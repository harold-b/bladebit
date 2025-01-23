#include "PlotReader.h"
#include "ChiaConsts.h"
#include "util/BitView.h"
#include "plotting/PlotTools.h"
#include "plotting/CTables.h"
#include "plotting/DTables.h"
#include "plotmem/LPGen.h"
#include "plotting/Compression.h"
#include "harvesting/GreenReaper.h"
#include "BLS.h"
#include "plotdisk/jobs/IOJob.h"

///
/// Plot Reader
///

//-----------------------------------------------------------
PlotReader::PlotReader( IPlotFile& plot )
    : _plot( plot )
{
    const size_t largestParkSize           = RoundUpToNextBoundaryT( CalculateParkSize( TableId::Table1, plot.K() ), sizeof( uint64 ) * 2 );
    const size_t maxDecompressedDeltasSize = RoundUpToNextBoundaryT( (size_t)0x7FFF, sizeof( uint64 ) );

    _parkBuffer   = bbmalloc<uint64>( largestParkSize );
    _deltasBuffer = bbmalloc<byte>  ( maxDecompressedDeltasSize );
}

//-----------------------------------------------------------
PlotReader::~PlotReader()
{
    free( _parkBuffer );    _parkBuffer = nullptr;
    free( _deltasBuffer );  _deltasBuffer = nullptr;

    bbvirtfreebounded( _c1Buffer );

    if( _c2Entries.Ptr() )
        delete[] _c2Entries.values;
    _c2Entries = {};

    bbvirtfreebounded_span( _c3Buffer );

    if( _grContext && _ownsGrContext )
        grDestroyContext( _grContext );
    _grContext = nullptr;
}

//-----------------------------------------------------------
uint64 PlotReader::GetC3ParkCount() const
{
    // We know how many C3 parks there are by how many 
    // entries we have in the C1 table - 1 (extra 0 entry added)
    // However, to make sure this is the case, we'll have to 
    // read-in all C1 entries and ensure we hit an empty one,
    // to ensure we don't run into dead/alignment-space
    const uint64 c3ParkCount = GetMaximumC1Entries();

    // Or just do this: 
    //  Same thing, but we use it
    //  because we want to validate the plot for farming, 
    //  and farming goes to C1 tables before it goes to C3
    // const size_t c3ParkSize  = CalculateC3Size();
    // const size_t c3TableSize = _plot.TableSize( PlotTable::C3 );
    // const size_t c3ParkCount = c3TableSize / c3ParkSize;

    return c3ParkCount;
}

//-----------------------------------------------------------
uint64 PlotReader::GetMaxF7EntryCount() const
{
    const size_t c3ParkCount = GetC3ParkCount();
    return c3ParkCount * kCheckpoint1Interval;
}

//-----------------------------------------------------------
size_t PlotReader::GetTableParkCount( const PlotTable table ) const
{
    switch( table )
    {
        case PlotTable::C3:
            return GetC3ParkCount();
    
        case PlotTable::Table7:
            return CDiv( GetMaxF7EntryCount(), kEntriesPerPark );

        case PlotTable::Table1:
        case PlotTable::Table2:
        case PlotTable::Table3:
        case PlotTable::Table4:
        case PlotTable::Table5:
        case PlotTable::Table6:
            return _plot.TableSize( table ) / GetParkSizeForTable( (TableId)table );

        default:
            return 0;
    }
}

//-----------------------------------------------------------
uint64 PlotReader::GetMaximumC1Entries() const
{
    // -1 because an extra 0 entry is added at the end
    const size_t c1TableSize = _plot.TableSize( PlotTable::C1 );
    const size_t f7Size      = CDiv( _plot.K(), 8 );
    const uint64 c3ParkCount = std::max( c1TableSize / f7Size, (size_t)1 ) - 1;

    return c3ParkCount;
}

//-----------------------------------------------------------
bool PlotReader::GetActualC1EntryCount( uint64& outC1Count )
{
    outC1Count = 0;

    const uint64 maxC1Entries = GetMaxF7EntryCount();

    if( maxC1Entries < 1 )
        return true;
        
    const size_t f7SizeBytes   = CDiv( _plot.K(), 8 );
    const uint64 c1Address     = _plot.TableAddress( PlotTable::C1 );
    const size_t c1TableSize   = _plot.TableSize( PlotTable::C1 );
          size_t c1ReadAddress = c1Address + c1TableSize - f7SizeBytes; 

    // Read entries from the end of the table until the start, until we find an entry that is
    // not zero/higher than the previous one
    if( !_plot.Seek( SeekOrigin::Begin, (int64)c1ReadAddress ) )
        return false;

    const uint32 k  = _plot.K();
          uint64 c1 = 0;
    while( c1ReadAddress >= c1Address )
    {
        uint64 newC1;
        if( _plot.Read( f7SizeBytes, &newC1 ) != (ssize_t)f7SizeBytes )
            return false;

        newC1 = Swap64( newC1 ) >> ( 64 - k );
        if( newC1 > c1 )
            break;

        if( c1ReadAddress <= c1Address )
            return false;
        
        c1ReadAddress -= f7SizeBytes;
    }

    outC1Count = ( c1ReadAddress - c1Address ) / f7SizeBytes;
    return true;
}

//-----------------------------------------------------------
int64 PlotReader::ReadC3Park( uint64 parkIndex, uint64* f7Buffer )
{
    const uint32 k              = _plot.K();
    const size_t f7SizeBytes    = CDiv( k, 8 );
    const size_t c3ParkSize     = CalculateC3Size();
    const uint64 c1Address      = _plot.TableAddress( PlotTable::C1 );
    const uint64 c3Address      = _plot.TableAddress( PlotTable::C3 );
    const size_t c1TableSize    = _plot.TableSize( PlotTable::C1 );
    const size_t c3TableSize    = _plot.TableSize( PlotTable::C3 );
    const uint64 c1EntryAddress = c1Address + parkIndex * f7SizeBytes;
    const uint64 parkAddress    = c3Address + parkIndex * c3ParkSize;


    // Ensure the C1 address is within the C1 table bounds.
    if( c1EntryAddress >= c1Address + c1TableSize - f7SizeBytes ) // - f7SizeBytes because the last C1 entry is an empty/dummy one
        return -1;

    // First we need to read the root F7 entry for the park,  which is in the C1 table.
    if( !_plot.Seek( SeekOrigin::Begin, (int64)c1EntryAddress ) )
        return -1;

    uint64 c1 = 0;
    if( _plot.Read( f7SizeBytes, &c1 ) != (ssize_t)f7SizeBytes )
        return -1;

    c1 = Swap64( c1 ) >> ( 64 - k );

    // Ensure we can read this park. If it's not present, it means
    // the C1 entry is the only entry in the park, so just return it.
    // Read the park into our buffer
    if( parkAddress >= c3Address + c3TableSize )
    {
        f7Buffer[0] = c1;
        return 1;
    }

    if( !_plot.Seek( SeekOrigin::Begin, (int64)parkAddress ) )
        return -1;

    // Read the size of the compressed C3 deltas
    uint16 compressedSize = 0;
    if( _plot.Read( sizeof( uint16 ), &compressedSize ) != (ssize_t)sizeof( uint16 ) )
        return -1;

    compressedSize = Swap16( compressedSize );
    if( compressedSize > c3ParkSize )
        return -1;

    // memset( _parkBuffer, 0, _parkBufferSize );
    if( _plot.Read( c3ParkSize - sizeof( uint16 ), _parkBuffer ) != c3ParkSize - sizeof( uint16 ) )
        return -1;

    // Now we can read the f7 deltas from the C3 park
    const size_t deltaCount = FSE_decompress_usingDTable( 
                                _deltasBuffer, kCheckpoint1Interval, 
                                _parkBuffer, compressedSize, 
                                (const FSE_DTable*)DTable_C3 );

    if( FSE_isError( deltaCount ) )
        return -1;           // #TODO: Set error message locally

    ASSERT( deltaCount );
    for( uint32 i = 0; i < deltaCount; i++ )
        if( _deltasBuffer[i] == 0xFF )
            return -1;

    // Unpack deltas into absolute values
    memset( f7Buffer, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    uint64 f7 = c1;
    f7Buffer[0] = f7;

    f7Buffer++;
    for( int32 i = 0; i < (int32)deltaCount; i++ )
        f7Buffer[i] = f7Buffer[i-1] + _deltasBuffer[i];

    return (int64)deltaCount+1;
}

//-----------------------------------------------------------
bool PlotReader::ReadP7Entries( uint64 parkIndex, uint64 p7ParkEntries[kEntriesPerPark] )
{
    if( LoadP7Park( parkIndex ) )
    {
        bbmemcpy_t( p7ParkEntries, _park7Entries, kEntriesPerPark );
        return true;
    }

    return false;
}

//-----------------------------------------------------------
bool PlotReader::ReadP7Entry( uint64 p7Index, uint64& outP7Entry )
{
    const uint64 parkIndex = p7Index / kEntriesPerPark;

    if( LoadP7Park( parkIndex ) )
    {
        const uint64 parkLocalP7Index = p7Index - parkIndex * kEntriesPerPark;
        outP7Entry = _park7Entries[parkLocalP7Index];
        return true;
    }

    outP7Entry = 0;
    return false;
}

//-----------------------------------------------------------
bool PlotReader::LoadP7Park( uint64 parkIndex )
{
    if( _park7Index != -1 && (uint64)_park7Index == parkIndex )
        return true;

    const uint32 k              = _plot.K();
    const uint32 p7EntrySize    = k + 1;
    const uint64 p7TableAddress = _plot.TableAddress( PlotTable::Table7 );
    const size_t p7TableMaxSize = _plot.TableSize( PlotTable::Table7 );
    const size_t parkSizeBytes  = CalculatePark7Size( k );

    const uint64 maxParks       = p7TableMaxSize / parkSizeBytes;

    // Park must be in the range of the maximum table parks encoded
    if( parkIndex >= maxParks )
        return false;

    const uint64 parkAddress = p7TableAddress + parkIndex * parkSizeBytes;

    if( !_plot.Seek( SeekOrigin::Begin, (int64)parkAddress ) )
        return false;

    if( _plot.Read( parkSizeBytes, _parkBuffer ) != (ssize_t)parkSizeBytes )
        return false;

    CPBitReader parkReader( (byte*)_parkBuffer, parkSizeBytes * 8 );

    for( uint32 i = 0; i < kEntriesPerPark; i++ )
        _park7Entries[i] = parkReader.Read64( p7EntrySize );

    _park7Index = (int64)parkIndex;
    return true;
}

//-----------------------------------------------------------
bool PlotReader::ReadLPParkComponents( TableId table, uint64 parkIndex, 
                                       CPBitReader& outStubs, byte*& outDeltas, 
                                       uint128& outBaseLinePoint, uint64& outDeltaCounts )
{
    outDeltaCounts = 0;

    ASSERT( table < TableId::Table7 );
    if( table >= TableId::Table7 )
        return false;

    const uint32 k                = _plot.K();
    const size_t lpSizeBytes      = LinePointSizeBytes( k );
    const size_t tableMaxSize     = _plot.TableSize( (PlotTable)table );
    const size_t tableAddress     = _plot.TableAddress( (PlotTable)table );
    const size_t parkSize         = GetParkSizeForTable( table );

    const uint64 maxParks       = tableMaxSize / parkSize;
    if( parkIndex >= maxParks )
        return false;
    
    const size_t parkAddress    = tableAddress + parkIndex * parkSize;
    
    if( !_plot.Seek( SeekOrigin::Begin, (int64)parkAddress ) )
        return false;

    // Read base full line point
    uint128 baseLinePoint;
    {
        uint64 baseLPBytes[CDiv(LinePointSizeBytes( 50 ), sizeof(uint64))] = { 0 };

        if( _plot.Read( lpSizeBytes, baseLPBytes ) != (ssize_t)lpSizeBytes )
            return false;

        const size_t lpSizeBits = (uint32)LinePointSizeBits( k );

        CPBitReader lpReader( (byte*)baseLPBytes, RoundUpToNextBoundary( lpSizeBits, 64 ) );
        baseLinePoint = lpReader.Read128Aligned( (uint32)lpSizeBits );
    }

    // Read stubs
    const size_t stubsSizeBytes = GetLPStubByteSize( table );
    uint64* stubsBuffer = _parkBuffer;

    if( _plot.Read( stubsSizeBytes, stubsBuffer ) != (ssize_t)stubsSizeBytes )
        return false;

    // Read deltas
    const size_t maxDeltasSizeBytes = GetParkDeltasSectionMaxSize( (TableId)table );
    byte* compressedDeltaBuffer = ((byte*)_parkBuffer) + RoundUpToNextBoundary( stubsSizeBytes, sizeof( uint64 ) );
    byte* deltaBuffer           = _deltasBuffer;

    uint16 compressedDeltasSize = 0;
    if( _plot.Read( 2, &compressedDeltasSize ) != 2 )
        return false;

    // Don't support uncompressed deltas
    if( compressedDeltasSize & 0x8000 )
        return false;

    size_t deltaCount = 0;

    // #TODO: Investigate this, but we should not support uncompressed deltas
    // if( compressedDeltasSize & 0x8000 ) 
    // {
    //     // Uncompressed
    //     compressedDeltasSize &= 0x7fff;
    //     if( _plot.Read( compressedDeltasSize, compressedDeltaBuffer ) != compressedDeltasSize )
    //         return false;

    //     deltaCount = compressedDeltasSize;
    // }
    // else
    {
        // Compressed deltas
        if( _plot.Read( compressedDeltasSize, compressedDeltaBuffer ) != compressedDeltasSize )
            return false;

        // Decompress deltas
        const FSE_DTable* dTable = GetDTableForTable( table );

        deltaCount = FSE_decompress_usingDTable( 
                        deltaBuffer, kEntriesPerPark - 1, 
                        compressedDeltaBuffer, compressedDeltasSize, 
                        dTable );

        if( FSE_isError( deltaCount ) )
            return false;
    }
    
    outStubs         = CPBitReader( (byte*)stubsBuffer, RoundUpToNextBoundary( stubsSizeBytes * 8, 64 ) );
    outBaseLinePoint = baseLinePoint;
    outDeltas        = deltaBuffer;
    outDeltaCounts   = deltaCount;

    return true;
}

//-----------------------------------------------------------
const FSE_DTable* PlotReader::GetDTableForTable( TableId table ) const
{
    if( !IsCompressedXTable( table ) )
        return DTables[(int)table];

    return CreateCompressionDTable( _plot.CompressionLevel() );
}

//-----------------------------------------------------------
bool PlotReader::ReadLPPark( TableId table, uint64 parkIndex, uint128 linePoints[kEntriesPerPark], uint64& outEntryCount )
{
    outEntryCount = 0;

    CPBitReader stubReader;
    byte*       deltaBuffer   = nullptr;
    uint128     baseLinePoint = 0;
    uint64      deltaCount    = 0;

    if( !ReadLPParkComponents( table, parkIndex, stubReader, deltaBuffer, baseLinePoint, deltaCount ) )
        return false;

    // Decode line points from stubs and deltas
    linePoints[0] = baseLinePoint;
    if( deltaCount > 0 )
    {
        const uint32 stubBitSize = GetLPStubBitSize( table );
        
        for( uint64 i = 1; i <= deltaCount; i++ )
        {
            // Since these entries are still deltafied, we can fit them in 64-bits
            const uint64 stub = stubReader.Read64( stubBitSize );
            const uint64 lp   = stub | (((uint64)deltaBuffer[i-1]) << stubBitSize );

            // Get absolute LP from delta
            linePoints[i] = linePoints[i-1] + (uint128)lp;
        }
    }

    outEntryCount = deltaCount + 1;
    return true;
}

//-----------------------------------------------------------
TableId PlotReader::GetLowestStoredTable() const
{
    const uint32 compressionLevel = _plot.CompressionLevel();

    const uint32 numDroppedTables = compressionLevel == 0 ? 0 :
                                    compressionLevel >= 40 ? 2 : 1;

    return TableId::Table1 + numDroppedTables;
}

//-----------------------------------------------------------
bool PlotReader::IsCompressedXTable( const TableId table ) const
{
    return _plot.CompressionLevel() > 0 && table == GetLowestStoredTable();
}

//-----------------------------------------------------------
size_t PlotReader::GetParkSizeForTable( TableId table ) const
{
    if( IsCompressedXTable( table ) )
        return GetCompressionInfoForLevel( _plot.CompressionLevel() ).tableParkSize;

    if( table < GetLowestStoredTable() )
    {
        ASSERT(0);
        return 0;
    }

    return CalculateParkSize( table, _plot.K() );
}

//-----------------------------------------------------------
uint32 PlotReader::GetLPStubBitSize( TableId table ) const
{
    FatalIf( table < GetLowestStoredTable(), "Getting stub bit size for invalid table." );

    if( !IsCompressedXTable( table ) )
        return _plot.K() - kStubMinusBits;

    auto info = GetCompressionInfoForLevel( _plot.CompressionLevel() );
    return info.stubSizeBits;
}

//-----------------------------------------------------------
uint32 PlotReader::GetLPStubByteSize( const TableId table ) const
{
    return CDiv( ( kEntriesPerPark - 1 ) * GetLPStubBitSize( table ), 8 );
}

//-----------------------------------------------------------
size_t PlotReader::GetParkDeltasSectionMaxSize( const TableId table ) const
{
    if( !IsCompressedXTable( table ) )
        return CalculateMaxDeltasSize( table );

    auto info = GetCompressionInfoForLevel( _plot.CompressionLevel() );

    const uint32 lpSize       = CDiv( _K * 2u, 8 );
    const uint32 stubByteSize = GetLPStubByteSize( table );
    
    return info.tableParkSize - ( lpSize + stubByteSize );
}

// #TODO: Add 64-bit outLinePoint (templatize)
//-----------------------------------------------------------
bool PlotReader::ReadLP( TableId table, uint64 index, uint128& outLinePoint )
{
    outLinePoint = 0;

    CPBitReader stubReader;
    byte*       deltaBuffer   = nullptr;
    uint128     baseLinePoint = 0;
    uint64      deltaCount    = 0;

    const uint64 parkIndex  = index / kEntriesPerPark;

    if( !ReadLPParkComponents( table, parkIndex, stubReader, deltaBuffer, baseLinePoint, deltaCount ) )
        return false;

    const uint64 lpLocalIdx = index - parkIndex * kEntriesPerPark;

    if( lpLocalIdx > 0 )
    {
        if( lpLocalIdx-1 >= deltaCount )
            return false;

        const uint64 maxIter     = std::min( lpLocalIdx, deltaCount );
        const uint32 stubBitSize = GetLPStubBitSize( table );

        for( uint64 i = 0; i < maxIter; i++ )
        {
            // Since these entries are still deltafied, we can fit them in 64-bits
            const uint64 stub    = stubReader.Read64( stubBitSize );
            const uint64 lpDelta = stub | (((uint64)deltaBuffer[i]) << stubBitSize );

            // Get absolute LP from delta
            baseLinePoint += (uint128)lpDelta;
        }
    }

    outLinePoint = baseLinePoint;
    return true;
}

//-----------------------------------------------------------
bool PlotReader::FetchProofFromP7Entry( uint64 p7Entry, uint64 proof[32] )
{
    // #TODO: Implement me
    ASSERT( 0 );
    return false;
}

//-----------------------------------------------------------
uint64 PlotReader::GetP7IndicesForF7( const uint64 f7, uint64& outStartT6Index )
{
    if( !LoadC2Entries() )
        return 0;

    uint64 c2Index = 0;

    for( uint64 i = 0; ; )
    {
        const uint64 c2 = _c2Entries[i];

        if( c2 > f7 || ++i >= _c2Entries.Length() )
        {
            if( c2Index > 0 ) c2Index--;
            break;
        }

        c2Index++;
    }

    const uint64 c1StartIndex = c2Index * kCheckpoint2Interval;

    const uint32 k              = _plot.K();
    const size_t f7SizeBytes    = CDiv( k, 8 );
    const size_t f7BitCount     = f7SizeBytes * 8;
    const uint64 c1TableAddress = _plot.TableAddress( PlotTable::C1 );
    const size_t c1TableSize    = _plot.TableSize( PlotTable::C1 );
    const uint64 c1TableEnd     = c1TableAddress + c1TableSize;
    const uint64 c1EntryAddress = c1TableAddress + c1StartIndex * f7SizeBytes;


    const uint64 c1EndAddress = std::min( c1EntryAddress + ( kCheckpoint1Interval * f7SizeBytes ), c1TableEnd );

    const size_t readSize     = c1EndAddress - c1EntryAddress;
    const uint64 c1EntryCount = readSize / f7SizeBytes;

    if( c1EntryCount < 1 )
        return {};

    if( !_plot.Seek( SeekOrigin::Begin, (int64)c1EntryAddress ) )
    {
        Log::Error( "Seek to C1 address failed: %d", _plot.GetError() );
        return {};
    }

    // Read C1 entries until we find one equal or larger than the f7 we're looking for
    if( !_c1Buffer )
        _c1Buffer = bbcvirtallocbounded<byte>( kCheckpoint1Interval * f7SizeBytes );
    
    if( _plot.Read( readSize, _c1Buffer ) != (ssize_t)readSize )
    {
        Log::Error( "Failed to read C1 entries: %d", _plot.GetError() );
        return {};
    }

    CPBitReader reader( _c1Buffer, readSize * 8 );
    uint64 c3Park = c1StartIndex;
    uint64 c1     = 0;

    for( uint64 i = 0; ; )
    {
        c1 = reader.Read64( (uint32)f7BitCount );

        if( c1 >= f7 || ++i >= c1EntryCount )
        {
            if( c3Park > 0 ) c3Park--;
            break;
        }

        c3Park++;
    }

    const uint64 parkCount = c1 == f7 && c3Park > 0 ? 2 : 1; // If we got the same c1 as f7, then the previous
                                                             // needs to be read as well because we may have duplicate f7s
                                                             // in the previous park's last entries.
        
    if( _c3Buffer.Ptr() == nullptr )
    {
        _c3Buffer.values = bbcvirtallocbounded<uint64>( kCheckpoint1Interval * 2 );
        _c3Buffer.length = kCheckpoint1Interval * 2;
    }

    int64 c3Count = ReadC3Park( c3Park, _c3Buffer.Ptr() );
    if( c3Count < 0)
        return {};

    if( parkCount > 1 )
    {
        ASSERT( parkCount == 2 );
        const int64 secondParkC3Count = ReadC3Park( c3Park+1, _c3Buffer.Ptr() + c3Count );
        if( secondParkC3Count < 0 )
            return {};

        c3Count += secondParkC3Count;
    }

    // Grab as many matches as we can
    const Span<uint64> c3Entries    = _c3Buffer.SliceSize( (size_t)c3Count );
          uint64       c3StartIndex = c3Park * kCheckpoint1Interval;

    for( uint64 i = 0; i < c3Entries.Length(); i++ )
    {
        if( c3Entries[i] == f7 )
        {
            uint64 matchCount = 1;

            outStartT6Index = c3StartIndex + i;
            while( ++i < (uint64)c3Count && c3Entries[i] == f7 )
                matchCount++;

            return matchCount;
        }
    }

    return 0;
}

//-----------------------------------------------------------
ProofFetchResult PlotReader::FetchProof( const uint64 t6LPIndex, uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] )
{
    uint64 lpIndices[2][BB_PLOT_PROOF_X_COUNT];

    uint64* lpIdxSrc = lpIndices[0];
    uint64* lpIdxDst = lpIndices[1];

    *lpIdxSrc = t6LPIndex;

    // Fetch line points to back pointers going through all our tables
    // from 6 to 1, grabbing all of the x's that make up a proof.
    uint32 lookupCount = 1;

    const bool    isCompressed = _plot.CompressionLevel() > 0;
    const TableId endTable     = GetLowestStoredTable();

    for( TableId table = TableId::Table6; table >= endTable; table-- )
    {
        ASSERT( lookupCount <= 32 );

        const bool use64BitLP = table < TableId::Table6 && _plot.K() <= 32;

        for( uint32 i = 0, dst = 0; i < lookupCount; i++, dst += 2 )
        {
            const uint64 idx = lpIdxSrc[i];

            uint128 lp = 0;
            if( !ReadLP( table, idx, lp ) )
                return ProofFetchResult::Error;

            const BackPtr ptr = use64BitLP ? LinePointToSquare64( (uint64)lp ) : LinePointToSquare( lp );

            ASSERT( ptr.x > ptr.y );
            lpIdxDst[dst+0] = ptr.y;
            lpIdxDst[dst+1] = ptr.x;
        }

        lookupCount <<= 1;

        std::swap( lpIdxSrc, lpIdxDst );
        // memset( lpIdxDst, 0, sizeof( uint64 ) * PROOF_X_COUNT );
    }

    const uint32  finalIndex = ((uint32)(endTable - TableId::Table1)) % 2;
    const uint64* xSource   = lpIndices[finalIndex];

    if( isCompressed )
        return DecompressProof( xSource, fullProofXs );

    // Full proof x's will be at the src ptr
    memcpy( fullProofXs, lpIdxSrc, sizeof( uint64 ) * BB_PLOT_PROOF_X_COUNT );
    return ProofFetchResult::OK;
}

//-----------------------------------------------------------
ProofFetchResult PlotReader::DecompressProof( const uint64 compressedProof[BB_PLOT_PROOF_X_COUNT], uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] )
{
    GreenReaperContext* gr = GetGRContext();
    if( !gr )
        return ProofFetchResult::Error;

    const uint32 compressionLevel = _plot.CompressionLevel();

    GRCompressedProofRequest req = {};
    req.plotId           = _plot.PlotId();
    req.compressionLevel = compressionLevel;

    const uint32 compressedProofCount = compressionLevel < 40 ? PROOF_X_COUNT / 2 : PROOF_X_COUNT / 4;

    for( uint32 i = 0; i < compressedProofCount; i++ )
        req.compressedProof[i] = compressedProof[i];

    const GRResult r = grFetchProofForChallenge( gr, &req );

    if( r == GRResult_OK )
    {
        bbmemcpy_t( fullProofXs, req.fullProof, BB_PLOT_PROOF_X_COUNT );
        return ProofFetchResult::OK;
    }

    return r == GRResult_NoProof ? ProofFetchResult::NoProof : ProofFetchResult::CompressionError;
}

//-----------------------------------------------------------
ProofFetchResult PlotReader::FetchQualityXsForP7Entry( 
    const uint64  t6Index, 
    const byte    challenge[BB_CHIA_CHALLENGE_SIZE], 
          uint64& outX1, uint64& outX2 )
{
    const bool    isCompressed = _plot.CompressionLevel() > 0;
    const TableId endTable     = GetLowestStoredTable();

    GreenReaperContext* gr = nullptr;
    if( isCompressed )
    {
        gr = GetGRContext();
        if( !gr )
            return ProofFetchResult::Error;
    }

    const uint32 last5Bits = (uint32)challenge[31] & 0x1f;

    uint64 lpIndex  = t6Index; 
    uint64 altIndex = 0;

    for( TableId table = TableId::Table6; table > endTable; table-- )
    {
        // Read line point
        uint128 lp;
        if( !ReadLP( table, lpIndex, lp ) )
            return ProofFetchResult::Error;


        const bool    use64BitLP = table < TableId::Table6 && _plot.K() <= 32;
        const BackPtr ptr        = use64BitLP ? LinePointToSquare64( (uint64)lp ) : LinePointToSquare( lp );
        ASSERT( ptr.x >= ptr.y );

        const bool isTableBitSet = ((last5Bits >> ((uint32)table-1)) & 1) == 1;

        if( !isTableBitSet )
        {
            lpIndex  = ptr.y;
            altIndex = ptr.x;
        }
        else
        {
            lpIndex  = ptr.x;
            altIndex = ptr.y;
        }
    }

    if( isCompressed )
    {
        const bool needBothLeaves = _plot.CompressionLevel() >= 6;

        // Read both back pointers, depending on compression level
        uint128 xLP0, xLP1;
        if( !ReadLP( endTable, lpIndex, xLP0 ) )
            return ProofFetchResult::Error;

        if( needBothLeaves )
        {
            if( !ReadLP( endTable, altIndex, xLP1 ) )
                return ProofFetchResult::Error;
        }

        // Now decompress the X's
        GRCompressedQualitiesRequest req = {};
        req.plotId                = _plot.PlotId();
        req.compressionLevel      = _plot.CompressionLevel();
        req.challenge             = challenge;
        req.xLinePoints[0].hi     = (uint64)(xLP0 >> 64);
        req.xLinePoints[0].lo     = (uint64)xLP0;

        if( needBothLeaves )
        {
            req.xLinePoints[1].hi = (uint64)(xLP1 >> 64);
            req.xLinePoints[1].lo = (uint64)xLP1;
        }

        const auto r = grGetFetchQualitiesXPair( gr, &req );
        if( r != GRResult_OK )
            return r == GRResult_NoProof ? ProofFetchResult::NoProof : ProofFetchResult::CompressionError;

        outX1 = req.x1;
        outX2 = req.x2;
    }
    else
    {
        uint128 lp;
        if( !ReadLP( endTable, lpIndex, lp ) )
            return ProofFetchResult::Error;

        const BackPtr ptr = _plot.K() <= 32 ? LinePointToSquare64( (uint64)lp ) : LinePointToSquare( lp );
        outX1 = ptr.x;
        outX2 = ptr.y;
    }

    return ProofFetchResult::OK;
}

//-----------------------------------------------------------
ProofFetchResult PlotReader::FetchQualityForP7Entry( 
    const uint64 t6Index, 
    const byte challenge [BB_CHIA_CHALLENGE_SIZE], 
          byte outQuality[BB_CHIA_QUALITY_SIZE] )
{
    uint64 x1, x2;

    const ProofFetchResult r = FetchQualityXsForP7Entry( t6Index, challenge, x1, x2 );
    if( r != ProofFetchResult::OK )
        return r;

    const size_t HASH_SIZE_MAX = BB_CHIA_QUALITY_SIZE + CDiv( 2*50, 8 );
    byte hashInput[HASH_SIZE_MAX] = {};

    const uint32 k        = _plot.K();
    const size_t hashSize = BB_CHIA_QUALITY_SIZE + CDiv( 2*k, 8 );

    memcpy( hashInput, challenge, BB_CHIA_QUALITY_SIZE );

    Bits<HASH_SIZE_MAX-BB_CHIA_QUALITY_SIZE> hashBits;
    hashBits.Write( x2, k );
    hashBits.Write( x1, k );

    hashBits.ToBytes( hashInput + BB_CHIA_QUALITY_SIZE );
    bls::Util::Hash256( outQuality, hashInput, hashSize );

    return r;
}

//-----------------------------------------------------------
bool PlotReader::LoadC2Entries()
{
    if( _c2Entries.Ptr() )
        return true;

    const size_t c2Size = _plot.TableSize( PlotTable::C2 );
    if( c2Size == 0 )
        return false;

    const size_t f7ByteSize = CDiv( _plot.K(), 8 );

    const uint64 c2MaxEntries = c2Size / f7ByteSize;
    if( c2MaxEntries < 1 )
        return false;

    if( !_plot.Seek( SeekOrigin::Begin, (int64)_plot.TableAddress( PlotTable::C2 ) ) )
        return false;

   
    byte* buffer = bbvirtallocbounded<byte>( c2Size );

    if( _plot.Read( c2Size, buffer ) != (ssize_t)c2Size )
    {
        bbvirtfreebounded( buffer );
        return false;
    }

    _c2Entries = Span<uint64>( new uint64[c2MaxEntries], c2MaxEntries );

    const size_t f7BitCount = f7ByteSize * 8;
    CPBitReader reader( buffer, c2Size * 8 );

    uint64 prevF7 = 0;
    uint64 i;
    for( i = 0; i < c2MaxEntries; i++ )
    {
        const uint64 f7 = reader.Read64( (uint32)f7BitCount );
        
        // Short circuit if we encounter an unsorted/out-of-order c2 entry
        if( f7 < prevF7 )
            break;

        _c2Entries[i] = f7;
        prevF7 = f7;
    }

    _c2Entries.length = i;

    bbvirtfreebounded( buffer );
    return true;
}

//-----------------------------------------------------------
void PlotReader::AssignDecompressionContext( struct GreenReaperContext* context )
{
    ASSERT( context );
    if( !context)
        return;

    if( _grContext && _ownsGrContext )
        grDestroyContext( _grContext );
    
    _grContext     = context;
    _ownsGrContext = false;
}

//-----------------------------------------------------------
void PlotReader::ConfigDecompressor( const uint32 threadCount, const bool disableCPUAffinity, const uint32 cpuOffset, bool useGpu, int gpuIndex )
{
    if( _grContext && _ownsGrContext )
        grDestroyContext( _grContext );

    _grContext     = nullptr;
    _ownsGrContext = true;

    GreenReaperConfig cfg = {};
    cfg.apiVersion         = GR_API_VERSION;
    cfg.threadCount        = bbclamp( threadCount, 1u, SysHost::GetLogicalCPUCount() );
    cfg.cpuOffset          = cpuOffset;
    cfg.disableCpuAffinity = disableCPUAffinity ? GR_TRUE : GR_FALSE;
    cfg.gpuRequest         = !useGpu ? GRGpuRequestKind_None : 
                                gpuIndex >= 0 ? GRGpuRequestKind_ExactDevice : GRGpuRequestKind_FirstAvailable;
    cfg.gpuDeviceIndex     = gpuIndex < 0 ? 0 : gpuIndex;

    auto result = grCreateContext( &_grContext, &cfg, sizeof( GreenReaperConfig ) );
    ASSERT( result == GRResult_OK );
}

//-----------------------------------------------------------
GreenReaperContext* PlotReader::GetGRContext()
{
    if( _grContext == nullptr )
    {
        GreenReaperConfig cfg = {};
        cfg.apiVersion  = GR_API_VERSION;
        cfg.threadCount = std::min( 8u, SysHost::GetLogicalCPUCount() );

        auto result = grCreateContext( &_grContext, &cfg, sizeof( GreenReaperConfig ) );
        ASSERT( result == GRResult_OK );

        _ownsGrContext = true;
    }

    return _grContext;
}

///
/// Memory Plot
///
//-----------------------------------------------------------
MemoryPlot::MemoryPlot()
    : _bytes( nullptr, 0 )
{}

//-----------------------------------------------------------
MemoryPlot::MemoryPlot( const MemoryPlot& plotFile )
{
    _bytes    = plotFile._bytes;
    _err      = 0;
    _position = 0;

    int headerError = 0;
    if( !ReadHeader( headerError ) )
    {
        if( headerError )
            _err = headerError;

        if( _err == 0 )
            _err = -1; // #TODO: Set generic plot header read error

        _bytes.values = nullptr;
        return;
    }

    _plotPath = plotFile._plotPath;
}

//-----------------------------------------------------------
MemoryPlot::~MemoryPlot()
{
    // #TODO: Don't destroy bytes unless we own them. Use a shared ptr here.
    if( _bytes.values )
        SysHost::VirtualFree( _bytes.values );

    _bytes = Span<byte>( nullptr, 0 );
}

//-----------------------------------------------------------
bool MemoryPlot::Open( const char* path )
{
    ASSERT( path );
    if( !path )
        return false;

    if( IsOpen() )
        return false;

    FileStream file;    // #TODO: Enable no buffering again, for now when testing disable to take advantage of caching.
    if( !file.Open( path, FileMode::Open, FileAccess::Read ) )//, FileFlags::LargeFile | FileFlags::NoBuffering ) )
    {
        _err = file.GetError();
        return false;
    }

    const ssize_t plotSize = file.Size();
    if( plotSize <= 0 )
    {
        if( plotSize < 0 )
            _err = file.GetError();
        else
            _err = -1;  // #TODO: Assign an actual user error.
        return false;
    }

    // Add an extra block at the end to be able to do an aligned read there if
    // we have any remainder that does not align to a block
    const size_t allocSize = RoundUpToNextBoundary( (size_t)plotSize, (int)file.BlockSize() ) + file.BlockSize();

    byte* bytes = (byte*)SysHost::VirtualAlloc( allocSize );
    if( !bytes )
    {
        _err = -1;      // #TODO: Assign an actual user error.
        return false;
    }

    // Read the whole thing to memory
    size_t readSize      = RoundUpToNextBoundary( plotSize, (int)file.BlockSize() );/// file.BlockSize() * file.BlockSize();
    // size_t readRemainder = plotSize - readSize;
    const size_t readEnd = readSize - plotSize;
    byte*  reader        = bytes;
    
    // Read blocks
    while( readSize > readEnd )
    {
        const ssize_t read = file.Read( reader, readSize );
        ASSERT( read );

        if( read < 0 )
        {
            _err = file.GetError();
            SysHost::VirtualFree( bytes );

            return false;
        }

        readSize -= (size_t)read;
        reader   += read;
    }

    // if( readRemainder )
    // {
    //     byte* block = (byte*)RoundUpToNextBoundary( (uintptr_t)reader, (int)file.BlockSize() );

    //     const ssize_t read = file.Read( block, (size_t)file.BlockSize() );
    //     ASSERT( read );
    //     ASSERT( read >= readRemainder );

    //     if( read < 0 )
    //     {
    //         _err = file.GetError();
    //         SysHost::VirtualFree( bytes );

    //         return false;
    //     }

    //     if( reader != block )
    //         memmove( reader, block, readRemainder );
    // }

    _bytes = Span<byte>( bytes, (size_t)plotSize );

    // Read the header
    int headerError = 0;
    if( !ReadHeader( headerError ) )
    {
        if( headerError )
            _err = headerError;

        if( _err == 0 )
            _err = -1; // #TODO: Set generic plot header read error

        _bytes.values = nullptr;
        SysHost::VirtualFree( bytes );
        return false;
    }

    // Lock the plot memory into read-only mode
    SysHost::VirtualProtect( bytes, allocSize, VProtect::Read );

    // Save data, good to go
    _plotPath = path;

    return true;
}

//-----------------------------------------------------------
bool MemoryPlot::IsOpen() const
{
    return _bytes.values != nullptr;
}

//-----------------------------------------------------------
size_t MemoryPlot::PlotSize() const
{
    return _bytes.length;
}

//-----------------------------------------------------------
bool MemoryPlot::Seek( SeekOrigin origin, int64 offset )
{
    ssize_t absPosition = 0;

    switch( origin )
    {
        case SeekOrigin::Begin:
            absPosition = offset;
            break;

        case SeekOrigin::Current:
            absPosition = _position + offset;
            break;

        case SeekOrigin::End:
            absPosition = (ssize_t)_bytes.length + offset;
            break;
    
        default:
            _err =  -1;     // #TODO: Set proper user error.
            return false;
    }

    if( absPosition < 0 || absPosition > (ssize_t)_bytes.length )
    {
        _err =  -1;     // #TODO: Set proper user error.
        return false;
    }

    _position = absPosition;
    return true;
}

//-----------------------------------------------------------
ssize_t MemoryPlot::Read( size_t size, void* buffer )
{
    if( size < 1 || !buffer )
        return 0;

    ASSERT( buffer );

    const size_t endPos = (size_t)_position + size;

    if( endPos > _bytes.length )
    {
        _err = -1; // #TODO: Set proper user error
        return false;
    }

    memcpy( buffer, _bytes.values + _position, size );
    _position = (ssize_t)endPos;

    return (ssize_t)size;
}

//-----------------------------------------------------------
int MemoryPlot::GetError() 
{
    return _err;
}



///
/// FilePlot
///
//-----------------------------------------------------------
FilePlot::FilePlot()
{

}

//-----------------------------------------------------------
FilePlot::FilePlot( const FilePlot& file )
{
    if( file.IsOpen() )
        Open( file._plotPath.c_str() ); // #TODO: Seek to same location
    else
        _plotPath = "";
}

//-----------------------------------------------------------
FilePlot::~FilePlot()
{

}

//-----------------------------------------------------------
bool FilePlot::Open( const char* path )
{
    if( !_file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::None ) )
        return false;

    // Read the header
    int headerError = 0;
    if( !ReadHeader( headerError ) )
    {
        // if( headerError )
        //     _err = headerError; // #TODO: Set local error

        // if( _err == 0 )
        //     _err = -1; // #TODO: Set generic plot header read error
        _file.Close();
        return false;
    }
    
    _plotPath = path;
    return true;
}

//-----------------------------------------------------------
bool FilePlot::IsOpen() const
{
    return _file.IsOpen();
}

//-----------------------------------------------------------
size_t FilePlot::PlotSize() const
{
    const ssize_t sz = ((FileStream*)&_file)->Size();
    if( sz < 0 )
        return 0;

    return (size_t)sz;
}

//-----------------------------------------------------------
ssize_t FilePlot::Read( size_t size, void* buffer )
{
    if( size > (size_t)std::numeric_limits<ssize_t>::max() )
        size = (size_t)std::numeric_limits<ssize_t>::max();

    int error = 0;
    if( !IOJob::ReadFromFileUnaligned( _file, buffer, size, error ) )
    {
        size = 0;
        (void)error;
        Log::Error( "Failed to read from plot with error %d", error );
    }

    return (ssize_t)size;
}

//-----------------------------------------------------------
bool FilePlot::Seek( SeekOrigin origin, int64 offset )
{
    return _file.Seek( offset, origin );
}

//-----------------------------------------------------------
int FilePlot::GetError()
{
    return _file.GetError();
}

