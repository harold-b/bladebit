#include "plotting/CTables.h"
#include "ChiaConsts.h"
#include "util/Log.h"
#include "util/BitView.h"
#include "io/FileStream.h"
#include "PlotTools.h"
#include "PlotReader.h"
#include "plotting/PlotTools.h"
#include "plotmem/LPGen.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"
#include "threading/MTJob.h"
#include "util/CliParser.h"
#include "plotting/GlobalPlotConfig.h"
#include <mutex>

#define PROOF_X_COUNT       64
#define MAX_K_SIZE          50
#define MAX_META_MULTIPLIER 4
#define MAX_Y_BIT_SIZE      ( MAX_K_SIZE + kExtraBits )
#define MAX_META_BIT_SIZE   ( MAX_K_SIZE * MAX_META_MULTIPLIER )
#define MAX_FX_BIT_SIZE     ( MAX_Y_BIT_SIZE + MAX_META_BIT_SIZE + MAX_META_BIT_SIZE )

typedef Bits<MAX_Y_BIT_SIZE>    YBits;
typedef Bits<MAX_META_BIT_SIZE> MetaBits;
typedef Bits<MAX_FX_BIT_SIZE>   FxBits;

//-----------------------------------------------------------
const char USAGE[] = R"(validate [OPTIONS] <plot_path>

Validates all of a plot's values to ensure they all contain valid proofs.

[NOTES]
You can specify the thread count in the bladebit global option '-t'.

[ARGUMENTS]
<plot_path>   : Path to the plot file to be validated.

[OPTIOINS]
 -m, --in-ram : Loads the whole plot file into memory before validating.

 -o, --offset : Percentage offset at which to start validating.
                Ex (start at 50%): bladebit validate -o 50 /path/to/my/plot

 -h, --help   : Print this help message and exit.
)";

void PlotValidatorPrintUsage()
{
    Log::Line( USAGE );
}

struct UnpackedK32Plot
{
    Span<uint32> table1;    // Xs
    Span<Pair>   table2;
    Span<Pair>   table3;
    Span<Pair>   table4;
    Span<Pair>   table5;
    Span<Pair>   table6;
    Span<Pair>   table7;
    Span<uint32> f7;
    
    static UnpackedK32Plot Load( IPlotFile** plotFile, ThreadPool& pool, uint32 threadCount );
    bool FetchProof( uint64 index );
};


void GetProofF1( uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64 fx[PROOF_X_COUNT] );

template<bool Use64BitLpToSquare>
bool FetchProof( PlotReader& plot, uint64 t6LPIndex, uint64 fullProofXs[PROOF_X_COUNT] );

bool ValidateFullProof( PlotReader& plot, uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 );
void ReorderProof( PlotReader& plot, uint64 fullProofXs[PROOF_X_COUNT] );
void GetProofF1( uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64 fx[PROOF_X_COUNT] );

uint64 BytesToUInt64( const byte bytes[8] );
uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount );

bool FxMatch( uint64 yL, uint64 yR );

void FxGen( const TableId table, const uint32 k, 
            const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
            uint64& outY, MetaBits& outMeta );

void PlotValidatorPrintUsage();
bool ValidatePlot( const ValidatePlotOptions& options );

struct ValidateJob : MTJob<ValidateJob>
{
    IPlotFile*  plotFile;
    uint64      failCount;
    std::mutex* logLock;
    float       startOffset;

    void Run() override;
    void Log( const char* msg, ... );
};

//-----------------------------------------------------------
void PlotValidatorMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    ValidatePlotOptions opts;

    while( cli.HasArgs() )
    {
        if( cli.ReadSwitch( opts.inRAM, "-m", "--in-ram" ) )
            continue;
        else if( cli.ReadSwitch( opts.unpacked, "-u", "--unpacked" ) )
            continue;
        else if( cli.ReadValue( opts.startOffset, "-o", "--offset" ) )
            continue;
        else if( cli.ArgConsume( "-h", "--help" ) )
        {
            PlotValidatorPrintUsage();
            exit( 0 );
        }
        else if( cli.IsLastArg() )
        {
            opts.plotPath = cli.ArgConsume();
        }
        else
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
        }
    }

    const uint32 maxThreads = SysHost::GetLogicalCPUCount();

    opts.threadCount = gCfg.threadCount == 0 ? maxThreads : std::min( maxThreads, gCfg.threadCount );
    opts.startOffset = std::max( std::min( opts.startOffset / 100.f, 100.f ), 0.f );

    ValidatePlot( opts );

    exit( 0 );
}


//-----------------------------------------------------------
bool ValidatePlot( const ValidatePlotOptions& options )
{
    LoadLTargets();

    const uint32 threadCount = options.threadCount;

    IPlotFile*  plotFile  = nullptr;
    IPlotFile** plotFiles = new IPlotFile*[threadCount];

    if( options.inRAM && !options.unpacked )
    {
        auto* memPlot = new MemoryPlot();
        plotFile = memPlot;
        
        Log::Line( "Reading plot file into memory..." );
        if( memPlot->Open( options.plotPath.c_str() ) )
        {
            for( uint32 i = 0; i < threadCount; i++ )
                plotFiles[i] = new MemoryPlot( *memPlot );
        }
    }
    else
    {
        auto* filePlot = new FilePlot();
        plotFile = filePlot;

        if( filePlot->Open( options.plotPath.c_str() ) )
        {
            for( uint32 i = 0; i < threadCount; i++ )
                plotFiles[i] = new FilePlot( *filePlot );
        }
    }

    ExitIf( !plotFile->IsOpen(), "Failed to open plot at path '%s'.", options.plotPath.c_str() );
    ExitIf( options.unpacked && plotFile->K() != 32, "Unpacked plots are only supported for k=32 plots." );

    Log::Line( "Validating plot %s", options.plotPath.c_str() );
    Log::Line( "K       : %u", plotFile->K() );
    Log::Line( "Unpacked: %s", options.unpacked? "true" : "false" );;

    const uint64 plotC3ParkCount = plotFile->TableSize( PlotTable::C1 ) / sizeof( uint32 ) - 1;
    Log::Line( "C3 Parks: %llu", plotC3ParkCount );
    Log::Line( "" );


    // Duplicate the plot file,     
    ThreadPool pool( threadCount );
    
    UnpackedK32Plot unpackedPlot;
    if( options.unpacked )
    {
        unpackedPlot = UnpackedK32Plot::Load( plotFiles, pool, threadCount );
        exit( 0 );
    }

    MTJobRunner<ValidateJob> jobs( pool );

    std::mutex logLock;
    
    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.logLock     = &logLock;
        job.plotFile    = plotFiles[i];
        job.startOffset = options.startOffset;
        job.failCount   = 0;
    }

    jobs.Run( threadCount );

    uint64 proofFailCount = 0;
    for( uint32 i = 0; i < threadCount; i++ )
        proofFailCount += jobs[i].failCount;

    if( proofFailCount )
        Log::Line( "Plot has %llu invalid proofs." );
    else
        Log::Line( "Perfect plot! All proofs are valid." );

    return proofFailCount == 0;
}



//-----------------------------------------------------------
void ValidateJob::Log( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    
    logLock->lock();
    fprintf( stdout, "[%3u] ", JobId() );
    vfprintf( stdout, msg, args );
    putc( '\n', stdout );
    logLock->unlock();

    va_end( args );
}

//-----------------------------------------------------------
void ValidateJob::Run()
{
    PlotReader plot( *plotFile );
    
    const uint32 k = plotFile->K();

    uint64 c3ParkCount = 0;
    uint64 c3ParkEnd   = 0;
    
    const uint32 threadCount     = this->JobCount();
    const uint64 plotC3ParkCount = plotFile->TableSize( PlotTable::C1 ) / sizeof( uint32 ) - 1;
    
    c3ParkCount = plotC3ParkCount / threadCount;

    uint64 startC3Park = this->JobId() * c3ParkCount;

    {
        uint64 trailingParks = plotC3ParkCount - c3ParkCount * threadCount;
        
        if( this->JobId() < trailingParks )
            c3ParkCount ++;

        startC3Park += std::min( trailingParks, (uint64)this->JobId() );
    }

    c3ParkEnd = startC3Park + c3ParkCount;

    if( startOffset > 0.0f )
    {
        startC3Park += std::min( c3ParkCount, (uint64)( c3ParkCount * startOffset ) );
        c3ParkCount = c3ParkEnd - startC3Park;
    }

    Log( "Park range: %10llu..%-10llu  Park count: %llu", startC3Park, c3ParkEnd, c3ParkCount );

    ///
    /// Start validating C3 parks
    ///
    uint64* f7Entries = bbcalloc<uint64>( kCheckpoint1Interval );
    memset( f7Entries, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    uint64* p7Entries = bbcalloc<uint64>( kEntriesPerPark );
    memset( p7Entries, 0, sizeof( kEntriesPerPark ) * sizeof( uint64 ) );


    uint64 curPark7 = 0;
    if( JobId() == 0 )
        FatalIf( !plot.ReadP7Entries( 0, p7Entries ), "Failed to read P7 0." );
    
    uint64 proofFailCount = 0;
    uint64 fullProofXs[PROOF_X_COUNT];

    for( uint64 c3ParkIdx = startC3Park; c3ParkIdx < c3ParkEnd; c3ParkIdx++ )
    {
        const auto timer = TimerBegin();

        const int64 f7EntryCount = plot.ReadC3Park( c3ParkIdx, f7Entries );

        FatalIf( f7EntryCount < 0, "Could not read C3 park %llu.", c3ParkIdx );
        ASSERT( f7EntryCount <= kCheckpoint1Interval );

        const uint64 f7IdxBase = c3ParkIdx * kCheckpoint1Interval;

        for( uint32 e = 0; e < (uint32)f7EntryCount; e++ )
        {
            const uint64 f7Idx       = f7IdxBase + e;
            const uint64 p7ParkIndex = f7Idx / kEntriesPerPark;
            const uint64 f7          = f7Entries[e];

            if( p7ParkIndex != curPark7 )
            {
                // ASSERT( p7ParkIndex == curPark7+1 );
                curPark7 = p7ParkIndex;

                FatalIf( !plot.ReadP7Entries( p7ParkIndex, p7Entries ), "Failed to read P7 %llu.", p7ParkIndex );
            }

            const uint64 p7LocalIdx = f7Idx - p7ParkIndex * kEntriesPerPark;
            const uint64 t6Index    = p7Entries[p7LocalIdx];

            bool success = true;

            if( k <= 32 )
                success = FetchProof<true>( plot, t6Index, fullProofXs );
            else
                success = FetchProof<false>( plot, t6Index, fullProofXs );

            if( success )
            {
                // ReorderProof( plot, fullProofXs );   // <-- No need for this for validation
                
                // Now we can validate the proof
                uint64 outF7;

                if( ValidateFullProof( plot, fullProofXs, outF7 ) )
                    success = f7 == outF7;
                else
                    success = false;
            }
            else
            {
                success = false;
                Log( "Park %llu proof fetch failed for f7[%llu] local(%llu) = %llu ( 0x%016llx ) ", 
                   c3ParkIdx, f7Idx, e, f7, f7 );
            }

            if( !success )
            {
                proofFailCount++;
            }
        }

        const double elapsed = TimerEnd( timer );
        Log( "%10llu..%-10llu ( %3.2lf%% ) C3 Park Validated in %2.2lf seconds | Proofs Failed: %llu", 
                c3ParkIdx, c3ParkEnd-1, 
                (double)(c3ParkIdx-startC3Park) / c3ParkCount * 100, elapsed,
                proofFailCount );
    }

    // All done
    this->failCount = proofFailCount;
}


//-----------------------------------------------------------
template<bool Use64BitLpToSquare>
bool FetchProof( PlotReader& plot, uint64 t6LPIndex, uint64 fullProofXs[PROOF_X_COUNT] )
{
    uint64 lpIndices[2][PROOF_X_COUNT];
    // memset( lpIndices, 0, sizeof( lpIndices ) );

    uint64* lpIdxSrc = lpIndices[0];
    uint64* lpIdxDst = lpIndices[1];

    *lpIdxSrc = t6LPIndex;

    // Fetch line points to back pointers going through all our tables
    // from 6 to 1, grabbing all of the x's that make up a proof.
    uint32 lookupCount = 1;

    for( TableId table = TableId::Table6; table >= TableId::Table1; table-- )
    {
        ASSERT( lookupCount <= 32 );

        for( uint32 i = 0, dst = 0; i < lookupCount; i++, dst += 2 )
        {
            const uint64 idx = lpIdxSrc[i];

            uint128 lp = 0;
            if( !plot.ReadLP( table, idx, lp ) )
                return false;

            BackPtr ptr;
            if constexpr ( Use64BitLpToSquare )
                ptr = LinePointToSquare64( (uint64)lp );
            else
                ptr = LinePointToSquare( lp );

            lpIdxDst[dst+0] = ptr.y;
            lpIdxDst[dst+1] = ptr.x;
        }

        lookupCount <<= 1;

        std::swap( lpIdxSrc, lpIdxDst );
        // memset( lpIdxDst, 0, sizeof( uint64 ) * PROOF_X_COUNT );
    }

    // Full proof x's will be at the src ptr
    memcpy( fullProofXs, lpIdxSrc, sizeof( uint64 ) * PROOF_X_COUNT );
    return true;
}

//-----------------------------------------------------------
bool ValidateFullProof( PlotReader& plot, uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 )
{
    const uint32 k = plot.PlotFile().K();

    uint64   fx  [PROOF_X_COUNT];
    MetaBits meta[PROOF_X_COUNT];

    // Convert these x's to f1 values
    {
        const uint32 xShift = k - kExtraBits;
        
        // Prepare ChaCha key
        byte key[32] = { 1 };
        memcpy( key + 1, plot.PlotFile().PlotId(), 31 );

        chacha8_ctx chacha;
        chacha8_keysetup( &chacha, key, 256, NULL );

        // Enough to hold 2 cha-cha blocks since a value my span over 2 blocks
        byte blocks[kF1BlockSize*2];

        for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
        {
            const uint64 x        = fullProofXs[i];
            const uint64 blockIdx = x * k / kF1BlockSizeBits; 

            chacha8_get_keystream( &chacha, blockIdx, 2, blocks );

            // Get the starting and end locations of y in bits relative to our block
            const uint64 bitStart = x * k - blockIdx * kF1BlockSizeBits;

            CPBitReader hashBits( blocks, sizeof( blocks ) * 8 );
            hashBits.Seek( bitStart );

            // uint64 y = SliceUInt64FromBits( blocks, bitStart, k ); // #TODO: Figure out what's wrong with this method.
            uint64 y = hashBits.Read64( k );
            y = ( y << kExtraBits ) | ( x >> xShift );

            fx  [i] = y;
            meta[i] = MetaBits( x, k );
        }
    }

    // Forward propagate f1 values to get the final f7
    uint32 iterCount = PROOF_X_COUNT;
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++, iterCount >>= 1)
    {
        for( uint32 i = 0, dst = 0; i < iterCount; i+= 2, dst++ )
        {
            uint64 y0 = fx[i+0];
            uint64 y1 = fx[i+1];

            const MetaBits* lMeta = &meta[i+0];
            const MetaBits* rMeta = &meta[i+1];

            if( y0 > y1 ) 
            {
                std::swap( y0, y1 );
                std::swap( lMeta, rMeta );
            }

            // Must be on the same group
            if( !FxMatch( y0, y1 ) )
                return false;

            // FxGen
            uint64 outY;
            MetaBits outMeta;
            FxGen( table, k, y0, *lMeta, *rMeta, outY, outMeta );

            fx  [dst] = outY;
            meta[dst] = outMeta;
        }
    }

    outF7 = fx[0] >> kExtraBits;

    return true;
}


// #TODO: Avoid code duplication here? At least for f1
//-----------------------------------------------------------
void ReorderProof( PlotReader& plot, uint64 fullProofXs[PROOF_X_COUNT] )
{
    const uint32 k = plot.PlotFile().K();

    uint64   fx  [PROOF_X_COUNT];
    MetaBits meta[PROOF_X_COUNT];

    uint64  xtmp[PROOF_X_COUNT];
    uint64* xs = fullProofXs;

    // Convert these x's to f1 values
    GetProofF1( k, plot.PlotFile().PlotId(), fullProofXs, fx );
    for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
        meta[i] = MetaBits( xs[i], k );

    // Forward propagate f1 values to get the final f7
    uint32 iterCount = PROOF_X_COUNT;
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++, iterCount >>= 1)
    {
        for( uint32 i = 0, dst = 0; i < iterCount; i+= 2, dst++ )
        {
            uint64 y0 = fx[i+0];
            uint64 y1 = fx[i+1];

            const MetaBits* lMeta = &meta[i+0];
            const MetaBits* rMeta = &meta[i+1];

            if( y0 > y1 ) 
            {
                std::swap( y0, y1 );
                std::swap( lMeta, rMeta );

                // Swap X's so far that have generated this y
                const uint32 count = 1u << ((int)table-1);
                uint64* x = xs + i * count;
                bbmemcpy_t( xtmp   , x      , count );
                bbmemcpy_t( x      , x+count, count );
                bbmemcpy_t( x+count, xtmp   , count );
            }

            // FxGen
            uint64 outY;
            MetaBits outMeta;
            FxGen( table, k, y0, *lMeta, *rMeta, outY, outMeta );

            fx  [dst] = outY;
            meta[dst] = outMeta;
        }
    }
}

//-----------------------------------------------------------
void GetProofF1( uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64 fx[PROOF_X_COUNT] )
{
    const uint32 xShift = k - kExtraBits;
        
    // Prepare ChaCha key
    byte key[32] = { 1 };
    memcpy( key + 1, plotId, 31 );

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    // Enough to hold 2 cha-cha blocks since a value my span over 2 blocks
    byte blocks[kF1BlockSize*2];

    for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
    {
        const uint64 x        = fullProofXs[i];
        const uint64 blockIdx = x * k / kF1BlockSizeBits; 

        chacha8_get_keystream( &chacha, blockIdx, 2, blocks );

        // Get the starting and end locations of y in bits relative to our block
        const uint64 bitStart = x * k - blockIdx * kF1BlockSizeBits;

        CPBitReader hashBits( blocks, sizeof( blocks ) * 8 );
        hashBits.Seek( bitStart );

        // uint64 y = SliceUInt64FromBits( blocks, bitStart, k ); // #TODO: Figure out what's wrong with this method.
        uint64 y = hashBits.Read64( k );
        y = ( y << kExtraBits ) | ( x >> xShift );

        fx[i] = y;
    }
}

//-----------------------------------------------------------
bool FxMatch( uint64 yL, uint64 yR )
{
    const uint64 groupL = yL / kBC;
    const uint64 groupR = yR / kBC;

    if( groupR - groupL != 1 )
        return false;

    // Groups are adjacent, check if the y values actually match
    const uint16 parity = groupL & 1;

    const uint64 groupLRangeStart = groupL * kBC;
    const uint64 groupRRangeStart = groupR * kBC;

    const uint64 localLY = yL - groupLRangeStart;
    const uint64 localRY = yR - groupRRangeStart;
    
    for( int iK = 0; iK < kExtraBitsPow; iK++ )
    {
        const uint64 targetR = L_targets[parity][localLY][iK];
        
        if( targetR == localRY )
            return true;
    } 

    return false;
}

//-----------------------------------------------------------
void FxGen( const TableId table, const uint32 k, 
            const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
            uint64& outY, MetaBits& outMeta )
{
    FxBits input( y, k + kExtraBits );

    if( table < TableId::Table4 )
    {
        outMeta = metaL + metaR;
        input += outMeta;
    }
    else
    {
        input += metaL;
        input += metaR;
    }

    byte inputBytes[64];
    byte hashBytes [32];

    input.ToBytes( inputBytes );

    blake3_hasher hasher;
    blake3_hasher_init    ( &hasher );
    blake3_hasher_update  ( &hasher, inputBytes, input.LengthBytes() );
    blake3_hasher_finalize( &hasher, hashBytes, sizeof( hashBytes ) );

    outY = BytesToUInt64( hashBytes ) >> ( 64 - (k + kExtraBits) );

    if( table >= TableId::Table4 && table < TableId::Table7 )
    {
        size_t multiplier = 0;
        switch( table )
        {
            case TableId::Table4: multiplier = TableMetaOut<TableId::Table4>::Multiplier; break;
            case TableId::Table5: multiplier = TableMetaOut<TableId::Table5>::Multiplier; break;
            case TableId::Table6: multiplier = TableMetaOut<TableId::Table6>::Multiplier; break;
            default: 
                ASSERT( 0 );
                break;
        }

        const uint32 metaBits  = k * multiplier;
        const uint32 yBits     = k + kExtraBits;
        const uint32 startByte = yBits / 8 ;
        const uint32 startBit  = yBits - startByte * 8;

        outMeta = MetaBits( hashBytes + startByte, metaBits, startBit );
    }
}

//-----------------------------------------------------------
/// Convertes 8 bytes to uint64 and endian-swaps it.
/// This takes any byte alignment, so that bytes does
/// not have to be aligned to 64-bit boundary.
/// This is for compatibility for how chiapos extracts
/// bytes into integers.
//-----------------------------------------------------------
inline uint64 BytesToUInt64( const byte bytes[8] )
{
    uint64 tmp;
    memcpy( &tmp, bytes, sizeof( uint64 ) );
    return Swap64( tmp );
}

//-----------------------------------------------------------
// Treats bytes as a set of 64-bit big-endian fields,
// from which it will extract a whole 64-bit value
// at the given bit offset. 
// The result may be truncated if the requested number of 
// bits + the number of bits overflows the 64-bit field.
// That is, if the local bit offset in the target bit field
// + the bitCount is greater than 64.
// This function is for compatibility with the way chiapos
// slices bits off of binary byte blobs.
//-----------------------------------------------------------
inline uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount )
{
    ASSERT( bitCount <= 64 );
     
    // #TODO: This is wrong, it's not treating the bytes as 64-bit fields.
    //        So that we may have swapped at the wrong position.
    //        In fact we might have fields that span 2 64-bit values.
    //        So we need to split it into 2, and do 2 swaps.
    const uint64 startByte = bitOffset / 8;
    bytes += startByte;

    // Convert bit offset to be local to the uint64 field
    bitOffset -= ( bitOffset >> 6 ) * 64; // bitOffset >> 6 == bitOffset / 64

    uint64 field = BytesToUInt64( bytes );
    
    field <<= bitOffset;     // Start bits from the MSBits
    field >>= 64 - bitCount; // Take the MSbits

    return field;
}


//-----------------------------------------------------------
UnpackedK32Plot UnpackedK32Plot::Load( IPlotFile** plotFile, ThreadPool& pool, uint32 threadCount )
{
    ASSERT( plotFile );
    const uint32 k = plotFile[0]->K();
    ExitIf( k != 32, "Only k=32 plots are supported for unpacked validation." );

    threadCount = threadCount == 0 ? pool.ThreadCount() : threadCount;

    auto LoadBackPtrTable = [=]() {
        
    };

    UnpackedK32Plot plot;

    PlotReader plotReader( *plotFile[0] );
    const uint64 f7Count = plotReader.GetMaxF7EntryCount(); ExitIf( f7Count < 1, "No F7s found." );

    // Load F7s
    {
        Log::Line( "Unpacking f7 values..." );
        uint32* f7 = bbcvirtallocboundednuma<uint32>( f7Count );

        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

            PlotReader reader( *plotFile[self->_jobId] );

            const uint64 plotParkCount = reader.GetC3ParkCount();

            uint64 parkCount, parkOffset, parkEnd;
            GetThreadOffsets( self, plotParkCount, parkCount, parkOffset, parkEnd );

            uint64 f7Buffer[kCheckpoint1Interval];
            uint32* f7Writer = f7 + parkOffset * kCheckpoint1Interval;

            for( uint64 i = parkOffset; i < parkEnd; i++ )
            {
                const int64 entryCount = reader.ReadC3Park( i, f7Buffer );

                ExitIf( entryCount == 0, "Empty C3 park @ %llu.", i );
                ExitIf( entryCount < kCheckpoint1Interval && i+1 != parkEnd, "C3 park is not full and it is not the last park." );

                for( int64 e = 0; e < entryCount; e++ )
                    f7Writer[e] = (uint32)f7Buffer[e];

                f7Writer += entryCount;
            }
        });

        plot.f7.length = f7Count;
        plot.f7.values = f7;
    }
    
    // Read Park 7
    Log::Line( "Reding park 7..." );
    {
        const size_t parkSize = CalculatePark7Size( k );
    }

}

//-----------------------------------------------------------
bool UnpackedK32Plot::FetchProof( uint64 index )
{
    return false;
}