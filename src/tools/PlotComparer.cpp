#include "io/FileStream.h"
#include "ChiaConsts.h"
#include "tools/PlotReader.h"
#include "util/Util.h"
#include "util/Log.h"
#include "util/CliParser.h"
#include "util/BitView.h"
#include "plotting/PlotTools.h"
#include "plotting/GlobalPlotConfig.h"
#include "plotdisk/jobs/IOJob.h"
#include "threading/ThreadPool.h"
#include <vector>
#include <algorithm>


void DumpPlotHeader( FilePlot& plot );
void TestTable( TableId table, FilePlot& ref, FilePlot& tgt );
void TestC3Table( FilePlot& ref, FilePlot& tgt );
void TestTable( FilePlot& ref, FilePlot& tgt, TableId table );

void UnpackPark7( uint32 k, const byte* srcBits, uint64* dstEntries );

void DumpP7( FilePlot& plot, const char* path );

Span<uint> ReadC1Table( FilePlot& plot );

//-----------------------------------------------------------
const char USAGE[] = R"(plotcmp <plot_a_path> <plot_b_path>

Compares 2 plots for matching tables.

[ARGUMENTS]
<plot_*_path> : Path to the plot files to be compared.
 -h, --help   : Print this help message and exit.
)";

//-----------------------------------------------------------
void PlotCompareMainPrintUsage()
{
    Log::Line( "" );
    Log::Flush();
}


/// Compares 2 plot's tables
//-----------------------------------------------------------
struct PlotCompareOptions
{
    const char* plotAPath = "";
    const char* plotBPath = "";
};

//-----------------------------------------------------------
void PlotCompareMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    PlotCompareOptions opts;

    while( cli.HasArgs() )
    {
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            PlotCompareMainPrintUsage();
            exit( 0 );
        }
        else
            break;
    }

    opts.plotAPath = cli.ArgConsume();
    opts.plotBPath = cli.ArgConsume();

    FilePlot refPlot; // Reference
    FilePlot tgtPlot; // Target

    {
        const char* refPath = opts.plotAPath;
        const char* tgtPath = opts.plotBPath;

        refPlot.Open( refPath );
        tgtPlot.Open( tgtPath );

        Log::Line( "[Reference Plot]" );
        DumpPlotHeader( refPlot );
        Log::NewLine();
        Log::Line( "[Target Plot]" );
        DumpPlotHeader( tgtPlot );
        Log::NewLine();
    }

    FatalIf( refPlot.K() != 32, "Plot A is k%u. Only k32 plots are currently supported.", refPlot.K() );
    FatalIf( tgtPlot.K() != 32, "Plot B is k%u. Only k32 plots are currently supported.", tgtPlot.K() );

    FatalIf( !MemCmp( refPlot.PlotId(), tgtPlot.PlotId(), BB_PLOT_ID_LEN ), "Plot id mismatch." );
    // FatalIf( !MemCmp( refPlot.PlotMemo(), tgtPlot.PlotMemo(), std::min( refPlot.PlotMemoSize(), tgtPlot.PlotMemoSize() ) ), "Plot memo mismatch." );
    FatalIf( refPlot.K() != tgtPlot.K(), "K value mismatch." );

    FatalIf( refPlot.CompressionLevel() != tgtPlot.CompressionLevel(), 
        "Compression mismatch. %u != %u.", refPlot.CompressionLevel(), tgtPlot.CompressionLevel() );

    // Test P7, dump it
    // DumpP7( refPlot, "/mnt/p5510a/reference/p7.tmp" );

    // TestC3Table( refPlot, tgtPlot ); Exit( 0 );

    // TestTable( refPlot, tgtPlot, TableId::Table7 );
    // TestTable( refPlot, tgtPlot, TableId::Table3 );

    TestC3Table( refPlot, tgtPlot );

    for( TableId table = TableId::Table1; table <= TableId::Table7; table++ )
        TestTable( refPlot, tgtPlot, table );

    // TestC3Table( refPlot, tgtPlot );
}

//-----------------------------------------------------------
void DumpPlotHeader( FilePlot& p )
{
    // Log::Line( "Id: %")
    Log::Line( "K: %u", p.K() );
    Log::Line( "Compression Level: %u", p.CompressionLevel() );

    Log::Line( "Table Addresses:" );
    for( uint32 i = 0; i < 10; i++ )
        Log::Line( " [%2u] : 0x%016llx", i+1, (llu)p.TableAddress( (PlotTable)i ) );

    if( p.Version() >= PlotVersion::v2_0 )
    {
        const auto sizes = p.TableSizes();

        Log::Line( "Table Sizes:" );
        for( uint32 i = 0; i < 10; i++ )
            Log::Line( " [%2u] : %-12llu B | %llu MiB", i+1, (llu)sizes[i], (llu)(sizes[i] BtoMB) );
    }
}

//-----------------------------------------------------------
Span<uint> ReadC1Table( FilePlot& plot )
{
    const size_t tableSize  = plot.TableSize( PlotTable::C1 );
    const uint32 entryCount = (uint)( tableSize / sizeof( uint32 ) );

    uint32* c1 = bbvirtalloc<uint32>( tableSize );

    FatalIf( !plot.SeekToTable( PlotTable::C1 ), "Failed to seek to table C1." );
    plot.Read( tableSize, c1 );

    for( uint i = 0; i < entryCount; i++ )
        c1[i] = Swap32( c1[i] );

    return Span<uint>( c1, entryCount );
}

//-----------------------------------------------------------
Span<uint> ReadC2Table( FilePlot& plot )
{
    const size_t tableSize  = plot.TableSize( PlotTable::C2 );
    const uint32 entryCount = (uint)( tableSize / sizeof( uint32 ) );
    
    uint32* c2 = bbvirtalloc<uint32>( tableSize );

    FatalIf( !plot.SeekToTable( PlotTable::C2 ), "Failed to seek to table C1." );
    plot.Read( tableSize, c2 );

    for( uint i = 0; i < entryCount; i++ )
        c2[i] = Swap32( c2[i] );

    return Span<uint>( c2, entryCount );
}

//-----------------------------------------------------------
void TestC3Table( FilePlot& ref, FilePlot& tgt )
{
    Log::Line( "Reading C tables..." );

    // Read C1 so that we know how many parks we got
    Span<uint> refC1 = ReadC1Table( ref );
    Span<uint> tgtC1 = ReadC1Table( tgt );

    // Compare C1
    const uint64 c1Length = (uint64)std::min( refC1.length, tgtC1.length );
    {
        Log::Line( "Validating C1 table... " );
        uint64 failCount = 0;
        for( uint i = 0; i < c1Length; i++ )
        {
            if( refC1[i] != tgtC1[i] )
            {
                if( failCount == 0 )
                    Log::Line( "C1 table mismatch @ index %u. Ref: %u Tgt: %u", i, refC1[i], tgtC1[i] );

                failCount++;
            }
        }

        if( failCount == 0 )
            Log::Line( "Success!" );
        else
            Log::Line( "C1 table mismatch: %llu entries failed.", failCount );
    }

    // Check C2
    Span<uint> refC2 = ReadC2Table( ref );
    Span<uint> tgtC2 = ReadC2Table( tgt );

    const uint64 c2Length = (uint64)std::min( refC2.length, tgtC2.length );
    {
        Log::Line( "Validating C2 table... " );

        uint64 failCount = 0;
        for( uint i = 0; i < c2Length; i++ )
        {
            if( refC2[i] != tgtC2[i] )
            {
                if( failCount == 0 )
                    Log::Line( "C2 table mismatch @ index %u. Ref: %u Tgt: %u", i, refC2[i], tgtC2[i] );

                failCount++;
            }
        }

        if( failCount == 0 )
            Log::Line( "Success!" );
        else
            Log::Line( "C2 table mismatch: %llu entries failed.", failCount );
    }

    const uint64 f7Count = c1Length * (uint64)kCheckpoint1Interval;
    Log::Line( "F7 Count: ~%llu", f7Count );

    Log::Line( "Validating C3 table..." );
    {
        const size_t refC3Size = ref.TableSize( PlotTable::C3 );
        const size_t tgtC3Size = tgt.TableSize( PlotTable::C3 );

        byte* refC3 = bbvirtalloc<byte>( refC3Size );
        byte* tgtC3 = bbvirtalloc<byte>( tgtC3Size );

        FatalIf( !ref.SeekToTable( PlotTable::C3 ), "Failed to seek ref plot to C3 table." );
        FatalIf( !tgt.SeekToTable( PlotTable::C3 ), "Failed to seek tgt plot to C3 table." );

        FatalIf( (ssize_t)refC3Size != ref.Read( refC3Size, refC3 ), "Failed to read ref C3 table." );
        FatalIf( (ssize_t)tgtC3Size != tgt.Read( tgtC3Size, tgtC3 ), "Failed to read tgt C3 table." );

        // const size_t c3Size = std::min( refC3Size, tgtC3Size );

        const int64  parkCount  = (int64)(c1Length - 1);
        const size_t c3ParkSize = CalculateC3Size();

        const byte* refC3Reader = refC3;
        const byte* tgtC3Reader = tgtC3;

        std::vector<int64> failures;

        for( int64 i = 0; i < parkCount; i++ )
        {
            const uint16 refSize = Swap16( *(uint16*)refC3Reader );
            const uint16 tgtSize = Swap16( *(uint16*)tgtC3Reader );

            if( refSize != tgtSize || !MemCmp( refC3Reader, tgtC3Reader, tgtSize ) )
            {
                Log::Line( " C3 park %lld failed.", i );
                failures.push_back( i );
            }
            // FatalIf( !MemCmp( refC3Reader, tgtC3Reader, c3ParkSize ), 
            //     "C3 park mismatch @ park %lld / %lld", i, parkCount );

            refC3Reader += c3ParkSize;
            tgtC3Reader += c3ParkSize;
        }

        if( failures.size() < 1 )
            Log::Line( "Success!" );
        else
            Log::Line( "%llu / %lld C3 parks failed!", failures.size(), parkCount );
        
        SysHost::VirtualFree( refC3 );
        SysHost::VirtualFree( tgtC3 );
    }

    SysHost::VirtualFree( refC2.values );
    SysHost::VirtualFree( tgtC2.values );
    SysHost::VirtualFree( refC1.values );
    SysHost::VirtualFree( tgtC1.values );
}

//-----------------------------------------------------------
uint64 CompareP7( FilePlot& ref, FilePlot& tgt, const byte* p7RefBytes, const byte* p7TgtBytes, const int64 parkCount )
{
    // Double-buffer parks at a time so that we can compare entries across parks
    uint64 refParks[2][kEntriesPerPark];
    uint64 tgtParks[2][kEntriesPerPark];

    uint64 refMatch[kEntriesPerPark];
    uint64 tgtMatch[kEntriesPerPark];

    // Unpack first park
    ASSERT( parkCount );

    const size_t parkSize = CalculatePark7Size( ref.K() );

    UnpackPark7( ref.K(), p7RefBytes, refParks[0] );
    UnpackPark7( tgt.K(), p7TgtBytes, tgtParks[0] );
    p7RefBytes += parkSize;
    p7TgtBytes += parkSize;

    uint64 parkFailCount = 0;

    for( int64 p = 0; p < parkCount; p++ )
    {
        const bool isLastPark = p + 1 == parkCount;

        // Load the next park, if we can
        if( !isLastPark )
        {
            UnpackPark7( ref.K(), p7RefBytes, refParks[1] );
            UnpackPark7( tgt.K(), p7TgtBytes, tgtParks[1] );
            p7RefBytes += parkSize;
            p7TgtBytes += parkSize;
        }

        const uint64* ref = refParks[0];
        const uint64* tgt = tgtParks[0];

        const int64 entriesEnd = isLastPark ? kEntriesPerPark : kEntriesPerPark * 2;

        for( int64 i = 0; i < kEntriesPerPark; i++ )
        {
            if( ref[i] == tgt[i] )
                continue;

            // Potential out-of-order entries
            // Try sorting and matching entry pairs until we run out of entries
            int64 j = i + 1;

            bool matched = false;
            for( ; j < entriesEnd; j++ )
            {
                const size_t matchCount = (size_t)( j - i ) + 1;

                bbmemcpy_t( refMatch, ref+i, matchCount );
                bbmemcpy_t( tgtMatch, tgt+i, matchCount );
                
                std::sort( refMatch, refMatch+matchCount );
                std::sort( tgtMatch, tgtMatch+matchCount );

                if( memcmp( refMatch, tgtMatch, matchCount * sizeof( uint64 ) ) == 0 )
                {
                    matched = true;
                    break;
                }
                else
                    continue;
            }

            if( !matched )
            {
                Log::Line( "T7 park %lld failed. First entry failed at index %lld", p, i );
                parkFailCount++;
            }
        
            i = j;
        }

        // Set next park as current park
        memcpy( refParks[0], refParks[1], sizeof( uint64 ) * kEntriesPerPark );
        memcpy( tgtParks[0], tgtParks[1], sizeof( uint64 ) * kEntriesPerPark );
    }
    
    return parkFailCount;
}

//-----------------------------------------------------------
void TestTable( FilePlot& ref, FilePlot& tgt, TableId table )
{
    if( table == TableId::Table1 && tgt.CompressionLevel() > 0 )
        return;

    if( table == TableId::Table2 && tgt.CompressionLevel() >= 40 )
        return;

    // if( table == TableId::Table7 ) return;

    Log::Line( "Reading Table %u...", table+1 );

    const uint32 numTablesDropped = tgt.CompressionLevel() >= 40 ? 2 :
                                    tgt.CompressionLevel() >= 1 ? 1 : 0;

    const size_t parkSize = table < TableId::Table7 ? 
                                (uint)table == numTablesDropped ? 
                                    GetCompressionInfoForLevel( tgt.CompressionLevel() ).tableParkSize : CalculateParkSize( table ) :
                                CalculatePark7Size( ref.K() );

    const size_t sizeRef = ref.TableSize( (PlotTable)table );
    const size_t sizeTgt = tgt.TableSize( (PlotTable)table );

    byte* tableParksRef = bbvirtalloc<byte>( sizeRef );
    byte* tableParksTgt = bbvirtalloc<byte>( sizeTgt );

    const size_t tableSize = std::min( sizeRef, sizeTgt );
    const int64  parkCount = (int64)( tableSize / parkSize );

    FatalIf( !ref.SeekToTable( (PlotTable)table ), "Failed to seek to table %u on reference plot.", (uint32)table+1 );
    FatalIf( !tgt.SeekToTable( (PlotTable)table ), "Failed to seek to table %u on target plot.", (uint32)table+1 );

    {
        const ssize_t refRead = ref.Read( tableSize, tableParksRef );
        FatalIf( (ssize_t)tableSize != refRead, "Failed to read reference table %u.", (uint32)table+1 );
        
        const ssize_t tgtRead = tgt.Read( tableSize, tableParksTgt );
        FatalIf( (ssize_t)tableSize != tgtRead, "Failed to read target table %u.", (uint32)table+1 );

    }

    const byte* parkRef = tableParksRef;
    const byte* parkTgt = tableParksTgt;
    Log::Line( "Validating Table %u...", table+1 );

    uint64 stubBitSize = (ref.K() - kStubMinusBits);
    if( ref.CompressionLevel() > 0 )
    {
        auto cInfo = GetCompressionInfoForLevel( ref.CompressionLevel() );
        stubBitSize = cInfo.stubSizeBits;
    }

    const size_t stubSectionBytes = CDiv( (kEntriesPerPark - 1) * stubBitSize, 8 );


    uint64 failureCount = 0;
    if( table == TableId::Table7 )
    {
        // Because entries can be found in different orders in P7,
        // we compare them in a different manner than the other parks.
        // (this can happen because these entries are sorted on f7,
        //  and if there's multiple entries with the same value
        //  there's no guarantee an implementation will sort it
        //  into the same index as another )
        failureCount = CompareP7( ref, tgt, tableParksRef, tableParksTgt, parkCount );
    }
    else
    {
        for( int64 i = 0; i < parkCount; i++ )
        {
            // Ignore buffer zone
            const uint16 pRefCSize = *(uint16*)(parkRef + stubSectionBytes + sizeof( uint64 ) );
            const uint16 pTgtCSize = *(uint16*)(parkTgt + stubSectionBytes + sizeof( uint64 ) );

            bool failed = pRefCSize != pTgtCSize;

            if( !failed )
            {
                const size_t realParkSize = sizeof( uint64 ) + stubSectionBytes + pRefCSize;
                failed = !MemCmp( parkRef, parkTgt, realParkSize );
            }
            // if( pRefCSize != pTgtCSize || !MemCmp( parkRef, parkTgt, parkSize ) )

            {
                // bool failed     = true;

                if( failed )
                {
                    // bool stubsEqual = MemCmp( parkRef, parkTgt, stubSectionBytes + sizeof( uint64 ) );
                    Log::Line( " T%u park %lld failed.", table+1, i );
                    failureCount++;
                }

                // {
                //     const byte* refBytes = parkRef;
                //     const byte* tgtBytes = parkTgt;

                //     // if( table != TableId::Table7 )
                //     // {
                //     //     refBytes += 8;
                //     // }

                //     for( uint64 b = 0; b < parkSize; b++ )
                //     {
                //         if( refBytes[b] != tgtBytes[b] )
                //         {
                //             Log::Line( "Byte mismatch @ byte %llu: 0x%02x (%3d) != 0x%02x (%3d)", b,
                //              (int)refBytes[b], (int)refBytes[b], (int)tgtBytes[b], (int)tgtBytes[b] );
                //             break;
                //         }
                //     }
                // }
            }

            parkRef += parkSize;
            parkTgt += parkSize;
        }
    }


    if( failureCount < 1 )
        Log::Line( "Success!" );
    else
        Log::Line( "%llu / %lld Table %u parks failed.", failureCount, parkCount, table+1 );

    SysHost::VirtualFree( tableParksRef );
    SysHost::VirtualFree( tableParksTgt );
}

// Unpack a single park 7,
// ensure srcBits is algined to uint64
//-----------------------------------------------------------
void UnpackPark7( const uint32 k, const byte* srcBits, uint64* dstEntries )
{
    ASSERT( ((uintptr_t)srcBits & 7 ) == 0 );

    const uint32 bitsPerEntry = k + 1;
    CPBitReader reader( srcBits, CalculatePark7Size( k ) * 8, 0  );

    for( int32 i = 0; i < kEntriesPerPark; i++ )
        dstEntries[i] = reader.Read64Aligned( bitsPerEntry );
}

//-----------------------------------------------------------
void DumpP7( FilePlot& plot, const char* path )
{
    FileStream file;
    FatalIf( !file.Open( path, FileMode::Create, FileAccess::Write, FileFlags::LargeFile | FileFlags::NoBuffering ),
        "Failed to open file at '%s' with error: %d.", path, file.GetError() );

    const size_t parkSize = CalculatePark7Size( plot.K() );

    const size_t tableSize  = plot.TableSize( PlotTable::Table7 );
    const int64  parkCount  = (int64)( tableSize / parkSize );
    const uint64 numEntries = (uint64)parkCount * kEntriesPerPark;

    byte* p7Bytes = bbvirtalloc<byte>( tableSize );

    Log::Line( "Reading Table7..." );
    // plot.ReadTable( (int)TableId::Table7, p7Bytes );

    Log::Line( "Unpacking Table 7..." );
    uint64* entries = bbvirtalloc<uint64>( RoundUpToNextBoundaryT( (size_t)numEntries* sizeof( uint64 ), file.BlockSize() ) );

    const byte*   parkReader  = p7Bytes;
          uint64* entryWriter = entries;
    for( int64 i = 0; i < parkCount; i++ )
    {
        UnpackPark7( plot.K(), p7Bytes, entryWriter );

        parkReader  += parkSize;
        entryWriter += kEntriesPerPark;
    }

    Log::Line( "Writing to disk..." );
    const size_t blockSize = file.BlockSize();
    
    uint64* block = bbvirtalloc<uint64>( blockSize );   
    
    int err;
    FatalIf( !IOJob::WriteToFile( file, entries, numEntries * sizeof( uint64 ), block, blockSize, err ),
        "Entry write failed with error: %d.", err );

    Log::Line( "All done." );
    bbvirtfree( p7Bytes );
    bbvirtfree( entries );
    bbvirtfree( block   );
}


