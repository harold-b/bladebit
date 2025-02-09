#include "plotting/GlobalPlotConfig.h"
#include "util/CliParser.h"
#include "plotdisk/DiskPlotter.h"
#include "plotmem/MemPlotter.h"
#include "plotting/PlotTools.h"
#include "commands/Commands.h"
#include "Version.h"

#if PLATFORM_IS_UNIX
    #include <sys/resource.h>
#endif

#if BB_CUDA_ENABLED
    #include "../cuda/CudaPlotter.h"
#endif

static void ParseCommandLine( GlobalPlotConfig& cfg, IPlotter*& outPlotter, int argc, const char* argv[] );
static void PrintUsage();

// See IOTester.cpp
void IOTestMain( GlobalPlotConfig& gCfg, CliParser& cli );
void IOTestPrintUsage();

// MemTester.cpp
void MemTestMain( GlobalPlotConfig& gCfg, CliParser& cli );
void MemTestPrintUsage();

// PlotValidator.cpp
void PlotValidatorMain( GlobalPlotConfig& gCfg, CliParser& cli );
void PlotValidatorPrintUsage();

// PlotComparer.cpp
void PlotCompareMain( GlobalPlotConfig& gCfg, CliParser& cli );
void PlotCompareMainPrintUsage();


//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    // Install a crash handler to dump our stack traces
    SysHost::InstallCrashHandler();

#if _DEBUG
    Log::Line( "*** Warning: Debug mode is ENABLED ***" );
#endif

    IPlotter* plotter = nullptr;

    auto& cfg = *new GlobalPlotConfig{};
    ParseCommandLine( cfg, plotter, --argc, ++argv );


    const int64 plotCount = cfg.plotCount > 0 ? (int64)cfg.plotCount : std::numeric_limits<int64>::max();
    // int64 failCount = 0;

    byte   plotId  [BB_PLOT_ID_LEN];
    byte   plotMemo[BB_PLOT_MEMO_MAX_SIZE];
    uint16 plotMemoSize = 0;

    char   plotIdStr[BB_PLOT_ID_LEN*2+1];

    // Prepare the output path
    char*  plotOutPath      = nullptr;
    uint32 plotOutPathIndex = 0;
    {
        // Get the largest buffer needed
        size_t outFolderLengthMax = cfg.outputFolders[0].length();

        for( uint32 i = 1; i < cfg.outputFolderCount; i++ )
            outFolderLengthMax = std::max( outFolderLengthMax, cfg.outputFolders[i].length() );

        plotOutPath = new char[outFolderLengthMax + BB_COMPRESSED_PLOT_FILE_LEN_TMP + 2]; // + '/' + null terminator
    }

    // Start plotting
    for( int64 i = 0; i < plotCount; i++ )
    {
        // Generate a plot id and memo
        PlotTools::GeneratePlotIdAndMemo( plotId, plotMemo, plotMemoSize,
                                          *cfg.farmerPublicKey, cfg.poolPublicKey, cfg.poolContractPuzzleHash );

        // Apply debug plot id and/or memo
        if( cfg.plotIdStr )
            HexStrToBytes( cfg.plotIdStr, BB_PLOT_ID_LEN*2, plotId, BB_PLOT_ID_LEN );

        if( cfg.plotMemoStr )
        {
            const size_t memoLen = strlen( cfg.plotMemoStr );
            HexStrToBytes( cfg.plotMemoStr, memoLen, plotMemo, memoLen/2 );
        }

        // Convert plot id to string
        PlotTools::PlotIdToString( plotId, plotIdStr );

        // Set the plot file name & get the full path to it
        const char* plotFileName  = nullptr;
        const char* plotOutFolder = nullptr;
        {
            // Select the next output folder
            const std::string& curOutputDir = cfg.outputFolders[plotOutPathIndex++];
            plotOutPathIndex %= cfg.outputFolderCount;

            plotOutFolder = curOutputDir.data();

            memcpy( plotOutPath, curOutputDir.data(), curOutputDir.length() );

            plotFileName = plotOutPath + curOutputDir.length();
            PlotTools::GenPlotFileName( plotId, (char*)plotFileName, cfg.compressionLevel );
        }

        // Begin plot
        if( cfg.plotCount == 0 )
            Log::Line( "Generating plot %lld: %s", i+1, plotIdStr );
        else
            Log::Line( "Generating plot %lld / %u: %s", i+1, cfg.plotCount, plotIdStr );

        Log::Line( "Plot temporary file: %s", plotOutPath );


        if( cfg.showMemo )
        {
            char plotMemoStr[BB_PLOT_MEMO_MAX_SIZE*2+1];

            size_t numEncoded = 0;
            BytesToHexStr( plotMemo, plotMemoSize, plotMemoStr, sizeof( plotMemoStr ) - 1, numEncoded );
            plotMemoStr[numEncoded*2] = 0;

            Log::Line( "Plot Memo: %s", plotMemoStr );
        }
        Log::Line( "" );

        PlotRequest req = {};
        req.plotId       = plotId;
        req.memo         = plotMemo;
        req.memoSize     = plotMemoSize;
        req.outDir       = plotOutFolder;
        req.plotFileName = plotFileName;
        req.plotOutPath  = plotOutPath;
        req.isFirstPlot  = i == 0;
        req.IsFinalPlot  = i == plotCount-1;

        plotter->Run( req );
    }
}

//-----------------------------------------------------------
void ParseCommandLine( GlobalPlotConfig& cfg, IPlotter*& outPlotter, int argc, const char* argv[] )
{
    CliParser cli( argc, argv );

    const char* farmerPublicKey     = nullptr;
    const char* poolPublicKey       = nullptr;
    const char* poolContractAddress = nullptr;

    outPlotter        = nullptr;
    IPlotter* plotter = nullptr;

    while( cli.HasArgs() )
    {
        if( cli.ReadU32( cfg.threadCount, "-t", "--threads" ) )
            continue;
        else if( cli.ReadU32( cfg.plotCount, "-n", "--count" ) )
            continue;
        else if( cli.ReadStr( farmerPublicKey, "-f", "--farmer-key" ) )
            continue;
        else if( cli.ReadStr( poolPublicKey, "-p", "--pool-key" ) )
            continue;
        else if( cli.ReadStr( poolContractAddress, "-c", "--pool-contract" ) )
            continue;
        else if( cli.ReadSwitch( cfg.warmStart, "-w", "--warm-start" ) )
            continue;
        else if( cli.ReadStr( cfg.plotIdStr, "-i", "--plot-id" ) )
            continue;
        else if( cli.ArgConsume( "-z", "--compress" ) )
        {
            cfg.compressionLevel = 1;   // Default to lowest compression

            // The next parameter is potentially the compression level
             if( IsNumber( cli.Peek() ) )
                cfg.compressionLevel = (uint32)cli.ReadU64();

            continue;
        }
        else if( cli.ReadSwitch( cfg.disableOutputDirectIO, "--no-direct-io" ) )
            continue;
        else if( cli.ReadStr( cfg.plotMemoStr, "--memo" ) )
            continue;
        else if( cli.ReadSwitch( cfg.showMemo, "--show-memo" ) )
            continue;
        else if( cli.ReadSwitch( cfg.disableNuma, "-m", "--no-numa" ) )
            continue;
        else if( cli.ReadSwitch( cfg.disableCpuAffinity, "--no-cpu-affinity" ) )
            continue;
        else if( cli.ReadSwitch( cfg.verbose, "-v", "--verbose" ) )
        {
            Log::SetVerbose( true );
        }
        else if( cli.ArgConsume( "--memory" ) )
        {
            // #TODO: We should move the required part to the memplot command
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "required : %llu", requiredMem  );
            Log::Line( "total    : %llu", totalMem     );
            Log::Line( "available: %llu", availableMem );

            exit( 0 );
        }
        else if( cli.ArgConsume( "--memory-json" ) )
        {
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "{ \"required\": %llu, \"total\": %llu, \"available\": %llu }",
                         requiredMem, totalMem, availableMem );

            exit( 0 );
        }
        else if( cli.ArgConsume( "--version" ) )
        {
            Log::Line( BLADEBIT_VERSION_STR );
            exit( 0 );
        }
        else if( cli.ArgConsume( "-h", "--help" ) )
        {
            PrintUsage();
            exit( 0 );
        }
        else if( cli.ArgConsume( "--about" ) )
        {
            Log::Line( "Bladebit Chia Plotter" );
            Log::Line( "Version      : %s", BLADEBIT_VERSION_STR   );
            Log::Line( "Git Commit   : %s", BLADEBIT_GIT_COMMIT    );
            Log::Line( "Compiled With: %s", BBGetCompilerVersion() );
            
            exit( 0 );
        }
        else if( cli.ReadSwitch( cfg.benchmarkMode, "--benchmark", "--dry-run" ) )
            continue;


        // Commands
        else if( cli.ArgConsume( "diskplot" ) )
        {
            // #TODO: Remove when fixed
            FatalIf( cfg.compressionLevel > 0, "diskplot is currently disabled for compressed plotting due to a bug." );

            plotter = new DiskPlotter();
            
            // Increase the file size limit on linux
            #if PLATFORM_IS_UNIX
                struct rlimit limit;
                getrlimit( RLIMIT_NOFILE, &limit );

                if( limit.rlim_cur < limit.rlim_max )
                {
                    Log::Line( "Increasing the file limit from %u to %u", limit.rlim_cur, limit.rlim_max );

                    limit.rlim_cur = limit.rlim_max;
                    if( setrlimit( RLIMIT_NOFILE, &limit ) != 0 )
                    {
                        const int err = errno;
                        Log::Line( "*** Warning: Failed to increase file limit to with error %d (0x%02x). Plotting may fail ***", err, err );
                    }
                }
            #endif
            break;
        }
        else if( cli.ArgConsume( "ramplot" ) )
        {
            FatalIf( cfg.compressionLevel > 16, "ramplot currently does not support compression levels greater than 7" );

            plotter = new MemPlotter();
            break;
        }
    #if BB_CUDA_ENABLED
        else if( cli.ArgConsume( "cudaplot" ) )
        {
            plotter = new CudaK32Plotter();
            break;
        }
    #endif
        else if( cli.ArgConsume( "iotest" ) )
        {
            IOTestMain( cfg, cli );
            Exit( 0 );
        }
        else if( cli.ArgConsume( "memtest" ) )
        {
            MemTestMain( cfg, cli );
            Exit( 0 );
        }
        else if( cli.ArgConsume( "validate" ) )
        {
            PlotValidatorMain( cfg, cli );
            Exit( 0 );
        }
        else if( cli.ArgConsume( "plotcmp" ) )
        {
            PlotCompareMain( cfg, cli );
            Exit( 0 );
        }
        else if( cli.ArgConsume( "simulate" ) )
        {
            CmdSimulateMain( cfg, cli );
            Exit( 0 );
        }
        else if( cli.ArgConsume( "check" ) )
        {
            CmdPlotsCheckMain( cfg, cli );
            Exit( 0 );
        }
        else if( cli.ArgConsume( "cudacheck" ) )
        {
            CmdCheckCUDA( cfg, cli );
            Exit( 1 );
        }
        else if( cli.ArgConsume( "help" ) )
        {
            if( cli.HasArgs() )
            {
                if( cli.ArgMatch( "diskplot" ) )
                    DiskPlotter::PrintUsage();
                else if( cli.ArgMatch( "ramplot" ) )
                    Log::Line( "bladebit -f ... -p/c ... ramplot <out_dirs>" );
            #if BB_CUDA_ENABLED
                else if( cli.ArgMatch( "cudaplot" ) )
                    CudaK32PlotterPrintHelp();
            #endif
                else if( cli.ArgMatch( "iotest" ) )
                    IOTestPrintUsage();
                else if( cli.ArgMatch( "memtest" ) )
                    MemTestPrintUsage();
                else if( cli.ArgMatch( "validate" ) )
                    PlotValidatorPrintUsage();
                else if( cli.ArgMatch( "plotcmp" ) )
                    PlotCompareMainPrintUsage();
                else if( cli.ArgMatch( "simulate" ) )
                    CmdSimulateHelp();
                else if( cli.ArgMatch( "check" ) )
                    CmdPlotsCheckHelp();
                else if( cli.ArgMatch( "cudacheck" ) )
                    CmdCheckCUDAHelp();
                else
                    Fatal( "Unknown command '%s'.", cli.Arg() );

                Exit( 0 );
            }

            Log::Line( "help [<command>]" );
            Log::Line( "Display help text for a command." );
            Log::Line( "" );
            PrintUsage();
            exit( 0 );
        }
        else
        {
            Fatal( "Unexpected argument '%s'", cli.Arg() );
        }
    }

    // The remainder should be output folders, which we parse after the plotter consumes it's config

    ///
    /// Validate global config
    ///
    FatalIf( farmerPublicKey == nullptr, "A farmer public key must be specified." );
    FatalIf( !KeyTools::HexPKeyToG1Element( farmerPublicKey, *(cfg.farmerPublicKey = new bls::G1Element()) ),
        "Invalid farmer public key '%s'", farmerPublicKey );

    if( poolContractAddress )
    {
        cfg.poolContractPuzzleHash = new PuzzleHash();
        FatalIf( !PuzzleHash::FromAddress( *cfg.poolContractPuzzleHash, poolContractAddress ),
            "Invalid pool contract puzzle hash '%s'", poolContractAddress );
    }
    else if( poolPublicKey )
    {
        cfg.poolPublicKey = new bls::G1Element();
        FatalIf( !KeyTools::HexPKeyToG1Element( poolPublicKey, *cfg.poolPublicKey ),
                 "Invalid pool public key '%s'", poolPublicKey );
    }
    else
        Fatal( "Error: Either a pool public key or a pool contract address must be specified." );


    // FatalIf( cfg.compressionLevel > 7, "Invalid compression level. Please specify a compression level between 0 and 7 (inclusive)." );
    FatalIf( cfg.compressionLevel > 40, "Invalid compression level. Please specify a compression level between 0 and 40 (inclusive)." );
    // If making compressed plots, get thr compression CTable, etc.
    if( cfg.compressionLevel > 0 )
    {
        // #TODO: Remove this when added
        if( cfg.compressionLevel > 7 )
            Log::Line( "WARNING: Compression levels greater than 7 are only for testing purposes and are not configured to the final plot size." );

        cfg.compressedEntryBits = 17 - cfg.compressionLevel;
        cfg.ctable              = CreateCompressionCTable( cfg.compressionLevel, &cfg.cTableSize );
        cfg.compressionInfo     = GetCompressionInfoForLevel( cfg.compressionLevel );
        cfg.compressedEntryBits = cfg.compressionInfo.entrySizeBits;
        cfg.numDroppedTables    = cfg.compressionLevel < 40 ? 1 : 2;

        cfg.ctable          = CreateCompressionCTable( cfg.compressionLevel );
        cfg.compressionInfo = GetCompressionInfoForLevel( cfg.compressionLevel );
    }

    const uint maxThreads = SysHost::GetLogicalCPUCount();
    if( cfg.threadCount == 0 )
        cfg.threadCount = maxThreads;
    else if( cfg.threadCount > maxThreads )
    {
        Log::Write( "Warning: Lowering thread count from %u to %u, the native maximum.",
                    cfg.threadCount, maxThreads );

        cfg.threadCount = maxThreads;
    }


    if( cfg.plotIdStr )
    {
        const size_t len = strlen( cfg.plotIdStr );
        if( len < 64 && len != 66 )
            Fatal( "Invalid plot id." );
        
        if( len == 66 )
        {
            if( cfg.plotIdStr[0] == '0' && cfg.plotIdStr[1] == 'x' )
                cfg.plotIdStr += 2;
            else
                Fatal( "Invalid plot id." );
        }
    }
    if( cfg.plotMemoStr )
    {
        size_t len = strlen( cfg.plotMemoStr );
        if( len > 2 && cfg.plotMemoStr[0] == '0' && cfg.plotMemoStr[1] == 'x' )
        {
            cfg.plotMemoStr += 2;
            len -= 2;
        }
        
        if( len/2 != (48 + 48 + 32) && len != (32 + 48 + 32) )
            Fatal( "Invalid plot memo." );
    }

    ///
    // Global Config Summary
    ///
    Log::Line( "" );
    Log::Line( "Bladebit Chia Plotter" );
    Log::Line( "Version      : %s", BLADEBIT_VERSION_STR   );
    Log::Line( "Git Commit   : %s", BLADEBIT_GIT_COMMIT    );
    Log::Line( "Compiled With: %s", BBGetCompilerVersion() );
    Log::Line( "" );

    Log::Line( "[Global Plotting Config]" );
    if( cfg.plotCount == 0 )
        Log::Line( " Will create plots indefinitely." );
    else
        Log::Line( " Will create %u plots.", cfg.plotCount );

    Log::Line( " Thread count          : %d", cfg.threadCount );
    Log::Line( " Warm start enabled    : %s", cfg.warmStart ? "true" : "false" );
    Log::Line( " NUMA disabled         : %s", cfg.disableNuma ? "true" : "false" );
    Log::Line( " CPU affinity disabled : %s", cfg.disableCpuAffinity ? "true" : "false" );

    Log::Line( " Farmer public key     : %s", farmerPublicKey );

    if( poolContractAddress )
        Log::Line( " Pool contract address : %s", poolContractAddress );
    else if( cfg.poolPublicKey )
        Log::Line( " Pool public key       : %s", poolPublicKey   );

    // Log::Line( " Compression           : %s", cfg.compressionLevel > 0 ? "enabled" : "disabled" );
    if( cfg.compressionLevel > 0 )
        Log::Line( " Compression Level     : %u", cfg.compressionLevel );

    Log::Line( " Benchmark mode        : %s", cfg.benchmarkMode ? "enabled" : "disabled" );
    // Log::Line( " Output path           : %s", cfg.outputFolder );
    // Log::Line( "" );


    FatalIf( plotter == nullptr, "No plotter type chosen." );

    // #TODO: Remove when C8 compression values added.
    FatalIf( cfg.compressionLevel == 8, "Compression level 8 is not currently supported." );

    // Parse plotter-specific CLI
    plotter->ParseCLI( cfg, cli );

    // Parse remaining args as output directories
    cfg.outputFolderCount = (uint32)cli.RemainingArgCount();
    FatalIf( cfg.outputFolderCount < 1, "At least one output folder must be specified." );

    cfg.outputFolders = new std::string[cfg.outputFolderCount];

    int32 folderIdx = 0;
    std::string outPath; 
    while( cli.HasArgs() )
    {
        outPath = cli.Arg();
        FatalIf( outPath[0] == '-', "Unrecognized argument '%s'.", outPath.c_str() );

        // Add trailing slash?
        const char endChar = outPath.back();
        if( endChar != '/' && endChar != '\\' )
            outPath += '/';
        
        cfg.outputFolders[folderIdx++] = outPath;
        cli.NextArg();
    }

    cfg.outputFolder = cfg.outputFolders[0].c_str();

    Log::Line( "" );
    Log::Flush();

    // Initialize plotter
    plotter->Init();

    Log::Line( "" );

    outPlotter = plotter;
}


//-----------------------------------------------------------
static const char* USAGE = "bladebit [GLOBAL_OPTIONS] <command> [COMMAND_OPTIONS]\n"
R"(
[COMMANDS]
 cudaplot   : Create a plot by using the a CUDA-capable GPU.
 diskplot   : Create a plot by making use of a disk.
 ramplot    : Create a plot completely in-ram.
 iotest     : Perform a write and read test on a specified disk.
 memtest    : Perform a memory (RAM) copy test.
 validate   : Validates all entries in a plot to ensure they all evaluate to a valid proof.
 simulate   : Simulation tool useful for compressed plot capacity.
 check      : Check and validate random proofs in a plot.
 help       : Output this help message, or help for a specific command, if specified.

[GLOBAL_OPTIONS]:
 -h, --help           : Shows this message and exits.

 -t, --threads        : Maximum number of threads to use.
                        By default, this is set to the maximum number of logical cpus present.

 -n, --count          : Number of plots to create. Default = 1.

 -f, --farmer-key     : Farmer public key, specified in hexadecimal format.
                        *REQUIRED*

 -c, --pool-contract  : Pool contract address.
                        Use this if you are creating pool plots.
                        *A pool contract address or a pool public key must be specified.*

 -p, --pool-key       : Pool public key, specified in hexadecimal format.
                        Use this if you are creating OG plots.
                        Only used if a pool contract address is not specified.

 -z,--compress [level]: Compress the plot. Optionally pass a compression level parameter.
                        If no level parameter is passed, the default compression level of 1 is used.
                        Current compression levels supported are from 0 to 7 (inclusive).
                        Where 0 means no compression, and 7 is the highest compression.
                        Higher compression means smaller plots, but more CPU usage during harvesting.

 --no-direct-io       : Disable direct I/O when writing plot files.
                        Enable this if writing to a storage destination 
                        that does not support direct I/O.

 --benchmark          : Enables benchmark mode. This is meant to test plotting without
                        actually writing a final plot to disk.

 -w, --warm-start     : Touch all pages of buffer allocations before starting to plot.

 -i, --plot-id        : Specify a plot id for debugging.

 --memo               : Specify a plot memo for debugging.

 --show-memo          : Output the memo of the next plot the be plotted.

 -v, --verbose        : Enable verbose output.

 -m, --no-numa        : Disable automatic NUMA aware memory binding.
                        If you set this parameter in a NUMA system you
                        will likely get degraded performance.

 --no-cpu-affinity    : Disable assigning automatic thread affinity.
                        This is useful when running multiple simultaneous
                        instances of Bladebit as you can manually
                        assign thread affinity yourself when launching Bladebit.

 --memory             : Display system memory available, in bytes, and the 
                        required memory to run Bladebit, in bytes.

 --memory-json        : Same as --memory, but formats the output as json.

 --version            : Display current version.


[EXAMPLES]
bladebit --help

bladebit help diskplot

# Simple config:
bladebit -t 24 -f <farmer_pub_key> -c <contract_address> diskplot -t1 /my/temporary/plot/dir /my/output/dir

# With fine-grained control over threads per phase/section (see bladebit -h diskplot):
bladebit -t 30 -f <farmer_pub_key> -c <contract_address> diskplot --f1-threads 16 --c-threads 16 --p2-threads 8 -t1 /my/temporary/plot/dir /my/output/dir
)";

//-----------------------------------------------------------
void PrintUsage()
{
    Log::Line( USAGE );
}