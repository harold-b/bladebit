#include "Commands.h"
#include "plotting/GlobalPlotConfig.h"
#include "tools/PlotReader.h"
#include "threading/MTJob.h"
#include "harvesting/GreenReaper.h"
#include "plotting/f1/F1Gen.h"

static constexpr double SECS_PER_DAY       = 24 * 60 * 60;
static constexpr double CHALLENGE_INTERVAL = 9.375;
static constexpr uint32 CHALLENGES_PER_DAY = (uint32)(SECS_PER_DAY / CHALLENGE_INTERVAL);

enum class SubCommand
{
    None,
    Farm,
};

struct Config
{
    GlobalPlotConfig* gCfg      = nullptr;
    const char* plotPath        = "";
    uint64      fetchCount      = 100;
    uint32      parallelCount   = 1;
    double      maxLookupTime   = 8.0;
    uint32      filterBits      = 512;
    uint32      partials        = 300;          // Partials per day
    size_t      farmSize        = 0;
    byte        randomSeed[BB_PLOT_ID_LEN] = {};
    double      powerSimSeconds = -1;
    bool        noCuda          = false;
    int32       cudaDevice      = 0;

    // Internally set
    double      partialRatio  = 0;


    // Set by the simulation job
    size_t      jobsMemoryUsed = 0;
};

struct JobStats
{
    std::atomic<uint64> nTotalProofs            = 0;                                  // Total proofs non-decayed f7s found in the plot
    std::atomic<uint64> nActualFetches          = 0;                                  // How many feches we actually made (f7 was found, no decay)
    std::atomic<uint64> nActualFullProofFetches = 0;                                  // Actual number of full proffs fetched (excluding decay misses)
    std::atomic<uint64> totalTimeNano           = 0;                                  // Total time accumulated fetching from plots (includes both qualities and full proofs)
    std::atomic<uint64> totalFullProofTimeNano  = 0;                                  // 
    std::atomic<uint64> maxFetchTimeNano        = 0;                                  // Largest amount of time spent fetching from plots (includes both qualities and full proof)
    std::atomic<uint64> minFPFetchTimeNano      = std::numeric_limits<uint64>::max(); // Smallest amount of time spent fetching from plots (includes both qualities and full proof)
};

struct SimulatorJob : MTJob<SimulatorJob>
{
    Config*        cfg;
    Span<FilePlot> plots;
    JobStats*      stats;
    uint32         decompressorThreadCount;

    virtual void Run() override;

    void RunFarm( PlotReader& reader, uint64 fetchCount, uint32 partialsCount );
};


static size_t CalculatePlotSizeBytes( const uint32 k, const uint32 compressionLevel );
static void DumpCompressedPlotCapacity( const Config& cfg, const uint32 k, const uint32 compressionLevel, const double fetchAverageSecs );

void CmdSimulateMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    Config cfg = {};
    cfg.gCfg = &gCfg;

    // Set initial random seed for F7s
    SysHost::Random( (byte*)&cfg.randomSeed, sizeof( cfg.randomSeed ) );

    while( cli.HasArgs() )
    {
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            CmdSimulateHelp();
            Exit( 0 );
        }
        else if( cli.ReadU64( cfg.fetchCount, "-n", "--iterations" ) ) continue;
        else if( cli.ReadU32( cfg.parallelCount, "-p", "--parallel" ) ) continue;
        else if( cli.ReadF64( cfg.maxLookupTime, "-l", "--lookup" ) ) continue;
        else if( cli.ReadU32( cfg.filterBits, "-f", "--filter" ) ) continue;
        else if( cli.ReadU32( cfg.partials, "--partials" ) ) continue;
        else if( cli.ReadF64( cfg.powerSimSeconds, "--power" ) ){
            FatalIf( cfg.powerSimSeconds < 0.0, "Invalid power simulation time." );
            continue;
        }
        else if( cli.ReadSize( cfg.farmSize, "-s", "--size" ) ) continue;
        else if( cli.ReadHexStrAsBytes( cfg.randomSeed, sizeof( cfg.randomSeed ), "--seed" ) ) continue;
        else if( cli.ReadSwitch( cfg.noCuda, "--no-cuda" ) ) continue;
        else if( cli.ReadI32( cfg.cudaDevice, "-d", "--device" ) ) continue;
        else
            break;
    }

    FatalIf( !cli.HasArgs(), "Expected a path to a plot file." );
    {
        cfg.plotPath = cli.Arg();
        cli.NextArg();

        if( cli.HasArgs() )
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
            Exit( 1 );
        }
    }

    FatalIf( cfg.fetchCount < 1      , "Invalid iteration count of %u.", cfg.fetchCount );
    FatalIf( cfg.maxLookupTime <= 0.0, "Invalid max lookup time of %lf.", cfg.maxLookupTime );
    FatalIf( cfg.filterBits < 1      , "Invalid filter bits value of %u.", cfg.filterBits );
    FatalIf( cfg.parallelCount * (uint64)gCfg.threadCount > MAX_THREADS, 
        "Too many thread combination (%llu) between -t and -p", cfg.parallelCount * (llu)gCfg.threadCount );

    const bool powerMode = cfg.powerSimSeconds > 0.0;

    // Lower the parallel count until all instances have at least 1 lookup
    if( !powerMode && cfg.parallelCount > cfg.fetchCount )
    {
        Log::Line( "Warning: Limiting parallel context count to %u, as it must be <= than the fetch count of %llu",
            cfg.parallelCount, (llu)cfg.fetchCount );
        cfg.parallelCount = (uint32)cfg.fetchCount;
    }

    // With cuda, only support a single context and thread
    if( !cfg.noCuda && cfg.parallelCount > 1 )
    {
        cfg.parallelCount = 1;
        Log::Line( "Warning: Limiting the number of parallel contexts to 1 for CUDA harvester simulation." );
    }


    FilePlot* plot = new FilePlot[cfg.parallelCount];
    for( uint32 i = 0; i < cfg.parallelCount; i++ )
    {
        if( !plot[i].Open( cfg.plotPath ) )
            Fatal( "Failed to open plot file at '%s' with error %d.", cfg.plotPath, plot[i].GetError() );
    }

    const uint32 compressionLevel = plot[0].CompressionLevel();
    FatalIf( compressionLevel < 1, "The plot %s is not compressed.", cfg.plotPath );

    if( powerMode ) 
    {
        if( cfg.farmSize == 0 )
        {
            // Set a default farm size when at least 1 plot per context passes the filter
            const size_t plotSize = CalculatePlotSizeBytes( plot->K(), compressionLevel );
            cfg.farmSize = (uint64)cfg.parallelCount * plotSize * cfg.filterBits;
            Log::Line( "Setting default farm size to %llu TB (use --size <size> to set a farm size manually).",
                (llu)BtoTBSi( cfg.farmSize ) );
        }
        
        // Adjust fetch count given the simulation time
        cfg.fetchCount = std::max( (uint64)1, (uint64)(cfg.powerSimSeconds / CHALLENGE_INTERVAL ) ) * cfg.parallelCount;
    }



    Log::Line( "[Simulator for harvester farm capacity for K%2u C%u plots]", plot->K(), compressionLevel );
    Log::Line( " Random seed: 0x%s", BytesToHexStdString( cfg.randomSeed, sizeof( cfg.randomSeed ) ).c_str() );
    Log::Line( " Simulating..." );
    Log::NewLine();


    ThreadPool pool( cfg.parallelCount, ThreadPool::Mode::Fixed, true );

    const uint32 decompressorThreadCount = std::min( gCfg.threadCount == 0 ? 8 : gCfg.threadCount, SysHost::GetLogicalCPUCount() );
    JobStats stats = {};

    {
        SimulatorJob job = {};
        job.cfg                     = &cfg;
        job.plots                   = Span<FilePlot>( plot, cfg.parallelCount );
        job.stats                   = &stats;
        job.decompressorThreadCount = decompressorThreadCount;

        MTJobRunner<SimulatorJob>::RunFromInstance( pool, cfg.parallelCount, job );

        if( stats.minFPFetchTimeNano == std::numeric_limits<uint64>::max() )
            stats.minFPFetchTimeNano = 0;
    }

    // Report
    {
        const uint64 actualFetchCount        = stats.nActualFetches;
        const double effectivePartialPercent = actualFetchCount == 0 ? 0 : stats.nActualFullProofFetches.load() / (double)actualFetchCount * 100.0;
        // const uint64 fetchCountAdjusted = CDiv( cfg.fetchCount, cfg.parallelCount ) * cfg.parallelCount;

        const uint64 totalTimeNanoAdjusted = stats.totalTimeNano / cfg.parallelCount;
        const uint64 fetchAverageNano      = actualFetchCount == 0 ? 0 : totalTimeNanoAdjusted / actualFetchCount;
        const double fetchAverageSecs      = NanoSecondsToSeconds( fetchAverageNano );
        const double fetchMaxSecs          = NanoSecondsToSeconds( stats.maxFetchTimeNano );
        const double fetchFpAverageSecs    = stats.nActualFullProofFetches > 0 ? NanoSecondsToSeconds( stats.totalFullProofTimeNano / stats.nActualFullProofFetches ) : 0;
        const size_t memoryUsed            = cfg.jobsMemoryUsed * cfg.parallelCount;

        Log::Line( " Context count                 : %llu", (llu)cfg.parallelCount );
        Log::Line( " Thread per context instance   : %llu", (llu)gCfg.threadCount );
        Log::Line( " Memory used                   : %.1lfMiB ( %.1lfGiB )", (double)memoryUsed BtoMB, (double)memoryUsed BtoGB );
        Log::Line( " Proofs / Challenges           : %llu / %llu ( %.2lf%% )", (llu)stats.nTotalProofs, (llu)cfg.fetchCount, (uint64)stats.nTotalProofs / (double)cfg.fetchCount * 100.0 );
        Log::Line( " Fetches / Challenges          : %llu / %llu", (llu)actualFetchCount, (llu)cfg.fetchCount );
        Log::Line( " Filter bits                   : %u", cfg.filterBits );
        Log::Line( " Effective partials            : %llu ( %.2lf%% )", (llu)stats.nActualFullProofFetches.load(), effectivePartialPercent );
        Log::Line( " Total fetch time elapsed      : %.3lf seconds", NanoSecondsToSeconds( totalTimeNanoAdjusted ) );
        Log::Line( " Average plot lookup time      : %.3lf seconds", fetchAverageSecs   );
        Log::Line( " Worst plot lookup lookup time : %.3lf seconds", fetchMaxSecs       );
        Log::Line( " Average full proof lookup time: %.3lf seconds", fetchFpAverageSecs );
        Log::Line( " Fastest full proof lookup time: %.3lf seconds", stats.nActualFullProofFetches == 0 ? 0.0 : NanoSecondsToSeconds( stats.minFPFetchTimeNano ) );
        Log::NewLine();

        if( fetchMaxSecs >= cfg.maxLookupTime )
        {
            Log::Line( "*** Warning *** : Your worst plot lookup time of %.3lf was over the maximum set of %.3lf.", 
                fetchMaxSecs, cfg.maxLookupTime );
        }

        // Calculate farm size for this compression level
        Log::Line( " %10s | %-10s | %-10s | %-10s ", "compression", "plot count", "size TB", "size PB" );
        Log::Line( "------------------------------------------------" );
        DumpCompressedPlotCapacity( cfg, plot->K(), compressionLevel, fetchAverageSecs );
    }

    Log::NewLine();
    Exit( 0 );
}

void DumpCompressedPlotCapacity( const Config& cfg, const uint32 k, const uint32 compressionLevel, const double fetchAverageSecs )
{
    // Calculate farm size for this compression level
    const size_t plotSize  = CalculatePlotSizeBytes( k, compressionLevel );
    const uint64 plotCount = (uint64)(cfg.maxLookupTime / fetchAverageSecs * cfg.filterBits);

    const size_t farmSizeBytes = plotCount * plotSize;
    const size_t farmSizeTB    = BtoTBSi( farmSizeBytes );
    const auto   farmSizePB    = BtoPBSiF( farmSizeBytes );

    Log::Line( " C%-10u | %-10llu | %-10llu | %-10.2lf ", compressionLevel, plotCount, farmSizeTB, farmSizePB );
}

size_t CalculatePlotSizeBytes( const uint32 k, const uint32 compressionLevel )
{
    ASSERT( compressionLevel > 0 );

    auto info = GetCompressionInfoForLevel( compressionLevel );

    const uint64 tableEntryCount      = 1ull << k;
    const double tablePrunedFactors[] = { 0.798, 0.801, 0.807, 0.823, 0.865, 1, 1 };

    const size_t parkSizes[] = {
        0, // Table 1 is dropped
        compressionLevel >= 40 ? 0 : info.tableParkSize,
        compressionLevel >= 40 ? info.tableParkSize : CalculateParkSize( TableId::Table3 ),
        CalculateParkSize( TableId::Table4 ),
        CalculateParkSize( TableId::Table5 ),
        CalculateParkSize( TableId::Table6 ),
        CalculatePark7Size( k )
    };

    size_t tableSizes[7] = {};
    for( uint32 table = (uint32)TableId::Table2; table <= (uint32)TableId::Table7; table++ )
    {
        const uint64 prunedEntryCount = (uint64)(tableEntryCount * tablePrunedFactors[table]);
        const uint64 parkCount        = CDiv( prunedEntryCount, kEntriesPerPark );

        tableSizes[table] = parkCount * parkSizes[table];
    }

    const size_t c1EntrySize = RoundUpToNextBoundary( k, 8 );
    const size_t c3ParkCount = CDiv( tableEntryCount, kCheckpoint1Interval );
    const size_t c3Size      = c3ParkCount * CalculateC3Size(); 
    const size_t c1Size      = c1EntrySize * ( tableEntryCount / kCheckpoint1Interval ) + c1EntrySize;
    const size_t c2Size      = c1EntrySize * ( tableEntryCount / (kCheckpoint1Interval * kCheckpoint2Interval) ) + c1EntrySize;

    const size_t plotSize = c3Size + c1Size + c2Size +
        tableSizes[0] +
        tableSizes[1] +
        tableSizes[2] +
        tableSizes[3] +
        tableSizes[4] +
        tableSizes[5] +
        tableSizes[6];
    
    return plotSize;
}


void SimulatorJob::Run()
{
    FilePlot& plot = plots[JobId()];

    PlotReader reader( plot );

    {
        GreenReaperConfig grCfg = {};
        grCfg.apiVersion         = GR_API_VERSION;
        grCfg.threadCount        = decompressorThreadCount;
        grCfg.cpuOffset          = decompressorThreadCount * JobId();
        grCfg.disableCpuAffinity = cfg->gCfg->disableCpuAffinity;
        grCfg.gpuRequest         = cfg->noCuda ? GRGpuRequestKind_None : GRGpuRequestKind_FirstAvailable;
        grCfg.gpuDeviceIndex     = cfg->cudaDevice;

        GreenReaperContext* grContext = nullptr;
        const auto result = grCreateContext( &grContext, &grCfg, sizeof( GreenReaperConfig ) );
        FatalIf( !grContext, "Failed to create decompression context with error %d.", (int)result );

        if( grCfg.gpuRequest != GRGpuRequestKind_None && !(bool)grHasGpuDecompressor( grContext ) )
            Log::Line( "Warning: No GPU device decompressor selected. Falling back to CPU-based simulation." );

        reader.AssignDecompressionContext( grContext );
    }
    // reader.ConfigDecompressor( decompressorThreadCount, cfg->gCfg->disableCpuAffinity, decompressorThreadCount * JobId() );


    const double challengeRatio = cfg->fetchCount / (double)CHALLENGES_PER_DAY;
    const uint64 actualPartials = (uint64)(cfg->partials * challengeRatio);

    uint64 fetchCountForJob, partialsForJob;
    {
        uint64 _;
        GetThreadOffsets( this, cfg->fetchCount, fetchCountForJob, _, _  );
        GetThreadOffsets( this, actualPartials, partialsForJob, _, _  );
        ASSERT( fetchCountForJob > 0 );
    }

    RunFarm( reader, fetchCountForJob, (uint32)partialsForJob );

    if( IsControlThread() )
        cfg->jobsMemoryUsed = grGetMemoryUsage( reader.GetDecompressorContext() );
}

void SimulatorJob::RunFarm( PlotReader& reader, const uint64 challengeCount, const uint32 partialCount )
{
    const uint32 k      = reader.PlotFile().K();
    const size_t f7Size = CDiv( k, 8 );
    
    uint64 plotsPerChallenge = 1;

    // In power simulation mode, determine how many plots we've got per challenge, if any,
    // based on the specified farm size.
    const bool powerSimMode = cfg->powerSimSeconds > 0.0;
    if( powerSimMode )
    {
        const size_t plotSize               = CalculatePlotSizeBytes( k, reader.PlotFile().CompressionLevel() );
        const uint64 farmPlotCount          = (uint64)cfg->farmSize / plotSize;
        const uint64 totalPlotsPerChallenge = farmPlotCount / cfg->filterBits;

        uint64 _;
        GetFairThreadOffsets( this, totalPlotsPerChallenge, plotsPerChallenge, _, _ );

        if( plotsPerChallenge < 1 )
            return;
    }

    const auto simulationStart = TimerBegin();

    uint64 totalFetchTimeNano     = 0;
    uint64 totalFullProofTimeNano = 0;
    uint64 maxFetchDurationNano   = 0;
    uint64 minFetchDurationNano   = std::numeric_limits<uint64>::max();
    uint64 nTotalProofs           = 0;
    uint64 nTotalFetches          = 0;
    uint64 nFullProofsFetches     = 0;

    uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] = {};
    byte   quality    [BB_CHIA_QUALITY_SIZE ] = {};

    byte challenge[BB_CHIA_CHALLENGE_SIZE] = {};
    {
        const char* challengeStr = "00000000000000ee9355068689bd558eafe07cc7af47ad1574b074fc34d6913a";
        HexStrToBytes( challengeStr, sizeof( challenge )*2, challenge, sizeof( challenge ) );
    }

    const int64 partialInterval = partialCount == 0 ? std::numeric_limits<int64>::max() : (int64)std::min( challengeCount, challengeCount / partialCount );
          int64 nextPartial     = partialCount == 0 ? std::numeric_limits<int64>::max() : 1;

    const uint64 f7Mask = (1ull << k) - 1;
          uint64 prevF7 = (uint64)_jobId & f7Mask;

    for( uint64 n = 0; n < challengeCount; n++ )
    {
        // How many plots are we simulating for this challenge?
        // When doing maximum farm simulation, we only try 1,
        // as we are simply calculating the maximum capacity based
        // on lookup times. However, when simulating for power,
        // we need to know how many plots would pass the filter per challenge
        // given a hypothetical farm.
        const auto challengeStartTime = TimerBegin();

        for( uint64 p = 0; p < plotsPerChallenge; p++ )
        {
            // Generate new f7 (challenge) & fetch from plot
            const uint64 f7 = F1GenSingleForK( k, cfg->randomSeed, prevF7 ) & f7Mask;
            prevF7 = f7;

            // Embed f7 into challenge as BE
            for( size_t i = 0; i < f7Size; i++ )
                challenge[i] = (byte)(f7 >> ((f7Size - i - 1) * 8));

            const auto fetchTimer = TimerBegin();

            // Read indices for F7
                  uint64 p7IndexBase = 0;
            const uint64 matchCount  = reader.GetP7IndicesForF7( f7, p7IndexBase );

            const bool fetchFullProof = --nextPartial <= 0;
            if( fetchFullProof )
                nextPartial = partialInterval;

            uint32 nFetchedFromMatches           = 0;
            uint64 nFullProofsFetchedFromMatches = 0;

            for( uint64 i = 0; i < matchCount; i++ )
            {
                uint64 p7Entry;
                FatalIf( !reader.ReadP7Entry( p7IndexBase + i, p7Entry ), 
                    "Failed to read P7 entry at %llu. (F7 = %llu)", (llu)p7IndexBase + i, (llu)f7 );

                ProofFetchResult rQ, rP = ProofFetchResult::OK;

                rQ = reader.FetchQualityForP7Entry( p7Entry, challenge, quality );
                if( rQ == ProofFetchResult::OK )
                {
                    nFetchedFromMatches++;

                    if( fetchFullProof )
                    {
                        rP = reader.FetchProof( p7Entry, fullProofXs );
                        if( rP == ProofFetchResult::OK )
                            nFullProofsFetchedFromMatches++;
                    }
                }

                const auto errR = rQ != ProofFetchResult::OK ? rQ : rP;
                if( errR != ProofFetchResult::OK )
                {
                    Log::Error( "Error %d while fetching proof for F7 %llu.", (int)errR, (llu)f7 );
                    // FatalIf( errR == ProofFetchResult::Error, "Error while fetching proof for F7 %llu.", (llu)f7 );
                    // FatalIf( errR == ProofFetchResult::CompressionError, "Decompression error while fetching proof for F7 %llu.", (llu)f7 );
                }
            }

            const auto fetchElapsedNano = (uint64)TicksToNanoSeconds( TimerEndTicks( fetchTimer ) );

            nTotalProofs += nFetchedFromMatches;
            if( nFetchedFromMatches > 0 )
            {
                nTotalFetches++;
                totalFetchTimeNano += fetchElapsedNano;

                maxFetchDurationNano = std::max( maxFetchDurationNano, fetchElapsedNano );
            }

            if( nFullProofsFetchedFromMatches > 0 )
            {
                nFullProofsFetches++;
                totalFullProofTimeNano += fetchElapsedNano;

                minFetchDurationNano = std::min( minFetchDurationNano, fetchElapsedNano );
            }
        }

        if( powerSimMode )
        {
            // End power simulation?
            const auto simulationElapsed = TimerEnd( simulationStart );

            if( simulationElapsed >= cfg->powerSimSeconds )
                break;

            // Wait for next challenge when in power usage simulation mode
            const double challengeTimeElapsed = TimerEnd( challengeStartTime );

            // #TODO: Count missed challenges
            double timeUntilNextChallenge = CHALLENGE_INTERVAL - challengeTimeElapsed;
            if( timeUntilNextChallenge < 0.0 )
                timeUntilNextChallenge = std::fmod( challengeTimeElapsed, CHALLENGE_INTERVAL );

            if( timeUntilNextChallenge >= 0.01 )
            {
                // Log::Line( "[%u] Sleep for %.2lf seconds. %.2lf / %.2lf", 
                //     timeUntilNextChallenge, simulationElapsed, cfg->powerSimSeconds );
                Thread::Sleep( (long)(timeUntilNextChallenge * 1000.0) );
            }
        }
    }

    stats->nTotalProofs            += nTotalProofs;
    stats->nActualFetches          += nTotalFetches;
    stats->nActualFullProofFetches += nFullProofsFetches;
    stats->totalTimeNano           += totalFetchTimeNano;
    stats->totalFullProofTimeNano  += totalFullProofTimeNano;

    {
        uint64 curMaxFetch = stats->maxFetchTimeNano.load( std::memory_order_relaxed );
        while( curMaxFetch < maxFetchDurationNano && 
            !stats->maxFetchTimeNano.compare_exchange_weak( curMaxFetch, maxFetchDurationNano, 
                                                            std::memory_order_release, std::memory_order_relaxed ) );
    }

    if( minFetchDurationNano != std::numeric_limits<uint64>::max() )
    {
        uint64 curMinFetch = stats->minFPFetchTimeNano.load( std::memory_order_relaxed );
        while( curMinFetch > minFetchDurationNano && 
            !stats->minFPFetchTimeNano.compare_exchange_weak( curMinFetch, minFetchDurationNano, 
                                                              std::memory_order_release, std::memory_order_relaxed ) );
    }
}


static const char _help[] = R"(simulate [OPTIONS] <plot_file_path>
OPTIONS:
 -h, --help               : Display this help message and exit.
 -n, --iterations <count> : How many iterations to run (default = %llu)
 -p, --parallel <count>   : How many instances to run in parallel (default = 1)
 -l, --lookup <seconds>   : Maximum allowed time per proof lookup (default = %.2lf)
 -f, --filter <count>     : Plot filter bit count (default = %u)
 --partials <count>       : Partials per-day simulation. (default = %u)
 --power <seconds>        : Time in seconds to run power simulation. -n is set automatically in this mode.
 -s, --size <size>        : Size of farm. Only used when `--power` is set.
 --seed <hex>             : 64 char hex string to use as a random seed for challenges.
 --no-cuda                : Don't use CUDA for decompression.
 -d, --device <index>     : Cuda device index. (default = 0)
)";

void CmdSimulateHelp()
{
    Config cfg = {};
    printf( _help, 
        (llu)cfg.fetchCount,
        cfg.maxLookupTime,
        cfg.filterBits,
        cfg.partials
    );
}