#include "CudaPlotter.h"
#include "CudaPlotContext.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"
#include "threading/MTJob.h"
#include "util/jobs/MemJobs.h"
#include "util/StackAllocator.h"
#include "CudaParkSerializer.h"
#include "plotting/CTables.h"
#include "plotting/TableWriter.h"
#include "plotting/PlotTools.h"
#include "util/VirtualAllocator.h"
#include "harvesting/GreenReaper.h"
#include "tools/PlotChecker.h"


// TEST/DEBUG
#if _DEBUG
    #include "algorithm/RadixSort.h"
    #include "plotdisk/jobs/IOJob.h"
    #include "io/FileStream.h"

    ThreadPool* _dbgThreadPool = nullptr;

    static void DbgPruneTableBuckets( CudaK32PlotContext& cx, const TableId rTable );
    static void DbgPruneTable( CudaK32PlotContext& cx, const TableId rTable );
#endif

static void InitContext( CudaK32PlotConfig& cfg, CudaK32PlotContext*& outContext );
static void CudaInit( CudaK32PlotContext& cx );

void GenF1Cuda( CudaK32PlotContext& cx );

static void MakePlot( CudaK32PlotContext& cx );
static void FpTable( CudaK32PlotContext& cx );
static void FpTableBucket( CudaK32PlotContext& cx, const uint32 bucket );
static void UploadBucketForTable( CudaK32PlotContext& cx, const uint64 bucket );
static void FinalizeTable7( CudaK32PlotContext& cx );
static void InlineTable( CudaK32PlotContext& cx, const uint32* devInX, cudaStream_t stream );

static void AllocBuffers( CudaK32PlotContext& cx );
static void AllocateP1Buffers( CudaK32PlotContext& cx, CudaK32AllocContext& acx );
static void AllocateParkSerializationBuffers( CudaK32PlotContext& cx, IAllocator& pinnedAllocator, bool dryRun );

template<typename T>
static void UploadBucketToGpu( CudaK32PlotContext& context, TableId table, const uint32* hostPtr, T* devPtr, uint64 bucket, uint64 stride );
static void LoadAndSortBucket( CudaK32PlotContext& cx, const uint32 bucket );

void CudaMatchBucketizedK32( CudaK32PlotContext& cx, const uint32* devY, cudaStream_t stream, cudaEvent_t event );
[[noreturn]] static void ListCudaDevices( bool json );

// Defined in FxCuda.cu
void GenFx( CudaK32PlotContext& cx, const uint32* devYIn, const uint32* devMetaIn, cudaStream_t stream );

static const char* USAGE = "bladebit_cuda ... cudaplot <out_dir>\n"
R"(
GPU-based (CUDA) plotter

[OPTIONS]:
 -h, --help           : Shows this help message and exits.
 -d, --device         : Select the CUDA device index. (default=0)

 -l, --list           : List availabe CUDA devices, showing their indices.

 --json               : Show output in json format. This is only valid for certain parameters:
                          --list

 --disk-128           : Enable hybrid disk plotting for 128G system RAM. 
                         Requires a --temp1 and --temp2 to be set.

 --disk-16            : (experimental) Enable hybrid disk plotting for 16G system RAM. 
                         Requires a --temp1 and --temp2 to be set.

 -t1, --temp1         : Temporary directory 1. Used for longer-lived, sequential writes.

 -t2, --temp2         : Temporary directory 2. Used for temporary, shorted-lived read and writes.
                         NOTE: If only one of -t1 or -t2 is specified, both will be
                               set to the same directory.

 --check <n>          : Perform a plot check for <n> proofs on the newly created plot.

 --check-threshold <f>: Proof threshold rate below which the plots that don't pass
                         the check will be deleted.
                         That is, the number of proofs fetched / proof check count
                         must be above or equal to this threshold to pass.
                         (default=0.6).
)";

///
/// CLI
///
//-----------------------------------------------------------
void CudaK32PlotterPrintHelp()
{
    Log::Line( USAGE );
}

//-----------------------------------------------------------
void CudaK32Plotter::ParseCLI( const GlobalPlotConfig& gCfg, CliParser& cli )
{
    CudaK32PlotConfig& cfg = _cfg;
    cfg.gCfg = &gCfg;

    bool listDevices = false;
    bool json        = false;

    while( cli.HasArgs() )
    {
        if( cli.ReadU32( cfg.deviceIndex, "-d", "--device" ) )
            continue;
        if( cli.ReadSwitch( cfg.hybrid128Mode, "--disk-128" ) )
            continue;
        if( cli.ReadSwitch( cfg.hybrid16Mode, "--disk-16" ) )
        {
            cfg.hybrid128Mode = true;
            continue;
        }
        if( cli.ReadStr( cfg.temp1Path, "-t1", "--temp1" ) )
        {
            if( !cfg.temp2Path )
                cfg.temp2Path = cfg.temp1Path;
            continue;
        }
        if( cli.ReadStr( cfg.temp2Path, "-t2", "--temp2" ) )
        {
            if( !cfg.temp1Path )
                cfg.temp1Path = cfg.temp2Path;
            continue;
        }
        if( cli.ReadUnswitch( cfg.temp1DirectIO, "--no-t1-direct" ) )
            continue;
        if( cli.ReadUnswitch( cfg.temp2DirectIO, "--no-t2-direct" ) )
            continue;

        if( cli.ReadU64( cfg.plotCheckCount, "--check" ) )
            continue;
        if( cli.ReadF64( cfg.plotCheckThreshhold, "--check-threshold" ) )
            continue;
        if( cli.ReadSwitch( json, "--json" ) )
            continue;
        if( cli.ReadSwitch( listDevices, "-l", "--list" ) )
            continue;
        // if( cli.ReadSwitch( cfg.disableDirectDownloads, "--no-direct-buffers" ) )
        //     continue;
        if( cli.ArgMatch( "--help", "-h" ) )
        {
            CudaK32PlotterPrintHelp();
            exit( 0 );
        }
        else
            break;  // Let the caller handle it
    }
    // The rest should be output directies, parsed by the global config parser.

    if( listDevices )
        ListCudaDevices( json );

    if( cfg.hybrid128Mode && gCfg.compressionLevel <= 0 )
    {
        Log::Error( "Error: Cannot plot classic (uncompressed) plots in 128G or 64G mode." );
        Exit( -1 );
    }

    if( cfg.hybrid16Mode )
    {
        #if PLATFORM_IS_WINDOWS
            Log::Error( "16G mode is currently unsupported on Windows." );
            Exit( -1 );
        #else
            Log::Line( "Warning: 16G mode is experimental and still under development." );
            Log::Line( "         Please use the --check <n> parameter to validate plots when using this mode." );

            if( cfg.temp1DirectIO || cfg.temp2DirectIO )
            {
                Log::Line( "         Direct I/O not supported in 16G mode at the moment. Disabing it." );
                cfg.temp1DirectIO = cfg.temp2DirectIO = false;
            }

        #endif
    }
}

//-----------------------------------------------------------
void CudaK32Plotter::Init()
{
    if( _cx )
        return;

    InitContext( _cfg, _cx );
}

//-----------------------------------------------------------
void InitContext( CudaK32PlotConfig& cfg, CudaK32PlotContext*& outContext )
{
    auto& cx = *new CudaK32PlotContext{};
    outContext = &cx;

    cx.cfg        = cfg;
    cx.gCfg       = cfg.gCfg;

    cx.firstStoredTable = TableId::Table2 + (TableId)cx.gCfg->numDroppedTables;

    Log::Line( "[Bladebit CUDA Plotter]" );
    Log::Line( " Host RAM            : %llu GiB", SysHost::GetTotalSystemMemory() BtoGB );
    
    if( cx.cfg.plotCheckCount == 0 )
        Log::Line( " Plot checks         : disabled" );
    else
    {
        Log::Line( " Plot checks         : enabled ( %llu )", (llu)cx.cfg.plotCheckCount );
        Log::Line( " Plot check threshold: %.3lf", cx.cfg.plotCheckThreshhold );
    }

    // Log::Line( " Direct transfers: %s", cfg.disableDirectDownloads ? "false" : "true" );
    Log::NewLine();

    CudaInit( cx );

    Log::Line( "Marker Set to %d", 48)
CudaErrCheck( cudaStreamCreateWithFlags( &cx.computeStream , cudaStreamNonBlocking ) );
    Log::Line( "Marker Set to %d", 49)
CudaErrCheck( cudaStreamCreateWithFlags( &cx.computeStreamB, cudaStreamNonBlocking ) );
    Log::Line( "Marker Set to %d", 50)
CudaErrCheck( cudaStreamCreateWithFlags( &cx.computeStreamC, cudaStreamNonBlocking ) );
    Log::Line( "Marker Set to %d", 51)
CudaErrCheck( cudaStreamCreateWithFlags( &cx.computeStreamD, cudaStreamNonBlocking ) );

    cudaEventCreateWithFlags( &cx.computeEventA, cudaEventDisableTiming );
    cudaEventCreateWithFlags( &cx.computeEventB, cudaEventDisableTiming );
    cudaEventCreateWithFlags( &cx.computeEventC, cudaEventDisableTiming );

    for( int32 i = 0; i < BBCU_GPU_STREAM_COUNT; i++ )
    {
        cx.gpuDownloadStream[i] = new GpuQueue( GpuQueue::Downloader );
        cx.gpuUploadStream  [i] = new GpuQueue( GpuQueue::Uploader   );
    }

    cx.threadPool = new ThreadPool( SysHost::GetLogicalCPUCount() );
    cx.plotFence  = new Fence();
    cx.parkFence  = new Fence();

    #if _WIN32
        // #MAYBE: Add a configurable option to enable direct downloads on windows?
        // On windows always default to using intermediate pinned buffers
        cx.downloadDirect = false;
    #else
        cx.downloadDirect = cfg.disableDirectDownloads ? false : true;
    #endif

    // cx.plotWriter = new PlotWriter( !cfg.gCfg->disableOutputDirectIO );
    // if( cx.gCfg->benchmarkMode )
    //     cx.plotWriter->EnableDummyMode();

    // Need to do allocations for park serialization differently under the following conditions
    if( cx.downloadDirect || cx.cfg.hybrid128Mode )
    {
        cx.parkContext    = new CudaK32ParkContext{};

        if( cx.cfg.hybrid16Mode )
            cx.useParkContext = true;
    }

    // Check for hybrid mode
    if( cx.cfg.hybrid128Mode )
    {
        cx.diskContext             = new CudaK32HybridMode{};
        cx.diskContext->temp1Queue = new DiskQueue( cx.cfg.temp1Path );

        // Re-use the same queue for temp2 if temp1 and temp2 are pointing to the same path
        auto t1Path = std::filesystem::canonical( cx.cfg.temp1Path );
        auto t2Path = std::filesystem::canonical( cx.cfg.temp2Path );
        if( t1Path.compare( t2Path ) == 0 )
            cx.diskContext->temp2Queue = cx.diskContext->temp1Queue;
        else
            cx.diskContext->temp2Queue = new DiskQueue( cx.cfg.temp2Path );
    }

    cx.phase2 = new CudaK32Phase2{};
    cx.phase3 = new CudaK32Phase3{};

    // #TODO: Support non-warm starting
    Log::Line( "Allocating buffers (this may take a few seconds)..." );
    AllocBuffers( cx );
    InitFSEBitMask( cx );
    Log::Line( "Done." );


    // Allocate GR Context if --check was specified
    if( cfg.plotCheckCount > 0 )
    {
        if( cfg.gCfg->compressionLevel > 0 )
        {
            GreenReaperConfig grCfg{};
            grCfg.apiVersion     = GR_API_VERSION;
            grCfg.threadCount    = 1;
            grCfg.gpuRequest     = GRGpuRequestKind_ExactDevice;
            grCfg.gpuDeviceIndex = cfg.deviceIndex;

            auto grResult = grCreateContext( &cx.grCheckContext, &grCfg, sizeof( grCfg ) );
            FatalIf( grResult != GRResult_OK, "Failed to create decompression context for plot check with error '%s' (%d).",
                    grResultToString( grResult ), (int)grResult );

            grResult = grPreallocateForCompressionLevel( cx.grCheckContext, BBCU_K, cfg.gCfg->compressionLevel );
            FatalIf( grResult != GRResult_OK, "Failed to preallocate memory for decompression context with error '%s' (%d).",
                    grResultToString( grResult ), (int)grResult );
        }

        PlotCheckerConfig checkerCfg{};
        checkerCfg.proofCount         = cfg.plotCheckCount;
        checkerCfg.noGpu              = false;
        checkerCfg.gpuIndex           = cfg.deviceIndex;
        checkerCfg.threadCount        = 1;
        checkerCfg.disableCpuAffinity = false;
        checkerCfg.silent             = false;
        checkerCfg.hasSeed            = false;
        checkerCfg.deletePlots        = true;
        checkerCfg.deleteThreshold    = cfg.plotCheckThreshhold;
        checkerCfg.grContext          = cx.grCheckContext;

        cx.plotChecker = PlotChecker::Create( checkerCfg );
    }
}

//-----------------------------------------------------------
void CudaInit( CudaK32PlotContext& cx )
{
    ASSERT( cx.cudaDevice == -1 );

    // CUDA init
    int deviceCount = 0;
    CudaFatalCheckMsg( cudaGetDeviceCount( &deviceCount ), "Failed to fetch CUDA devices." );
    FatalIf( deviceCount < 1, "No CUDA-capable devices found." );
    FatalIf( cx.cfg.deviceIndex >= deviceCount, "CUDA device %u is out of range out of %d CUDA devices", 
            cx.cfg.deviceIndex, deviceCount );
    
    CudaFatalCheckMsg( cudaSetDevice( (int)cx.cfg.deviceIndex ), "Failed to set cuda device at index %u", cx.cfg.deviceIndex );
    cx.cudaDevice = (int32)cx.cfg.deviceIndex;

    cudaDeviceProp* cudaDevProps = new cudaDeviceProp{};
    Log::Line( "Marker Set to %d", 52)
CudaErrCheck( cudaGetDeviceProperties( cudaDevProps, cx.cudaDevice ) );
    cx.cudaDevProps = cudaDevProps;

    Log::Line( "Selected cuda device %u : %s", cx.cudaDevice, cudaDevProps->name );

    // Get info & limites
    size_t stack = 0, memFree = 0, memTotal = 0;
    cudaMemGetInfo( &memFree, &memTotal );
    cudaDeviceGetLimit( &stack, cudaLimitStackSize );

    Log::Line( " CUDA Compute Capability   : %u.%u", cudaDevProps->major, cudaDevProps->minor );
    Log::Line( " SM count                  : %d", cudaDevProps->multiProcessorCount );
    Log::Line( " Max blocks per SM         : %d", cudaDevProps->maxBlocksPerMultiProcessor );
    Log::Line( " Max threads per SM        : %d", cudaDevProps->maxThreadsPerMultiProcessor );
    Log::Line( " Async Engine Count        : %d", cudaDevProps->asyncEngineCount );
    Log::Line( " L2 cache size             : %.2lf MB", (double)cudaDevProps->l2CacheSize BtoMB );
    Log::Line( " L2 persist cache max size : %.2lf MB", (double)cudaDevProps->persistingL2CacheMaxSize BtoMB );
    Log::Line( " Stack Size                : %.2lf KB", (double)stack   BtoKB );
    Log::Line( " Memory:" );
    Log::Line( "  Total                    : %.2lf GB", (double)memTotal BtoGB );
    Log::Line( "  Free                     : %.2lf GB", (double)memFree  BtoGB );
    Log::Line( "" );

    // Ensure we have the correct capabilities    
    //int supportsCoopLaunch = 0;
    //cudaDeviceGetAttribute( &supportsCoopLaunch, cudaDevAttrCooperativeLaunch, cx.cudaDevice );
    //FatalIf( supportsCoopLaunch != 1, "This CUDA device does not support cooperative kernel launches." );
}

//-----------------------------------------------------------
void ListCudaDevices( const bool json )
{
    cudaError_t err         = cudaSuccess;
    int         deviceCount = 0;

    #define CheckCudaSuccess( x ) if( (err = x) != cudaSuccess ) goto CUDA_ERROR_EXIT;

    {
        CheckCudaSuccess( cudaGetDeviceCount( &deviceCount ) );

        if( deviceCount < 1 )
        {
            const char* e = "No CUDA devices available.";
            if( json )
                Log::Line( R"({"error": "%s"})", e );
            else
                Log::Line( e );
            exit(0);
        }

        if( json )
            Log::Line("[");

        for( int i = 0; i < deviceCount; i++ )
        {
            cudaDeviceProp cudaDevProps{};
            CheckCudaSuccess( cudaGetDeviceProperties( &cudaDevProps, i ) );

            if( json )
            {
                Log::Write( R"(  {"id": %d, "name": "%s"})", i, cudaDevProps.name );
                if( i+1 < deviceCount )
                    Log::Write( "," );

                Log::NewLine();
            }
            else
                Log::Line( "%-2d: %s", i, cudaDevProps.name );
        }

        if( json )
            Log::Line("]");

        exit(0);
    }

    #undef CheckCudaSuccess
    CUDA_ERROR_EXIT:

    if( json )
        Log::Error( R"({ "error": "Failed to list CUDA devices with error 0x%llx: '%s'"})", (llu)err, cudaGetErrorString( err ) );
    else
        Log::Error( "Failed to list CUDA devices with error 0x%llx: '%s'.", (llu)err, cudaGetErrorString( err ) );

    exit(1);
}


///
/// Plotting entry point
///
//-----------------------------------------------------------
void CudaK32Plotter::Run( const PlotRequest& req )
{
    SysHost::InstallCrashHandler();

    // Initialize if needed
    if( _cx == nullptr )
        Init();

    auto&       cx  = *_cx;
    const auto& cfg = _cfg;

    // Only start profiling from here (don't profile allocations)
    Log::Line( "Marker Set to %d", 53)
CudaErrCheck( cudaProfilerStart() );

    ASSERT( cx.plotWriter == nullptr );
    cx.plotWriter = new PlotWriter( !cfg.gCfg->disableOutputDirectIO );
    if( cx.gCfg->benchmarkMode )
        cx.plotWriter->EnableDummyMode();
    if( cx.plotChecker )
        cx.plotWriter->EnablePlotChecking( *cx.plotChecker );

    FatalIf( !cx.plotWriter->BeginPlot( cfg.gCfg->compressionLevel > 0 ? PlotVersion::v2_0 : PlotVersion::v1_0, 
            req.outDir, req.plotFileName, req.plotId, req.memo, req.memoSize, cfg.gCfg->compressionLevel ), 
        "Failed to open plot file with error: %d", cx.plotWriter->GetError() );

    cx.plotRequest = req;
    MakePlot( cx );

    cx.plotWriter->EndPlot( true );

    // Ensure the last plot has ended
    // #TODO: Move it elsewhere, using different buffers for parks
    //        so that we can continue writing to disk until we get to
    //        actually writing the next plot in table 7 finalization.
    {
        const auto pltoCompleteTimer = TimerBegin();
        cx.plotWriter->WaitForPlotToComplete();
        const double plotIOTime = TimerEnd( pltoCompleteTimer );
        Log::Line( "Completed writing plot in %.2lf seconds", plotIOTime );

        if( !cx.plotChecker || !cx.plotChecker->LastPlotDeleted() )
        {
            cx.plotWriter->DumpTables();
            Log::NewLine();
        }
    }
    
    delete cx.plotWriter;
    cx.plotWriter = nullptr;


    // Delete any temporary files
    #if !(DBG_BBCU_KEEP_TEMP_FILES)
        if( cx.plotRequest.IsFinalPlot && cx.cfg.hybrid128Mode )
        {
            if( cx.diskContext->yBuffer )    delete cx.diskContext->yBuffer;
            if( cx.diskContext->metaBuffer ) delete cx.diskContext->metaBuffer;
            if( cx.diskContext->unsortedL )  delete cx.diskContext->unsortedL;
            if( cx.diskContext->unsortedR )  delete cx.diskContext->unsortedR;

            for( TableId t = TableId::Table1; t <= TableId::Table7; t++ )
            {
                if( cx.diskContext->tablesL[(int)t] ) delete cx.diskContext->tablesL[(int)t];
                if( cx.diskContext->tablesR[(int)t] ) delete cx.diskContext->tablesR[(int)t];
            }
        }
    #endif
}

//-----------------------------------------------------------
void MakePlot( CudaK32PlotContext& cx )
{
    memset( cx.bucketCounts    , 0, sizeof( cx.bucketCounts ) );
    memset( cx.bucketSlices    , 0, sizeof( cx.bucketSlices ) );
    memset( cx.tableEntryCounts, 0, sizeof( cx.tableEntryCounts ) );

    cx.table = TableId::Table1;

    const auto plotTimer = TimerBegin();
    const auto p1Timer   = plotTimer;

    #if BBCU_DBG_SKIP_PHASE_1
        DbgLoadContextAndPairs( cx );
    #else

    if( cx.cfg.hybrid128Mode )
    {
        cx.sortedXPairsOut.AssignDiskBuffer( nullptr );
        cx.sortedPairsLOut.AssignDiskBuffer( nullptr );
        cx.sortedPairsROut.AssignDiskBuffer( nullptr );

        if( !cx.plotRequest.isFirstPlot )
        {
            for( TableId t = TableId::Table1; t <= TableId::Table7; t++ )
            {
                if( cx.diskContext->tablesL[(int)t] ) cx.diskContext->tablesL[(int)t]->Swap();
                if( cx.diskContext->tablesR[(int)t] ) cx.diskContext->tablesR[(int)t]->Swap();
            }
            
        }
    }

    /// Generate F1 entries
    {
        Log::Line( "Generating F1" );
        const auto timer = TimerBegin();

        GenF1Cuda( cx );

        const auto elapsed = TimerEnd( timer );
        Log::Line( "Finished F1 in %.2lf seconds.", elapsed );
    }

    /// Forward-propagate the rest of the tables
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
    {
        cx.table  = table;
        cx.bucket = 0;

        FpTable( cx );
    }

    const auto p1Elapsed = TimerEnd( p1Timer );
    Log::Line( "Completed Phase 1 in %.2lf seconds", p1Elapsed );
    #endif

    // Prune
    #if !BBCU_DBG_SKIP_PHASE_2
    const auto p2Timer = TimerBegin();
    CudaK32PlotPhase2( cx );
    const auto p2Elapsed = TimerEnd( p2Timer );
    Log::Line( "Completed Phase 2 in %.2lf seconds", p2Elapsed );
    #endif

    // Compress & write plot tables
    const auto p3Timer = TimerBegin();
    CudaK32PlotPhase3( cx );
    const auto p3Elapsed = TimerEnd( p3Timer );
    Log::Line( "Completed Phase 3 in %.2lf seconds", p3Elapsed );

    auto plotElapsed = TimerEnd( plotTimer );
    Log::Line( "Completed Plot 1 in %.2lf seconds ( %.2lf minutes )", plotElapsed, plotElapsed / 60.0 );
    Log::Line( "" );
}

//-----------------------------------------------------------
void FpTable( CudaK32PlotContext& cx )
{
    memset( &cx.timings, 0, sizeof( cx.timings ) );
    const TableId inTable = cx.table - 1;

    cx.prevTablePairOffset = 0;

    if( cx.cfg.hybrid128Mode )
    {
        auto* diskBufferL = cx.diskContext->tablesL[(int)inTable];
        auto* diskBufferR = cx.diskContext->tablesR[(int)inTable];

        if( inTable == cx.firstStoredTable )
        {
            cx.sortedXPairsOut.AssignDiskBuffer( diskBufferL );
        }
        else if( inTable > cx.firstStoredTable )
        {
            cx.sortedPairsLOut.AssignDiskBuffer( diskBufferL );
            cx.sortedPairsROut.AssignDiskBuffer( diskBufferR );
        }
    }

    // Clear slice counts
    Log::Line( "Marker Set to %d", 54)
CudaErrCheck( cudaMemsetAsync( cx.devSliceCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, cx.computeStream ) );

    // Load initial buckets
    UploadBucketForTable( cx, 0 );

    const auto timer = TimerBegin();
    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        FpTableBucket( cx, bucket );
    }

    Log::Line( "Marker Set to %d", 55)
CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );

    // Copy bucket slices to host
    cudaMemcpyAsync( cx.hostBucketSlices, cx.devSliceCounts, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, 
                        cudaMemcpyDeviceToHost, cx.gpuDownloadStream[0]->GetStream() );
    CudaErrCheck( cudaStreamSynchronize( cx.gpuDownloadStream[0]->GetStream() ) );

    // #TODO: Don't do this copy and instead just use the hostBucketSlices one
    const uint32 outIdx = CudaK32PlotGetOutputIndex( cx );
    memcpy( &cx.bucketSlices[outIdx], cx.hostBucketSlices, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT );

    // #TODO: Do this on the GPU and simply copy it over
    for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
        for( uint32 j = 0; j < BBCU_BUCKET_COUNT; j++ )
            cx.bucketCounts[(int)cx.table][i] += cx.bucketSlices[outIdx][j][i];

    cx.tableEntryCounts[(int)cx.table] = 0;
    for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
        cx.tableEntryCounts[(int)cx.table] += cx.bucketCounts[(int)cx.table][i];

    // Cap entry counts to 2^k
    if( cx.tableEntryCounts[(int)cx.table] > BBCU_TABLE_ENTRY_COUNT )
    {
        const uint32 overflow = (uint32)( cx.tableEntryCounts[(int)cx.table] - BBCU_TABLE_ENTRY_COUNT );
        cx.tableEntryCounts[(int)cx.table] = BBCU_TABLE_ENTRY_COUNT;
        cx.bucketCounts[(int)cx.table][BBCU_BUCKET_COUNT-1] -= overflow;
    }

    cx.yOut.WaitForCompletion();
    cx.yOut.Reset();
    
    cx.xPairsOut.WaitForCompletion();
    cx.xPairsOut.Reset();

    cx.xPairsIn.Reset();

    cx.pairsLOut.WaitForCompletion();
    cx.pairsLOut.Reset();
    cx.pairsROut.WaitForCompletion();
    cx.pairsROut.Reset();

    // #NOTE: Must do this to ensure the buffers are
    //        free for the next go, which use the same underlying buffers
    //        but a different downloader object.
    cx.sortedXPairsOut.WaitForCompletion();
    cx.sortedXPairsOut.Reset();

    cx.sortedPairsLOut.WaitForCompletion();//cx.sortedPairsLOut.WaitForCopyCompletion();
    cx.sortedPairsLOut.Reset();
    cx.sortedPairsROut.WaitForCompletion();//cx.sortedPairsROut.WaitForCopyCompletion();
    cx.sortedPairsROut.Reset();

    if( cx.cfg.hybrid128Mode && inTable >= cx.firstStoredTable )
    {
        if( cx.diskContext->tablesL[(int)inTable] ) cx.diskContext->tablesL[(int)inTable]->Swap();
        if( cx.diskContext->tablesR[(int)inTable] ) cx.diskContext->tablesR[(int)inTable]->Swap();
    }

    if( cx.table < TableId::Table7 )
        cx.metaOut.WaitForCompletion(); cx.metaOut.Reset();

    if( cx.cfg.hybrid128Mode )
    {
        if( cx.cfg.hybrid16Mode || cx.table == cx.firstStoredTable || cx.table == cx.firstStoredTable + 1 )
        {
            cx.diskContext->unsortedL->Swap();
        }

        if( cx.cfg.hybrid16Mode )
        {
            cx.diskContext->yBuffer->Swap();
            cx.diskContext->metaBuffer->Swap();
            cx.diskContext->unsortedR->Swap();
        }
    }

    cx.yIn     .Reset();
    cx.pairsLIn.Reset();
    cx.pairsRIn.Reset();
    cx.metaIn  .Reset();

    const auto elapsed = TimerEnd( timer );
    Log::Line( "Table %u completed in %.2lf seconds with %llu entries.", 
               (uint32)cx.table+1, elapsed, cx.tableEntryCounts[(int)cx.table] );

    /// DEBUG
    #if DBG_BBCU_P1_WRITE_PAIRS
        // Write them sorted, so have to wait until table 3 completes
        if( cx.table > TableId::Table2 )
            DbgWritePairs( cx, cx.table - 1 );
    #endif

    if( cx.table == TableId::Table7 )
    {
       FinalizeTable7( cx );

        // DEBUG
        #if DBG_BBCU_P1_WRITE_PAIRS
           DbgWritePairs( cx, TableId::Table7 );
        #endif
        #if DBG_BBCU_P1_WRITE_CONTEXT
           DbgWriteContext( cx );
        #endif
    }
}

//-----------------------------------------------------------
void FpTableBucket( CudaK32PlotContext& cx, const uint32 bucket )
{
    cx.bucket = bucket;

    // Load next bucket in the background
    if( bucket + 1 < BBCU_BUCKET_COUNT )
        UploadBucketForTable( cx, bucket + 1 );

    const TableId inTable    = cx.table - 1;
    const uint32  entryCount = cx.bucketCounts[(int)inTable][bucket];

    // #NOTE: Ensure these match the ones in UploadBucketForTable()
    cudaStream_t mainStream  = cx.computeStream;
    cudaStream_t metaStream  = cx.computeStream;//B;
    cudaStream_t pairsStream = cx.computeStream;//C;

    uint32* sortKeyIn  = (uint32*)cx.devMatches;
    uint32* sortKeyOut = cx.devSortKey;
    if( cx.table > TableId::Table2 )
    {
        // Generate a sorting key
        CudaK32PlotGenSortKey( entryCount, sortKeyIn, mainStream );
    }
 
    uint32* devYUnsorted    = (uint32*)cx.yIn.GetUploadedDeviceBuffer( mainStream );
    uint32* devMetaUnsorted = nullptr;

    uint32* devYSorted      = cx.devYWork;
    uint32* devMetaSorted   = cx.devMetaWork;

    if( cx.table == TableId::Table2 )
    {
        devMetaUnsorted = (uint32*)cx.metaIn.GetUploadedDeviceBuffer( mainStream );
        sortKeyIn       = devMetaUnsorted;
        sortKeyOut      = devMetaSorted;
    }

    // Sort y w/ key
    CudaErrCheck( cub::DeviceRadixSort::SortPairs<uint32, uint32>( 
        cx.devSortTmp, cx.devSortTmpAllocSize, 
        devYUnsorted,  devYSorted, 
        sortKeyIn,     sortKeyOut, 
        entryCount, 0, 32, mainStream ) );

    CudaErrCheck( cudaEventRecord( cx.computeEventC, mainStream ) );
    CudaErrCheck( cudaEventRecord( cx.computeEventA, mainStream ) );

    cx.yIn.ReleaseDeviceBuffer( mainStream );
    if( cx.table == TableId::Table2 )
        cx.metaIn.ReleaseDeviceBuffer( mainStream );

    // Sort and download prev table's pairs
    const bool isLTableInlineable = cx.table == TableId::Table2 || (uint32)cx.table <= cx.gCfg->numDroppedTables+1;

    if( !isLTableInlineable )
    {
        CudaErrCheck( cudaStreamWaitEvent( pairsStream, cx.computeEventC ) );   // Ensure sort key is ready

        const bool isLTableInlinedPairs = (uint32)cx.table == cx.gCfg->numDroppedTables + 2;

        if( isLTableInlinedPairs )
        {
            // Table 2's pairs are inlined x's. Treat as Pairs
            Pair* pairsIn     = (Pair*)cx.xPairsIn.GetUploadedDeviceBuffer( pairsStream );
            Pair* sortedPairs = (Pair*)cx.sortedXPairsOut.LockDeviceBuffer( pairsStream );

            CudaK32PlotSortByKey( entryCount, sortKeyOut, pairsIn, sortedPairs, pairsStream );
            cx.xPairsIn.ReleaseDeviceBuffer( pairsStream );

            Pair* hostPairs = ((Pair*)cx.hostBackPointers[(int)inTable].left) + cx.prevTablePairOffset;

            // Write sorted pairs back to host
            cx.sortedXPairsOut.DownloadT( hostPairs, entryCount, pairsStream, cx.downloadDirect );
        }
        else
        {
            // uint32* hostPairsL; 
            // uint16* hostPairsR; 

            // Wait for pairs to complete loading and sort on Y (or do this before match? Giving us time to write to disk while matching?)
            uint32* pairsLIn     = (uint32*)cx.pairsLIn       .GetUploadedDeviceBuffer( pairsStream );
            uint32* sortedPairsL = (uint32*)cx.sortedPairsLOut.LockDeviceBuffer( pairsStream );
            CudaK32PlotSortByKey( entryCount, sortKeyOut, pairsLIn, sortedPairsL, pairsStream );
            cx.pairsLIn.ReleaseDeviceBuffer( pairsStream );
            // hostPairsL      = cx.hostTableSortedL + cx.prevTablePairOffset;

            uint32* hostPairsLFinal = cx.hostBackPointers[(int)inTable].left  + cx.prevTablePairOffset;
            cx.sortedPairsLOut.DownloadT( hostPairsLFinal, entryCount, pairsStream, cx.downloadDirect );
            // cx.sortedPairsLOut.DownloadAndCopyT( hostPairsL, hostPairsLFinal, entryCount, pairsStream );

            // if( !isOutputCompressed )
            {
                uint16* pairsRIn     = (uint16*)cx.pairsRIn       .GetUploadedDeviceBuffer( pairsStream );
                uint16* sortedPairsR = (uint16*)cx.sortedPairsROut.LockDeviceBuffer( pairsStream );
                CudaK32PlotSortByKey( entryCount, sortKeyOut, pairsRIn, sortedPairsR, pairsStream );
                cx.pairsRIn.ReleaseDeviceBuffer( pairsStream );
                // hostPairsR      = cx.hostTableSortedR + cx.prevTablePairOffset; 

                uint16* hostPairsRFinal = cx.hostBackPointers[(int)inTable].right + cx.prevTablePairOffset;
                
                cx.sortedPairsROut.DownloadT( hostPairsRFinal, entryCount, pairsStream, cx.downloadDirect );
                // cx.sortedPairsROut.DownloadAndCopyT( hostPairsR, hostPairsRFinal, entryCount, pairsStream );
            }
        }
    }

    // Match pairs
    CudaMatchBucketizedK32( cx, devYSorted, mainStream, nullptr );

    // Inline input x's or compressed x's
    if( isLTableInlineable )
    {
        uint32* inlineInput = devMetaSorted;

        if( cx.table > TableId::Table2 )
        {
            uint32* pairsLIn = (uint32*)cx.pairsLIn.GetUploadedDeviceBuffer( pairsStream );
            inlineInput = cx.devXInlineInput;

            CudaK32PlotSortByKey( entryCount, sortKeyOut, pairsLIn, inlineInput, pairsStream );
            cx.pairsLIn.ReleaseDeviceBuffer( pairsStream );
        }

        // Inline x values into our new pairs (merge L table into R table)
        InlineTable( cx, inlineInput, mainStream );
    }

    // Upload and sort metadata
    if( cx.table > TableId::Table2 )
    {
        const uint32 metaMultiplier = GetTableMetaMultiplier( cx.table - 1 );

        // Wait for meta to complete loading, and sort on Y
        devMetaUnsorted = (uint32*)cx.metaIn.GetUploadedDeviceBuffer( metaStream );

        // Ensure the sort key is ready
        CudaErrCheck( cudaStreamWaitEvent( metaStream, cx.computeEventA ) );

        switch( metaMultiplier )
        {
            case 2: CudaK32PlotSortByKey( entryCount, sortKeyOut, (K32Meta2*)devMetaUnsorted, (K32Meta2*)devMetaSorted, metaStream ); break;
            case 3: CudaK32PlotSortByKey( entryCount, sortKeyOut, (K32Meta3*)devMetaUnsorted, (K32Meta3*)devMetaSorted, metaStream ); break;
            case 4: CudaK32PlotSortByKey( entryCount, sortKeyOut, (K32Meta4*)devMetaUnsorted, (K32Meta4*)devMetaSorted, metaStream ); break;
            default: ASSERT( 0 ); break;
        }
        cx.metaIn.ReleaseDeviceBuffer( metaStream );
        CudaErrCheck( cudaEventRecord( cx.computeEventB, metaStream ) );
    }

    // Ensure metadata is sorted
    CudaErrCheck( cudaStreamWaitEvent( mainStream, cx.computeEventB ) );

    // Compute Fx
    GenFx( cx, devYSorted, devMetaSorted, mainStream );

    CudaK32PlotDownloadBucket( cx );

    cx.prevTablePairOffset += entryCount;
}

//-----------------------------------------------------------
void FinalizeTable7( CudaK32PlotContext& cx )
{
    Log::Line( "Finalizing Table 7" );

    const auto timer = TimerBegin();

    cx.table               = TableId::Table7+1;   // Set a false table
    cx.prevTablePairOffset = 0;

    // Upload initial bucket
    UploadBucketForTable( cx, 0 );


    // Prepare C1 & 2 tables
    const uint32 c1Interval       = kCheckpoint1Interval;
    const uint32 c2Interval       = kCheckpoint1Interval * kCheckpoint2Interval;

    const uint64 tableLength      = cx.tableEntryCounts[(int)TableId::Table7];
    const uint32 c1TotalEntries   = (uint32)CDiv( tableLength, (int)c1Interval ) + 1; // +1 because chiapos adds an extra '0' entry at the end
    const uint32 c2TotalEntries   = (uint32)CDiv( tableLength, (int)c2Interval ) + 1; // +1 because we add a short-circuit entry to prevent C2 lookup overflows

    const size_t c1TableSizeBytes = c1TotalEntries * sizeof( uint32 );
    const size_t c2TableSizeBytes = c2TotalEntries * sizeof( uint32 );

    if( cx.cfg.hybrid128Mode )
    {
        cx.sortedPairsLOut.AssignDiskBuffer( cx.diskContext->tablesL[(int)TableId::Table7] );
        cx.sortedPairsROut.AssignDiskBuffer( cx.diskContext->tablesR[(int)TableId::Table7] );
    }


    // Re-use meta GPU downloader to download parks
    GpuDownloadBuffer& parkDownloader = cx.metaOut;

    // Store disk buffer temporarily, if there is one, since we don't want to write to meta now
    DiskBufferBase* metaDiskBuffer = parkDownloader.GetDiskBuffer();

    // Reset park buffer chain, if we're using it
    if( cx.parkContext )
    {
        cx.parkContext->parkBufferChain->Reset();
        parkDownloader.AssignDiskBuffer( nullptr ); // We want direct downloads to the park buffers, which are pinned already
    }

    // Prepare host allocations
    constexpr size_t c3ParkSize = CalculateC3Size();

    const uint64 totalParkSize = CDivT( tableLength, (uint64)kCheckpoint1Interval ) * c3ParkSize;

    StackAllocator hostAlloc = cx.parkContext
        ? StackAllocator( cx.parkContext->table7Memory.Ptr(), cx.parkContext->table7Memory.Length() )
        : StackAllocator( cx.hostMeta, BBCU_TABLE_ALLOC_ENTRY_COUNT * sizeof( uint32 ) * 4 );

    uint32* hostC1Buffer        = hostAlloc.CAlloc<uint32>( c1TotalEntries );
    uint32* hostC2Buffer        = hostAlloc.CAlloc<uint32>( c2TotalEntries );
    uint32* hostLastParkEntries = hostAlloc.CAlloc<uint32>( kCheckpoint1Interval );
    byte*   hostLastParkBuffer  = (byte*)hostAlloc.CAlloc<uint32>( kCheckpoint1Interval );
    byte*   hostCompressedParks = cx.parkContext ? nullptr : hostAlloc.AllocT<byte>( totalParkSize );

    byte*   hostParkWriter      = hostCompressedParks;
    uint32* hostC1Writer        = hostC1Buffer;

    // Prepare device allocations
    constexpr size_t devAllocatorSize = BBCU_BUCKET_ALLOC_ENTRY_COUNT * BBCU_HOST_META_MULTIPLIER * sizeof( uint32 );
    StackAllocator devAlloc( cx.devMetaWork, devAllocatorSize );

    constexpr uint32 maxParksPerBucket = CuCDiv( BBCU_BUCKET_ENTRY_COUNT, kCheckpoint1Interval ) + 1;
    static_assert( maxParksPerBucket * c3ParkSize < devAllocatorSize );

    uint32* devC1Buffer = devAlloc.CAlloc<uint32>( c1TotalEntries );
    uint32* devC1Writer = devC1Buffer;

    const size_t parkBufferSize = kCheckpoint1Interval * sizeof( uint32 );

    cudaStream_t mainStream     = cx.computeStream;
    cudaStream_t metaStream     = cx.computeStream;//B;
    cudaStream_t pairsStream    = cx.computeStream;//C;
    cudaStream_t downloadStream = cx.gpuDownloadStream[0]->GetStream();

    // Load CTable
    FSE_CTable* devCTable = devAlloc.AllocT<FSE_CTable>( sizeof( CTable_C3 ), sizeof( uint64 ) );
    CudaErrCheck( cudaMemcpyAsync( devCTable, CTable_C3, sizeof( CTable_C3 ), cudaMemcpyHostToDevice, cx.computeStream ) );
    CudaErrCheck( cudaStreamSynchronize( cx.computeStream  ) );

    // Prepare plot tables
    cx.plotWriter->ReserveTableSize( PlotTable::C1, c1TableSizeBytes );
    cx.plotWriter->ReserveTableSize( PlotTable::C2, c2TableSizeBytes );
    cx.plotWriter->BeginTable( PlotTable::C3 );

    // Save a buffer with space before the start of it for us to copy retained entries for the next park.
    uint32  retainedC3EntryCount = 0;
    uint32* devYSorted           = cx.devYWork + kCheckpoint1Interval;

    uint32* sortKeyIn  = (uint32*)cx.devMatches;
    uint32* sortKeyOut = cx.devSortKey;

    // Compress parks
    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        cx.bucket = bucket;

        // Upload next bucket
        if( bucket + 1 < BBCU_BUCKET_COUNT )
            UploadBucketForTable( cx, bucket+1 );

        const uint32 entryCount = cx.bucketCounts[(int)TableId::Table7][bucket];
        ASSERT( entryCount > kCheckpoint1Interval );


        // Generate a sorting key
        CudaK32PlotGenSortKey( entryCount, sortKeyIn, mainStream );

        // Sort y w/ key
        uint32* devYUnsorted = (uint32*)cx.yIn.GetUploadedDeviceBuffer( mainStream );

        CudaErrCheck( cub::DeviceRadixSort::SortPairs<uint32, uint32>( 
            cx.devSortTmp, cx.devSortTmpAllocSize, 
            devYUnsorted, devYSorted,
            sortKeyIn, sortKeyOut, 
            entryCount, 0, 32, mainStream ) );

        CudaErrCheck( cudaEventRecord( cx.computeEventA, mainStream ) );
        cx.yIn.ReleaseDeviceBuffer( mainStream ); devYUnsorted = nullptr;

        // Sort pairs
        {
            CudaErrCheck( cudaStreamWaitEvent( pairsStream, cx.computeEventA ) );   // Wait for the sort key to be ready

            uint32* sortedPairsL = (uint32*)cx.sortedPairsLOut.LockDeviceBuffer( pairsStream );
            uint32* pairsLIn     = (uint32*)cx.pairsLIn.GetUploadedDeviceBuffer( pairsStream );
            CudaK32PlotSortByKey( entryCount, sortKeyOut, pairsLIn, sortedPairsL, pairsStream );
            cx.pairsLIn.ReleaseDeviceBuffer( pairsStream );

            uint16* sortedPairsR = (uint16*)cx.sortedPairsROut.LockDeviceBuffer( pairsStream );
            uint16* pairsRIn     = (uint16*)cx.pairsRIn.GetUploadedDeviceBuffer( pairsStream );
            CudaK32PlotSortByKey( entryCount, sortKeyOut, pairsRIn, sortedPairsR, pairsStream );
            cx.pairsRIn.ReleaseDeviceBuffer( pairsStream );


            // Download sorted pairs back to host
            // uint32* hostPairsL      = cx.hostTableSortedL + cx.prevTablePairOffset;
            // uint16* hostPairsR      = cx.hostTableSortedR + cx.prevTablePairOffset;
            uint32* hostPairsLFinal = cx.hostBackPointers[(int)TableId::Table7].left  + cx.prevTablePairOffset;
            uint16* hostPairsRFinal = cx.hostBackPointers[(int)TableId::Table7].right + cx.prevTablePairOffset;

            // cx.sortedPairsLOut.DownloadAndCopyT( hostPairsL, hostPairsLFinal, entryCount, pairsStream );
            // cx.sortedPairsROut.DownloadAndCopyT( hostPairsR, hostPairsRFinal, entryCount, pairsStream );
            cx.sortedPairsLOut.DownloadT( hostPairsLFinal, entryCount, pairsStream, true );
            cx.sortedPairsROut.DownloadT( hostPairsRFinal, entryCount, pairsStream, true );

            cx.prevTablePairOffset += entryCount;
        }


        // If we previously had entries retained, adjust our buffer and counts accordingly
        uint32* devF7Entries = devYSorted - retainedC3EntryCount;
        uint32  f7EntryCount = entryCount + retainedC3EntryCount;

        const uint32 parkCount = f7EntryCount / kCheckpoint1Interval;

        // Copy C1 entries
        CudaErrCheck( cudaMemcpy2DAsync( devC1Writer, sizeof( uint32 ), devF7Entries, sizeof( uint32 ) * c1Interval,
                                         sizeof( uint32 ), parkCount, cudaMemcpyDeviceToDevice, mainStream ) );
        devC1Writer += parkCount;

        // Compress C tables
        // This action mutates the f7 buffer in-place, so ensure the C1 copies happen before this call
        byte* devParkBuffer = (byte*)parkDownloader.LockDeviceBuffer( mainStream );
        CompressC3ParksInGPU( parkCount, devF7Entries, devParkBuffer, c3ParkSize, devCTable, mainStream );

        // Retain any new f7 entries for the next bucket, if ndeeded
        retainedC3EntryCount = f7EntryCount - (parkCount * kCheckpoint1Interval);
        if( retainedC3EntryCount > 0 )
        {
            // Last bucket?
            const bool isLastBucket = bucket + 1 == BBCU_BUCKET_COUNT;

            const uint32  compressedEntryCount = parkCount * kCheckpoint1Interval;
            const uint32* copySource           = devF7Entries + compressedEntryCount;
            const size_t  copySize             = sizeof( uint32 ) * retainedC3EntryCount;

            if( !isLastBucket )
            {
                // Not the last bucket, so retain entries for the next GPU compression bucket
                CudaErrCheck( cudaMemcpyAsync( devYSorted - retainedC3EntryCount, copySource, copySize, 
                                                cudaMemcpyDeviceToDevice, mainStream ) );
            }
            else
            {
                // No more buckets so we have to compress this last park on the CPU
                CudaErrCheck( cudaMemcpyAsync( hostLastParkEntries, copySource, copySize, 
                                                cudaMemcpyDeviceToHost, downloadStream ) );
            }
        }

        // Download compressed parks to host
        const size_t parkDownloadSize = c3ParkSize * parkCount;

        if( cx.parkContext )
        {
            ASSERT( parkDownloadSize <= cx.parkContext->parkBufferChain->BufferSize() );

            // Override the park buffer to be used when using a park context
            hostParkWriter = cx.parkContext->parkBufferChain->PeekBuffer( bucket );

            // Wait for the next park buffer to be available to be used for download
            parkDownloader.HostCallback([&cx]{
               (void)cx.parkContext->parkBufferChain->GetNextBuffer();
            });
        }

        const bool directOverride = cx.parkContext != nullptr;
    
        parkDownloader.DownloadWithCallback( hostParkWriter, parkDownloadSize, 
            []( void* parksBuffer, size_t size, void* userData ) {

                auto& cx = *reinterpret_cast<CudaK32PlotContext*>( userData );

                cx.plotWriter->WriteTableData( parksBuffer, size );

                // Release the buffer after the plot writer is done with it.
                if( cx.parkContext )
                {
                    cx.plotWriter->CallBack([&cx](){
                        cx.parkContext->parkBufferChain->ReleaseNextBuffer();
                    });
                }

            }, &cx, mainStream, directOverride );
        hostParkWriter += parkDownloadSize;

        if( cx.parkContext )
            hostParkWriter = nullptr;
    }

    // Download c1 entries
    const size_t devC1EntryCount = (size_t)(uintptr_t)(devC1Writer - devC1Buffer);
    CudaErrCheck( cudaMemcpyAsync( hostC1Buffer, devC1Buffer, sizeof( uint32 ) * devC1EntryCount, cudaMemcpyDeviceToHost, downloadStream ) );
    hostC1Writer += devC1EntryCount;

    // Wait for parks to finish downloading
    parkDownloader.WaitForCompletion();
    parkDownloader.Reset();

    // Was there a left-over park?
    if( retainedC3EntryCount > 0 )
    {
        // Copy c1 entry
        *hostC1Writer++ = hostLastParkEntries[0];
        ASSERT( hostC1Writer - hostC1Buffer == c1TotalEntries - 1 );

        // Serialize and trailing park and submit it to the plot
        if( retainedC3EntryCount > 1 )
        {
            TableWriter::WriteC3Park( retainedC3EntryCount - 1, hostLastParkEntries, hostLastParkBuffer );
            cx.plotWriter->WriteTableData( hostLastParkBuffer, c3ParkSize );
        }
    }

    // Write final empty C entries
    hostC1Buffer[c1TotalEntries-1] = 0;
    hostC2Buffer[c2TotalEntries-1] = 0;

    // Byte-swap C1 
    for( uint32 i = 0; i < c1TotalEntries-1; i++ )
        hostC1Buffer[i] = Swap32( hostC1Buffer[i] );

    // Calculate C2 entries
    for( uint32 i = 0; i < c2TotalEntries-1; i++ )
    {
        ASSERT( i * kCheckpoint2Interval < c1TotalEntries - 1 );
        hostC2Buffer[i] = hostC1Buffer[i * kCheckpoint2Interval];
    }

    // End C3 table & write C1 & C2 tables
    cx.plotWriter->EndTable();
    cx.plotWriter->WriteReservedTable( PlotTable::C1, hostC1Buffer );
    cx.plotWriter->WriteReservedTable( PlotTable::C2, hostC2Buffer );
    cx.plotWriter->SignalFence( *cx.plotFence );    // Signal the fence for the start of Phase 3 when we have to use our tmp2 host buffer again


    // Cleanup
    cx.sortedPairsLOut.WaitForCompletion();
    cx.sortedPairsROut.WaitForCompletion();
    cx.sortedPairsLOut.Reset();
    cx.sortedPairsROut.Reset();

    cx.prevTablePairOffset = 0;

    // Restore disk buffer on repurposed meta download stream
    parkDownloader.AssignDiskBuffer( metaDiskBuffer );

    if( cx.cfg.hybrid128Mode )
    {
        cx.diskContext->tablesL[(int)TableId::Table7]->Swap();
        cx.diskContext->tablesR[(int)TableId::Table7]->Swap();

        if( cx.cfg.hybrid16Mode )
            cx.diskContext->yBuffer->Swap();
    }

    auto elapsed = TimerEnd( timer );
    Log::Line( "Finalized Table 7 in %.2lf seconds.", elapsed );
}

//-----------------------------------------------------------
__global__ void CudaInlineTable( const uint32* entryCount, const uint32* inX, const Pair* matches, Pair* inlinedPairs, uint32 entryBits = 0 )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    if( gid >= *entryCount )
        return;

    const Pair pair = matches[gid];

    Pair inlined;
    inlined.left  = inX[pair.left ];
    inlined.right = inX[pair.right];

    CUDA_ASSERT( inlined.left || inlined.right );

    inlinedPairs[gid] = inlined;
}

//-----------------------------------------------------------
template<bool UseLP>
__global__ void CudaCompressTable( const uint32* entryCount, const uint32* inLEntries, const Pair* matches, uint32* outREntries, const uint32 bitShift )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    if( gid >= *entryCount )
        return;

    const Pair pair = matches[gid];

    const uint32 x0 = inLEntries[pair.left ];
    const uint32 x1 = inLEntries[pair.right];

    // Convert to linepoint
    if constexpr ( UseLP )         
        outREntries[gid] = (uint32)CudaSquareToLinePoint64( x1 >> bitShift, x0 >> bitShift );
    else
        outREntries[gid] =  ((x1 >> bitShift) << (32-bitShift) ) | (x0 >> bitShift);
}

//-----------------------------------------------------------
void InlineTable( CudaK32PlotContext& cx, const uint32* devInX, cudaStream_t stream )
{
    static_assert( alignof( Pair ) == sizeof( uint32 ) );

    const bool isCompressedInput = cx.gCfg->compressionLevel > 0 && (uint32)cx.table <= cx.gCfg->numDroppedTables;

    const uint32 kthreads = 256;
    const uint32 kblocks  = CDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, (int)kthreads );

    if( isCompressedInput )
    {
        const bool   isFinalTable = cx.table == TableId::Table1 + (TableId)cx.gCfg->numDroppedTables;
        const uint32 bitShift     = ( isFinalTable && cx.gCfg->numDroppedTables > 1 ) ? 0 : BBCU_K - cx.gCfg->compressedEntryBits;

        if( isFinalTable )
            CudaCompressTable<true><<<kblocks, kthreads, 0, stream>>>( cx.devMatchCount, devInX, cx.devMatches, cx.devCompressedXs, bitShift );
        else
            CudaCompressTable<false><<<kblocks, kthreads, 0, stream>>>( cx.devMatchCount, devInX, cx.devMatches, cx.devCompressedXs, bitShift );
    }
    else
    {
        CudaInlineTable<<<kblocks, kthreads, 0, stream>>>( cx.devMatchCount, devInX, cx.devMatches, cx.devInlinedXs );
    }
}

//-----------------------------------------------------------
void CudaK32PlotDownloadBucket( CudaK32PlotContext& cx )
{
    const bool   writeVertical  = CudaK32PlotIsOutputVertical( cx );
    const size_t metaMultiplier = GetTableMetaMultiplier( cx.table );

    const bool   downloadCompressed   = cx.table > TableId::Table1 && (uint32)cx.table <= cx.gCfg->numDroppedTables;
    const bool   downloadInlinedPairs = !downloadCompressed && (uint32)cx.table == cx.gCfg->numDroppedTables+1;

    uint32* hostY        = cx.hostY;
    uint32* hostMeta     = cx.hostMeta;

    uint32* hostPairsL   = cx.hostTableL;
    uint16* hostPairsR   = cx.hostTableR;
    Pair*   t2HostPairs  = (Pair*)cx.hostBackPointers[4].left;

    const size_t startOffset  = cx.bucket * ( writeVertical ? BBCU_MAX_SLICE_ENTRY_COUNT : BBCU_BUCKET_ALLOC_ENTRY_COUNT );  // vertical: offset to starting col. horizontal: to starting row
    const size_t width        = BBCU_MAX_SLICE_ENTRY_COUNT;
    const size_t height       = BBCU_BUCKET_COUNT;
    const size_t dstStride    = writeVertical ? BBCU_BUCKET_ALLOC_ENTRY_COUNT : BBCU_MAX_SLICE_ENTRY_COUNT;
    const size_t srcStride    = BBCU_MAX_SLICE_ENTRY_COUNT;

    cx.yOut.Download2DT<uint32>( hostY + startOffset, width, height, dstStride, srcStride, cx.computeStream );

    // Metadata
    if( metaMultiplier > 0 )
    {
        const size_t metaSizeMultiplier = metaMultiplier == 3 ? 4 : metaMultiplier;
        const size_t metaSize           = sizeof( uint32 ) * metaSizeMultiplier;

        const size_t  metaSrcStride = srcStride * metaSize;
        const size_t  metaDstStride = dstStride * sizeof( K32Meta4 );
        const size_t  metaWidth     = width * metaSize;
              uint32* meta          = hostMeta + startOffset * 4;

        cx.metaOut.Download2D( meta, metaWidth, height, metaDstStride, metaSrcStride, cx.computeStream );
    }

    if( cx.table > TableId::Table1 )
    {
        if( downloadInlinedPairs )
        {
            cx.xPairsOut.Download2DT<Pair>( t2HostPairs + startOffset, width, height, dstStride, srcStride, cx.computeStream );
        }
        else
        {
            cx.pairsLOut.Download2DT<uint32>( hostPairsL + startOffset, width, height, dstStride, srcStride, cx.computeStream );

            if( !downloadCompressed )
                cx.pairsROut.Download2DT<uint16>( hostPairsR + startOffset, width, height, dstStride, srcStride, cx.computeStream );
        }
    }
}

//-----------------------------------------------------------
void UploadBucketForTable( CudaK32PlotContext& cx, const uint64 bucket )
{
    const TableId rTable  = cx.table;
    const TableId inTable = rTable - 1;

    const uint32  metaMultiplier = GetTableMetaMultiplier( inTable );

    const uint32  inIdx        = CudaK32PlotGetInputIndex( cx );
    const bool    readVertical = CudaK32PlotIsOutputVertical( cx );

    const uint32* hostY        = cx.hostY;
    const uint32* hostMeta     = cx.hostMeta;
    const uint32* hostPairsL   = cx.hostTableL;
    const uint16* hostPairsR   = cx.hostTableR;

    const bool   uploadCompressed   = cx.table > TableId::Table2 && (uint32)cx.table-1 <= cx.gCfg->numDroppedTables;
    const bool   uploadInlinedPairs = !uploadCompressed && (uint32)cx.table == cx.gCfg->numDroppedTables+2;
    const Pair*  t2HostPairs        = (Pair*)cx.hostBackPointers[4].left; // Table 2 will use table 5, and overflow onto 6

    if( cx.cfg.hybrid128Mode )
        t2HostPairs = (Pair*)hostPairsL;

    uint32 stride = BBCU_BUCKET_ALLOC_ENTRY_COUNT;          // Start as vertical
    size_t offset = (size_t)bucket * BBCU_MAX_SLICE_ENTRY_COUNT;

    if( !readVertical )
    {
        // Adjust to starting row
        stride = BBCU_MAX_SLICE_ENTRY_COUNT;
        offset = (size_t)bucket * BBCU_BUCKET_ALLOC_ENTRY_COUNT;
    }

    cudaStream_t mainStream  = cx.computeStream;
    cudaStream_t metaStream  = cx.computeStream;//B;
    cudaStream_t pairsStream = cx.computeStream;//C;

    const uint32* counts = &cx.bucketSlices[inIdx][0][bucket];

    cx.yIn.UploadArrayT<uint32>( hostY + offset, BBCU_BUCKET_COUNT, stride, BBCU_BUCKET_COUNT, counts, cx.computeStream );

    // Upload pairs, also
    if( cx.table > TableId::Table2 )
    {
        if( uploadInlinedPairs )
        {
            cx.xPairsIn.UploadArrayT<Pair>( t2HostPairs + offset, BBCU_BUCKET_COUNT, stride, BBCU_BUCKET_COUNT, counts, pairsStream );
        }
        else
        {
            cx.pairsLIn.UploadArrayT<uint32>( hostPairsL + offset, BBCU_BUCKET_COUNT, stride, BBCU_BUCKET_COUNT, counts, pairsStream );

            if( !uploadCompressed )
                cx.pairsRIn.UploadArrayT<uint16>( hostPairsR + offset, BBCU_BUCKET_COUNT, stride, BBCU_BUCKET_COUNT, counts, pairsStream );
        }
    }

    // Meta
    if( metaMultiplier > 0 )
    {
        const size_t metaSizeMultiplier = metaMultiplier == 3 ? 4 : metaMultiplier;
        const size_t metaSize           = sizeof( uint32 ) * metaSizeMultiplier;

        auto actualMetaStream = inTable == TableId::Table1 ? cx.computeStream : metaStream;

        cx.metaIn.UploadArray( hostMeta + offset * 4, BBCU_BUCKET_COUNT, metaSize, stride * sizeof( K32Meta4 ), BBCU_BUCKET_COUNT, counts, actualMetaStream );
    }
}



///
/// Allocations
///
//-----------------------------------------------------------
void AllocBuffers( CudaK32PlotContext& cx )
{
    // Determine initially the largest required size

    const size_t alignment = bbclamp<size_t>( SysHost::GetPageSize(), sizeof( K32Meta4 ), 4096 );
    cx.allocAlignment     = alignment;
    cx.pinnedAllocSize    = 0;
    cx.hostTableAllocSize = 0;
    cx.hostTempAllocSize  = 0;
    cx.devAllocSize       = 0;

    // If on <= 128G mode or not using direct downloads, 
    // we need to use a separate buffer for downloading parks, instead of re-using exisintg ones.
    // If on <= 64G mode or not using direct downloads, 
    const bool allocateParkBuffers = cx.downloadDirect || cx.cfg.hybrid128Mode;
    size_t parksPinnedSize = 0;

    // Gather the size needed first
    {
        CudaK32AllocContext acx = {};

        acx.alignment = alignment;
        acx.dryRun    = true;

        DummyAllocator pinnedAllocator;
        DummyAllocator hostTableAllocator;
        DummyAllocator hostTempAllocator;
        DummyAllocator devAllocator;

        acx.pinnedAllocator    = &pinnedAllocator;
        acx.hostTableAllocator = &hostTableAllocator;
        acx.hostTempAllocator  = &hostTempAllocator;
        acx.devAllocator       = &devAllocator;

        AllocateP1Buffers( cx, acx );
        cx.pinnedAllocSize    = pinnedAllocator   .Size();
        cx.hostTableAllocSize = hostTableAllocator.Size();
        cx.hostTempAllocSize  = hostTempAllocator .Size();
        cx.devAllocSize       = devAllocator      .Size();

        /// Phase 2
        pinnedAllocator    = {};
        hostTableAllocator = {};
        hostTempAllocator  = {};
        devAllocator       = {};

        CudaK32PlotPhase2AllocateBuffers( cx, acx );
        cx.pinnedAllocSize    = std::max( cx.pinnedAllocSize   , pinnedAllocator   .Size() );
        cx.hostTableAllocSize = std::max( cx.hostTableAllocSize, hostTableAllocator.Size() );
        cx.hostTempAllocSize  = std::max( cx.hostTempAllocSize , hostTempAllocator .Size() );
        cx.devAllocSize       = std::max( cx.devAllocSize      , devAllocator      .Size() );

        /// Phase 3
        pinnedAllocator    = {};
        hostTableAllocator = {};
        hostTempAllocator  = {};
        devAllocator       = {};

        CudaK32PlotPhase3AllocateBuffers( cx, acx );
        cx.pinnedAllocSize    = std::max( cx.pinnedAllocSize   , pinnedAllocator   .Size() );
        cx.hostTableAllocSize = std::max( cx.hostTableAllocSize, hostTableAllocator.Size() );
        cx.hostTempAllocSize  = std::max( cx.hostTempAllocSize , hostTempAllocator .Size() );
        cx.devAllocSize       = std::max( cx.devAllocSize      , devAllocator      .Size() );

        // May need to allocate extra pinned buffers for park buffers
        if( allocateParkBuffers )
        {
            pinnedAllocator = {};
            AllocateParkSerializationBuffers( cx, *acx.pinnedAllocator, acx.dryRun );
            parksPinnedSize = pinnedAllocator.Size();
        }
    }


    const size_t totalPinnedSize = cx.pinnedAllocSize + cx.hostTempAllocSize + parksPinnedSize;
    const size_t totalHostSize   = cx.hostTableAllocSize + totalPinnedSize;
    Log::Line( "Kernel RAM required       : %-12llu bytes ( %-9.2lf MiB or %-6.2lf GiB )", totalPinnedSize,
                   (double)totalPinnedSize BtoMB, (double)totalPinnedSize BtoGB );

    Log::Line( "Intermediate RAM required : %-12llu bytes ( %-9.2lf MiB or %-6.2lf GiB )", cx.pinnedAllocSize,
                   (double)cx.pinnedAllocSize BtoMB, (double)cx.pinnedAllocSize BtoGB );

    Log::Line( "Host RAM required         : %-12llu bytes ( %-9.2lf MiB or %-6.2lf GiB )", cx.hostTableAllocSize,
                    (double)cx.hostTableAllocSize BtoMB, (double)cx.hostTableAllocSize BtoGB );

    Log::Line( "Total Host RAM required   : %-12llu bytes ( %-9.2lf MiB or %-6.2lf GiB )", totalHostSize,
                    (double)totalHostSize BtoMB, (double)totalHostSize BtoGB );

    Log::Line( "GPU RAM required          : %-12llu bytes ( %-9.2lf MiB or %-6.2lf GiB )", cx.devAllocSize,
                   (double)cx.devAllocSize BtoMB, (double)cx.devAllocSize BtoGB );

    // Now actually allocate the buffers
    Log::Line( "Allocating buffers..." );
    CudaErrCheck( cudaMallocHost( &cx.pinnedBuffer, cx.pinnedAllocSize, cudaHostAllocDefault ) );

    #if _DEBUG
        cx.hostBufferTables = bbvirtallocboundednuma<byte>( cx.hostTableAllocSize );
    #else

        bool allocateHostTablesPinned = cx.downloadDirect;
        #if _WIN32
            // On windows we always force the use of intermediate buffers, so we allocate on the host
            allocateHostTablesPinned = false;
        #endif

        // Log::Line( "Table pairs allocated as pinned: %s", allocateHostTablesPinned ? "true" : "false" );
        if( allocateHostTablesPinned )
            CudaErrCheck( cudaMallocHost( &cx.hostBufferTables, cx.hostTableAllocSize, cudaHostAllocDefault ) );
        else
            cx.hostBufferTables = bbvirtallocboundednuma<byte>( cx.hostTableAllocSize );
    #endif

    cx.hostBufferTemp = nullptr;
    #if _DEBUG || _WIN32
        if( cx.hostTempAllocSize )
            cx.hostBufferTemp = bbvirtallocboundednuma<byte>( cx.hostTempAllocSize );
    #endif

    if( cx.hostBufferTemp == nullptr && cx.hostTempAllocSize )
        CudaErrCheck( cudaMallocHost( &cx.hostBufferTemp, cx.hostTempAllocSize, cudaHostAllocDefault ) );

    CudaErrCheck( cudaMalloc( &cx.deviceBuffer, cx.devAllocSize ) );

    // Warm start
    if( true )// cx.gCfg->warmStart )
    {
        FaultMemoryPages::RunJob( *cx.threadPool, cx.threadPool->ThreadCount(), cx.pinnedBuffer    , cx.pinnedAllocSize    );
        FaultMemoryPages::RunJob( *cx.threadPool, cx.threadPool->ThreadCount(), cx.hostBufferTables, cx.hostTableAllocSize );

        if( cx.hostTempAllocSize )
            FaultMemoryPages::RunJob( *cx.threadPool, cx.threadPool->ThreadCount(), cx.hostBufferTemp, cx.hostTempAllocSize );
    }

    {
        CudaK32AllocContext acx = {};

        acx.alignment = alignment;
        acx.dryRun    = false;

        StackAllocator pinnedAllocator   ( cx.pinnedBuffer    , cx.pinnedAllocSize    );
        StackAllocator hostTableAllocator( cx.hostBufferTables, cx.hostTableAllocSize );
        StackAllocator hostTempAllocator ( cx.hostBufferTemp  , cx.hostTempAllocSize  );
        StackAllocator devAllocator      ( cx.deviceBuffer    , cx.devAllocSize       );

        acx.pinnedAllocator    = &pinnedAllocator;
        acx.hostTableAllocator = &hostTableAllocator;
        acx.hostTempAllocator  = &hostTempAllocator;
        acx.devAllocator       = &devAllocator;
        AllocateP1Buffers( cx, acx );

        pinnedAllocator   .PopToMarker( 0 );
        hostTableAllocator.PopToMarker( 0 );
        hostTempAllocator .PopToMarker( 0 );
        devAllocator      .PopToMarker( 0 );
        CudaK32PlotPhase2AllocateBuffers( cx, acx );

        pinnedAllocator   .PopToMarker( 0 );
        hostTableAllocator.PopToMarker( 0 );
        hostTempAllocator .PopToMarker( 0 );
        devAllocator      .PopToMarker( 0 );
        CudaK32PlotPhase3AllocateBuffers( cx, acx );

        if( allocateParkBuffers )
        {
            // Fine to leak. App-lifetime buffer
            void* parksBuffer = nullptr;
            CudaErrCheck( cudaMallocHost( &parksBuffer, parksPinnedSize, cudaHostAllocDefault ) );
            StackAllocator parkAllocator( parksBuffer, parksPinnedSize );
            AllocateParkSerializationBuffers( cx, parkAllocator, acx.dryRun );
        }
    }
}

//-----------------------------------------------------------
void AllocateP1Buffers( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    const size_t    alignment     = acx.alignment;
    const bool      isCompressed  = cx.gCfg->compressionLevel > 0;
    const TableId   firstTable    = cx.firstStoredTable;

    const FileFlags tmp1FileFlags = cx.cfg.temp1DirectIO ? FileFlags::NoBuffering | FileFlags::LargeFile : FileFlags::LargeFile;
    const FileFlags tmp2FileFlags = cx.cfg.temp2DirectIO ? FileFlags::NoBuffering | FileFlags::LargeFile : FileFlags::LargeFile;

    /// Host allocations
    {
        // Temp allocations are pinned host buffers that can be re-used for other means in different phases.
        // This is roughly equivalent to temp2 dir during disk plotting.


        if( !cx.cfg.hybrid16Mode )
        {
            cx.hostY = acx.hostTempAllocator->CAlloc<uint32>( BBCU_TABLE_ALLOC_ENTRY_COUNT, alignment );
            cx.hostMeta = acx.hostTempAllocator->CAlloc<uint32>( BBCU_TABLE_ALLOC_ENTRY_COUNT * BBCU_HOST_META_MULTIPLIER, alignment );
        }
        else if( !cx.diskContext->metaBuffer )
        {
            const size_t ySliceSize    = sizeof( uint32 ) * BBCU_MAX_SLICE_ENTRY_COUNT;
            const size_t metaSliceSize = sizeof( uint32 ) * BBCU_META_SLICE_ENTRY_COUNT;

            cx.diskContext->yBuffer = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, CudaK32HybridMode::Y_DISK_BUFFER_FILE_NAME.data(), 
                                            BBCU_BUCKET_COUNT, ySliceSize, FileMode::Create, FileAccess::ReadWrite, tmp2FileFlags );
            FatalIf( !cx.diskContext->yBuffer, "Failed to create y disk buffer." );

            cx.diskContext->metaBuffer = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, CudaK32HybridMode::META_DISK_BUFFER_FILE_NAME.data(), 
                                            BBCU_BUCKET_COUNT, metaSliceSize, FileMode::Create, FileAccess::ReadWrite, tmp2FileFlags );
            FatalIf( !cx.diskContext->metaBuffer, "Failed to create metadata disk buffer." );
        }

        // Marking tables used to prune back pointers
        {
            const size_t markingTableBitFieldSize = GetMarkingTableBitFieldSize();

            cx.hostMarkingTables[0] = nullptr;
            cx.hostMarkingTables[1] = isCompressed ? nullptr : acx.hostTableAllocator->AllocT<uint64>( markingTableBitFieldSize, alignment );
            cx.hostMarkingTables[2] = acx.hostTableAllocator->AllocT<uint64>( markingTableBitFieldSize, alignment );
            cx.hostMarkingTables[3] = acx.hostTableAllocator->AllocT<uint64>( markingTableBitFieldSize, alignment );
            cx.hostMarkingTables[4] = acx.hostTableAllocator->AllocT<uint64>( markingTableBitFieldSize, alignment );
            cx.hostMarkingTables[5] = acx.hostTableAllocator->AllocT<uint64>( markingTableBitFieldSize, alignment );
        }

        if( !cx.cfg.hybrid128Mode )
        {
            // NOTE: The first table has their values inlined into the backpointers of the next table
            cx.hostBackPointers[0] = {};

            Pair* firstTablePairs = acx.hostTableAllocator->CAlloc<Pair>( BBCU_TABLE_ALLOC_ENTRY_COUNT, alignment );

            cx.hostBackPointers[(int)firstTable] = { (uint32*)firstTablePairs, nullptr };

            for( TableId table = firstTable + 1; table <= TableId::Table7; table++ )
            {
                cx.hostBackPointers[(int)table] = { 
                    acx.hostTableAllocator->CAlloc<uint32>( BBCU_TABLE_ALLOC_ENTRY_COUNT, alignment ), 
                    acx.hostTableAllocator->CAlloc<uint16>( BBCU_TABLE_ALLOC_ENTRY_COUNT, alignment )
                };
            }

            // These buffers, belonging to table 7, are re-used
            // to store the unsorted back-pointers.
            // For this to work, the reading ot table 7 must be horizontal (see CudaK32PlotIsOutputVertical()).
            // This way, when we store the sorted pairs, we don't
            // overwrite the unsorted data from other buckets.
            cx.hostTableL = cx.hostBackPointers[6].left;
            cx.hostTableR = cx.hostBackPointers[6].right;
        }
        else
        {
            char tableName[] = "table_l_000.tmp";

            size_t multiplier = 2; // First table entries are Pair, not uint32s...

            #if BBCU_DBG_SKIP_PHASE_1
                const FileMode fileMode = FileMode::Open;
            #else
                const FileMode fileMode = FileMode::Create;
            #endif

            for( TableId table = firstTable; table <= TableId::Table7; table++ )
            {
                if( cx.diskContext->tablesL[(int)table] == nullptr )
                {
                    sprintf( tableName, "table_l_%d.tmp", (int32)table+1 );
                    cx.diskContext->tablesL[(int)table] = DiskBuffer::Create(
                        *cx.diskContext->temp1Queue, tableName, BBCU_BUCKET_COUNT, sizeof( uint32 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT * multiplier,
                        fileMode, FileAccess::ReadWrite, tmp1FileFlags );

                    FatalIf( !cx.diskContext->tablesL[(int)table], "Failed to create table %d L disk buffer.", (int)table+1 );
                }

                if( table > firstTable && cx.diskContext->tablesR[(int)table] == nullptr )
                {
                    sprintf( tableName, "table_r_%d.tmp", (int32)table+1 );
                    cx.diskContext->tablesR[(int)table] = DiskBuffer::Create(
                        *cx.diskContext->temp1Queue, tableName, BBCU_BUCKET_COUNT, sizeof( uint16 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT,
                        fileMode, FileAccess::ReadWrite, tmp1FileFlags );

                    FatalIf( !cx.diskContext->tablesR[(int)table], "Failed to create table %d R disk buffer.", (int)table+1 );
                }

                multiplier = 1;
            }

            // When storing unsorted inlined x's, we don't have enough space in RAM, store i disk instead.
            const size_t xSliceSize = BBCU_MAX_SLICE_ENTRY_COUNT * sizeof( Pair );
            cx.diskContext->unsortedL = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, CudaK32HybridMode::LPAIRS_DISK_BUFFER_FILE_NAME.data(), 
                                                                   BBCU_BUCKET_COUNT, xSliceSize, FileMode::OpenOrCreate, FileAccess::ReadWrite, tmp2FileFlags );
            FatalIf( !cx.diskContext->unsortedL, "Failed to create unsorted L disk buffer." );

            if( cx.cfg.hybrid16Mode )
            {
                cx.diskContext->unsortedR = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, "p1unsorted_r.tmp", 
                                                                    BBCU_BUCKET_COUNT, BBCU_MAX_SLICE_ENTRY_COUNT * sizeof( uint16 ), FileMode::OpenOrCreate, FileAccess::ReadWrite, tmp2FileFlags );
                FatalIf( !cx.diskContext->unsortedR, "Failed to create unsorted R disk buffer." );
            }
            else
            {
                // In 128G mode we can store intermediate pairs in the host
                cx.hostTableL = acx.hostTableAllocator->CAlloc<uint32>( BBCU_TABLE_ALLOC_ENTRY_COUNT, alignment );
                cx.hostTableR = acx.hostTableAllocator->CAlloc<uint16>( BBCU_TABLE_ALLOC_ENTRY_COUNT, alignment );
            }
        }
    }

    /// Device & Pinned allocations
    {
        GpuStreamDescriptor yDesc{};
        yDesc.entriesPerSlice = BBCU_MAX_SLICE_ENTRY_COUNT;
        yDesc.sliceCount      = BBCU_BUCKET_COUNT;
        yDesc.sliceAlignment  = alignment;
        yDesc.bufferCount     = BBCU_DEFAULT_GPU_BUFFER_COUNT;
        yDesc.deviceAllocator = acx.devAllocator;
        yDesc.pinnedAllocator = nullptr;             // Start in direct mode (no intermediate pinined buffers)

        // In disk-backed mode, we always have pinned buffers,
        // which are the same buffers used to write and read from disk.
        GpuStreamDescriptor descTablePairs       = yDesc;
        GpuStreamDescriptor descTableSortedPairs = yDesc;
        GpuStreamDescriptor descXPairs           = yDesc;
        GpuStreamDescriptor descMeta             = yDesc;

        if( cx.cfg.hybrid128Mode )
        {
            // Temp 1 Queue
            descTableSortedPairs.pinnedAllocator = acx.pinnedAllocator;
            descTableSortedPairs.sliceAlignment  = cx.diskContext->temp1Queue->BlockSize();

            // Temp 2 Queue
            descXPairs.pinnedAllocator   = acx.pinnedAllocator;
            descXPairs.sliceAlignment    = cx.diskContext->temp2Queue->BlockSize();

            if( cx.cfg.hybrid16Mode )
            {
                yDesc.pinnedAllocator = acx.pinnedAllocator;
                yDesc.sliceAlignment  = cx.diskContext->temp2Queue->BlockSize();

                descMeta.pinnedAllocator = acx.pinnedAllocator;
                descMeta.sliceAlignment  = cx.diskContext->temp2Queue->BlockSize();

                descTablePairs.pinnedAllocator = acx.pinnedAllocator;
                descTablePairs.sliceAlignment  = cx.diskContext->temp2Queue->BlockSize();
            }
        }

        if( !cx.downloadDirect )
        {
            // Use intermediate pinned buffer for transfers to non-pinned destinations
            yDesc.pinnedAllocator                = acx.pinnedAllocator;
            descTablePairs.pinnedAllocator       = acx.pinnedAllocator;
            descTableSortedPairs.pinnedAllocator = acx.pinnedAllocator;
            descXPairs.pinnedAllocator           = acx.pinnedAllocator;
            descMeta.pinnedAllocator             = acx.pinnedAllocator;
        }


        ///
        /// Downloads
        ///
        cx.yOut    = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint32>( yDesc, acx.dryRun );
        cx.metaOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<K32Meta4>( descMeta, acx.dryRun );

        {
            // These download buffers share the same backing buffers
            const size_t devMarker    = acx.devAllocator->Size();
            const size_t pinnedMarker = acx.pinnedAllocator->Size();

            cx.pairsLOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint32>( descTablePairs, acx.dryRun );
            cx.pairsROut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint16>( descTablePairs, acx.dryRun );

            acx.devAllocator->PopToMarker( devMarker );
            acx.pinnedAllocator->PopToMarker( pinnedMarker );

            // Allocate Pair at the end, to ensure we grab the highest value
            cx.xPairsOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<Pair>( descXPairs, acx.dryRun );
        }

        {
            // These download buffers share the same backing buffers
            const size_t devMarker    = acx.devAllocator->Size();
            const size_t pinnedMarker = acx.pinnedAllocator->Size();

            cx.sortedPairsLOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint32>( descTableSortedPairs, acx.dryRun );
            cx.sortedPairsROut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint16>( descTableSortedPairs, acx.dryRun );

            acx.devAllocator->PopToMarker( devMarker );
            acx.pinnedAllocator->PopToMarker( pinnedMarker );

            // Allocate Pair at the end, to ensure we grab the highest value
            cx.sortedXPairsOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<Pair>( descXPairs, acx.dryRun );
        }

        ///
        /// Uploads
        ///
        cx.yIn    = cx.gpuUploadStream[0]->CreateUploadBufferT<uint32>( yDesc, acx.dryRun );
        cx.metaIn = cx.gpuUploadStream[0]->CreateUploadBufferT<K32Meta4>( descMeta, acx.dryRun );

        // These uploaded buffers share the same backing buffers
        {
            const size_t devMarker    = acx.devAllocator->Size();
            const size_t pinnedMarker = acx.pinnedAllocator->Size();

            cx.pairsLIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint32>( descTablePairs, acx.dryRun );
            cx.pairsRIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint16>( descTablePairs, acx.dryRun );

            acx.devAllocator->PopToMarker( devMarker );
            acx.pinnedAllocator->PopToMarker( pinnedMarker );

            // Allocate Pair at the end, to ensure we grab the highest value
            cx.xPairsIn = cx.gpuUploadStream[0]->CreateUploadBufferT<Pair>( descXPairs, acx.dryRun );
        }

        /// Device-only allocations
        if( acx.dryRun )
        {
            cx.devSortTmpAllocSize = 0;
            cub::DeviceRadixSort::SortPairs<uint32, uint32>( nullptr, cx.devSortTmpAllocSize, nullptr, nullptr, nullptr, nullptr, BBCU_BUCKET_ALLOC_ENTRY_COUNT );
        }

        cx.devSortTmp         = acx.devAllocator->AllocT<byte>( cx.devSortTmpAllocSize, alignment );

        cx.devYWork           = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );
        cx.devMetaWork        = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT * BBCU_HOST_META_MULTIPLIER, alignment );
        cx.devXInlineInput    = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );   // #TODO: Maybe we can avoid this allocation?
        cx.devMatches         = acx.devAllocator->CAlloc<Pair>  ( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );
        cx.devInlinedXs       = acx.devAllocator->CAlloc<Pair>  ( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );

        cx.devSortKey         = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );
        cx.devChaChaInput     = (uint32*)acx.devAllocator->AllocT<byte>( 64, alignment );
        cx.devGroupBoundaries = acx.devAllocator->CAlloc<uint32>( CU_MAX_BC_GROUP_BOUNDARIES, alignment );
        cx.devMatchCount      = acx.devAllocator->CAlloc<uint32>( 1 );
        cx.devGroupCount      = acx.devAllocator->CAlloc<uint32>( 1 );
        cx.devBucketCounts    = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_COUNT, alignment );
        cx.devSliceCounts     = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, alignment );


        /// Pinned-only allocations
        cx.hostMatchCount   = acx.pinnedAllocator->CAlloc<uint32>( 1, alignment );
        cx.hostBucketCounts = acx.pinnedAllocator->CAlloc<uint32>( BBCU_BUCKET_COUNT, alignment );
        cx.hostBucketSlices = acx.pinnedAllocator->CAlloc<uint32>( BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, alignment );
    }

    /// In disk-backed mode, assign disk buffers to gpu buffers
    if( cx.cfg.hybrid128Mode && !acx.dryRun )
    {
        cx.xPairsOut.AssignDiskBuffer( cx.diskContext->unsortedL );
        cx.xPairsIn .AssignDiskBuffer( cx.diskContext->unsortedL );

        if( cx.cfg.hybrid16Mode )
        {
            cx.pairsLOut.AssignDiskBuffer( cx.diskContext->unsortedL );
            cx.pairsLIn .AssignDiskBuffer( cx.diskContext->unsortedL );

            cx.pairsROut.AssignDiskBuffer( cx.diskContext->unsortedR );
            cx.pairsRIn .AssignDiskBuffer( cx.diskContext->unsortedR );

            cx.yOut.AssignDiskBuffer( cx.diskContext->yBuffer );
            cx.yIn .AssignDiskBuffer( cx.diskContext->yBuffer );

            cx.metaOut.AssignDiskBuffer( cx.diskContext->metaBuffer );
            cx.metaIn .AssignDiskBuffer( cx.diskContext->metaBuffer );
        }
    }
}

//-----------------------------------------------------------
void AllocateParkSerializationBuffers( CudaK32PlotContext& cx, IAllocator& pinnedAllocator, bool dryRun )
{
    ASSERT( cx.parkContext );

    auto& pc = *cx.parkContext;
    pc.maxParkBuffers = 3;

    // Get the largest park size
    const size_t maxParkSize = cx.cfg.gCfg->compressionLevel == 0 ?
                                CalculateParkSize( TableId::Table1 ) :
                                GetLargestCompressedParkSize();

    const size_t parksPerBuffer       = CDivT<size_t>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, kEntriesPerPark ) + 2;
    // CDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, kCheckpoint1Interval ) + 1; // Need an extra park for left-over entries
    const size_t bucketParkBufferSize = parksPerBuffer * maxParkSize;
    const size_t alignment            = 4096;

    // Allocate some extra space for C tables (see FinalizeTable7)
    pc.hostRetainedLinePoints = pinnedAllocator.CAlloc<uint64>( kEntriesPerPark );
    pc.table7Memory           = pinnedAllocator.CAllocSpan<byte>( 8 MiB, alignment );
    pc.parkBufferChain        = BufferChain::Create( pinnedAllocator, pc.maxParkBuffers, bucketParkBufferSize, alignment, dryRun );
}



///
/// Debug
///
#if _DEBUG

void DbgWritePairs( CudaK32PlotContext& cx, const TableId table )
{
    if( cx.cfg.hybrid128Mode )
        return;

    const TableId earliestTable = TableId::Table1 + (TableId)cx.gCfg->numDroppedTables+1;
    if( table < earliestTable )
        return;

    char lPath[512];
    char rPath[512];

    Log::Line( "[DEBUG] Writing pairs to disk..." );
    {
        sprintf( lPath, "%st%d.l.tmp", DBG_BBCU_DBG_DIR, (int)table+1 );
        sprintf( rPath, "%st%d.r.tmp", DBG_BBCU_DBG_DIR, (int)table+1 );

        const uint64 entryCount = cx.tableEntryCounts[(int)table];
        const Pairs  pairs      = cx.hostBackPointers[(int)table];

        int err;

        if( table == earliestTable )
        {
            FatalIf( !IOJob::WriteToFile( lPath, pairs.left, sizeof( Pair ) * entryCount, err ),
                "Failed to write table pairs: %d", err );
        }
        else
        {
            FatalIf( !IOJob::WriteToFile( lPath, pairs.left, sizeof( uint32 ) * entryCount, err ),
                "Failed to write table L pairs: %d", err );

            // if( (uint32)table > cx.gCfg->numDroppedTables )
                FatalIf( !IOJob::WriteToFile( rPath, pairs.right, sizeof( uint16 ) * entryCount, err),
                    "Failed to write table R pairs: %d", err );
        }
    }

    // if( cx.table == TableId::Table7 )
    // {
    //     // Now write our context data
    //     Log::Line( "[DEBUG] Writing context file." );
    //     FileStream contxetFile;
    //     sprintf( lPath, "%scontext.tmp", DBG_BBCU_DBG_DIR );
    //     FatalIf( !contxetFile.Open( lPath, FileMode::Create, FileAccess::Write ), "Failed to open context file." );
    //     FatalIf( contxetFile.Write( &cx, sizeof( CudaK32PlotContext ) ) != (ssize_t)sizeof( CudaK32PlotContext ), "Failed to write context data." );
    //     contxetFile.Close();
    // }
    Log::Line( "[DEBUG] Done." );
}

void DbgWriteContext( CudaK32PlotContext& cx )
{
    char path[512];
    
    // Now write our context data
    Log::Line( "[DEBUG] Writing context file." );
    FileStream contxetFile;
    sprintf( path, "%scontext.tmp", DBG_BBCU_DBG_DIR );
    FatalIf( !contxetFile.Open( path, FileMode::Create, FileAccess::Write ), "Failed to open context file at '%s'.", path );
    FatalIf( contxetFile.Write( &cx, sizeof( CudaK32PlotContext ) ) != (ssize_t)sizeof( CudaK32PlotContext ), "Failed to write context data." );
    
    contxetFile.Close();
    
    Log::Line( "[DEBUG] Done." );
}

void DbgLoadContextAndPairs( CudaK32PlotContext& cx, bool loadTables )
{
    char lPath[512];
    char rPath[512];

    // Log::Line( "[DEBUG] Loading table pairs..." );
    {
        Log::Line( "[DEBUG] Reading context" );
        CudaK32PlotContext tmpCx = {};

        FileStream contxetFile;
        sprintf( lPath, "%scontext.tmp", DBG_BBCU_DBG_DIR );
        FatalIf( !contxetFile.Open( lPath, FileMode::Open, FileAccess::Read ), "Failed to open context file." );
        FatalIf( contxetFile.Read( &tmpCx, sizeof( CudaK32PlotContext ) ) != (ssize_t)sizeof( CudaK32PlotContext ), "Failed to read context data." );
        contxetFile.Close();

        memcpy( cx.bucketCounts, tmpCx.bucketCounts, sizeof( tmpCx.bucketCounts ) );
        memcpy( cx.bucketSlices, tmpCx.bucketSlices, sizeof( tmpCx.bucketSlices ) );
        memcpy( cx.tableEntryCounts, tmpCx.tableEntryCounts, sizeof( tmpCx.tableEntryCounts ) );        
    }

    if( !loadTables )
        return;

    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
    {
        Log::Line( "[DEBUG] Loading table %d", (int)table+1 );

        sprintf( lPath, "%st%d.l.tmp", DBG_BBCU_DBG_DIR, (int)table+1 );
        sprintf( rPath, "%st%d.r.tmp", DBG_BBCU_DBG_DIR, (int)table+1 );

        const uint64  entryCount = cx.tableEntryCounts[(int)table];
              Pairs&  pairs      = cx.hostBackPointers[(int)table];


        int err;
        pairs.left = (uint32*)IOJob::ReadAllBytesDirect( lPath, err );
        FatalIf( pairs.left == nullptr, "Failed to read table L pairs: %d", err );

        pairs.right = (uint16*)IOJob::ReadAllBytesDirect( rPath, err );
        FatalIf( pairs.right == nullptr, "Failed to read table R pairs: %d", err );
    }
}

void DbgLoadTablePairs( CudaK32PlotContext& cx, const TableId table, bool useDiskHybridData )
{
    if( cx.cfg.hybrid128Mode )
        return;

    char lPath[512];
    char rPath[512];

    const TableId earliestTable = TableId::Table1 + (TableId)cx.gCfg->numDroppedTables+1;
    if( table < earliestTable )
        return;

    const uint64 entryCount = cx.tableEntryCounts[(int)table];
    Pairs& pairs = cx.hostBackPointers[(int)table];

    {
        Log::Line( "[DEBUG] Loading table %d", (int)table + 1 );

        sprintf( lPath, "%st%d.l.tmp", DBG_BBCU_DBG_DIR, (int)table + 1 );
        sprintf( rPath, "%st%d.r.tmp", DBG_BBCU_DBG_DIR, (int)table + 1 );

        // cx.hostBackPointers[(int)table].left  = bbcvirtallocbounded<uint32>( entryCount );
        // cx.hostBackPointers[(int)table].right = bbcvirtallocbounded<uint16>( entryCount );

        int err;

        static DiskQueue* diskQueue = nullptr;

        // Load disk-hybrid tables
        // #NOTE: Enable (and disable the block below this one), to load tables from 
        //        the disk-hybrid output. Also adjust path in the DiskQueue below.

        // useDiskHybridData = true;
        if( useDiskHybridData )
        {
            if( diskQueue == nullptr )
                diskQueue = new DiskQueue( "/home/harold/plotdisk" );

            char lname[64] = {};
            sprintf( lname, "table_l_%d.tmp", (int)table + 1 );

            if( table == earliestTable )
            {
                DiskBuffer* buf = DiskBuffer::Create( *diskQueue, lname, BBCU_BUCKET_COUNT, sizeof( Pair ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT,
                    FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering );
                PanicIf( !buf, "No table file" );

                VirtualAllocator valloc;
                buf->ReserveBuffers( valloc );

                Span<Pair> pairsWriter( (Pair*)pairs.left, BBCU_TABLE_ALLOC_ENTRY_COUNT );
                buf->ReadNextBucket();

                for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
                {
                    const size_t bucketLength = cx.bucketCounts[(int)table][bucket];

                    buf->TryReadNextBucket();
                    auto entries = buf->GetNextReadBufferAs<Pair>().SliceSize( bucketLength );

                    entries.CopyTo( pairsWriter );
                    pairsWriter = pairsWriter.Slice( entries.Length() );
                }

                delete buf;
            }
            else
            {
                char rname[64] = {};
                sprintf( rname, "table_r_%d.tmp", (int)table + 1 );

                DiskBuffer* lBuf = DiskBuffer::Create( *diskQueue, lname, BBCU_BUCKET_COUNT, sizeof( uint32 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT,
                    FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering );
                DiskBuffer* rBuf = DiskBuffer::Create( *diskQueue, rname, BBCU_BUCKET_COUNT, sizeof( uint16 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT,
                    FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering );
                PanicIf( !lBuf, "No table L file" );
                PanicIf( !rBuf, "No table R file" );

                VirtualAllocator valloc;
                lBuf->ReserveBuffers( valloc );
                rBuf->ReserveBuffers( valloc );

                Span<uint32> lWriter( pairs.left , BBCU_TABLE_ALLOC_ENTRY_COUNT );
                Span<uint16> rWriter( pairs.right, BBCU_TABLE_ALLOC_ENTRY_COUNT );

                lBuf->ReadNextBucket();
                rBuf->ReadNextBucket();

                for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
                {
                    const size_t bucketLength = cx.bucketCounts[(int)table][bucket];

                    lBuf->TryReadNextBucket();
                    rBuf->TryReadNextBucket();

                    auto lEntries = lBuf->GetNextReadBufferAs<uint32>().SliceSize( bucketLength );
                    lEntries.CopyTo( lWriter );

                    auto rEntries = rBuf->GetNextReadBufferAs<uint16>().SliceSize( bucketLength );
                    rEntries.CopyTo( rWriter );

                    lWriter = lWriter.Slice( lEntries.Length() );
                    rWriter = rWriter.Slice( rEntries.Length() );
                }

                delete lBuf;
                delete rBuf;
            }
        }
        else
        {
            if( table == earliestTable )
            {
                FatalIf( !IOJob::ReadFromFile( lPath, pairs.left, entryCount * sizeof( Pair ), err ), "Failed to read table X pairs: %d", err );
            }
            else
            {
                FatalIf( !IOJob::ReadFromFile( lPath, pairs.left , entryCount * sizeof( uint32 ), err ), "Failed to read table L pairs: %d", err );
                
                // if( (uint32)table > cx.gCfg->numDroppedTables )
                    FatalIf( !IOJob::ReadFromFile( rPath, pairs.right, entryCount * sizeof( uint16 ), err ), "Failed to read table R pairs: %d", err );
            }
        }
    }


    // if( table == earliestTable && !useDiskHybridData )
    // {
    //     uint64* tmpBucket = bbcvirtallocboundednuma<uint64>( BBCU_BUCKET_ALLOC_ENTRY_COUNT );

    //     std::vector<std::string> hashesRam{};
    //     std::vector<std::string> hashesDisk{};

    //     byte hash[32];
    //     char hashstr[sizeof(hash)*2+1] = {};
    
    //     for( uint32 run = 0; run < 2; run++ )
    //     {
    //         auto& hashes = run == 0 ? hashesRam : hashesDisk;

    //         uint64* xs = (uint64*)pairs.left;
    
    //         for( uint32 b = 0; b < BBCU_BUCKET_COUNT; b++ )
    //         {
    //             const uint64 bucketEntryCount = cx.bucketCounts[(int)table][b];

    //             RadixSort256::Sort<BB_MAX_JOBS>( DbgGetThreadPool( cx ), xs, tmpBucket, bucketEntryCount );

    //             // Hash
    //             {
    //                 blake3_hasher hasher;
    //                 blake3_hasher_init( &hasher );
    //                 blake3_hasher_update( &hasher, xs, bucketEntryCount * sizeof( uint64 ) );
    //                 blake3_hasher_finalize( &hasher, hash, sizeof( hash ) );

    //                 size_t _;
    //                 BytesToHexStr( hash, sizeof( hash ), hashstr, sizeof( hashstr ), _ );
    //                 Log::Line( "[%3u] : 0x%s", b, hashstr );

    //                 hashes.push_back( hashstr );
        
    //                 // DbgPrintHash( " :", xs, sizeof( uint64 ) * bucketEntryCount );
    //             }
            
    //             xs += bucketEntryCount;
    //         }

    //         if( run == 0 )
    //         {
    //             DbgLoadTablePairs( cx, table, true );
    //         }
    //     }

    //     // Compare hashes
    //     {
    //         for( uint32 b = 0; b < BBCU_BUCKET_COUNT; b++ )
    //         {
    //             if( hashesRam[b] != hashesDisk[b] )
    //             {
    //                 Panic( "Hash mismatch at bucket %u. %s != %s", b, hashesRam[b].c_str(), hashesDisk[b].c_str() );
    //             }
    //         }
    //         Log::Line( "All hashes match!" );
    //     }
        

    //     // DbgPrintHash( "Inlined X Table", cx.hostBackPointers[(int)table].left, sizeof( Pair ) * cx.tableEntryCounts[(int)table] );
    //     Log::Line( "" );
    //     bbvirtfreebounded( tmpBucket );
    //     Exit( 0 );
    // }
    // else
    // {
    //     // DbgPrintHash( "L Table", cx.hostBackPointers[(int)table].left, sizeof( uint32 ) * cx.tableEntryCounts[(int)table] );
    //     // DbgPrintHash( "R Table", cx.hostBackPointers[(int)table].right, sizeof( uint16 ) * cx.tableEntryCounts[(int)table] );
    //     // Log::Line( "" );
    // }

    // Sort inlined xs
    // if( table == earliestTable )
    // {
    //     uint64* tmpBucket = bbcvirtallocboundednuma<uint64>( BBCU_BUCKET_ALLOC_ENTRY_COUNT );
    //     uint64* xs = (uint64*)pairs.left;

    //     for( uint32 b = 0; b < BBCU_BUCKET_COUNT; b++ )
    //     {
    //         const uint64 bucketEntryCount = cx.bucketCounts[(int)table][b];
    //         RadixSort256::Sort<BB_MAX_JOBS>( DbgGetThreadPool( cx ), xs, tmpBucket, bucketEntryCount );
    //         xs += bucketEntryCount;
    //     }

    //     DbgPrintHash( "pre_sorted_xs", pairs.left, sizeof( uint64 ) * entryCount );
    // }

    Log::Line( "[DEBUG] Done." );
}

void DbgLoadMarks( CudaK32PlotContext& cx )
{
    char path[512];

    std::string baseUrl = DBG_BBCU_DBG_DIR;
    if( cx.cfg.hybrid128Mode )
        baseUrl += "disk/";

    // const size_t tableSize = ((1ull << BBCU_K) / 64) * sizeof(uint64);
    Log::Line( "[DEBUG] Loadinging marking tables" );

    const TableId startTable = cx.firstStoredTable;

    for( TableId table = startTable; table < TableId::Table7; table++ )
    {
        sprintf( path, "%smarks%d.tmp", baseUrl.c_str(), (int)table+1 );

        int err = 0;
        cx.hostMarkingTables[(int)table] = (uint64*)IOJob::ReadAllBytesDirect( path, err );
    }

    Log::Line( "[DEBUG] Done." );
}

void DbgPruneTable( CudaK32PlotContext& cx, const TableId rTable )
{
    const size_t MarkingTableSize = 1ull << 32;
    byte* bytefield = bbvirtalloc<byte>( MarkingTableSize );
    memset( bytefield, 0, MarkingTableSize );
    
    std::atomic<uint64> totalPrunedEntryCount = 0;

    ThreadPool& pool = DbgGetThreadPool( cx );
    AnonMTJob::Run( pool, [&]( AnonMTJob* self ){

        const uint64 rEntryCount = cx.tableEntryCounts[(int)rTable];
        {
            uint64 count, offset, end;
            GetThreadOffsets( self, rEntryCount, count, offset, end );

            auto marks = bytefield;
            Pairs rTablePairs = cx.hostBackPointers[(int)rTable];

            for( uint64 i = offset; i < end; i++ )
            {
                const uint32 l = rTablePairs.left[i];
                const uint32 r = l + rTablePairs.right[i];
                
                marks[l] = 1;
                marks[r] = 1;
            }

            self->SyncThreads();

                  uint64 localPrunedEntryCount = 0;
            const uint64 lEntryCount           = cx.tableEntryCounts[(int)rTable-1];

            GetThreadOffsets( self, lEntryCount, count, offset, end );
            for( uint64 i = offset; i < end; i++ )
            {
                if( marks[i] == 1 )
                    localPrunedEntryCount++;
            }

            totalPrunedEntryCount += localPrunedEntryCount;
        }
    });

    const uint64 prunedEntryCount = totalPrunedEntryCount.load();
    const uint64 lEntryCount      = cx.tableEntryCounts[(int)rTable-1];
    Log::Line( "Table %u pruned entry count: %llu / %llu ( %.2lf %% )", rTable, 
        prunedEntryCount, lEntryCount, prunedEntryCount / (double)lEntryCount * 100.0 );

    bbvirtfree( bytefield );
}

void DbgPruneTableBuckets( CudaK32PlotContext& cx, const TableId rTable )
{
    const size_t MarkingTableSize = 1ull << 32;
    byte* bytefield = bbvirtalloc<byte>( MarkingTableSize );
    memset( bytefield, 0, MarkingTableSize );
    
    std::atomic<uint64> totalPrunedEntryCount = 0;

    AnonMTJob::Run( *_dbgThreadPool, [&]( AnonMTJob* self ){

        auto  marks = bytefield;

        Pairs rTablePairs = cx.hostBackPointers[6];

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            const uint64 rEntryCount = cx.bucketCounts[(int)rTable][bucket];

            uint64 count, offset, end;
            GetThreadOffsets( self, rEntryCount, count, offset, end );

            for( uint64 i = offset; i < end; i++ )
            {
                const uint32 l = rTablePairs.left[i];
                const uint32 r = l + rTablePairs.right[i];
                
                marks[l] = 1;
                marks[r] = 1;
            }

            rTablePairs.left  += BBCU_BUCKET_ALLOC_ENTRY_COUNT;
            rTablePairs.right += BBCU_BUCKET_ALLOC_ENTRY_COUNT;
        }

        self->SyncThreads();

        {
                  uint64 localPrunedEntryCount = 0;
            const uint64 lEntryCount           = cx.tableEntryCounts[(int)rTable-1];

            uint64 count, offset, end;
            GetThreadOffsets( self, lEntryCount, count, offset, end );

            for( uint64 i = offset; i < end; i++ )
            {
                if( marks[i] == 1 )
                    localPrunedEntryCount++;
            }

            totalPrunedEntryCount += localPrunedEntryCount;
        }
    });

    const uint64 prunedEntryCount = totalPrunedEntryCount.load();
    const uint64 lEntryCount      = cx.tableEntryCounts[(int)rTable-1];
    Log::Line( "Table %u pruned entry count: %llu / %llu ( %.2lf %% )", rTable, 
        prunedEntryCount, lEntryCount, prunedEntryCount / (double)lEntryCount * 100.0 );

    bbvirtfree( bytefield );
}

#endif // _DEBUG


