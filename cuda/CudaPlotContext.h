#pragma once

#include "CudaPlotConfig.h"
#include "CudaUtil.h"
#include "ChiaConsts.h"
#include "CudaPlotter.h"
#include "plotting/PlotTypes.h"
#include "plotting/PlotWriter.h"
#include "GpuStreams.h"
#include "GpuQueue.h"
#include "util/StackAllocator.h"
#include "fse/fse.h"
#include "threading/Fence.h"
#include "plotting/GlobalPlotConfig.h"
#include "threading/ThreadPool.h"
#include "plotting/BufferChain.h"
#include "plotting/DiskBuffer.h"
#include "plotting/DiskBucketBuffer.h"
#include <filesystem>

#include "cub/device/device_radix_sort.cuh"
// #include <cub/device/device_radix_sort.cuh>

// Fix for cooperative_groups.h on windows
#ifdef __LITTLE_ENDIAN__
    #undef __LITTLE_ENDIAN__
    #define __LITTLE_ENDIAN__ 1 
#endif
#include <cooperative_groups.h>
using namespace cooperative_groups;

#if _DEBUG
    #include <assert.h>
#endif


struct CudaK32ParkContext
{
    Span<byte>        table7Memory;             // Memory buffer reserved for finalizing table7 and writing C parks
    BufferChain*      parkBufferChain;
    uint32            maxParkBuffers;           // Maximum number of park buffers
    uint64*           hostRetainedLinePoints;
};

struct CudaK32HybridMode
{
    // For clarity, these are the file names for the disk buffers
    // whose disk space will be shared for temp data in both phase 1 and phase 3.
    // The name indicates their usage and in which phase.
    static constexpr std::string_view Y_DISK_BUFFER_FILE_NAME      = "p1y-p3index.tmp";
    static constexpr std::string_view META_DISK_BUFFER_FILE_NAME   = "p1meta-p3rmap.tmp";
    static constexpr std::string_view LPAIRS_DISK_BUFFER_FILE_NAME = "p1unsortedx-p1lpairs-p3lp-p3-lmap.tmp";

    static constexpr std::string_view P3_RMAP_DISK_BUFFER_FILE_NAME        = META_DISK_BUFFER_FILE_NAME;
    static constexpr std::string_view P3_INDEX_DISK_BUFFER_FILE_NAME       = Y_DISK_BUFFER_FILE_NAME;
    static constexpr std::string_view P3_LP_AND_LMAP_DISK_BUFFER_FILE_NAME = LPAIRS_DISK_BUFFER_FILE_NAME;

    DiskQueue*  temp1Queue;  // Tables Queue
    DiskQueue*  temp2Queue;  // Metadata Queue (could be the same as temp1Queue)

    DiskBucketBuffer* metaBuffer;   // Enabled in < 128G mode
    DiskBucketBuffer* yBuffer;      // Enabled in < 128G mode
    DiskBucketBuffer* unsortedL;    // Unsorted Xs (or L pairs in < 128G) are written to disk (uint64 entries)
    DiskBucketBuffer* unsortedR;    // Unsorted R pairs in < 128G mode

    DiskBuffer*       tablesL[7];
    DiskBuffer*       tablesR[7];

    GpuDownloadBuffer _tablesL[7];
    GpuDownloadBuffer _tablesR[7];

    struct
    {
        // #NOTE: These buffers shared the same file-backed storage as
        //        with other buffers in phase 1.
        DiskBucketBuffer* rMapBuffer;           // Step 1
        DiskBucketBuffer* indexBuffer;          // X-step/Step 2
        DiskBucketBuffer* lpAndLMapBuffer;      // X-step/Step 2 (LP) | Step 3 (LMap)

    } phase3;
};

struct CudaK32Phase2
{
    GpuUploadBuffer   pairsLIn;
    GpuUploadBuffer   pairsRIn;
    GpuDownloadBuffer outMarks;

    uint64            pairsLoadOffset;
    byte*             devMarkingTable;          // bytefield marking table
    const uint64*     devRMarks[6];             // Right table's marks as a bitfield
    uint32*           devPrunedCount;

    StackAllocator*   hostBitFieldAllocator;    // Pinned bitfield buffers

    TableId           endTable;
};

struct CudaK32Phase3
{
    struct LMap
    {
        uint32 sourceIndex; // Initial unsorted (or y-sorted) index
        uint32 sortedIndex; // Final LinePoint-sorted index
    };
    static_assert( sizeof( LMap ) == sizeof( uint64 ) );

    struct RMap
    {
        uint32 src;
        uint32 dstL;
        uint32 dstR;
    };

    uint64  pairsLoadOffset;

    // Device buffers
    uint32* devBucketCounts;
    uint32* devPrunedEntryCount;

    // Host buffers
    union {
        RMap*   hostRMap;
        uint32* hostIndices;
    };

    union {
        LMap*   hostLMap;
        uint64* hostLinePoints;
    };

    uint32 prunedBucketCounts[7][BBCU_BUCKET_COUNT];
    uint64 prunedTableEntryCounts[7];


    // Inlined x table
    struct {
        const uint64*     devRMarks;    // R-Marking table
        GpuUploadBuffer   xIn;          // 64-bit Pair
        GpuDownloadBuffer lpOut;        // Output line points (uint64)
        GpuDownloadBuffer indexOut;     // Output source line point index (uint32) (taken from the rMap source value)

    } xTable;

    // Step 1
    struct {
        uint64*           rTableMarks;
        GpuUploadBuffer   pairsLIn;
        GpuUploadBuffer   pairsRIn;
        GpuDownloadBuffer rMapOut;

        uint32 prunedBucketSlices[BBCU_BUCKET_COUNT][BBCU_BUCKET_COUNT];
    } step1;

    // Step 2
    struct {
        GpuUploadBuffer   rMapIn;       // RMap from step 1
        GpuUploadBuffer   lMapIn;       // Output map (uint64) from the previous table run. Or, when L table is the first stored table, it is inlined x values
        GpuDownloadBuffer lpOut;        // Output line points (uint64)
        GpuDownloadBuffer indexOut;     // Output source line point index (uint32) (taken from the rMap source value)
        GpuDownloadBuffer parksOut;     // Output P7 parks on the last table
        uint32*           devLTable[2]; // Unpacked L table bucket

        uint32 prunedBucketSlices[BBCU_BUCKET_COUNT][BBCU_BUCKET_COUNT];
    } step2;

    // Step 3
    struct {
        GpuUploadBuffer   lpIn;         // Line points from step 2
        GpuUploadBuffer   indexIn;      // Indices from step 2
        GpuDownloadBuffer mapOut;       // lTable for next step 2
        GpuDownloadBuffer parksOut;     // Downloads park buffers to host

        uint32*           hostParkOverrunCount;

        size_t            sizeTmpSort;
        byte*             devSortTmpData;

        uint64*           devLinePoints;
        uint64*           devDeltaLinePoints;
        uint32*           devIndices;
        FSE_CTable*       devCTable;
        uint32*           devParkOverrunCount;

        std::atomic<uint32> parkBucket;

        uint32 prunedBucketSlices[BBCU_BUCKET_COUNT][BBCU_BUCKET_COUNT];

    } step3;
};

struct CudaK32AllocContext
{
    size_t alignment;
    bool   dryRun;

    IStackAllocator* pinnedAllocator;
    IStackAllocator* devAllocator;
    IStackAllocator* hostTableAllocator;
    IStackAllocator* hostTempAllocator;
};

// struct CudaK32PlotRequest
// {
//     const char* plotOutDir;
//     const char* plotFileName;

//     const byte* plotId;
//     const char* plotIdStr;

//     const byte* plotMemo;
//     uint16      plotMemoSize;

//     uint32      plotCount;
// };

struct CudaK32PlotContext
{
          CudaK32PlotConfig cfg       = {};
    const GlobalPlotConfig* gCfg      = nullptr;

    int32           cudaDevice        = -1;
    cudaDeviceProp* cudaDevProps      = nullptr;
    bool            downloadDirect    = false;
    TableId         firstStoredTable  = TableId::Table2;    // First non-dropped table that has back pointers
    ThreadPool*     threadPool        = nullptr;

    TableId      table                = TableId::Table1;    // Current table being generated
    uint32       bucket               = 0;                  // Current bucket being processed

    uint64       prevTablePairOffset  = 0;                  // Offset at which to write the previous table's sorted pairs

    uint32       bucketCounts[7][BBCU_BUCKET_COUNT]  = {};
    uint32       bucketSlices[2][BBCU_BUCKET_COUNT][BBCU_BUCKET_COUNT] = {};
    uint64       tableEntryCounts[7]  = {};

    PlotRequest  plotRequest;
    PlotWriter*  plotWriter           = nullptr;
    Fence*       plotFence            = nullptr;
    Fence*       parkFence            = nullptr;

    // Root allocations
    size_t allocAlignment             = 0;
    size_t pinnedAllocSize            = 0;
    size_t devAllocSize               = 0;
    size_t hostTableAllocSize         = 0;
    size_t hostTempAllocSize          = 0;

    void* pinnedBuffer                = nullptr;
    void* deviceBuffer                = nullptr;
    void* hostBufferTemp              = nullptr;
    void* hostBufferTables            = nullptr;

    // Device stuff
    cudaStream_t computeStream        = nullptr;
    cudaStream_t computeStreamB       = nullptr;
    cudaStream_t computeStreamC       = nullptr;
    cudaStream_t computeStreamD       = nullptr;
    cudaEvent_t  computeEventA        = nullptr;
    cudaEvent_t  computeEventB        = nullptr;
    cudaEvent_t  computeEventC        = nullptr;
    GpuQueue*    gpuDownloadStream[BBCU_GPU_STREAM_COUNT] = {};
    GpuQueue*    gpuUploadStream  [BBCU_GPU_STREAM_COUNT] = {};

    GpuDownloadBuffer yOut;
    GpuDownloadBuffer metaOut;
    GpuUploadBuffer   yIn;
    GpuUploadBuffer   metaIn;


    GpuDownloadBuffer xPairsOut;        // This shares the same backing buffer with pairsLOut & pairsROut
    GpuDownloadBuffer pairsLOut;
    GpuDownloadBuffer pairsROut;
    GpuUploadBuffer   xPairsIn;         // This shares the same backing buffer with pairsLIn & pairsRIn
    GpuUploadBuffer   pairsLIn;
    GpuUploadBuffer   pairsRIn;
    GpuDownloadBuffer sortedXPairsOut;  // This shares the same backing buffer with sortedPairsLOut & sortedPairsROut
    GpuDownloadBuffer sortedPairsLOut;
    GpuDownloadBuffer sortedPairsROut;

    
    size_t       devSortTmpAllocSize  = 0;
    void*        devSortTmp           = nullptr;
    uint32*      devYWork             = nullptr;
    uint32*      devMetaWork          = nullptr;
    uint32*      devXInlineInput      = nullptr;
    Pair*        devMatches           = nullptr;
    union {
        Pair*    devInlinedXs         = nullptr;
        uint32*  devCompressedXs;
    };
    uint32*      devBucketCounts      = nullptr;
    uint32*      devSliceCounts       = nullptr;
    uint32*      devSortKey           = nullptr;
    uint32*      devChaChaInput       = nullptr;
    
    uint32*      devGroupBoundaries   = nullptr;

    uint32*      devMatchCount        = nullptr;
    uint32*      devGroupCount        = nullptr;


    /// Host stuff

    // Host "Temp 2"
    uint32*      hostY                = nullptr;
    uint32*      hostMeta             = nullptr;
    uint32*      hostBucketCounts     = nullptr;
    uint32*      hostBucketSlices     = nullptr;
    uint32*      hostTableL           = nullptr;
    uint16*      hostTableR           = nullptr;

    union {
        uint32*  hostMatchCount       = nullptr;
        uint32*  hostGroupCount;
    };

    // Host "Temp 1"
    Pairs        hostBackPointers [7] = {};
    uint64*      hostMarkingTables[6] = {};


    CudaK32Phase2* phase2 = nullptr;
    CudaK32Phase3* phase3 = nullptr;

    CudaK32HybridMode*  diskContext    = nullptr;
    CudaK32ParkContext* parkContext    = nullptr;
    bool                useParkContext = false;

    // Used when '--check' is enabled
    struct GreenReaperContext* grCheckContext = nullptr;
    class  PlotChecker*        plotChecker    = nullptr;

    struct
    {
        Duration uploadTime   = Duration::zero();   // Host-to-device wait time
        Duration downloadTime = Duration::zero();   // Device-to-host wait time
        Duration matchTime    = Duration::zero();
        Duration sortTime     = Duration::zero();
        Duration fxTime       = Duration::zero();

    } timings;
};

#if _DEBUG
    extern ThreadPool* _dbgThreadPool;

    void DbgLoadTablePairs( CudaK32PlotContext& cx, const TableId table, bool copyToPinnedBuffer = false );
    void DbgWritePairs( CudaK32PlotContext& cx, TableId table );
    void DbgWriteContext( CudaK32PlotContext& cx );
    void DbgLoadContextAndPairs( CudaK32PlotContext& cx, bool loadTables = false );
    void DbgLoadMarks( CudaK32PlotContext& cx );
    ThreadPool& DbgGetThreadPool( CudaK32PlotContext& cx );
#endif

void CudaK32PlotDownloadBucket( CudaK32PlotContext& cx );
//void CudaK32PlotUploadBucket( CudaK32PlotContext& cx );


void CudaK32PlotGenSortKey( const uint32 entryCount, uint32* devKey, cudaStream_t stream = nullptr, bool synchronize = false );

template<typename T>
void CudaK32PlotSortByKey( const uint32 entryCount, const uint32* devKey, const T* devInput, T* devOutput, cudaStream_t stream = nullptr, bool synchronize = false );

void CudaK32InlineXsIntoPairs(
    const uint32  entryCount,
          Pair*   devOutPairs,
    const Pair*   devInPairs,
    const uint32* devXs,
    cudaStream_t  stream );

void CudaK32ApplyPairOffset(
    const uint32 entryCount,
    const uint32 offset,
          Pair*  devOutPairs,
    const Pair*  devInPairs,
    cudaStream_t stream );

///
/// Phase 2
///
void CudaK32PlotPhase2( CudaK32PlotContext& cx );
void CudaK32PlotPhase2AllocateBuffers( CudaK32PlotContext& cx, CudaK32AllocContext& acx );

///
/// Phase 3
///
void CudaK32PlotPhase3( CudaK32PlotContext& cx );
void CudaK32PlotPhase3AllocateBuffers( CudaK32PlotContext& cx, CudaK32AllocContext& acx );

///
/// Debug
///
uint64 CudaPlotK32DbgXtoF1( CudaK32PlotContext& cx, const uint32 x );



///
/// Internal
///
//-----------------------------------------------------------
inline uint32 CudaK32PlotGetInputIndex( CudaK32PlotContext& cx )
{
    return ((uint32)cx.table-1) & 1;
}

//-----------------------------------------------------------
inline uint32 CudaK32PlotGetOutputIndex( CudaK32PlotContext& cx )
{
    return (uint32)cx.table & 1;
}

//-----------------------------------------------------------
inline bool CudaK32PlotIsOutputVertical( CudaK32PlotContext& cx )
{
    return CudaK32PlotGetOutputIndex( cx ) == 0;
}

//-----------------------------------------------------------
inline size_t GetMarkingTableBitFieldSize()
{
    return ((1ull << BBCU_K) / 64) * sizeof(uint64);
}

#define CuCDiv( a, b ) (( (a) + (b) - 1 ) / (b))

//-----------------------------------------------------------
template <typename T>
__host__ __device__ __forceinline__ constexpr T CuBBLog2( T x )
{
    T r = 0;
    while( x >>= 1 )
        r++;
    return r;
}



// Calculates x * (x-1) / 2. Division is done before multiplication.
//-----------------------------------------------------------
__device__ __forceinline__ uint64 CudaGetXEnc64( uint64 x )
{
    uint64 a = x, b = x - 1;

    if( (a & 1) == 0 )
        a >>= 1;
    else
        b >>= 1;

    return a * b;
}

//-----------------------------------------------------------
__device__ __forceinline__ uint64 CudaSquareToLinePoint64( uint64 x, uint64 y )
{
    return CudaGetXEnc64( max( x, y ) ) + min( x, y );
}

//-----------------------------------------------------------
template<typename T>
__device__ inline void CuGetThreadOffsets( const uint32 id, const uint32 threadCount, const T totalCount, T& count, T& offset, T& end )
{
    const T countPerThread = totalCount / (T)threadCount;
    const T remainder      = totalCount - countPerThread * (T)threadCount;

    count  = countPerThread;
    offset = (T)id * countPerThread;

    if( id == threadCount - 1 )
        count += remainder;

    end = offset + count;
}

//-----------------------------------------------------------
__host__ __device__ __forceinline__ bool CuBitFieldGet( const uint64* bitfield, uint64 index )
{
    const uint64 fieldIdx = index >> 6;                          // Divide by 64. Safe to do with power of 2. (shift right == log2(64))
    const uint64 field    = bitfield[fieldIdx];

    const uint32 rShift   = (uint32)(index - (fieldIdx << 6));  // Multiply by fieldIdx (shift left == log2(64))
    return (bool)((field >> rShift) & 1u);
}


//-----------------------------------------------------------
__device__ __forceinline__ uint32 atomicAggrInc( uint32* dst )
{
    // Increment from coallesced group first
    coalesced_group g = coalesced_threads();
    
    uint32 prev;
    if( g.thread_rank() == 0 )
        prev = atomicAdd( dst, g.size() );

    prev = g.thread_rank() + g.shfl( prev, 0 );
    return prev;
}

//-----------------------------------------------------------
__device__ __forceinline__ uint32 atomicGlobalOffset( uint32* globalCount )
{
    __shared__ uint32 sharedCount;

    thread_block block = this_thread_block();

    if( block.thread_rank() == 0 )
        sharedCount = 0;

    // Store block-wide offset
    block.sync();
    const uint32 blockOffset = atomicAggrInc( &sharedCount );
    block.sync();

    // Store global offset
    if( block.thread_rank() == 0 )
        sharedCount = atomicAdd( globalCount, sharedCount );

    block.sync();

    // Broadcast the shared count to each thread
    const uint32 gOffset = sharedCount + blockOffset;
    return gOffset;
}

//-----------------------------------------------------------
__device__ __forceinline__ uint32 atomicAddShared( uint32* globalCount, const uint32 count  )
{
    __shared__ uint32 sharedCount;

    thread_block block = this_thread_block();

    if( block.thread_rank() == 0 )
        sharedCount = 0;

    // Store shared offset
    block.sync();
    const uint32 offset = atomicAdd( &sharedCount, count );
    block.sync();

    // Store global offset
    if( block.thread_rank() == 0 )
        sharedCount = atomicAdd( globalCount, sharedCount );

    block.sync();

    return sharedCount + offset;
}


#if _DEBUG


#include "b3/blake3.h"

//-----------------------------------------------------------
inline void DbgPrintHash( const char* msg, const void* ptr, const size_t size )
{
    byte hash[32];
    
    blake3_hasher hasher;
    blake3_hasher_init( &hasher );
    blake3_hasher_update( &hasher, ptr, size );
    blake3_hasher_finalize( &hasher, hash, sizeof( hash ) );

    char hashstr[sizeof(hash)*2+1] = {};
    size_t _;
    BytesToHexStr( hash, sizeof( hash ), hashstr, sizeof( hashstr ), _ );

    Log::Line( "%s 0x%s", msg, hashstr );
}

//-----------------------------------------------------------
inline void DbgPrintDeviceHash( const char* msg, const void* ptr, const size_t size, cudaStream_t stream )
{
    byte hash[32];

    void* hostBuffer = bbvirtallocboundednuma<byte>( size );
    Log::Line( "Marker Set to %d", 5)
CudaErrCheck( cudaMemcpyAsync( hostBuffer, ptr, size, cudaMemcpyDeviceToHost, stream ) );
    Log::Line( "Marker Set to %d", 6)
CudaErrCheck( cudaStreamSynchronize( stream ) );

    blake3_hasher hasher;
    blake3_hasher_init( &hasher );
    blake3_hasher_update( &hasher, hostBuffer, size );
    blake3_hasher_finalize( &hasher, hash, sizeof( hash ) );

    bbvirtfreebounded( hostBuffer );

    char hashstr[sizeof( hash ) * 2 + 1] = {};
    size_t _;
    BytesToHexStr( hash, sizeof( hash ), hashstr, sizeof( hashstr ), _ );

    Log::Line( "%s 0x%s", msg, hashstr );
}

//-----------------------------------------------------------
template<typename T>
inline void DbgPrintDeviceHashT( const char* msg, const T* ptr, const size_t count, cudaStream_t stream )
{
    return DbgPrintDeviceHash( msg, ptr, count * sizeof( T ), stream );
}

//-----------------------------------------------------------
inline ThreadPool& DbgGetThreadPool( CudaK32PlotContext& cx )
{
    if( _dbgThreadPool == nullptr )
        _dbgThreadPool = new ThreadPool( SysHost::GetLogicalCPUCount() );

    return *_dbgThreadPool;
}

#endif