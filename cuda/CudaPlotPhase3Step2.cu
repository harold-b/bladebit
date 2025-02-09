#include "CudaPlotPhase3Internal.h"
#include "CudaParkSerializer.h"
#include "plotting/TableWriter.h"
#include "algorithm/RadixSort.h"
#include "plotdisk/jobs/IOJob.h"

#define P3_CalculateMaxLPValue( x ) ((((uint64)(x))/2)*((uint64)(x))+x)
#define P3_CalculateTableDivisor( p ) (P3_CalculateMaxLPValue( (uint64)(BBCU_TABLE_ENTRY_COUNT*(p)) ) / BBCU_BUCKET_COUNT)

__constant__ uint64 BucketDivisor;

static void CudaK32PlotPhase3Step2Compressed( CudaK32PlotContext& cx );

//-----------------------------------------------------------
__global__ static void CudaUnpackLMap( const uint32 entryCount, const LMap* devLMap, uint32* devLTable
#if _DEBUG
    , const uint32 bucket
#endif
)
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if( gid >= entryCount )
        return;

    const uint32 bucketMask = (1u << (BBCU_K - BBC_BUCKET_BITS)) - 1;
    const LMap   map        = devLMap[gid];
        
    const uint32 dst = map.sourceIndex & bucketMask;

    CUDA_ASSERT( ( map.sourceIndex >> ( 32 - BBC_BUCKET_BITS ) ) == bucket );

    devLTable[dst] = map.sortedIndex;
}

//-----------------------------------------------------------
static void UnpackLMap( CudaK32PlotContext& cx, const uint32 entryCount, const LMap* devLMap, uint32* devLTable,
                        const uint32 bucket, cudaStream_t stream )
{
    const uint32 threads = 256;
    const uint32 blocks  = CDiv( entryCount, threads );

    CudaUnpackLMap<<<blocks, threads, 0, stream>>>( entryCount, devLMap, devLTable
#if _DEBUG
        , bucket 
#endif
    );
}


//-----------------------------------------------------------
template<bool isCompressed=false>
__global__ static void CudaConvertRMapToLinePoints( 
    const uint64 entryCount, const uint32 rOffset, const uint32 lTableOffset,
    const uint32* lTable, const RMap* rmap, uint64* outLPs, uint32* outIndices, uint32* gBucketCounts, const uint32 lpShift = 0 )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    __shared__ uint32 sharedBuckets[BBCU_BUCKET_COUNT];

    CUDA_ASSERT( gridDim.x >= BBCU_BUCKET_COUNT );
    if( id < BBCU_BUCKET_COUNT )
        sharedBuckets[id] = 0;

    __syncthreads();

    uint32 bucket;
    uint32 offset;
    uint32 rIndex;
    uint64 lp;

    if( gid < entryCount )
    {
        const RMap map = rmap[gid];

        const uint32 left  = map.dstL - lTableOffset;
        const uint32 right = map.dstR - lTableOffset;

        CUDA_ASSERT( left  < BBCU_BUCKET_ALLOC_ENTRY_COUNT );
        CUDA_ASSERT( right < BBCU_BUCKET_ALLOC_ENTRY_COUNT );
        CUDA_ASSERT( left < right );

        rIndex = map.src;

        const uint32 x = lTable[left ];
        const uint32 y = lTable[right];

        lp = CudaSquareToLinePoint64( x, y );

        if constexpr( !isCompressed )
        {
            CUDA_ASSERT( x || y );
            CUDA_ASSERT( lp );
            bucket = (uint32)( lp / BucketDivisor );
        }
        else
            bucket = (uint32)( lp >> lpShift );

        CUDA_ASSERT( bucket < BBCU_BUCKET_COUNT );

        offset = atomicAdd( &sharedBuckets[bucket], 1 );
    }
    __syncthreads();

    // Global offset
    if( id < BBCU_BUCKET_COUNT )
    {
        sharedBuckets[id] = atomicAdd( &gBucketCounts[id], sharedBuckets[id] );
        CUDA_ASSERT( sharedBuckets[id] <= P3_PRUNED_SLICE_MAX );
    }
    __syncthreads();

    if( gid >= entryCount )
        return;

    const uint32 dst = bucket * P3_PRUNED_SLICE_MAX + sharedBuckets[bucket] + offset;
    CUDA_ASSERT( dst < P3_PRUNED_BUCKET_MAX );

    outLPs    [dst] = lp;
    outIndices[dst] = rIndex;
}

//-----------------------------------------------------------
static void ConvertRMapToLinePoints( CudaK32PlotContext& cx, const uint32 entryCount, const uint32 rOffset,
                                     const uint32* lTable, const RMap* rMap, uint64* outLPs, uint32* outIndices, cudaStream_t stream )
{
    const TableId rTable = cx.table;
    auto& p3 = *cx.phase3;
    auto& s2 = p3.step2;

    const uint32 threads = 256;
    const uint32 blocks  = CDiv( entryCount, threads );

    const uint32 lTableOffset = cx.bucket * BBCU_BUCKET_ENTRY_COUNT;
    
    uint32* devSliceCounts = cx.devSliceCounts + cx.bucket * BBCU_BUCKET_COUNT;
    #define Rmap2LPParams entryCount, rOffset, lTableOffset, lTable, rMap, outLPs, outIndices, devSliceCounts

    const bool isCompressed = rTable - 1 <= (TableId)cx.gCfg->numDroppedTables;

    if( !isCompressed )
    {
        if( cx.bucket == 0 )
        {
            // Calculate the divisor needed to generate a uniform distribution across buckets
            // and set it as a constant for our kernel.
            const uint64 prunedEntryCount = p3.prunedTableEntryCounts[(int)rTable - 1];
            const uint64 divisor          = P3_CalculateMaxLPValue( prunedEntryCount ) / BBCU_BUCKET_COUNT;

            // #TODO: Use upload stream?
            Log::Line( "Marker Set to %d", 20)
CudaErrCheck( cudaMemcpyToSymbolAsync( BucketDivisor, &divisor, sizeof( divisor ), 0, cudaMemcpyHostToDevice, cx.computeStream ) );
        }

        CudaConvertRMapToLinePoints<false><<<blocks, threads, 0, stream>>>( Rmap2LPParams, 0 );
    }
    else
    {
        const uint32 xBits      = cx.gCfg->compressedEntryBits;
        const uint32 lpBits     = (xBits * 2 - 1) * 2 - 1;
        const uint32 lpBitShift = lpBits - BBC_BUCKET_BITS;

        CudaConvertRMapToLinePoints<true><<<blocks, threads, 0, stream>>>( Rmap2LPParams, lpBitShift );
    }

    #undef Rmap2LPParams
}

/**
 * Load RMap and L table and generate line points from RMap and L table.
 * Write line points to their buckets, along with their origin index.
*/
//-----------------------------------------------------------
void CudaK32PlotPhase3Step2( CudaK32PlotContext& cx )
{
    auto LoadLBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void {

        auto& p3 = *cx.phase3;
        auto& s2 = p3.step2;

        const bool isCompressed = (uint32)cx.table-1 <= cx.gCfg->numDroppedTables;

        if( !isCompressed )
        {
            ASSERT( p3.prunedBucketCounts[(int)cx.table-1][cx.bucket] > 0 );

            // Load lMap
            // Horizontal load
            const LMap* lmap = p3.hostLMap + (size_t)bucket * P3_PRUNED_BUCKET_MAX;

            const uint32* lSliceCounts = &p3.step3.prunedBucketSlices[0][bucket];

            s2.lMapIn.UploadArrayT<LMap>( lmap, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, lSliceCounts );
        }
        else
        {
            ASSERT( cx.gCfg->compressionLevel > 0 );

            if( bucket == 0 )
                p3.pairsLoadOffset = 0;

            // Load the compressed entries from the table pairs
            const uint32* lEntries = (cx.hostBackPointers[(int)cx.table-1].left) + p3.pairsLoadOffset;
            // const uint32* lEntries         = cx.hostTableL + p3.pairsLoadOffset;   // Our compressed x's are copied to the LMap buffer before we get to this point

            // #TODO: Do a preload here instead and have each bucket start at the max bucket offset
            // const uint32 bucketEntryCount = cx.bucketCounts[(int)cx.table-1][bucket];

            s2.lMapIn.UploadT<uint32>( lEntries, BBCU_BUCKET_ENTRY_COUNT );
            p3.pairsLoadOffset += BBCU_BUCKET_ENTRY_COUNT;
        }
    };

    auto UnpackLBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void {

        auto& p3 = *cx.phase3;
        auto& s2 = p3.step2;

        const bool isCompressed = (uint32)cx.table-1 <= cx.gCfg->numDroppedTables;

        const auto* lMap   = (LMap*)s2.lMapIn.GetUploadedDeviceBuffer( cx.computeStream );
        uint32*     lTable = s2.devLTable[bucket & 1];

        if( isCompressed )
        {
            // Copy from upload buffer to working buffer
            Log::Line( "Marker Set to %d", 21)
CudaErrCheck( cudaMemcpyAsync( lTable, lMap, BBCU_BUCKET_ENTRY_COUNT * sizeof( uint32 ), cudaMemcpyDeviceToDevice, cx.computeStream ) );
        }
        else
        {
            // Unpack next LMap and copy to the end of the current map
            const uint32 lEntryCount = p3.prunedBucketCounts[(int)cx.table-1][bucket];
            ASSERT( lEntryCount > 0 );
        
            UnpackLMap( cx, lEntryCount, lMap, lTable, bucket, cx.computeStream );
        }
    };

    auto LoadRBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void {

        auto& p3 = *cx.phase3;
        auto& s2 = p3.step2;

        // Load rMap
        // Horizontal load
        const RMap* rmap = p3.hostRMap + (size_t)bucket * P3_PRUNED_BUCKET_MAX;
        
        const uint32* rSliceCounts = &p3.step1.prunedBucketSlices[0][bucket];

        s2.rMapIn.UploadArrayT<RMap>( rmap, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, rSliceCounts );
    };


    const TableId rTable = cx.table;
    const TableId lTable = rTable-1;
    auto&         p3     = *cx.phase3;
    auto&         s2     = p3.step2;


    // We always have 1 L bucket loaded ahead since we need the next bucket to be 
    // loaded also so that we can include the next bucket's initial 
    // entries as part of the current bucket.
    LoadLBucket( cx, 0 );
    LoadLBucket( cx, 1 );
    LoadRBucket( cx, 0 );

    // Clear pruned entry count
    Log::Line( "Marker Set to %d", 22)
CudaErrCheck( cudaMemsetAsync( p3.devPrunedEntryCount, 0, sizeof( uint32 ), cx.computeStream ) );

    // Unpack the first map beforehand
    UnpackLBucket( cx, 0 );


    ///
    /// Process buckets
    /// 
    uint32 rTableOffset = 0; // Track the global origin index of R entry/line point 

    Log::Line( "Marker Set to %d", 23)
CudaErrCheck( cudaMemsetAsync( cx.devSliceCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, cx.computeStream ) );

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        cx.bucket = bucket;
        const uint32 nextBucket  = bucket + 1;
        const uint32 nextBucketL = bucket + 2;

        const uint32* devLTable = s2.devLTable[bucket & 1];

        // Preload next buckets
        if( nextBucket < BBCU_BUCKET_COUNT )
        {
            LoadRBucket( cx, nextBucket );

            UnpackLBucket( cx, nextBucket );
            s2.lMapIn.ReleaseDeviceBuffer( cx.computeStream );

            // Copy start of next bucket to the end of the current one
            const uint32 copyCount = BBCU_BUCKET_COUNT * BBCU_XTRA_ENTRIES_PER_SLICE;
            static_assert( BBCU_BUCKET_ALLOC_ENTRY_COUNT - BBCU_BUCKET_ENTRY_COUNT == copyCount );

            uint32* nextLTable = s2.devLTable[nextBucket & 1];
            Log::Line( "Marker Set to %d", 24)
CudaErrCheck( cudaMemcpyAsync( (uint32*)devLTable + BBCU_BUCKET_ENTRY_COUNT, nextLTable, copyCount * sizeof( uint32 ), cudaMemcpyDeviceToDevice, cx.computeStream ) );
        }

        if( nextBucketL < BBCU_BUCKET_COUNT )
            LoadLBucket( cx, nextBucketL );


        // Generate line points given the unpacked LMap as input and the RMap
        const auto*  rMap        = (RMap*)s2.rMapIn.GetUploadedDeviceBuffer( cx.computeStream );
        const uint32 rEntryCount = p3.prunedBucketCounts[(int)rTable][bucket];


        uint64* devOutLPs     = (uint64*)s2.lpOut   .LockDeviceBuffer( cx.computeStream );
        uint32* devOutIndices = (uint32*)s2.indexOut.LockDeviceBuffer( cx.computeStream );

        ConvertRMapToLinePoints( cx, rEntryCount, rTableOffset, devLTable, rMap, devOutLPs, devOutIndices, cx.computeStream );
        s2.rMapIn.ReleaseDeviceBuffer( cx.computeStream );
        rTableOffset += rEntryCount;

        // Horizontal download (write 1 row)
        s2.lpOut   .Download2DT<uint64>( p3.hostLinePoints + (size_t)bucket * P3_PRUNED_BUCKET_MAX  , P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX  , P3_PRUNED_SLICE_MAX, cx.computeStream );
        s2.indexOut.Download2DT<uint32>( p3.hostIndices    + (size_t)bucket * P3_PRUNED_BUCKET_MAX*3, P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX*3, P3_PRUNED_SLICE_MAX, cx.computeStream );
    }

    #if _DEBUG
    {
        size_t tableLength       = 0;
        uint32 activeBucketCount = 0;
        for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
        {
            ASSERT( p3.prunedBucketCounts[(int)rTable][i] <= P3_PRUNED_BUCKET_MAX );
            tableLength += p3.prunedBucketCounts[(int)rTable][i];

            if( p3.prunedBucketCounts[(int)rTable][i] ) activeBucketCount++;
        }

        ASSERT( tableLength <= BBCU_TABLE_ALLOC_ENTRY_COUNT );
        ASSERT( tableLength == p3.prunedTableEntryCounts[(int)rTable] );
    }
    #endif

    s2.lpOut.WaitForCompletion();
    s2.lpOut.Reset();

    s2.indexOut.WaitForCompletion();
    s2.indexOut.Reset();

    s2.lMapIn.Reset();
    s2.rMapIn.Reset();

    // Copy slice counts & bucket count
    cudaStream_t downloadStream = s2.lpOut.GetQueue()->GetStream();

    Log::Line( "Marker Set to %d", 25)
CudaErrCheck( cudaMemcpyAsync( cx.hostBucketSlices, cx.devSliceCounts, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT,
                    cudaMemcpyDeviceToHost, downloadStream ) );

    memset( p3.prunedBucketCounts[(int)rTable], 0, BBCU_BUCKET_COUNT * sizeof( uint32 ) );

    Log::Line( "Marker Set to %d", 26)
CudaErrCheck( cudaStreamSynchronize( downloadStream ) );
    bbmemcpy_t( &s2.prunedBucketSlices[0][0], cx.hostBucketSlices, BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT );
    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
        {
            ASSERT( s2.prunedBucketSlices[slice][bucket] <= P3_PRUNED_SLICE_MAX );
            p3.prunedBucketCounts[(int)rTable][bucket] += s2.prunedBucketSlices[slice][bucket];
        }
    //     //ASSERT( p3.hostBucketCounts[i] );
        ASSERT( p3.prunedBucketCounts[(int)rTable][bucket] <= P3_PRUNED_BUCKET_MAX );
    }

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->phase3.rMapBuffer->Swap();
        cx.diskContext->phase3.lpAndLMapBuffer->Swap();
        cx.diskContext->phase3.indexBuffer->Swap();
    }

    // #if _DEBUG
    // // if( cx.table > TableId::Table3 )
    // {
    //    DbgValidateStep2Output( cx );
    // }
    // #endif
}

//-----------------------------------------------------------
void WritePark7( CudaK32PlotContext& cx )
{
    auto LoadBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void {

        auto& p3 = *cx.phase3;
        auto& s2 = p3.step2;

        ASSERT( p3.prunedBucketCounts[(int)TableId::Table7][cx.bucket] > 0 );

        // Load lMap
        // Horizontal load
        const LMap* lmap = p3.hostLMap + (size_t)bucket * P3_PRUNED_BUCKET_MAX;

        const uint32* lSliceCounts = &p3.step3.prunedBucketSlices[0][bucket];

        s2.lMapIn.UploadArrayT<LMap>( lmap, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, lSliceCounts );
    };

    ASSERT( cx.table == TableId::Table7 );

    auto& p3 = *cx.phase3;
    auto& s2 = p3.step2;


    // Load initial bucket
    LoadBucket( cx, 0 );

    // Begin park 7 table in plot
    cx.plotWriter->BeginTable( PlotTable::Table7 );

    constexpr size_t parkSize       = P3_PARK_7_SIZE;
    constexpr size_t parkFieldCount = parkSize / sizeof( uint64 );
    static_assert( parkFieldCount * sizeof( uint64 ) == parkSize );

    GpuDownloadBuffer& parkDownloader = cx.useParkContext ? s2.parksOut : s2.lpOut;

    constexpr size_t maxParksPerBucket = P3_MAX_P7_PARKS_PER_BUCKET;
    static_assert( sizeof( uint64 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT >= maxParksPerBucket * parkSize );

    if( cx.useParkContext )
    {
        cx.parkContext->parkBufferChain->Reset();
    }

    // Host stuff
    constexpr size_t hostMetaTableSize = sizeof( RMap ) * BBCU_TABLE_ALLOC_ENTRY_COUNT;
    StackAllocator hostAllocator( p3.hostRMap, hostMetaTableSize );

    const uint64 tableEntryCount = cx.tableEntryCounts[(int)cx.table];
    const size_t totalParkCount  = CDiv( (size_t)tableEntryCount, kEntriesPerPark );

    byte*   hostParks           = cx.useParkContext ? nullptr : hostAllocator.AllocT<byte>( totalParkCount * parkSize );
    byte*   hostParksWriter     = cx.useParkContext ? nullptr : hostParks;
    uint32* hostLastParkEntries = cx.useParkContext ? (uint32*)cx.parkContext->hostRetainedLinePoints : 
                                                      hostAllocator.CAlloc<uint32>( kEntriesPerPark );

    static_assert( kEntriesPerPark * maxParksPerBucket <= BBCU_BUCKET_ALLOC_ENTRY_COUNT * 2 );
    uint32* devIndexBuffer     = s2.devLTable[0] + kEntriesPerPark;
    uint32  retainedEntryCount = 0;

    // Begin serialization
    cudaStream_t downloadStream = parkDownloader.GetQueue()->GetStream();

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        if( bucket + 1 < BBCU_BUCKET_COUNT )
            LoadBucket( cx, bucket+1 );

        const uint32 bucketEntryCount = p3.prunedBucketCounts[(int)TableId::Table7][bucket];

        // Unmap bucket
        auto* lMap = (LMap*)s2.lMapIn.GetUploadedDeviceBuffer( cx.computeStream );
        UnpackLMap( cx, bucketEntryCount, lMap, devIndexBuffer, bucket, cx.computeStream );
        s2.lMapIn.ReleaseDeviceBuffer( cx.computeStream );

        // Serialize indices into a park
        uint32* indices    = devIndexBuffer - retainedEntryCount;
        uint32  indexCount = bucketEntryCount + retainedEntryCount;

        const uint32 parkCount = indexCount / kEntriesPerPark;

        uint64* devParkFields = (uint64*)parkDownloader.LockDeviceBuffer( cx.computeStream );
        SerializePark7InGPU( parkCount, indices, devParkFields, parkFieldCount, cx.computeStream );


        // Retain any entries that did not fit into a park
        retainedEntryCount = indexCount - (parkCount * kEntriesPerPark);
        if( retainedEntryCount > 0 )
        {
            const bool isLastBucket = bucket + 1 == BBCU_BUCKET_COUNT;

            const uint32  serializedEntryCount = parkCount * kEntriesPerPark;
            const uint32* copySource           = indices + serializedEntryCount;
            const size_t  copySize             = sizeof( uint32 ) * retainedEntryCount;

            if( !isLastBucket )
                Log::Line( "Marker Set to %d", 27)
CudaErrCheck( cudaMemcpyAsync( devIndexBuffer - retainedEntryCount, copySource, copySize, cudaMemcpyDeviceToDevice, cx.computeStream ) );
            else
                Log::Line( "Marker Set to %d", 28)
CudaErrCheck( cudaMemcpyAsync( hostLastParkEntries, copySource, copySize, cudaMemcpyDeviceToHost, cx.computeStream ) );
        }

        // Download parks & write to plot
        const size_t downloadSize = parkCount * parkSize;

        if( cx.useParkContext )
        {
            ASSERT( downloadSize <= cx.parkContext->parkBufferChain->BufferSize() );

            // Override the park buffer to be used when using a park context
            hostParksWriter = cx.parkContext->parkBufferChain->PeekBuffer( bucket );

            // Wait for the next park buffer to be available
            parkDownloader.HostCallback([&cx]{
               (void)cx.parkContext->parkBufferChain->GetNextBuffer();
            });
        }

        parkDownloader.DownloadWithCallback( hostParksWriter, downloadSize,
              []( void* parksBuffer, size_t size, void* userData ) {

                auto& cx = *reinterpret_cast<CudaK32PlotContext*>( userData );
                cx.plotWriter->WriteTableData( parksBuffer, size );

                // Release the buffer after the plot writer is done with it.
                if( cx.useParkContext )
                {
                    cx.plotWriter->CallBack([&cx](){
                        cx.parkContext->parkBufferChain->ReleaseNextBuffer();
                    });
                }
                
            }, &cx, cx.computeStream );

        hostParksWriter += downloadSize;
        if( cx.useParkContext )
            hostParksWriter = nullptr;
    }

    // Wait for parks to complete downloading
    parkDownloader.WaitForCompletion();
    parkDownloader.Reset();

    CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );
    CudaErrCheck( cudaStreamSynchronize( downloadStream ) );

    // Was there a left-over park?
    if( retainedEntryCount > 0 )
    {
        if( cx.useParkContext )
            hostParksWriter = cx.parkContext->parkBufferChain->GetNextBuffer();

        // Submit last park to plot
        TableWriter::WriteP7Parks( 1, hostLastParkEntries, hostParksWriter );
        cx.plotWriter->WriteTableData( hostParksWriter, parkSize );

        if( cx.useParkContext )
        {
            cx.plotWriter->CallBack([&cx](){
                cx.parkContext->parkBufferChain->ReleaseNextBuffer();
            });
        }
    }
    cx.plotWriter->EndTable();

    // Cleanup
    s2.lMapIn.Reset();
}


#if _DEBUG

//-----------------------------------------------------------
static void _DbgValidateOutput( CudaK32PlotContext& cx );
void DbgValidateStep2Output( CudaK32PlotContext& cx )
{
    // New stack (prevent overflow)
    auto* thread = new Thread();
    thread->Run( []( void* p ) {
        _DbgValidateOutput( *(CudaK32PlotContext*)p );
    }, &cx );

    thread->WaitForExit();
    delete thread;
}

//-----------------------------------------------------------
void _DbgValidateOutput( CudaK32PlotContext& cx )
{
    const TableId rTable = cx.table;
    auto& p3 = *cx.phase3;
    auto& s2 = p3.step2;

    // Validate line points...
    Log::Debug( "[DEBUG] Validating line points..." );
    uint64* refLinePoints = bbcvirtallocboundednuma<uint64>( BBCU_TABLE_ALLOC_ENTRY_COUNT );
    uint64* tmpLinePoints = bbcvirtallocboundednuma<uint64>( BBCU_TABLE_ALLOC_ENTRY_COUNT );
    uint32* indices       = bbcvirtallocboundednuma<uint32>( BBCU_TABLE_ALLOC_ENTRY_COUNT );

    uint64* writer    = refLinePoints;
    uint32* idxWriter = indices;

    const uint64 prunedEntryCount = p3.prunedTableEntryCounts[(int)rTable];

    const uint32 lpBits        = 63; // #TODO: Change when compressing here
    const uint32 lpBucketShift = lpBits - BBC_BUCKET_BITS;

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        uint64* reader    = p3.hostLinePoints + bucket * P3_PRUNED_SLICE_MAX;
        uint32* idxReader = p3.hostIndices    + bucket * P3_PRUNED_SLICE_MAX*3;

        for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice ++ )
        {
            const size_t count = s2.prunedBucketSlices[slice][bucket];
            bbmemcpy_t( writer   , reader   , count );
            bbmemcpy_t( idxWriter, idxReader, count );
            
            // The line points must be in their given buckets if inlined x's
            if( cx.table-1 == TableId::Table1 )
            {
                for( size_t i = 0; i < count; i++ )
                {
                    const uint64 lp = writer[i];
                    const uint32 b  = lp >> lpBucketShift;
                    ASSERT( b == bucket );
                }
            }

            writer    += count;
            idxWriter += count;
            reader    += P3_PRUNED_BUCKET_MAX;
            idxReader += P3_PRUNED_BUCKET_MAX*3;
        }
    }

    const uint64 readEntries = (uint64)( (uintptr_t)writer - (uintptr_t)refLinePoints ) / sizeof( uint64 );
    ASSERT( readEntries == prunedEntryCount );

    ThreadPool& pool = DbgGetThreadPool( cx );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, refLinePoints, tmpLinePoints, prunedEntryCount );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, indices, (uint32*)tmpLinePoints, prunedEntryCount );

    for( uint32 i = 1; i < (uint32)prunedEntryCount; i++ )
    {
        ASSERT( indices[i] >= indices[i-1] );
    }

    for( uint64 i = 1; i < prunedEntryCount; i++ )
    {
        ASSERT( refLinePoints[i] >= refLinePoints[i-1] );
    }

    // Delta test
    // #TODO: Get correct stub bit size depending on compression
    const uint32 stubBitSize = (BBCU_K - kStubMinusBits);
    for( uint32 i = 0; i < (uint32)prunedEntryCount; i+=kEntriesPerPark )
    {
        const uint32 parkCount = std::min( prunedEntryCount - i, (uint64)kEntriesPerPark );

        const uint64* park = refLinePoints + i;

        uint64 prevLp = park[0];

        for( uint32 j = 1; j < parkCount; j++ )
        {
            uint64 lp         = park[j];
            uint64 delta      = lp - prevLp;
            uint64 smallDelta = delta >> stubBitSize;
            ASSERT( smallDelta < 256 );

            prevLp = lp;
        }
    }

    DbgHashDataT( refLinePoints, prunedEntryCount, "line_points", (uint32)cx.table+1 );

    bbvirtfreebounded( refLinePoints );
    bbvirtfreebounded( tmpLinePoints );
    bbvirtfreebounded( indices );

    Log::Debug( "[DEBUG] Line point validation OK" );
}

#endif

//-----------------------------------------------------------
void DbgDumpSortedLinePoints( CudaK32PlotContext& cx )
{
    Log::Line( "[DEBUG] Prpaparing line ponts for writing to file." );
    const TableId rTable = cx.table;

    auto& p3 = *cx.phase3;
    auto& s2 = p3.step2;


    uint64* sortedLinePoints = bbcvirtallocboundednuma<uint64>( BBCU_TABLE_ALLOC_ENTRY_COUNT );
    uint64* tmpLinePoints    = bbcvirtallocboundednuma<uint64>( BBCU_TABLE_ALLOC_ENTRY_COUNT );

    uint64* writer = sortedLinePoints;

    const uint64 prunedEntryCount = p3.prunedTableEntryCounts[(int)rTable];

    const uint32 lpBits        = 63; // #TODO: Change when compressing here
    const uint32 lpBucketShift = lpBits - BBC_BUCKET_BITS;

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        uint64* reader = p3.hostLinePoints + bucket * P3_PRUNED_SLICE_MAX;

        for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice ++ )
        {
            const size_t count = s2.prunedBucketSlices[slice][bucket];
            bbmemcpy_t( writer, reader, count );

            writer    += count;
            reader    += P3_PRUNED_BUCKET_MAX;
        }
    }

    // Sort
    ThreadPool& pool = *cx.threadPool; //DbgGetThreadPool( cx );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, sortedLinePoints, tmpLinePoints, prunedEntryCount );

    // DbgHashDataT( sortedLinePoints, prunedEntryCount, "sorted_line_points", (uint32)cx.table+1 );

    // Write to disk
    {
        char filePath[1024] = {};
        sprintf( filePath, "%s/lp.c%u.ref", "/home/harold/plot/ref/compressed-lps", (uint32)cx.gCfg->compressionLevel );

        FileStream file;
        if( file.Open( filePath, FileMode::Open, FileAccess::Read ) )
        {
            Log::Line( "[DEBUG]File %s already exists. Cannot overwrite.", filePath );
        }
        else
        {
            Log::Line( "[DEBUG] Writing line points to %s", filePath );
            file.Close();
            file.Open( filePath, FileMode::Create, FileAccess::Write );

            void* block = bbvirtalloc( file.BlockSize() );
            int err;
            if( !IOJob::WriteToFile( file, sortedLinePoints, prunedEntryCount * sizeof( uint64 ), block, file.BlockSize(), err ) )
                Log::Line( "Failed to to file %s with error %d.", filePath, err );

            bbvirtfree( block );

            Log::Line( "[DEBUG] Wrote %llu line points", prunedEntryCount );
        }

        file.Close();
    }

    bbvirtfreebounded( sortedLinePoints );
    bbvirtfreebounded( tmpLinePoints );
}
