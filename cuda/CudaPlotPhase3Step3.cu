#include "CudaPlotPhase3Internal.h"
#include "CudaParkSerializer.h"
#include "plotmem/ParkWriter.h"

static void GenerateLMap( CudaK32PlotContext& cx, const uint32 entryCount, const uint32 finalOffset, const uint32* indices, cudaStream_t stream );
static void DeltafyLinePoints( CudaK32PlotContext& cx, const uint32 entryCount, const uint64* linePoints, uint64* deltaLinePoints, cudaStream_t stream );

#if _DEBUG
    #include "plotdisk/jobs/IOJob.h"
    static void DbgSaveLMap( CudaK32PlotContext& cx );
    static void DbgValidateLMapData( CudaK32PlotContext& cx );
    static void DbgValidateLMap( CudaK32PlotContext& cx );
#endif

//-----------------------------------------------------------
void CudaK32PlotPhase3Step3( CudaK32PlotContext& cx )
{
    auto LoadBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void {

        auto& p3 = *cx.phase3;
        auto& s2 = p3.step2;
        auto& s3 = p3.step3;

        // if( bucket == 0 )
        //     p3.pairsLoadOffset = 0;

        // Load line points and their source indices
        const TableId rTable     = cx.table;
        const uint32  entryCount = p3.prunedBucketCounts[(int)rTable][bucket];
        ASSERT( entryCount <= P3_PRUNED_BUCKET_MAX );

        if( entryCount < 1 )
            return;

        // Vertical input layout of data: Start at row 0, column according to the current bucket
        const uint64* linePoints = p3.hostLinePoints + (size_t)bucket * P3_PRUNED_SLICE_MAX;
        const uint32* indices    = p3.hostIndices    + (size_t)bucket * P3_PRUNED_SLICE_MAX * 3; // This buffer is shared with RMap ((uint32)*3) (which we're about to write to),
                                                                                                 // which is why we multiply by 3

        const uint32* counts     = &s2.prunedBucketSlices[0][bucket];

        // Load 1 column
        s3.lpIn   .UploadArrayT( linePoints, BBCU_BUCKET_COUNT, P3_PRUNED_BUCKET_MAX  , BBCU_BUCKET_COUNT, counts );
        s3.indexIn.UploadArrayT( indices   , BBCU_BUCKET_COUNT, P3_PRUNED_BUCKET_MAX*3, BBCU_BUCKET_COUNT, counts );
    };

    auto& p3 = *cx.phase3;
    auto& s3 = p3.step3;

    const TableId rTable = cx.table;
    const TableId lTable = cx.table-1;

    // Load CTable
    const bool    isCompressed = cx.gCfg->compressionLevel > 0 && lTable <= (TableId)cx.gCfg->numDroppedTables;
    const uint32  stubBitSize  = !isCompressed ? (BBCU_K - kStubMinusBits) : cx.gCfg->compressionInfo.stubSizeBits;
    const TableId firstTable   = TableId::Table2 + (TableId)cx.gCfg->numDroppedTables;

    const bool    isFirstSerializedTable = firstTable == rTable;

    const size_t      cTableSize = !isCompressed ? sizeof( CTable_0 )   : cx.gCfg->cTableSize;             ASSERT( cTableSize <= P3_MAX_CTABLE_SIZE );
    const FSE_CTable* hostCTable = !isCompressed ? CTables[(int)lTable] : cx.gCfg->ctable;

    // (upload must be loaded before first bucket, on the same stream)
    Log::Line( "Marker Set to %d", 29)
CudaErrCheck( cudaMemcpyAsync( s3.devCTable, hostCTable, cTableSize, cudaMemcpyHostToDevice,
                    s3.lpIn.GetQueue()->GetStream() ) );

    // Load initial bucket
    LoadBucket( cx, 0 );

    // Begin plot table
    cx.plotWriter->BeginTable( (PlotTable)lTable );


    uint32 mapOffset       = 0;
    uint32 retainedLPCount = 0;                     // Line points retained for the next bucket to write to park

    const size_t hostParkSize = isCompressed ? cx.gCfg->compressionInfo.tableParkSize : CalculateParkSize( lTable );
    ASSERT( DEV_MAX_PARK_SIZE >= hostParkSize );

    byte*   hostParksWriter     = (byte*)cx.hostBackPointers[(int)rTable].left;  //(byte*)cx.hostTableL; 
    uint64* hostRetainedEntries = nullptr;

    if( cx.cfg.hybrid128Mode )
    {
        hostParksWriter = (byte*)cx.hostTableL;

        if( !isFirstSerializedTable && !cx.useParkContext )
        {
            // Ensure the this buffer is no longer in use (the last table finished writing to disk.)
            const bool willWaitForParkFence = cx.parkFence->Value() < BBCU_BUCKET_COUNT;
            if( willWaitForParkFence )
                Log::Line( " Waiting for parks buffer to become available." );

            Duration parkWaitTime;
            cx.parkFence->Wait( BBCU_BUCKET_COUNT, parkWaitTime );

            if( willWaitForParkFence )
                Log::Line( " Waited %.3lf seconds for the park buffer to be released.", TicksToSeconds( parkWaitTime ) );
        }
    }
    if( cx.useParkContext )
    {
        cx.parkContext->parkBufferChain->Reset();
    }

    // if( !isCompressed && lTable == TableId::Table1 )
    //     hostParksWriter = (byte*)cx.hostBackPointers[(int)TableId::Table2].left;

    ///
    /// Process buckets
    ///
    uint64* sortedLinePoints = s3.devLinePoints + kEntriesPerPark;
    uint32* sortedIndices    = s3.devIndices;

    cudaStream_t sortAndMapStream = cx.computeStream;
    cudaStream_t lpStream         = cx.computeStream;//B;
    cudaStream_t downloadStream   = cx.gpuDownloadStream[0]->GetStream();

    Log::Line( "Marker Set to %d", 30)
CudaErrCheck( cudaMemsetAsync( cx.devSliceCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, sortAndMapStream ) );
    Log::Line( "Marker Set to %d", 31)
CudaErrCheck( cudaMemsetAsync( s3.devParkOverrunCount, 0, sizeof( uint32 ), sortAndMapStream ) );

    // Set initial event LP stream event as set.
    Log::Line( "Marker Set to %d", 32)
CudaErrCheck( cudaEventRecord( cx.computeEventA, lpStream ) );

    cx.parkFence->Reset( 0 );
    s3.parkBucket = 0;

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        cx.bucket = bucket;

        const uint32 bucketEntryCount = p3.prunedBucketCounts[(int)rTable][bucket];

        if( bucketEntryCount == 0 )
            break;

        if( bucket + 1 < BBCU_BUCKET_COUNT )
            LoadBucket( cx, bucket + 1 );

        // Wait for upload to finish
        uint64* unsortedLinePoints = (uint64*)s3.lpIn   .GetUploadedDeviceBuffer( sortAndMapStream );
        uint32* unsortedIndices    = (uint32*)s3.indexIn.GetUploadedDeviceBuffer( sortAndMapStream );

        // Sort line points
        #if _DEBUG
        {
            size_t sortRequiredSize = 0;
            Log::Line( "Marker Set to %d", 33)
CudaErrCheck( cub::DeviceRadixSort::SortPairs<uint64, uint32>( nullptr, sortRequiredSize, nullptr, nullptr, nullptr, nullptr, bucketEntryCount, 0, 64 ) );
            ASSERT( s3.sizeTmpSort >= sortRequiredSize );
        }
        #endif

        // Wait for the previous bucket's LP work to finish, so we can re-use the device buffer
        Log::Line( "Marker Set to %d", 34)
CudaErrCheck( cudaStreamWaitEvent( sortAndMapStream, cx.computeEventA ) );

        // #TODO: We can use 63-7 (log2(128 buckets)), which might be faster
        // #NOTE: I did change it and the sort failed. Investigate.
        Log::Line( "Marker Set to %d", 35)
CudaErrCheck( cub::DeviceRadixSort::SortPairs<uint64, uint32>(
            s3.devSortTmpData,  s3.sizeTmpSort,
            unsortedLinePoints, sortedLinePoints,
            unsortedIndices,    sortedIndices,
            bucketEntryCount, 0, 64, sortAndMapStream ) );

        Log::Line( "Marker Set to %d", 36)
CudaErrCheck( cudaEventRecord( cx.computeEventB, sortAndMapStream ) );

        s3.lpIn   .ReleaseDeviceBuffer( sortAndMapStream ); unsortedLinePoints = nullptr;
        s3.indexIn.ReleaseDeviceBuffer( sortAndMapStream ); unsortedIndices    = nullptr;

        ///
        /// Map
        /// 
        // Generate map and download to it to host
        GenerateLMap( cx, bucketEntryCount, mapOffset, sortedIndices, sortAndMapStream );
        mapOffset += bucketEntryCount;

        // Vertical download map (write 1 column)
        s3.mapOut.Download2DT<LMap>( p3.hostLMap + (size_t)bucket * P3_PRUNED_SLICE_MAX,
                 P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, P3_PRUNED_BUCKET_MAX, P3_PRUNED_SLICE_MAX, sortAndMapStream );


        ///
        /// Line points
        ///
        // If we have retained entries, let's account for them in this bucket
        uint64* parkLinePoints = sortedLinePoints - retainedLPCount;

        const uint32 totalEntryCount = bucketEntryCount + retainedLPCount;
        const uint32 parkCount       = totalEntryCount / kEntriesPerPark;
        const uint32 entryCount      = parkCount * kEntriesPerPark;
        ASSERT( parkCount <= P3_PRUNED_MAX_PARKS_PER_BUCKET );

        // Wait for sort to finish
        Log::Line( "Marker Set to %d", 37)
CudaErrCheck( cudaStreamWaitEvent( lpStream, cx.computeEventB ) );

        // Deltafy line points
        DeltafyLinePoints( cx, entryCount, parkLinePoints, s3.devDeltaLinePoints, lpStream );

        Log::Line( "Marker Set to %d", 38)
CudaErrCheck( cudaEventRecord( cx.computeEventC, lpStream ) );  // Signal download stream can download remaining line points for last park

        // Compress line point parks
        byte* devParks = (byte*)s3.parksOut.LockDeviceBuffer( lpStream );
        CompressToParkInGPU( parkCount, hostParkSize, s3.devDeltaLinePoints, devParks, DEV_MAX_PARK_SIZE, stubBitSize, s3.devCTable, s3.devParkOverrunCount, lpStream );

        // Retain any entries that did not maked it into parks for the next bucket to process
        retainedLPCount = totalEntryCount - (parkCount * kEntriesPerPark);
        if( retainedLPCount > 0 )
        {
            // Last bucket?
            const bool isLastBucket = bucket + 1 == BBCU_BUCKET_COUNT;

            const uint64* copySource = parkLinePoints + entryCount;
            const size_t  copySize   = sizeof( uint64 ) * retainedLPCount;

            if( !isLastBucket )
            {
                // Not the last bucket, so retain entries for the next GPU compression bucket
                Log::Line( "Marker Set to %d", 39)
CudaErrCheck( cudaMemcpyAsync( sortedLinePoints - retainedLPCount, copySource, copySize, cudaMemcpyDeviceToDevice, lpStream ) );
            }
            else
            {       
                // No more buckets so we have to compress this last park on the CPU
                Log::Line( "Marker Set to %d", 40)
CudaErrCheck( cudaStreamWaitEvent( downloadStream, cx.computeEventC ) );

                hostRetainedEntries = cx.useParkContext ? cx.parkContext->hostRetainedLinePoints :
                                                       (uint64*)( hostParksWriter + hostParkSize * parkCount );
                Log::Line( "Marker Set to %d", 41)
CudaErrCheck( cudaMemcpyAsync( hostRetainedEntries, copySource, copySize, cudaMemcpyDeviceToHost, downloadStream ) );
            }
        }

        Log::Line( "Marker Set to %d", 42)
CudaErrCheck( cudaEventRecord( cx.computeEventA, lpStream ) );  // Signal sortedLinePoints buffer ready for use again


        // Download parks
        if( cx.useParkContext )
        {
            ASSERT( hostParkSize * parkCount <= cx.parkContext->parkBufferChain->BufferSize() );

            // Override the park buffer to be used when using a park context
            hostParksWriter = cx.parkContext->parkBufferChain->PeekBuffer( bucket );

            // Wait for the next park buffer to be available
            s3.parksOut.HostCallback([&cx]{
               (void)cx.parkContext->parkBufferChain->GetNextBuffer();
            });
        }

        s3.parksOut.Download2DWithCallback( hostParksWriter, hostParkSize, parkCount, hostParkSize, DEV_MAX_PARK_SIZE, 
            []( void* parksBuffer, size_t size, void* userData ) {

                auto& cx = *reinterpret_cast<CudaK32PlotContext*>( userData );
                auto& s3 = cx.phase3->step3;

                cx.plotWriter->WriteTableData( parksBuffer, size );
                cx.plotWriter->SignalFence( *cx.parkFence, ++s3.parkBucket );

                // Release the buffer after the plot writer is done with it.
                if( cx.useParkContext )
                {
                    cx.plotWriter->CallBack([&cx](){
                        cx.parkContext->parkBufferChain->ReleaseNextBuffer();
                    });
                }

            }, &cx, lpStream, cx.downloadDirect );

        hostParksWriter += hostParkSize * parkCount;
    
        if( cx.useParkContext )
            hostParksWriter = nullptr;
    }

    // Copy park overrun count
    Log::Line( "Marker Set to %d", 43)
CudaErrCheck( cudaMemcpyAsync( s3.hostParkOverrunCount, s3.devParkOverrunCount, sizeof( uint32 ), cudaMemcpyDeviceToHost, downloadStream ) );

    // Wait for parks to complete downloading
    s3.parksOut.WaitForCompletion();
    s3.parksOut.Reset();

    // Copy map slice counts (for the next step 2)
    Log::Line( "Marker Set to %d", 44)
CudaErrCheck( cudaMemcpyAsync( cx.hostBucketSlices, cx.devSliceCounts, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT,
                    cudaMemcpyDeviceToHost, downloadStream ) );

    Log::Line( "Marker Set to %d", 45)
CudaErrCheck( cudaStreamSynchronize( downloadStream ) );
    memcpy( &s3.prunedBucketSlices[0][0], cx.hostBucketSlices, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT );

    FatalIf( *s3.hostParkOverrunCount > 0, "Park buffer overrun." );

    // Was there a left-over park?
    if( retainedLPCount > 0 )
    {
        if( cx.useParkContext )
            hostParksWriter = cx.parkContext->parkBufferChain->GetNextBuffer();

        uint64 lastParkEntries[kEntriesPerPark];
        bbmemcpy_t( lastParkEntries, hostRetainedEntries, retainedLPCount );

        WritePark( hostParkSize, retainedLPCount, lastParkEntries, hostParksWriter, stubBitSize, hostCTable );
        cx.plotWriter->WriteTableData( hostParksWriter, hostParkSize );

        if( cx.useParkContext )
        {
            cx.plotWriter->CallBack([&cx](){
                cx.parkContext->parkBufferChain->ReleaseNextBuffer();
            });
        }
    }
    cx.plotWriter->EndTable();


    memset( p3.prunedBucketCounts[(int)rTable], 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT );
    for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
        for( uint32 j = 0; j < BBCU_BUCKET_COUNT; j++ )
            p3.prunedBucketCounts[(int)rTable][i] += s3.prunedBucketSlices[j][i];

    s3.mapOut.WaitForCompletion();
    s3.mapOut.Reset();

    s3.lpIn   .Reset();
    s3.indexIn.Reset();

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->phase3.lpAndLMapBuffer->Swap();
        cx.diskContext->phase3.indexBuffer->Swap();
    }


    // #if _DEBUG
    // //if( cx.table >= TableId::Table6 )
    // //{
    //     // DbgValidateLMap( cx );
    //     // DbgValidateLMapData( cx );

    //     // DbgSaveLMap( cx );
    // //}
    // #endif
}


//-----------------------------------------------------------
__global__ void CudaGenerateLMap( const uint32 entryCount, const uint32 finalOffset, const uint32* indices, LMap* gMap, uint32* gBucketCounts )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    __shared__ uint32 sharedBucketCounts[BBCU_BUCKET_COUNT];
    if( id < BBCU_BUCKET_COUNT )
        sharedBucketCounts[id] = 0;

    __syncthreads();

    uint32 index;
    uint32 bucket;
    uint32 offset;

    if( gid < entryCount )
    {
        index = indices[gid];
        
        bucket = ( index >> (32 - BBC_BUCKET_BITS) );
        offset = atomicAdd( &sharedBucketCounts[bucket], 1 );
    }

    __syncthreads();

    // Global offset
    if( id < BBCU_BUCKET_COUNT )
        sharedBucketCounts[id] = atomicAdd( &gBucketCounts[id], sharedBucketCounts[id] );

    __syncthreads();
    
    if( gid >= entryCount )
        return;

    const uint32 dst = bucket * P3_PRUNED_SLICE_MAX + sharedBucketCounts[bucket] + offset;
    
    //CUDA_ASSERT( index != 0 );

    LMap map;
    map.sortedIndex = finalOffset + gid;
    map.sourceIndex = index;
#if _DEBUG
    CUDA_ASSERT( map.sortedIndex != 0 || map.sourceIndex != 0 );
#endif
    gMap[dst] = map;
}

//-----------------------------------------------------------
void GenerateLMap( CudaK32PlotContext& cx, const uint32 entryCount, const uint32 finalOffset, const uint32* indices, cudaStream_t stream )
{
    const uint32 threads = 256;
    const uint32 blocks  = CDiv( entryCount, threads );

    auto& p3 = *cx.phase3;
    auto& s3 = p3.step3;

    auto*   devMap         = (LMap*)s3.mapOut.LockDeviceBuffer( stream );
    uint32* devSliceCounts = cx.devSliceCounts + cx.bucket * BBCU_BUCKET_COUNT;

    CudaErrCheck( cudaMemsetAsync( devSliceCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT, stream ) );
    
    CudaGenerateLMap<<<blocks, threads, 0, stream>>>( entryCount, finalOffset, indices, devMap, devSliceCounts );
}

//-----------------------------------------------------------
__global__ void CudaDeltafyLinePoints( const uint32 entryCount, const uint64* linePoints, uint64* deltaLinePoints )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if( gid >= entryCount )
        return;

    const bool isFirstParkEntry = ( gid & ( kEntriesPerPark - 1 ) ) == 0;

    if( isFirstParkEntry )
    {
        deltaLinePoints[gid] = linePoints[gid];
    }
    else
    {
        //CUDA_ASSERT( linePoints[gid] && linePoints[gid - 1] );
        CUDA_ASSERT( linePoints[gid] >= linePoints[gid - 1] );
        deltaLinePoints[gid] = linePoints[gid] - linePoints[gid - 1];
    }
}

//-----------------------------------------------------------
void DeltafyLinePoints( CudaK32PlotContext& cx, const uint32 entryCount, const uint64* linePoints, uint64* deltaLinePoints, cudaStream_t stream )
{
    ASSERT( entryCount / kEntriesPerPark * kEntriesPerPark == entryCount );

    const uint32 threadsPerBlock = 256;
    const uint32 blockCount      = CDivT( entryCount, threadsPerBlock );
    CudaDeltafyLinePoints<<<blockCount, threadsPerBlock, 0, stream>>>( entryCount, linePoints, deltaLinePoints );
}



#if _DEBUG

//-----------------------------------------------------------
void DbgSaveLMap( CudaK32PlotContext& cx )
{
    Log::Line( "[DEBUG] Saving table %u LMap", (uint)cx.table+1 );
    auto& p3 = *cx.phase3;

    char path[512];
    sprintf( path, DBG_BBCU_DBG_DIR "p3.lmap.t%u.tmp", (uint)cx.table+1 );

    const size_t writeSize = sizeof( LMap ) * BBCU_TABLE_ALLOC_ENTRY_COUNT;
    int err;
    FatalIf( !IOJob::WriteToFile( path, p3.hostLMap, writeSize, err ),
        "[DEBUG] Failed to write LMap with error: %d", err );

    sprintf( path, DBG_BBCU_DBG_DIR "p3.lmap.t%u.slices.tmp", (uint)cx.table+1 );
    FatalIf( !IOJob::WriteToFileUnaligned( path, p3.step3.prunedBucketSlices, sizeof( p3.step3.prunedBucketSlices ), err ),
        "[DEBUG] Failed to write LMap slices with error: %d", err );

    sprintf( path, DBG_BBCU_DBG_DIR "p3.lmap.t%u.buckets.tmp", (uint)cx.table+1 );
    FatalIf( !IOJob::WriteToFileUnaligned( path, p3.prunedBucketCounts[(int)cx.table], sizeof( uint32 ) * BBCU_BUCKET_COUNT, err ),
        "[DEBUG] Failed to write LMap buckets with error: %d", err );

    Log::Line( " [DEBUG] OK" );
}

//-----------------------------------------------------------
void DbgLoadLMap( CudaK32PlotContext& cx )
{
    auto& p3 = *cx.phase3;

    char path[512];
    sprintf( path, DBG_BBCU_DBG_DIR "p3.lmap.t%u.tmp", (uint)cx.table+1 );

    const size_t writeSize = sizeof( LMap ) * BBCU_TABLE_ALLOC_ENTRY_COUNT;
    int err;
    FatalIf( !IOJob::ReadFromFile( path, p3.hostLMap, writeSize, err ),
        "[DEBUG] Failed to read LMap with error: %d", err );

    sprintf( path, DBG_BBCU_DBG_DIR "p3.lmap.t%u.slices.tmp", (uint)cx.table+1 );
    FatalIf( !IOJob::ReadFromFileUnaligned( path, p3.step3.prunedBucketSlices, sizeof( p3.step3.prunedBucketSlices ), err ),
        "[DEBUG] Failed to read LMap slices with error: %d", err );

    sprintf( path, DBG_BBCU_DBG_DIR "p3.lmap.t%u.buckets.tmp", (uint)cx.table+1 );

    FatalIf( !IOJob::ReadFromFileUnaligned( path, p3.prunedBucketCounts[(int)cx.table], sizeof( uint32 ) * BBCU_BUCKET_COUNT, err ),
        "[DEBUG] Failed to read LMap buckets with error: %d", err );

    //DbgValidateLMapData( cx );
}

//-----------------------------------------------------------
void DbgValidateLMap( CudaK32PlotContext& cx )
{
    Log::Line( "[DEBUG] Validating LMap..." );

    ThreadPool& pool = DbgGetThreadPool( cx );

    auto& p3 = *cx.phase3;
    auto& s3 = p3.step3;

    LMap* lMap = bbcvirtallocbounded<LMap>( BBCU_BUCKET_ALLOC_ENTRY_COUNT );

    {
        // blake3_hasher hasher;
        // blake3_hasher_init( &hasher );

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            const LMap* reader = p3.hostLMap + bucket * P3_PRUNED_BUCKET_MAX;

            uint64 entryCount = 0;
            LMap*  writer     = lMap;

            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
            {
                // Read counts vertically, but read data horizontally
                const uint32 copyCount = s3.prunedBucketSlices[slice][bucket];

                bbmemcpy_t( writer, reader, copyCount );

                writer     += copyCount;
                entryCount += copyCount;
                reader     += P3_PRUNED_SLICE_MAX;
            }

            // All source entries should belong to the same bucket
            ASSERT( entryCount == p3.prunedBucketCounts[(int)cx.table][bucket] );

            for( uint64 i = 0; i < entryCount; i++ )
            {
                const LMap map = lMap[i];

                ASSERT( map.sourceIndex || map.sortedIndex );
                ASSERT( ( map.sourceIndex >> ( 32 - BBC_BUCKET_BITS ) ) == bucket );
            }

            // Hash bucket
            // blake3_hasher_update( &hasher, lMap, sizeof( LMap ) * entryCount );
        }

        // Print hash
        // DbgFinishAndPrintHash( hasher, "l_map", (uint)cx.table + 1 );
    }

    bbvirtfreebounded( lMap );

    Log::Line( "[DEBUG] LMap OK" );
}

//-----------------------------------------------------------
static void _DbgValidateLMapData( CudaK32PlotContext& cx );
void DbgValidateLMapData( CudaK32PlotContext& cx )
{
    // New stack (prevent overflow)
    auto* thread = new Thread();
    thread->Run( []( void* p ) {
        _DbgValidateLMapData( *(CudaK32PlotContext*)p );
    }, &cx );

    thread->WaitForExit();
    delete thread;
}

void _DbgValidateLMapData( CudaK32PlotContext& cx )
{
    Log::Line( "[DEBUG] Validating LMap uniquenes..." );

    ThreadPool& pool = DbgGetThreadPool( cx );

    auto& p3 = *cx.phase3;
    auto& s3 = p3.step3;

    uint32* srcIndices = bbcvirtallocbounded<uint32>( BBCU_TABLE_ENTRY_COUNT );
    uint32* dstIndices = bbcvirtallocbounded<uint32>( BBCU_TABLE_ENTRY_COUNT );
    uint32* tmpIndices = bbcvirtallocbounded<uint32>( BBCU_TABLE_ENTRY_COUNT );

    uint64 entryCount = 0;
    uint32 twoCount = 0;
    {
        uint32* srcWriter = srcIndices;
        uint32* dstWriter = dstIndices;

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            const LMap* reader = p3.hostLMap + bucket * P3_PRUNED_BUCKET_MAX;

            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
            {
                // Read counts vertically, but read data horizontally
                const uint32 copyCount = s3.prunedBucketSlices[slice][bucket];

                for( uint32 i = 0; i < copyCount; i++ )
                {
                    if( reader[i].sourceIndex == 2 )
                        twoCount++;
                    if( reader[i].sourceIndex == 0 && reader[i].sortedIndex == 0 )
                    {
                        ASSERT( 0 );
                    }

                    srcWriter[i] = reader[i].sourceIndex;
                    dstWriter[i] = reader[i].sortedIndex;
                }

                srcWriter += copyCount;
                dstWriter += copyCount;
                entryCount += copyCount;
                reader += P3_PRUNED_SLICE_MAX;
            }
        }

        ASSERT( entryCount == p3.prunedTableEntryCounts[(int)cx.table] );
    }

    RadixSort256::Sort<BB_MAX_JOBS>( pool, srcIndices, tmpIndices, entryCount );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, dstIndices, tmpIndices, entryCount );

    // Indices must not repeat:
    for( uint64 i = 1; i < entryCount; i++ )
    {
        ASSERT( srcIndices[i] > srcIndices[i-1] );
    }

    Log::Line( "Maximum source index: %u", srcIndices[entryCount-1] );

    for( uint64 i = 0; i < entryCount; i++ )
    {
        ASSERT( dstIndices[i] == i );
    }

    bbvirtfreebounded( srcIndices );
    bbvirtfreebounded( dstIndices );
    bbvirtfreebounded( tmpIndices );

    Log::Line( "[DEBUG] LMap uniqueness OK" );
}

#endif

