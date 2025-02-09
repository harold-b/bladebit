#include "CudaPlotPhase3Internal.h"
#include "CudaParkSerializer.h"


static void CompressInlinedTable( CudaK32PlotContext& cx );
static void Step1( CudaK32PlotContext& cx );

void CudaK32PlotPhase3Step2( CudaK32PlotContext& cx );
void CudaK32PlotPhase3Step3( CudaK32PlotContext& cx );
void WritePark7( CudaK32PlotContext& cx );


static void AllocXTableStep( CudaK32PlotContext& cx, CudaK32AllocContext& acx );
static void CudaK32PlotAllocateBuffersStep1( CudaK32PlotContext& cx, CudaK32AllocContext& acx );
static void CudaK32PlotAllocateBuffersStep2( CudaK32PlotContext& cx, CudaK32AllocContext& acx );
static void CudaK32PlotAllocateBuffersStep3( CudaK32PlotContext& cx, CudaK32AllocContext& acx );



#if _DEBUG
    static void DbgValidateRMap( CudaK32PlotContext& cx );
    static void DbgValidateIndices( CudaK32PlotContext& cx );
    void DbgLoadLMap( CudaK32PlotContext& cx );
    void DbgDumpSortedLinePoints( CudaK32PlotContext& cx );
#endif


//-----------------------------------------------------------
__global__ void CudaConvertInlinedXsToLinePoints( 
    const uint64 entryCount, const uint32 rOffset, const uint32 bucketShift,
    const Pair* inXs, const uint64* rMarks,
    uint64* outLPs, uint32* outIndices, uint32* gBucketCounts )
{
    const uint32 id     = threadIdx.x;
    const uint32 gid    = blockIdx.x * blockDim.x + id;
    const uint32 rIndex = rOffset + gid;

    __shared__ uint32 sharedBuckets[BBCU_BUCKET_COUNT];

    CUDA_ASSERT( gridDim.x >= BBCU_BUCKET_COUNT );
    if( id < BBCU_BUCKET_COUNT )
        sharedBuckets[id] = 0;

    __syncthreads();

    uint32 bucket;
    uint32 offset;
    uint64 lp;
    uint32 count = 0;

    const bool isPruned = gid >= entryCount || !CuBitFieldGet( rMarks, rIndex );
    if( !isPruned )
    {
        const Pair p = inXs[gid];
        CUDA_ASSERT( p.left || p.right );

        lp     = CudaSquareToLinePoint64( p.left, p.right );
        bucket = (uint32)(lp >> bucketShift);
        offset = atomicAdd( &sharedBuckets[bucket], 1 );

        count = 1;
    }
    __syncthreads();

    // Global offset
    if( id < BBCU_BUCKET_COUNT )
        sharedBuckets[id] = atomicAdd( &gBucketCounts[id], sharedBuckets[id] );
    __syncthreads();

    if( isPruned )
        return;

    const uint32 dst = bucket * P3_PRUNED_SLICE_MAX + sharedBuckets[bucket] + offset;

    CUDA_ASSERT( lp );
    // CUDA_ASSERT( outLPs[dst] == 0 );

    outLPs    [dst] = lp;
    outIndices[dst] = rIndex;
}

//-----------------------------------------------------------
__global__ void CudaTestPrune(
    const uint64 entryCount, const uint32 rOffset, const uint64* rTableMarks, uint32* gPrunedEntryCount )
{
    const uint32 id     = threadIdx.x;
    const uint32 gid    = blockIdx.x * blockDim.x + id;
    
    const uint32 count = ( gid >= entryCount || !CuBitFieldGet( rTableMarks, rOffset + gid ) ) ? 0 : 1;

    atomicAddShared( gPrunedEntryCount, count );
}

//-----------------------------------------------------------
__global__ void CudaConvertToLinePoints( 
    const uint64 entryCount, const uint32 rOffset, const uint32 lpBitSize,
    const uint32* lTable, const uint32* lPairs, const uint16* rPairs,
    const byte* marks, uint64* outLPs, uint32* gPrunedCount )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    if( gid == 0 )
        gPrunedCount = 0;

    // Filter-out entries that are not marked
    // if( !CuBitFieldGet( rMarks, rIndex ) )
    // {
        
    // }

    // Grab L table values
    const uint32 l = lPairs[gid];
    const uint32 r = l + rPairs[gid];

    const uint32 x = lTable[l];
    const uint32 y = lTable[r];

    // Convert to line point
    const uint64 lp  = CudaSquareToLinePoint64( x, y );

    const uint32 dst = atomicGlobalOffset( gPrunedCount );

    outLPs[dst] = lp;
}


//-----------------------------------------------------------
template<bool prune>
__global__ void PruneAndWriteRMap( 
    const uint32 entryCount, const uint64 rOffset,
    uint32* gBucketCounts, uint32* gPrunedEntryCount, RMap* gRMap,
    const uint32* lPairs, const uint16* rPairs, const uint64* rMarks )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    __shared__ uint32 sharedBuckets[BBCU_BUCKET_COUNT];

    CUDA_ASSERT( gridDim.x >= BBCU_BUCKET_COUNT );
    if( id < BBCU_BUCKET_COUNT )
        sharedBuckets[id] = 0;

    __syncthreads();

    // if( gid >= entryCount )
    //     return;

    const uint64 rIndex = rOffset + gid;

    bool isPruned = gid >= entryCount;

    if constexpr ( prune )
        isPruned = isPruned || !CuBitFieldGet( rMarks, rIndex );

    RMap entry;
    uint32 bucket, offset;

    if( !isPruned )
    {
        entry.dstL = lPairs[gid];
        entry.dstR = entry.dstL + rPairs[gid];
        entry.src  = (uint32)rIndex;   // It's original index

        bucket = (uint32)(entry.dstL >> (BBCU_K - BBC_BUCKET_BITS));

        // Block-level offset
        offset = atomicAdd( &sharedBuckets[bucket], 1 );
    }

    // Global offset
    __syncthreads();
    if( id < BBCU_BUCKET_COUNT )
        sharedBuckets[id] = atomicAdd( &gBucketCounts[id], sharedBuckets[id] );
    __syncthreads();

    if( isPruned )
        return;

    const uint32 dst = bucket * P3_PRUNED_SLICE_MAX + sharedBuckets[bucket] + offset;
    gRMap[dst] = entry;   
}


/**
 * #TODO: Optimize Steps 1 & 2 w/ packing.
 * Phase 3 works in 4 steps:
 * Step 1:
 *  - Prune table R and for each pair write a mapping
 *      at the back pointer locations, which points to the index of the pair.
 * 
 * Step 2:
 *  - Load the RMap
 *  - Load the LTable
 *  - Create line points given RMap with LTable values
 *  - Write line points to the their buckets along with the indices from the RMap
 * 
 * Step 3:
 *  - Load line points and index
 *  - Sort line points w/ index
 *  - Compress line points to park
 *  - Write parks
 *  - Write index as a map, this will be the next iteration's L table
*/
//-----------------------------------------------------------
void CudaK32PlotPhase3( CudaK32PlotContext& cx )
{
    // Set-up our context
    memset( cx.phase3->prunedBucketCounts    , 0, sizeof( cx.phase3->prunedBucketCounts ) );
    memset( cx.phase3->prunedTableEntryCounts, 0, sizeof( cx.phase3->prunedTableEntryCounts ) );

    InitFSEBitMask( cx );

#if _DEBUG
    //#define SKIP_TO_TABLE TableId::Table3
#endif

#if BBCU_DBG_SKIP_PHASE_2 && !defined( SKIP_TO_TABLE )
    DbgLoadMarks( cx );

    // if( cx.gCfg->compressionLevel > 0 )
    {
        DbgLoadTablePairs( cx, TableId::Table1 + (TableId)cx.gCfg->numDroppedTables + 2, false );
    }
#endif

    // Ensure the host buffers are not being used by the plot writer anymore
    #if !BBCU_DBG_SKIP_PHASE_1
    {
        Duration waitTime = Duration::zero();
        cx.plotFence->Wait( waitTime );
        cx.plotFence->Reset();

        if( TicksToSeconds( waitTime ) > 0.001 )
            Log::Line( "Waited %.2lf seconds for C tables to finish writing.", TicksToSeconds( waitTime ) );
    }
    #endif

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->phase3.rMapBuffer->Swap();
        cx.diskContext->phase3.indexBuffer->Swap();
        cx.diskContext->phase3.lpAndLMapBuffer->Swap();
    }


    const uint32 compressionLevel = cx.gCfg->compressionLevel;

    // Special case with the starting table, since it has the values inlined already
    cx.table = TableId::Table2 + cx.gCfg->numDroppedTables;

    // if( compressionLevel == 0 )
    {
        Log::Line( "Compressing Table %u and %u...", cx.table, cx.table+1 );

        auto tableTimer = TimerBegin();

        auto timer = tableTimer;
        CompressInlinedTable( cx );
        auto elapsed = TimerEnd( timer );
        Log::Line( " Step 1 completed step in %.2lf seconds.", elapsed );

        timer = TimerBegin();
        CudaK32PlotPhase3Step3( cx );

        auto tableElapsed = TimerEnd( tableTimer );
        elapsed = TimerEnd( timer );
        Log::Line( " Step 2 completed step in %.2lf seconds.", elapsed );

        const uint64 baseEntryCount   = cx.tableEntryCounts[(int)cx.table];
        const uint64 prunedEntryCount = cx.phase3->prunedTableEntryCounts[(int)cx.table];
        Log::Line( "Completed table %u in %.2lf seconds with %llu / %llu entries ( %.2lf%% ).",
            cx.table, tableElapsed, prunedEntryCount, baseEntryCount, (prunedEntryCount / (double)baseEntryCount) * 100.0 );

    }
    // else if( compressionLevel > 0 )
    // {
    //     const TableId startLTable = TableId::Table1 + (TableId)cx.gCfg->numDroppedTables;
    //     cx.phase3->prunedTableEntryCounts[(int)startLTable] = cx.tableEntryCounts[(int)startLTable];
    //     if( cx.gCfg->numDroppedTables > 1 )
    //         cx.table = TableId::Table3;
    // }

#ifdef SKIP_TO_TABLE
    cx.table = SKIP_TO_TABLE;
    DbgLoadLMap( cx );
#endif

    auto& p3 = *cx.phase3;
    const TableId startRTable = cx.table + 1;

    for( TableId rTable = startRTable; rTable <= TableId::Table7; rTable++ )
    {
        Log::Line( "Compressing tables %u and %u...", (uint)rTable, (uint)rTable+1 );

        cx.table = rTable;

        #if BBCU_DBG_SKIP_PHASE_2
            if( rTable < TableId::Table7 )
                DbgLoadTablePairs( cx, rTable+1, false );
        #endif

        auto tableTimer = TimerBegin();

        // Step 1
        auto timer = tableTimer;
        Step1( cx );
        double elapsed = TimerEnd( timer );
        Log::Line( " Step 1 completed step in %.2lf seconds.", elapsed );

        // Step 2
        timer = TimerBegin();
        CudaK32PlotPhase3Step2( cx );
        elapsed = TimerEnd( timer );
        Log::Line( " Step 2 completed step in %.2lf seconds.", elapsed );

        // Step 3
        timer = TimerBegin();
        CudaK32PlotPhase3Step3( cx );
        elapsed = TimerEnd( timer );
        Log::Line( " Step 3 completed step in %.2lf seconds.", elapsed );

        auto tableElapsed = TimerEnd( tableTimer );

        const uint64 baseEntryCount   = cx.tableEntryCounts[(int)rTable];
        const uint64 prunedEntryCount = p3.prunedTableEntryCounts[(int)rTable];
        Log::Line( "Completed table %u in %.2lf seconds with %llu / %llu entries ( %.2lf%% ).",
            rTable, tableElapsed, prunedEntryCount, baseEntryCount, (prunedEntryCount / (double)baseEntryCount) * 100.0 );
    }

    // Park 7
    {
        Log::Line( "Serializing P7 entries" );

        const auto timer = TimerBegin();
        WritePark7( cx );
        const auto elapsed = TimerEnd( timer );
        Log::Line( "Completed serializing P7 entries in %.2lf seconds.", elapsed );
    }
}

//-----------------------------------------------------------
void Step1( CudaK32PlotContext& cx )
{
    auto LoadBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void
    {
        const TableId rTable = cx.table;
        auto&         p3     = *cx.phase3;
        auto&         s1     = p3.step1;

        if( bucket == 0 && cx.cfg.hybrid128Mode )
        {
            cx.diskContext->tablesL[(int)rTable]->Swap();
            cx.diskContext->tablesR[(int)rTable]->Swap();

            s1.pairsLIn.AssignDiskBuffer( cx.diskContext->tablesL[(int)rTable] );
            s1.pairsRIn.AssignDiskBuffer( cx.diskContext->tablesR[(int)rTable] );
        }

        const uint32 entryCount = cx.bucketCounts[(int)rTable][bucket]; //BBCU_BUCKET_ENTRY_COUNT;

        uint32* hostPairsL = cx.hostBackPointers[(int)rTable].left  + p3.pairsLoadOffset;
        uint16* hostPairsR = cx.hostBackPointers[(int)rTable].right + p3.pairsLoadOffset;

        s1.pairsLIn.UploadT( hostPairsL, entryCount );
        s1.pairsRIn.UploadT( hostPairsR, entryCount );

        p3.pairsLoadOffset += entryCount;
    };

    auto& p2 = *cx.phase2;
    auto& p3 = *cx.phase3;
    auto& s1 = p3.step1;

    const TableId rTable = cx.table;

    // Clear pruned table count
    Log::Line( "Marker Set to %d", 12)
CudaErrCheck( cudaMemsetAsync( p3.devPrunedEntryCount, 0, sizeof( uint32 ), cx.computeStream ) );

    // Load marking table (must be loaded before first bucket, on the same stream)
    if( cx.table < TableId::Table7 )
    {
        Log::Line( "Marker Set to %d", 13)
CudaErrCheck( cudaMemcpyAsync( s1.rTableMarks, cx.hostMarkingTables[(int)rTable],
                        GetMarkingTableBitFieldSize(), cudaMemcpyHostToDevice, s1.pairsLIn.GetQueue()->GetStream() ) );
    }

    // Load initial bucket
    p3.pairsLoadOffset = 0;
    LoadBucket( cx, 0 );

    ///
    /// Process buckets
    ///
    const uint32 threadPerBlock = 256;
    const uint32 blocksPerGrid  = CDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, (int)threadPerBlock ); 

    uint64 rTableOffset = 0;
    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        cx.bucket = bucket;

        if( bucket + 1 < BBCU_BUCKET_COUNT )
            LoadBucket( cx, bucket + 1 );

        // Wait for R table pairs to be ready
        const uint32* devLPairs = (uint32*)s1.pairsLIn.GetUploadedDeviceBuffer( cx.computeStream );
        const uint16* devRPairs = (uint16*)s1.pairsRIn.GetUploadedDeviceBuffer( cx.computeStream );

        const uint32 entryCount = cx.bucketCounts[(int)rTable][bucket];// bucket == BBCU_BUCKET_COUNT-1 ?
                                //   ( cx.tableEntryCounts[(int)rTable] - (BBCU_BUCKET_ENTRY_COUNT * (BBCU_BUCKET_COUNT-1)) ) :    // Get only the remaining entries for the last bucket
                                //   BBCU_BUCKET_ENTRY_COUNT;                                                                      // Otherwise, use a whole bucket's worth.

        auto* devRMap = (RMap*)s1.rMapOut.LockDeviceBuffer( cx.computeStream );

        uint32* devSliceCounts = cx.devSliceCounts + bucket * BBCU_BUCKET_COUNT;

        // Generate map
        #define KERN_RMAP_ARGS entryCount, rTableOffset, devSliceCounts, p3.devPrunedEntryCount, devRMap, devLPairs, devRPairs, s1.rTableMarks

        Log::Line( "Marker Set to %d", 14)
CudaErrCheck( cudaMemsetAsync( devSliceCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT, cx.computeStream ) );

        if( cx.table < TableId::Table7 )
            PruneAndWriteRMap<true><<<blocksPerGrid, threadPerBlock, 0, cx.computeStream>>>( KERN_RMAP_ARGS );
        else
            PruneAndWriteRMap<false><<<blocksPerGrid, threadPerBlock, 0, cx.computeStream>>>( KERN_RMAP_ARGS );

        #undef KERN_RMAP_ARGS
        s1.pairsLIn.ReleaseDeviceBuffer( cx.computeStream );
        s1.pairsRIn.ReleaseDeviceBuffer( cx.computeStream );
        rTableOffset += entryCount;

        // Download data (Vertical download (write 1 column))
        s1.rMapOut.Download2DT<RMap>( p3.hostRMap + (size_t)bucket * P3_PRUNED_SLICE_MAX,
            P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, P3_PRUNED_BUCKET_MAX, P3_PRUNED_SLICE_MAX, cx.computeStream );
    }

    // Download slice counts
    cudaStream_t downloadStream = s1.rMapOut.GetQueue()->GetStream();

    Log::Line( "Marker Set to %d", 15)
CudaErrCheck( cudaMemcpyAsync( cx.hostBucketSlices, cx.devSliceCounts, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT,
                    cudaMemcpyDeviceToHost, downloadStream ) );

    // Wait for completion
    s1.rMapOut.WaitForCompletion();
    s1.rMapOut.Reset();

    s1.pairsLIn.Reset();
    s1.pairsRIn.Reset();

    Log::Line( "Marker Set to %d", 16)
CudaErrCheck( cudaStreamSynchronize( downloadStream ) );

    // Add-up pruned bucket counts and tables counts
    memcpy( &s1.prunedBucketSlices[0][0], cx.hostBucketSlices, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT );
    {
        uint32* hostSliceCounts = cx.hostBucketSlices;

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
                p3.prunedBucketCounts[(int)rTable][bucket] += s1.prunedBucketSlices[slice][bucket];

            // hostSliceCounts += BBCU_BUCKET_COUNT;
        }

        p3.prunedTableEntryCounts[(int)rTable] = 0;

        for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
            p3.prunedTableEntryCounts[(int)rTable] += p3.prunedBucketCounts[(int)rTable][i];
    }

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->phase3.rMapBuffer->Swap();
    }

    // #if _DEBUG
    //     DbgValidateRMap( cx );
    // #endif
}

//-----------------------------------------------------------
// Table 2 (or 3,4,etc., depending on comrpession level), already has
// the x values inlined into the pairs. Therefore, we can skip step 1
// and go directly into converting to line point, then sorting it to the target 
//-----------------------------------------------------------
void CompressInlinedTable( CudaK32PlotContext& cx )
{
    auto LoadBucket = []( CudaK32PlotContext& cx, const uint32 bucket ) -> void {

        auto& p3 = *cx.phase3;
        auto& tx = p3.xTable;

        // Load inlined x's
        const TableId rTable     = TableId::Table2 + (TableId)cx.gCfg->numDroppedTables;
        const uint32  entryCount = cx.bucketCounts[(int)rTable][bucket];

        if( bucket == 0 )
        {
            p3.pairsLoadOffset = 0;

            if( cx.cfg.hybrid128Mode )
            {
                cx.diskContext->tablesL[(int)rTable]->Swap();
                tx.xIn.AssignDiskBuffer( cx.diskContext->tablesL[(int)rTable] );
            }
        }

        const Pair* inlinedXs = ((Pair*)cx.hostBackPointers[(int)rTable].left) + p3.pairsLoadOffset;

        tx.xIn.UploadT( inlinedXs, entryCount, cx.computeStream );

        p3.pairsLoadOffset += entryCount;
    };

    const TableId rTable = TableId::Table2 + (TableId)cx.gCfg->numDroppedTables;
    auto& p3 = *cx.phase3;
    auto& tx = p3.xTable;
    auto& s2 = p3.step2;

    #if BBCU_DBG_SKIP_PHASE_2
        DbgLoadTablePairs( cx, rTable );
    #endif

    // Load R Marking table (must be loaded before first bucket, on the same stream)
    Log::Line( "Marker Set to %d", 17)
CudaErrCheck( cudaMemcpyAsync( (void*)tx.devRMarks, cx.hostMarkingTables[(int)rTable],
                GetMarkingTableBitFieldSize(), cudaMemcpyHostToDevice, p3.xTable.xIn.GetQueue()->GetStream() ) );

    // Load initial bucket
    LoadBucket( cx, 0 );

    const bool   isCompressed     = cx.gCfg->compressionLevel > 0;
    const uint32 compressedLPBits = isCompressed ? GetCompressedLPBitCount( cx.gCfg->compressionLevel ) : 0;

    const uint32 lpBits           = isCompressed ? compressedLPBits : BBCU_K * 2 - 1;
    const uint32 lpBucketShift    = lpBits - BBC_BUCKET_BITS;

    uint64 tablePrunedEntryCount = 0;
    uint32 rTableOffset          = 0;

    Log::Line( "Marker Set to %d", 18)
CudaErrCheck( cudaMemsetAsync( cx.devSliceCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, cx.computeStream ) );

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        cx.bucket = bucket;

        if( bucket + 1 < BBCU_BUCKET_COUNT )
            LoadBucket( cx, bucket + 1 );

        // Wait for pairs to be ready
        const Pair* devXs = (Pair*)tx.xIn.GetUploadedDeviceBuffer( cx.computeStream );

        uint64* outLps     = (uint64*)tx.lpOut   .LockDeviceBuffer( cx.computeStream );
        uint32* outIndices = (uint32*)tx.indexOut.LockDeviceBuffer( cx.computeStream );

        const uint32 entryCount     = cx.bucketCounts[(int)rTable][bucket];

        const uint32 threadPerBlock = 256;
        const uint32 blocksPerGrid  = CDiv( entryCount, (int)threadPerBlock ); 

        uint32* devSliceCounts = cx.devSliceCounts + bucket * BBCU_BUCKET_COUNT;

        #if _DEBUG
            Log::Line( "Marker Set to %d", 19)
CudaErrCheck( cudaMemsetAsync( outLps, 0, sizeof( uint64 ) * P3_PRUNED_BUCKET_MAX, cx.computeStream ) );
        #endif

        CudaConvertInlinedXsToLinePoints<<<blocksPerGrid, threadPerBlock, 0, cx.computeStream>>>(
            entryCount, rTableOffset, lpBucketShift,
            devXs, tx.devRMarks, outLps, outIndices, devSliceCounts );

        tx.xIn.ReleaseDeviceBuffer( cx.computeStream );

        // Download output
        // Horizontal download (write 1 row)
        tx.lpOut   .Download2DT<uint64>( p3.hostLinePoints + (size_t)bucket * P3_PRUNED_BUCKET_MAX  , P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX    , P3_PRUNED_SLICE_MAX, cx.computeStream );
        tx.indexOut.Download2DT<uint32>( p3.hostIndices    + (size_t)bucket * P3_PRUNED_BUCKET_MAX*3, P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX * 3, P3_PRUNED_SLICE_MAX, cx.computeStream );

        rTableOffset += entryCount;
    }

    cudaStream_t downloadStream = tx.lpOut.GetQueue()->GetStream();

    CudaErrCheck( cudaMemcpyAsync( cx.hostBucketSlices, cx.devSliceCounts, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, 
                    cudaMemcpyDeviceToHost, downloadStream ) );

    tx.lpOut   .WaitForCompletion();
    tx.indexOut.WaitForCompletion();
    tx.lpOut   .Reset();
    tx.indexOut.Reset();

    CudaErrCheck( cudaStreamSynchronize( downloadStream ) );

    #if _DEBUG
        for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
        {
            ASSERT( p3.prunedBucketCounts[(int)rTable][i] <= P3_PRUNED_BUCKET_MAX );
        }
    #endif

    // Add-up pruned bucket counts and tables counts
    {
        bbmemcpy_t( &s2.prunedBucketSlices[0][0], cx.hostBucketSlices, BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT );

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
                p3.prunedBucketCounts[(int)rTable][bucket] += s2.prunedBucketSlices[slice][bucket];
        }

        p3.prunedTableEntryCounts[(int)rTable] = 0;

        for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
            p3.prunedTableEntryCounts[(int)rTable] += p3.prunedBucketCounts[(int)rTable][i];
    }

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->phase3.lpAndLMapBuffer->Swap();
        cx.diskContext->phase3.indexBuffer->Swap();
    }

// #if _DEBUG
//     DbgValidateIndices( cx );
//     // DbgValidateStep2Output( cx );
//     // DbgDumpSortedLinePoints( cx );
// #endif
}


///
/// Allocation
///
//-----------------------------------------------------------
void CudaK32PlotPhase3AllocateBuffers( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    static_assert( sizeof( LMap ) == sizeof( uint64 ) );

    auto& p3 = *cx.phase3;

    // Shared allocations
    p3.devBucketCounts     = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_COUNT, acx.alignment );
    p3.devPrunedEntryCount = acx.devAllocator->CAlloc<uint32>( 1, acx.alignment );

    // Host allocations
    if( !cx.cfg.hybrid16Mode )
    {
        p3.hostRMap       = acx.hostTempAllocator->CAlloc<RMap>( BBCU_TABLE_ALLOC_ENTRY_COUNT );     // Used for rMap and index
        p3.hostLinePoints = acx.hostTempAllocator->CAlloc<uint64>( BBCU_TABLE_ALLOC_ENTRY_COUNT );   // Used for lMap and LPs
    }
    else if( !cx.diskContext->phase3.rMapBuffer )
    {
        const size_t RMAP_SLICE_SIZE        = sizeof( RMap )   * P3_PRUNED_SLICE_MAX;
        const size_t INDEX_SLICE_SIZE       = sizeof( uint32 ) * P3_PRUNED_SLICE_MAX;
        const size_t LP_AND_LMAP_SLICE_SIZE = sizeof( uint64 ) * P3_PRUNED_SLICE_MAX;

        const FileFlags TMP2_QUEUE_FILE_FLAGS = cx.cfg.temp2DirectIO ? FileFlags::NoBuffering | FileFlags::LargeFile : FileFlags::LargeFile;

        cx.diskContext->phase3.rMapBuffer = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, CudaK32HybridMode::P3_RMAP_DISK_BUFFER_FILE_NAME.data(), 
                                            BBCU_BUCKET_COUNT, RMAP_SLICE_SIZE, FileMode::OpenOrCreate, FileAccess::ReadWrite, TMP2_QUEUE_FILE_FLAGS );
        FatalIf( !cx.diskContext->phase3.rMapBuffer, "Failed to create R Map disk buffer." );

        cx.diskContext->phase3.indexBuffer = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, CudaK32HybridMode::P3_INDEX_DISK_BUFFER_FILE_NAME.data(), 
                                            BBCU_BUCKET_COUNT, INDEX_SLICE_SIZE, FileMode::OpenOrCreate, FileAccess::ReadWrite, TMP2_QUEUE_FILE_FLAGS );
        FatalIf( !cx.diskContext->phase3.indexBuffer, "Failed to create index disk buffer." );

        cx.diskContext->phase3.lpAndLMapBuffer = DiskBucketBuffer::Create( *cx.diskContext->temp2Queue, CudaK32HybridMode::P3_LP_AND_LMAP_DISK_BUFFER_FILE_NAME.data(), 
                                            BBCU_BUCKET_COUNT, RMAP_SLICE_SIZE, FileMode::OpenOrCreate, FileAccess::ReadWrite, TMP2_QUEUE_FILE_FLAGS );
        FatalIf( !cx.diskContext->phase3.lpAndLMapBuffer, "Failed to create LP/LMap disk buffer." );
    }

    #if _DEBUG
        if( !acx.dryRun && !cx.cfg.hybrid128Mode )
        {
            ASSERT( (uintptr_t)(p3.hostLinePoints + BBCU_TABLE_ALLOC_ENTRY_COUNT ) <= (uintptr_t)cx.hostTableL );
        }
    #endif

    if( acx.dryRun )
    {
        CudaK32AllocContext dacx = acx;

        DummyAllocator devAlloc    = {};
        DummyAllocator pinnedAlloc = {};

        dacx.devAllocator     = &devAlloc;
        dacx.pinnedAllocator  = &pinnedAlloc;

        AllocXTableStep( cx, dacx );

        size_t sharedDevSize    = devAlloc.Size();
        size_t sharedPinnedSize = pinnedAlloc.Size();

        devAlloc    = {};
        pinnedAlloc = {};
        CudaK32PlotAllocateBuffersStep1( cx, dacx );

        sharedDevSize    = std::max( sharedDevSize   , devAlloc.Size()    );
        sharedPinnedSize = std::max( sharedPinnedSize, pinnedAlloc.Size() );
        devAlloc         = {};
        pinnedAlloc      = {};
        CudaK32PlotAllocateBuffersStep2( cx, dacx );

        sharedDevSize    = std::max( sharedDevSize   , devAlloc.Size()    );
        sharedPinnedSize = std::max( sharedPinnedSize, pinnedAlloc.Size() );
        devAlloc         = {};
        pinnedAlloc      = {};
        CudaK32PlotAllocateBuffersStep3( cx, dacx );

        sharedDevSize    = std::max( sharedDevSize   , devAlloc.Size()    );
        sharedPinnedSize = std::max( sharedPinnedSize, pinnedAlloc.Size() );

        acx.devAllocator   ->Alloc( sharedDevSize   , acx.alignment );
        acx.pinnedAllocator->Alloc( sharedPinnedSize, acx.alignment );
    }
    else
    {
        StackAllocator* devAllocator    = (StackAllocator*)acx.devAllocator;
        StackAllocator* pinnedAllocator = (StackAllocator*)acx.pinnedAllocator;

        const size_t devMarker = devAllocator   ->Size();
        const size_t pinMarker = pinnedAllocator->Size();

        AllocXTableStep( cx, acx );
        devAllocator   ->PopToMarker( devMarker );
        pinnedAllocator->PopToMarker( pinMarker );

        CudaK32PlotAllocateBuffersStep1( cx, acx );
        devAllocator   ->PopToMarker( devMarker );
        pinnedAllocator->PopToMarker( pinMarker );

        CudaK32PlotAllocateBuffersStep2( cx, acx );
        devAllocator   ->PopToMarker( devMarker );
        pinnedAllocator->PopToMarker( pinMarker );

        CudaK32PlotAllocateBuffersStep3( cx, acx );
    }
}

//-----------------------------------------------------------
void AllocXTableStep( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    GpuStreamDescriptor desc{};
    desc.entriesPerSlice = BBCU_MAX_SLICE_ENTRY_COUNT;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = acx.alignment;
    desc.bufferCount     = BBCU_DEFAULT_GPU_BUFFER_COUNT;
    desc.deviceAllocator = acx.devAllocator;
    desc.pinnedAllocator = nullptr;

    GpuStreamDescriptor uploadDesc = desc;
    if( cx.cfg.hybrid128Mode )
    {
        uploadDesc.pinnedAllocator = acx.pinnedAllocator;

        if( cx.cfg.hybrid16Mode )
            desc.pinnedAllocator = acx.pinnedAllocator;
    }

    auto& tx = cx.phase3->xTable;

    tx.devRMarks = (uint64*)acx.devAllocator->AllocT<uint64>( GetMarkingTableBitFieldSize(), acx.alignment );

    tx.xIn       = cx.gpuUploadStream[0]->CreateUploadBufferT<Pair>( uploadDesc, acx.dryRun );
    tx.lpOut     = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint64>( desc, acx.dryRun );
    tx.indexOut  = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint32>( desc, acx.dryRun );

    if( !acx.dryRun && cx.cfg.hybrid16Mode )
    {
        tx.lpOut   .AssignDiskBuffer( cx.diskContext->phase3.lpAndLMapBuffer );
        tx.indexOut.AssignDiskBuffer( cx.diskContext->phase3.indexBuffer );
    }
}

//-----------------------------------------------------------
void CudaK32PlotAllocateBuffersStep1( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    GpuStreamDescriptor desc{};
    desc.entriesPerSlice = BBCU_MAX_SLICE_ENTRY_COUNT;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = acx.alignment;
    desc.bufferCount     = BBCU_DEFAULT_GPU_BUFFER_COUNT;
    desc.deviceAllocator = acx.devAllocator;
    desc.pinnedAllocator = nullptr;

    GpuStreamDescriptor uploadDesc = desc;
    if( cx.cfg.hybrid128Mode )
    {
        uploadDesc.pinnedAllocator = acx.pinnedAllocator;

        if( cx.cfg.hybrid16Mode )
            desc.pinnedAllocator = acx.pinnedAllocator;
    }

    auto&        s1        = cx.phase3->step1;
    const size_t alignment = acx.alignment;

    s1.pairsLIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint32>( uploadDesc,  acx.dryRun );
    s1.pairsRIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint16>( uploadDesc,  acx.dryRun );
    s1.rMapOut  = cx.gpuDownloadStream[0]->CreateDownloadBufferT<RMap>( desc, acx.dryRun );

    s1.rTableMarks = (uint64*)acx.devAllocator->AllocT<uint64>( GetMarkingTableBitFieldSize(), acx.alignment );

    if( !acx.dryRun && cx.cfg.hybrid16Mode )
    {
        s1.rMapOut.AssignDiskBuffer( cx.diskContext->phase3.rMapBuffer );
    }
}

//-----------------------------------------------------------
void CudaK32PlotAllocateBuffersStep2( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    GpuStreamDescriptor desc{};
    desc.entriesPerSlice = BBCU_MAX_SLICE_ENTRY_COUNT;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = acx.alignment;
    desc.bufferCount     = BBCU_DEFAULT_GPU_BUFFER_COUNT;
    desc.deviceAllocator = acx.devAllocator;
    desc.pinnedAllocator = nullptr;

    GpuStreamDescriptor uploadDesc = desc;
    if( cx.cfg.hybrid16Mode )
    {
        desc.pinnedAllocator = acx.pinnedAllocator;
    }

    auto&        s2        = cx.phase3->step2;
    const size_t alignment = acx.alignment;

    s2.rMapIn = cx.gpuUploadStream[0]->CreateUploadBufferT<RMap>( desc, acx.dryRun );
    s2.lMapIn = cx.gpuUploadStream[0]->CreateUploadBufferT<LMap>( desc, acx.dryRun );

    s2.lpOut    = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint64>( desc, acx.dryRun );
    s2.indexOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint32> (desc, acx.dryRun );


    const size_t devParkAllocSize = P3_PARK_7_SIZE * P3_MAX_P7_PARKS_PER_BUCKET;

    GpuStreamDescriptor parksDesc = desc;
    parksDesc.sliceCount      = 1;
    parksDesc.entriesPerSlice = devParkAllocSize;
    parksDesc.sliceAlignment  = RoundUpToNextBoundaryT<size_t>( P3_PARK_7_SIZE, sizeof( uint64 ) );

    s2.parksOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<byte>( parksDesc, acx.dryRun );

    s2.devLTable[0] = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );
    s2.devLTable[1] = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );

    if( !acx.dryRun && cx.cfg.hybrid16Mode )
    {
        s2.rMapIn.AssignDiskBuffer( cx.diskContext->phase3.rMapBuffer );
        s2.lMapIn.AssignDiskBuffer( cx.diskContext->phase3.lpAndLMapBuffer );

        s2.lpOut   .AssignDiskBuffer( cx.diskContext->phase3.lpAndLMapBuffer );
        s2.indexOut.AssignDiskBuffer( cx.diskContext->phase3.indexBuffer );
    }
}

//-----------------------------------------------------------
void CudaK32PlotAllocateBuffersStep3( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    GpuStreamDescriptor desc{};
    desc.entriesPerSlice = BBCU_MAX_SLICE_ENTRY_COUNT;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = acx.alignment;
    desc.bufferCount     = BBCU_DEFAULT_GPU_BUFFER_COUNT;
    desc.deviceAllocator = acx.devAllocator;
    desc.pinnedAllocator = nullptr;

    if( cx.cfg.hybrid16Mode )
    {
        desc.pinnedAllocator = acx.pinnedAllocator;
    }

    auto&        s3        = cx.phase3->step3;
    const size_t alignment = acx.alignment;

    s3.hostParkOverrunCount = acx.pinnedAllocator->CAlloc<uint32>( 1 );

    s3.lpIn    = cx.gpuUploadStream[0]->CreateUploadBufferT<uint64>( desc, acx.dryRun );
    s3.indexIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint32>( desc, acx.dryRun );

    s3.mapOut  = cx.gpuDownloadStream[0]->CreateDownloadBufferT<uint64>( desc, acx.dryRun );

    const size_t devParkAllocSize = DEV_MAX_PARK_SIZE * P3_PRUNED_MAX_PARKS_PER_BUCKET;

    GpuStreamDescriptor parksDesc = desc;
    parksDesc.sliceCount      = 1;
    parksDesc.entriesPerSlice = devParkAllocSize;
    parksDesc.sliceAlignment  = RoundUpToNextBoundaryT<size_t>( DEV_MAX_PARK_SIZE, sizeof( uint64 ) );

    s3.parksOut = cx.gpuDownloadStream[0]->CreateDownloadBufferT<byte>( parksDesc, acx.dryRun );

    if( acx.dryRun )
    {
        s3.sizeTmpSort = 0;
        cub::DeviceRadixSort::SortPairs<uint64, uint32>( nullptr, s3.sizeTmpSort, nullptr, nullptr, nullptr, nullptr, BBCU_BUCKET_ALLOC_ENTRY_COUNT );
    }

    s3.devSortTmpData = acx.devAllocator->AllocT<byte>( s3.sizeTmpSort, alignment );


    // Allocate 1 more park's worth of line points so we can have space to retain the line points
    // that did not make it into a park for the next bucket.
    const size_t linePointAllocCount = P3_PRUNED_MAX_PARKS_PER_BUCKET * (size_t)kEntriesPerPark;
    static_assert( linePointAllocCount > BBCU_BUCKET_ALLOC_ENTRY_COUNT );

    s3.devLinePoints      = acx.devAllocator->CAlloc<uint64>( linePointAllocCount, alignment );
    s3.devDeltaLinePoints = acx.devAllocator->CAlloc<uint64>( linePointAllocCount, alignment );
    s3.devIndices         = acx.devAllocator->CAlloc<uint32>( BBCU_BUCKET_ALLOC_ENTRY_COUNT, alignment );

    s3.devCTable           = acx.devAllocator->AllocT<FSE_CTable>( P3_MAX_CTABLE_SIZE, alignment );
    s3.devParkOverrunCount = acx.devAllocator->CAlloc<uint32>( 1 );

    if( !acx.dryRun && cx.cfg.hybrid16Mode )
    {
        s3.lpIn   .AssignDiskBuffer( cx.diskContext->phase3.lpAndLMapBuffer );
        s3.indexIn.AssignDiskBuffer( cx.diskContext->phase3.indexBuffer );

        s3.mapOut.AssignDiskBuffer( cx.diskContext->phase3.lpAndLMapBuffer );
    }
}



#if _DEBUG

//-----------------------------------------------------------
__global__ static void DbgCudaValidateRMap( const uint64 entryCount, const uint32 lTableOffset, const RMap* rmap )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    if( gid >= entryCount )
        return;

    
    const RMap map = rmap[gid];

    const uint32 left  = map.dstL - lTableOffset;
    const uint32 right = map.dstR - lTableOffset;

    // if( left >= BBCU_BUCKET_ALLOC_ENTRY_COUNT )
    if( left >= right || left >= BBCU_BUCKET_ALLOC_ENTRY_COUNT || right >= BBCU_BUCKET_ALLOC_ENTRY_COUNT )
    {
        printf( "gid: %u | left: %u | right: %u | loffset: %u\n"
            " dstL: %u | dstR: %u | src: %u\n",
        gid, left, right, lTableOffset, map.dstL, map.dstR, map.src );
        CUDA_ASSERT( false );
    }

    CUDA_ASSERT( left  < BBCU_BUCKET_ALLOC_ENTRY_COUNT );
    CUDA_ASSERT( right < BBCU_BUCKET_ALLOC_ENTRY_COUNT );
    CUDA_ASSERT( left < right );
}

//-----------------------------------------------------------
void DbgValidateRMap( CudaK32PlotContext& cx )
{
    Log::Line( "[DEBUG] Validating RMap..." );

    auto& p3 = *cx.phase3;
    auto& s1 = p3.step1;
 
    {
        ThreadPool& pool = DbgGetThreadPool( cx );

        RMap* rMap = bbcvirtallocbounded<RMap>( BBCU_BUCKET_ALLOC_ENTRY_COUNT );

        // blake3_hasher hasher;
        // blake3_hasher_init( &hasher );

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            const RMap* reader = p3.hostRMap + bucket * P3_PRUNED_BUCKET_MAX;
            RMap* writer = rMap;

            uint32 entryCount = 0;

            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
            {
                const uint32 copyCount = s1.prunedBucketSlices[slice][bucket];
                bbmemcpy_t( writer, reader, copyCount );

                writer     += copyCount;
                entryCount += copyCount;

                reader += P3_PRUNED_SLICE_MAX;
            }

            // Validate bucket
            const uint32 bucketOffset = bucket * BBCU_BUCKET_ENTRY_COUNT;
            for( uint32 i = 0; i < entryCount; i++ )
            {
                const RMap map = rMap[i];
                ASSERT( map.dstL || map.dstR );
                ASSERT( map.dstR - map.dstL < 0x10000u );
                ASSERT( map.dstL >> ( 32 - BBC_BUCKET_BITS ) == bucket );

                const uint32 left  = map.dstL - bucketOffset;
                const uint32 right = map.dstR - bucketOffset;
                ASSERT( left  < BBCU_BUCKET_ALLOC_ENTRY_COUNT );
                ASSERT( right < BBCU_BUCKET_ALLOC_ENTRY_COUNT );
                ASSERT( left < right );
            }

            // Hash bucket
            // blake3_hasher_update( &hasher, rMap, sizeof( RMap ) * entryCount );
        }

        // Print hash
        // DbgFinishAndPrintHash( hasher, "r_map", (uint)cx.table + 1 );

        bbvirtfreebounded( rMap );
        Log::Line( " [DEBUG] CPU OK" );
    }

    // Validate in CUDA
    {
        uint64 pairsLoadOffset = 0;
        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            uint64 entryCount = 0;
            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
            {
                const uint32 copyCount = s1.prunedBucketSlices[slice][bucket];
                entryCount += copyCount;
            }

            const RMap*   rmap         = p3.hostRMap + (size_t)bucket * P3_PRUNED_BUCKET_MAX;
            const uint32* rSliceCounts = &p3.step1.prunedBucketSlices[0][bucket];
            
            p3.step2.rMapIn.UploadArrayT<RMap>( rmap, BBCU_BUCKET_COUNT, P3_PRUNED_SLICE_MAX, BBCU_BUCKET_COUNT, rSliceCounts );

            const uint32 rEntryCount = p3.prunedBucketCounts[(int)cx.table][bucket];
                  RMap*  devRMap     = p3.step2.rMapIn.GetUploadedDeviceBufferT<RMap>( cx.computeStream );

            ASSERT( entryCount == rEntryCount );

            const uint32 threads = 256;
            const uint32 blocks  = CDiv( rEntryCount, threads );

            const uint32 lTableOffset = bucket * BBCU_BUCKET_ENTRY_COUNT;

            DbgCudaValidateRMap<<<blocks, threads, 0, cx.computeStream>>>( rEntryCount, lTableOffset, devRMap );
            CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );

            p3.step2.rMapIn.ReleaseDeviceBuffer( cx.computeStream );
        }
        Log::Line( " [DEBUG] CUDA OK" );

        p3.step2.lMapIn.Reset();
    }

    Log::Line( "[DEBUG] RMap validation OK" );
}

//-----------------------------------------------------------
void DbgValidateIndices( CudaK32PlotContext& cx )
{
    // Ensure all origin output indices are not repeated and well distributed
    Log::Line( "[DEBUG] Validating indices..." );

    auto& p3 = *cx.phase3;
    auto& s2 = p3.step2;
    
    ThreadPool& pool  = DbgGetThreadPool( cx );

    uint32* indices   = bbcvirtallocbounded<uint32>( BBCU_TABLE_ENTRY_COUNT );
    uint32* idxTmp    = bbcvirtallocbounded<uint32>( BBCU_TABLE_ENTRY_COUNT );
    uint32* idxWriter = indices;

    const uint32* reader       = p3.hostIndices;
    const size_t  readerStride = P3_PRUNED_SLICE_MAX * 3;
    uint64 entryCount = 0;

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        if( cx.cfg.hybrid16Mode )
        {
            const uint32* sizeSlices = &s2.prunedBucketSlices[0][bucket];

            cx.diskContext->phase3.indexBuffer->OverrideReadSlices( bucket, sizeof( uint32 ), sizeSlices, BBCU_BUCKET_COUNT );
            cx.diskContext->phase3.indexBuffer->ReadNextBucket();
            const auto readBucket = cx.diskContext->phase3.indexBuffer->GetNextReadBufferAs<uint32>();
            ASSERT( readBucket.Length() == p3.prunedBucketCounts[(int)cx.table][bucket] );

            bbmemcpy_t( idxWriter, readBucket.Ptr(), readBucket.Length() );

            idxWriter  += readBucket.Length();
            entryCount += readBucket.Length();
        }
        else
        {
            for( uint32 slice = 0; slice < BBCU_BUCKET_COUNT; slice++ )
            {
                const uint32 copyCount = s2.prunedBucketSlices[slice][bucket];

                bbmemcpy_t( idxWriter, reader, copyCount );

                idxWriter  += copyCount;
                entryCount += copyCount;
                reader     += readerStride;
            }
        }
    }

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->phase3.indexBuffer->Swap();
        cx.diskContext->phase3.indexBuffer->Swap();
    }

    ASSERT( entryCount == p3.prunedTableEntryCounts[(int)cx.table] );

    RadixSort256::Sort<BB_MAX_JOBS>( pool, indices, idxTmp, entryCount );

    // Indices must not repeat:
    for( uint64 i = 1; i < entryCount; i++ )
    {
        ASSERT( indices[i] > indices[i-1] );
    }

    DbgHashDataT( indices, entryCount, "indices", (uint32)cx.table+1 );

    bbvirtfreebounded( indices );
    bbvirtfreebounded( idxTmp );

    Log::Line( "[DEBUG] Index validation OK" );
}

//-----------------------------------------------------------
void DbgHashData( const void* data, size_t size, const char* name, uint32 index )
{
    blake3_hasher hasher;
    blake3_hasher_init( &hasher );
    blake3_hasher_update( &hasher, data, size );

    DbgFinishAndPrintHash( hasher, name, index );
}

//-----------------------------------------------------------
void DbgFinishAndPrintHash( blake3_hasher& hasher, const char* name, uint32 index )
{
    constexpr size_t HASH_LEN = 256/8;
    byte output[HASH_LEN];
    blake3_hasher_finalize( &hasher, output, HASH_LEN );

    Log::Write( "[DEBUG] %s_%u hash: 0x", name, index );
    for( uint32 i = 0; i < HASH_LEN; i++ )
        Log::Write( "%02x", output[i] );
    
    Log::NewLine();
}

#endif

