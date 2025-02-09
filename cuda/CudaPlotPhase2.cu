#include "CudaPlotContext.h"
#include "util/StackAllocator.h"

#if _DEBUG
    #include "util/BitField.h"
    #include "threading/MTJob.h"
    #include "plotdisk/jobs/IOJob.h"

    byte* _dbgRMarks = nullptr;
    
    static void DbgValidateTable( CudaK32PlotContext& cx, const TableId table );
    static void DbgWriteMarks( CudaK32PlotContext& cx, const TableId table );
    static void DebugPruneInCPU( CudaK32PlotContext& cx );

    #ifndef DBG_BBCU_P2_COUNT_PRUNED_ENTRIES
        #define DBG_BBCU_P2_COUNT_PRUNED_ENTRIES 1
    #endif
#endif

static void CudaK32PlotAllocateBuffersTest( CudaK32PlotContext& cx );

#define MARK_TABLE_BLOCK_THREADS 128
#define P2_ENTRIES_PER_BUCKET    BBCU_BUCKET_ALLOC_ENTRY_COUNT //((1ull<<BBCU_K)/BBCU_BUCKET_COUNT)


inline size_t GetMarkingTableByteSize()
{
    return 1ull << BBCU_K;
}

template<bool useRMarks>
__global__ void CudaMarkTables( const uint32 entryCount, const uint32* lPairs, const uint16* rPairs,
                                byte* marks, const uint64* rTableMarks, const uint32 rOffset )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles 1 entry
    if( gid >= entryCount )
        return;

    if constexpr ( useRMarks )
    {
        if( !CuBitFieldGet( rTableMarks, rOffset + gid ) )
            return;
    }

    const uint32 l = lPairs[gid];
    const uint32 r = l + rPairs[gid];

    marks[l] = 1;
    marks[r] = 1;
}


__global__ void CudaBytefieldToBitfield( const byte* bytefield, uint64* bitfield
#if DBG_BBCU_P2_COUNT_PRUNED_ENTRIES
    , uint32* gPrunedCount
#endif
 )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_ASSERT( gid < 67108864 );

    // if( gid >= fieldCount )
    //     return;

    // Each thread reads a full 64-bit field, so 64 bytes
    bytefield += gid * 64ull;

    // Convert 64 bytes to a 64-bit field
    uint64 bits = (uint64)bytefield[0];
    
    #pragma unroll
    for( int32 i = 1; i < 64; i++ )
        bits |= (((uint64)bytefield[i]) << i);

    CUDA_ASSERT( (uintptr_t)bitfield / 8 * 8 == (uintptr_t)bitfield );
    bitfield[gid] = bits;

#if DBG_BBCU_P2_COUNT_PRUNED_ENTRIES
    
    uint32 markCount = 0;

    #pragma unroll
    for( uint32 i = 0; i < 64; i++ )
    {
        // if( (bits & (1ull << i)) != 0 )
        //     markCount++;
        if( bytefield[i] == 1 )
            markCount++;
    }

    __shared__ uint32 sharedMarkCount;
    thread_block block = this_thread_block();

    // #TODO: Use warp-aware reduction via CUB
    block.sync();
    if( block.thread_rank() == 0 )
        sharedMarkCount = 0;
    block.sync();
    
    atomicAdd( &sharedMarkCount, markCount );
    block.sync();

    if( block.thread_rank() == 0 )
        atomicAdd( gPrunedCount, sharedMarkCount );
#endif
}

static void BytefieldToBitfield( CudaK32PlotContext& cx, const byte* bytefield, uint64* bitfield, cudaStream_t stream )
{
    const uint64 tableEntryCount    = 1ull << BBCU_K;
    const uint32 fieldCount         = (uint32)( tableEntryCount / 64 );
    
    const uint32 blockThreadCount   = 256;
    const uint32 blockCount         = CDivT( fieldCount, blockThreadCount );

    ASSERT( (uint64)blockCount * blockThreadCount * 64 == tableEntryCount );

    #if DBG_BBCU_P2_COUNT_PRUNED_ENTRIES
        #define G_PRUNED_COUNTS ,cx.phase2->devPrunedCount
        Log::Line( "Marker Set to %d", 7)
CudaErrCheck( cudaMemsetAsync( cx.phase2->devPrunedCount, 0, sizeof( uint32 ), stream ) );
    #else
        #define G_PRUNED_COUNTS 
    #endif
    
    ASSERT_DOES_NOT_OVERLAP2( bitfield, bytefield, GetMarkingTableBitFieldSize(), GetMarkingTableByteSize() );

    CudaBytefieldToBitfield<<<blockCount, blockThreadCount, 0, stream>>>( bytefield, bitfield G_PRUNED_COUNTS );
}

void LoadPairs( CudaK32PlotContext& cx, CudaK32Phase2& p2, const TableId rTable, const uint32 bucket )
{
    if( bucket >= BBCU_BUCKET_COUNT )
        return;

    const uint64 tableEntryCount = cx.tableEntryCounts[(int)rTable];
    const uint32 entryCount      = cx.bucketCounts[(int)rTable][bucket];

        //   uint32* hostPairsL     = cx.hostTableSortedL + p2.pairsLoadOffset;
        //   uint16* hostPairsR     = cx.hostTableSortedR + p2.pairsLoadOffset;
          uint32* hostPairsL     = cx.hostBackPointers[(int)rTable].left  + p2.pairsLoadOffset;
          uint16* hostPairsR     = cx.hostBackPointers[(int)rTable].right + p2.pairsLoadOffset;
    // const uint32* nextHostPairsL = cx.hostBackPointers[(int)rTable-1].left  + p2.pairsLoadOffset;
    // const uint16* nextHostPairsR = cx.hostBackPointers[(int)rTable-1].right + p2.pairsLoadOffset;

    // if( rTable > p2.endTable )
    {
        // Copy the next table to our pinned host pairs
        // p2.pairsLIn.UploadAndPreLoadT( hostPairsL, entryCount, nextHostPairsL, entryCount );
        // p2.pairsRIn.UploadAndPreLoadT( hostPairsR, entryCount, nextHostPairsR, entryCount );
    }
    // else
    // {
        p2.pairsLIn.UploadT( hostPairsL, entryCount );
        p2.pairsRIn.UploadT( hostPairsR, entryCount );
    // }

    p2.pairsLoadOffset += entryCount;
}

void MarkTable( CudaK32PlotContext& cx, CudaK32Phase2& p2 )
{
    const TableId lTable = cx.table;
    const TableId rTable = lTable + 1;

    byte* devLMarks = p2.devMarkingTable;

    if( cx.cfg.hybrid128Mode )
    {
        cx.diskContext->tablesL[(int)rTable]->Swap();
        cx.diskContext->tablesR[(int)rTable]->Swap();

        p2.pairsLIn.AssignDiskBuffer( cx.diskContext->tablesL[(int)rTable] );
        p2.pairsRIn.AssignDiskBuffer( cx.diskContext->tablesR[(int)rTable] );
    }

    // Zero-out marks
    Log::Line( "Marker Set to %d", 8)
CudaErrCheck( cudaMemsetAsync( devLMarks, 0, GetMarkingTableByteSize(), cx.computeStream ) );

    // Load first bucket's worth of pairs
    LoadPairs( cx, p2, rTable, 0 );

    // Mark the table, buckey by bucket
    uint32 rTableGlobalIndexOffset = 0;

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        // Load next set of pairs in the background (if there is another bucket)
        LoadPairs( cx, p2, rTable, bucket + 1 );

        const uint64 tableEntryCount = cx.tableEntryCounts[(int)rTable];
        const uint32 entryCount      = cx.bucketCounts[(int)rTable][bucket];

        // Wait for pairs to be ready
        const uint32* devLPairs = p2.pairsLIn.GetUploadedDeviceBufferT<uint32>( cx.computeStream );
        const uint16* devRPairs = p2.pairsRIn.GetUploadedDeviceBufferT<uint16>( cx.computeStream );

        // Mark
        const uint32 blockCount = (uint32)CDiv( entryCount, MARK_TABLE_BLOCK_THREADS );

        if( rTable == TableId::Table7 )
            CudaMarkTables<false><<<blockCount, MARK_TABLE_BLOCK_THREADS, 0, cx.computeStream>>>( entryCount, devLPairs, devRPairs, devLMarks, nullptr, 0 );
        else
            CudaMarkTables<true ><<<blockCount, MARK_TABLE_BLOCK_THREADS, 0, cx.computeStream>>>( entryCount, devLPairs, devRPairs, devLMarks, p2.devRMarks[(int)rTable], rTableGlobalIndexOffset );

        p2.pairsLIn.ReleaseDeviceBuffer( cx.computeStream );
        p2.pairsRIn.ReleaseDeviceBuffer( cx.computeStream );

        rTableGlobalIndexOffset += entryCount;
    }

    // Convert the bytefield marking table to a bitfield
    uint64* bitfield = (uint64*)p2.outMarks.LockDeviceBuffer( cx.computeStream );

    BytefieldToBitfield( cx, devLMarks, bitfield, cx.computeStream );

    // Download bitfield marks
    // uint64* hostBitField = p2.hostBitFieldAllocator->AllocT<uint64>( GetMarkingTableBitFieldSize() );
    uint64* hostBitField = cx.hostMarkingTables[(int)lTable];

    // #TODO: Do download and copy again, for now just store all of them in this pinned buffer
    // cx.phase3->hostMarkingTables[(int)lTable] = hostBitField;
    p2.outMarks.Download( hostBitField, GetMarkingTableBitFieldSize(), cx.computeStream );

    // p2.outMarks.DownloadAndCopy( hostBitField, cx.hostMarkingTables[(int)lTable], GetMarkingTableBitFieldSize(), cx.computeStream );
    // p2.outMarks.Download( cx.hostMarkingTables[(int)lTable], GetMarkingTableBitFieldSize() );


#if DBG_BBCU_P2_COUNT_PRUNED_ENTRIES
    {
        uint32 prunedEntryCount = 0;
        Log::Line( "Marker Set to %d", 9)
CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );
        Log::Line( "Marker Set to %d", 10)
CudaErrCheck( cudaMemcpyAsync( &prunedEntryCount, p2.devPrunedCount, sizeof( uint32 ), cudaMemcpyDeviceToHost, cx.computeStream ) );
        Log::Line( "Marker Set to %d", 11)
CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );

        const uint64 lEntryCount = cx.tableEntryCounts[(int)lTable];
        Log::Line( "Table %u now has %u / %llu  ( %.2lf%% ) entries.", (uint)lTable+1, 
            prunedEntryCount, lEntryCount, ((double)prunedEntryCount / lEntryCount ) * 100.0 );
    }

    // Check on CPU
    if( 0 )
    {
        #if _DEBUG
        p2.outMarks.WaitForCompletion();


        uint64* hBitField = cx.hostMarkingTables[(int)lTable];
        
        std::atomic<uint64> bitfieldPrunedEntryCount = 0;
        // std::atomic<uint64> totalPrunedEntryCount    = 0;
        // std::atomic<uint64> rTablePrunedEntryCount   = 0;

        AnonMTJob::Run( DbgGetThreadPool( cx ), [&]( AnonMTJob* self ){
            
            const TableId rt          = lTable + 1;
            const uint64  rEntryCount = cx.tableEntryCounts[(int)rTable];
            const uint64  lEntryCount = cx.tableEntryCounts[(int)lTable];

            uint64 localPrunedEntryCount = 0;
            uint64 rPrunedEntryCount     = 0;


            uint64 count, offset, end;



            GetThreadOffsets( self, lEntryCount, count, offset, end );
    //         for( uint64 i = offset; i < end; i++ )
    //         {
    //             if( bytefield[i] == 1 )
    //                 localPrunedEntryCount++;
    //         }
    //         totalPrunedEntryCount += localPrunedEntryCount;

            BitField bits( hBitField, lEntryCount );
            localPrunedEntryCount = 0;
            for( uint64 i = offset; i < end; i++ )
            {
                if( bits.Get( i ) )
                    localPrunedEntryCount++;
            }
            bitfieldPrunedEntryCount += localPrunedEntryCount;
        });
        
        uint64 prunedEntryCount;
        const uint64 lEntryCount      = cx.tableEntryCounts[(int)lTable];
    //           prunedEntryCount = totalPrunedEntryCount.load();
    //     Log::Line( "*** BYTEfield pruned entry count: %llu / %llu ( %.2lf %% )", 
    //         prunedEntryCount, lEntryCount, prunedEntryCount / (double)lEntryCount * 100.0 );

        prunedEntryCount = bitfieldPrunedEntryCount.load();
        Log::Line( "*** Bitfield pruned entry count: %llu / %llu ( %.2lf %% )", 
            prunedEntryCount, lEntryCount, prunedEntryCount / (double)lEntryCount * 100.0 );

    //     if( rTable < TableId::Table7 )
    //     {
    //         prunedEntryCount = rTablePrunedEntryCount.load();
    //         const uint64 rEntryCount = cx.tableEntryCounts[(int)rTable];
    //         Log::Line( "*** R pruned entry count: %llu / %llu ( %.2lf %% )", 
    //             prunedEntryCount, rEntryCount, prunedEntryCount / (double)rEntryCount * 100.0 );

    //     }

    //     // Full CPU method

    //     bbvirtfree( hByteField );
    //     bbvirtfree( hBitField  );
    //     bbvirtfree( rBitField );
    #endif
    }
#endif

    // Set right table marks for the next step
    p2.devRMarks[(int)lTable] = bitfield;    
}

void CudaK32PlotPhase2( CudaK32PlotContext& cx )
{
    CudaK32Phase2& p2 = *cx.phase2;
    // p2.hostBitFieldAllocator->PopToMarker( 0 );

    const uint32 compressionLevel = cx.gCfg->compressionLevel;

    const TableId startRTable = TableId::Table7;    
    const TableId endRTable   = TableId::Table3 + (TableId)cx.gCfg->numDroppedTables;

    p2.endTable = endRTable;

// #if _DEBUG
//     DebugPruneInCPU( cx );
// #endif

#if BBCU_DBG_SKIP_PHASE_1
    DbgLoadTablePairs( cx, TableId::Table7, true );
#endif
    // CudaK32PlotAllocateBuffersTest( cx );

    for( TableId rTable = startRTable; rTable >= endRTable; rTable-- )
    {
    #if BBCU_DBG_SKIP_PHASE_1
        DbgLoadTablePairs( cx, rTable-1, false );
    // DbgValidateTable( cx, rTable );
    #endif
        const auto timer = TimerBegin();

        cx.table           = rTable-1;
        p2.pairsLoadOffset = 0;

        MarkTable( cx, p2 );
        p2.outMarks.WaitForCompletion();
        p2.outMarks.Reset();
        p2.pairsLIn.Reset();
        p2.pairsRIn.Reset();

        const auto elapsed = TimerEnd( timer );
        Log::Line( "Marked Table %u in %.2lf seconds.", rTable, elapsed );

        #if _DEBUG && DBG_BBCU_P2_WRITE_MARKS
            p2.outMarks.WaitForCompletion();
            DbgWriteMarks( cx, rTable-1 );
        #endif
    }

    // Wait for everything to complete

    // p2.outMarks.WaitForCopyCompletion(); // #TODO: Re-activate this when re-enabling copy
    p2.outMarks.WaitForCompletion();
    p2.outMarks.Reset();
}


///
/// Allocation
///
void CudaK32PlotPhase2AllocateBuffers( CudaK32PlotContext& cx, CudaK32AllocContext& acx )
{
    GpuStreamDescriptor desc{};

    desc.entriesPerSlice = P2_ENTRIES_PER_BUCKET;
    desc.sliceCount      = 1;
    desc.sliceAlignment  = cx.allocAlignment;
    desc.bufferCount     = BBCU_DEFAULT_GPU_BUFFER_COUNT;
    desc.deviceAllocator = acx.devAllocator;
    desc.pinnedAllocator = nullptr;             // Start in direct mode (no intermediate pinined buffers)

    if( cx.cfg.hybrid128Mode )
    {
        desc.pinnedAllocator = acx.pinnedAllocator;
        desc.sliceAlignment  = cx.diskContext->temp1Queue->BlockSize();
    }

    if( !cx.downloadDirect )
        desc.pinnedAllocator = acx.pinnedAllocator;

    CudaK32Phase2& p2 = *cx.phase2;

    const size_t markingTableByteSize     = GetMarkingTableByteSize();
    const size_t markingTableBitFieldSize = GetMarkingTableBitFieldSize();

    // Device buffers
    p2.devPrunedCount  = acx.devAllocator->CAlloc<uint32>( 1, acx.alignment );
    p2.devMarkingTable = acx.devAllocator->AllocT<byte>( markingTableByteSize, acx.alignment );

    // Upload/Download streams
    p2.pairsLIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint32>( desc, acx.dryRun );
    p2.pairsRIn = cx.gpuUploadStream[0]->CreateUploadBufferT<uint16>( desc, acx.dryRun );

    desc.entriesPerSlice = markingTableBitFieldSize;
    p2.outMarks          = cx.gpuDownloadStream[0]->CreateDownloadBufferT<byte>( desc, acx.dryRun );
}


#if _DEBUG

void DebugPruneInCPU( CudaK32PlotContext& cx )
{
    ThreadPool& pool = DbgGetThreadPool( cx );
    byte* bytefields[2] = {
        bbvirtalloc<byte>( GetMarkingTableByteSize() ),
        bbvirtalloc<byte>( GetMarkingTableByteSize() )
    };

    
    // uint64* bitfield = bbvirtalloc<uint64>( GetMarkingTableBitFieldSize() );
    // BitField marks( bitfield, 1ull << BBCU_K );
    // memset( bitfield, 0, GetMarkingTableBitFieldSize() );

    // uint64 prunedEntryCount = 0;
    // const uint64 entryCount = cx.tableEntryCounts[6];
    

    // for( uint64 i = 0; i < entryCount; i++ )
    // {
    //     const uint32 l = rTable.left[i];
    //     const uint32 r = l + rTable.right[i];

    //     marks.Set( l );
    //     marks.Set( r );
    // }

    // for( uint64 i = 0; i < 1ull << BBCU_K; i++ )
    // {
    //     if( marks.Get( i ) )
    //         prunedEntryCount++;
    // }
    // const TableId rTableId = TableId::Table7;

    for( TableId rTableId = TableId::Table7; rTableId >= cx.phase2->endTable; rTableId-- )
    {
        const TableId lTableId = rTableId - 1;

        const byte* rTableByteField = bytefields[(int)lTableId % 2];
              byte* bytefield       = bytefields[(int)rTableId % 2];

        memset( bytefield, 0, GetMarkingTableByteSize() );
        
        // DbgLoadTablePairs( cx, rTableId );
        // Pairs rTable = cx.hostBackPointers[(int)rTableId];
        
        std::atomic<uint64> totalPrunedEntryCount = 0;

        AnonMTJob::Run( pool, [&]( AnonMTJob* self ) {

            const uint64 rEntryCount = cx.tableEntryCounts[(int)rTableId];
            {
                uint64 count, offset, end;
                GetThreadOffsets( self, rEntryCount, count, offset, end );

                const TableId rId    = rTableId;
                      Pairs   rTable = cx.hostBackPointers[(int)rTableId];

                for( uint64 i = offset; i < end; i++ )
                {
                    if( rId < TableId::Table7 && rTableByteField[i] == 0 )
                        continue;

                    const uint32 l = rTable.left[i];
                    const uint32 r = l + rTable.right[i];
                    
                    bytefield[l] = 1;
                    bytefield[r] = 1;
                }

                self->SyncThreads();

                      uint64 localPrunedEntryCount = 0;
                const uint64 lEntryCount           = cx.tableEntryCounts[(int)lTableId];
                GetThreadOffsets( self, lEntryCount, count, offset, end );
                for( uint64 i = offset; i < end; i++ )
                {
                    if( bytefield[i] == 1 )
                        localPrunedEntryCount++;
                }

                totalPrunedEntryCount += localPrunedEntryCount;
            }
        });

        const uint64 prunedEntryCount = totalPrunedEntryCount.load();
        const uint64 lEntryCount      = cx.tableEntryCounts[(int)lTableId];
        Log::Line( "Table %u Pruned entry count: %llu / %llu ( %.2lf %% )", (uint)rTableId,
            prunedEntryCount, lEntryCount, prunedEntryCount / (double)lEntryCount * 100.0 );
    }
}

void DbgValidateTable( CudaK32PlotContext& cx )
{
    ThreadPool& pool = DbgGetThreadPool( cx );

    byte* bytefieldL = bbvirtalloc<byte>( GetMarkingTableByteSize() );
    byte* bytefieldR = bbvirtalloc<byte>( GetMarkingTableByteSize() );
    memset( bytefieldL, 0, GetMarkingTableByteSize() );
    memset( bytefieldR, 0, GetMarkingTableByteSize() );

    // uint64* bitfield = bbvirtalloc<uint64>( GetMarkingTableBitFieldSize() );
    // BitField marks( bitfield, 1ull << BBCU_K );
    // memset( bitfield, 0, GetMarkingTableBitFieldSize() );

    // uint64 prunedEntryCount = 0;
    // const uint64 entryCount = cx.tableEntryCounts[6];
    // Pairs rTable = cx.hostBackPointers[6];

    // for( uint64 i = 0; i < entryCount; i++ )
    // {
    //     const uint32 l = rTable.left[i];
    //     const uint32 r = l + rTable.right[i];

    //     marks.Set( l );
    //     marks.Set( r );
    // }

    // for( uint64 i = 0; i < 1ull << BBCU_K; i++ )
    // {
    //     if( marks.Get( i ) )
    //         prunedEntryCount++;
    // }
    Log::Line( "[DEBUG] Validating table" );

    // for( TableId rt = TableId::Table7; rt >= TableId::Table3; rt-- )
    TableId rt = TableId::Table7;
    {
        {
            uint64 totalCount = 0;
            for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
                totalCount += cx.bucketCounts[(int)rt][bucket];

            ASSERT( totalCount == cx.tableEntryCounts[(int)rt] );
        }

        std::atomic<uint64> totalPrunedEntryCount = 0;

        memset( bytefieldL, 0, GetMarkingTableByteSize() );

        Pairs hostRTablePairs = cx.hostBackPointers[(int)rt];

        for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
        {
            const uint32 rTableBucketEntryCount = cx.bucketCounts[(int)rt][bucket];

            // Mark
            AnonMTJob::Run( pool, [&]( AnonMTJob* self ){
                
                    //   Pairs  rTable      = cx.hostBackPointers[(int)rt];
                // const uint64 rEntryCount = cx.tableEntryCounts[(int)rt];
                const uint64 rBucketEntryCount = rTableBucketEntryCount;

                {
                    uint64 count, offset, end;
                    GetThreadOffsets( self, rBucketEntryCount, count, offset, end );

                    Pairs rTable = hostRTablePairs;

                    if( offset == 0 )
                        Log::Line( "[%-3u] %u, %u", bucket, rTable.left[offset], (uint32)rTable.right[offset] );

                    const bool readR = rt < TableId::Table7;

                    const byte* rBytes = bytefieldR;
                          byte* lBytes = bytefieldL;

                    for( uint64 i = offset; i < end; i++ )
                    {
                        // if( readR && rBytes[i] == 0 )
                        //     continue;
        
                        const uint32 l = rTable.left[i];
                        const uint32 r = l + rTable.right[i];
                        
                        lBytes[l] = 1;
                        lBytes[r] = 1;
                    }
                }
            });

            hostRTablePairs.left  += rTableBucketEntryCount;
            hostRTablePairs.right += rTableBucketEntryCount;
        }

        // Count
        AnonMTJob::Run( pool, [&]( AnonMTJob* self ){

                  uint64 localPrunedEntryCount = 0;
            const uint64 lEntryCount           = cx.tableEntryCounts[(int)rt-1];
            const byte * lBytes                = bytefieldL;

            uint64 count, offset, end;
            GetThreadOffsets( self, lEntryCount, count, offset, end );
            for( uint64 i = offset; i < end; i++ )
            {
                if( lBytes[i] == 1 )
                    localPrunedEntryCount++;
            }

            totalPrunedEntryCount += localPrunedEntryCount;
        });

        // if( _dbgRMarks == nullptr )
        //     _dbgRMarks = bb
        std::swap( bytefieldL, bytefieldR );

        const uint64 prunedEntryCount = totalPrunedEntryCount.load();
        const uint64 lEntryCount      = cx.tableEntryCounts[(int)rt-1];
        Log::Line( "Table %u pruned entry count: %llu / %llu ( %.2lf %% )", (uint)rt,
            prunedEntryCount, lEntryCount, prunedEntryCount / (double)lEntryCount * 100.0 );
    }
}

void DbgWriteMarks( CudaK32PlotContext& cx, const TableId table )
{
    char path[512];

    std::string baseUrl = DBG_BBCU_DBG_DIR;
    if( cx.cfg.hybrid128Mode )
        baseUrl += "disk/";

    Log::Line( "[DEBUG] Writing marking table %u to disk...", table+1 );
    {
        sprintf( path, "%smarks%d.tmp", baseUrl.c_str(), (int)table+1 );

        const uint64* marks = cx.hostMarkingTables[(int)table];

        int err;
        FatalIf( !IOJob::WriteToFile( path, marks, GetMarkingTableBitFieldSize(), err ),
            "Failed to write marking table with error: %d", err );
    }
}

#endif

