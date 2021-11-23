#include "FxGenBucketized.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "diskplot/DiskBufferQueue.h"
#include "b3/blake3.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

struct FxBucketJob : MTJob<FxBucketJob>
{
    // Inputs
    uint32        bucketIdx;
    uint32        entryCount;
    uint32        fileBlockSize;        // For direct I/O
    TableId       table;

    
    const Pairs   pairs;
    const uint32* yIn;
    const uint64* metaInA;
    const uint64* metaInB;
    const uint32* counts;               // Each thread's entry count per bucket

    // Temp/working space
    uint32*       yTmp;
    uint64*       metaTmpA;
    uint64*       metaTmpB;

    // Outputs, if not disk-based
    uint32*       yOut;
    uint64*       metaOutA;
    uint64*       metaOutB;
    byte*         bucketIdOut;

    uint32*       totalBucketCounts;    // Total counts per for all buckets. Used by the control thread

    // For chunked jobs:
    uint32           chunkSize;
    uint32           entriesPerChunk;
    uint32           chunkCount;
    uint32           trailingChunkEntries;
    DiskBufferQueue* queue;
    bool             writeToDisk;

    void Run() override;

    template<TableId tableId, typename TMetaA, typename TMetaB>
    void DistributeIntoBuckets(
        const uint32  entryCount,
        const byte*   bucketIndices,
        const uint32* y,                // Unsorted table data
        const TMetaA* metaA,
        const TMetaB* metaB,
        uint32*       yBuckets,         // Output buckets
        TMetaA*       metaABuckets,
        TMetaB*       metaBBuckets,
        uint32        outBucketCounts[BB_DP_BUCKET_COUNT], // Entry count per bucket (across all threads)
        const size_t  fileBlockSize
    );

    void CalculatePrefixSum( 
        uint32       counts      [BB_DP_BUCKET_COUNT],  // Entry count for this thread
        uint32       pfxSum      [BB_DP_BUCKET_COUNT],  // Prefix sum for this thread
        uint32       bucketCounts[BB_DP_BUCKET_COUNT],  // Entries per bucket, across all threads
        const size_t fileBlockSize                      // For aligning data when direct IO is enabled
    );

    template<TableId tableId, bool WriteToDisk>
    void RunForTable();
};

//-----------------------------------------------------------
template<TableId tableId>
static void ComputeFxForTable( 
    const uint64  bucket, 
    uint32        entryCount, 
    const Pairs   pairs,
    const uint32* yIn, 
    const uint64* metaInA, 
    const uint64* metaInB,
    uint32*       yOut,
    byte*         bucketOut, 
    uint64*       metaOutA, 
    uint64*       metaOutB );


//-----------------------------------------------------------
template<TableId tableId>
void GenFxBucketizedChunked(
    DiskBufferQueue* diskQueue,
    ThreadPool&      threadPool,
    uint             threadCount,
    size_t           chunkSize,
    bool             directIO,
    
    uint             bucketIdx,        // Inputs
    uint             entryCount, 
    Pairs            pairs,
    const uint32*    yIn,
    const uint64*    metaInA, 
    const uint64*    metaInB,

    byte*            bucketIdOut,     // Tmp
    uint32*          yTmp,
    uint64*          metaTmpA,
    uint64*          metaTmpB,

    uint32*          yOut,             // Outputs
    uint64*          metaOutA,
    uint64*          metaOutB,

    uint32           bucketCounts[BB_DP_BUCKET_COUNT]
)
{
    const size_t outMetaSizeA = TableMetaOut<tableId>::SizeA;
    const size_t outMetaSizeB = TableMetaOut<tableId>::SizeB;

    const size_t sizePerEntry = sizeof( uint32 ) + sizeof( outMetaSizeA ) + sizeof( outMetaSizeB );

    // We need to adjust the chunk buffer size to leave some space for us to be
    // able to align each bucket start pointer to the block size of the output device
    const size_t fileBlockSize         = diskQueue ? 1ull : diskQueue->BlockSize();
    const size_t bucketBlockAlignSize  = fileBlockSize * BB_DP_BUCKET_COUNT;
    const size_t usableChunkSize       = directIO ? chunkSize : chunkSize - bucketBlockAlignSize * 2;

    const uint32 entriesPerChunk       = chunkSize == 0 ? entryCount : (uint32)( usableChunkSize / sizePerEntry );

    uint32 chunkCount            = entriesPerChunk / entryCount;
    uint32 chunkTrailingEntries  = entryCount - entriesPerChunk * chunkCount;
    uint32 entriesPerThread      = entriesPerChunk / threadCount;

    ASSERT( entriesPerThread > 0 );

    // If the leftover entries per chunk add up to a full chunk, then count it
    while( chunkTrailingEntries >= chunkCount )
    {
        chunkCount++;
        chunkTrailingEntries -= entriesPerChunk;
    }

    // Left-over entries per thread, per chunk can be spread out
    // between threads since we are guaranteed they still fit in a chunk,
    // and are less than thread count

    uint32 threadTrailingEntries = entriesPerChunk - entriesPerThread * threadCount;
    
    MTJobRunner<FxBucketJob> jobs( threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        FxBucketJob& job = jobs[i];

        job.queue         = diskQueue;
        job.bucketIdx     = bucketIdx;
        job.entryCount    = entriesPerThread;
        job.fileBlockSize = (uint32)fileBlockSize;
        job.table         = tableId;

        job.pairs         = pairs;
        job.yIn           = yIn;
        job.metaInA       = metaInA;
        job.metaInB       = metaInB;
        job.counts        = nullptr;

        job.yTmp          = yTmp;
        job.metaTmpA      = metaTmpA;
        job.metaTmpB      = metaTmpB;

        job.yOut          = yOut;
        job.metaOutA      = metaOutA;
        job.metaOutB      = metaOutB;
        job.bucketIdOut   = bucketIdOut;

        job.totalBucketCounts    = bucketCounts;

        job.chunkSize            = usableChunkSize;
        job.chunkCount           = chunkCount;
        job.entriesPerChunk      = entriesPerChunk;
        job.trailingChunkEntries = chunkTrailingEntries;

        if( threadTrailingEntries )
        {
            job.entryCount ++;
            threadTrailingEntries --;
        }

        pairs.left  += job.entryCount;
        pairs.right += job.entryCount;

        yTmp        += job.entryCount;
        metaTmpA    += job.entryCount;

        if constexpr ( outMetaSizeB > 0 )
            metaTmpB += job.entryCount;
    }
}

//-----------------------------------------------------------
template<TableId table>
void FxGenBucketized<table>::GenerateFxBucketizedInMemory(
        ThreadPool&      pool, 
        uint32           threadCount,
        uint32           bucketIdx,
        const uint32     entryCount,
        Pairs            pairs,
        byte*            bucketIndices,

        const uint32*    yIn,
        const uint64*    metaAIn,
        const uint64*    metaBIn,

        uint32*          yTmp,
        uint32*          metaATmp,
        uint32*          metaBTmp,

        uint32*          yOut,
        uint32*          metaAOut,
        uint32*          metaBOut,

        uint32           bucketCounts[BB_DP_BUCKET_COUNT]
    )
{
    GenFxBucketizedChunked<table>(
        nullptr,
        pool, threadCount,
        0,                  // Chunk size of 0 means unchunked
        false,

        bucketIdx,
        entryCount,
        pairs,
        
        yIn,
        metaAIn,
        metaBIn,

        bucketIndices,
        yTmp,
        metaATmp,
        metaBTmp,
        
        yOut,
        metaAOut,
        metaBOut,

        bucketCounts
    );
}

//-----------------------------------------------------------
template<TableId tableId>
FORCE_INLINE
void GenFxBucketized(
    
    ThreadPool&   threadPool,
    uint          threadCount,
    size_t        fileBlockSize,    // For direct I/O alignment

    uint          bucketIdx,        // Inputs
    uint          entryCount, 
    Pairs         pairs,
    const uint32* yIn,
    const uint64* metaInA, 
    const uint64* metaInB,

    uint32*       yTmp,             // Tmp
    uint64*       metaTmpA,
    uint64*       metaTmpB,

    uint32*       yOut,             // Outputs
    uint64*       metaOutA,
    uint64*       metaOutB,
    byte*         bucketIdOut,

    uint32        bucketCounts[BB_DP_BUCKET_COUNT]
)
{
    const size_t outMetaSizeA = TableMetaOut<tableId>::SizeA;
    const size_t outMetaSizeB = TableMetaOut<tableId>::SizeB;

    const uint32 entriesPerThread = entryCount / threadCount;

    uint32 trailingEntries = entryCount - entriesPerThread * threadCount;

    MTJobRunner<FxBucketJob> jobs( threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        FxBucketJob& job = jobs[i];

        job.bucketIdx     = bucketIdx;
        job.entryCount    = entriesPerThread;
        job.fileBlockSize = (uint32)fileBlockSize;
        job.table         = tableId;

        job.pairs         = pairs;
        job.yIn           = yIn;
        job.metaInA       = metaInA;
        job.metaInB       = metaInB;
        job.counts        = nullptr;

        job.yTmp          = yTmp;
        job.metaTmpA      = metaTmpA;
        job.metaTmpB      = metaTmpB;

        job.yOut          = yOut;
        job.metaOutA      = metaOutA;
        job.metaOutB      = metaOutB;
        job.bucketIdOut   = bucketIdOut;

        job.totalBucketCounts = bucketCounts;

        if( trailingEntries )
        {
            job.entryCount ++;
            trailingEntries --;
        }

        pairs.left  += job.entryCount;
        pairs.right += job.entryCount;

        yTmp        += job.entryCount;
        metaTmpA    += job.entryCount;

        if constexpr ( outMetaSizeB > 0 )
            metaTmpB += job.entryCount;
    }

    jobs.Run( threadCount );
}

//-----------------------------------------------------------
void FxBucketJob::Run()
{
    switch( this->table )
    {
        case TableId::Table1: 
            this->writeToDisk ? RunForTable<TableId::Table1, true>() : RunForTable<TableId::Table1, false>(); 
            return;
        case TableId::Table2: 
            this->writeToDisk ? RunForTable<TableId::Table2, true>() : RunForTable<TableId::Table2, false>(); 
            return;
        case TableId::Table3: 
            this->writeToDisk ? RunForTable<TableId::Table3, true>() : RunForTable<TableId::Table3, false>(); 
            return;
        case TableId::Table4: 
            this->writeToDisk ? RunForTable<TableId::Table4, true>() : RunForTable<TableId::Table4, false>(); 
            return;
        case TableId::Table5: 
            this->writeToDisk ? RunForTable<TableId::Table5, true>() : RunForTable<TableId::Table5, false>(); 
            return;
        case TableId::Table6: 
            this->writeToDisk ? RunForTable<TableId::Table6, true>() : RunForTable<TableId::Table6, false>(); 
            return;
        case TableId::Table7: 
            this->writeToDisk ? RunForTable<TableId::Table7, true>() : RunForTable<TableId::Table7, false>(); 
            return;
        
        case TableId::_Count:
        default:
            ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
template<TableId tableId, bool WriteToDisk>
void FxBucketJob::RunForTable()
{
    using TMetaA = typename TableMetaOut<tableId>::MetaA;
    using TMetaB = typename TableMetaOut<tableId>::MetaB;

    DiskBufferQueue* diskQueue = this->queue;

    uint32       chunkCount      = this->chunkCount;
    uint32       entryCount      = this->entryCount;
    const uint32 bucketIdx       = this->bucketIdx;
    const size_t fileBlockSize   = this->fileBlockSize;
    const uint32 entriesPerChunk = this->entriesPerChunk;
    
    Pairs         pairs        = this->pairs;
    const uint32* yIn          = this->yIn;
    const uint64* metaInA      = this->metaInA;
    const uint64* metaInB      = this->metaInB;

    // Temporary buffers used when we're calculating FX, but before distribution into buckets.
    uint32* yTmp     = this->yTmp ;
    uint64* metaATmp = this->metaTmpA;
    uint64* metaBTmp = this->metaTmpB;

    // If writing to mmemory, these are not null.
    // When writing to disk, we get a buffer from the queue.
    // #TODO: We should be able to ping-pong the in/out buffers
    //        instead when processing in-memory.
    uint32* yOut          = this->yOut;
    uint64* metaOutA      = this->metaOutA;
    uint64* metaOutB      = this->metaOutB;
    byte*   bucketIndices = this->bucketIdOut;


    uint bucketCounts[BB_DP_BUCKET_COUNT];  // Count how many entries we have per bucket, across all threads.
                                            // This is only used by the main thread

    // If there's a left-over chunk, account for it here
    uint32 trailingChunk = 0xFFFFFFFF;
    if( this->trailingChunkEntries )
    {
        trailingChunk = chunkCount;
        chunkCount ++;
    }

    // Start processing chunks
    for( uint chunk = 0; chunk < chunkCount; chunk++ )
    {
        // Adjust entry count if its the trailing chunk
        if( chunk == trailingChunk )
        {
            ASSERT( this->trailingChunkEntries );
            // #TODO: This
        }

        // Grab a buffer
        if constexpr( WriteToDisk )
        {
            // #TODO: This
            ASSERT( diskQueue );
        }

        // Calculate fx for chunk
        ComputeFxForTable<tableId>( 
            bucketIdx, entryCount, pairs,
            yIn, metaInA, metaInB,
            yOut, bucketIdOut, metaOutA, metaOutB );

        // Distribute entries into their corresponding buckets
        DistributeIntoBuckets<tableId, TMetaA, TMetaB>(
            entryCount, bucketIndices, 
            yTmp, (TMetaA*)metaATmp, (TMetaB*)metaBTmp,
            yOut, (TMetaA*)metaOutA, (TMetaB*)metaOutB,
            bucketCounts, fileBlockSize );

        if( this->IsControlThread() )
        {
            uint32* totalBucketCounts = this->totalBucketCounts;

            // Add to the the total count
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                totalBucketCounts[i] += bucketCounts[i];
        }

        // Submit bucket buffers
        if constexpr( WriteToDisk )
        {
            ASSERT( diskQueue );
            // #TODO: This
        }

        // Go to next chunk
        pairs.left  += entriesPerChunk;
        pairs.right += entriesPerChunk;
    }
}

//-----------------------------------------------------------
template<TableId tableId>
FORCE_INLINE 
void ComputeFxForTable( const uint64 bucket, uint32 entryCount, const Pairs pairs, 
                        const uint32* yIn, const uint64* metaInA, const uint64* metaInB, 
                        uint32* yOut, byte* bucketOut, uint64* metaOutA, uint64* metaOutB )
{
    constexpr size_t metaKMultiplierIn  = TableMetaIn <tableId>::Multiplier;
    constexpr size_t metaKMultiplierOut = TableMetaOut<tableId>::Multiplier;

    // Helper consts
    // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
    // so we need to shift by 32 bits, instead of 26.
    // constexpr uint extraBitsShift = tableId == TableId::Table7 ? 0 : kExtraBits;

    constexpr uint   shiftBits   = metaKMultiplierOut == 0 ? 0 : kExtraBits;
    constexpr uint   k           = _K;
    constexpr uint32 ySize       = k + kExtraBits;         // = 38
    constexpr uint32 yShift      = 64u - (k + shiftBits);  // = 26 or 32
    constexpr size_t metaSize    = k * metaKMultiplierIn;
    constexpr size_t metaSizeLR  = metaSize * 2;
    constexpr size_t bufferSize  = CDiv( ySize + metaSizeLR, 8u );

    // Bucket for extending y
//     const uint64 bucket = ( (uint64)bucketIdx ) << 32;

    // Meta extraction
    uint64 l0, l1, r0, r1;

    // Hashing
    uint64 input [5];       // y + L + R
    uint64 output[4];       // blake3 hashed output

    static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

    blake3_hasher hasher;

    #if _DEBUG
        uint64 prevY = bucket | yIn[pairs.left[0]];
    #endif

    for( uint i = 0; i < entryCount; i++ )
    {
        const uint32 left  = pairs.left[i];
        const uint32 right = left + pairs.right[i];

        const uint64 y     = bucket | yIn[left];

        #if _DEBUG
            ASSERT( y >= prevY );
            prevY = y;
        #endif

        // Extract metadata
        if constexpr( metaKMultiplierIn == 1 )
        {
            l0 = reinterpret_cast<const uint32*>( metaInA )[left ];
            r0 = reinterpret_cast<const uint32*>( metaInA )[right];

            input[0] = Swap64( y  << 26 | l0 >> 6  );
            input[1] = Swap64( l0 << 58 | r0 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 2 )
        {
            l0 = metaInA[left ];
            r0 = metaInA[right];

            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | r0 >> 38 );
            input[2] = Swap64( r0 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 3 )
        {
            l0 = metaInA[left];
            l1 = reinterpret_cast<const uint32*>( metaInB )[left ];
            r0 = metaInA[right];
            r1 = reinterpret_cast<const uint32*>( metaInB )[right];
        
            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 6  );
            input[2] = Swap64( l1 << 58 | r0 >> 6  );
            input[3] = Swap64( r0 << 58 | r1 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 4 )
        {
            l0 = metaInA[left ];
            l1 = metaInB[left ];
            r0 = metaInA[right];
            r1 = metaInB[right];

            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 38 );
            input[2] = Swap64( l1 << 26 | r0 >> 38 );
            input[3] = Swap64( r0 << 26 | r1 >> 38 );
            input[4] = Swap64( r1 << 26 );
        }

        // Hash input
        blake3_hasher_init( &hasher );
        blake3_hasher_update( &hasher, input, bufferSize );
        blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

        uint64 fx = Swap64( *output ) >> yShift;

        yOut[i] = (uint32)fx;

        if constexpr( tableId != TableId::Table7 )
        {
            // Store the bucket id for this y value
            bucketOut[i] = (byte)( fx >> 32 );
        }
        else
        {
            // For table 7 we don't have extra bits,
            // but we do want to be able to store per bucket,
            // in order to sort. So let's just use the high 
            // bits of the 32 bit values itself
            bucketOut[i] = (byte)( ( fx >> 26 ) & 0b111111 );
        }

        // Calculate output metadata
        if constexpr( metaKMultiplierOut == 2 && metaKMultiplierIn == 1 )
        {
            metaOutA[i] = l0 << 32 | r0;
        }
        else if constexpr ( metaKMultiplierOut == 2 && metaKMultiplierIn == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );

            metaOutA[0] = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( metaKMultiplierOut == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
            reinterpret_cast<uint32*>( metaOutB )[i] = (uint32)( ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58 );
        }
        else if constexpr( metaKMultiplierOut == 4 && metaKMultiplierIn == 2 )
        {
            metaOutA[i] = l0;
            metaOutB[i] = r0;
        }
        else if constexpr ( metaKMultiplierOut == 4 && metaKMultiplierIn != 2 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
            metaOutB[i] = h1 << 38    | h2 >> 26;
        }
    }
}


//-----------------------------------------------------------
template<TableId tableId, typename TMetaA, typename TMetaB>
void FxBucketJob::DistributeIntoBuckets(
    const uint32  entryCount,
    const byte*   bucketIndices,
    const uint32* y,
    const TMetaA* metaA,
    const TMetaB* metaB,
    uint32*       yBuckets,
    TMetaA*       metaABuckets,
    TMetaB*       metaBBuckets,
    uint32        outBucketCounts[BB_DP_BUCKET_COUNT],
    const size_t  fileBlockSize
)
{
    const size_t metaSizeA = TableMetaOut<tableId>::SizeA;
    const size_t metaSizeB = TableMetaOut<tableId>::SizeB;

    uint32 counts[BB_DP_BUCKET_COUNT];
    uint32 pfxSum[BB_DP_BUCKET_COUNT];

    // Count our buckets
    memset( counts, 0, sizeof( counts ) );
    for( const byte* ptr = bucketIndices, *end = ptr + entryCount; ptr < end; ptr++ )
    {
        ASSERT( *ptr <= ( 0b111111u ) );
        counts[*ptr] ++;
    }

    this->CalculatePrefixSum( counts, pfxSum, outBucketCounts, fileBlockSize );

    // #TODO: Unroll this a bit?
    // Distribute values into buckets at each thread's given offset
    for( uint i = 0; i < entryCount; i++ )
    {
        const uint32 dstIdx = --pfxSum[bucketIndices[i]];

        yBuckets[dstIdx] = y[i];

        if constexpr ( metaSizeA > 0 )
            metaABuckets[dstIdx] = metaA[i];

        if constexpr ( metaSizeB > 0 )
            metaBBuckets[dstIdx] = metaB[i];
    }
}

//-----------------------------------------------------------
inline void FxBucketJob::CalculatePrefixSum( 
    uint32       counts      [BB_DP_BUCKET_COUNT],
    uint32       pfxSum      [BB_DP_BUCKET_COUNT],
    uint32       bucketCounts[BB_DP_BUCKET_COUNT],
    const size_t fileBlockSize
)
{
    const uint32 jobId    = this->JobId();
    const uint32 jobCount = this->JobCount();

    // This holds the count of extra entries added per-bucket
    // to align each bucket starting address to disk block size.
    // Only used when fileBlockSize > 0
    uint32 entryPadding[BB_DP_BUCKET_COUNT];

    this->counts = counts;
    this->SyncThreads();

    // Add up all of the jobs counts
    memset( pfxSum, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );

    for( uint i = 0; i < jobCount; i++ )
    {
        const uint* tCounts = this->GetJob( i ).counts;

        for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
            pfxSum[j] += tCounts[j];
    }

    // #TODO: Only do this for the control thread?
    // uint32 totalCount = 0;
    // for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
    //     totalCount += pfxSum[i];

    // If we're the control thread, retain the total bucket count
    if( this->IsControlThread() )
    {
        memcpy( bucketCounts, pfxSum, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
    }

    // #TODO: Only do this if using Direct IO
    // We need to align our bucket totals to the  file block size boundary
    // so that each block buffer is properly aligned for direct io.
    if( fileBlockSize )
    {
        #if _DEBUG
            size_t bucketAddress = 0;
        #endif

        for( uint i = 0; i < BB_DP_BUCKET_COUNT-1; i++ )
        {
            const uint32 count = pfxSum[i];

            pfxSum[i]       = RoundUpToNextBoundary( count * sizeof( uint32 ), (int)fileBlockSize ) / sizeof( uint32 );
            entryPadding[i] = pfxSum[i] - count;

            #if _DEBUG
                bucketAddress += pfxSum[i] * sizeof( uint32 );
                ASSERT( bucketAddress / fileBlockSize * fileBlockSize == bucketAddress );
            #endif
        }

        #if _DEBUG
        // if( this->IsControlThread() )
        // {
        //     size_t totalSize = 0;
        //     for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        //         totalSize += pfxSum[i];

        //     totalSize *= sizeof( uint32 );
        //     Log::Line( "Total Size: %llu", totalSize );
        // }   
        #endif
    }

    // Calculate the prefix sum
    for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Subtract the count from all threads after ours 
    // to get the correct prefix sum for this thread
    for( uint t = jobId+1; t < jobCount; t++ )
    {
        const uint* tCounts = this->GetJob( t ).counts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] -= tCounts[i];
    }

    if( fileBlockSize )
    {
        // Now that we have the starting addresses of the buckets
        // at a block-aligned position, we need to substract
        // the padding that we added to align them, so that
        // the entries actually get writting to the starting
        // point of the address

        for( uint i = 0; i < BB_DP_BUCKET_COUNT-1; i++ )
            pfxSum[i] -= entryPadding[i];
    }
}


#pragma GCC diagnostic pop

