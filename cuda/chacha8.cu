#include "pos/chacha8.h"
#include "CudaPlotContext.h"
#include "plotting/DiskBucketBuffer.h"

// #TEST
#if _DEBUG
    #include "threading/MTJob.h"

    uint64 CudaPlotK32DbgXtoF1( CudaK32PlotContext& cx, const uint32 x );
    static void DbgValidateBucket( CudaK32PlotContext& cx, const uint32 bucket );
#endif

#define U32TO32_LITTLE(v) CuBSwap32(v)
#define U8TO32_LITTLE(p) (*(const uint32_t *)(p))
#define U32TO8_LITTLE(p, v) (((uint32_t *)(p))[0] = U32TO32_LITTLE(v))
#define ROTL32(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define ROTATE(v, c) (ROTL32(v, c))
#define XOR(v, w) ((v) ^ (w))
#define PLUS(v, w) ((v) + (w))
#define PLUSONE(v) (PLUS((v), 1))

#define QUARTERROUND(a, b, c, d) \
    a = PLUS(a, b);              \
    d = ROTATE(XOR(d, a), 16);   \
    c = PLUS(c, d);              \
    b = ROTATE(XOR(b, c), 12);   \
    a = PLUS(a, b);              \
    d = ROTATE(XOR(d, a), 8);    \
    c = PLUS(c, d);              \
    b = ROTATE(XOR(b, c), 7)


// 128 threads per cuda block, each thread will do one chacha block
#define CHACHA_BLOCKS_PER_CUDA_BLOCK 128ull

//-----------------------------------------------------------
__global__ void get_keystream_gpu( const uint32_t* input, uint64_t chachaBlock, uint32* outY, uint32* outX, uint32* gBucketCounts )
{
    const uint32   id          = (uint32)threadIdx.x;
    const uint64_t blockOffset = blockIdx.x * CHACHA_BLOCKS_PER_CUDA_BLOCK + id;
    
    chachaBlock += blockOffset;

    uint32_t x[16];// , x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
    uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;

    j0  = input[0];
    j1  = input[1];
    j2  = input[2];
    j3  = input[3];
    j4  = input[4];
    j5  = input[5];
    j6  = input[6];
    j7  = input[7];
    j8  = input[8];
    j9  = input[9];
    j10 = input[10];
    j11 = input[11];
    j12 = (uint32_t)chachaBlock;
    j13 = (uint32_t)(chachaBlock >> 32);
    j14 = input[14];
    j15 = input[15];

    x[0 ] = j0;
    x[1 ] = j1;
    x[2 ] = j2;
    x[3 ] = j3;
    x[4 ] = j4;
    x[5 ] = j5;
    x[6 ] = j6;
    x[7 ] = j7;
    x[8 ] = j8;
    x[9 ] = j9;
    x[10] = j10;
    x[11] = j11;
    x[12] = j12;
    x[13] = j13;
    x[14] = j14;
    x[15] = j15;

    #pragma unroll
    for( int i = 8; i > 0; i -= 2 )
    {
        QUARTERROUND( x[0], x[4], x[8 ], x[12] );
        QUARTERROUND( x[1], x[5], x[9 ], x[13] );
        QUARTERROUND( x[2], x[6], x[10], x[14] );
        QUARTERROUND( x[3], x[7], x[11], x[15] );
        QUARTERROUND( x[0], x[5], x[10], x[15] );
        QUARTERROUND( x[1], x[6], x[11], x[12] );
        QUARTERROUND( x[2], x[7], x[8 ], x[13] );
        QUARTERROUND( x[3], x[4], x[9 ], x[14] );
    }

    x[0 ] = CuBSwap32( PLUS( x[0 ], j0  ) );
    x[1 ] = CuBSwap32( PLUS( x[1 ], j1  ) );
    x[2 ] = CuBSwap32( PLUS( x[2 ], j2  ) );
    x[3 ] = CuBSwap32( PLUS( x[3 ], j3  ) );
    x[4 ] = CuBSwap32( PLUS( x[4 ], j4  ) );
    x[5 ] = CuBSwap32( PLUS( x[5 ], j5  ) );
    x[6 ] = CuBSwap32( PLUS( x[6 ], j6  ) );
    x[7 ] = CuBSwap32( PLUS( x[7 ], j7  ) );
    x[8 ] = CuBSwap32( PLUS( x[8 ], j8  ) );
    x[9 ] = CuBSwap32( PLUS( x[9 ], j9  ) );
    x[10] = CuBSwap32( PLUS( x[10], j10 ) );
    x[11] = CuBSwap32( PLUS( x[11], j11 ) );
    x[12] = CuBSwap32( PLUS( x[12], j12 ) );
    x[13] = CuBSwap32( PLUS( x[13], j13 ) );
    x[14] = CuBSwap32( PLUS( x[14], j14 ) );
    x[15] = CuBSwap32( PLUS( x[15], j15 ) );

    // Distribute our values to their buckets
    __shared__ uint32 sharedBucketCounts[BBCU_BUCKET_COUNT];

    {
        constexpr uint32 xShift      = BBCU_K - kExtraBits;
        constexpr uint32 bucketShift = BBCU_K - BBC_BUCKET_BITS;

        uint32 offsets[16];
        // #TODO: Above 128 threads we need to loop to set the others that need to be zeroed out
        if( id < BBCU_BUCKET_COUNT )
        {
            sharedBucketCounts[id] = 0;
        
        #if BBCU_BUCKET_COUNT > 128
            sharedBucketCounts[CHACHA_BLOCKS_PER_CUDA_BLOCK+id] = 0;
        #endif
        }

        // Record local offsets to the shared bucket count
        __syncthreads();

        #pragma unroll
        for( uint32 i = 0; i < 16; i++ )
            offsets[i] = atomicAdd( &sharedBucketCounts[x[i] >> bucketShift], 1 );

        __syncthreads();

        // Store global bucket counts, from the block-shared count,
        // and get the block-wide offsets into the destination bucket slice
        CUDA_ASSERT( gridDim.x >= BBCU_BUCKET_COUNT );
        if( id < BBCU_BUCKET_COUNT )
        {
            sharedBucketCounts[id] = atomicAdd( &gBucketCounts[id], sharedBucketCounts[id] );

            #if BBCU_BUCKET_COUNT > 128
                const uint32 id2 = CHACHA_BLOCKS_PER_CUDA_BLOCK + id;
                sharedBucketCounts[id2] = atomicAdd( &gBucketCounts[id2], sharedBucketCounts[id2] );
            #endif
        }

        __syncthreads();


        const uint32 xOffset = (uint32)(chachaBlock * 16);
        
        const uint32 yBits = (uint32)( BBC_Y_BITS - BBC_BUCKET_BITS );
        const uint32 yMask = (uint32)((1ull << yBits) - 1);

        #pragma unroll
        for( uint32 i = 0; i < 16; i++ )
        {
            const uint32 y      = x[i];
            const uint32 bucket = y >> bucketShift;
            const uint32 xo     = xOffset + i;

            CUDA_ASSERT( bucket < BBCU_BUCKET_COUNT );

            const uint32 offsetInSlice = sharedBucketCounts[bucket] + offsets[i];
            const uint32 sliceOffset   = bucket * (uint32)BBCU_MAX_SLICE_ENTRY_COUNT;
            // const uint32 sliceOffsetX  = bucket * (uint32)BBCU_META_SLICE_ENTRY_COUNT;
            const uint32 dst           = sliceOffset  + offsetInSlice;
            // const uint32 dstX          = sliceOffsetX + offsetInSlice;

            outY[dst] = ((y << kExtraBits) | (xo >> xShift)) & yMask;
            outX[dst] = xo;
        }
    }
}

//-----------------------------------------------------------
void GenF1Cuda( CudaK32PlotContext& cx )
{
    const uint32 k                      = BBCU_K;
    const uint64 bucketEntryCount       = (1ull << k) / BBCU_BUCKET_COUNT;
    const uint32 f1EntriesPerBlock      = kF1BlockSize / sizeof( uint32 );
    const uint32 chachaBucketBlockCount = (uint32)(bucketEntryCount / f1EntriesPerBlock);
    const int32  cudaBlockCount         = (int32)(chachaBucketBlockCount / CHACHA_BLOCKS_PER_CUDA_BLOCK);
    ASSERT( (uint64)cudaBlockCount * CHACHA_BLOCKS_PER_CUDA_BLOCK * f1EntriesPerBlock == bucketEntryCount );


    uint32* devBucketCounts = cx.devSliceCounts;
    uint32* devChaChaInput  = cx.devChaChaInput;

    const size_t bucketCopySize = BBCU_BUCKET_ALLOC_ENTRY_COUNT * sizeof( uint32 );

    // Init chacha context
    byte key[32] = { 1 };
    memcpy( key + 1, cx.plotRequest.plotId, 32 - 1 );
    
    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, nullptr );
    
    Log::Line( "Marker Set to %d", 94)
CudaErrCheck( cudaMemcpyAsync( devChaChaInput, chacha.input, 64, cudaMemcpyHostToDevice, cx.computeStream ) );
    Log::Line( "Marker Set to %d", 95)
CudaErrCheck( cudaMemsetAsync( devBucketCounts, 0, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT, cx.computeStream ) );
   
    const uint32 outIndex = CudaK32PlotGetOutputIndex( cx );

    uint32* hostY    = cx.hostY;
    uint32* hostMeta = cx.hostMeta;

    for( uint32 bucket = 0; bucket < BBCU_BUCKET_COUNT; bucket++ )
    {
        cx.bucket = bucket;
        const uint32 chachaBlockIdx = bucket * chachaBucketBlockCount;
        
        uint32* devY    = (uint32*)cx.yOut   .LockDeviceBuffer( cx.computeStream );
        uint32* devMeta = (uint32*)cx.metaOut.LockDeviceBuffer( cx.computeStream );

        #if _DEBUG
            Log::Line( "Marker Set to %d", 96)
CudaErrCheck( cudaMemsetAsync( devY, 0, sizeof( uint32 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT, cx.computeStream ) );
        #endif

        // Gen chacha blocks
        get_keystream_gpu<<<cudaBlockCount, CHACHA_BLOCKS_PER_CUDA_BLOCK, 0, cx.computeStream>>>( devChaChaInput, chachaBlockIdx, devY, devMeta, devBucketCounts );
        CudaK32PlotDownloadBucket( cx );

        devBucketCounts += BBCU_BUCKET_COUNT;
    }

    // Copy bucket slices to host
    Log::Line( "Marker Set to %d", 97)
CudaErrCheck( cudaMemcpyAsync( cx.hostBucketSlices, cx.devSliceCounts, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT,
                        cudaMemcpyDeviceToHost, cx.computeStream ) );

    Log::Line( "Marker Set to %d", 98)
CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );

    memcpy( &cx.bucketSlices[0], cx.hostBucketSlices, sizeof( uint32 ) * BBCU_BUCKET_COUNT * BBCU_BUCKET_COUNT );

    // Count-up bucket counts
    for( uint32 i = 0; i < BBCU_BUCKET_COUNT; i++ )
        for( uint32 j = 0; j < BBCU_BUCKET_COUNT; j++ )
            cx.bucketCounts[(int)0][i] += cx.bucketSlices[0][j][i];

    cx.tableEntryCounts[0] = 1ull << BBCU_K;

    // Ensure last bucket finished downloading
    cx.yOut   .WaitForCompletion();
    cx.metaOut.WaitForCompletion();
    cx.yOut   .Reset();
    cx.metaOut.Reset();

    if( cx.cfg.hybrid16Mode )
    {
        cx.diskContext->yBuffer->Swap();
        cx.diskContext->metaBuffer->Swap();
    }
}

///
/// DEBUG
///

//-----------------------------------------------------------
uint64 CudaPlotK32DbgXtoF1( CudaK32PlotContext& cx, const uint32 x )
{
    constexpr uint32 xShift = BBCU_K - kExtraBits;
    constexpr uint32 yBits = (uint32)( BBC_Y_BITS - BBC_BUCKET_BITS );
    constexpr uint32 yMask = (uint32)((1ull << yBits) - 1);

    byte key[32] = { 1 };
    memcpy( key + 1, cx.plotRequest.plotId, 32 - 1 );

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    uint64 chachaBlock  = x / 16;
    uint32 indexInBlock = x - chachaBlock * 16;
    uint32 blocks[16];

    chacha8_get_keystream( &chacha, chachaBlock, 1, (byte*)blocks );

    uint64 y = Swap32( blocks[indexInBlock] );

    y = ((y << kExtraBits) | (x >> xShift)) & yMask;
    return y;
}

#if _DEBUG
static ThreadPool* _dbgPool = nullptr;
//-----------------------------------------------------------
static void DbgValidateBucket( CudaK32PlotContext& cx, const uint32 bucket )
 {
    if( _dbgPool == nullptr )
        _dbgPool = new ThreadPool( SysHost::GetLogicalCPUCount() );


    Log::Line( "Validating bucket %u", bucket );
    AnonMTJob::Run( *_dbgPool, [&cx, bucket]( AnonMTJob* self ) {

        // divide-up slices between threads
        uint64 _, sliceOffset, sliceEnd;
        GetThreadOffsets( self, (uint64)BBCU_BUCKET_COUNT, _, sliceOffset, sliceEnd );

        for( uint64 slice = sliceOffset; slice < sliceEnd; slice++ )
        {
            const uint64  offset = (uint64)slice * BBCU_BUCKET_ALLOC_ENTRY_COUNT + bucket * BBCU_MAX_SLICE_ENTRY_COUNT;
            const uint32* py     = cx.hostY    + offset;
            const uint32* px     = cx.hostMeta + offset * BBCU_HOST_META_MULTIPLIER;

            const uint32 sliceSize = cx.bucketSlices[0][slice][bucket];

            for( uint32 j = 0; j < sliceSize; j++ )
            {
                const uint32 y = py[j];
                const uint32 x = px[j];

                const uint32 oy = (uint32)CudaPlotK32DbgXtoF1( cx, x );
                ASSERT( oy == y );
            }
        }
    });

    Log::Line( " OK" );
            
    //for( uint64 slice = 0; slice < 64; slice++ )
    //{
    //    const uint64  offset = (uint64)i * BBCU_BUCKET_ALLOC_ENTRY_COUNT + slice * BBCU_MAX_SLICE_ENTRY_COUNT;
    //    const uint32* py     = cx.hostY    + offset;
    //    const uint32* px     = cx.hostMeta + offset * BBCU_HOST_META_MULTIPLIER;

    //    const uint32 sliceSize = cx.bucketSlices[outIndex][i][slice];

    //    for( uint32 j = 0; j < sliceSize; j++ )
    //    {
    //        const uint32 y = py[j];
    //        const uint32 x = px[j];

    //        const uint32 oy = (uint32)CudaPlotK32DbgXtoF1( cx, x );
    //        ASSERT( oy == y );
    //    }
    //}
}
#endif
