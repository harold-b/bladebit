#include "CudaPlotContext.h"
#include "ChiaConsts.h"
#include "CudaMatch.h"

#define CU_MATCH_THREAD_COUNT (kExtraBitsPow)

#define BBCU_SCAN_GROUP_THREADS      128
#define BBCU_THREADS_PER_MATCH_GROUP 352
static constexpr uint32 BBCU_MAX_ENTRIES_PER_GROUP = 238;
static constexpr uint32 BBCU_MIN_ENTRIES_PER_GROUP = 230;
static constexpr uint32 BBCU_MIN_GROUP_COUNT       = ( CuCDiv( BBCU_BUCKET_ENTRY_COUNT, BBCU_MAX_ENTRIES_PER_GROUP ) );
static constexpr uint32 BBCU_MAX_GROUP_COUNT       = ( CuCDiv( BBCU_BUCKET_ENTRY_COUNT, BBCU_MIN_ENTRIES_PER_GROUP ) );

static_assert( CU_MAX_BC_GROUP_BOUNDARIES >= BBCU_MAX_GROUP_COUNT );

// #NOTE: The above have been tuned for 128 buckets, should check them for other bucket counts.
//static_assert( BBCU_BUCKET_COUNT == 128, "Unexpected bucket count" );

//-----------------------------------------------------------
__forceinline__ __device__ uint16 GenLTarget( const uint16 parity, const uint16 rTargetIdx, const uint16 m )
{
    const uint16 indJ = rTargetIdx / kC;
    return ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + rTargetIdx) % kC);
}

//-----------------------------------------------------------
__global__ void CudaInitGroupsBucket( uint32* entries )
{
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    entries[gid] = 0xFFFFFFFF;
}

//-----------------------------------------------------------
__global__ void CudaInitGroups( uint32* entries, const uint32 entryCount )
{
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    if( gid >= entryCount )
        return;

    entries[gid] = 0xFFFFFFFF;
}

//-----------------------------------------------------------
__global__ void CudaSetFirstAndLastGroup( uint32* groups, const uint32 entryCount )
{
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    if( id == 0 )
        groups[id] = 0;
    else
        groups[id] = entryCount;
}

//-----------------------------------------------------------
__global__ void ScanGroupsCudaK32Bucket( const uint32* yEntries, uint32* groupBounadries, uint32* gGroupCount, const uint32 entryCount, const uint64 bucketMask )
{
    const uint32 id           = threadIdx.x;
    const uint32 gid          = blockIdx.x * blockDim.x + id;

    if( gid >= entryCount-1 )
        return;
    
    __shared__ uint32 sharedGroupCount;
    if( id == 0 )
        sharedGroupCount = 0;

    __syncthreads();

    const uint64 currentGroup = ( bucketMask | yEntries[gid]   ) / kBC;
    const uint64 nextGroup    = ( bucketMask | yEntries[gid+1] ) / kBC;

    uint32 offset;
    if( currentGroup != nextGroup )
    {
        // #TODO: Use cooperative groups here instead, so we can just sync these threads
        offset = atomicAdd( &sharedGroupCount, 1 );
    }

    __syncthreads();

    // Global sync
    if( id == 0 )
        sharedGroupCount = atomicAdd( gGroupCount, sharedGroupCount );

    __syncthreads();

    if( currentGroup != nextGroup )
    {
        CUDA_ASSERT( sharedGroupCount + offset < CU_MAX_BC_GROUP_BOUNDARIES );
        groupBounadries[sharedGroupCount + offset] = gid+1;
    }
}

//-----------------------------------------------------------
__global__ void MatchCudaK32Bucket( const uint64 bucketMask, const uint32 entryCount, const uint32* gGroupCounts, const uint32* yEntries, const uint32* groupBoundaries, uint32* gMatchCount, Pair* outPairs )
{
    // 1 thread per y
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    if( groupIdx >= *gGroupCounts )
        return;

    const uint32 groupLIdx    = groupBoundaries[groupIdx];
    const uint32 groupRIdx    = groupBoundaries[groupIdx+1];
    const uint32 groupREnd    = groupBoundaries[groupIdx+2];
    const uint64 groupL       = ( bucketMask | yEntries[groupLIdx] ) / kBC;
    const uint64 groupR       = ( bucketMask | yEntries[groupRIdx] ) / kBC;
    const uint32 groupRLength = groupREnd - groupRIdx;
    const uint64 groupLYStart = groupL * kBC;
    const uint64 groupRYStart = groupR * kBC;
    const uint32 groupLLength = groupRIdx - groupLIdx;

#if _DEBUG
    if( groupLLength >= BBCU_THREADS_PER_MATCH_GROUP || groupRLength >= BBCU_THREADS_PER_MATCH_GROUP )
        printf( "[%u] Group %u is too large: %u\n", gid, groupRIdx, ( groupRIdx - groupLIdx ) );
#endif
    CUDA_ASSERT( groupLLength <= BBCU_THREADS_PER_MATCH_GROUP );
    CUDA_ASSERT( groupRLength <= BBCU_THREADS_PER_MATCH_GROUP );

    // Generate R group map
    __shared__ uint32 rMap[kBC/2+1];
    __shared__ uint32 sharedMatchCount;

    if( groupR - groupL != 1 )
        return;

    if( id == 0 )
        sharedMatchCount = 0;

    const uint16 localLY = (uint16)(( bucketMask | yEntries[groupLIdx + min(id, groupLLength-1)] ) - groupLYStart );
    const uint16 localRY = (uint16)(( bucketMask | yEntries[groupRIdx + min(id, groupRLength-1)] ) - groupRYStart );

    // #TODO: See about using coop_threads here
    {
        {
            uint32 mapIdx = id;
            while( mapIdx < kBC/2+1 )
            { 
                // Each entry is:
                // hi           lo
                // 7     9      7     9
                //(count,offset|count,offset) 
                rMap[mapIdx] = 0x01FF01FF;
                mapIdx += BBCU_THREADS_PER_MATCH_GROUP;
            }
        }

        __syncthreads();
        
        const uint16 shift  = ( ( localRY & 1 ) << 4 );   // Shift left by 16 bits if odd
        const uint32 idx    = localRY >> 1;               // Divide by 2

        // First set the offsets for the even ones (lower bits)
        if( id < groupRLength && ( localRY & 1 ) == 0 )
            atomicMin( &rMap[idx], id | 0x01FF0000 );
            
        __syncthreads();

        // Then set offset for the odd ones
        if( id < groupRLength && ( localRY & 1 ) )
            atomicMin( &rMap[idx], (id << 16) | (rMap[idx] & 0x0000FFFF) );

        __syncthreads();

        // Finally, add the counts
        if( id < groupRLength )
            atomicAdd( &rMap[idx], 0x200ul << shift );
    }

    if( id >= groupLLength )
        return;

    __syncthreads();


    // Begin matching
    constexpr uint32 MAX_MATCHES = 16;
    Pair matches[MAX_MATCHES];
    uint32 matchCount = 0;

    #pragma unroll
    for( uint32 i = 0; i < kExtraBitsPow; i++ )
    {
        const uint16 lTarget = GenLTarget( (byte)(groupL & 1), localLY, (uint16)i );
        const uint16 shift   = ( ( lTarget & 1 ) << 4 );   // Shift left by 16 bits if odd
        const uint16 rValue  = (uint16)(rMap[lTarget>>1] >> shift);
        const int16  rCount  = (int16)(rValue >> 9);

        for( int32 j = 0; j < rCount; j++ )
        {
            CUDA_ASSERT( matchCount < MAX_MATCHES );
            matches[matchCount++] = { groupLIdx + id, groupRIdx + (rValue & 0x1FF) + j };
        }
    }

    // Store final values
    const uint32 copyOffset = atomicAdd( &sharedMatchCount, matchCount );
    __syncthreads();
        
    // Store our shared match count and get our global offset
    if( id == 0 )
        sharedMatchCount = atomicAdd( gMatchCount, sharedMatchCount );
    __syncthreads();

    outPairs += copyOffset + sharedMatchCount;

    for( uint32 i = 0; i < matchCount; i++ )
    {
        CUDA_ASSERT( matches[i].left < entryCount );
        CUDA_ASSERT( matches[i].right < entryCount );

        outPairs[i] = matches[i];
    }
}

/// This kernel, meant for harvesting compressed k32 plots,
/// matches adjacent BC groups with 64 threads per block.
/// Where each block represents 1 L entry, and 
/// each thread is one iteration of log2( kExtraBits ) required
/// per L entry during normal matching. 
/// Since compressed groups are small, we expect this
/// to be a reasonable way to implement matching
/// vs the way is imlpemented in plotting where the group
/// sizes are exploited.
//-----------------------------------------------------------
__global__ void HarvestMatchK32Kernel(
          Pair*   gOutMatches,
          uint32* gOutMatchCount,
    const uint64* yEntries,
    const uint32  entryCount,
    const uint32  matchOffset
)
{
    const uint32 id   = threadIdx.x;
    const uint32 yIdx = blockIdx.x;
    const uint32 gid  = yIdx + id;

    CUDA_ASSERT( id < 64 );

    constexpr uint32 SHARED_R_BUF_SIZE = 64;
    constexpr uint32 MAX_MATCHES       = 16;

    // Read rGroup entries into a shared buffer
    __shared__ uint64 rBuf[SHARED_R_BUF_SIZE];
    __shared__ uint32 sharedMatchCount;
    __shared__ uint64 lYShared;

    // Find group boundaries
    __shared__ uint32 lGroupStart;
    __shared__ uint32 rGroupStartShared;
    __shared__ uint32 rGroupEnd;

    uint64 myY = 0xFFFFFFFFFFFFFFFF;
    if( gid < entryCount )
        myY = yEntries[gid];


    if( id == 0 )
    {
        lYShared          = myY;
        sharedMatchCount  = 0;
        rGroupStartShared = 0xFFFFFFFF;
        rGroupEnd         = 0xFFFFFFFF;
    }
    __syncthreads();

    const uint32 groupL  = (uint32)(lYShared / kBC);
    const uint32 myGroup = (uint32)(myY / kBC);

    if( myGroup - groupL == 1 )
        atomicMin( &rGroupStartShared, id );

    __syncthreads();

    // Not an adjacent group, exit
    if( rGroupStartShared == 0xFFFFFFFF )
        return;

    const uint32 rGroupStart = rGroupStartShared;

    // Store our read Y into shared value buffer
    rBuf[id] = myY;
    __syncthreads();

    // Now find the R group end
    // Notice we store the global index here, not the block-local one,
    // like we did for rGroupStart
    const uint32 groupR = (uint32)( rBuf[rGroupStart] / kBC);
    if( myGroup > groupR )
        atomicMin( &rGroupEnd, gid );

    __syncthreads();

    // Is it the last R group?
    if( id == 0 && rGroupEnd == 0xFFFFFFFF )
        rGroupEnd = entryCount;

    __syncthreads();
    CUDA_ASSERT( rGroupEnd < 0xFFFFFFFF );

    // We should have all the info we need to match this Y now
    const uint32 rGroupLength = rGroupEnd - (yIdx + rGroupStart);

    const uint64 lY           = lYShared;
    const uint64 groupLYStart = ((uint64)groupL) * kBC;
    const uint64 groupRYStart = ((uint64)groupR) * kBC;
    const uint16 localLY      = (uint16)(lY - groupLYStart);

    const uint16 lTarget      = GenLTarget( (byte)(groupL & 1), localLY, (uint16)id );

    Pair   matches[MAX_MATCHES];
    uint32 matchCount = 0;

    #pragma unroll
    for( uint32 i = rGroupStart; i < (rGroupStart+rGroupLength); i++ )
    {
        const uint64 rY      = rBuf[i];
        const uint16 localRY = (uint16)(rY - groupRYStart);

        if( lTarget == localRY )
        {
            CUDA_ASSERT( matchCount <= MAX_MATCHES );
            matches[matchCount++] = { matchOffset + yIdx, matchOffset + yIdx + i };
        }
    }

    // Store matches into global memory
    const uint32 offset = atomicAdd( &sharedMatchCount, matchCount );

    __syncthreads();
    if( sharedMatchCount == 0 )
        return;

    if( id == 0 )
        sharedMatchCount = atomicAdd( gOutMatchCount, sharedMatchCount );

    __syncthreads();

    // Copy matches to global buffer
    const uint32 out = sharedMatchCount + offset;

    for( uint32 i = 0; i < matchCount; i++ )
        gOutMatches[out+i] = matches[i];
}

//-----------------------------------------------------------
__global__ void MatchCudaK32KernelInternal( 
    Pair*         outPairs,
    uint32*       gMatchCount,
    const uint32  entryCount,
    const uint32* gGroupCounts,
    const uint64* yEntries,
    const uint32* groupBoundaries )
{
    // 1 thread per y
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    if( groupIdx >= *gGroupCounts )
        return;

    const uint32 groupLIdx    = groupBoundaries[groupIdx];
    const uint32 groupRIdx    = groupBoundaries[groupIdx+1];
    const uint32 groupREnd    = groupBoundaries[groupIdx+2];
    const uint64 groupL       = yEntries[groupLIdx] / kBC;
    const uint64 groupR       = yEntries[groupRIdx] / kBC;
    const uint32 groupRLength = groupREnd - groupRIdx;
    const uint64 groupLYStart = groupL * kBC;
    const uint64 groupRYStart = groupR * kBC;
    const uint32 groupLLength = groupRIdx - groupLIdx;

#if _DEBUG
    if( groupLLength >= BBCU_THREADS_PER_MATCH_GROUP || groupRLength >= BBCU_THREADS_PER_MATCH_GROUP )
        printf( "[%u] Group %u is too large: %u\n", gid, groupRIdx, ( groupRIdx - groupLIdx ) );
#endif
    CUDA_ASSERT( groupLLength <= BBCU_THREADS_PER_MATCH_GROUP );
    CUDA_ASSERT( groupRLength <= BBCU_THREADS_PER_MATCH_GROUP );

    // Each rMap entry is:
    // hi           lo
    // 7     9      7     9
    //(count,offset|count,offset) 
    uint32 rMap[kBC/2+1] = {};

    __shared__ uint32 sharedMatchCount;
    if( id == 0 )
        sharedMatchCount = 0;
    __syncthreads();

    if( groupR - groupL != 1 )
        return;

    const uint16 localLY     = (uint16)( yEntries[groupLIdx + min(id, groupLLength-1)] - groupLYStart );
    const uint16 localRYBase = (uint16)( yEntries[groupRIdx + min(id, groupRLength-1)] - groupRYStart );

    // Packed rMap. 2 entries (of count and offset) per DWORD
    for( uint32 i = 0; i < groupRLength; i++ )
    {
        const uint16 localRY = localRYBase + (uint16)i;

        const uint32 idx   = localRY >> 1; // Index in the rMap (Divide by 2)
        const uint32 value = rMap[idx];

        // Increase the count and sets the index
        if( (localRY & 1) == 0 )
        {
            // Even value, store in the LSbits
            rMap[idx] = (value + 0x200) | i;
        }
        else
        {
            // Odd value, store in the MSbits
            rMap[idx] = (value + 0x2000000) | (i << 16);
        }
    }
    __syncthreads();


    // Begin matching
    constexpr uint32 MAX_MATCHES = 16;
    Pair   matches[MAX_MATCHES];
    uint32 matchCount = 0;

    #pragma unroll
    for( uint32 i = 0; i < kExtraBitsPow; i++ )
    {
        const uint16 lTarget = GenLTarget( (byte)(groupL & 1), localLY, (uint16)i );
        const uint16 shift   = ( ( lTarget & 1 ) << 4 );   // Shift left by 16 bits if odd
        const uint16 rValue  = (uint16)(rMap[lTarget>>1] >> shift);
        const int16  rCount  = (int16)(rValue >> 9);

        for( int32 j = 0; j < rCount; j++ )
        {
            if( matchCount >= MAX_MATCHES )
            {
                printf( "[%u] We got too many (i=%u) matches: %u\n", gid, i, matchCount );
            }
            CUDA_ASSERT( matchCount < MAX_MATCHES );
            matches[matchCount++] = { groupLIdx + id, groupRIdx + (rValue & 0x1FF) + j };
        }
    }

    // Store final values
    const uint32 copyOffset = atomicAdd( &sharedMatchCount, matchCount );
    __syncthreads();
        
    // Store our shared match count and get our global offset
    if( id == 0 )
        sharedMatchCount = atomicAdd( gMatchCount, sharedMatchCount );
    __syncthreads();

    outPairs += copyOffset + sharedMatchCount;

    for( uint32 i = 0; i < matchCount; i++ )
    {
        CUDA_ASSERT( matches[i].left < entryCount );
        CUDA_ASSERT( matches[i].right < entryCount );

        outPairs[i] = matches[i];
    }
}

//-----------------------------------------------------------
__global__ void MatchK32Kernel( 
    Pair*         outPairs,
    uint32*       gMatchCount,
    const uint32  entryCount,
    const uint32* gGroupCounts,
    const uint64* yEntries,
    const uint32* groupBoundaries )
{
    // CUDA_ASSERT( blockDim.x == 1 );
    // CUDA_ASSERT( blockIdx.x == 0 );
    // CUDA_ASSERT( threadIdx.x == 0 );


    // const uint32 groupCount      = *gGroupCounts;
    // const uint32 entriesPerGroup = (entryCount / groupCount) + 6;

    // const uint32 blocks  = groupCount;
    // const uint32 threads = entriesPerGroup;

    // HarvestMatchK32Kernel<<<blocks, 64>>>(
    //       gMatchCount,
    // const uint32  lGroupIdx,
    // const uint32  lYIdx,
    // const uint32  rGroupIdx,
    // const uint32  rGroupLength,
    // const uint64* yEntries
    
    // MatchCudaK32KernelInternal<<<blocks, threads>>>( outPairs, gMatchCount, entryCount, gGroupCounts, yEntries, groupBoundaries );
    
    // const cudaError_t err = cudaGetLastError();
    // assert( err == cudaSuccess );
}

//-----------------------------------------------------------
__global__ void ScanGroupsK32Kernel( 
    uint32*       groupIndices,
    uint32*       outGroupCount,
    const uint64* yEntries,
    const uint32  entryCount )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    __shared__ uint32 sharedGroupCount;

    if( id == 0 ) 
        sharedGroupCount = 0;
    __syncthreads();

    if( gid >= entryCount-1 )
        return;

    const uint32 currentGroup = (uint32)(yEntries[gid] / kBC);
    const uint32 nextGroup    = (uint32)(yEntries[gid+1] / kBC);

    uint32 offset;
    if( currentGroup != nextGroup )
        offset = atomicAdd( &sharedGroupCount, 1 );
    
    __syncthreads();

    // Global sync
    if( id == 0 )
        sharedGroupCount = atomicAdd( outGroupCount, sharedGroupCount );

    __syncthreads();

    if( currentGroup != nextGroup )
        groupIndices[sharedGroupCount + offset] = gid+1;
    // // CUDA_ASSERT( sharedGroupCount + offset < CU_MAX_BC_GROUP_BOUNDARIES );
}

//-----------------------------------------------------------
cudaError CudaHarvestMatchK32(
    Pair*         devOutPairs,
    uint32*       devMatchCount,
    const uint32  maxMatches,
    const uint64* devYEntries,
    const uint32  entryCount,
    const uint32  matchOffset,
    cudaStream_t  stream )
{
    uint32 kthreads = 64;
    uint32 kblocks  = entryCount-1;

    cudaError cErr = cudaMemsetAsync( devMatchCount, 0, sizeof( uint32 ), stream );
    if( cErr != cudaSuccess )
        return cErr;

    HarvestMatchK32Kernel<<<kblocks, kthreads, 0, stream>>>(
        devOutPairs, devMatchCount, devYEntries, entryCount, matchOffset );


    return cudaSuccess;
}


//-----------------------------------------------------------
void CudaMatchBucketizedK32( 
    CudaK32PlotContext& cx,
    const uint32*       devY,
    cudaStream_t        stream,
    cudaEvent_t         event )
{
    const TableId inTable    = cx.table - 1;
    const uint32  entryCount = cx.bucketCounts[(int)inTable][cx.bucket];
    const uint64  bucketMask = BBC_BUCKET_MASK( cx.bucket );

    constexpr uint32 kscanblocks = CuCDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, BBCU_SCAN_GROUP_THREADS );

    uint32* tmpGroupCounts = (uint32*)cx.devMatches;

    {
        // Initialize the entries to the max value so that they are not included in the sort
        CudaInitGroupsBucket<<<kscanblocks, BBCU_SCAN_GROUP_THREADS, 0, stream>>>( tmpGroupCounts );

        // Add first group and last ghost group
        CudaSetFirstAndLastGroup<<<1,2,0,stream>>>( tmpGroupCounts, entryCount );
    }

    Log::Line( "Marker Set to %d", 1)
CudaErrCheck( cudaMemsetAsync( cx.devGroupCount, 0, sizeof( uint32 ), stream ) );
    Log::Line( "Marker Set to %d", 2)
CudaErrCheck( cudaMemsetAsync( cx.devMatchCount, 0, sizeof( uint32 ), stream ) );
    ScanGroupsCudaK32Bucket<<<kscanblocks, BBCU_SCAN_GROUP_THREADS, 0, stream>>>( devY, tmpGroupCounts+2, cx.devGroupCount, entryCount, bucketMask );

    byte*  sortTmpAlloc = (byte*)( tmpGroupCounts + BBCU_MAX_GROUP_COUNT );
    size_t sortTmpSize  = ( BBCU_BUCKET_ALLOC_ENTRY_COUNT - BBCU_MAX_GROUP_COUNT ) * sizeof( uint32 );

#if _DEBUG
    size_t sortSize = 0;
    cub::DeviceRadixSort::SortKeys<uint32, uint32>( nullptr, sortSize, nullptr, nullptr, BBCU_MAX_GROUP_COUNT, 0, 32 );
    ASSERT( sortSize <= sortTmpSize );
#endif

    cub::DeviceRadixSort::SortKeys<uint32, uint32>( sortTmpAlloc, sortTmpSize, tmpGroupCounts, cx.devGroupBoundaries, BBCU_MAX_GROUP_COUNT, 0, 32, stream );

    MatchCudaK32Bucket<<<BBCU_MAX_GROUP_COUNT, BBCU_THREADS_PER_MATCH_GROUP, 0, stream>>>( bucketMask, entryCount, cx.devGroupCount, devY, cx.devGroupBoundaries, cx.devMatchCount, cx.devMatches );
}

