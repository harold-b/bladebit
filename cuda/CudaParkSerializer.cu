#include "CudaParkSerializer.h"
#include "CudaFSE.cuh"


//-----------------------------------------------------------
void InitFSEBitMask( CudaK32PlotContext& cx )
{
    static bool _initialized = false;
    if( _initialized )
        return;

    _initialized = true;

    uint32 bitmask[] = {
        0,          1,         3,         7,         0xF,       0x1F,
        0x3F,       0x7F,      0xFF,      0x1FF,     0x3FF,     0x7FF,
        0xFFF,      0x1FFF,    0x3FFF,    0x7FFF,    0xFFFF,    0x1FFFF,
        0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,  0x3FFFFF,  0x7FFFFF,
        0xFFFFFF,   0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF,
        0x3FFFFFFF, 0x7FFFFFFF
    };

    Log::Line( "Marker Set to %d", 3)
CudaErrCheck( cudaMemcpyToSymbolAsync( CUDA_FSE_BIT_mask, bitmask, sizeof( bitmask ), 0, cudaMemcpyHostToDevice, cx.computeStream ) );
    Log::Line( "Marker Set to %d", 4)
CudaErrCheck( cudaStreamSynchronize( cx.computeStream ) );
}


//-----------------------------------------------------------
void CompressToParkInGPU( const uint32 parkCount, const size_t parkSize, 
    uint64* devLinePoints, byte* devParkBuffer, const size_t parkBufferSize, 
    const uint32 stubBitSize, const FSE_CTable* devCTable, uint32* devParkOverrunCount, cudaStream_t stream )
{
    const uint32 kThreadCount = 256;
    const uint32 kBlocks      = CDivT( parkCount, kThreadCount );
    CudaCompressToPark<<<kBlocks, kThreadCount, 0, stream>>>( parkCount, parkSize, devLinePoints, devParkBuffer, parkBufferSize, stubBitSize, devCTable, devParkOverrunCount );
}

//-----------------------------------------------------------
__global__ void CudaCompressToPark( 
    const uint32 parkCount, const size_t parkSize, 
    uint64* linePoints, byte* parkBuffer, const size_t parkBufferSize,
    const uint32 stubBitSize, const FSE_CTable* cTable, uint32* gParkOverrunCount )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    if( gid >= parkCount )
        return;

    linePoints += kEntriesPerPark * (size_t)gid;
    parkBuffer += parkBufferSize  * (size_t)gid;

    // __shared__ uint16 sharedCTable[34812/2];


    CUDA_ASSERT( (uintptr_t)parkBuffer / sizeof( uint64 ) * sizeof( uint64 ) == (uintptr_t)parkBuffer ); // Must be 64-bit aligned
    uint64* writer = (uint64*)parkBuffer;

    // Write the first LinePoint as a full LinePoint
    uint64 prevLinePoint = linePoints[0];

    *writer++ = CuBSwap64( prevLinePoint );

    // Grab the writing location after the stubs
    const size_t stubSectionBytes = CuCDiv( (kEntriesPerPark - 1) * (size_t)stubBitSize, 8 );

    byte* deltaBytesWriter = ((byte*)writer) + stubSectionBytes;

    // Write stubs
    {
        const uint64 stubMask = ((1ULL << stubBitSize) - 1);

        uint64 field = 0;   // Current field to write
        uint   bits  = 0;   // Bits occupying the current field (always shifted to the leftmost bits)

        #pragma unroll
        for( uint32 i = 1; i < kEntriesPerPark; i++ )
        {
            const uint64 lpDelta = linePoints[i];
            const uint64 stub    = lpDelta & stubMask;

            // Serialize into bits, one uint64 field at a time
            // Always store it all the way to the MSbits
            const uint freeBits = 64 - bits;
            if( freeBits <= stubBitSize )
            {
                // Update the next field bits to what the stub bits that were not written into the current field
                bits = (uint32)stubBitSize - freeBits;

                // Write what we can (which may be nothing) into the free bits of the current field
                field |= stub >> bits;

                // Store field
                *writer++ = CuBSwap64( field );

                const uint remainder = 64 - bits;
                uint64 mask = ( ( 1ull << bits ) - 1 ) << (remainder & 63);
                field = ( stub << remainder ) & mask;
            }
            else
            {
                // The stub completely fits into the current field with room to spare
                field |= stub << (freeBits - stubBitSize);
                bits += stubBitSize;
            }
        }

        // Write any trailing fields
        if( bits > 0 )
            *writer++ = CuBSwap64( field );

        // Zero-out any remaining unused bytes
        // const size_t stubUsedBytes  = CDiv( (kEntriesPerPark - 1) * (size_t)stubBitSize, 8 );
        // const size_t remainderBytes = stubSectionBytes - stubUsedBytes;
        
        // memset( deltaBytesWriter - remainderBytes, 0, remainderBytes );
    }
    
    
    // Convert to small deltas
    byte* smallDeltas = (byte*)&linePoints[1];
    
    #pragma unroll
    for( uint32 i = 1; i < kEntriesPerPark; i++ )
    {
        const uint64 smallDelta = linePoints[i] >> stubBitSize;
        CUDA_ASSERT( smallDelta < 256 );

        smallDeltas[i-1] = (byte)smallDelta;
    }

    // Write small deltas
    size_t parkSizeWritten = 0;
    {
        byte* deltaSizeWriter = (byte*)deltaBytesWriter;
        deltaBytesWriter += 2;

        // CUDA_ASSERT( smallDeltas[0] == 3 );
        size_t deltasSize = CUDA_FSE_compress_usingCTable<kEntriesPerPark-1>(
                                deltaBytesWriter, (kEntriesPerPark-1) * 8,
                                smallDeltas, kEntriesPerPark-1, cTable );

        if( deltasSize == 0 )
        {
            // #TODO: Set error
            CUDA_ASSERT( 0 );
        }
        else
        {
            // Deltas were compressed
            
            //memcpy( deltaSizeWriter, &deltasSize, sizeof( uint16 ) );
            // *deltaSizeWriter = (uint16)deltasSize;
            deltaSizeWriter[0] = (byte)( deltasSize ); // Stored as LE
            deltaSizeWriter[1] = (byte)( deltasSize >> 8 );
        }

        if( ((deltaBytesWriter + deltasSize) - parkBuffer)  > parkSize )
        {
            *gParkOverrunCount++;
        }
// #if _DEBUG
        // deltaBytesWriter += deltasSize;
        // parkSizeWritten = deltaBytesWriter - parkBuffer;

        // if( parkSizeWritten > parkSize )
            // printf( "[CUDA KERN ERROR] Overran park buffer: %llu / %llu\n", parkSizeWritten, parkSize );
        // CUDA_ASSERT( parkSizeWritten <= parkSize );
// #endif
        
        // Zero-out any remaining bytes in the deltas section
        // const size_t parkSizeRemainder = parkSize - parkSizeWritten;

        // memset( deltaBytesWriter, 0, parkSizeRemainder );
    }

    // return parkSizeWritten;
}


// #TODO: Check if deltafying in a different kernel would be good
//-----------------------------------------------------------
__global__ void CudaCompressC3Park( const uint32 parkCount, uint32* f7Entries, byte* parkBuffer, const size_t c3ParkSize, const FSE_CTable* cTable )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    if( gid >= parkCount )
        return;

    f7Entries  += gid * kCheckpoint1Interval;
    parkBuffer += gid * c3ParkSize;

    byte* deltaWriter = (byte*)f7Entries;

    // Convert to deltas

    // f7Entries must always start at an interval of kCheckpoint1Interval
    // Therefore its first entry is a C1 entry, and not written as a delta.
    uint32 prevF7 = *f7Entries;
    
    #pragma unroll
    for( uint32 i = 1; i < kCheckpoint1Interval; i++ )
    {
        const uint32 f7    = f7Entries[i];
        const uint32 delta = f7 - prevF7;
        prevF7 = f7;

        CUDA_ASSERT( delta < 255 );
        *deltaWriter++ = (byte)delta;
    }

    CUDA_ASSERT( (uintptr_t)(deltaWriter - (byte*)f7Entries) == kCheckpoint1Interval-1 );

    // Serialize them into the C3 park buffer
    const size_t compressedSize = CUDA_FSE_compress_usingCTable<kCheckpoint1Interval-1>(
        parkBuffer+2, c3ParkSize, (byte*)f7Entries,
        kCheckpoint1Interval-1, cTable );
    
    CUDA_ASSERT( (compressedSize+2) < c3ParkSize );
    CUDA_ASSERT( (compressedSize+2) < 3000 );
    
    // Store size in the first 2 bytes
    //memcpy( parkBuffer, &sizeu16, sizeof( uint16)  );
    parkBuffer[0] = (byte)( compressedSize >> 8 ); // Stored as BE
    parkBuffer[1] = (byte)( compressedSize );
}

//-----------------------------------------------------------
void CompressC3ParksInGPU( const uint32 parkCount, uint32* devF7, byte* devParkBuffer, 
                           const size_t parkBufSize, const FSE_CTable* cTable, cudaStream_t stream )
{
    const uint32 kthreads = 128;
    const uint32 kblocks  = CDiv( parkCount, kthreads );

    CudaCompressC3Park<<<kblocks, kthreads, 0, stream>>>( parkCount, devF7, devParkBuffer, parkBufSize, cTable );
}


//-----------------------------------------------------------
__global__ void CudaWritePark7( const uint32 parkCount, const uint32* indices, uint64* fieldWriter, const size_t parkFieldCount )
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    if( gid >= parkCount )
        return;

    indices     += gid * kEntriesPerPark;
    fieldWriter += gid * parkFieldCount;

    const uint32 bitsPerEntry = BBCU_K + 1;

    uint64 field = 0;
    uint32 bits  = 0;

    #pragma unroll
    for( int32 i = 0; i < kEntriesPerPark; i++ )
    {
        const uint64 index    = indices[i];
        const uint32 freeBits = 64 - bits;

        // Filled a field?
        if( freeBits <= bitsPerEntry )
        {
            bits = bitsPerEntry - freeBits;
            field |= index >> bits;

            // Store field
            *fieldWriter++ = CuBSwap64( field );

            const uint remainder = 64 - bits;
            uint64 mask = ( ( 1ull << bits ) - 1 ) << (remainder & 63);
            field = ( index << remainder ) & mask;
        }
        else
        {
            // The entry completely fits into the current field with room to spare
            field |= index << ( freeBits - bitsPerEntry );
            bits += bitsPerEntry;
        }
    }

    // Write any trailing fields
    if( bits > 0 )
        *fieldWriter = CuBSwap64( field );
}

//-----------------------------------------------------------
void SerializePark7InGPU( const uint32 parkCount, const uint32* indices, uint64* fieldWriter, const size_t parkFieldCount, cudaStream_t stream )
{
    const uint32 kthreads = 256;
    const uint32 kblocks  = CDiv( parkCount, kthreads );

    CudaWritePark7<<<kblocks, kthreads, 0, stream>>>( parkCount, indices, fieldWriter, parkFieldCount );
}
