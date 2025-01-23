#include "GpuStreams.h"
#include "GpuQueue.h"
#include "plotting/DiskBucketBuffer.h"
#include "plotting/DiskBuffer.h"



///
/// UploadBuffer
///
void* GpuUploadBuffer::GetNextPinnedBuffer()
{
    // Wait for the pinned host buffer to be available
    //if( self->outgoingSequence > self->bufferCount-1 )
    //    self->fence.Wait( self->outgoingSequence - self->bufferCount + 1 );
    //
    const uint32 index = self->outgoingSequence % self->bufferCount;

    void* pinnedBuffer = self->pinnedBuffer[index];

    return pinnedBuffer;
}

void GpuUploadBuffer::Upload( const void* hostBuffer, size_t size, cudaStream_t workStream, bool directOverride )
{
    ASSERT( size );

    const bool isDirect = (!self->pinnedBuffer[0] || directOverride) && !self->diskBuffer;
    PanicIf( isDirect && !hostBuffer, "No host buffer provided for direct upload." );

    const uint32 index = SynchronizeOutgoingSequence();

    auto uploadStream = self->queue->GetStream();

    DiskBuffer* diskBuffer = nullptr;
    if( self->diskBuffer )
    {
        // Preload data from disk into pinned buffer

        diskBuffer = dynamic_cast<DiskBuffer*>( self->diskBuffer );
        PanicIf( !diskBuffer, "Not a DiskBucketBuffer" );
        ASSERT( diskBuffer->GetAlignedBufferSize() >= size );

        hostBuffer = self->pinnedBuffer[index];
        ASSERT( hostBuffer == diskBuffer->PeekReadBufferForBucket( self->outgoingSequence - 1 ) );
        ASSERT( self->outgoingSequence <= BBCU_BUCKET_COUNT );

        CallHostFunctionOnStream( uploadStream, [=](){
            // Read on disk queue's thread
            diskBuffer->ReadNextBucket();

            // Block until the buffer is fully read from disk
            // #TODO: Also should not do this here, but in a host-to-host background stream,
            //        so that the next I/O read can happen in the background while
            //        the previous upload to disk is happening, if needed.
            (void)diskBuffer->GetNextReadBuffer();
        });
    }
    else if( !isDirect )
    {
        // Copy from unpinned to pinned first
        // #TODO: This should be done in a different backgrund host-to-host copy stream
        Log::Line( "Marker Set to %d", 76)
CudaErrCheck( cudaStreamWaitEvent( uploadStream, self->pinnedEvent[index] ) );
        Log::Line( "Marker Set to %d", 77)
CudaErrCheck( cudaMemcpyAsync( self->pinnedBuffer[index], hostBuffer, size, cudaMemcpyHostToHost, uploadStream ) );

        hostBuffer = self->pinnedBuffer[index];
    }

    // Ensure the device buffer is ready for use
    Log::Line( "Marker Set to %d", 78)
CudaErrCheck( cudaStreamWaitEvent( uploadStream, self->deviceEvents[index] ) );

    // Upload to the device buffer
    Log::Line( "Marker Set to %d", 79)
CudaErrCheck( cudaMemcpyAsync( self->deviceBuffer[index], hostBuffer, size, cudaMemcpyHostToDevice, uploadStream ) );

    if( !isDirect )
    {
        // Signal that the pinned buffer is ready for re-use
        Log::Line( "Marker Set to %d", 80)
CudaErrCheck( cudaEventRecord( self->pinnedEvent[index], uploadStream ) );
    }

    // Signal work stream that the device buffer is ready to be used
    Log::Line( "Marker Set to %d", 81)
CudaErrCheck( cudaEventRecord( self->readyEvents[index], uploadStream ) );
}

void GpuUploadBuffer::UploadAndPreLoad( void* hostBuffer, const size_t size, const void* copyBufferSrc, const size_t copySize )
{
    ASSERT(0);
    // ASSERT( size >= copySize );

    // Upload( hostBuffer, size, nullptr );

    // // Add callback for copy
    // const uint32 sequence = self->outgoingSequence - 1;
    // auto& cpy = self->copies[sequence];
    // cpy.self            = self;
    // cpy.sequence        = sequence;
    // cpy.copy.hostBuffer = hostBuffer;
    // cpy.copy.srcBuffer  = copyBufferSrc;
    // cpy.copy.size       = copySize;

    // // Launch copy command

    //     const CopyInfo& c = *reinterpret_cast<CopyInfo*>( userData );
    //     IGpuBuffer* self = c.self;

    //     auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Copy );
    //     cmd.copy.info = &c;

    //     self->queue->SubmitCommands();
    // }, &cpy ) );
}

void GpuUploadBuffer::UploadArray( const void* hostBuffer, uint32 length, uint32 elementSize, uint32 srcStride, 
                                   uint32 countStride, const uint32* counts, cudaStream_t workStream )
{
    const uint32 index    = SynchronizeOutgoingSequence();
    const bool   isDirect = self->pinnedBuffer[0] == nullptr && !self->diskBuffer;

    auto uploadStream = self->queue->GetStream();

    DiskBucketBuffer* diskBuffer      = nullptr;
    size_t            totalBufferSize = 0;

    if( self->diskBuffer )
    {
        diskBuffer = dynamic_cast<DiskBucketBuffer*>( self->diskBuffer );
        PanicIf( !diskBuffer, "Not a DiskBucketBuffer" );

        hostBuffer = diskBuffer->PeekReadBufferForBucket( self->outgoingSequence-1 );
        ASSERT( self->outgoingSequence <= BBCU_BUCKET_COUNT );

        // if( nextReadBucket < BBCU_BUCKET_COUNT )
        {
            // Override the input slice sizes with the correct ones (as we wrote them with fixed size)

            // Preload the bucket buffer from disk
            CallHostFunctionOnStream( uploadStream, [=](){

                const uint32 nextReadBucket = diskBuffer->GetNextReadBucketId();
                diskBuffer->OverrideReadSlices( nextReadBucket, elementSize, counts, countStride );

                // Preloads in the background
                diskBuffer->ReadNextBucket();

                // Upload the next one too, if needed 
                // #NOTE: This is a hacky way to do it for now. 
                //        We ought to have a synchronized, separate, disk stream later
                // if( nextReadBucket < BBCU_BUCKET_COUNT )
                //     diskBuffer->ReadNextBucket();
            });
        }

        // Wait for disk buffer to be ready
        CallHostFunctionOnStream( uploadStream, [diskBuffer](){

            // Wait until next buffer is ready
            (void)diskBuffer->GetNextReadBuffer();
        });
    }
    else
    {
        // Perform fragmented uploads
        const auto waitEvent = isDirect ? self->deviceEvents[index] : self->pinnedEvent[index];
        const auto copyMode  = isDirect ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;

        // Wait on device or pinned buffer to be ready (depending if a direct copy or not)
        Log::Line( "Marker Set to %d", 82)
CudaErrCheck( cudaStreamWaitEvent( uploadStream, waitEvent ) );

        const byte*   src   = (byte*)hostBuffer;
              byte*   dst   = (byte*)( isDirect ? self->deviceBuffer[index] : self->pinnedBuffer[index] );
        const uint32* sizes = counts;

        for( uint32 i = 0; i < length; i++ )
        {
            const size_t size = *sizes * (size_t)elementSize;

            Log::Line( "Marker Set to %d", 83)
CudaErrCheck( cudaMemcpyAsync( dst, src, size, copyMode, uploadStream ) );

            dst    += size;
            src    += srcStride;
            sizes += countStride;
        }

        if( !isDirect )
        {
            // Set the pinned buffer as the host buffer so that we can do a sequential copy to the device now
            hostBuffer = self->pinnedBuffer[index];
        }
    }

    // Upload to device buffer if in non-direct mode
    if( !isDirect )
    {
        for( uint32 i = 0; i < length; i++ )
        {
            ASSERT( *counts );
            totalBufferSize += *counts * (size_t)elementSize;
            counts += countStride;
        }

        // #TODO: This should be done in a copy stream to perform the copies in the background
        Log::Line( "Marker Set to %d", 84)
CudaErrCheck( cudaStreamWaitEvent( uploadStream, self->deviceEvents[index] ) );
        Log::Line( "Marker Set to %d", 84)
CudaErrCheck( cudaMemcpyAsync( self->deviceBuffer[index], hostBuffer, totalBufferSize, cudaMemcpyHostToDevice, uploadStream ) );

        if( !self->diskBuffer )
            Log::Line( "Marker Set to %d", 85)
CudaErrCheck( cudaEventRecord( self->pinnedEvent[index], uploadStream ) );
    }

    // Signal work stream that the device buffer is ready to be used
    Log::Line( "Marker Set to %d", 86)
CudaErrCheck( cudaEventRecord( self->readyEvents[index], uploadStream ) );
}

void GpuUploadBuffer::UploadArrayForIndex( const uint32 index, const void* hostBuffer, uint32 length, 
                                           uint32 elementSize, uint32 srcStride, uint32 countStride, const uint32* counts )
{
    ASSERT( hostBuffer );

    auto stream = self->queue->GetStream();

    // Ensure the device buffer is ready for use
    Log::Line( "Marker Set to %d", 87)
CudaErrCheck( cudaStreamWaitEvent( stream, self->events[index] ) );

    // Perform uploads
    //size_t deviceCopySize = 0;
    const byte* src = (byte*)hostBuffer;
          byte* dst = (byte*)self->deviceBuffer[index];

    for( uint32 i = 0; i < length; i++ )
    {
        const size_t size = *counts * (size_t)elementSize;
        //memcpy( dst, src, size );
        Log::Line( "Marker Set to %d", 88)
CudaErrCheck( cudaMemcpyAsync( dst, src, size, cudaMemcpyHostToDevice, stream ) );

        //deviceCopySize += size;

        dst    += size;
        src    += srcStride;
        counts += countStride;
    }

    // Copy to device buffer
    //Log::Line( "Marker Set to %d", 89)
CudaErrCheck( cudaMemcpyAsync( self->deviceBuffer[index], cpy.dstBuffer, deviceCopySize, cudaMemcpyHostToDevice, _stream ) );

    // Signal work stream that the device buffer is ready to be used
    Log::Line( "Marker Set to %d", 90)
CudaErrCheck( cudaEventRecord( self->readyEvents[index], stream ) );
}

void GpuUploadBuffer::Upload( const void* hostBuffer, const size_t size )
{
    Upload( hostBuffer, size, nullptr );
}

void GpuUploadBuffer::UploadArray( const void* hostBuffer, uint32 length,
                                   uint32 elementSize, uint32 srcStride, uint32 countStride, const uint32* counts )
{
    UploadArray( hostBuffer, length, elementSize, srcStride, countStride, counts, nullptr );
}

void* GpuUploadBuffer::GetUploadedDeviceBuffer( cudaStream_t workStream )
{
    ASSERT( workStream );

    if( self->outgoingSequence < 1 )
    {
        ASSERT( 0 );
        return nullptr;
    }

    const uint32 index = self->completedSequence % self->bufferCount;
    self->completedSequence++;

    Log::Line( "Marker Set to %d", 91)
CudaErrCheck( cudaStreamWaitEvent( workStream, self->readyEvents[index] ) );

    return self->deviceBuffer[index];
}

void GpuUploadBuffer::ReleaseDeviceBuffer( cudaStream_t workStream )
{
    ASSERT( self->outgoingSequence > self->lockSequence );
    ASSERT( self->outgoingSequence - self->lockSequence <= 2 );
    ASSERT( self->completedSequence > 0 );

    const uint32 index = self->lockSequence % self->bufferCount;
    self->lockSequence++;

    Log::Line( "Marker Set to %d", 92)
CudaErrCheck( cudaEventRecord( self->deviceEvents[index], workStream ) );
}

void GpuUploadBuffer::WaitForPreloadsToComplete()
{
    if( self->outgoingSequence > 0 )
    {
        self->copyFence.Wait( self->outgoingSequence );
    }
}

uint32 GpuUploadBuffer::SynchronizeOutgoingSequence()
{
    PanicIf( self->outgoingSequence < self->lockSequence || self->outgoingSequence - self->lockSequence >= 2,
            "Invalid outgoing synchro sequence state." );

    const uint32 index = self->outgoingSequence % self->bufferCount;
    self->outgoingSequence++;

    return index;
}

void GpuUploadBuffer::Reset()
{
    self->lockSequence      = 0;
    self->outgoingSequence  = 0;
    self->completedSequence = 0;
    self->copySequence      = 0;
    self->fence.Reset( 0 );
    self->copyFence.Reset( 0 );
}

GpuQueue* GpuUploadBuffer::GetQueue() const
{
    return self->queue;
}

void GpuUploadBuffer::AssignDiskBuffer( DiskBufferBase* diskBuffer )
{
    ASSERT( self->pinnedBuffer[0] );

    void* nullBuffers[2] = { nullptr, nullptr };
    if( self->diskBuffer )
        self->diskBuffer->AssignReadBuffers( nullBuffers );

    self->diskBuffer = diskBuffer;
    if( self->diskBuffer )
        self->diskBuffer->AssignReadBuffers( self->pinnedBuffer );
}

DiskBufferBase* GpuUploadBuffer::GetDiskBuffer() const
{
    return self->diskBuffer;
}

void GpuUploadBuffer::CallHostFunctionOnStream( cudaStream_t stream, std::function<void()> func )
{
    auto* fnCpy = new std::function<void()>( std::move( func ) );
    Log::Line( "Marker Set to %d", 93)
CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ) {

        auto& fn = *reinterpret_cast<std::function<void()>*>( userData );
        fn();
        delete& fn;

    }, fnCpy ) );
}
