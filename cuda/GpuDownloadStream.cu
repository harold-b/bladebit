#include "GpuStreams.h"
#include "GpuQueue.h"
#include "plotting/DiskBucketBuffer.h"
#include "plotting/DiskBuffer.h"


///
/// DownloadBuffer
///
void* GpuDownloadBuffer::GetDeviceBuffer()
{
    const uint32 index = self->outgoingSequence % self->bufferCount;

    Log::Line( "Marker Set to %d", 56)
CudaErrCheck( cudaEventSynchronize( self->events[index] ) );

    return self->deviceBuffer[index];
}

void* GpuDownloadBuffer::LockDeviceBuffer( cudaStream_t stream )
{
    ASSERT( self->lockSequence >= self->outgoingSequence );
    ASSERT( self->lockSequence - self->outgoingSequence < self->bufferCount );

    const uint32 index = self->lockSequence % self->bufferCount;
    self->lockSequence++;

    // Wait for the device buffer to be free to be used by kernels
    Log::Line( "Marker Set to %d", 57)
CudaErrCheck( cudaStreamWaitEvent( stream, self->events[index] ) );
    return self->deviceBuffer[index];
}

void GpuDownloadBuffer::Download( void* hostBuffer, const size_t size )
{
    Download2D( hostBuffer, size, 1, size, size );
}

void GpuDownloadBuffer::Download( void* hostBuffer, const size_t size, cudaStream_t workStream, bool directOverride )
{
    Download2D( hostBuffer, size, 1, size, size, workStream, directOverride );
}

void GpuDownloadBuffer::DownloadAndCopy( void* hostBuffer, void* finalBuffer, const size_t size, cudaStream_t workStream  )
{
    Panic( "Unavailable" );
    // ASSERT( self->outgoingSequence < BBCU_BUCKET_COUNT );
    // ASSERT( hostBuffer );
    // ASSERT( workStream );
    // ASSERT( self->lockSequence > 0 );
    // ASSERT( self->outgoingSequence < self->lockSequence );
    // ASSERT( self->lockSequence - self->outgoingSequence <= self->bufferCount );

    // auto& cpy = self->copies[self->outgoingSequence];
    // cpy.self            = self;
    // cpy.sequence        = self->outgoingSequence;
    // cpy.copy.hostBuffer = finalBuffer;
    // cpy.copy.srcBuffer  = hostBuffer;
    // cpy.copy.size       = size;


    // const uint32 index = self->outgoingSequence % self->bufferCount;
    // self->outgoingSequence++;

    //       void* pinnedBuffer = self->pinnedBuffer[index];
    // const void* devBuffer    = self->deviceBuffer[index];

    // // Signal from the work stream when it has finished doing kernel work with the device buffer


    // // Ensure the work stream has completed writing data to the device buffer
    // cudaStream_t stream = self->queue->_stream;


    // // Copy

    // // Signal that the device buffer is free to be re-used
    // Log::Line( "Marker Set to %d", 58)
CudaErrCheck( cudaEventRecord( self->events[index], stream ) );

    // // Launch copy command
    // Log::Line( "Marker Set to %d", 59)
CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ){

    //     const CopyInfo& c = *reinterpret_cast<CopyInfo*>( userData );
    //     IGpuBuffer* self = c.self;

    //     auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Copy );
    //     cmd.copy.info = &c;

    //     self->queue->SubmitCommands();
        
    //     // Signal the download completed
    //     self->fence.Signal( ++self->completedSequence );
    // }, &cpy ) );
}

void GpuDownloadBuffer::DownloadWithCallback( void* hostBuffer, const size_t size, GpuDownloadCallback callback, void* userData, cudaStream_t workStream, bool directOverride )
{
    Download2DWithCallback( hostBuffer, size, 1, size, size, callback, userData, workStream, directOverride );
}

void GpuDownloadBuffer::Download2D( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, cudaStream_t workStream, bool directOverride )
{
    Download2DWithCallback( hostBuffer, width, height, dstStride, srcStride, nullptr, nullptr, workStream, directOverride );
}

void GpuDownloadBuffer::Download2DWithCallback( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, 
                                                GpuDownloadCallback callback, void* userData, cudaStream_t workStream, bool directOverride )
{
    PerformDownload2D( hostBuffer, width, height, dstStride, srcStride,
                       callback, userData, 
                       workStream, directOverride );
}

void GpuDownloadBuffer::PerformDownload2D( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                                           GpuDownloadCallback postCallback, void* postUserData, 
                                           cudaStream_t workStream, bool directOverride )
{
    PanicIf( !(hostBuffer || self->pinnedBuffer[0] ), "" );
    ASSERT( workStream );
    ASSERT( self->lockSequence > 0 );
    ASSERT( self->outgoingSequence < self->lockSequence );
    ASSERT( self->lockSequence - self->outgoingSequence <= self->bufferCount );

    const uint32 index = self->outgoingSequence++ % self->bufferCount;

          void*  pinnedBuffer     = self->pinnedBuffer[index];
          void*  finalHostBuffer  = hostBuffer;
    const void*  devBuffer        = self->deviceBuffer[index];

    const bool   isDirect         = (directOverride || self->pinnedBuffer[0] == nullptr) && !self->diskBuffer;   ASSERT( isDirect || self->pinnedBuffer[0] );
    const bool   isSequentialCopy = dstStride == srcStride;
    const size_t totalSize        = height * width;


    // Signal from the work stream when it has finished doing kernel work with the device buffer
    Log::Line( "Marker Set to %d", 60)
CudaErrCheck( cudaEventRecord( self->workEvent[index], workStream ) );

    // From the download stream, wait for the work stream to finish
    cudaStream_t downloadStream = self->queue->_stream;
    Log::Line( "Marker Set to %d", 61)
CudaErrCheck( cudaStreamWaitEvent( downloadStream, self->workEvent[index] ) );


    if( self->diskBuffer )
    {
        // Wait until the next disk buffer is ready for use.
        // This also signals that the pinned buffer is ready for re-use
        CallHostFunctionOnStream( downloadStream, [this](){
            self->diskBuffer->GetNextWriteBuffer();
        });

        pinnedBuffer = self->diskBuffer->PeekWriteBufferForBucket( self->outgoingSequence-1 ); 
    }

    if( !isDirect )
    {
        // Ensure that the pinned buffer is ready for use
        // (we signal pinned buffers are ready when using disks without events)
        if( !self->diskBuffer )
            CudaErrCheck( cudaStreamWaitEvent( downloadStream, self->pinnedEvent[index] ) );

        // Set host buffer as the pinned buffer
        hostBuffer = pinnedBuffer;
    }


    // Copy from device to host buffer
    // #NOTE: Since the pinned buffer is simply the same size (a full bucket) as the device buffer
    //        we also always copy as 1D if we're copying to our pinned buffer.
    ASSERT( hostBuffer );
    if( isSequentialCopy || hostBuffer == pinnedBuffer )
        CudaErrCheck( cudaMemcpyAsync( hostBuffer, devBuffer, totalSize, cudaMemcpyDeviceToHost, downloadStream ) );
    else
        CudaErrCheck( cudaMemcpy2DAsync( hostBuffer, dstStride, devBuffer, srcStride, width, height, cudaMemcpyDeviceToHost, downloadStream ) );

    // Dispatch a host callback if one was set
    if( postCallback )
    {
        CallHostFunctionOnStream( downloadStream, [=](){
            (*postCallback)( finalHostBuffer, totalSize, postUserData );
        });
    }


    // Signal that the device buffer is free to be re-used
    CudaErrCheck( cudaEventRecord( self->deviceEvents[index], downloadStream ) );

    if( self->diskBuffer )
    {
        // If it's a disk-based copy, then write the pinned buffer to disk
        CallHostFunctionOnStream( downloadStream, [=]() {

            auto* diskBucketBuffer = dynamic_cast<DiskBucketBuffer*>( self->diskBuffer );
            if( diskBucketBuffer != nullptr )
                diskBucketBuffer->Submit( srcStride );
            else
                static_cast<DiskBuffer*>( self->diskBuffer )->Submit( totalSize );
        });

        // #NOTE: We don't need to signal that the pinned buffer is ready for re-use here as
        //        we do that implicitly with DiskBuffer::GetNextWriteBuffer (see above).
    }
    else if( !isDirect )
    {
        // #TODO: Do this in a different host copy stream, and signal from there.
        // #MAYBE: Perhaps use multiple host threads/streams to do host-to-host copies.
        //        for now do it on the same download stream, but we will be blocking the download stream,
        //        unless other download streams are used by other buffers.


        ASSERT( hostBuffer == pinnedBuffer );
        if( isSequentialCopy )
            CudaErrCheck( cudaMemcpyAsync( finalHostBuffer, hostBuffer, totalSize, cudaMemcpyHostToHost, downloadStream ) );
        else
            CudaErrCheck( cudaMemcpy2DAsync( finalHostBuffer, dstStride, hostBuffer, srcStride, width, height, cudaMemcpyHostToHost, downloadStream ) );

        // Signal the pinned buffer is free to be re-used
        CudaErrCheck( cudaEventRecord( self->pinnedEvent[index], downloadStream ) );
    }
}

void GpuDownloadBuffer::CallHostFunctionOnStream( cudaStream_t stream, std::function<void()> func )
{
    auto* fnCpy = new std::function<void()>( std::move( func ) );
    CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ) {

        auto& fn = *reinterpret_cast<std::function<void()>*>( userData );
        fn();
        delete& fn;

    }, fnCpy ) );
}

void GpuDownloadBuffer::HostCallback( std::function<void()> func )
{
    CallHostFunctionOnStream( self->queue->GetStream(), func );
}

void GpuDownloadBuffer::GetDownload2DCommand( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                                              uint32& outIndex, void*& outPinnedBuffer, const void*& outDevBuffer, GpuDownloadCallback callback, void* userData )
{
    ASSERT( width      );
    ASSERT( height     );
    ASSERT( hostBuffer );

    const uint32 index = self->outgoingSequence % self->bufferCount;

    // We need to block until the pinned buffer is available.
    if( self->outgoingSequence > self->bufferCount-1 )
        self->fence.Wait( self->outgoingSequence - self->bufferCount + 1 );

          void* pinnedBuffer = self->pinnedBuffer[index];
    const void* devBuffer    = self->deviceBuffer[index];

    //auto& cmd = self->commands[index];
    //cmd.type             = GpuQueue::CommandType::Copy2D;
    //cmd.sequenceId       = self->outgoingSequence++;
    //cmd.finishedSignal   = &self->fence;
    //cmd.dstBuffer        = hostBuffer;
    //cmd.srcBuffer        = pinnedBuffer;
    //cmd.copy2d.width     = width;
    //cmd.copy2d.height    = height;
    //cmd.copy2d.dstStride = dstStride;
    //cmd.copy2d.srcStride = srcStride;
    //cmd.copy2d.callback  = callback;
    //cmd.copy2d.userData  = userData;

    outIndex        = index;
    outPinnedBuffer = pinnedBuffer;
    outDevBuffer    = devBuffer;
}


void GpuDownloadBuffer::DownloadAndPackArray( void* hostBuffer, const uint32 length, size_t srcStride, const uint32* counts, const uint32 elementSize )
{
    ASSERT( length      );
    ASSERT( elementSize );
    ASSERT( counts      );

    uint32 totalElements = 0;
    for( uint32 i = 0; i < length; i++ )
        totalElements += counts[i];

    const size_t totalSize = (size_t)totalElements * elementSize;

    uint32      index;
    void*       pinnedBuffer;
    const void* devBuffer;
    GetDownload2DCommand( hostBuffer, totalSize, 1, totalSize, totalSize, index, pinnedBuffer, devBuffer );


    srcStride *= elementSize;

          byte* dst = (byte*)pinnedBuffer;
    const byte* src = (byte*)devBuffer;

    cudaStream_t stream = self->queue->_stream;

    // Copy all buffers from device to pinned buffer
    for( uint32 i = 0; i < length; i++ )
    {
        const size_t copySize = counts[i] * (size_t)elementSize;

        // #TODO: Determine if there's a cuda (jagged) array copy
        CudaErrCheck( cudaMemcpyAsync( dst, src, copySize, cudaMemcpyDeviceToHost, stream ) );

        src += srcStride;
        dst += copySize;
    }

    // Signal that the device buffer is free
    CudaErrCheck( cudaEventRecord( self->events[index], stream ) );

    // Submit command to do the final copy from pinned to host
    CudaErrCheck( cudaLaunchHostFunc( stream, GpuQueue::CopyPendingDownloadStream, self ) );
}

void GpuDownloadBuffer::WaitForCompletion()
{
    if( self->outgoingSequence > 0 )
    {
        //const uint32 index = (self->outgoingSequence - 1) % self->bufferCount;

        //      cudaEvent_t event = self->completedEvents[index];
        //const cudaError_t r     = cudaEventQuery( event );

        //if( r == cudaSuccess )
        //    return;

        //if( r != cudaErrorNotReady )
        //    CudaErrCheck( r );

        //CudaErrCheck( cudaEventSynchronize( event ) );
        

        cudaStream_t downloadStream = self->queue->_stream;
        // this->self->fence.Reset( 0 );
        CallHostFunctionOnStream( downloadStream, [this](){
            this->self->fence.Signal( this->self->outgoingSequence );
        });
        self->fence.Wait( self->outgoingSequence );

    }
}

void GpuDownloadBuffer::WaitForCopyCompletion()
{
    if( self->outgoingSequence > 0 )
    {
        self->copyFence.Wait( self->outgoingSequence );
    }
}

void GpuDownloadBuffer::Reset()
{
    self->lockSequence      = 0;
    self->outgoingSequence  = 0;
    self->completedSequence = 0;
    self->copySequence      = 0;
    self->fence.Reset( 0 );
    self->copyFence.Reset( 0 );
}

GpuQueue* GpuDownloadBuffer::GetQueue() const
{
    return self->queue;
}

void GpuDownloadBuffer::AssignDiskBuffer( DiskBufferBase* diskBuffer )
{
    // ASSERT( self->pinnedBuffer[0] );

    void* nullBuffers[2] = { nullptr, nullptr };
    if( self->diskBuffer )
        self->diskBuffer->AssignWriteBuffers( nullBuffers );

    self->diskBuffer = diskBuffer;
    if( self->diskBuffer )
        self->diskBuffer->AssignWriteBuffers( self->pinnedBuffer );
}

DiskBufferBase* GpuDownloadBuffer::GetDiskBuffer() const
{
    return self->diskBuffer;
}
