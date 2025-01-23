#include "GpuQueue.h"
#include "util/IAllocator.h"
#include "plotting/DiskBucketBuffer.h"
#include "plotting/DiskBuffer.h"

///
/// Shared GpuStream Inteface
///
GpuQueue::GpuQueue( Kind kind ) : _kind( kind )
    , _bufferReadySignal( BBCU_BUCKET_COUNT )
{
    Log::Line( "Marker Set to %d", 62)
CudaErrCheck( cudaStreamCreateWithFlags( &_stream        , cudaStreamNonBlocking ) );
    Log::Line( "Marker Set to %d", 63)
CudaErrCheck( cudaStreamCreateWithFlags( &_preloadStream , cudaStreamNonBlocking ) );
    Log::Line( "Marker Set to %d", 64)
CudaErrCheck( cudaStreamCreateWithFlags( &_callbackStream, cudaStreamNonBlocking ) );

    _queueThread.Run( QueueThreadEntryPoint, this );
}

GpuQueue::~GpuQueue()
{
    _exitQueueThread.store( true, std::memory_order_release );
    _bufferReadySignal.Release();
    _waitForExitSignal.Wait();
    

    if( _stream         ) cudaStreamDestroy( _stream );
    if( _preloadStream  ) cudaStreamDestroy( _preloadStream );
    if( _callbackStream ) cudaStreamDestroy( _callbackStream );
    
    _stream         = nullptr;
    _preloadStream  = nullptr;
    _callbackStream = nullptr;
}

GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( const GpuStreamDescriptor& desc, bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue." );
    GpuDownloadBuffer r = { CreateGpuBuffer( desc, dryRun ) };

    if( !dryRun )
        r.Reset();
    
    return r;
}

GpuDownloadBuffer GpuQueue::CreateDirectDownloadBuffer( const size_t size, IAllocator& devAllocator, const size_t alignment, const bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );

    ASSERT( 0 );    // #TODO: Deprecated function. Replace with the new one.
    GpuStreamDescriptor desc{};
    desc.entrySize       = 1;
    desc.entriesPerSlice = 1;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = alignment;
    desc.bufferCount     = 2;
    desc.deviceAllocator = &devAllocator;
    desc.pinnedAllocator = nullptr;

    return CreateDownloadBuffer( desc, dryRun );
}

GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );
    GpuDownloadBuffer r = { CreateGpuBuffer( size, devAllocator, pinnedAllocator, alignment, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}

GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( const size_t size, const uint32 bufferCount, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );

    ASSERT( 0 );    // #TODO: Deprecated function. Replace with the new one.
    GpuStreamDescriptor desc{};
    desc.entrySize       = 1;
    desc.entriesPerSlice = 1;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = alignment;
    desc.bufferCount     = bufferCount;
    desc.deviceAllocator = &devAllocator;
    desc.pinnedAllocator = &pinnedAllocator;

    GpuDownloadBuffer r = { CreateGpuBuffer( desc, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}

GpuUploadBuffer GpuQueue::CreateUploadBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    Panic( "Deprecated" );
    FatalIf( _kind != Uploader, "Attempted to create GpuUploadBuffer on an DownloadQueue" );

    GpuUploadBuffer r = { CreateGpuBuffer( size, devAllocator, pinnedAllocator, alignment, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}

GpuUploadBuffer GpuQueue::CreateUploadBuffer( const GpuStreamDescriptor& desc, bool dryRun )
{
    FatalIf( _kind != Uploader, "Attempted to create GpuUploadBuffer on an DownloadQueue." );

    GpuUploadBuffer r = { CreateGpuBuffer( desc, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}



struct IGpuBuffer* GpuQueue::CreateGpuBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    Panic( "Deprecated" );
    // ASSERT( 0 );    // #TODO: Deprecated function. Replace with the new one.
    GpuStreamDescriptor desc{};
    desc.entrySize       = 1;
    desc.entriesPerSlice = size;
    desc.sliceCount      = BBCU_BUCKET_COUNT;
    desc.sliceAlignment  = alignment;
    desc.bufferCount     = 2;
    desc.deviceAllocator = &devAllocator;
    desc.pinnedAllocator = &pinnedAllocator;

    return CreateGpuBuffer( desc, dryRun );
}

struct IGpuBuffer* GpuQueue::CreateGpuBuffer( const GpuStreamDescriptor& desc, bool dryRun )
{
    PanicIf( desc.bufferCount > BBCU_GPU_BUFFER_MAX_COUNT || !desc.bufferCount, "Invalid GPUBuffer buffer count." );
    PanicIf( !desc.deviceAllocator, "Null device allocator." );
    PanicIf( !desc.entrySize, "Invalid entry size." );
    PanicIf( !desc.entriesPerSlice, "Invalid entries per slice." );
    PanicIf( !desc.sliceCount || desc.sliceCount > BBCU_BUCKET_COUNT, "Invalid slice count." );
    PanicIf( !desc.sliceAlignment, "Invalid slice alignment." );
    PanicIf( desc.diskQueue && (!desc.diskFileName || !*desc.diskFileName), "Invalid disk offload config." );
    PanicIf( desc.diskQueue && !desc.pinnedAllocator, "A pinned allocator must be set in disk offload mode." );

    const size_t allocSize = CalculateBufferSizeFromDescriptor( desc );

    void* devBuffers   [BBCU_GPU_BUFFER_MAX_COUNT] = {};
    void* pinnedBuffers[BBCU_GPU_BUFFER_MAX_COUNT] = {};

    for( int32 i = 0; i < desc.bufferCount; i++ )
    {
        devBuffers[i] = desc.deviceAllocator->Alloc( allocSize, desc.sliceAlignment );

        if( desc.pinnedAllocator )
            pinnedBuffers[i] = desc.pinnedAllocator->Alloc( allocSize, desc.sliceAlignment );
    }

    struct IGpuBuffer* buf = nullptr;

    if( !dryRun )
    {
        buf = new IGpuBuffer{};

        for( int32 i = 0; i < desc.bufferCount; i++ )
        {
            Log::Line( "Marker Set to %d", 65)
CudaErrCheck( cudaEventCreateWithFlags( &buf->events[i]         , cudaEventDisableTiming ) );
            Log::Line( "Marker Set to %d", 66)
CudaErrCheck( cudaEventCreateWithFlags( &buf->completedEvents[i], cudaEventDisableTiming ) );
            Log::Line( "Marker Set to %d", 67)
CudaErrCheck( cudaEventCreateWithFlags( &buf->readyEvents[i]    , cudaEventDisableTiming ) );
            Log::Line( "Marker Set to %d", 68)
CudaErrCheck( cudaEventCreateWithFlags( &buf->pinnedEvent[i]  , cudaEventDisableTiming ) );

            Log::Line( "Marker Set to %d", 69)
CudaErrCheck( cudaEventCreateWithFlags( &buf->callbackLockEvent     , cudaEventDisableTiming ) );
            Log::Line( "Marker Set to %d", 70)
CudaErrCheck( cudaEventCreateWithFlags( &buf->callbackCompletedEvent, cudaEventDisableTiming ) );
            
            buf->deviceBuffer[i] = devBuffers[i];
            buf->pinnedBuffer[i] = pinnedBuffers[i];
        }

            buf->size        = allocSize;
            buf->bufferCount = desc.bufferCount;
            buf->queue       = this;
    }

    // Disk offload mode?
    if( desc.diskQueue )
    {
        const size_t sliceSize = CalculateSliceSizeFromDescriptor( desc );

        if( !dryRun )
        {
            if( desc.bucketedDiskBuffer )
            {
                buf->diskBuffer = DiskBucketBuffer::Create( 
                    *desc.diskQueue, desc.diskFileName,
                    desc.sliceCount, sliceSize,
                    FileMode::Create, FileAccess::ReadWrite, 
                    desc.directIO ? FileFlags::NoBuffering | FileFlags::LargeFile : FileFlags::None );
            }
            else
            {
                buf->diskBuffer = DiskBuffer::Create(
                    *desc.diskQueue, desc.diskFileName,
                    desc.sliceCount, allocSize,
                    FileMode::Create, FileAccess::ReadWrite, 
                    desc.directIO ? FileFlags::NoBuffering | FileFlags::LargeFile : FileFlags::None );
            }

            PanicIf( !buf->diskBuffer, "Failed to create DiskBuffer for GpuBuffer." );

            void* readBuffers [2] = { nullptr, nullptr };
            void* writeBuffers[2] = { pinnedBuffers[0], pinnedBuffers[1] };

            buf->diskBuffer->AssignBuffers( readBuffers, writeBuffers );
        }
        else
        {
            size_t diskAllocSize = 0;
            if( desc.bucketedDiskBuffer )
            {
                diskAllocSize = DiskBucketBuffer::GetReserveAllocSize( *desc.diskQueue, desc.sliceCount, sliceSize );
            }
            else
            {
                diskAllocSize = DiskBuffer::GetReserveAllocSize( *desc.diskQueue, allocSize );
            }

            ASSERT( diskAllocSize == allocSize * 4 );
        }
    }

    return buf;
}

void GpuQueue::DispatchHostFunc( GpuCallbackDispath func, cudaStream_t stream, cudaEvent_t lockEvent, cudaEvent_t completedEvent )
{
    // #MAYBE: Perhaps support having multiple callback streams, and multiple copy streams.

    // Signal from the work stream into the callback stream that we are ready for callback
    Log::Line( "Marker Set to %d", 71)
CudaErrCheck( cudaEventRecord( lockEvent, stream ) );

    // Wait on the callback stream until it's ready to dsitpatch
    Log::Line( "Marker Set to %d", 72)
CudaErrCheck( cudaStreamWaitEvent( _callbackStream, lockEvent ) );

    // #MAYBE: Use a bump allocator perhaps later to avoid locking here by new/delete if needed for performance.
    auto* fnCpy = new std::function<void()>( std::move( func ) );
    Log::Line( "Marker Set to %d", 73)
CudaErrCheck( cudaLaunchHostFunc( _callbackStream, []( void* userData ){

        auto& fn = *reinterpret_cast<std::function<void()>*>( userData );
        fn();
        delete &fn;

    }, fnCpy ) );

    // Signal from the callback stream that the callback finished
    Log::Line( "Marker Set to %d", 74)
CudaErrCheck( cudaEventRecord( completedEvent, _callbackStream ) );

    // Wait on work stream for the callback to complete
    Log::Line( "Marker Set to %d", 75)
CudaErrCheck( cudaStreamWaitEvent( stream, completedEvent ) );
}

size_t GpuQueue::CalculateSliceSizeFromDescriptor( const GpuStreamDescriptor& desc )
{
    const size_t alignment = desc.diskQueue ? desc.diskQueue->BlockSize() : desc.sliceAlignment; 
    return RoundUpToNextBoundaryT( desc.entrySize * desc.entriesPerSlice, alignment );
}

size_t GpuQueue::CalculateBufferSizeFromDescriptor( const GpuStreamDescriptor& desc )
{
    return CalculateSliceSizeFromDescriptor( desc ) * desc.sliceCount;
}

void GpuQueue::CopyPendingDownloadStream( void* userData )
{
    auto* buf = reinterpret_cast<IGpuBuffer*>( userData );

    GpuQueue* queue = buf->queue;

    //const uint32 index = buf->completedSequence % buf->bufferCount;
    buf->completedSequence++;

    //queue->GetCommand( CommandType::Download2D ) = buf->commands[index];
    queue->SubmitCommands();
}

void GpuQueue::SubmitCommands()
{
    const uint64 ticket = _commitTicketOut++;

    // Wait for our ticket to come up
    while( _commitTicketIn.load( std::memory_order_relaxed ) != ticket );

    _queue.Commit();
    _bufferReadySignal.Release();
    //_bufferReadySignal.Signal();

    // Use our ticket
    _commitTicketIn.store( ticket+1, std::memory_order_release );
}

GpuQueue::Command& GpuQueue::GetCommand( CommandType type )
{
    const uint64 ticket = _cmdTicketOut++;

    // Wait for our ticket to come up
    while( _cmdTicketIn.load( std::memory_order_relaxed ) != ticket );
    
    Command* cmd;
    while( !_queue.Write( cmd ) )
    {
        Log::Line( "[GpuQueue] Queue is depleted. Waiting for copies to complete." );
        auto waitTimer = TimerBegin();

        // Block and wait until we have commands free in the buffer
        _bufferCopiedSignal.Wait();
        
        Log::Line( "[GpuQueue] Waited %.6lf seconds for availability.", TimerEnd( waitTimer ) );
    }

    // Use our ticket
    _cmdTicketIn.store( ticket+1, std::memory_order_release );

    ZeroMem( cmd );
    cmd->type = type;

    return *cmd;
}

///
/// Command thread
///
void GpuQueue::QueueThreadEntryPoint( GpuQueue* self )
{
    ASSERT( self );
    self->QueueThreadMain();
    self->_waitForExitSignal.Signal();
}

void GpuQueue::QueueThreadMain()
{
    const int32 CMD_BUF_SIZE = 256;
    Command buffers[CMD_BUF_SIZE];

    for( ;; )
    {
        _bufferReadySignal.Wait();

        if( ShouldExitQueueThread() )
            return;

        // 1 command per semaphore release
        int32 bufCount;
        while( ( ( bufCount = _queue.Dequeue( buffers, CMD_BUF_SIZE ) ) ) )
        // if( ( ( bufCount = _queue.Dequeue( buffers, CMD_BUF_SIZE ) ) ) )
        {
            ASSERT( bufCount <= CMD_BUF_SIZE );
            _bufferCopiedSignal.Signal();

            for( int i = 0; i < bufCount; i++ )
                ExecuteCommand( buffers[i] );
        }
    }
}

void GpuQueue::ExecuteCommand( const Command& cmd )
{

    // const uint32 index = cmd.sequenceId % BBCU_GPU_BUFFER_MAX_COUNT;

    if( cmd.type == CommandType::Copy )
    {
        auto& cpy = *cmd.copy;

        const bool   isSequentialCopy = cpy.dstStride == cpy.srcStride;
        const size_t totalSize        = cpy.height * cpy.width;

              byte* dst = (byte*)cpy.dstBuffer;
        const byte* src = (byte*)cpy.srcBuffer;

        if( isSequentialCopy )
            memcpy( cpy.dstBuffer, cpy.srcBuffer, totalSize );
        else
        {
            const byte* src = (byte*)cpy.srcBuffer;
                  byte* dst = (byte*)cpy.dstBuffer;

            for( size_t i = 0; i < cpy.height; i++ )
            {
                memcpy( dst, src, cpy.width );

                dst += cpy.dstStride;
                src += cpy.srcStride;
            }
        }

        cpy.self->fence.Signal( cpy.sequence+1 );
        cpy.self->copyFence.Signal( cpy.sequence+1 );

        if( cpy.callback )
            cpy.callback( cpy.dstBuffer, totalSize, cpy.userData );
    }
    else if( cmd.type == CommandType::CopyArray )
    {
        
    }
    else if( cmd.type == CommandType::Callback )
    {
        cmd.callback.callback( cmd.callback.dstbuffer, cmd.callback.copySize, cmd.callback.userData );
    }
    // else if( cmd.type == CommandType::Sync )
    // {
    //     _syncFence.Signal();
    //     return;
    // }
    else
    {
        ASSERT( 0 );
    }

    // Signal that the pinned buffer is free
    //cpy.finishedSignal->Signal( cpy.sequenceId + 1 );
}

inline bool GpuQueue::ShouldExitQueueThread()
{
    return _exitQueueThread.load( std::memory_order_acquire );
}

