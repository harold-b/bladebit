#include "DiskBufferQueue.h"
#include "Util.h"
#include "diskplot/DiskPlotConfig.h"
#include "SysHost.h"

#include "jobs/IOJob.h"
#include "util/Log.h"


#define NULL_BUFFER -1


//-----------------------------------------------------------
DiskBufferQueue::DiskBufferQueue( 
    const char* workDir, byte* workBuffer, 
    size_t workBufferSize, uint ioThreadCount,
    bool useDirectIO )
    : _workDir       ( workDir     )
    , _workHeap      ( workBufferSize, workBuffer )
    , _useDirectIO   ( useDirectIO )
    , _threadPool    ( ioThreadCount, ThreadPool::Mode::Fixed, true )
    , _dispatchThread()
{
    ASSERT( workDir );

    // Initialize Files
    size_t workDirLen = strlen( workDir );
    char*  pathBuffer = bbmalloc<char>( workDirLen + 64 );

    memcpy( pathBuffer, workDir, workDirLen + 1 );

    if( pathBuffer[workDirLen - 1] != '/' && pathBuffer[workDirLen - 1] != '\\' )
    {
        pathBuffer[workDirLen]   = '/';
        pathBuffer[++workDirLen] = '\0';
    }

    InitFileSet( FileId::Y0      , "y0"      , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::Y1      , "y1"      , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::META_A_0, "meta_a0" , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::META_A_1, "meta_a1" , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::META_B_0, "meta_b0" , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::META_B_1, "meta_b1" , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::X       , "x"       , BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
//     InitFileSet( FileId::T2_L    , "table_2l", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T2_R    , "table_2r", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T3_L    , "table_3l", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T3_R    , "table_3r", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T4_L    , "table_4l", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T4_R    , "table_4r", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T5_L    , "table_5l", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T5_R    , "table_5r", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T6_L    , "table_6l", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T6_R    , "table_6r", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T7_L    , "table_7l", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::T7_R    , "table_7r", 1,                  pathBuffer, workDirLen );
//     InitFileSet( FileId::F7      , "f7"      , 1,                  pathBuffer, workDirLen );
    free( pathBuffer );

    // Initialize I/O thread
    _dispatchThread.Run( CommandThreadMain, this );
}

//-----------------------------------------------------------
DiskBufferQueue::~DiskBufferQueue()
{
}

//-----------------------------------------------------------
void DiskBufferQueue::ResetHeap( const size_t heapSize, void* heapBuffer )
{
    _workHeap.ResetHeap( heapSize, heapBuffer );
}

//-----------------------------------------------------------
void DiskBufferQueue::InitFileSet( FileId fileId, const char* name, uint bucketCount, char* pathBuffer, size_t workDirLength )
{
    char* baseName = pathBuffer + workDirLength;

    FileFlags flags = FileFlags::LargeFile;
    if( _useDirectIO )
        flags |= FileFlags::NoBuffering;

    FileSet& fileSet = _files[(uint)fileId];
    
    fileSet.name         = name;
    fileSet.files.values = new FileStream[bucketCount];
    fileSet.files.length = bucketCount;

    for( uint i = 0; i < bucketCount; i++ )
    {
        FileStream& file = fileSet.files[i];

        constexpr FileMode fileMode =
        #if BB_DP_DBG_READ_EXISTING_F1
            FileMode::OpenOrCreate;
        #else
            FileMode::Create;
        #endif

        sprintf( baseName, "%s_%u.tmp", name, i );
        if( !file.Open( pathBuffer, fileMode, FileAccess::ReadWrite, flags ) )
            Fatal( "Failed to open temp work file @ %s with error: %d.", pathBuffer, file.GetError() );

        if( !_blockBuffer )
        {
            _blockSize = file.BlockSize();
            FatalIf( _blockSize < 2, "Invalid temporary file block size." );
            
            _blockBuffer = (byte*)SysHost::VirtualAlloc( _blockSize, false );
            FatalIf( !_blockBuffer, "Out of memory." );
        }
        else
        {
            if( file.BlockSize() != _blockSize )
                Fatal( "Temporary work files have differing block sizes." );
        }
    }
}

//-----------------------------------------------------------
byte* DiskBufferQueue::GetBuffer( size_t size )
{
    return _workHeap.Alloc( size, _blockSize );
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteBuckets( FileId id, const void* buckets, const uint* sizes )
{
    Command* cmd = GetCommandObject( Command::WriteBuckets );
    cmd->buckets.buffers = (byte*)buckets;
    cmd->buckets.sizes   = sizes;
    cmd->buckets.fileId  = id;
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteFile( FileId id, uint bucket, const void* buffer, size_t size )
{
    Command* cmd = GetCommandObject( Command::WriteFile );
    cmd->file.buffer = (byte*)buffer;
    cmd->file.size   = size;
    cmd->file.fileId = id;
    cmd->file.bucket = bucket;
}

//-----------------------------------------------------------
void DiskBufferQueue::ReadFile( FileId id, uint bucket, void* dstBuffer, size_t readSize )
{
    Command* cmd = GetCommandObject( Command::ReadFile );
    cmd->file.buffer = (byte*)dstBuffer;
    cmd->file.size   = readSize;
    cmd->file.fileId = id;
    cmd->file.bucket = bucket;
}

//-----------------------------------------------------------
void DiskBufferQueue::SeekFile( FileId id, uint bucket, int64 offset, SeekOrigin origin )
{
    Command* cmd = GetCommandObject( Command::SeekFile );
    cmd->seek.fileId = id;
    cmd->seek.bucket = bucket;
    cmd->seek.offset = offset;
    cmd->seek.origin = origin;
}

//-----------------------------------------------------------
void DiskBufferQueue::SeekBucket( FileId id, int64 offset, SeekOrigin origin )
{
    Command* cmd = GetCommandObject( Command::SeekBucket );
    cmd->seek.fileId = id;
    cmd->seek.offset = offset;
    cmd->seek.origin = origin;
}

//-----------------------------------------------------------
void DiskBufferQueue::ReleaseBuffer( void* buffer )
{
    ASSERT( buffer );

    Command* cmd = GetCommandObject( Command::ReleaseBuffer );
    cmd->releaseBuffer.buffer = (byte*)buffer;
}

//-----------------------------------------------------------
void DiskBufferQueue::AddFence( AutoResetSignal& signal )
{
    Command* cmd = GetCommandObject( Command::MemoryFence );
    cmd->fence.signal = &signal;
}

//-----------------------------------------------------------
void DiskBufferQueue::CompletePendingReleases()
{
    _workHeap.CompletePendingReleases();
}

//-----------------------------------------------------------
inline DiskBufferQueue::Command* DiskBufferQueue::GetCommandObject( Command::CommandType type )
{
    Command* cmd;
    while( !_commands.Write( cmd ) )
    {
        Log::Line( "[DiskBufferQueue] Command buffer full. Waiting for commands." );
        auto waitTimer = TimerBegin();

        // Block and wait until we have commands free in the buffer
        _cmdConsumedSignal.Wait();
        
        Log::Line( "[DiskBufferQueue] Waited %.6lf seconds for a Command to be available.", TimerEnd( waitTimer ) );
    }

    ZeroMem( cmd );
    cmd->type = type;

    #if DBG_LOG_ENABLE
        Log::Debug( "[DiskBufferQueue] > Snd: %s (%d)", DbgGetCommandName( type ), type );
    #endif

    return cmd;
}

//-----------------------------------------------------------
void DiskBufferQueue::CommitCommands()
{
    //Log::Debug( "Committing %d commands.", _commands._pendingCount );
    _commands.Commit();
    _cmdReadySignal.Signal();
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandThreadMain( DiskBufferQueue* self )
{
    self->CommandMain();
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandMain()
{
    const int CMD_BUF_SIZE = 64;
    Command commands[CMD_BUF_SIZE];

    for( ;; )
    {
        _cmdReadySignal.Wait();

        int cmdCount;
        while( cmdCount = _commands.Dequeue( commands, CMD_BUF_SIZE ) )
        {
            _cmdConsumedSignal.Signal();

            for( int i = 0; i < cmdCount; i++ )
                ExecuteCommand( commands[i] );
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::ExecuteCommand( Command& cmd )
{
    //#if DBG_LOG_ENABLE
    //    Log::Debug( "[DiskBufferQueue] ^ Cmd Execute: %s (%d)", DbgGetCommandName( cmd.type ), cmd.type );
    //#endif

    switch( cmd.type )
    {
        case Command::WriteBuckets:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WriteBuckets: (%u) addr:0x%p", cmd.buckets.fileId, cmd.buckets.buffers );
            #endif
            CmdWriteBuckets( cmd );
        break;

        case Command::WriteFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WriteFile: (%u) bucket:%u sz:%llu addr:0x%p", cmd.file.fileId, cmd.file.bucket, cmd.file.size, cmd.file.buffer );
            #endif
            CndWriteFile( cmd );
        break;

        case Command::ReadFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd ReadFile: (%u) bucket:%u sz:%llu addr:0x%p", cmd.file.fileId, cmd.file.bucket, cmd.file.size, cmd.file.buffer );
            #endif
            CmdReadFile( cmd );
        break;

        case Command::SeekFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd SeekFile: (%u) bucket:%u offset:%lld origin:%ld", cmd.seek.fileId, cmd.seek.bucket, cmd.seek.offset, (int)cmd.seek.origin );
            #endif
                if( !_files[(uint)cmd.seek.fileId].files[cmd.seek.bucket].Seek( cmd.seek.offset, cmd.seek.origin ) )
                {
                    int err = _files[(uint)cmd.seek.fileId].files[cmd.seek.bucket].GetError();
                    Fatal( "[DiskBufferQueue] Failed to seek file %s.%u with error %d (0x%x)", 
                           _files[(uint)cmd.seek.fileId].name, cmd.seek.bucket, err, err );
                }
        break;

        case Command::SeekBucket:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd SeekBucket: (%u) offset:%lld origin:%ld", cmd.seek.fileId, cmd.seek.offset, (int)cmd.seek.origin );
            #endif

            CmdSeekBucket( cmd );
        break;

        case Command::ReleaseBuffer:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd ReleaseBuffer: 0x%p", cmd.releaseBuffer.buffer );
            #endif
            _workHeap.Release( cmd.releaseBuffer.buffer );
        break;

        case Command::MemoryFence:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd MemoryFence" );
            #endif
            ASSERT( cmd.fence.signal );
            cmd.fence.signal->Signal();
        break;


        default:
            ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdWriteBuckets( const Command& cmd )
{
    const FileId fileId      = cmd.buckets.fileId;
    const uint*  sizes       = cmd.buckets.sizes;
    const byte*  buffers     = cmd.buckets.buffers;

    FileSet&     fileBuckets = _files[(int)fileId];

    const uint   bucketCount = (uint)fileBuckets.files.length;

    Log::Debug( "  >>> Write 0x%p", buffers );

    // Single-threaded for now... We don't have file handles for all the threads yet!
    const size_t blockSize = _blockSize;
    const byte*  buffer    = buffers;
    
    for( uint i = 0; i < bucketCount; i++ )
    {
        const size_t bufferSize = sizes[i];
        
        // Only write up-to the block-aligned boundary.
        // The caller is in charge of writing any remainders manually
        const size_t writeSize = _useDirectIO == false ? bufferSize :
                                 bufferSize / blockSize * blockSize;

        WriteToFile( fileBuckets.files[i], writeSize, buffer, _blockBuffer, fileBuckets.name, i );

        // Each bucket buffer must start at the next block-aligned boundary
        const size_t bufferOffset = _useDirectIO == false ? bufferSize :
                                    RoundUpToNextBoundaryT( bufferSize, blockSize );

        buffer += bufferOffset;
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CndWriteFile( const Command& cmd )
{
    FileSet& fileBuckets = _files[(int)cmd.file.fileId];
    WriteToFile( fileBuckets.files[cmd.file.bucket], cmd.file.size, cmd.file.buffer, _blockBuffer, fileBuckets.name, cmd.file.bucket );
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdReadFile( const Command& cmd )
{
    FileSet& fileBuckets = _files[(int)cmd.file.fileId];
    ReadFromFile( fileBuckets.files[cmd.file.bucket], cmd.file.size, cmd.file.buffer, _blockBuffer, fileBuckets.name, cmd.file.bucket );
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdSeekBucket( const Command& cmd )
{
    FileSet&     fileBuckets = _files[(int)cmd.seek.fileId];
    const uint   bucketCount = (uint)fileBuckets.files.length;

    const int64      seekOffset = cmd.seek.offset;
    const SeekOrigin seekOrigin = cmd.seek.origin;

    for( uint i = 0; i < bucketCount; i++ )
    {
        if( !fileBuckets.files[i].Seek( seekOffset, seekOrigin ) )
        {
            int err = fileBuckets.files[i].GetError();
            Fatal( "[DiskBufferQueue] Failed to seek file %s.%u with error %d (0x%x)", fileBuckets.name, i, err, err );
        }
    }
}

//-----------------------------------------------------------
inline void DiskBufferQueue::WriteToFile( FileStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket )
{
    if( !_useDirectIO )
    {
        while( size )
        {
            ssize_t sizeWritten = file.Write( buffer, size );

            if( sizeWritten < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to write to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            size -= (size_t)sizeWritten;
            buffer += sizeWritten;
        }
    }
    else
    {
        const size_t blockSize   = _blockSize;
        size_t       sizeToWrite = size / blockSize * blockSize;
        const size_t remainder   = size - sizeToWrite;

        while( sizeToWrite )
        {
            ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
            if( sizeWritten < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to write to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            ASSERT( sizeWritten <= (ssize_t)sizeToWrite );

            sizeToWrite -= (size_t)sizeWritten;
            buffer      += sizeWritten;
        }
        
        if( remainder )
        {
            ASSERT( blockBuffer );
            
            // Unnecessary zeroing of memory, but might be useful for debugging
            memset( blockBuffer, 0, blockSize );
            memcpy( blockBuffer, buffer, remainder );

            ssize_t sizeWritten = file.Write( blockBuffer, blockSize );

            if( sizeWritten < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to write block to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::ReadFromFile( FileStream& file, size_t size, byte* buffer, byte* blockBuffer, const char* fileName, uint bucket )
{
    if( !_useDirectIO )
    {
        while( size )
        {
            const ssize_t sizeRead = file.Read( buffer, size );
            if( sizeRead < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to read from '%s_%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            size  -= (size_t)sizeRead;
            buffer += sizeRead;
        }
    }
    else
    {
        const size_t blockSize  = _blockSize;
        
        /// #NOTE: All our buffers should be block aligned so we should be able to freely read all blocks to them...
        ///       Only implement remainder block reading if we really need to.
    //     size_t       sizeToRead = size / blockSize * blockSize;
    //     const size_t remainder  = size - sizeToRead;

        size_t sizeToRead = CDivT( size, blockSize ) * blockSize;

        while( sizeToRead )
        {
            ssize_t sizeRead = file.Read( buffer, sizeToRead );
            if( sizeRead < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to read from '%s_%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            ASSERT( sizeRead <= (ssize_t)sizeToRead );
            
            sizeToRead -= (size_t)sizeRead;
            buffer     += sizeRead;
        }
    }

//     if( remainder )
//     {
//         ASSERT( blockBuffer );
// 
//         file.read
//     }
}

//-----------------------------------------------------------
inline const char* DiskBufferQueue::DbgGetCommandName( Command::CommandType type )
{
    switch( type )
    {
        case DiskBufferQueue::Command::WriteFile:
            return "WriteFile";

        case DiskBufferQueue::Command::WriteBuckets:
            return "WriteBuckets";

        case DiskBufferQueue::Command::ReadFile:
            return "ReadFile";

        case DiskBufferQueue::Command::ReleaseBuffer:
            return "ReleaseBuffer";

        case DiskBufferQueue::Command::SeekFile:
            return "SeekFile";

        case DiskBufferQueue::Command::SeekBucket:
            return "SeekBucket";

        case DiskBufferQueue::Command::MemoryFence:
            return "MemoryFence";

        default:
            ASSERT( 0 );
            return nullptr;
    }
}
