#include "PlotWriter.h"
#include "ChiaConsts.h"
#include "plotdisk/jobs/IOJob.h"
#include "plotdisk/DiskBufferQueue.h"
#include "harvesting/GreenReaper.h"

//-----------------------------------------------------------
PlotWriter::PlotWriter() : PlotWriter( true ) {}

//-----------------------------------------------------------
PlotWriter::PlotWriter( bool useDirectIO )
    : _writerThread( new Thread( 4 MiB ) )
    , _directIO    ( useDirectIO )
    , _queue()
{
    _readyToPlotSignal.Signal();    // Start ready to plot

    // #MAYBE: Start the thread at first plot?
    _writerThread->Run( WriterThreadEntry, this );
}

//-----------------------------------------------------------
PlotWriter::PlotWriter( DiskBufferQueue& ownerQueue ) : PlotWriter( true )
{
    _owner = &ownerQueue;
}

//-----------------------------------------------------------
PlotWriter::~PlotWriter()
{
    if( _writerThread )
    {
        ExitWriterThread();
        ASSERT( _writerThread->HasExited() );
        delete _writerThread;
    }

    if( _plotPathBuffer.Ptr() )
        free( _plotPathBuffer.Ptr() );
    if( _plotFinalPathName )
        free( _plotFinalPathName );
    if( _writeBuffer.Ptr() )
        bbvirtfree( _writeBuffer.Ptr() );
}

//-----------------------------------------------------------
void PlotWriter::EnablePlotChecking( PlotChecker& checker )
{
    _plotChecker = &checker;
}

//-----------------------------------------------------------
bool PlotWriter::BeginPlot( PlotVersion version, 
    const char* plotFileDir, const char* plotFileName, const byte plotId[32],
    const byte* plotMemo, const uint16 plotMemoSize, const uint32 compressionLevel )
{
    _readyToPlotSignal.Wait();

    const bool r = BeginPlotInternal( version, plotFileDir, plotFileName, plotId, plotMemo, plotMemoSize, compressionLevel );

    if( !r )
        _readyToPlotSignal.Signal();

    return r;
}

//-----------------------------------------------------------
// bool PlotWriter::BeginCompressedPlot( PlotVersion version, 
//         const char* plotFileDir, const char* plotFileName, const byte plotId[32],
//         const byte* plotMemo, const uint16 plotMemoSize,
//         uint32 compressionLevel )
// {
//     ASSERT( compressionLevel );

//     return BeginPlot( version, plotFileDir, plotFileName, plotId, plotMemo, plotMemoSize, (int32)compressionLevel );
// }

//-----------------------------------------------------------
bool PlotWriter::BeginPlotInternal( PlotVersion version,
        const char* plotFileDir, const char* plotFileName, const byte plotId[32],
        const byte* plotMemo, const uint16 plotMemoSize,
        int32 compressionLevel )
{
    if( _dummyMode ) return true;

    ASSERT( !_stream.IsOpen() );

    // Ensure we don't start a 
    if( _stream.IsOpen() )
        return false;

    if( !plotFileDir || !*plotFileDir || !plotFileDir || !*plotFileName )
        return false;

    if( !plotMemo || !plotMemoSize )
        return false;

    ASSERT( compressionLevel >= 0 && compressionLevel <= 40 );

    if( compressionLevel > 0 && version < PlotVersion::v2_0 )
        return false;

    /// Copy plot file path
    {
        const size_t dirLength   = strlen( plotFileDir  );
        const size_t nameLength  = strlen( plotFileName );
        const size_t nameBufSize = dirLength + nameLength + 2;

        if( _plotPathBuffer.length < nameBufSize )
        {
            _plotPathBuffer.values = (char*)realloc( _plotPathBuffer.values, nameBufSize );
            _plotPathBuffer.length = nameBufSize;
        }

        auto plotFilePath = _plotPathBuffer;

        memcpy( plotFilePath.Ptr(), plotFileDir, dirLength );
        plotFilePath = plotFilePath.Slice( dirLength );
        if( plotFileDir[dirLength-1] != '/' && plotFileDir[dirLength-1] != '\\' )
        {
            *plotFilePath.values = '/';
            plotFilePath = plotFilePath.Slice( 1 );
        }

        memcpy( plotFilePath.Ptr(), plotFileName, nameLength );
        plotFilePath[nameLength] = 0;
    }

    /// Open the plot file
    //  #NOTE: We need to read access because we allow seeking, but in order to
    //         remain block-aligned, we might have to read data from the seek location.
    const FileFlags flags = FileFlags::LargeFile | ( _directIO ? FileFlags::NoBuffering : FileFlags::None );
    if( !_stream.Open( _plotPathBuffer.Ptr(), FileMode::Create, FileAccess::ReadWrite, flags ) )
        return false;

    /// Make the header
    size_t headerSize = 0;

    if( version == PlotVersion::v1_0 )
    {
        headerSize =
            ( sizeof( kPOSMagic ) - 1 ) +
            32 +            // plot id
            1  +            // k
            2  +            // kFormatDescription length
            ( sizeof( kFormatDescription ) - 1 ) +
            2  +            // Memo length
            plotMemoSize +  // Memo
            80              // Table pointers
            ;
    }
    else if( version == PlotVersion::v2_0 )
    {
        headerSize =
            4  +            // magic
            4  +            // version number
            32 +            // plot id
            1  +            // k
            2  +            // Memo length
            plotMemoSize +  // Memo
            4  +            // flags
            80 +            // Table pointers
            80              // Table sizes
            ;

        if( compressionLevel > 0 )
            headerSize += 1;
    }
    else
        return false;

    if( _writeBuffer.Ptr() == nullptr )
    {
        const size_t allocSize = RoundUpToNextBoundaryT( BUFFER_ALLOC_SIZE, _stream.BlockSize() );

        if( _writeBuffer.Ptr() && allocSize > _writeBuffer.Length() )
            bbvirtfree_span( _writeBuffer );

        _writeBuffer.values = bbvirtalloc<byte>( allocSize );
        _writeBuffer.length = allocSize;
    }

    _headerSize = headerSize;

    if( version == PlotVersion::v1_0 )
    {
        byte* headerWriter = _writeBuffer.Ptr();

        // Magic
        memcpy( headerWriter, kPOSMagic, sizeof( kPOSMagic ) - 1 );
        headerWriter += sizeof( kPOSMagic ) - 1;

        // Plot Id
        memcpy( headerWriter, plotId, 32 );
        headerWriter += 32;

        // K
        *headerWriter++ = (byte)_K;

        // Format description
        *((uint16*)headerWriter) = Swap16( (uint16)(sizeof( kFormatDescription ) - 1) );
        headerWriter += 2;
        memcpy( headerWriter, kFormatDescription, sizeof( kFormatDescription ) - 1 );
        headerWriter += sizeof( kFormatDescription ) - 1;

        // Memo
        *((uint16*)headerWriter) = Swap16( plotMemoSize );
        headerWriter += 2;
        memcpy( headerWriter, plotMemo, plotMemoSize );
        headerWriter += plotMemoSize;
        
        // Empty table pointers
        headerWriter += sizeof( _tablePointers );

        ASSERT( (size_t)(uintptr_t)( headerWriter - _writeBuffer.Ptr() ) == headerSize );
    }
    else if( version == PlotVersion::v2_0 )
    {
        byte* headerWriter = _writeBuffer.Ptr();
        
        // The start of the header buffer should be 4-byte aligned, so we can do this
        *((uint32*)headerWriter) = (uint32)CHIA_PLOT_V2_MAGIC;  headerWriter += 4;  // magic
        *((uint32*)headerWriter) = (uint32)CHIA_PLOT_VERSION_2; headerWriter += 4;  // file version

        // Plot Id
        memcpy( headerWriter, plotId, 32 );
        headerWriter += 32;

        // K
        *headerWriter++ = (byte)_K;

        // Memo
        *((uint16*)headerWriter) = Swap16( plotMemoSize );
        headerWriter += 2;
        memcpy( headerWriter, plotMemo, plotMemoSize );
        headerWriter += plotMemoSize;

        // Flags
        PlotFlags flags = PlotFlags::None;
        static_assert( sizeof( flags ) == 4 );

        if( compressionLevel > 0 )
            flags |= PlotFlags::Compressed;

        memcpy( headerWriter, &flags, sizeof( flags ) ); headerWriter += sizeof( flags );

        if( compressionLevel > 0 )
            *headerWriter++ = (byte)compressionLevel;

        // Empty tables pointer and sizes
        headerWriter += sizeof( _tablePointers ) * 2;

        ASSERT( (size_t)(uintptr_t)(headerWriter - _writeBuffer.Ptr()) == headerSize );
    }

    // Write header, block-aligned, tables will start at the aligned position
    const ssize_t headerWriteSize = (ssize_t)RoundUpToNextBoundaryT( _headerSize, _stream.BlockSize() );
    FatalIf( headerWriteSize != _stream.Write( _writeBuffer.Ptr(), (size_t)headerWriteSize ),
        "Failed to write plot header with error: %d.", _stream.GetError() );

    // Reset state
    _plotVersion        = version;
    _bufferBytes        = 0;
    _haveTable          = false;
    _currentTable       = PlotTable::Table1;
    _position           = headerWriteSize;
    // _tablesBeginAddress = headerWriteSize;
    _tableStart         = 0;
    _unalignedFileSize  = headerWriteSize;
    _alignedFileSize    = headerWriteSize;

    memset( _tablePointers, 0, sizeof( _tablePointers ) );
    memset( _tableSizes   , 0, sizeof( _tablePointers ) );

    return true;
}

//-----------------------------------------------------------
void PlotWriter::EndPlot( const bool rename )
{
    if( _dummyMode ) return;

    ASSERT( _stream.IsOpen() );

    // auto& cmd = GetCommand( CommandType::EndPlot );
    // cmd.endPlot.fence    = &_completedFence;
    // cmd.endPlot.rename   = rename;
    // SubmitCommands();

    SubmitCommand({ .type = CommandType::EndPlot,
        .endPlot{ .fence    = &_completedFence,
                  .rename   = rename
        }
    });
}

//-----------------------------------------------------------
bool PlotWriter::CheckPlot()
{
    if( _dummyMode || !_plotChecker ) return false;

    const char* plotPath = _plotPathBuffer.Ptr();

    PlotCheckResult checksResult{};
    _plotChecker->CheckPlot( plotPath, &checksResult );

    if( !checksResult.error.empty() )
        return false;

    return !checksResult.deleted;
}


//-----------------------------------------------------------
void PlotWriter::WaitForPlotToComplete()
{
    if( _dummyMode ) return;

    ASSERT( _headerSize );
    _completedFence.Wait();

    _headerSize = 0;
    ASSERT( !_stream.IsOpen() );
}

//-----------------------------------------------------------
void PlotWriter::DumpTables()
{
    const uint64* tablePointers = _tablePointers;
    const uint64* tableSizes    = _tableSizes;

    Log::Line( "Final plot table pointers: " );
    for( int i = 0; i < 10; i++ )
    {
        const uint64 addy = tablePointers[i];

        if( i < 7 )
            Log::Line( " Table %d: %16llu ( 0x%016llx )", i+1, addy, addy );
        else
            Log::Line( " C %d    : %16llu ( 0x%016llx )", i-6, addy, addy );
    }
    Log::Line( "" );

    Log::Line( "Final plot table sizes: " );
    for( int i = 0; i < 10; i++ )
    {
        const uint64 size = tableSizes[i];

        if( i < 7 )
            Log::Line( " Table %d: %.2lf MiB", i+1, (double)size BtoMB );
        else
            Log::Line( " C %d    : %.2lf MiB", i-6, (double)size BtoMB );
    }
    Log::Line( "" );
}

//-----------------------------------------------------------
void PlotWriter::BeginTable( const PlotTable table )
{
    if( _dummyMode ) return;

    SubmitCommand({
        .type = CommandType::BeginTable,
        .beginTable{ .table = table }
    });
    // auto& cmd = GetCommand( CommandType::BeginTable );
    // auto cmd = GetCommand( CommandType::BeginTable );
    // cmd.beginTable.table = table;
    // SubmitCommands();
}

//-----------------------------------------------------------
void PlotWriter::ReserveTableSize( const PlotTable table, const size_t size )
{
    if( _dummyMode ) return;

    // auto& cmd = GetCommand( CommandType::ReserveTable );
    // cmd.reserveTable.table = table;
    // cmd.reserveTable.size  = size;
    // SubmitCommands();

     SubmitCommand({
        .type = CommandType::ReserveTable,
        .reserveTable { 
            .table = table,
            .size  = size 
        }
    });
}

//-----------------------------------------------------------
void PlotWriter::EndTable()
{
    if( _dummyMode ) return;

    // auto& cmd = GetCommand( CommandType::EndTable );
    // SubmitCommands();
    SubmitCommand({ .type = CommandType::EndTable });
}

//-----------------------------------------------------------
void PlotWriter::WriteTableData( const void* data, const size_t size )
{
    if( _dummyMode ) return;

    // auto& cmd = GetCommand( CommandType::WriteTable );
    // cmd.writeTable.buffer = (byte*)data;
    // cmd.writeTable.size   = size;
    // SubmitCommands();

    SubmitCommand({ .type = CommandType::WriteTable,
        .writeTable{ .buffer = (byte*)data,
                     .size   = size,
        }
    });
}

//-----------------------------------------------------------
void PlotWriter::WriteReservedTable( const PlotTable table, const void* data )
{
    if( _dummyMode ) return;

    // auto& cmd = GetCommand( CommandType::WriteReservedTable );
    // cmd.writeReservedTable.table  = table;
    // cmd.writeReservedTable.buffer = (byte*)data;
    // SubmitCommands();

    SubmitCommand({ .type = CommandType::WriteReservedTable,
        .writeReservedTable{ 
            .table  = table,
            .buffer = (byte*)data
        }
    });
}

//-----------------------------------------------------------
void PlotWriter::SignalFence( Fence& fence )
{
    if( _dummyMode ) 
    {
        fence.Signal();
        return;
    }

    // auto& cmd = GetCommand( CommandType::SignalFence );
    // cmd.signalFence.fence    = &fence;
    // cmd.signalFence.sequence = -1;
    // SubmitCommands();

    SubmitCommand({ .type = CommandType::SignalFence,
        .signalFence{ .fence    = &fence,
                      .sequence = -1
        }
    });
}

//-----------------------------------------------------------
void PlotWriter::SignalFence( Fence& fence, uint32 sequence )
{
    if( _dummyMode )
    {
        fence.Signal( sequence );
        return;
    }

    // auto& cmd = GetCommand( CommandType::SignalFence );
    // cmd.signalFence.fence    = &fence;
    // cmd.signalFence.sequence = (int64)sequence;
    // SubmitCommands();
    
    SubmitCommand({ .type = CommandType::SignalFence,
        .signalFence{ .fence    = &fence,
                      .sequence = (int64)sequence
        }
    });
}

//-----------------------------------------------------------
void PlotWriter::CallBack( std::function<void()> func )
{
    if( _dummyMode )
    {
        func();
        return;
    }

    // auto& cmd = GetCommand( CommandType::CallBack );
    // cmd.callback.func = new std::function<void()>( std::move( func ) );
    // SubmitCommands();

    SubmitCommand({ .type =  CommandType::CallBack,
        .callback{ .func = new std::function<void()>( std::move( func ) ) }
    });
}

//-----------------------------------------------------------
void PlotWriter::ExitWriterThread()
{
    // Signal writer thread to exit after it finishes its commands
    // auto& cmd = GetCommand( CommandType::Exit );
    // cmd.signalFence.fence = &_completedFence;
    // SubmitCommands();

    SubmitCommand({ .type = CommandType::Exit,
        .signalFence{ .fence = &_completedFence }
    });

    // Wait for writer thread to exit
    _completedFence.Wait();
    ASSERT( _writerThread->HasExited() );
}

//-----------------------------------------------------------
PlotWriter::Command& PlotWriter::GetCommand( CommandType type )
{
    Panic( "Don't use me!" );

    // if( _owner != nullptr )
    // {
    //     auto* cmd = _owner->GetCommandObject( DiskBufferQueue::Command::CommandType::PlotWriterCommand );
    //     ASSERT( cmd );

    //     ZeroMem( &cmd->plotWriterCmd );
    //     cmd->plotWriterCmd.writer   = this;
    //     cmd->plotWriterCmd.cmd.type = type;
    //     return cmd->plotWriterCmd.cmd;
    // }
    // else
    // {
    //     Command* cmd = nullptr;
    //     while( !_queue.Write( cmd ) )
    //     {
    //         Log::Line( "[PlotWriter] Command buffer full. Waiting for commands." );
    //         auto waitTimer = TimerBegin();

    //         // Block and wait until we have commands free in the buffer
    //         _cmdConsumedSignal.Wait();
            
    //         Log::Line( "[PlotWriter] Waited %.6lf seconds for a Command to be available.", TimerEnd( waitTimer ) );
    //     }
        
    //     ASSERT( cmd );
    //     ZeroMem( cmd );
    //     cmd->type = type;

    //     return *cmd;
    // }
}

//-----------------------------------------------------------
void PlotWriter::SubmitCommand( const Command cmd )
{
    std::unique_lock lock( _queueLock );
    _queue.push( cmd );
    _cmdReadySignal.Signal();
}

//-----------------------------------------------------------
void PlotWriter::SubmitCommands()
{Panic( "" );
    // if( _owner != nullptr )
    // {
    //     _owner->CommitCommands();
    // }
    // else
    // {
    //     _queue.Commit();
    //     _cmdReadySignal.Signal();
    // }
}


///
/// Writer Thread
///
//-----------------------------------------------------------
void PlotWriter::WriterThreadEntry( PlotWriter* self )
{
    self->WriterThreadMain();
}

//-----------------------------------------------------------
void PlotWriter::WriterThreadMain()
{
    const uint32 MAX_COMMANDS = 64;
    Command commands[MAX_COMMANDS];

    for( ;; )
    {
        // Wait for commands
        _cmdReadySignal.Wait();

        // Load commands from the queue
        // int32 cmdCount;
        // while( ( ( cmdCount = _queue.Dequeue( commands, MAX_COMMANDS ) ) ) )
        // {
        //     // Notify we consumed commands
        //     _cmdConsumedSignal.Signal();

        //     for( int32 i = 0; i < cmdCount; i++ )
        //     {
        //         if( commands[i].type == CommandType::Exit )
        //         {
        //             commands[i].signalFence.fence->Signal();
        //             return;
        //         }

        //         ExecuteCommand( commands[i] );
        //     }
        // }

        // Consume commands from the queue and execute them
        // until there are none more found in the queue
        size_t cmdCount = 0;
        for( ;; )
        {
            // Consume commands from queue
            {
                std::unique_lock lock( _queueLock );
                cmdCount = std::min<size_t>( _queue.size(), MAX_COMMANDS );
                
                for( size_t i = 0; i < cmdCount; i++ )
                {
                    commands[i] = _queue.front();
                    _queue.pop();
                }
            }

            // Notify we consumed commands
            _cmdConsumedSignal.Signal();

            if( cmdCount < 1 )
                break;

            // Execute commands
            for( int32 i = 0; i < cmdCount; i++ )
            {
                if( commands[i].type == CommandType::Exit )
                {
                    commands[i].signalFence.fence->Signal();
                    return;
                }

                ExecuteCommand( commands[i] );
            }
        }
    }
}

//-----------------------------------------------------------
void PlotWriter::ExecuteCommand( const Command& cmd )
{
    switch( cmd.type )
    {
        default: return;

        case CommandType::BeginTable         : CmdBeginTable( cmd ); break;
        case CommandType::EndTable           : CmdEndTable( cmd ); break;
        case CommandType::WriteTable         : CmdWriteTable( cmd ); break;
        case CommandType::ReserveTable       : CmdReserveTable( cmd ); break;
        case CommandType::WriteReservedTable : CmdWriteReservedTable( cmd ); break;
        case CommandType::EndPlot            : CmdEndPlot( cmd ); break;
        case CommandType::CallBack           : CmdCallBack( cmd ); break;

        case CommandType::SignalFence:
            if( cmd.signalFence.sequence >= 0 )
                cmd.signalFence.fence->Signal( (uint32)cmd.signalFence.sequence );
            else
                cmd.signalFence.fence->Signal();
        break;
    }
}


//-----------------------------------------------------------
void PlotWriter::SeekToLocation( const size_t location )
{
    // We need to read the block we seeked to if:
    // - The seeked-to block is NOT the current block AND
    // - The seeked-to block already existed

    const size_t blockSize              = _stream.BlockSize();
    // const size_t currentAlignedLocation = _position / blockSize * blockSize;
    const size_t alignedLocation        = location / blockSize * blockSize;

    if( _bufferBytes )
    {
        FlushRetainedBytes();
        
        // Seek back to the location
        FatalIf( !_stream.Seek( -(int64)blockSize, SeekOrigin::Current ),
            "Plot file seek failed with error: %d", _stream.GetError() );
    }
    ASSERT( _bufferBytes == 0 );
    

    FatalIf( !_stream.Seek( (int64)alignedLocation, SeekOrigin::Begin ),
        "Plot file seek failed with error: %d", _stream.GetError() );
    
    // Read the block we just seeked-to,
    // unless it is at the unaligned end, and the end is block-aligned (start of a block)
    if( alignedLocation < _unalignedFileSize )
    {
        FatalIf( (ssize_t)blockSize != _stream.Read( _writeBuffer.Ptr(), blockSize ),
            "Plot file read failed with error: %d", _stream.GetError() );

        // Seek back to the location
        FatalIf( !_stream.Seek( -(int64)blockSize, SeekOrigin::Current ),
            "Plot file seek failed with error: %d", _stream.GetError() );
    }

    _bufferBytes = location - alignedLocation;

    _position          = location;
    _unalignedFileSize = std::max( _position, _unalignedFileSize );
}

//-----------------------------------------------------------
size_t PlotWriter::BlockAlign( const size_t size ) const
{
    return RoundUpToNextBoundaryT( size, _stream.BlockSize() );
}

//-----------------------------------------------------------
void PlotWriter::FlushRetainedBytes()
{
    if( _bufferBytes > 0 )
    {
        const size_t blockSize = _stream.BlockSize();
        ASSERT( RoundUpToNextBoundaryT( _bufferBytes, blockSize ) == blockSize )

        int32 err;
        PanicIf( !IOJob::WriteToFile( _stream, _writeBuffer.Ptr(), blockSize, nullptr, blockSize, err ),
            "Failed to write to plot with error %d:", err );

        _bufferBytes     = 0;
        _alignedFileSize = std::max( _alignedFileSize, CDivT( _unalignedFileSize, blockSize ) * blockSize );
    }
}

//-----------------------------------------------------------
void PlotWriter::WriteData( const byte* src, const size_t size )
{
    // #TODO: If the input data is aligned, and our position is aligned, bypass this.

    // Determine how many blocks will be written
    const size_t capacity    = _writeBuffer.Length();
    const size_t blockSize   = _stream.BlockSize();
    ASSERT( _bufferBytes < blockSize );

    const size_t startBlock  = _position / blockSize;
    const size_t endBlock    = ( _position + size ) / blockSize;
    // const size_t blocksWritten = endBlock - startBlock;


    byte* writeBuffer = _writeBuffer.Ptr();
    int32 err         = 0;

    size_t sizeRemaining = _bufferBytes + size;                    // Data that we will keep in the temporary buffer for later writing
    size_t sizeToWrite   = sizeRemaining / blockSize * blockSize;  // Block-aligned data we can write
    
    // Substract the blocks that will be written, or if none
    // will be written, we only need to copy over the size (no blocks filled)
    sizeRemaining = std::min( sizeRemaining - sizeToWrite, size );

    // Write as much block-aligned data as we can
    while( sizeToWrite )
    {
        const size_t spaceAvailable = capacity - _bufferBytes;
        const size_t copySize       = std::min( spaceAvailable, sizeToWrite - _bufferBytes );
        ASSERT( (copySize + _bufferBytes) / blockSize * blockSize == (copySize + _bufferBytes) );

        memcpy( writeBuffer + _bufferBytes, src, copySize );

        size_t writeSize = _bufferBytes + copySize;
        sizeToWrite -= writeSize;
        src         += copySize;
        _bufferBytes = 0;

        ASSERT( writeSize / blockSize * blockSize == writeSize );


        size_t totalSizeWritten = 0;
        size_t sizeWritten      = 0;
        while( !IOJob::WriteToFile( _stream, writeBuffer, writeSize, nullptr, blockSize, err, &sizeWritten ) )
        {
            ASSERT( writeSize / blockSize * blockSize == writeSize );

            bool isOutOfSpace = false;

            #if !defined( _WIN32 )
                isOutOfSpace = err == ENOSPC;
            #else
                // #TODO: Add out of space error check for windows
            #endif

            // Wait indefinitely until there's more space
            if( isOutOfSpace )
            {
                const long SLEEP_TIME = 10 * (long)1000;

                Log::Line( "No space left in plot output directory for plot '%s'. Waiting %.1lf seconds before trying again...",
                            this->_plotPathBuffer.Ptr(), (double)SLEEP_TIME/1000.0 );
                Thread::Sleep( SLEEP_TIME );
            }
            else
                Log::Line( "Error %d encountered when writing to plot '%s.", err, this->_plotPathBuffer.Ptr() );

            totalSizeWritten += sizeWritten;
            if( totalSizeWritten >= writeSize )
                break;

            ASSERT( sizeWritten >= writeSize );

            writeBuffer += sizeWritten;
            writeSize   -= sizeWritten;
            sizeWritten = 0;
        }
    }


    // Data remaining in the last block has to be read back if the last block had already been written to
    const size_t maxBlockWritten = _unalignedFileSize / blockSize;

    if( maxBlockWritten >= endBlock && endBlock > startBlock )
    {
        ASSERT( _bufferBytes == 0 );
        PanicIf( _stream.Read( _writeBuffer.Ptr(), blockSize ) != (ssize_t)blockSize, 
            "Plot file read failed: %d", _stream.GetError() );
        
        // Seek back to the last block
        PanicIf( !_stream.Seek( -(int64)blockSize, SeekOrigin::Current ),
            "Plot file seek failed: %d", _stream.GetError() );
    }

    if( sizeRemaining > 0 )
    {
        memcpy( writeBuffer + _bufferBytes, src, sizeRemaining );
        _bufferBytes += sizeRemaining;
    }
    ASSERT( _bufferBytes < blockSize );

    // Update position and file size
    _position        += size;
    _unalignedFileSize = std::max( _position, _unalignedFileSize );
    _alignedFileSize   = std::max( _alignedFileSize, _unalignedFileSize / blockSize * blockSize );
}

///
/// Commands
///
//-----------------------------------------------------------
void PlotWriter::CmdBeginTable( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::BeginTable );
    ASSERT( !_haveTable );
    if( _haveTable )
        return;

    const PlotTable table = cmd.beginTable.table;

    _currentTable = table;
    _haveTable    = true;
    
    _tableStart                = _position;
    _tablePointers[(int)table] = _tableStart;
}

//-----------------------------------------------------------
void PlotWriter::CmdEndTable( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::EndTable );
    ASSERT( _haveTable );
    if( !_haveTable )
        return;

    ASSERT( _position >= _tableStart );
    _tableSizes[(int)_currentTable] = _position - _tableStart;
    _haveTable = false;
}

//-----------------------------------------------------------
void PlotWriter::CmdWriteTable( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::WriteTable );

    auto& c = cmd.writeTable;
    ASSERT( c.size );
    
    WriteData( c.buffer, c.size );
}

//-----------------------------------------------------------
void PlotWriter::CmdReserveTable( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::ReserveTable );
    ASSERT( !_haveTable );
    if( _haveTable )
        return;

    auto& c = cmd.reserveTable;
    ASSERT( _tablePointers[(int)c.table] == 0 );

    _tablePointers[(int)c.table] = _position;
    _tableSizes   [(int)c.table] = c.size;

    SeekToLocation( _position + c.size );
}

//-----------------------------------------------------------
void PlotWriter::CmdWriteReservedTable( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::WriteReservedTable );

    const auto& c = cmd.writeReservedTable;

    const size_t currentLocation = _position;

    const size_t tableLocation   = _tablePointers[(int)c.table];
    const size_t tableSize       = _tableSizes[(int)c.table];
    
    ASSERT( tableSize );
    ASSERT( tableLocation != 0 );

    SeekToLocation( tableLocation );
    WriteData( c.buffer, tableSize );
    SeekToLocation( currentLocation );
}

//-----------------------------------------------------------
void PlotWriter::CmdSignalFence( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::SignalFence || cmd.type == CommandType::EndPlot );

    if( cmd.signalFence.sequence >= 0 )
        cmd.signalFence.fence->Signal( (uint32)cmd.signalFence.sequence );
    else
        cmd.signalFence.fence->Signal();
}

//-----------------------------------------------------------
void PlotWriter::CmdEndPlot( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::EndPlot );
    ASSERT( _stream.IsOpen() );
    ASSERT( !_haveTable );

    // Write table sizes
    size_t tablePointersLoc;
    if( _plotVersion == PlotVersion::v1_0 )
        tablePointersLoc = _headerSize - 80;
    else if( _plotVersion == PlotVersion::v2_0 )
        tablePointersLoc = _headerSize - 160;
    else
    {
        ASSERT( 0 );
    }

    uint64 tablePointersBE[10];
    for( uint32 i = 0; i < 10; i++ )
        tablePointersBE[i] = Swap64( _tablePointers[i] );

    SeekToLocation( tablePointersLoc );
    WriteData( (byte*)tablePointersBE, sizeof( tablePointersBE ) );

    if( _plotVersion == PlotVersion::v2_0 )
        WriteData( (byte*)_tableSizes, sizeof( _tableSizes ) );
    
    ASSERT( _position == _headerSize );

    FlushRetainedBytes();
    _stream.Close();

    bool renamePlot = cmd.endPlot.rename;
    if( _plotChecker )
    {
        renamePlot = CheckPlot();
    }

    // Now rename to its final non-temp name
    if( renamePlot )
    {
        const uint32 RETRY_COUNT  = 10;
        const long   MS_WAIT_TIME = 1000;

        const char* tmpName = _plotPathBuffer.Ptr();
        const size_t pathLen = strlen( tmpName );

        _plotFinalPathName = (char*)realloc( _plotFinalPathName, pathLen + 1 );
        memcpy( _plotFinalPathName, tmpName, pathLen );
        _plotFinalPathName[pathLen-4] = '\0';

        Log::Line( "%s -> %s", tmpName, _plotFinalPathName );

        int32 error = 0;

        for( uint32 i = 0; i < RETRY_COUNT; i++ )
        {
            const bool success = FileStream::Move( tmpName, _plotFinalPathName, &error );

            if( success )
                break;
            
            Log::Line( "[PlotWriter] Error: Could not rename plot file with error: %d.", error );

            if( i+1 == RETRY_COUNT)
            {
                Log::Line( "[PlotWriter] Error: Failed to to rename plot file after %u retries. Please rename manually.", RETRY_COUNT );
                break;
            }

            Log::Line( "[PlotWriter] Retrying in %.2lf seconds...", MS_WAIT_TIME / 1000.0 );
            Thread::Sleep( MS_WAIT_TIME );
        }
    }

    _readyToPlotSignal.Signal();
    cmd.endPlot.fence->Signal();
}

//-----------------------------------------------------------
void PlotWriter::CmdCallBack( const Command& cmd )
{
    ASSERT( cmd.type == CommandType::CallBack );

    (*cmd.callback.func)();
    delete cmd.callback.func;
}