#pragma once
#include "util/Util.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotting/PlotTools.h"
#include "pos/chacha8.h"
#include "threading/MTJob.h"
#include "BitBucketWriter.h"
#include "util/StackAllocator.h"

template<uint32 _numBuckets>
struct DiskF1
{
    using Info = DiskPlotInfo<TableId::Table1, _numBuckets>;
    
    static constexpr uint32 _k = _K;
    static constexpr int64  _entriesPerBucket = (int64)CDiv( 1ull << _k, _numBuckets );


    //-----------------------------------------------------------
    inline DiskF1( DiskPlotContext& context, const FileId fileId )
        : _context( context )
        // , _entriesPerBucket( (int64)CDiv( _k, _numBuckets ) )
        , _entriesPerThread( CDiv( _entriesPerBucket, (int32)context.f1ThreadCount ) )
        , _bitWriter( *context.ioQueue, fileId, context.t1FsBlocks )
    {
        DiskBufferQueue& ioQueue = *context.ioQueue;
        
        const uint32 threadCount     = context.f1ThreadCount;
        const uint32 entriesPerBlock = kF1BlockSizeBits / _k;
        const uint64 blocksPerThread = CDiv( _entriesPerThread, entriesPerBlock );
        const size_t entryAllocCount = blocksPerThread * entriesPerBlock * threadCount;
        
        StackAllocator stack( context.heapBuffer, context.heapSize );
        _blocks  = stack.CAlloc<uint32>( entryAllocCount );
        _entries = stack.CAlloc<uint64>( entryAllocCount );

        // Must have at least n bits left
        const size_t ioBitsPerBucket  = RoundUpToNextBoundary( Info::EntrySizePackedBits * entryAllocCount, 64u );
        const size_t ioBytesPerBucket = ioBitsPerBucket / 8; 
        Log::Line( "Minimum IO size required per bucket @ %u buckets: %.2lf MiB", 
            _numBuckets, (double)ioBytesPerBucket BtoMB );
        
        Log::Line( "F1 IO size @ %u buckets: %.2lf MiB", 
            _numBuckets, (double)stack.Remainder() BtoMB );

        FatalIf( stack.Remainder() < ioBitsPerBucket, "Not enough IO reserve size." );

        ioQueue.ResetHeap( stack.Remainder(), stack.Top() );
    }

    //-----------------------------------------------------------
    inline void GenF1()
    {
        using Job = AnonPrefixSumJob<uint64>;
        Job::Run( *_context.threadPool, _context.f1ThreadCount, [=]( Job* self ) {
            
            const uint32 threadCount      = self->JobCount();
            const uint32 id               = self->JobId();
            const uint32 kMinusKExtraBits = _k - kExtraBits;
            const uint32 bucketBitShift   = _k - bblog2( _numBuckets );
            
            const int32 entriesPerBlock  = kF1BlockSizeBits / (int32)_k;
            const int64 blocksPerThread  = CDiv( _entriesPerThread, entriesPerBlock );

            const size_t entrySizeBits   = Info::YBitSize + _k;    // y + x

            // uint32* blocks  = _context.heapBuffer + ( (size_t)blocksPerThread * kF1BlockSize ) * id;
            uint32* blocks  = _blocks + (size_t)blocksPerThread * (uint32)entriesPerBlock * id;
            uint64* entries = _entries;

            int64 tableEntryCount = 1ll << _k;
            
            byte key[BB_PLOT_ID_LEN] = { 1 };
            memcpy( key + 1, _context.plotId, BB_PLOT_ID_LEN-1 );

            chacha8_ctx chacha;
            chacha8_keysetup( &chacha, key, 256, NULL );

            uint64 nextX = 0;

            auto& bitWriter = _bitWriter;

            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            {
                const int64 bucketEntryCount = std::min( _entriesPerBucket, tableEntryCount );

                int64  entriesPerThread = bucketEntryCount / (int64)threadCount;
                uint64 x                = nextX + (uint64)entriesPerThread * id;   // Esure this is set before we update entriesPerThread on the last job

                if( self->IsLastThread() )
                    entriesPerThread = bucketEntryCount - entriesPerThread * (int64)( threadCount - 1 );

                const uint64 chachaBlock = x / kF1BlockSize;
                const uint64 blockCount  = (uint64)CDivT( entriesPerThread, (int64)entriesPerBlock );

                // ChaCha gen
                chacha8_get_keystream( &chacha, chachaBlock, blockCount, (byte*)blocks );

                // Distribue to buckets
                uint64 counts     [_numBuckets];
                uint64 pfxSum     [_numBuckets];
                uint64 totalCounts[_numBuckets];

                memset( counts, 0, sizeof( counts ) );
                for( int64 i = 0; i < entriesPerThread; i++ )
                    counts[Swap32( blocks[i] ) >> bucketBitShift] ++;

                self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

                if( self->IsControlThread() )
                {
                    self->LockThreads();

                    for( uint32 i = 0; i < _numBuckets; i++ )
                        _context.bucketCounts[0][i] += (uint32)totalCounts[i];

                    // Convert counts to bit sizes
                    for( uint32 i = 0; i < _numBuckets; i++ )
                        totalCounts[i] *= entrySizeBits;
                    
                    bitWriter.BeginWriteBuckets( totalCounts );
                    self->ReleaseThreads();
                }
                else
                {
                    self->WaitForRelease();
                }

                const uint32 yBits = Info::YBitSize;
                const uint64 yMask = ( 1ull << yBits ) - 1;

                for( int64 i = 0; i < entriesPerThread; i++ )
                {
                          uint64 y   = Swap32( blocks[i] );
                    const uint32 dst = --pfxSum[y >> bucketBitShift];

                    // Store bit-compressed already
                    const uint64 xi = ( x + (uint64)i );
                    y = ( ( y << kExtraBits ) | ( xi >> kMinusKExtraBits ) ) & yMask;
                    entries[dst] = ( xi << yBits ) | y;
                    // entries[dst] = ( y << kExtraBits ) | ( ( x + (uint64)i ) >> kMinusKExtraBits );
                }
                

                // Bit-compress each bucket
                for( uint32 i = 0; i < _numBuckets; i++ )
                {
                    const uint64 offset = pfxSum[i];

                    BitWriter writer = bitWriter.GetWriter( bucket, offset * 8 );
                    
                    const uint64* entry = entries + offset;
                    const uint64* end   = entry + counts[i];
                    ASSERT( counts[i] >= 2 );

                    // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                    writer.Write( entry[0], entrySizeBits );
                    writer.Write( entry[1], entrySizeBits );
                    entry += 2;

                    self->SyncThreads();

                    while( entry < end )
                    {
                        writer.Write( *entry, entrySizeBits );
                        entry++;
                    }
                }
                
                self->SyncThreads();

                if( self->IsControlThread() )
                    bitWriter.Submit();

                // Next bucket
                tableEntryCount -= bucketEntryCount;
                nextX           += bucketEntryCount;
            }

            if( self->IsControlThread() )
                bitWriter.SubmitLeftOvers();
        });
    }

private:
    DiskPlotContext& _context;
    int64            _entriesPerThread;
    uint64           _blocksPerBucket;
    uint32*          _blocks;
    uint64*          _entries;  // Work buffer for distributed bucket entries
    FileId           _fileId;
    BitBucketWriter<_numBuckets>  _bitWriter;
};