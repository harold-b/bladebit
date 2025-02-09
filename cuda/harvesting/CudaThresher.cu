#include "pch.h"            // Make intellisens happy in some IDEs...
#include "harvesting/Thresher.h"
#include "harvesting/GreenReaperInternal.h"
#include "harvesting/GreenReaper.h"
#include "pos/chacha8.h"
#include "CudaF1.h"
#include "CudaFx.h"
#include "CudaMatch.h"
#include "CudaUtil.h"
#include "CudaPlotContext.h"
#include "cub/device/device_radix_sort.cuh"
#include "ChiaConsts.h"
#include "plotting/PlotTypes.h"

// #define BB_CUDA_HARVEST_USE_TIMINGS 1

namespace {
    struct Timings
    {
        NanoSeconds f1       = NanoSeconds::zero();
        NanoSeconds match    = NanoSeconds::zero();
        NanoSeconds fx       = NanoSeconds::zero();
        NanoSeconds sort     = NanoSeconds::zero();
        NanoSeconds inlineX  = NanoSeconds::zero();
        NanoSeconds download = NanoSeconds::zero();
        NanoSeconds upload   = NanoSeconds::zero();
    };
}

class CudaThresher : public IThresher
{
    GreenReaperConfig _config;
    int               _deviceId;

    bool      _isDecompressing = false;             // Are we currently decompressing a proof?
    TableId   _currentTable    = TableId::Table1;   // Current table being decompressed
    
    uint32    _maxCompressionLevel = 0; // Max compression level for which we have allocated buffers

    size_t    _bufferCapacity = 0;
    size_t    _matchCapacity  = 0;
    size_t    _sortBufferSize = 0;

    uint32    _tableOffsetIn    = 0;  // Offset in the input table values. That is, the read position
    uint32    _tableOffsetOut   = 0;  // Offset in the output table. That is, how many entries/pairs have been generated so far.

    // Host buffers
    uint32*   _hostMatchCount   = nullptr;
    
    // Device buffers
    byte*     _devSortTmpBuffer = nullptr;
    uint32*   _devChaChaInput   = nullptr;  // Modified plot id for chacha seed

    // F1
    uint64*   _devYBufferF1     = nullptr;
    uint32*   _devXBuffer       = nullptr;
    uint32*   _devXBufferTmp    = nullptr;

    uint32*   _devSortKey       = nullptr;
    uint32*   _devSortKeyTmp    = nullptr;

    // Fx
    Pair*     _devMatchesIn     = nullptr;
    Pair*     _devMatchesOut    = nullptr;
    uint64*   _devYBufferIn     = nullptr;
    uint64*   _devYBufferOut    = nullptr;
    byte*     _devMetaBufferIn  = nullptr;
    byte*     _devMetaBufferOut = nullptr;

    uint32*   _devMatchCount    = nullptr;

    // Temporary sorted buffers
    // Pair*   _devMatchesSorted = nullptr;
    // uint64* _devYSorted       = nullptr;
    // byte*   _devMetaSorted    = nullptr;


    // Cuda objects
    cudaStream_t _computeStream  = nullptr;
    cudaStream_t _uploadStream   = nullptr;
    cudaStream_t _downloadStream = nullptr;

    cudaEvent_t  _computeEvent   = nullptr;
    cudaEvent_t  _uploadEvent    = nullptr;
    cudaEvent_t  _downloadEvent  = nullptr;

    CudaPlotInfo _info;

    Timings      _timings = {};

public:
    CudaThresher( const GreenReaperConfig& config, int deviceId )
        : _config  ( config )
        , _deviceId( deviceId )
    {}

    virtual ~CudaThresher()
    {
        ReleaseBuffers();
    }

    bool AllocateBuffers( const uint k, uint maxCompressionLevel ) override
    {
        // Only support C7 max for now
        if( maxCompressionLevel > 16 )
            return false;

        // #NOTE: For now we always preallocate for the maximum compression level
        maxCompressionLevel = 16;

        if( _maxCompressionLevel >= maxCompressionLevel )
            return true;
    
        _info.k              = 32;
        _info.bucketCount    = 64;                          // #TODO: Make this configurable
        _info.yBits          = _info.k + kExtraBits;
        _info.bucketBits     = bblog2( _info.bucketCount );

        // #TODO: Needs to be configured per k
        //const uint64 kTableEntryCount  = 1ull << k;
        const uint32 entriesPerF1Block   = kF1BlockSizeBits / k;

        const uint64 allocEntryCountBase = GetEntriesPerBucketForCompressionLevel( k, maxCompressionLevel );
        const uint64 bucketCapcity       = (allocEntryCountBase / _info.bucketCount + ( 4096 )) * 2;
        const uint64 allocEntryCount     = RoundUpToNextBoundary( bucketCapcity * _info.bucketCount, entriesPerF1Block );

        _bufferCapacity = allocEntryCount;
        //const uint64 kSliceEntryCount  = kBucketEntryCount / _info.bucketCount;
        //const uint64 kSliceCapcity     = kSliceEntryCount + _info.bucketCount;

        _info.bucketCapacity   = (uint32)_bufferCapacity;
        _info.sliceCapacity    = (uint32)bucketCapcity;
        _info.metaMaxSizeBytes = _info.k * 4 / 8;

        ASSERT( _info.sliceCapacity * _info.bucketCount == _info.bucketCapacity );


        ////    cuda.info.sliceCapacity     = cuda.info.bucketCapacity / cuda.info.bucketCount;
        ////    cuda.info.metaMaxSizeBytes;
        //    // context.cuda.bufferCapacity =

        // Allocate CUDA buffers
        cudaError cErr = cudaSuccess;
        {
            #define CuFailCheck()  if( cErr != cudaSuccess ) goto FAIL

            /// Host pinned allocations
            cErr = cudaMallocHost( &_hostMatchCount, sizeof( uint32 ), cudaHostAllocDefault ); CuFailCheck();
            // cErr = cudaMallocHost( &_hostBucketCounts, _info.bucketCount * sizeof( uint32 ), cudaHostAllocDefault ); CuFailCheck();

            /// Cuda allocations
            _sortBufferSize = 0;
            cErr = cub::DeviceRadixSort::SortPairs<uint64, uint32>( nullptr, _sortBufferSize, nullptr, nullptr, nullptr, nullptr, allocEntryCount ); CuFailCheck();
            ASSERT( _sortBufferSize );

            cErr = cudaMalloc( &_devSortTmpBuffer, _sortBufferSize ); CuFailCheck();
            // cErr = cudaMalloc( &_devBucketCounts , _info.bucketCount * sizeof( uint32 ) ); CuFailCheck();
            cErr = CudaCallocT( _devChaChaInput, 32 ); CuFailCheck();

            cErr = CudaCallocT( _devYBufferF1 , allocEntryCount ); CuFailCheck();
            cErr = CudaCallocT( _devYBufferIn , allocEntryCount ); CuFailCheck();
            cErr = CudaCallocT( _devYBufferOut, allocEntryCount ); CuFailCheck();
            cErr = CudaCallocT( _devXBuffer   , allocEntryCount ); CuFailCheck();
            cErr = CudaCallocT( _devXBufferTmp, allocEntryCount ); CuFailCheck();


            const uint64 maxPairsPerTable = std::max( (uint64)GR_MIN_TABLE_PAIRS, GetMaxTablePairsForCompressionLevel( k, maxCompressionLevel ) );
            _matchCapacity = (size_t)maxPairsPerTable;

            cErr = CudaCallocT( _devMatchCount, 1 ); CuFailCheck();

            cErr = CudaCallocT( _devMatchesIn,  maxPairsPerTable ); CuFailCheck();
            cErr = CudaCallocT( _devMatchesOut, maxPairsPerTable ); CuFailCheck();

            cErr = CudaCallocT( _devMetaBufferIn , maxPairsPerTable * sizeof( uint32 ) * 4 ); CuFailCheck();
            cErr = CudaCallocT( _devMetaBufferOut, maxPairsPerTable * sizeof( uint32 ) * 4 ); CuFailCheck();

            cErr = CudaCallocT( _devSortKey   , maxPairsPerTable ); CuFailCheck();
            cErr = CudaCallocT( _devSortKeyTmp, maxPairsPerTable ); CuFailCheck();
            

            // Sorted temp buffers
            // cErr = cudaMalloc( &_devMatchesSorted, maxPairsPerTable * sizeof( Pair ) ); CuFailCheck();
            // cErr = cudaMalloc( &_devYSorted      , maxPairsPerTable * sizeof( uint64 ) ); CuFailCheck();
            // cErr = cudaMalloc( &_devMetaSorted   , maxPairsPerTable * sizeof( uint32 )*4 ); CuFailCheck();

            // CUDA objects
            cErr = cudaStreamCreate( &_computeStream  ); CuFailCheck();
            cErr = cudaStreamCreate( &_uploadStream   ); CuFailCheck();
            cErr = cudaStreamCreate( &_downloadStream ); CuFailCheck();

            cErr = cudaEventCreate( &_computeEvent  ); CuFailCheck();
            cErr = cudaEventCreate( &_uploadEvent   ); CuFailCheck();
            cErr = cudaEventCreate( &_downloadEvent ); CuFailCheck();

            #undef CuFailCheck
        }

        //cErr = cudaMalloc( &cuda.devYBufferF1, sizeof( uint32 ) * allocEntryCount );
        _maxCompressionLevel = maxCompressionLevel;
        return true;

    FAIL:
        ReleaseBuffers();
        return false;
    }

    void ReleaseBuffers() override
    {
        _bufferCapacity      = 0;
        _maxCompressionLevel = 0;

        // Release all buffers
        CudaSafeFreeHost( _hostMatchCount );

        CudaSafeFree( _devSortTmpBuffer );
        CudaSafeFree( _devChaChaInput );

        CudaSafeFree( _devYBufferF1  );
        CudaSafeFree( _devYBufferIn  );
        CudaSafeFree( _devYBufferOut );
        CudaSafeFree( _devXBuffer    );
        CudaSafeFree( _devXBufferTmp );

        CudaSafeFree( _devMatchCount );
        CudaSafeFree( _devMatchesIn );
        CudaSafeFree( _devMatchesOut );

        CudaSafeFree( _devMetaBufferIn );
        CudaSafeFree( _devMetaBufferOut );

        CudaSafeFree( _devSortKey    );
        CudaSafeFree( _devSortKeyTmp );

        // CUDA objects
        if( _computeStream ) cudaStreamDestroy( _computeStream ); _computeStream = nullptr;
        if( _uploadStream )  cudaStreamDestroy( _uploadStream );  _uploadStream  = nullptr;
        if( _computeStream ) cudaStreamDestroy( _computeStream ); _computeStream = nullptr;

        if( _computeEvent )  cudaEventDestroy( _computeEvent );  _computeEvent  = nullptr;
        if( _uploadEvent )   cudaEventDestroy( _uploadEvent );   _uploadEvent   = nullptr;
        if( _downloadEvent ) cudaEventDestroy( _downloadEvent ); _downloadEvent = nullptr;
    }

    ThresherResult DecompressInitialTable( 
        GreenReaperContext& cx,
        const byte   plotId[32],
        const uint32 entryCountPerX,
        Pair*        outPairs,
        uint64*      outY,
        void*        outMeta,
        uint32&      outMatchCount,
        const uint64 x0, const uint64 x1 ) override
    {
        // Only k32 for now
        ASSERT( x0 <= 0xFFFFFFFF );
        ASSERT( x1 <= 0xFFFFFFFF );
        ASSERT( entryCountPerX*2 < _bufferCapacity );

        ThresherResult result{};
        result.kind = ThresherResultKind::Success; 

        if( entryCountPerX*2 > _bufferCapacity )
        {
            result.kind  = ThresherResultKind::Error;
            result.error = ThresherError::UnexpectedError;
            return result;
        }

        uint64    table1EntryCount = 0;
        cudaError cErr             = cudaSuccess;

        // Ensure we're in a good state
        cErr = cudaStreamSynchronize( _computeStream ); if( cErr != cudaSuccess ) goto FAIL;
        cErr = cudaStreamSynchronize( _downloadStream ); if( cErr  != cudaSuccess ) goto FAIL;


        {
            byte key[32] = { 1 };
            memcpy( key + 1, plotId, 32 - 1 );

            chacha8_ctx chacha;
            chacha8_keysetup( &chacha, key, 256, nullptr );

            const uint32 f1EntriesPerBlock = kF1BlockSize / sizeof( uint32 );

            const uint32 x0Start     = (uint32)(x0 * entryCountPerX);
            const uint32 x1Start     = (uint32)(x1 * entryCountPerX);
            const uint32 block0Start = x0Start / f1EntriesPerBlock;
            const uint32 block1Start = x1Start / f1EntriesPerBlock;

            const uint32 f1BlocksPerX = (uint32)(entryCountPerX * sizeof( uint32 ) / kF1BlockSize);
            ASSERT( entryCountPerX == f1BlocksPerX * (kF1BlockSize / sizeof( uint32 ) ) );

            uint32 f1BlocksToCompute = f1BlocksPerX;
            uint32 f1Iterations      = 2;
            uint32 f1BlockStart      = block0Start;

            // #TODO: Re-enable 
            // If we overlap chacha blocks, then  only calculate 1 range
            {
                const uint32 blockMin = std::min( block0Start, block1Start );
                const uint32 blockMax = std::max( block0Start, block1Start );

                if( blockMin + f1BlocksPerX >= blockMax )
                {
                    f1BlocksToCompute = blockMax - blockMin + f1BlocksPerX;
                    f1Iterations      = 1;
                    f1BlockStart      = blockMin;
                }
            }

            // Setup initial data
            {
                #if BB_CUDA_HARVEST_USE_TIMINGS
                    const auto timer = TimerBegin();
                #endif

                uint64* f1Y = _devYBufferF1;
                uint32* f1X = _devXBufferTmp;

                cErr = cudaMemcpyAsync( _devChaChaInput, chacha.input, 64, cudaMemcpyHostToDevice, _computeStream );
                if( cErr != cudaSuccess ) goto FAIL;

                for( uint32 i = 0; i < f1Iterations; i++ )
                {
                    CudaGenF1K32(
                        _info,
                        _devChaChaInput,    // Seed
                        f1BlockStart,       // Starting chacha block 
                        f1BlocksToCompute,  // How many chacha blocks to compute
                        f1Y,
                        f1X,
                        _computeStream
                    );

                    f1Y += entryCountPerX;
                    f1X += entryCountPerX;
                    f1BlockStart = block1Start;
                }

                cErr = cudaStreamSynchronize( _computeStream );
                if( cErr != cudaSuccess ) goto FAIL;

                #if BB_CUDA_HARVEST_USE_TIMINGS
                    _timings.f1 += TimerEndTicks( timer );
                #endif
            }



            // Sort entries on Y
            {
                #if BB_CUDA_HARVEST_USE_TIMINGS
                    const auto timer = TimerBegin();
                #endif

                const uint64 entriesPerChaChaBlock = kF1BlockSize / sizeof( uint32 );
                const uint64 f1EntryCount          = f1BlocksToCompute * entriesPerChaChaBlock * f1Iterations;
                table1EntryCount = f1EntryCount;

                cErr = cub::DeviceRadixSort::SortPairs<uint64, uint32>(
                    _devSortTmpBuffer, _sortBufferSize,
                    _devYBufferF1,  _devYBufferIn,
                    _devXBufferTmp, _devXBuffer,
                    f1EntryCount, 0, _info.k+kExtraBits,
                    _computeStream );
                if( cErr != cudaSuccess ) goto FAIL;

                cErr = cudaStreamSynchronize( _computeStream );
                if( cErr != cudaSuccess ) goto FAIL;

                #if BB_CUDA_HARVEST_USE_TIMINGS
                    _timings.sort += TimerEndTicks( timer );
                #endif
            }
        }

        // Perform T2 matches
        {
            #if BB_CUDA_HARVEST_USE_TIMINGS
                auto timer = TimerBegin();
            #endif

            cErr = CudaHarvestMatchK32(
                    _devMatchesOut,
                    _devMatchCount,
                    (uint32)_bufferCapacity,
                    _devYBufferIn,
                    (uint32)table1EntryCount,
                    0,
                    _computeStream );
            if( cErr != cudaSuccess ) goto FAIL;

            // Get match count
            cErr = cudaMemcpyAsync( _hostMatchCount, _devMatchCount, sizeof( uint32 ), cudaMemcpyDeviceToHost, _computeStream );
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaStreamSynchronize( _computeStream );
            if( cErr != cudaSuccess ) goto FAIL;

            const uint32 matchCount = *_hostMatchCount;

            #if BB_CUDA_HARVEST_USE_TIMINGS
                _timings.match += TimerEndTicks( timer );
                timer = TimerBegin();
            #endif

            if( matchCount < 1 )
            {
                result.kind = ThresherResultKind::NoMatches;
                return result;
            }

            // Compute table 2 Fx
            CudaFxHarvestK32( 
                TableId::Table2,
                _devYBufferOut,
                _devMetaBufferOut,
                matchCount,
                _devMatchesOut,
                _devYBufferIn,
                _devXBuffer,
                _computeStream );

            #if BB_CUDA_HARVEST_USE_TIMINGS
                cErr = cudaStreamSynchronize( _computeStream );
                _timings.fx += TimerEndTicks( timer );
                timer = TimerBegin();
            #endif

            // Inline x's into pairs
            CudaK32InlineXsIntoPairs(
                matchCount,
                _devMatchesOut,
                _devMatchesOut,
                _devXBuffer,
                _computeStream );

            #if BB_CUDA_HARVEST_USE_TIMINGS
                cErr = cudaStreamSynchronize( _computeStream );
                _timings.inlineX += TimerEndTicks( timer );
                timer = TimerBegin();
            #endif

            // Sync download stream w/ compute stream
            // #TODO: Use pinned
            const size_t metaSize = CDiv( _info.k * 2, 8 );

            /// Copy new entries back to host
            cErr = cudaEventRecord( _computeEvent, _computeStream );    // Signal from compute stream
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaStreamWaitEvent( _downloadStream, _computeEvent ); // Download stream sync w/ compute stream
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaMemcpyAsync( outPairs, _devMatchesOut, sizeof( Pair ) * matchCount, cudaMemcpyDeviceToHost, _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;
            
            cErr = cudaMemcpyAsync( outY, _devYBufferOut, sizeof( uint64 ) * matchCount, cudaMemcpyDeviceToHost, _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaMemcpyAsync( outMeta, _devMetaBufferOut, metaSize * matchCount, cudaMemcpyDeviceToHost, _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaStreamSynchronize( _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;

            outMatchCount = matchCount;

            #if BB_CUDA_HARVEST_USE_TIMINGS
                _timings.download += TimerEndTicks( timer );
            #endif

            if( matchCount < 1 )
            {
                result.kind = ThresherResultKind::NoMatches;
                return result;
            }
        }

        return result;

    FAIL:
// Log::Line( "DecompressInitialTable() Failed with CUDA error '%s': %s", cudaGetErrorName( cErr ), cudaGetErrorString( cErr ) );
        ASSERT( cErr == cudaSuccess );              // Force debugger break

        result.kind          = ThresherResultKind::Error;
        result.error         = ThresherError::CudaError;
        result.internalError = (i32)cErr;

        return result;
    }

    ThresherResult DecompressTableGroup(
        GreenReaperContext& cx,
        const TableId   table,
        uint32          entryCount,
        uint32          matchOffset,
        uint32          maxPairs,
        uint32&         outMatchCount,
        Pair*           outPairs,
        uint64*         outY,
        void*           outMeta,
        Pair*           outLPairs,
        const Pair*     inLPairs,
        const uint64*   inY,
        const void*     inMeta ) override
    {
        ASSERT( maxPairs );

        outMatchCount = 0;

        ThresherResult result{};
        result.kind = ThresherResultKind::Success; 

        cudaError_t cErr = cudaSuccess;

        const size_t inMetaMultiplier = GetTableMetaMultiplier( table - 1 );
        const size_t inMetaByteSize   = CDiv( _info.k * inMetaMultiplier, 8 );
        uint32 matchCount = 0;

        // Ensure we're in a good state
        cErr = cudaStreamSynchronize( _uploadStream ); if( cErr != cudaSuccess ) goto FAIL;
        cErr = cudaStreamSynchronize( _computeStream ); if( cErr != cudaSuccess ) goto FAIL;
        cErr = cudaStreamSynchronize( _downloadStream ); if( cErr != cudaSuccess ) goto FAIL;

        /// Upload input data
    #if BB_CUDA_HARVEST_USE_TIMINGS
        auto timer = TimerBegin();
    #endif

        cErr = cudaMemcpyAsync( _devMatchesIn, inLPairs, sizeof( Pair ) * entryCount, cudaMemcpyHostToDevice, _uploadStream ); if( cErr != cudaSuccess ) goto FAIL;
        cErr = cudaMemcpyAsync( _devYBufferOut, inY, sizeof( uint64 ) * entryCount, cudaMemcpyHostToDevice, _uploadStream ); if( cErr != cudaSuccess ) goto FAIL;
        cErr = cudaMemcpyAsync( _devMetaBufferOut, inMeta, inMetaByteSize * entryCount, cudaMemcpyHostToDevice, _uploadStream ); if( cErr != cudaSuccess ) goto FAIL;

        cErr = cudaEventRecord( _uploadEvent, _uploadStream );
        if( cErr != cudaSuccess ) goto FAIL;

        // Sync w/ upload stream
        cErr = cudaStreamWaitEvent( _computeStream, _uploadEvent );
        if( cErr != cudaSuccess ) goto FAIL;

        #if BB_CUDA_HARVEST_USE_TIMINGS
            cErr = cudaStreamSynchronize( _computeStream ); if( cErr != cudaSuccess ) goto FAIL;
            _timings.upload += TimerEndTicks( timer );
            timer = TimerBegin();
        #endif

        /// Sort on Y
        SortEntriesOnY( table-1,
            _devYBufferIn,
            _devMatchesOut,
            _devMetaBufferIn,
            _devYBufferOut,
            _devMatchesIn,
            _devMetaBufferOut,
            entryCount,
            _computeStream );

        #if BB_CUDA_HARVEST_USE_TIMINGS
            cErr = cudaStreamSynchronize( _computeStream ); if( cErr != cudaSuccess ) goto FAIL;
            _timings.sort += TimerEndTicks( timer );
            timer = TimerBegin();
        #endif

        // Sync download stream w/ compute stream
        cErr = cudaEventRecord( _computeEvent, _computeStream );
        if( cErr != cudaSuccess ) goto FAIL;
        cErr = cudaStreamWaitEvent( _downloadStream, _computeEvent );
        if( cErr != cudaSuccess ) goto FAIL;

        // Copy sorted input matches back to device
        cErr = cudaMemcpyAsync( outLPairs, _devMatchesOut, sizeof( Pair ) * entryCount, cudaMemcpyDeviceToHost, _downloadStream );
        if( cErr != cudaSuccess ) goto FAIL;

        /// Perform new matches w/ sorted input Y
        ASSERT( maxPairs <= _matchCapacity );

        cErr = CudaHarvestMatchK32(
                    _devMatchesOut,
                    _devMatchCount,
                    maxPairs,
                    _devYBufferIn,
                    entryCount,
                    0,
                    _computeStream );
        if( cErr != cudaSuccess ) goto FAIL;

        // Get match count
        cErr = cudaMemcpyAsync( _hostMatchCount, _devMatchCount, sizeof( uint32 ), cudaMemcpyDeviceToHost, _computeStream );
        if( cErr != cudaSuccess ) goto FAIL;

        cErr = cudaStreamSynchronize( _computeStream );
        if( cErr != cudaSuccess ) goto FAIL;

        matchCount = *_hostMatchCount;
        outMatchCount = matchCount;

        #if BB_CUDA_HARVEST_USE_TIMINGS
            _timings.match += TimerEndTicks( timer );
            timer = TimerBegin();
        #endif

        if( matchCount < 1 )
        {
// Log::Line( "CUDA: No matches!" );
            result.kind = ThresherResultKind::NoMatches;
            goto FAIL;
        }

        // Generate Fx
        CudaFxHarvestK32( 
            table,
            _devYBufferOut,
            _devMetaBufferOut,
            matchCount,
            _devMatchesOut,
            _devYBufferIn,
            _devMetaBufferIn,
            _computeStream );

        // Apply matchOffset
        CudaK32ApplyPairOffset(
            matchCount,
            matchOffset,
            _devMatchesOut,
            _devMatchesOut,
            _computeStream );

        cErr = cudaEventRecord( _computeEvent, _computeStream );    // Signal from compute stream
        if( cErr != cudaSuccess ) goto FAIL;

        #if BB_CUDA_HARVEST_USE_TIMINGS
            cErr = cudaStreamSynchronize( _computeStream ); if( cErr != cudaSuccess ) goto FAIL;
            _timings.fx += TimerEndTicks( timer );
            timer = TimerBegin();
        #endif

        /// Copy new, unsorted entries back to host
        {
            const size_t outMetaMultiplier = GetTableMetaMultiplier( table );
            const size_t outMetaByteSize    = CDiv( _info.k * outMetaMultiplier, 8 );
            ASSERT( outMetaByteSize );

            // Download stream sync w/ compute stream
            cErr = cudaStreamWaitEvent( _downloadStream, _computeEvent ); 
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaMemcpyAsync( outPairs, _devMatchesOut, sizeof( Pair ) * matchCount, cudaMemcpyDeviceToHost, _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;
            
            cErr = cudaMemcpyAsync( outY, _devYBufferOut, sizeof( uint64 ) * matchCount, cudaMemcpyDeviceToHost, _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaMemcpyAsync( outMeta, _devMetaBufferOut, outMetaByteSize * matchCount, cudaMemcpyDeviceToHost, _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;

            cErr = cudaStreamSynchronize( _downloadStream );
            if( cErr != cudaSuccess ) goto FAIL;
        }

        #if BB_CUDA_HARVEST_USE_TIMINGS    
            _timings.download += TimerEndTicks( timer );
        #endif

        outMatchCount = matchCount;
        return result;

    FAIL:
// Log::Line( "DecompressTableGroup() Failed with CUDA error '%s': %s", cudaGetErrorName( cErr ), cudaGetErrorString( cErr ) );

        ASSERT( cErr == cudaSuccess );  // Force debugger break

        if( result.kind == ThresherResultKind::Success )
        {
            result.kind          = ThresherResultKind::Error;
            result.error         = ThresherError::CudaError;
            result.internalError = (i32)cErr;
        }

        return result;
    }

    cudaError_t SortEntriesOnY( 
        const TableId table,
              uint64* yOut,
              Pair*   pairsOut,
              void*   metaOut,
        const uint64* yIn,
        const Pair*   pairsIn,
        const void*   metaIn ,
        const uint32  entryCount,
        cudaStream_t  stream )
    {
        cudaError_t cErr;

        uint32* sortKeyIn = _devSortKeyTmp;
        uint32* sortKey   = _devSortKey;

        // Generate sort key (for meta and pairs)
        CudaK32PlotGenSortKey( entryCount, sortKeyIn, stream );

        // Sort y, with the sort key
        cErr = cub::DeviceRadixSort::SortPairs<uint64, uint32>(
                _devSortTmpBuffer, _sortBufferSize,
                yIn, yOut, 
                sortKeyIn, sortKey, 
                entryCount, 0, _info.k+kExtraBits,
                stream );

        if( cErr != cudaSuccess ) return cErr;

        // Sort matches on key
        CudaK32PlotSortByKey( entryCount, sortKey, pairsIn, pairsOut, stream );

        // Sort meta on key
        const size_t metaMultiplier = GetTableMetaMultiplier( table );
        const size_t metaByteSize   = CDiv( _info.k * metaMultiplier, 8 );
        ASSERT( metaMultiplier > 0 );

        switch( metaMultiplier )
        {
            case 2: CudaK32PlotSortByKey( entryCount, sortKey, (K32Meta2*)metaIn, (K32Meta2*)metaOut, stream ); break;
            case 3: CudaK32PlotSortByKey( entryCount, sortKey, (K32Meta3*)metaIn, (K32Meta3*)metaOut, stream ); break;
            case 4: CudaK32PlotSortByKey( entryCount, sortKey, (K32Meta4*)metaIn, (K32Meta4*)metaOut, stream ); break;
            default: ASSERT( 0 ); break;
        }

        return cErr;
    }

    void DumpTimings() override
    {
        #if BB_CUDA_HARVEST_USE_TIMINGS
            auto logTiming = []( const char* title, NanoSeconds ns ) {

                Log::Line( "%8s: %.3lf", title, TicksToSeconds( ns ) );
            };

            logTiming( "F1"      , _timings.f1       );
            logTiming( "Sort"    , _timings.sort     );
            logTiming( "Fx"      , _timings.fx       );
            logTiming( "Match"   , _timings.match    );
            logTiming( "Inline X", _timings.inlineX  );
            logTiming( "Download", _timings.download );
            logTiming( "Upload"  , _timings.upload   );

            ClearTimings();
        #endif
    }

    void ClearTimings() override
    {
        _timings = {};
    }
};


IThresher* CudaThresherFactory_Private( const GreenReaperConfig& config )
{
    ASSERT( config.gpuRequest != GRGpuRequestKind_None );

    // Attempt to init CUDA first
    cudaError cErr;

    int deviceCount = 0;
    cErr = cudaGetDeviceCount( &deviceCount );
    if( cErr != cudaSuccess || deviceCount < 1 )
        return nullptr;

    int deviceId = (int)config.gpuDeviceIndex;
    if( config.gpuDeviceIndex >= (uint)deviceCount )
    {
        // Match exact device?
        if( config.gpuRequest == GRGpuRequestKind_ExactDevice )
            return nullptr;
        
        deviceId = 0;
    }

    cErr = cudaSetDevice( deviceId );
    if( cErr != cudaSuccess )
    {
        // Try the default device then, if requested
        if( deviceId != 0 && config.gpuRequest == GRGpuRequestKind_FirstAvailable )
        {
            deviceId = 0;
            cErr = cudaSetDevice( deviceId );
            if( cErr != cudaSuccess )
                return nullptr;
        }
        else
            return nullptr;
    }

    auto* thresher = new CudaThresher( config, deviceId );
    return thresher;
}
