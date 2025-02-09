#include "GreenReaper.h"
#include "GreenReaperInternal.h"
#include "harvesting/Thresher.h"
#include "threading/ThreadPool.h"
#include "threading/GenJob.h"
#include "plotting/Tables.h"
#include "tools/PlotReader.h"
#include "plotmem/LPGen.h"
#include "ChiaConsts.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"
#include "algorithm/RadixSort.h"
#include "util/jobs/SortKeyJob.h"
#include "util/BitView.h"
#include "util/VirtualAllocator.h"
#include "plotting/matching/GroupScan.h"
#include <mutex>
#include <algorithm>


static std::mutex _lTargetLock;

// Internal types
struct ProofContext
{
    uint32    leftLength;
    uint32    rightLength;

    uint64*   yLeft;        // L table Y
    uint64*   yRight;       // R table Y
    K32Meta4* metaLeft;     // L table metadata
    K32Meta4* metaRight;    // R table metadata

    uint64*   proof;
};

struct ProofTable
{
    Pair*  _pairs    = nullptr;
    uint64 _capacity = 0;
    uint64 _length   = 0;

    struct {
        uint32 count;
        uint32 offset;
    } _groups[16] = {};

    inline Span<Pair> GetGroupPairs( uint32 groupIdx ) const
    {
        ASSERT( groupIdx < 16 );
        
        const uint32 offset = _groups[groupIdx].offset;
        const uint32 count  = _groups[groupIdx].count;
        ASSERT( offset + count <= _length );

        return MakeSpan( _pairs + offset,  count );
    }

    inline Span<Pair> GetUsedTablePairs() const
    {
        return MakeSpan( _pairs, _length );
    }

    inline Span<Pair> GetFreeTablePairs() const
    {
        return MakeSpan( _pairs+_length, _capacity - _length );
    }

    inline Pair& PushGroupPair( uint32 groupIdx )
    {
        ASSERT( _length < _capacity );

        _groups[groupIdx].count++;

        return _pairs[_length++];
    }

    inline void BeginGroup( uint32 groupIdx )
    {
        ASSERT( groupIdx < 16 );
        _groups[groupIdx].count = 0;

        if( groupIdx > 0 )
            _groups[groupIdx].offset = _groups[groupIdx-1].offset + _groups[groupIdx-1].count;
        else
            _length = 0;
    }

    inline void AddGroupPairs( uint32 groupIdx, uint32 pairCount )
    {
        ASSERT( groupIdx < 16 );

        _groups[groupIdx].count += pairCount;
        _length += pairCount;
    }

    template<TableId lTable>
    void GetLTableGroupEntries( GreenReaperContext& cx, const uint32 group, Span<uint64>& outY, Span<typename K32MetaType<lTable>::Out>& outMeta );

    // inline void FinishGroup( uint32 groupIdx )
    // {
    //     ASSERT( groupIdx < 16 );

    //     if( groupIdx < 15 )
    //         _groups[groupIdx+1].offset = _groups[groupIdx].offset + _groups[groupIdx].count;
    // }
};

// Used for qualities fetch. Line point and index.
// Represents a line point and its original, pre-sort index.
struct LPIndex { uint64 lp; uint32 index; };

struct GreenReaperContext
{
    enum State
    {
        None      = 0,  // Awaiting a proof request or qualities request
        Qualities = 1,  // Currently fetching qualities
    };

    GreenReaperConfig config;

    State          state          = None;
    ThreadPool*    pool           = nullptr;
    size_t         allocationSize = 0;

    uint64         maxEntriesPerBucket;
    uint32         maxCompressionLevelReserved = 0; // Compression level for which we have reserved memory

    Span<uint64>   yBufferF1;   // F1 tmp
    Span<uint64>   yBuffer;     // F1/In/Out
    Span<uint64>   yBufferTmp;  // In/Out

    Span<uint32>   xBuffer;
    Span<uint32>   xBufferTmp;

    Span<uint32>   sortKey;             // #TODO: Remove this, can use x for it

    Span<K32Meta4> metaBuffer;
    Span<K32Meta4> metaBufferTmp;
    Span<Pair>     pairs;
    Span<Pair>     pairsTmp;
    Span<uint32>   groupsBoundaries;    // BC group boundaries

    ProofTable     tables[7]  = {};

    ProofContext   proofContext;

    Pair           backTraceTables[6][32] = {};
    LPIndex        lpTables       [6][32] = {};

    IThresher*     cudaThresher         = nullptr;
    bool           cudaRecreateThresher = false;    // In case a CUDA error occurred or the device was lost,
                                                    // we need to re-create it.
};

enum class ForwardPropResult
{
    Failed   = 0,
    Success  = 1,
    Continue = 2
};


template<TableId lTable>
inline void ProofTable::GetLTableGroupEntries( GreenReaperContext& cx, const uint32 group, Span<uint64>& outY, Span<typename K32MetaType<lTable>::Out>& outMeta )
{
    using TMeta = typename K32MetaType<lTable>::Out;

    ASSERT( group < 16 );
    outY    = MakeSpan( cx.proofContext.yLeft + _groups[group].offset, _groups[group].count );
    outMeta = MakeSpan( (TMeta*)cx.proofContext.metaLeft + _groups[group].offset, _groups[group].count );
}

struct Table1BucketContext
{
    GreenReaperContext* cx;
    const byte*    plotId;
    uint64         entriesPerBucket;

    // Mutated after each bucket to point to the next
    // free section of the buffers.
    Span<uint64>   outY;
    Span<K32Meta2> outMeta;
    Span<Pair>     outPairs;
};


/// Internal functions
static GRResult RequestSetup( GreenReaperContext* cx, const uint32 k, const uint32 compressionLevel );

static void SortQualityXs( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64* xs, const uint32 count );

static GRResult ProcessTable1Bucket( Table1BucketContext& tcx, const uint64 x1, const uint64 x2, const uint32 groupIndex );
static GRResult ProcessTable1BucketCPU( Table1BucketContext& tcx, const uint64 x1, const uint64 x2, const uint32 groupIndex );

static void FreeBucketBuffers( GreenReaperContext& cx );
static bool ReserveBucketBuffers( GreenReaperContext& cx, uint32 k, uint32 compressionLevel );

static void GenerateF1( GreenReaperContext& cx, const byte plotId[32], const uint64 bucketEntryCount, const uint32 x0, const uint32 x1 );
static Span<Pair> Match( GreenReaperContext& cx, const Span<uint64> yEntries, Span<Pair> outPairs, const uint32 pairOffset  );

template<TableId rTable>
ForwardPropResult ForwardPropTable( GreenReaperContext& cx, uint32 tableGroupCount, bool returnSuccessOnSingleMatch );

static void FlipBuffers( GreenReaperContext& cx, const TableId rTable );
template<TableId rTable>
static void SortTableAndFlipBuffers( GreenReaperContext& cx, const uint32 groupCount );

template<TableId rTable, typename TMetaIn, typename TMetaOut>
static void GenerateFxForPairs( GreenReaperContext& cx, const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> outMeta );

template<TableId rTable, typename TMetaIn, typename TMetaOut>
static void GenerateFx( const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> outMeta );

static bool ForwardPropTables( GreenReaperContext& cx );
static void BacktraceProof( GreenReaperContext& cx, const TableId tableStart, uint64 proof[GR_POST_PROOF_X_COUNT] );


///
/// Public API
///
//-----------------------------------------------------------
GRResult grPopulateApi( GRApi* api, const size_t apiStructSize, const int apiVersion )
{
    if( api == nullptr )
        return GRResult_Failed;

    if( apiVersion != GR_API_VERSION )
        return GRResult_WrongVersion;

    if( apiStructSize != sizeof( GRApi ) )
        return GRResult_WrongVersion;

    api->CreateContext                  = &grCreateContext;
    api->DestroyContext                 = &grDestroyContext;
    api->PreallocateForCompressionLevel = &grPreallocateForCompressionLevel;
    api->FetchProofForChallenge         = &grFetchProofForChallenge;
    api->GetFetchQualitiesXPair         = &grGetFetchQualitiesXPair;
    api->GetMemoryUsage                 = &grGetMemoryUsage;
    api->HasGpuDecompressor             = &grHasGpuDecompressor;
    api->GetCompressionInfo             = &grGetCompressionInfo;

    return GRResult_OK;
}

//-----------------------------------------------------------
GRResult grCreateContext( GreenReaperContext** outContext, 
                          GreenReaperConfig* config,
                          const size_t configStructSize )
{
    if( outContext == nullptr )
        return GRResult_Failed;

    auto* context = new GreenReaperContext{};
    if( context == nullptr )
        return GRResult_OutOfMemory;

    if( config )
    {
        if( configStructSize != sizeof( GreenReaperConfig ) || config->apiVersion != GR_API_VERSION )
            return GRResult_WrongVersion;

        context->config = *config;
    }
    else
    {
        context->config.threadCount = std::min( 2u, SysHost::GetLogicalCPUCount() );
    }

    context->pool = new ThreadPool( context->config.threadCount, ThreadPool::Mode::Fixed,
                                    (bool)config->disableCpuAffinity, 
                                    (bool)config->disableCpuAffinity ? 0 : config->cpuOffset );
    if( !context->pool )
    {
        grDestroyContext( context );
        return GRResult_OutOfMemory;
    }

    if( config->gpuRequest != GRGpuRequestKind_None )
    {
        context->cudaThresher = CudaThresherFactory::Create( *config );
        if( context->cudaThresher == nullptr && config->gpuRequest == GRGpuRequestKind_ExactDevice )
        {
            grDestroyContext( context );
            return GRResult_InvalidGPU;
        }
    }

    *outContext = context;
    return GRResult_OK;
}

//-----------------------------------------------------------
void grDestroyContext( GreenReaperContext* context )
{
    if( context == nullptr )
        return;

    FreeBucketBuffers( *context );

    if( context->pool )
        delete context->pool;

    delete context;
}

//-----------------------------------------------------------
GRResult grPreallocateForCompressionLevel( GreenReaperContext* context, const uint32_t k, const uint32_t maxCompressionLevel )
{
    if( context == nullptr )
        return GRResult_Failed;

    if( k != 32 )
        return GRResult_Failed;

    // Ensure our buffers have enough for the specified entry bit count
    if( !ReserveBucketBuffers( *context, k, maxCompressionLevel ) )
        return GRResult_OutOfMemory;

    return GRResult_OK;
}

//-----------------------------------------------------------
size_t grGetMemoryUsage( GreenReaperContext* context )
{
    if( !context )
        return 0;

    return context->allocationSize;
}

//-----------------------------------------------------------
GRBool grHasGpuDecompressor( GreenReaperContext* context )
{
    return (context != nullptr && context->cudaThresher != nullptr ) ? GR_TRUE : GR_FALSE;
}

//-----------------------------------------------------------
GRResult grGetCompressionInfo( GRCompressionInfo* outInfo, const size_t infoStructSize, 
                               const uint32_t k, const uint32_t compressionLevel )
{
    if( infoStructSize != sizeof( GRCompressionInfo ) )
        return GRResult_WrongVersion;

    if( outInfo == nullptr 
        || k != 32
        || compressionLevel < 1
        || compressionLevel > 40 )
    {
        return GRResult_InvalidArg;
    }

    auto c = GetCompressionInfoForLevel( compressionLevel );
    outInfo->entrySizeBits = c.entrySizeBits;
    outInfo->stubSizeBits  = c.stubSizeBits;
    outInfo->tableParkSize = c.tableParkSize;
    outInfo->ansRValue     = c.ansRValue;

    return GRResult_OK;
}

//-----------------------------------------------------------
GRResult grFetchProofForChallenge( GreenReaperContext* cx, GRCompressedProofRequest* req )
{
    if( !req || !req->plotId )
        return GRResult_Failed;

    const uint32 k = 32;
    {
        auto r = RequestSetup( cx, k, req->compressionLevel );
        if( r != GRResult_OK )
            return r;
    }

    const uint32 numGroups        = GR_POST_PROOF_CMP_X_COUNT;
    const uint64 entriesPerBucket = GetEntriesPerBucketForCompressionLevel( k, req->compressionLevel );
    ASSERT( entriesPerBucket <= 0xFFFFFFFF );


    bool proofMightBeDropped = false;

    if( cx->cudaThresher ) cx->cudaThresher->ClearTimings();

    uint32 xGroups[GR_POST_PROOF_X_COUNT] = {};

    // Unpack x groups first
    if( req->compressionLevel < 40 )
    {
        for( uint32 i = 0, j = 0; i < numGroups; i++, j+=2 )
        {
            const uint32  xLinePoint = (uint32)req->compressedProof[i];
            const BackPtr xs         = LinePointToSquare64( xLinePoint );

            proofMightBeDropped = proofMightBeDropped || (xs.x == 0 || xs.y == 0);

            xGroups[j+0] = (uint32)xs.y;
            xGroups[j+1] = (uint32)xs.x;
        }
    }
    else
    {
        for( uint32 i = 0, j = 0; i < numGroups / 2; i++, j+=4 )
        {
            const uint32  xLinePoint = (uint32)req->compressedProof[i];
            const BackPtr xs         = LinePointToSquare64( xLinePoint );

            const uint32 entrybits = GetCompressionInfoForLevel( req->compressionLevel ).entrySizeBits;
            const uint32 mask      = (1u << entrybits) - 1;

            proofMightBeDropped = proofMightBeDropped || (xs.x == 0 || xs.y == 0);

            xGroups[j+0] = (uint32)(xs.y) & mask;
            xGroups[j+1] = (uint32)(xs.y) >> entrybits;
            xGroups[j+2] = (uint32)(xs.x) & mask;
            xGroups[j+3] = (uint32)(xs.x) >> entrybits;
        }
    }

    {
        Table1BucketContext tcx;
        tcx.cx               = cx;
        tcx.plotId           = req->plotId;
        tcx.entriesPerBucket = entriesPerBucket;
        tcx.outY             = cx->yBufferTmp;
        tcx.outMeta          = cx->metaBufferTmp.template As<K32Meta2>();
        tcx.outPairs         = cx->pairs;

        for( uint32 i = 0, j = 0; i < numGroups; i++, j+= 2 )
        {
            const uint64 x1 = xGroups[j+0];
            const uint64 x2 = xGroups[j+1];

                  auto&  table      = cx->tables[1];
            const uint32 groupIndex = i / 2;

            if( (i % 2 ) == 0 )
                table.BeginGroup( groupIndex );

            const GRResult r = ProcessTable1Bucket( tcx, x1, x2, groupIndex );
             if( r != GRResult_OK )
                return proofMightBeDropped ? GRResult_NoProof : GRResult_Failed;
        }
    }

    // Test
    if( cx->cudaThresher )
        cx->cudaThresher->DumpTimings();

    // #NOTE: Sanity check, but should never happen w/ our starting compression levels.
    if( cx->tables[1]._length <= 2 )
    {
        Log::Line( "[GR_WARNING] Unexpected proof match on first table." );
        return GRResult_Failed;
    }

    // Continue forward propagation to the next table
    const uint32 table2Length = (uint32)cx->tables[1]._length;

    cx->proofContext.leftLength = table2Length;
    // cx->proofContext.rightLength = cx->tables[2]._capacity;

    cx->proofContext.yLeft     = cx->yBuffer.Ptr();
    cx->proofContext.metaLeft  = cx->metaBuffer.Ptr();
    cx->proofContext.yRight    = cx->yBufferTmp.Ptr();
    cx->proofContext.metaRight = cx->metaBufferTmp.Ptr();
    cx->proofContext.proof     = req->fullProof;

    SortTableAndFlipBuffers<TableId::Table2>( *cx, 32 >> (int)TableId::Table2 );

    return ForwardPropTables( *cx ) ? GRResult_OK : 
        proofMightBeDropped ? GRResult_NoProof : GRResult_Failed;
}

//-----------------------------------------------------------
GRResult grGetFetchQualitiesXPair( GreenReaperContext* cx, GRCompressedQualitiesRequest* req )
{
    if( !req || !req->plotId )
        return GRResult_Failed;

    const uint32 k = 32;
    {
        auto r = RequestSetup( cx, k, req->compressionLevel );
        if( r != GRResult_OK )
            return r;
    }

    if( cx->cudaThresher ) cx->cudaThresher->ClearTimings();

    const uint64 entriesPerBucket = GetEntriesPerBucketForCompressionLevel( k, req->compressionLevel );
    ASSERT( entriesPerBucket <= 0xFFFFFFFF );

    // Begin decompression
    Table1BucketContext tcx;
    tcx.cx               = cx;
    tcx.plotId           = req->plotId;
    tcx.entriesPerBucket = entriesPerBucket;
    tcx.outY             = cx->yBufferTmp;
    tcx.outMeta          = cx->metaBufferTmp.template As<K32Meta2>();
    tcx.outPairs         = cx->pairs;

    uint32 xGroups[16] = {};
    uint32 numXGroups  = 0;

    bool proofMightBeDropped = false;

    // #TODO: Support more K values
    {
        const BackPtr p = LinePointToSquare( ((uint128)req->xLinePoints[0].hi) << 64 | (uint128)req->xLinePoints[0].lo );

        // For level < 9, we have 4 x groups inlined
        const BackPtr x1x2 = LinePointToSquare64( p.x );
        const BackPtr x2x3 = LinePointToSquare64( p.y );

        proofMightBeDropped = (x1x2.x == 0 || x1x2.y == 0) || (x2x3.x == 0 || x2x3.y == 0);

        if( req->compressionLevel < 40 )
        {
            numXGroups = 2;
            xGroups[0] = (uint32)x1x2.x;
            xGroups[1] = (uint32)x1x2.y;
            xGroups[2] = (uint32)x2x3.x;
            xGroups[3] = (uint32)x2x3.y;
        }
        else
        {

            // Level 9 and above have 8 packed entries
            const uint32 entrybits = GetCompressionInfoForLevel( req->compressionLevel ).entrySizeBits;
            const uint32 mask      = (1u << entrybits) - 1;

            numXGroups = 4;
            xGroups[0] = ((uint32)x1x2.x) & mask;
            xGroups[1] = ((uint32)x1x2.x) >> entrybits;
            xGroups[2] = ((uint32)x1x2.y) & mask;
            xGroups[3] = ((uint32)x1x2.y) >> entrybits;
            xGroups[4] = ((uint32)x2x3.x) & mask;
            xGroups[5] = ((uint32)x2x3.x) >> entrybits;
            xGroups[6] = ((uint32)x2x3.y) & mask;
            xGroups[7] = ((uint32)x2x3.y) >> entrybits;
        }
    }

    if( req->compressionLevel >= 6 && req->compressionLevel < 40 )
    {
        const BackPtr p = LinePointToSquare( ((uint128)req->xLinePoints[1].hi) << 64 | (uint128)req->xLinePoints[1].lo );

        const BackPtr x1x2 = LinePointToSquare64( p.x );
        const BackPtr x2x3 = LinePointToSquare64( p.y );

        proofMightBeDropped = proofMightBeDropped || (x1x2.x == 0 || x1x2.y == 0) || (x2x3.x == 0 || x2x3.y == 0);

        if( req->compressionLevel < 40 )
        {
            numXGroups = 4;
            xGroups[4] = (uint32)x1x2.x;
            xGroups[5] = (uint32)x1x2.y;
            xGroups[6] = (uint32)x2x3.x;
            xGroups[7] = (uint32)x2x3.y;
        }
        else
        {
            const uint32 entrybits = GetCompressionInfoForLevel( req->compressionLevel ).entrySizeBits;
            const uint32 mask      = (1u << entrybits) - 1;

            numXGroups = 8;
            xGroups[8 ] = ((uint32)x1x2.x) & mask;
            xGroups[9 ] = ((uint32)x1x2.x) >> entrybits;
            xGroups[10] = ((uint32)x1x2.y) & mask;
            xGroups[11] = ((uint32)x1x2.y) >> entrybits;
            xGroups[12] = ((uint32)x2x3.x) & mask;
            xGroups[13] = ((uint32)x2x3.x) >> entrybits;
            xGroups[14] = ((uint32)x2x3.y) & mask;
            xGroups[15] = ((uint32)x2x3.y) >> entrybits;
        }
    }

    for( uint32 i = 0; i < numXGroups; i++ )
    {
        // Gen sorted f1
        const uint32 xIdx = i*2;
        const uint64 x1   = xGroups[xIdx+0];
        const uint64 x2   = xGroups[xIdx+1];

        const uint32 groupIndex = i / 2;
        if( (i & 1) == 0 )
            cx->tables[1].BeginGroup( groupIndex );

        const auto r = ProcessTable1Bucket( tcx, x1, x2, groupIndex );

        if( r != GRResult_OK )
            return proofMightBeDropped ? GRResult_NoProof : GRResult_Failed;
    }

    if( cx->tables[1]._length == 1 )
    {
        // #TODO: Add this path, but this should never happen
        Log::Line( "[GR_WARNING] Unexpected qualities match on first 2nd table." );
        return GRResult_Failed;
    }

    // Continue forward propagation to the next table
    cx->proofContext.leftLength  = (uint32)cx->tables[1]._length;

    uint64 fullProof[GR_POST_PROOF_X_COUNT] = {};

    cx->proofContext.yLeft     = cx->yBuffer.Ptr();
    cx->proofContext.metaLeft  = cx->metaBuffer.Ptr();
    cx->proofContext.yRight    = cx->yBufferTmp.Ptr();
    cx->proofContext.metaRight = cx->metaBufferTmp.Ptr();
    cx->proofContext.proof     = fullProof;

    SortTableAndFlipBuffers<TableId::Table2>( *cx, numXGroups / 2 );


    TableId matchTable = TableId::Table3;

    auto fpResult = ForwardPropTable<TableId::Table3>( *cx, numXGroups /= 2, true );

    if( fpResult == ForwardPropResult::Continue )
    {
        matchTable = TableId::Table4;
        fpResult = ForwardPropTable<TableId::Table4>( *cx, numXGroups /= 2, true );
    }
    if( fpResult == ForwardPropResult::Continue )
    {
        matchTable = TableId::Table5;
        fpResult = ForwardPropTable<TableId::Table5>( *cx, numXGroups /= 2, true );
    }
    if( fpResult == ForwardPropResult::Continue )
    {
        matchTable = TableId::Table6;
        fpResult = ForwardPropTable<TableId::Table6>( *cx, numXGroups /= 2, true );
    }

    if( fpResult != ForwardPropResult::Success )
        return proofMightBeDropped ? GRResult_NoProof : GRResult_Failed;

    // If we have more than 1 group, we need to only trace back to
    // the ones that belong to the first group
    uint64 qualityXs[8] = {};

    {
        Pair  pairs[2][4] = {};
        Pair* pairsIn  = pairs[0];
        Pair* pairsOut = pairs[1];

        if( cx->cudaThresher )
            bbmemcpy_t( cx->tables[(int)matchTable]._pairs, cx->pairs.Ptr(), cx->tables[(int)matchTable]._length );

        bbmemcpy_t( pairs[0], cx->tables[(int)matchTable]._pairs, cx->tables[(int)matchTable]._length );

        for( TableId rTable = matchTable; rTable > TableId::Table3; rTable-- )
        {
            auto& inTable  = cx->tables[(int)rTable];
            auto& outTable = cx->tables[(int)rTable-1];

            for( uint32 i = 0; i < inTable._length; i++ )
            {
                const auto pair = inTable._pairs[i];

                pairsOut[i*2+0] = outTable._pairs[pair.left ];
                pairsOut[i*2+1] = outTable._pairs[pair.right];
            }

            std::swap( pairsOut, pairsIn );
        }

        // From table 3, only take the pair that points to the first group
        const Pair t3Pair = pairsIn[0].left < cx->tables[1]._groups[0].count ?
                            pairsIn[0] : pairsIn[1];

        // Grab the x's from the first group only
        const Pair xPair0 = cx->tables[1]._pairs[t3Pair.left ];
        const Pair xPair1 = cx->tables[1]._pairs[t3Pair.right];

        qualityXs[0] = xPair0.left;
        qualityXs[1] = xPair0.right;
        qualityXs[2] = xPair1.left;
        qualityXs[3] = xPair1.right;
    }

    // We need to now sort the X's on y, in order to chose the right path
    SortQualityXs( k, req->plotId, qualityXs, 4 );

    // Follow the last path, based on the challenge in our x's
    const uint32 last5Bits      = (uint32)req->challenge[31] & 0x1f;
    const bool   isTable1BitSet = (last5Bits & 1) == 1;

    if( !isTable1BitSet )
    {
        req->x1 = qualityXs[0];
        req->x2 = qualityXs[1];
    }
    else
    {
        req->x1 = qualityXs[2];
        req->x2 = qualityXs[3];
    }

    return GRResult_OK;
}


///
/// Private Funcs
///
//-----------------------------------------------------------
GRResult ProcessTable1Bucket( Table1BucketContext& tcx, const uint64 x1, const uint64 x2, const uint32 groupIndex )
{
    GreenReaperContext& cx = *tcx.cx;

    // #TODO: Not supported proofs, these should be droped
    const bool proofMightBeDropped = x1 == 0 || x2 == 0 ;
    
    // if( false )
    if( cx.cudaThresher != nullptr )
    {
// #if _DEBUG
//         ProcessTable1BucketCPU( tcx, x1, x2, groupIndex );
// #endif
        uint32 matchCount = 0;

        const auto r = cx.cudaThresher->DecompressInitialTable(
                        cx,
                        tcx.plotId,
                        (uint32)tcx.entriesPerBucket,
                        tcx.outPairs.Ptr(),
                        tcx.outY.Ptr(),
                        tcx.outMeta.Ptr(),
                        matchCount,
                        x1, x2 );

        if( r.kind == ThresherResultKind::Error )
        {
            // #NOTE: Perhaps find a way to log this or pass it back to the caller.
            delete cx.cudaThresher;
            cx.cudaThresher         = nullptr;
            cx.cudaRecreateThresher = true;

            return GRResult_Failed;
        }
        else if( r.kind == ThresherResultKind::Success )
        {
            auto& table = cx.tables[1];
            table.AddGroupPairs( groupIndex, matchCount );
// Log::Line( "[%u] CUDA Pairs: %u", groupIndex, matchCount );
            tcx.outPairs = tcx.outPairs.Slice( matchCount );
            tcx.outY     = tcx.outY    .Slice( matchCount );
            tcx.outMeta  = tcx.outMeta .Slice( matchCount );

            return GRResult_OK;
        }

        return proofMightBeDropped ? GRResult_NoProof : GRResult_Failed;
    }
    else
    {
        return ProcessTable1BucketCPU( tcx, x1, x2, groupIndex );
    }

    return GRResult_Failed;
}

//-----------------------------------------------------------
GRResult ProcessTable1BucketCPU( Table1BucketContext& tcx, const uint64 x1, const uint64 x2, const uint32 groupIndex )
{
    GreenReaperContext& cx = *tcx.cx;

    // #TODO: Not supported proofs, these should be droped
    const bool proofMightBeDropped = x1 == 0 || x2 == 0 ;

    // Gen F1 for both X buckets into a single bucket sorted on Y
    GenerateF1( cx, tcx.plotId, tcx.entriesPerBucket, (uint32)x1, (uint32)x2 );

    // Perform matches
    auto yEntries = cx.yBuffer;
    auto xEntries = cx.xBuffer;

    const Span<Pair> pairs = Match( cx, yEntries, tcx.outPairs, 0 );
// Log::Line( "[%u] CPU Pairs: %u", groupIndex, (uint)pairs.Length() );
    // Expect at least one match
    if( pairs.Length() < 1 )
        return proofMightBeDropped ? GRResult_NoProof: GRResult_Failed;

    auto& table = cx.tables[1];
    table.AddGroupPairs( groupIndex, (uint32)pairs.Length() );

    // Perform fx for table2 pairs
    GenerateFxForPairs<TableId::Table2>( cx, pairs, yEntries, xEntries, tcx.outY, tcx.outMeta );

    // Inline x's into the pairs
    {
        #if SHOW_TIMINGS
            const auto timer = TimerBegin();
        #endif
        const uint32 threadCount = std::min( cx.config.threadCount, (uint32)pairs.Length() );

        AnonMTJob::Run( *cx.pool, threadCount, [=]( AnonMTJob* self ){
            
            const Span<uint32> xs          = xEntries;
                  Span<Pair>   threadPairs = GetThreadOffsets( self, pairs );

            for( uint64 i = 0; i < threadPairs.Length(); i++ )
            {
                Pair p = threadPairs[i];
                p.left  = xs[p.left ];
                p.right = xs[p.right];

                threadPairs[i] = p;
            }
        });
        #if SHOW_TIMINGS
            Log::Line( "Inline x elapsed: %.3lf s", TimerEnd( timer ) );
        #endif
    }

    tcx.outPairs = tcx.outPairs.Slice( pairs.Length() );
    tcx.outY     = tcx.outY    .Slice( pairs.Length() );
    tcx.outMeta  = tcx.outMeta .Slice( pairs.Length() );

    return GRResult_OK;
}

//-----------------------------------------------------------
void BacktraceProof( GreenReaperContext& cx, const TableId tableStart, uint64 proof[GR_POST_PROOF_X_COUNT] )
{
    Pair _backtrace[2][64] = {};

    Pair* backTraceIn  = _backtrace[0];
    Pair* backTraceOut = _backtrace[1];

    // #TODO: If in table 2, get the x's directly

    // Fill initial back-trace
    {
        ProofTable& table = cx.tables[(int)tableStart];

        const Pair* pairsSrc = cx.cudaThresher != nullptr ? cx.pairs.Ptr() : table._pairs;
        bbmemcpy_t( backTraceIn, pairsSrc, table._length );

        // const uint32 groupCount = 32 >> (int)tableStart;
            

        // for( uint32 i = 0; i < groupCount; i++ )
        // {
        //     ASSERT( table._length == 2 );

        //     Pair p0 = table._pairs[0];
        //     Pair p1 = table._pairs[1];

        //     const uint32 idx = i*4;

        //     backTraceIn[idx+0] = e0.parentL;
        //     backTraceIn[idx+1] = e0.parentR;
        //     backTraceIn[idx+2] = e1.parentL;
        //     backTraceIn[idx+3] = e1.parentR;
        // }
    }

    for( TableId table = tableStart; table > TableId::Table2; table-- )
    {
        const uint32 entryCount = (32 >> (int)table)*2;

        ProofTable& lTable = cx.tables[(int)table-1];

        for( uint32 i = 0; i < entryCount; i++ )
        {
            const Pair p = backTraceIn[i];
            
            const uint32 idx = i * 2;
            backTraceOut[idx+0] = lTable._pairs[p.left ];
            backTraceOut[idx+1] = lTable._pairs[p.right];
        }

        std::swap( backTraceIn, backTraceOut );
    }

    for( uint32 i = 0; i < GR_POST_PROOF_CMP_X_COUNT; i++ )
    {
        const uint32 idx = i * 2;
        proof[idx+0] = backTraceIn[i].left;
        proof[idx+1] = backTraceIn[i].right;
    }

// #if _DEBUG
//     Log::Line( "" );
//     Log::Line( "Recovered Proof:" );
//     for( uint32 i = 0; i < GR_POST_PROOF_X_COUNT; i++ )
//         Log::WriteLine( "[%2u]: %-10u ( 0x%08x )", i, (uint32)proof[i], (uint32)proof[i] );
// #endif
}

//-----------------------------------------------------------
GRResult RequestSetup( GreenReaperContext* cx, const uint32 k, const uint32 compressionLevel )
{
    if( compressionLevel < 1 || compressionLevel > 40 )
        return GRResult_Failed;

    // Make sure we have our CUDA decompressor working in case it was deleted after a failure
    {
        // if( cx->config.gpuRequest != GRGpuRequestKind_None )
        if( cx->cudaRecreateThresher )
        {
            ASSERT( !cx->cudaThresher );
            cx->cudaThresher = CudaThresherFactory::Create( cx->config );
            if( !cx->cudaThresher )
                return GRResult_Failed;

            cx->cudaRecreateThresher = false;
        }
    }

    const GRResult r = grPreallocateForCompressionLevel( cx, k, compressionLevel );
    if( r != GRResult_OK )    
        return r;

    // Always make sure this has been done
    {
        // #TODO: Make The function itself thread-safe, don't do it out here
        _lTargetLock.lock();
        LoadLTargets();
        _lTargetLock.unlock();
    }

    return r;
}

//-----------------------------------------------------------
bool ReserveBucketBuffers( GreenReaperContext& cx, const uint32 k, const uint32 compressionLevel )
{
    // #TODO: Support other K values
    ASSERT( k == 32 );

    if( compressionLevel > cx.maxCompressionLevelReserved )
    {
        const uint64 entriesPerBucket = GetEntriesPerBucketForCompressionLevel( k, compressionLevel );
        ASSERT( entriesPerBucket > cx.maxEntriesPerBucket );

        cx.maxEntriesPerBucket = 0;
        FreeBucketBuffers( cx );

        const size_t allocCount = (size_t)entriesPerBucket * 2;

        // The pair requirements ought to be much less as the number of matches we get per group is not as high.
        uint64 maxPairsPerTable = std::max( (uint64)GR_MIN_TABLE_PAIRS, GetMaxTablePairsForCompressionLevel( k, compressionLevel ) );

        VirtualAllocator alloc;

        cx.yBufferF1        = alloc.TryCAllocBoundedSpan<uint64>  ( allocCount );
        cx.yBuffer          = alloc.TryCAllocBoundedSpan<uint64>  ( allocCount );
        cx.yBufferTmp       = alloc.TryCAllocBoundedSpan<uint64>  ( allocCount );
        cx.xBuffer          = alloc.TryCAllocBoundedSpan<uint32>  ( allocCount );
        cx.xBufferTmp       = alloc.TryCAllocBoundedSpan<uint32>  ( allocCount );
        cx.sortKey          = alloc.TryCAllocBoundedSpan<uint32>  ( maxPairsPerTable );
        cx.metaBuffer       = alloc.TryCAllocBoundedSpan<K32Meta4>( maxPairsPerTable );
        cx.metaBufferTmp    = alloc.TryCAllocBoundedSpan<K32Meta4>( maxPairsPerTable );
        cx.pairs            = alloc.TryCAllocBoundedSpan<Pair>    ( maxPairsPerTable );
        cx.pairsTmp         = alloc.TryCAllocBoundedSpan<Pair>    ( maxPairsPerTable );
        cx.groupsBoundaries = alloc.TryCAllocBoundedSpan<uint32>  ( allocCount );

        // Allocate proof tables
        // Table 1 needs no groups, as we write to the R table's merged group, always
        for( uint32 i = 1; i < 7; i++ )
        {
            cx.tables[i] = {};
            cx.tables[i]._pairs    = alloc.TryCAllocBounded<Pair>( maxPairsPerTable );
            cx.tables[i]._capacity = maxPairsPerTable;

            // Reduce the match count for each subsequent table by nearly half
            maxPairsPerTable = std::max<uint64>( (uint64)(maxPairsPerTable * 0.6), GR_MIN_TABLE_PAIRS );
        }

        cx.allocationSize = alloc.AllocSize();

        bool allocFailed = alloc.FailCount() > 0;
        if( !allocFailed && cx.cudaThresher != nullptr )
            allocFailed = cx.cudaThresher->AllocateBuffers( k, compressionLevel );


        if( alloc.FailCount() > 0 )
        {   
            FreeBucketBuffers( cx );
            return false;
        }

        cx.maxEntriesPerBucket         = entriesPerBucket;
        cx.maxCompressionLevelReserved = compressionLevel;
    }

    if( cx.cudaThresher != nullptr )
    {
        if( !cx.cudaThresher->AllocateBuffers( k, compressionLevel ) )
        {
            FreeBucketBuffers( cx );
            delete cx.cudaThresher;
            cx.cudaThresher         = nullptr;
            cx.cudaRecreateThresher = true;
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------
void FreeBucketBuffers( GreenReaperContext& cx )
{
    cx.allocationSize = 0;
    cx.maxCompressionLevelReserved = 0;

    bbvirtfreebounded_span( cx.yBufferF1 );
    bbvirtfreebounded_span( cx.yBuffer );
    bbvirtfreebounded_span( cx.yBufferTmp );
    bbvirtfreebounded_span( cx.xBuffer );
    bbvirtfreebounded_span( cx.xBufferTmp );
    bbvirtfreebounded_span( cx.sortKey );
    bbvirtfreebounded_span( cx.metaBuffer );
    bbvirtfreebounded_span( cx.metaBufferTmp );
    bbvirtfreebounded_span( cx.pairs );
    bbvirtfreebounded_span( cx.pairsTmp );
    bbvirtfreebounded_span( cx.groupsBoundaries );

    for( uint32 i = 0; i < 7; i++ )
        bbvirtfreebounded( cx.tables[i]._pairs );

    if( cx.cudaThresher != nullptr )
        cx.cudaThresher->ReleaseBuffers();
}


//-----------------------------------------------------------
inline void FlipBuffers( GreenReaperContext& cx, const TableId rTable )
{
    ProofTable& table = cx.tables[(int)rTable];

    const uint32 tableLength = (uint32)table._length;

    ASSERT( tableLength <= cx.pairs.Length() );
    ASSERT( tableLength <= table._capacity );

    cx.proofContext.leftLength  = (uint32)tableLength;
    cx.proofContext.rightLength = (uint32)cx.tables[(int)rTable+1]._capacity;

    std::swap( cx.proofContext.yLeft, cx.proofContext.yRight );
    std::swap( cx.proofContext.metaLeft, cx.proofContext.metaRight );
}

//-----------------------------------------------------------
template<TableId rTable>
inline void SortTableAndFlipBuffers( GreenReaperContext& cx, const uint32 groupCount )
{
    if( cx.cudaThresher != nullptr )
    {
        ASSERT( cx.cudaThresher );
        FlipBuffers( cx, rTable );
        return;
    }

    #if SHOW_TIMINGS
        const auto timer = TimerBegin();
    #endif
    using TMeta = typename K32MetaType<rTable>::Out;

    ProofTable& table = cx.tables[(int)rTable];

    const uint32 tableLength = (uint32)table._length;
    const uint32 threadCount = cx.config.threadCount;

    ASSERT( tableLength <= cx.pairs.Length() );
    ASSERT( tableLength <= table._capacity );

    // At this point the yRight/metaRight hold the unsorted fx output 
    // from the left table pairs/y/meta and sort onto the right buffers
    Span<uint64> tableYUnsorted     = MakeSpan( cx.proofContext.yRight, tableLength );
    Span<uint64> tableYSorted       = MakeSpan( cx.proofContext.yLeft , tableLength );
    Span<TMeta>  tableMetaUnsorted  = MakeSpan( (TMeta*)cx.proofContext.metaRight, tableLength );
    Span<TMeta>  tableMetaSorted    = MakeSpan( (TMeta*)cx.proofContext.metaLeft , tableLength );
    Span<Pair>   tablePairsUnsorted = cx.pairs.SliceSize( tableLength );
    Span<Pair>   tablePairsSorted   = MakeSpan( table._pairs, tableLength );

    Span<uint32> keyUnsorted        = cx.xBufferTmp.SliceSize( tableLength );
    Span<uint32> keySorted          = cx.xBuffer   .SliceSize( tableLength );

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint32 groupLength  = table._groups[i].count;
        const uint32 groupThreads = std::min( threadCount, groupLength );

        auto yUnsorted     = tableYUnsorted    .SliceSize( groupLength );
        auto ySorted       = tableYSorted      .SliceSize( groupLength );
        auto metaUnsorted  = tableMetaUnsorted .SliceSize( groupLength );
        auto metaSorted    = tableMetaSorted   .SliceSize( groupLength );
        auto pairsUnsorted = tablePairsUnsorted.SliceSize( groupLength );
        auto pairsSorted   = tablePairsSorted  .SliceSize( groupLength );

        tableYUnsorted     = tableYUnsorted    .Slice( groupLength );
        tableYSorted       = tableYSorted      .Slice( groupLength );
        tableMetaUnsorted  = tableMetaUnsorted .Slice( groupLength );
        tableMetaSorted    = tableMetaSorted   .Slice( groupLength );
        tablePairsUnsorted = tablePairsUnsorted.Slice( groupLength ); 
        tablePairsSorted   = tablePairsSorted  .Slice( groupLength );


        auto kUnsorted = keyUnsorted.SliceSize( groupLength );
        auto kSorted   = keySorted  .SliceSize( groupLength );

        // #TODO: Perhaps do all groups and this in a single job (more code repetition, though)?
        SortKeyJob::GenerateKey( *cx.pool, groupThreads, keyUnsorted );

        RadixSort256::SortYWithKey<BB_MAX_JOBS>( *cx.pool, groupThreads, yUnsorted.Ptr(), ySorted.Ptr(),
                                                 kUnsorted.Ptr(), kSorted.Ptr(), groupLength );

        SortKeyJob::SortOnKey( *cx.pool, groupThreads, kSorted, metaUnsorted , metaSorted );
        SortKeyJob::SortOnKey( *cx.pool, groupThreads, kSorted, pairsUnsorted, pairsSorted );
    }

    cx.proofContext.leftLength  = (uint32)tableLength;
    cx.proofContext.rightLength = (uint32)cx.tables[(int)rTable+1]._capacity;

    #if SHOW_TIMINGS
        Log::Line( "Sort elapsed: %.3lf s", TimerEnd( timer ) );
    #endif
}

///
/// F1
///
//-----------------------------------------------------------
void GenerateF1( GreenReaperContext& cx, const byte plotId[32], const uint64 bucketEntryCount, const uint32 x0, const uint32 x1 )
{
    #if SHOW_TIMINGS
        const auto timer = TimerBegin();
    #endif

    const uint32 k = 32;
    const uint32 f1BlocksPerBucket = (uint32)(bucketEntryCount * sizeof( uint32 ) / kF1BlockSize);

    uint32 threadCount = cx.pool->ThreadCount();

    while( f1BlocksPerBucket < threadCount )
        threadCount--;
    
    // Out buffers are continuous, so that we can merge both buckets into one
    uint32* xBuffer = cx.xBufferTmp.Ptr();
    uint64* yBuffer = cx.yBufferF1 .Ptr();

    auto blocks = Span<uint32>( (uint32*)cx.yBuffer.Ptr(), bucketEntryCount );

    const uint32 xSources[2] = { x0, x1 };

    Span<uint32> xEntries[2] = {
        Span<uint32>( xBuffer, bucketEntryCount ),
        Span<uint32>( xBuffer + bucketEntryCount, bucketEntryCount )
    };

    Span<uint64> yEntries[2] = {
        Span<uint64>( yBuffer, bucketEntryCount ),
        Span<uint64>( yBuffer + bucketEntryCount, bucketEntryCount )
    };

    byte key[32] = { 1 };
    memcpy( key+1, plotId, 31 );

    AnonMTJob::Run( *cx.pool, threadCount, [=]( AnonMTJob* self ) {

        const uint32 xShift            = k - kExtraBits;
        const uint32 f1EntriesPerBlock = kF1BlockSize / sizeof( uint32 );

        uint32 numBlocks, blockOffset, _;
        GetThreadOffsets( self, f1BlocksPerBucket, numBlocks, blockOffset, _ );

        const uint32 entriesPerThread = numBlocks * f1EntriesPerBlock;
        const uint32 entriesOffset    = blockOffset * f1EntriesPerBlock;

        Span<uint32> f1Blocks = blocks.Slice( blockOffset * f1EntriesPerBlock, numBlocks * f1EntriesPerBlock );

        chacha8_ctx chacha;
        chacha8_keysetup( &chacha, key, 256, NULL );

        for( uint32 i = 0; i < 2; i++ )
        {
            Span<uint32> xSlice = xEntries[i].Slice( entriesOffset, entriesPerThread );
            Span<uint64> ySlice = yEntries[i].Slice( entriesOffset, entriesPerThread );

            const uint32 xStart     = (uint32)(( xSources[i] * bucketEntryCount ) + entriesOffset);
            const uint32 blockIndex = xStart / f1EntriesPerBlock;

            chacha8_get_keystream( &chacha, blockIndex, numBlocks, (byte*)f1Blocks.Ptr() );

            for( uint32 j = 0; j < entriesPerThread; j++ )
            {
                // Get the starting and end locations of y in bits relative to our block
                const uint32 x = xStart + j;
                
                uint64 y = Swap32( f1Blocks[j] );
                y = ( y << kExtraBits ) | ( x >> xShift );

                xSlice[j] = x;
                ySlice[j] = y;
            }

// #if _DEBUG
//             {
//                 for( uint32 j = 0; j < entriesPerBucket; j++ )
//                     Log::Line( "[%2u] %-10u | 0x%08x", j,  xEntries[j], xEntries[j] );
//             }
// #endif
        }
    });

    // const auto elapsed = TimerEnd( timer );

    // Log::Line( "Completed F1 in %.2lf seconds.", TimerEnd( timer ) );


    // Sort f1 on y
    const uint64 mergedEntryCount = bucketEntryCount * 2;
    RadixSort256::SortYWithKey<BB_MAX_JOBS>( *cx.pool, yBuffer, cx.yBuffer.Ptr(), xBuffer, cx.xBuffer.Ptr(), mergedEntryCount );
    
    #if SHOW_TIMINGS
        Log::Line( "F1 elapsed: %.3lf s", TimerEnd( timer ) );
    #endif
}

//-----------------------------------------------------------
void SortQualityXs( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64* x, const uint32 count )
{
    ASSERT( count <= 16 );
    
    uint128 lp[8] = {};
    const uint32 lpCount = count / 2;

    for( uint32 i = 0; i < lpCount; i++ )
        lp[i] = SquareToLinePoint128( x[i*2], x[i*2+1] );

    std::sort( &lp[0], &lp[lpCount] );

    for( uint32 i = 0; i < lpCount; i++ )
    {
        const auto v = LinePointToSquare( lp[i] );

        x[i*2]   = v.x;
        x[i*2+1] = v.y;
    }
}

///
/// Forward propagation
///
//-----------------------------------------------------------
template<TableId rTableId>
inline uint64 ForwardPropTableGroup( GreenReaperContext& cx, const uint32 lGroup, Span<Pair> outPairs, Span<uint64> yRight, Span<typename K32MetaType<rTableId>::Out> metaRight  )
{
    using TMetaIn  = typename K32MetaType<rTableId>::In;
    using TMetaOut = typename K32MetaType<rTableId>::Out;

    ASSERT( yRight.Length() == metaRight.Length() );

    ProofTable& lTable = cx.tables[(int)rTableId-1];
    ProofTable& rTable = cx.tables[(int)rTableId];

    Span<uint64>  yLeft;    
    Span<TMetaIn> metaLeft;
    lTable.GetLTableGroupEntries<rTableId-1>( cx, lGroup, yLeft, metaLeft );

    Span<Pair> pairs;

    const uint32 rGroup = lGroup / 2;
    if( (lGroup & 1) == 0)
        rTable.BeginGroup( rGroup );

    if( cx.cudaThresher != nullptr )
    {
        auto inLeftPairs = Span<Pair>( cx.pairs.Ptr() + lTable._groups[lGroup].offset,
                                       lTable._groups[lGroup].count );

        auto outLeftPairs = Span<Pair>( lTable._pairs + lTable._groups[lGroup].offset,
                                       lTable._groups[lGroup].count );

        const uint32 matchOffset = lTable._groups[lGroup].offset;
              uint32 matchCount  = 0;

        const auto r = cx.cudaThresher->DecompressTableGroup(
            cx,
            rTableId,
            (uint32)yLeft.Length(),
            matchOffset,
            (uint32)outPairs.Length(),
            matchCount,                   // Output
            outPairs.Ptr(),
            yRight.Ptr(),
            metaRight.Ptr(),
            outLeftPairs.Ptr(),
            inLeftPairs.Ptr(),            // Input (pairs are in/out as they get sorted here)
            yLeft.Ptr(),
            metaLeft.Ptr() );

        if( r.kind == ThresherResultKind::Error )
        {
            // Perhaps find a way to log this or pass the error back to the caller.
            delete cx.cudaThresher;
            cx.cudaThresher         = nullptr;
            cx.cudaRecreateThresher = true;
        }

         if( r.kind != ThresherResultKind::Success || matchCount < 1 )
            return 0;

        ASSERT( matchCount <= inLeftPairs.Length() );

        pairs = outPairs.SliceSize( matchCount );
    }
    else
    {
        // Match
        #if SHOW_TIMINGS
            const auto timer = TimerBegin();
        #endif
        
        pairs = Match( cx, yLeft, outPairs, lTable._groups[lGroup].offset );
        
        #if SHOW_TIMINGS
            Log::Line( " Match elapsed: %.3lf s", TimerEnd( timer ) );
        #endif

        if( pairs.Length() > yRight.Length() )
            return 0;

        // Fx
        if( pairs.Length() > 0 )
        {
            // Since pairs have the global L table offset applied to them,
            // we need to turn the left values back to global table y and meta, instead
            // of group-local y and meta
            yLeft    = MakeSpan( cx.proofContext.yLeft, cx.proofContext.leftLength );
            metaLeft = MakeSpan( (TMetaIn*)cx.proofContext.metaLeft, cx.proofContext.leftLength );

            #if SHOW_TIMINGS
                const auto timer = TimerBegin();
            #endif

            GenerateFxForPairs<rTableId, TMetaIn, TMetaOut>( cx, pairs, yLeft, metaLeft, yRight, metaRight );

            #if SHOW_TIMINGS
                Log::Line( " Fx elapsed: %.3lf s", TimerEnd( timer ) );
            #endif
        }
    }

    rTable.AddGroupPairs( rGroup, (uint32)pairs.Length() );

    return pairs.Length();
}

//-----------------------------------------------------------
template<TableId rTable>
ForwardPropResult ForwardPropTable( GreenReaperContext& cx, uint32 tableGroupCount, bool returnSuccessOnSingleMatch )
{
    #if SHOW_TIMINGS
        const auto timer = TimerBegin();
    #endif
    using TMetaOut = typename K32MetaType<rTable>::Out;

    if( cx.cudaThresher ) cx.cudaThresher->ClearTimings();

    auto& table = cx.tables[(int)rTable];
    // ASSERT( table._length == 0 );

    auto yRight    = MakeSpan( cx.proofContext.yRight, table._capacity );
    auto metaRight = MakeSpan( (TMetaOut*)cx.proofContext.metaRight, table._capacity );
    auto outPairs  = cx.pairs.SliceSize( table._capacity );

    // Fp all groups in the table
    uint64 tableMatchCount = 0;

    for( uint32 i = 0; i < tableGroupCount; i++ )
    {
        uint64 matchCount = ForwardPropTableGroup<rTable>( cx, i, outPairs, yRight, metaRight );
// Log::Line( "[%u] : %llu", i, (llu)matchCount );

        if( matchCount == 0 )
            return ForwardPropResult::Failed;

// #if _DEBUG
//         {
//             for( uint64 j = 0; j < matchCount; j++ )
//             {
//                 Log::Line( "  [%u] L: %-8u | R: %u", j, outPairs[j].left, outPairs[j].right );
//             }
//         }
// #endif
        tableMatchCount += matchCount;
        outPairs  = outPairs .Slice( matchCount ); 
        yRight    = yRight   .Slice( matchCount );
        metaRight = metaRight.Slice( matchCount );
    }

    // if( cx.cudaThresher ) cx.cudaThresher->DumpTimings();

    const uint32 groupsToFlip = std::max( 1u, tableGroupCount / 2 );
    SortTableAndFlipBuffers<rTable>( cx, groupsToFlip );

    // The last table makes no group entries, it should simply have a single match
    // if( rTable == TableId::Table6 )
    //     return tableMatchCount == 1 ? ForwardPropResult::Success : ForwardPropResult::Failed;

    // When getting qualities, we may have a single match before the last table
    // if( returnSuccessOnSingleMatch && tableMatchCount == 1 )
    //     return ForwardPropResult::Success;

    // Check if we found a match already
    const bool hasProof = (returnSuccessOnSingleMatch && tableMatchCount == 1) || tableMatchCount == 2;
    // bool hasProof = true;

    // const uint32 groupsToCheck = tableGroupCount == 1 ? 1 : tableGroupCount / 2;
    // for( uint32 i = 0; i < groupsToCheck; i++ )
    // {
    //     if( table._groups[i].count != 2 )
    //     {
    //         hasProof = false;
    //         break;
    //     }
    // }
    #if SHOW_TIMINGS
        Log::Line( "FP table %u: %.3lf s", rTable+1, TimerEnd( timer ) );
    #endif

    return hasProof ? ForwardPropResult::Success : ForwardPropResult::Continue;
}

//-----------------------------------------------------------
bool ForwardPropCudaTables( GreenReaperContext& cx, bool returnSuccessOnSingleMatch )
{
    ASSERT( cx.cudaThresher );
    return false;

    // for( TableId rTableId = TableId::Table3; rTableId < TableId::Table7; rTableId++ )
    // {
    //     const uint32 groupCount  = 32 >> ((int)rTableId-1);
    //     const bool   isLastTable = rTableId == TableId::Table6;

    //     ProofTable& lTable = cx.tables[(int)rTableId-1];
    //     ProofTable& rTable = cx.tables[(int)rTableId];

    //     Span<Pair> outPairs = cx.pairs.SliceSize( rTable._capacity );

    //     uint32 tableMatchCount = 0;

    //     for( uint32 lGroup = 0; lGroup < groupCount; lGroup++ )
    //     {
    //         const uint32 rGroup = lGroup / 2;

    //         if( (lGroup & 1) == 0)
    //             rTable.BeginGroup( rGroup );

    //         uint32 matchCount = 0;
    //         const bool r = cx.cudaThresher->DecompressNextTableGroup(
    //             cx,
    //             outPairs,
    //             matchCount,
    //             lTable._groups[lGroup].offset
    //         );

    //         if( !r || matchCount < 1 )
    //             return false;


    //         rTable.AddGroupPairs( rGroup, matchCount );
    //         tableMatchCount += matchCount;

    //         outPairs = cx.pairs.SliceSize( rTable._capacity );
    //     }

    //     const bool hasProof = (returnSuccessOnSingleMatch && tableMatchCount == 1) || tableMatchCount == 2;
    //     return hasProof;
    // }
}

//-----------------------------------------------------------
bool ForwardPropTables( GreenReaperContext& cx )
{
    for( TableId rTable = TableId::Table3; rTable < TableId::Table7; rTable++ )
    {
        ForwardPropResult r;

        const uint32 groupCount  = 32 >> ((int)rTable-1);
        const bool   isLastTable = rTable == TableId::Table6;

        switch( rTable )
        {
            // case TableId::Table2: r = ForwardPropTable<TableId::Table2>( cx ); break;
            case TableId::Table3: r = ForwardPropTable<TableId::Table3>( cx, groupCount, isLastTable ); break;
            case TableId::Table4: r = ForwardPropTable<TableId::Table4>( cx, groupCount, isLastTable ); break;
            case TableId::Table5: r = ForwardPropTable<TableId::Table5>( cx, groupCount, isLastTable ); break;
            case TableId::Table6: r = ForwardPropTable<TableId::Table6>( cx, groupCount, isLastTable ); break;

            default:
                Log::Error( "[GR_ERROR] Unexpected table encountered %u", (uint32)rTable+1 );
                return false;
        }

        switch( r )
        {
            case ForwardPropResult::Success:
                BacktraceProof( cx, std::min( rTable, TableId::Table6 ), cx.proofContext.proof );
                return true;

            case ForwardPropResult::Continue:
                break;

            default:
                // ASSERT( 0 );
                return false;
        }   
    }

    return false;
}


///
/// Matching
///
//-----------------------------------------------------------
struct GRMatchJob : MTJob<GRMatchJob>
{
    const uint64* _yBuffer;
          uint64  _entryCount;
    const uint32* _groupIndices;
          uint32  _groupCount;
          Pair*   _tmpPairs;
          Pair*   _outPairs;
          uint64  _maxMatches;
          uint32  _pairOffset;
          std::atomic<uint64>* _totalMatchCount;

          uint32 _matchCount;

    virtual void Run() override;
};

//-----------------------------------------------------------
Span<Pair> Match( GreenReaperContext& cx, const Span<uint64> yEntries, Span<Pair> outputPairs, const uint32 pairOffset )
{
    ASSERT( yEntries.length <= 0xFFFFFFFF );
    ASSERT( cx.groupsBoundaries.length <= 0xFFFFFFFF );

    // Get the group boundaries and the adjusted thread count
    const uint64 groupCount = ScanBCGroupMT32( 
        *cx.pool,
        cx.config.threadCount,
        yEntries.Ptr(),
        (uint32)yEntries.Length(),
        cx.xBufferTmp.Ptr(),
        cx.groupsBoundaries.Ptr(),
        (uint32)cx.groupsBoundaries.Length()
    );

    ASSERT( cx.pairsTmp.Length() >= outputPairs.Length() );
    std::atomic<uint64> matchCount = 0;

    GRMatchJob job = {};
    job._yBuffer         = yEntries.Ptr();
    job._entryCount      = yEntries.Length();
    job._groupIndices    = cx.groupsBoundaries.Ptr();
    job._groupCount      = (uint32)groupCount;
    job._tmpPairs        = cx.pairsTmp.Ptr();
    job._outPairs        = outputPairs.Ptr();
    job._maxMatches      = outputPairs.Length();
    job._pairOffset      = pairOffset;
    job._totalMatchCount = &matchCount;
    job._matchCount      = 0;

    const uint32 matchThreadCount = std::min( cx.config.threadCount, (uint32)groupCount );
    MTJobRunner<GRMatchJob>::RunFromInstance( *cx.pool, matchThreadCount, job );

    return outputPairs.SliceSize( matchCount );
}

//-----------------------------------------------------------
static uint32 MatchJob(
    const Span<uint64> yEntries, 
    const uint32*      groupBoundaries,
    const uint32       groupCount,
          Span<Pair>   pairs, 
    const uint32       pairOffset,
    const uint32       id );

void GRMatchJob::Run()
{
    uint32 groupCount, offset, _;
    GetThreadOffsets( this, _groupCount, groupCount, offset, _ );

    uint64 maxMatches, matchOffset, __;
    GetThreadOffsets( this, _maxMatches, maxMatches, matchOffset, __ );

    auto tmpPairs = Span<Pair>( _tmpPairs + matchOffset, maxMatches );

    const uint32 matchCount = MatchJob( 
        Span<uint64>( (uint64*)_yBuffer, _entryCount ),
        _groupIndices + offset,
        groupCount,
        tmpPairs,
        _pairOffset,
        _jobId );

    _totalMatchCount->fetch_add( matchCount, std::memory_order_relaxed );

    _matchCount = matchCount;
    SyncThreads();

    uint64 copyOffset = 0;
    for( uint32 i = 0; i < _jobId; i++ )
        copyOffset += GetJob( i )._matchCount;

    bbmemcpy_t( _outPairs + copyOffset, tmpPairs.Ptr(), matchCount );
}

//-----------------------------------------------------------
uint32 MatchJob(
    const Span<uint64> yEntries, 
    const uint32*      groupBoundaries,
    const uint32       groupCount,
          Span<Pair>   pairs, 
    const uint32       pairOffset,
    const uint32       id )
{
    // const uint32 groupCount = (uint32)groupBoundaries.Length();
    const uint32 maxPairs = (uint32)pairs.Length();

    uint32 pairCount = 0;

    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    uint64 groupLStart = groupBoundaries[0];
    uint64 groupL      = yEntries[groupLStart] / kBC;

    for( uint32 i = 1; i <= groupCount; i++ )
    {
        const uint64 groupRStart = groupBoundaries[i];
        const uint64 groupR      = yEntries[groupRStart] / kBC;

        if( groupR - groupL == 1 )
        {
            // Groups are adjacent, calculate matches
            const uint16 parity           = groupL & 1;
            const uint64 groupREnd        = groupBoundaries[i+1];

            const uint64 groupLRangeStart = groupL * kBC;
            const uint64 groupRRangeStart = groupR * kBC;

            ASSERT( groupREnd - groupRStart <= 350 );
            ASSERT( groupLRangeStart == groupRRangeStart - kBC );

            // Prepare a map of range kBC to store which indices from groupR are used
            // For now just iterate our bGroup to find the pairs

            // #NOTE: memset(0) works faster on average than keeping a separate a clearing buffer
            memset( rMapCounts, 0, sizeof( rMapCounts ) );

            for( uint64 iR = groupRStart; iR < groupREnd; iR++ )
            {
                uint64 localRY = yEntries[iR] - groupRRangeStart;
                ASSERT( yEntries[iR] / kBC == groupR );

                if( rMapCounts[localRY] == 0 )
                    rMapIndices[localRY] = (uint16)( iR - groupRStart );

                rMapCounts[localRY] ++;
            }

            // For each group L entry
            for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
            {
                const uint64 yL     = yEntries[iL];
                const uint64 localL = yL - groupLRangeStart;

                for( int iK = 0; iK < kExtraBitsPow; iK++ )
                {
                    const uint16 targetR = L_targets[parity][localL][iK];

                    // for( uint j = 0; j < rMapCounts[targetR]; j++ )
                    if( rMapCounts[targetR] > 0 )
                    {
                        for( uint j = 0; j < rMapCounts[targetR]; j++ )
                        {
                            const uint64 iR = groupRStart + rMapIndices[targetR] + j;
                            ASSERT( iL < iR );

                            // Add a new pair
                            if( pairCount >= maxPairs )
                            {
                                // #TODO: Set error
                                ASSERT( 0 );
                                return pairCount;
                            }

                            Pair& pair = pairs[pairCount++];
                            pair.left  = (uint32)iL + pairOffset;
                            pair.right = (uint32)iR + pairOffset;
                        }
                    }
                }
            }
        }
        // Else: Not an adjacent group, skip to next one.

        // Go to next group
        groupL      = groupR;
        groupLStart = groupRStart;
    }

    return pairCount;
}


///
/// Fx
///
//-----------------------------------------------------------
template<TableId rTable, typename TMetaIn, typename TMetaOut>
void GenerateFxForPairs( GreenReaperContext& cx, const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> metaOut )
{
    ASSERT( yOut.Length() >= pairs.Length() );
    ASSERT( metaOut.Length() >= pairs.Length() );

    const uint32 threadCount = std::min( cx.config.threadCount, (uint32)pairs.Length() );

    if( threadCount == 1 )
    {
        GenerateFx<rTable, TMetaIn, TMetaOut>( pairs, yIn, metaIn, yOut, metaOut );
        return;
    }

    AnonMTJob::Run( *cx.pool, cx.config.threadCount, [&]( AnonMTJob* self ){
        
        uint64 count, offset, _;
        GetThreadOffsets( self, (uint64)pairs.Length(), count, offset, _ );
        
        GenerateFx<rTable, TMetaIn, TMetaOut>( pairs.Slice( offset, count ), yIn, metaIn, yOut.Slice( offset, count ), metaOut.Slice( offset, count ) );
    });
}

//-----------------------------------------------------------
template<TableId rTable, typename TMetaIn, typename TMetaOut>
inline void GenerateFx( const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> outMeta )
{
    constexpr size_t MetaInMulti  = TableMetaIn <rTable>::Multiplier;
    constexpr size_t MetaOutMulti = TableMetaOut<rTable>::Multiplier;
    static_assert( MetaInMulti != 0, "Invalid metaKMultiplier" );

    const uint32 k           = 32;
    const uint32 shiftBits   = MetaOutMulti == 0 ? 0 : kExtraBits;
    const uint32 ySize       = k + kExtraBits;         // = 38
    const uint32 yShift      = 64 - (k + shiftBits);   // = 26 or 32
    
    const size_t metaSize    = k * MetaInMulti;
    const size_t metaSizeLR  = metaSize * 2;
    const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );

    // Hashing
    uint64 input [5]; // y + L + R
    uint64 output[4]; // blake3 hashed output

    blake3_hasher hasher;
    static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

    for( uint64 i = 0; i < pairs.Length(); i++ )
    {
        const Pair   pair = pairs[i];
        const uint64 y    = yIn[pair.left];

        const TMetaIn metaL = metaIn[pair.left ];
        const TMetaIn metaR = metaIn[pair.right];

        TMetaOut& mOut = outMeta[i];

        if constexpr( MetaInMulti == 1 )
        {
            const uint64 l = metaL;
            const uint64 r = metaR;

            input[0] = Swap64( y << 26 | l >> 6  );
            input[1] = Swap64( l << 58 | r << 26 );

            // Metadata is just L + R of 8 bytes
            if constexpr( MetaOutMulti == 2 )
                mOut = l << 32 | r;
        }
        else if constexpr( MetaInMulti == 2 )
        {
            const uint64 l = metaL;
            const uint64 r = metaR;

            input[0] = Swap64( y << 26 | l >> 38 );
            input[1] = Swap64( l << 26 | r >> 38 );
            input[2] = Swap64( r << 26 );

            // Metadata is just L + R again of 16 bytes
            if constexpr( MetaOutMulti == 4 )
            {
                mOut.m0 = l;
                mOut.m1 = r;
            }
        }
        else if constexpr( MetaInMulti == 3 )
        {
            const uint64 l0 = metaL.m0;
            const uint64 l1 = metaL.m1 & 0xFFFFFFFF;
            const uint64 r0 = metaR.m0;
            const uint64 r1 = metaR.m1 & 0xFFFFFFFF;
        
            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 6  );
            input[2] = Swap64( l1 << 58 | r0 >> 6  );
            input[3] = Swap64( r0 << 58 | r1 << 26 );
        }
        else if constexpr( MetaInMulti == 4 )
        {
            const auto l = metaL;
            const auto r = metaR;

            input[0] = Swap64( y    << 26 | l.m0 >> 38 );
            input[1] = Swap64( l.m0 << 26 | l.m1 >> 38 );
            input[2] = Swap64( l.m1 << 26 | r.m0 >> 38 );
            input[3] = Swap64( r.m0 << 26 | r.m1 >> 38 );
            input[4] = Swap64( r.m1 << 26 );
        }

        // Hash the input
        blake3_hasher_init( &hasher );
        blake3_hasher_update( &hasher, input, bufferSize );
        blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

        const uint64 f = Swap64( *output ) >> yShift;
        yOut[i] = f;

        if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );

            mOut = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( MetaOutMulti == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            mOut.m0 = h0 << ySize | h1 >> 26;
            mOut.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
        }
        else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 ) // In = 2 is calculated above with L + R
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            mOut.m0 = h0 << ySize | h1 >> 26;
            mOut.m1 = h1 << 38    | h2 >> 26;
        }
    }
}
