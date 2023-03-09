#pragma once 
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#define GR_POST_PROOF_X_COUNT 64
#define GR_POST_PROOF_CMP_X_COUNT (GR_POST_PROOF_X_COUNT/2)

typedef int32_t grBool;

typedef struct GreenReaperContext GreenReaperContext;

typedef struct GreenReaperConfig
{
    uint32_t threadCount;
    uint32_t cpuOffset;
    grBool   disableCpuAffinity;

} GreenReaperConfig;

typedef enum GRResult
{
    GRResult_Failed       = 0,
    GRResult_OK           = 1,
    GRResult_OutOfMemory  = 2,
    GRResult_NoProof      = 3,  // A dropped proof due to line point compression

} GRResult;


// Timings expressed in nanoseconds
// typedef struct GRProofTimings
// {
//     uint64_t totalElapsedNS;
//     uint64_t f1ElapsedNS;
//     uint64_t sortElapsedNS;
//     uint64_t matchElapsedNS;
//     uint64_t fxElapsedNS;
// };

typedef struct GRCompressedProofRequest
{
    union {
        uint32_t compressedProof[GR_POST_PROOF_CMP_X_COUNT];   // Corresponds to the x buckets in line points form
        uint64_t fullProof      [GR_POST_PROOF_X_COUNT];
    };

          uint32_t  compressionLevel;
    const uint8_t*  plotId;

    // Pass a pointer to a timings struct if 
    // you'd like detailed timings output
    // GRProofTimings* outTimings;
          
} GRCompressedProofRequest;

typedef struct GRLinePoint
{
    uint64_t hi;  // High-order bytes
    uint64_t lo;  // Low-order bytes
} GRLinePoint;

typedef struct GRCompressedQualitiesRequest
{
    // Input
    const uint8_t*  plotId;
    const uint8_t*  challenge;
    uint32_t        compressionLevel;
    GRLinePoint     xLinePoints[2];     // Line points with compressed x's

    // Output
    uint64_t        x1, x2;             // Output x qualities

} GRCompressedQualitiesRequest;

///
/// API
///
GreenReaperContext* grCreateContext( GreenReaperConfig* config );
void                grDestroyContext( GreenReaperContext* context );

// Full proof of space request given a challenge
GRResult grFetchProofForChallenge( GreenReaperContext* context, GRCompressedProofRequest* req );

// Request plot qualities for a challenge
GRResult grGetFetchQualitiesXPair( GreenReaperContext* context, GRCompressedQualitiesRequest* req );

size_t grGetMemoryUsage( GreenReaperContext* context );


#ifdef __cplusplus
}
#endif