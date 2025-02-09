#pragma once

// Determined from the average match count at compression level 11, which was 0.288% of the bucket count * 2.
// We round up to a reasonable percentage of 0.5%.
// #NOTE: This must be modified for higher compression levels.
static constexpr double GR_MAX_MATCHES_MULTIPLIER         = 0.005;
static constexpr double GR_MAX_MATCHES_MULTIPLIER_2T_DROP = 0.018;  // For C9+
static constexpr uint32 GR_MAX_BUCKETS                    = 32;
static constexpr uint64 GR_MIN_TABLE_PAIRS                = 1024;

struct CudaThresherConfig
{
    uint deviceId;
    
};


inline uint64 GetEntriesPerBucketForCompressionLevel( const uint32 k, const uint32 cLevel )
{
    const uint32 entryBits = 17u - cLevel;
    const uint32 bucketBits = k - entryBits;
    const uint64 bucketEntryCount = 1ull << bucketBits;

    return bucketEntryCount;
}

inline uint64 GetMaxTablePairsForCompressionLevel( const uint32 k, const uint32 cLevel )
{
    const double factor = cLevel >= 40 ? GR_MAX_MATCHES_MULTIPLIER_2T_DROP : GR_MAX_MATCHES_MULTIPLIER;
    return (uint64)( GetEntriesPerBucketForCompressionLevel( k, cLevel ) * factor ) * (uint64)GR_MAX_BUCKETS;
}