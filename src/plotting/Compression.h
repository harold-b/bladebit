#pragma once
#include "fse/fse.h"

struct CompressionInfo
{
    uint32_t entrySizeBits;
    uint32_t stubSizeBits;
    size_t   tableParkSize;
    double   ansRValue;
};

// #TODO: Change this API to C, and move it to GreenReaper.h
// #TODO: Rename to GetCompressionCTable/DTable
// #TODO: Add to a namespace
FSE_CTable*     CreateCompressionCTable( const uint32_t compressionLevel, size_t* outTableSize = nullptr );
FSE_DTable*     CreateCompressionDTable( const uint32_t compressionLevel, size_t* outTableSize = nullptr );
CompressionInfo GetCompressionInfoForLevel( const uint32_t compressionLevel );
uint32_t        GetCompressedLPBitCount( const uint32_t compressionLevel );
size_t          GetLargestCompressedParkSize();

template<uint32_t level>
struct CompressionLevelInfo 
{
    // static_assert( false, "Invalid compression level." );
};

template<>
struct CompressionLevelInfo<1>
{
    static constexpr uint32_t ENTRY_SIZE      = 16;
    static constexpr uint32_t STUB_BIT_SIZE   = 29;
    static constexpr size_t TABLE_PARK_SIZE = 8336;
    static constexpr double ANS_R_VALUE     = 2.51;
};

template<>
struct CompressionLevelInfo<2>
{
    static constexpr uint32_t ENTRY_SIZE      = 15;
    static constexpr uint32_t STUB_BIT_SIZE   = 25;
    static constexpr size_t   TABLE_PARK_SIZE = 7360;
    static constexpr double   ANS_R_VALUE     = 3.44;
};

template<>
struct CompressionLevelInfo<3>
{
    static constexpr uint32_t ENTRY_SIZE      = 14;
    static constexpr uint32_t STUB_BIT_SIZE   = 21;
    static constexpr size_t   TABLE_PARK_SIZE = 6352;
    static constexpr double   ANS_R_VALUE     = 4.36;
};

template<>
struct CompressionLevelInfo<4>
{
    static constexpr uint32_t ENTRY_SIZE      = 13;
    static constexpr uint32_t STUB_BIT_SIZE   = 16;
    static constexpr size_t   TABLE_PARK_SIZE = 5325; //5312;
    static constexpr double   ANS_R_VALUE     = 9.30;
};

template<>
struct CompressionLevelInfo<5>
{
    static constexpr uint32_t ENTRY_SIZE      = 12;
    static constexpr uint32_t STUB_BIT_SIZE   = 12;
    static constexpr size_t   TABLE_PARK_SIZE = 4300; //4290;
    static constexpr double   ANS_R_VALUE     = 9.30;
};

template<>
struct CompressionLevelInfo<6>
{
    static constexpr uint32_t ENTRY_SIZE      = 11;
    static constexpr uint32_t STUB_BIT_SIZE   = 8;
    static constexpr size_t   TABLE_PARK_SIZE = 3273; //3263;
    static constexpr double   ANS_R_VALUE     = 9.10;
};

template<>
struct CompressionLevelInfo<7>
{
    static constexpr uint32_t ENTRY_SIZE      = 10;
    static constexpr uint32_t STUB_BIT_SIZE   = 4;
    static constexpr size_t   TABLE_PARK_SIZE = 2250; //2240;
    static constexpr double   ANS_R_VALUE     = 8.60;
};

// #TODO: These are dummy values for now... Update with real values
template<>
struct CompressionLevelInfo<8>
{
    static constexpr uint32_t ENTRY_SIZE      = 9;
    static constexpr uint32_t STUB_BIT_SIZE   = 4;
    static constexpr size_t   TABLE_PARK_SIZE = 6350; //2240;
    static constexpr double   ANS_R_VALUE     = 8.60;
};

template<>
struct CompressionLevelInfo<9>
{
    static constexpr uint32_t ENTRY_SIZE      = 8;
    static constexpr uint32_t STUB_BIT_SIZE   = 30;
    static constexpr size_t   TABLE_PARK_SIZE = 8808;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<10>
{
    static constexpr uint32_t ENTRY_SIZE      = 7;
    static constexpr uint32_t STUB_BIT_SIZE   = 26;
    static constexpr size_t   TABLE_PARK_SIZE = 7896;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<11>
{
    static constexpr uint32_t ENTRY_SIZE      = 6;
    static constexpr uint32_t STUB_BIT_SIZE   = 22;
    static constexpr size_t   TABLE_PARK_SIZE = 6930;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<12>
{
    static constexpr uint32_t ENTRY_SIZE      = 5;
    static constexpr uint32_t STUB_BIT_SIZE   = 18;
    static constexpr size_t   TABLE_PARK_SIZE = 5953;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<13>
{
    static constexpr uint32_t ENTRY_SIZE      = 4;
    static constexpr uint32_t STUB_BIT_SIZE   = 14;
    static constexpr size_t   TABLE_PARK_SIZE = 4956;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<14>
{
    static constexpr uint32_t ENTRY_SIZE      = 3;
    static constexpr uint32_t STUB_BIT_SIZE   = 10;
    static constexpr size_t   TABLE_PARK_SIZE = 3944;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<15>
{
    static constexpr uint32_t ENTRY_SIZE      = 2;
    static constexpr uint32_t STUB_BIT_SIZE   = 6;
    static constexpr size_t   TABLE_PARK_SIZE = 2930;
    static constexpr double   ANS_R_VALUE     = 4.54;
};

template<>
struct CompressionLevelInfo<16>
{
    static constexpr uint32_t ENTRY_SIZE      = 1;
    static constexpr uint32_t STUB_BIT_SIZE   = 2;
    static constexpr size_t   TABLE_PARK_SIZE = 1913;
    static constexpr double   ANS_R_VALUE     = 4.54;
};