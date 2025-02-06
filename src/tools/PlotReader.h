#pragma once
// #include "plotting/PlotTools.h"
#include "plotting/FSETableGenerator.h"
#include "plotting/PlotTypes.h"
#include "plotting/PlotHeader.h"
#include "io/FileStream.h"
#include "util/Util.h"
#include <vector>
#include <memory>

class CPBitReader;

enum class ProofFetchResult
{
    OK = 0,
    NoProof,
    Error,
    CompressionError
};

// Base Abstract class for read-only plot files
class IPlotFile
{
public:
    inline uint K() const { return _header.k; }

    inline PlotFlags Flags() const 
    {
        if( _version < PlotVersion::v2_0 )
            return PlotFlags::None;

         return _header.flags;
    }

    inline uint CompressionLevel() const
    {
        if( _version < PlotVersion::v2_0 )
            return 0;

        return _header.compressionLevel;
    }

    inline const byte* PlotId() const { return _header.id; }

    inline uint PlotMemoSize() const { return _header.memoLength; }

    inline const byte* PlotMemo() const { return _header.memo; }

    inline uint64 TableAddress( PlotTable table ) const
    {
        ASSERT( table >= PlotTable::Table1 && table <= PlotTable::C3 );
        return _header.tablePtrs[(int)table];
    }

    inline size_t TableSize( PlotTable table )
    {
        ASSERT( table >= PlotTable::Table1 && table <= PlotTable::C3 );

        const uint64 address    = _header.tablePtrs[(int)table];
        uint64       endAddress = PlotSize();

        // Check all table entris where we find and address that is 
        // greater than ours and less than the current end address
        for( int i = 0; i < 10; i++ )
        {
            const uint64 a = _header.tablePtrs[i];
            if( a > address && a < endAddress )
                endAddress = a;
        }

        return (size_t)( endAddress - address );
    }

    // #NOTE: User must check for read errors!
    inline uint16 ReadUInt16()
    {
        uint16 value = 0;
        const ssize_t read = Read( sizeof( value ), &value );
        if( read < 0 )
            return 0;
        
        ASSERT( read == sizeof( uint16 ) );
        if( read != sizeof( uint16 ) )
        {
            // #TODO: Set proper error
            return 0;
        }

        return Swap16( value );
    }

    inline bool SeekToTable( PlotTable table )
    {
        return Seek( SeekOrigin::Begin, (int64)TableAddress( table ) );
    }

    inline PlotVersion Version() { return _version; }

    inline Span<uint64> TableSizes() { return Span<uint64>( _header.tableSizes, 10 ); }

    // Abstract Interface
public:
    virtual bool Open( const char* path ) = 0;
    virtual bool IsOpen() const = 0;

    // Plot size in bytes
    virtual size_t PlotSize() const = 0;
    
    // Read data from the plot file
    virtual ssize_t Read( size_t size, void* buffer ) = 0;

    // Seek to the specific location on the plot stream,
    // whatever the underlying implementation may be.
    virtual bool Seek( SeekOrigin origin, int64 offset ) = 0;

    // Get last error ocurred
    virtual int GetError() = 0;

protected:

    // Implementors can call this to load the header
    bool ReadHeader( int& error );

protected:
    PlotFileHeaderV2 _header;
    PlotVersion      _version;
};

class MemoryPlot : public IPlotFile
{
public:
    MemoryPlot() = default;
    MemoryPlot( const MemoryPlot& plotFile ) = default;
    MemoryPlot( MemoryPlot&& plotFile ) = default;
    ~MemoryPlot() = default;

    bool Open( const char* path ) override;
    bool IsOpen() const override;

    size_t PlotSize() const override;
    
    ssize_t Read( size_t size, void* buffer ) override;

    bool Seek( SeekOrigin origin, int64 offset ) override;

    int GetError() override;

private:

    std::shared_ptr<byte[]> _buffer;
    size_t      _size     = 0;
    int         _err      = 0;
    ssize_t     _position = 0;
    std::string _plotPath = "";
};

class FilePlot : public IPlotFile
{
public:
    FilePlot();
    FilePlot( const FilePlot& file );
    ~FilePlot();

    bool Open( const char* path ) override;
    bool IsOpen() const override;

    size_t PlotSize() const override;

    ssize_t Read( size_t size, void* buffer ) override;

    bool Seek( SeekOrigin origin, int64 offset ) override;

    int GetError() override;

private:
    FileStream  _file;
    std::string _plotPath = "";
};

class PlotReader
{
public:
    PlotReader( IPlotFile& plot );
    ~PlotReader();

    // Start reading a different plot
    void SetPlot( IPlotFile& plot );

    uint64 GetC3ParkCount() const;

    // Get the maximum potential F7 count.
    // This may be more than the actual number of F7s that we have,
    // since the last park is likely not full.
    uint64 GetMaxF7EntryCount() const;

    size_t GetTableParkCount( const PlotTable table ) const;

    uint64 GetMaximumC1Entries() const;

    // Returns the number of apparent C1 entries (same as C3 park count),
    // excluding any empty space appearing after the table for alignment,
    // This may not be accurate if the C1 entries are malformed and 
    // not sorted in ascending order.
    // This method performs various reads until it find a seemingly valid C1 entries
    bool GetActualC1EntryCount( uint64& outC1Count );

    // Read a whole C3 park into f7Buffer.
    // f7Buffer must hold at least as many as the amount of entries
    // required per C3Park. (kCheckpoint1Interval).
    // The return value should be the entries in the park.
    // It should never be 0.
    // If the return value is negative, there was an error reading the park.
    int64 ReadC3Park( uint64 parkIndex, uint64* f7Buffer );

    bool ReadP7Entries( uint64 parkIndex, uint64 p7ParkEntries[kEntriesPerPark] );

    bool ReadP7Entry( uint64 p7Index, uint64& outP7Entry );

    uint64 GetFullProofForF7Index( uint64 f7Index, byte* fullProof );

    ProofFetchResult FetchProof( uint64 t6LPIndex, uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] );

    // void   FindF7ParkIndices( uintt64 f7, std::vector<uint64> indices );
    bool ReadLPPark( TableId table, uint64 parkIndex, uint128 linePoints[kEntriesPerPark], uint64& outEntryCount );

    bool ReadLP( TableId table, uint64 index, uint128& outLinePoint );

    bool FetchProofFromP7Entry( uint64 p7Entry, uint64 proof[32] );

    inline IPlotFile& PlotFile() const { return _plot; }

    // Returns the number of f7s found and the starting index as outStartT6Index.
    uint64 GetP7IndicesForF7( const uint64 f7, uint64& outStartT6Index );

    ProofFetchResult FetchQualityXsForP7Entry( uint64 t6Index,const byte challenge[BB_CHIA_CHALLENGE_SIZE], uint64& outX1, uint64& outX2 );
    ProofFetchResult FetchQualityForP7Entry( uint64 t6Index, const byte challenge[BB_CHIA_CHALLENGE_SIZE], byte outQuality[BB_CHIA_QUALITY_SIZE] );

    TableId           GetLowestStoredTable() const;
    bool              IsCompressedXTable( TableId table ) const;
    size_t            GetParkSizeForTable( TableId table ) const;
    uint32            GetLPStubBitSize( TableId table ) const;
    uint32            GetLPStubByteSize( TableId table ) const;
    size_t            GetParkDeltasSectionMaxSize( TableId table ) const;
    const FSE_DTable* GetDTableForTable( TableId table ) const;

    // Takes ownership of a decompression context
    void AssignDecompressionContext( struct GreenReaperContext* context );

    void ConfigDecompressor( uint32 threadCount, bool disableCPUAffinity, uint32 cpuOffset = 0, bool useGpu = false, int gpuIndex = -1 );

    inline void ConfigGpuDecompressor( uint32 threadCount, bool disableCPUAffinity, uint32 cpuOffset = 0 )
    {
        ConfigDecompressor( threadCount, disableCPUAffinity, cpuOffset, true );
    }

    inline struct GreenReaperContext* GetDecompressorContext() const { return _grContext; }

private:
    ProofFetchResult DecompressProof( const uint64 compressedProof[BB_PLOT_PROOF_X_COUNT], uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] );

    bool ReadLPParkComponents( TableId table, uint64 parkIndex, 
                               CPBitReader& outStubs, byte*& outDeltas, 
                               uint128& outBaseLinePoint, uint64& outDeltaCounts );

    bool LoadP7Park( uint64 parkIndex );

    bool LoadC2Entries();

    struct GreenReaperContext* GetGRContext();

private:
    IPlotFile& _plot;
    uint32     _version;

    // size_t  _parkBufferSize;
    uint64*      _parkBuffer;           // Buffer for loading compressed park data.
    byte*        _deltasBuffer;         // Buffer for decompressing deltas in parks that have delta. 

    byte*        _c1Buffer = nullptr;
    Span<uint64> _c2Entries;
    Span<uint64> _c3Buffer;

    struct GreenReaperContext* _grContext     = nullptr;    // Used for decompressing
    bool                       _ownsGrContext = true;

    int64  _park7Index = -1;
    uint64 _park7Entries[kEntriesPerPark];
};

