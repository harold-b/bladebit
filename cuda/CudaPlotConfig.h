#pragma once

#define BBCU_GPU_STREAM_COUNT         4
#define BBCU_GPU_BUFFER_MAX_COUNT     4
#define BBCU_DEFAULT_GPU_BUFFER_COUNT 2

#define BBCU_K                          (32u)
#define BBCU_BUCKET_COUNT               (256u)
#define BBC_Y_BITS                      (BBCU_K+kExtraBits)
#define BBC_Y_BITS_T7                   (BBCU_K)
#define BBC_BUCKET_BITS                 (CuBBLog2( BBCU_BUCKET_COUNT ))
#define BBC_BUCKET_SHIFT                (BBC_Y_BITS-BBC_BUCKET_BITS)
#define BBC_BUCKET_SHIFT_T7             (BBC_Y_BITS_T7-BBC_BUCKET_BITS)
#define BBC_Y_MASK                      ((uint32)((1ull << BBC_Y_BITS) - 1))
#define BBC_Y_MASK_T7                   (0xFFFFFFFFu)
#define BBC_BUCKET_MASK( bucket )       ( ((uint64)bucket) << BBC_BUCKET_SHIFT )


#define BBCU_TABLE_ENTRY_COUNT          (1ull<<32)
#define BBCU_BUCKET_ENTRY_COUNT         (BBCU_TABLE_ENTRY_COUNT/BBCU_BUCKET_COUNT)
//#define BBCU_XTRA_ENTRIES_PER_SLICE     (1024u*64u)
#define BBCU_XTRA_ENTRIES_PER_SLICE     (4096+1024)
#define BBCU_MAX_SLICE_ENTRY_COUNT      ((BBCU_BUCKET_ENTRY_COUNT/BBCU_BUCKET_COUNT)+BBCU_XTRA_ENTRIES_PER_SLICE)
#define BBCU_BUCKET_ALLOC_ENTRY_COUNT   (BBCU_MAX_SLICE_ENTRY_COUNT*BBCU_BUCKET_COUNT)
#define BBCU_TABLE_ALLOC_ENTRY_COUNT    (((uint64)BBCU_BUCKET_ALLOC_ENTRY_COUNT)*BBCU_BUCKET_COUNT)

// The host always needs to start slices at the meta4 size, to avoid overwriting by subsequent tables
#define BBCU_HOST_META_MULTIPLIER       (4ull)
#define BBCU_META_SLICE_ENTRY_COUNT     (BBCU_MAX_SLICE_ENTRY_COUNT*BBCU_HOST_META_MULTIPLIER)
#define BBCU_META_BUCKET_ENTRY_COUNT    (BBCU_BUCKET_ALLOC_ENTRY_COUNT*BBCU_HOST_META_MULTIPLIER)

#define BBCU_SCAN_GROUP_THREAD_COUNT       128
#define BBCU_SCAN_GROUP_ENTRIES_PER_THREAD 512

static constexpr uint32 CU_MAX_BC_GROUP_BOUNDARIES = ( BBCU_BUCKET_ENTRY_COUNT / 210 ); // Should be enough for all threads


static_assert( BBCU_BUCKET_ALLOC_ENTRY_COUNT / BBCU_BUCKET_COUNT == BBCU_MAX_SLICE_ENTRY_COUNT );

#if _DEBUG

    #ifdef _WIN32
        #define DBG_BBCU_DBG_DIR "D:/dbg/cuda/"
    #else
        #define DBG_BBCU_DBG_DIR "/home/harold/plotdisk/dbg/cuda/"
        // #define DBG_BBCU_DBG_DIR "/home/harito/plots/dbg/cuda/"
    #endif
    // #define DBG_BBCU_REF_DIR       "/home/harold/plots/ref/"


    // #define BBCU_DBG_SKIP_PHASE_1   1   // Skip phase 1 and load pairs from disk
    // #define BBCU_DBG_SKIP_PHASE_2   1   // Skip phase 1 and 2 and load pairs and marks from disk

    #if (defined( BBCU_DBG_SKIP_PHASE_2 ) && !defined( BBCU_DBG_SKIP_PHASE_1 ) )
        #define BBCU_DBG_SKIP_PHASE_1 1
    #endif

    // #define DBG_BBCU_P1_WRITE_CONTEXT 1
    // #define DBG_BBCU_P1_WRITE_PAIRS   1
    // #define DBG_BBCU_P2_WRITE_MARKS   1

    // #define DBG_BBCU_P2_COUNT_PRUNED_ENTRIES 1
    // #define DBG_BBCU_KEEP_TEMP_FILES 1


    #define _ASSERT_DOES_NOT_OVERLAP( b0, b1, size ) ASSERT( (b1+size) <= b0 || b1 >= (b0+size) )
    #define ASSERT_DOES_NOT_OVERLAP( b0, b1, size ) _ASSERT_DOES_NOT_OVERLAP( ((byte*)b0), ((byte*)b1), (size) )

    #define _ASSERT_DOES_NOT_OVERLAP2( b0, b1, sz0, sz1 )ASSERT( (b1+sz1) <= b0 || b1 >= (b0+sz0) )
    #define ASSERT_DOES_NOT_OVERLAP2( b0, b1, size0, size1 ) _ASSERT_DOES_NOT_OVERLAP2( ((byte*)b0), ((byte*)b1), (size0), (size1) )

#else

    #define _ASSERT_DOES_NOT_OVERLAP( b0, b1, size ) 
    #define ASSERT_DOES_NOT_OVERLAP( b0, b1, size ) 
    #define _ASSERT_DOES_NOT_OVERLAP2( b0, b1, sz0, sz1 ) 
    #define ASSERT_DOES_NOT_OVERLAP2( b0, b1, size0, size1 ) 
#endif
