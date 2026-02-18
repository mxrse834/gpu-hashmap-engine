#include <cstdint>
#ifndef HASHMAP
#define HASHMAP
#define TPB 256
#define BPG 140
// Forward declaration
typedef struct hashmap_engine
{
    uint32_t n = 10000000;
    uint32_t o_n = 100000;
    // int master_offset_current = 0;
    uint32_t master_byte_current = 0;
    uint8_t *master_bytes = NULL;
    // int *master_offset;
    uint32_t last_offset_val = 0;
    // these 2 store the complete string and the offsets to seperate its parts ( inclusive of all the elements in the current hash table)
    uint32_t *key;
    uint32_t *value;
    uint32_t *o_key;
    uint32_t *o_value;
} hashmap_engine;

//
////DELETE KERNEL -> delete_device
//
__global__ void delete_kernel(hashmap_engine *h,
                              uint8_t *qbytes,
                              uint32_t *qoffset,
                              uint32_t length_qoffset,
                              uint32_t length_qbytes);

//
////INSERT KERNEL -> insert_device
//
__global__ void insert_kernel(hashmap_engine *h,
                              uint32_t *words,
                              uint32_t *offset,
                              uint32_t *data,
                              uint32_t length_offset,
                              uint32_t length_bytes);

//
////LOOKUP KERNEL -> lookup_device
//
__global__ void lookup_kernel(hashmap_engine *h,
                              uint8_t *qbytes,
                              uint32_t *qoffset,
                              uint32_t length_qoffset,
                              uint32_t length_qbytes,
                              uint32_t *results);

#endif