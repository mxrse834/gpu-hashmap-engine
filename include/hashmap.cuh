#include <cstdint>
#ifndef HASHMAP
#define HASHMAP

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
                              uint32_t length_bytes,
                              uint32_t last_offset_val,
                              uint32_t master_byte_current);

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