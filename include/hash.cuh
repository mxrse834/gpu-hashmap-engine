#include<cstdint>
#ifndef HASH_H
#define HASH_H


__device__ void xh332(
    uint8_t *bytes,
    uint32_t tid, 
    uint32_t wid,
    uint32_t *offset,
    uint32_t *words, 
    uint32_t res1,
    uint32_t res2,
    uint32_t res3,
    uint32_t length_bytes,
    uint32_t length_offset
);

extern uint32_t start;
extern uint32_t len;
extern uint32_t posn;


#endif