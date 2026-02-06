#include <cstdint>
#ifndef HASH
#define HASH

// if linker errors are persistent here in xh332 try :
// 1)making it inline
// 2)forcing it inline :)

__device__ void xh332(
    uint8_t *bytes,
    uint32_t tid,
    uint32_t start,
    uint32_t len,
    uint32_t posn,
    uint32_t wid,
    uint32_t *offset,
    uint32_t *words,
    uint32_t &res1,
    uint32_t &res2,
    uint32_t &res3,
    uint32_t length_bytes,
    uint32_t length_offset);

#endif