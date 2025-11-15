#pragma once
#include<cstdint>


__global__ void xh332(
    uint8_t *bytes, 
    uint32_t *offset, 
    uint32_t *results_h1,
    uint32_t *results_h2,
    uint32_t *results_h3
);