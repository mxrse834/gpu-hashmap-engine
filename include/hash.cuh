#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

// Computes three independently seeded XXH32 hashes for one byte string.
// Exactly four threads in the caller's cooperative-group tile participate.
// The final values are defined in tile lane 0 and are broadcast by callers.
__device__ void hash3_xxh32(const uint8_t *bytes,
                            uint32_t start,
                            uint32_t length,
                            uint32_t *hash1,
                            uint32_t *hash2,
                            uint32_t *hash3);
