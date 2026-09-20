#pragma once

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                                  \
    do                                                                                    \
    {                                                                                     \
        const cudaError_t cuda_status_ = (call);                                           \
        if (cuda_status_ != cudaSuccess)                                                   \
        {                                                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(cuda_status_));                                     \
            exit(2);                                                                       \
        }                                                                                 \
    } while (0)

static inline void check_kernel(const char *kernel_name)
{
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
    {
        fprintf(stderr, "%s launch failed: %s\n", kernel_name,
                cudaGetErrorString(launch_status));
        exit(2);
    }

    const cudaError_t runtime_status = cudaDeviceSynchronize();
    if (runtime_status != cudaSuccess)
    {
        fprintf(stderr, "%s execution failed: %s\n", kernel_name,
                cudaGetErrorString(runtime_status));
        exit(2);
    }
}
