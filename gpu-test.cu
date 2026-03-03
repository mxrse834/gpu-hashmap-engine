#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::cout << "CUDA devices: " << count << std::endl;
}
