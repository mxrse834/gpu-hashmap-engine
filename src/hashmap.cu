#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// ==============================
// GPU HashMap Engine - Core File
// ==============================

// Kernel: Insert Key-Value Pair into Hash Table (out goal in this part is to generate a key for a specific value to get a key,value pair and store this in our dictionary)
__global__ void insert_kernel(int *keys, int *values, int *hash_table_keys, int *hash_table_values, int num_items) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int h_keys[] = {12, 44, 13, 88, 23, 94, 11, 39};  // sample keys
    int h_values[] = {100, 200, 300, 400, 500, 600, 700, 800};  // sample values
    int num_items = 8;
    if(tid<n)
    {int index[tid]=-1;//declare an array in global mem that can carry 8 elements initialize aal to -1
    if(index[tid]!=-1) 
    int index[tid]=h_keys[tid]%n;
    //okay now lets store the values we have in our keys lets copy to shared memory
    
}

// Kernel: Lookup Key in Hash Table
__global__ void lookup_kernel(int *keys, int *hash_table_keys, int *hash_table_values, int *results, int num_queries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Placeholder for lookup logic
}

// Kernel: Delete Key from Hash Table
__global__ void delete_kernel(int *keys, int *hash_table_keys, int *hash_table_values, int num_items) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Placeholder for delete logic
}

// Main Testing Function
int main() {
    std::cout << "GPU HashMap Engine - CUDA Kernel Skeleton Initialized" << std::endl;
    return 0;
}
