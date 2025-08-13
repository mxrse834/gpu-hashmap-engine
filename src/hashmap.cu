//cudaMemset(IMP
//cudaMemset use dedicated copy engines (also called DMA engines).)

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// ==============================
// GPU HashMap Engine - Core File
// ==============================

// Kernel: Insert Key-Value Pair into Hash Table (out goal in this part is to generate a key for a specific value to get a key,value pair and store this in our dictionary)
__global__ void insert_kernel(int n, int *h_keys, int *h_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if(tid<n)
    {h_keys[tid]=-1;//declare an array in global mem that can carry 8 elements initialize arr to -1
    int i=0;
    int ptv;
    int base=h_values[tid]%n;
    do
    { if(i<n)
      {
        int idx = (base+i)%n;
        ptv=atomicCAS(&h_keys[idx],-1,h_values[tid]); //will lock the the position address given 
        //and allow only one thread to execute at a a time to prevent any race condition
        //ptv launches a per thread variable and in atomicCAS it will store the old value in index[tid] that was replaced 
      i++;}
      else 
      break;
    }
    while( ptv!=-1);
    //okay now lets store the values we have in our keys lets copy to shared memory
    
}
}

// Kernel: Lookup Key in Hash Table
__global__ void lookup_kernel(int *keys, int *hash_table_keys, int *hash_table_values, int *results, int num_queries) 
{
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
    /*int h_values[] = {10, 20, 30, 40, 50, 60, 70, 80};  // sample values*/
    /*int n=8;*/
    return 0;

    int *g_a;
    size_t size=n*sizeof(int);
    cudaMalloc(&g_a,size);
    cudaMemset(g_a,0xFF,size);
}
