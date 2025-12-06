// small frequently used hashmaps entries we place in shared memory manually

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <hash.cuh>
#define TPB 1024
using namespace std;

// ==============================
// GPU HashMap Engine - Core File
// ==============================

/*uint8_t *bytes,
    uint32_t *offset,
    uint32_t *results_h1,
    uint32_t *results_h2,
    uint32_t *results_h3*/
/*struct BurstMetadata
{
    uint8_t *flat_bytes; // Which batch
    uint32_t *offset;     // Where in that batch
    uint32_t length;     // How long
};*/

struct hashmap_engine
{
    int n = 4294967296;
    int o_n = 10000;
    vector<uint8_t> MASTERBYTES;
    vector<uint32_t> MASTEROFFSET;

    // these 2 store the complete string and the offsets to seperate its parts ( inclusive of all the elements in the current hash table)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = tid / 4;
    int *key;
    int *value;
    int *o_key;
    int *o_value;

    // lets say n is size of our hash table , lets maintain a healthy
    // Kernel: Insert Key-Value Pair into Hash Table (out goal in this part is to generate a key for a specific value to get a key,value pair and store this in our dictionary)
    __device__ void insert_kernel(uint8_t *bytes, uint32_t *offset, uint32_t *data)
    {
        if (tid < sizeof(bytes) / sizeof(uint32_t))
        { /// h_keys[tid]=-1;//declare an array in global mem that can carry 8 elements initialize arr to -1
            uint32_t results_h1;
            uint32_t results_h2;
            uint32_t results_h3;
            uint32_t *words = reinterpret_cast<uint32_t *>(bytes);

            xh332(
                bytes,
                tid, wid,
                offset, words,
                results_h1,
                results_h2,
                results_h3);
            // upto this point we have hashed the input value with 3 different seeds stored in 3 different arrays results_h1,h2,h3
            if (tid % 4 == 0)
            {
                if (atomicCAS(&key[results_h1], -1, words[wid]) == -1)
                {
                    value[results_h1] = data[wid];
                    return;
                }

                if (atomicCAS(&key[results_h2], -1, words[wid]) == -1)
                {
                    value[results_h2] = data[wid];
                    return;
                }
                if (atomicCAS(&key[results_h3], -1, words[wid]) == -1)
                {
                    value[results_h3] = data[wid];
                    return;
                }
                // upto this point we have the key,value pairs inserted into the hashmap most likely (unless there is collision even after the third hash is calculcated so now we go for probing/overflow buffer)
                uint32_t overflow_slot = ((results_h1 ^ results_h2 ^ results_h3) * 0x9e3779b9) % o_n;
                for (int i = 0; i < o_n; i++)
                {
                    int idx = (overflow_slot + i) % o_n; //  the for loop over "i" allows for linear probing
                    if (atomicCAS(&o_key[idx], -1, words[wid]) == -1)
                    {
                        o_value[wid] = data[wid];
                        return;
                    }
                }
                if (tid == 0)
                {
                    MASTERBYTES.insert(MASTERBYTES.end(), bytes, bytes + sizeof(bytes) / sizeof(bytes[0]));
                    MASTEROFFSET.insert(MASTEROFFSET.end(), offset, offset + sizeof(offset) / sizeof(offset[0]));
                }
                /*do
                { if(i<n)
                  {
                    int idx = (base+i)%n;
                    ptv=atomicCAS(&h_keys[idx],-1,h_values[tid]); //will lock the the position address given
                    //and allow only one thread to execute at a a time to prevent any race condition
                    //ptv launches a per thread variable and in atomicCAS it will store the old value in index[tid] that was replaced
                 if(ptv==-1)
                 return;
                 i++;}
                  else
                 return;
                }
                while(i<n);*/
                // same code as the above for loop less modular and easy to understand
                // okay now lets store the values we have in our keys lets copy to shared memory
            }
        }
    }

    // Kernel: Lookup Key in Hash Table
    __device__ void lookup_kernel(int n)
    {
        // so the lookup kenerl with take some value and use the hashing function (%) on it and return a a place value where the lement may be stored or maybe have to linearly probe further
        if (tid < l)
        {
            int base = vals[tid] % n; // thi will give the ele location without collision
            for (int i = 0; i < n; i++)
            {
                if (vals[tid] == h_keys[(base + i) % n])
                {
                    locn[tid] = (base + i) % n;
                    return;
                }
            }
        }
    }
    // Kernel: Delete Key from Hash Table
    __device__ void delete_kernel(int *keys, int *hash_table_keys, int *hash_table_values, int num_items)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // Placeholder for delete logic
    }
};

__global__ void hash_kernel(hashmap_engine h) {

};

// Main Testing Function
int main()
{
    std::cout << "GPU HashMap Engine - CUDA Kernel Skeleton Initialized" << std::endl;
    int h[] = {10, 20, 30, 40, 50, 60, 70, 80}; // sample values*/
    int n = 8;
    string x;
    vector<int> temp;
    cout << "Type in the elements u want to search for, when ur done type in 'eol' standing for end of list";
    int i = 0;
    while (cin >> x && x != "eol")
    {
        try
        {
            int number = stoi(x);
            temp.push_back(number);
        }
        catch (const invalid_argument &e)
        {
            cerr << "Error is:" << e.what() << '\n';
        }
        i++;
    }
    int *g_k, *g_v, *locn, *g_temp;
    int *output = new int[i];
    int *h_temp = new int[i];
    size_t size = n * sizeof(int);
    size_t size1 = i * sizeof(int);
    cudaMalloc(&g_k, size);
    cudaMalloc(&g_v, size);
    cudaMemset(g_k, 0xFF, size);
    cudaMalloc(&locn, size1);
    cudaMemset(locn, 0xFF, size1);
    cudaMalloc(&g_temp, size1);
    cudaMemcpy(g_v, h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_temp, temp.data(), size1, cudaMemcpyHostToDevice);
    // cudaMemset(locn,0xFF,size1);
    dim3 block(TPB);
    dim3 grid((TPB + n - 1) / TPB);
    dim3 grid1((TPB + i - 1) / TPB);
    // insert_kernel<<<grid,block>>>(n,g_k,g_v);
    cudaDeviceSynchronize();
    // lookup_kernel<<<grid1,block>>>(n,i,g_temp,locn,g_k);
    cudaDeviceSynchronize();
    cudaMemcpy(output, locn, size1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_temp, g_temp, size1, cudaMemcpyDeviceToHost);
    for (int l = 0; l < i; l++)
    {
        if (output[l] == -1)
            cout << "The element " << h_temp[l] << " is not a part of the hahs table";
        else
            cout << h_temp[l] << " was found at " << output[l];
    }
    cudaFree(g_k);
    cudaFree(g_v);
    cudaFree(locn);
    cudaFree(g_temp);
    delete[] output;
    return 0;
}
