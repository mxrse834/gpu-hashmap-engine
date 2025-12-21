// small frequently used hashmaps entries we place in shared memory manually
// #include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <hash.cuh>
#define TPB 256
#define BPG 140
using namespace std;
namespace cg = cooperative_groups;

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
    int master_offset_current = 0;
    int master_byte_current = 0;
    int *master_bytes;
    int *master_offset;
    // these 2 store the complete string and the offsets to seperate its parts ( inclusive of all the elements in the current hash table)
    int *key;
    int *value;
    int *o_key;
    int *o_value;

    // lets say n is size of our hash table , lets maintain a healthy
    // Kernel: Insert Key-Value Pair into Hash Table (out goal in this part is to generate a key for a specific value to get a key,value pair and store this in our dictionary)
    __device__ void insert_kernel(uint32_t *words,
                                  uint32_t *offset,
                                  uint32_t *data,
                                  uint32_t length_offset,
                                  uint32_t length_bytes,
                                  uint32_t master_byte_current,
                                  uint32_t master_offset_current)
    // void *shmem)
    {
        cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
        // implementation of double buffering into shared memory to deal with waves efficiently
        // inour implementation we have 1 wave = 256*160 = 40960threads( maximum number of threads that can exist on the 40 SM's of 2070 super)
        // above is a example of our current gpu
        // shared memory calculation per wave ->
        // 1) 4 threads map to one shared memory string therefore 1 offset
        // 2) therefore we have 256/4= 64 offsets per block each uint32_t therefore 256 bytes per block (there is a hrdware limit of 48kb(configurable upto 64kb) which we are well within)
        // 3) total shared memory used(for 1 wave is) is 40960 bytes which is 40kb
        // 4) trying to understand hardware limitation->
        //    in this  case we have 4 blocks that will be in a SM at any point in time each SM has its own shared memory limited to exactly 48(or 64)
        //    so we have 48/4=12 kb for each block and each block requires 256 b to store offsets + 8 bytes for storing master_offset_current and master_byte_current
        //    in total we are using 264 bytes out of 12288(12kb).
        // 5) SETUP FOR SINGLE BUFFERING
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint8_t *bytes = reinterpret_cast<uint8_t *>(words);

        // uint32_t *words = reinterpret_cast<uint32_t *>(byte);
        /*uint32_t *master_vals = (uint32_t *)shmem;
        master_vals[0] = master_byte_currentv;
        master_vals[1] = master_offset_currentv;
        master_byte_current = master_vals[0];
        master_offset_current = master_vals[1];
        // uint8_t *bytes = (uint8_t *)(master_vals + 2);
        uint32_t *words = (uint32_t *)(master_vals + 2);
        uint32_t *offset = (uint32_t *)(words + (length_bytes / 4));*/
        for (uint32_t wid = tid / 4; wid < length_offset; wid += (gridDim.x * blockDim.x) / 4)
        {

            uint32_t results_h1;
            uint32_t results_h2;
            uint32_t results_h3;
            // calling the custom hashing function from the included hash.cuh file
            xh332(
                bytes,
                tid, wid,
                offset, words,
                results_h1,
                results_h2,
                results_h3,
                length_bytes,
                length_offset);

            // each thread takes up one string to copy into the master bytes

            if (wid < length_offset)
            {
                uint32_t editaT = offset[wid];
                uint32_t tile_no = tile.thread_rank();
                for (int i = tile_no; i < len; i = i + 4)
                {
                    master_bytes[master_byte_current + editaT + i + 1] = bytes[start + i]; // CODE FOR COPYING INTO MASTERBYTE
                }
                //(each wid : one offset)
                if (tile_no == 0)
                    master_offset[master_offset_current + wid + 1] = master_offset[master_offset_current] + editaT;
            }
            tile.sync(); // upto this point we have hashed the input value with 3 different seeds stored in 3 different arrays results_h1,h2,h3
            if (tid == 0)
            {
                master_offset_current += length_offset;
                master_byte_current += length_bytes;
            }
            tile.sync();
            if (tile.thread_rank() == 0)
            {
                if (atomicCAS(&key[results_h1], -1, master_offset[wid]) == -1)
                {
                    value[results_h1] = data[wid];
                    return;
                }

                if (atomicCAS(&key[results_h2], -1, master_offset[wid]) == -1)
                {
                    value[results_h2] = data[wid];
                    return;
                }
                if (atomicCAS(&key[results_h3], -1, master_offset[wid]) == -1)
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
                        o_value[idx] = data[wid];
                        return;
                    }
                }
            }
            
            /*if (tid == 0)
                      {
                          MASTERBYTES.insert(MASTERBYTES.end(), bytes, bytes + sizeof(bytes) / sizeof(bytes[0]));
                          MASTEROFFSET.insert(MASTEROFFSET.end(), offset, offset + sizeof(offset) / sizeof(offset[0]));
                      }*/
            // start,posn,len,bytes,offset,
            //  on my gpu vram = 8 gb keeping a 2 gb overhead we can store about 6.4 bil bytes
        }
    }

    // Kernel: Lookup Key in Hash Table
    __device__ void lookup_kernel(uint8_t *qbytes,
                                  uint32_t *qoffset,
                                  uint32_t length_qoffset,
                                  uint32_t length_qbytes,
                                  uint32_t *results)
    {
        // we need to first apply the 3 hash functions on our input , then we must must calc address ( check overflow buffer also)
        // and probe until we find it , now we return the data associated with it if found
        cg::thread_block_tile<4> tile= cg::tiled_partition<4>(cg::this_thread_block());
        uint32_t *qwords = reinterpret_cast<uint32_t *>(qbytes);
        uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        for (uint wid = tid / 4; wid < length_qoffset; wid += blockDim.x * gridDim.x)
        {
            uint32_t results_h1;
            uint32_t results_h2;
            uint32_t results_h3;
            uint32_t os;
            xh332(
                qbytes,
                tid, wid,
                qoffset, qwords,
                results_h1,
                results_h2,
                results_h3,
                length_qbytes,
                length_qoffset);
            //now we have each thread_rank==0 holding three variables namely results_h1,results_h2,results_h3
            //once lookup kernel finds the hashed value(ie the address), we check the offset for the master byte array stored there , then we go to that offset in master and check if both are equal
            if(tile.thread_rank()==0)
            {
                os=master_offset[results_h1]; //now all threads that hacve id 0 in the tile hold the offsets that may hold the right string 
            }
            tile.shfl(os,0);
            for(uint32_t i = tile.thread_rank();i<master_offset[results_h1+1]||i<length_qoffset;i+=4)
            {
                if(tile.all(master_bytes[master_offset[os]+i]==qbytes[qoffset[wid]+i]))
                {}
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
