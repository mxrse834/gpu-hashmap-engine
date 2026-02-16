// #include <thrust/device_vector.h>
#include <iostream>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include "hash.cuh"
#define TPB 1024 /// threads per block 256
#define BPG 35   /// blocks per grid 140
using namespace std;
namespace cg = cooperative_groups;

typedef struct hashmap_engine
{
    uint32_t n = 10000000;
    uint32_t o_n = 100000;
    uint32_t master_byte_current = 0;
    uint8_t *master_bytes = NULL;
    uint32_t last_offset_val = 0;
    uint32_t *key = NULL;
    uint32_t *value = NULL;
    uint32_t *o_key = NULL;
    uint32_t *o_value = NULL;
} hashmap_engine;

////INSERT DEVICE (PER THREAD)
// Kernel: Insert Key-Value Pair into Hash Table (our goal in this part is to generate a key for a specific value to get a key,value pair and store this in our "dictionary")
__device__ void insert_device(
    hashmap_engine *h,
    uint32_t *words,
    uint32_t lov,
    uint32_t mbc,
    uint32_t *offset,
    uint32_t *data,
    uint32_t length_offset,
    uint32_t length_bytes)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    // implementation of buffering into shared memory to deal with waves efficiently
    // inour implementation we have 1 wave = 256*160 = 40960threads( maximum number of threads that can exist on the 40 SM's of 2070 super)
    // each SM can schedule a maximum of 64 warps at a time, and ttrue hardware parallelism is only 4 warp schedulers selecting 4 warps to run at a given time
    // theres zero cost abstraction as the warps are already resident on the gpu
    // above is a example of our current gpu(check README)
    // shared memory calculation per wave ->
    // 1) 4 threads map to one shared memory string therefore 1 offset
    // 2) therefore we have 256/4= 64 offsets per block each uint32_t therefore 256 bytes per block (there is a hrdware limit of 48kb(configurable upto 64kb) which we are well within)
    // 3) total shared memory used(for 1 wave is) is 40960 bytes which is 40kb
    //    (Dont misinterpret the wording, since shared memory is unique to each block we will have 264(bytes per block) * 4(blocks per sm)= about 1KB
    // 4) trying to understand hardware limitation->
    //    in this  case we have 4 blocks that will be in a SM at any point in time each SM has its own shared memory limited to exactly 48(or 64)
    //    so we have 48/4=12 kb for each block and each block requires 256 b to store offsets + 8 bytes for storing master_offset_current and master_byte_current
    //    in total we are using 1kb(for calc look at 3)) out of 12288(12kb).
    // 5) SETUP FOR SINGLE BUFFERING

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(words); // ensure 4 byte padding
    // uint32_t *words = reinterpret_cast<uint32_t *>(byte);
    /*uint32_t *master_vals = (uint32_t *)shmem;
    master_vals[0] = master_byte_currentv;
    master_vals[1] = master_offset_currentv;
    master_byte_current = master_vals[0];
    master_offset_current = master_vals[1];
    // uint8_t *bytes = (uint8_t *)(master_vals + 2);
    uint32_t *words = (uint32_t *)(master_vals + 2);
    uint32_t *offset = (uint32_t *)(words + (length_bytes / 4));*/
    for (uint32_t wid = tid / 4; wid < length_offset; wid += (gridDim.x * blockDim.x) / 4) /// REMEmBER WID IS outn indexing method
    {
        uint32_t start = offset[wid];
        uint32_t len;
        if (wid < length_offset - 1)
            len = offset[wid + 1] - start;
        else
            len = length_bytes - start;
        uint32_t posn = (start / 4) + tile.thread_rank();

        uint32_t results_h1;
        uint32_t results_h2;
        uint32_t results_h3;
        // calling the custom hashing function from the included hash.cuh file
        xh332(
            bytes,
            tid, start, len, posn, wid,
            offset, words,
            results_h1,
            results_h2,
            results_h3,
            length_bytes,
            length_offset);

        // Reduce raw 32-bit hashes into table index range
        results_h1 = results_h1 % h->n;
        results_h2 = results_h2 % h->n;
        results_h3 = results_h3 % h->n;

        // each thread takes up one string to copy into the master bytes

        // uint32_t editaT = offset[wid];
        // uint32_t tile_no = tile.thread_rank();
        int i = tile.thread_rank();
        for (i = i << 2; i + 4 <= len; i += tile.size() << 2)
        {
            *(uint32_t *)(h->master_bytes + mbc + start + i) = *(uint32_t *)(bytes + start + i);
            // the unfortunate method we must use of tricking the compiler into type_casting then dereferencing
        }
        // for trailing bytes(ie <4 (non ints))
        uint32_t tail = (len & ~3); // last 4 byte "copyable" part
        if (tail + tile.thread_rank() < len)
            h->master_bytes[mbc + start + tail + tile.thread_rank()] = bytes[start + tail + tile.thread_rank()]; // CODE FOR COPYING INTO MASTERBYTE

        //(each wid : one offset)
        /*if (tile_no == 0)
            master_offset[master_offset_current + wid + 1] = master_offset[master_offset_current] + editaT;*/

        tile.sync(); // upto this point we have hashed the input value with 3 different seeds stored in 3 different arrays results_h1,h2,h3
        /*pif (tid == 0)
        {
            master_offset_current += length_offset;
            master_byte_current += length_bytes;
        }
        tile.sync();*/
        if (tile.thread_rank() == 0)
        {
            if (h->key[results_h1] == 0xFFFFFFFF)
            {
                if (atomicCAS(&h->key[results_h1], 0xFFFFFFFF, lov + offset[wid]) == 0xFFFFFFFF)
                {
                    h->value[results_h1] = data[wid];
                    continue;
                }
            }

            if (h->key[results_h2] == 0xFFFFFFFF)
            {

                if (atomicCAS(&h->key[results_h2], 0xFFFFFFFF, lov + offset[wid]) == 0xFFFFFFFF)
                {
                    h->value[results_h2] = data[wid];
                    continue;
                }
            }
            if (h->key[results_h3] == 0xFFFFFFFF)
            {
                if (atomicCAS(&h->key[results_h3], 0xFFFFFFFF, lov + offset[wid]) == 0xFFFFFFFF)
                {
                    h->value[results_h3] = data[wid];
                    continue;
                }
            }
            // upto this point we have the key,value pairs inserted into the hashmap most likely
            // (unless there is collision even after the third hash is calculcated so now we go for probing/overflow buffer)
            uint32_t overflow_slot = ((results_h1 ^ results_h2 ^ results_h3) * 0x9e3779b9) % h->o_n;
            for (int i = 0; i < h->o_n; i++)
            {
                int idx = (overflow_slot + i) % h->o_n; //  the for loop over "i" allows for linear probing
                if (atomicCAS(&h->o_key[idx], 0xFFFFFFFF, lov + offset[wid]) == 0xFFFFFFFF)
                {
                    h->o_value[idx] = data[wid];
                    continue;
                }
            }
        }
        /*if (tid == 0)
                  {
                      MASTERBYTES.insert(MASTERBYTES.end(), bytes, bytes + sizeof(bytes) / sizeof(bytes[0]));
                      MASTEROFFSET.insert(MASTEROFFSET.end(), offset, offset + sizeof(offset) / sizeof(offset[0]));
                  }*/
    }
    if (tid == 1)
    {
        h->last_offset_val += offset[length_offset]; // length_offset is a cumulative aggregation
        h->master_byte_current += length_bytes;      // length_bytes is not its simply a counter
    }
}

// Kernel: Lookup Key in Hash Table
__device__ void lookup_device(hashmap_engine *h,
                              uint8_t *qbytes,
                              uint32_t *qoffset,
                              uint32_t length_qoffset,
                              uint32_t length_qbytes,
                              uint32_t *results)
{
    // we need to first apply the 3 hash functions on our input , then we must must calc address ( check overflow buffer also)
    // and probe until we find it , now we return the data associated with it if found
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    // cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    uint32_t *qwords = reinterpret_cast<uint32_t *>(qbytes);
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint wid = tid / 4; wid < length_qoffset; wid += (blockDim.x * gridDim.x) / 4)
    {
        tile.sync();
        uint32_t start = qoffset[wid];
        uint32_t len;
        if (wid < length_qoffset - 1)
            len = qoffset[wid + 1] - start;
        else
            len = length_qbytes - start;
        uint32_t posn = (start / 4) + tile.thread_rank();
        uint32_t results_h1;
        uint32_t results_h2;
        uint32_t results_h3;
        uint32_t os1;
        uint32_t os2;
        uint32_t os3;
        xh332(
            qbytes,
            tid, start, len, posn, wid,
            qoffset, qwords,
            results_h1,
            results_h2,
            results_h3,
            length_qbytes,
            length_qoffset);
        // Ensure hash indices are within primary table bounds
        results_h1 = results_h1 % h->n;
        results_h2 = results_h2 % h->n;
        results_h3 = results_h3 % h->n;
        // now we have each thread_rank==0 holding three variables namely results_h1,results_h2,results_h3
        // once lookup kernel finds the hashed value(ie the address), we check the offset for the master byte array stored there , then we go to that offset in master and check if both are equal
        if (tile.thread_rank() == 0)
        {
            os1 = h->key[results_h1];
            os2 = h->key[results_h2];
            os3 = h->key[results_h3];
        }
        os1 = tile.shfl(os1, 0); // now all threads hold the value of offset (given to us by our first hash)( ie all threads in a tile will hold the same offset value to probe at)
        if (os1 == 0xFFFFFFFF)
        {
            if (tile.thread_rank() == 0)
            {
                results[wid] = 0xFFFFFFFF;
            }
            continue;
        }
        bool failed = false;
        /*
        int i = tile.thread_rank();
        for (i = i << 2; i + 4 <= len; i += tile.size() << 2)
        {
            *(uint32_t *)(h->master_bytes + h->master_byte_current + start + i) = *(uint32_t *)(bytes + start + i);
            // the unfortunate trick we must use of tricking the compiler into type_casting then dereferencing
        }
        // for trailing bytes(ie <4 (non ints))
        uint32_t tail = (len & ~3); // last 4 byte "copyable" part
        if (tail + tile.thread_rank() < len)
            h->master_bytes[h->master_byte_current + start + tail + tile.thread_rank()] = bytes[start + tail + tile.thread_rank()]; // CODE FOR COPYING INTO MASTERBYTE
        */

        //// here our 1st loop compares in groups of 16 bytes( or maybe more realistically MULTIPLES of 4 (since each thread does 4 bytes and threads are concurrent))
        //// so here we can ensure we will only have at max 3 trailing bytes ledt that may not be comapred
        //// conveniently(not so much as intentionally :)) we have exactly 4 threads ina tile so each can do a byte by byte comp in a single iteration to satisfy a maximum case of 3)

        int i = tile.thread_rank();
        for (i = i << 2; i + 4 <= len; i += tile.size() << 2)
        {
            if (*(uint32_t *)(h->master_bytes + os1 + i) != *(uint32_t *)(qbytes + qoffset[wid] + i))
            {
                failed = true;
            }
            if (tile.any(failed))
                break;
        }
        // for trailing bytes(ie <4 (non ints))
        uint32_t tail = (len & ~3); // last 4 byte "copyable" part
        if (tail + tile.thread_rank() < len)
        {
            if (h->master_bytes[os1 + tail + thread_rank()] != qbytes[qoffset[wid] + tail + thread_rank()]) // CODE FOR COPYING INTO MASTERBYTE
            {
                failed = true;
            }
        }
        // we cannot return threads until all the for loops have completed for a given string
        failed = tile.any(failed);
        if (!failed)
        {
            if (tile.thread_rank() == 0)
                results[wid] = h->value[results_h1];
            continue;
        }
        ///// essentially what were doing here is returning all the threads from the function that have found the key value pair from the first hash function itself
        /// when threads are returned the entire tile of threads will be returned as if one thread fails the full tile has failed)
        // all the threads that fail the first function ie the key,value pair is not foud move to the 2nd for loop to probe the next hash function given location

        os2 = tile.shfl(os2, 0);
        failed = false;

        i = tile.thread_rank();
        for (i = i << 2; i + 4 <= len; i += tile.size() << 2)
        {
            if (*(uint32_t *)(h->master_bytes + os2 + i) != *(uint32_t *)(qbytes + qoffset[wid] + i))
            {
                failed = true;
            }
            if (tile.any(failed))
                break;
        }
        // for trailing bytes(ie <4 (non ints))
        tail = (len & ~3); // last 4 byte "copyable" part
        if (tail + tile.thread_rank() < len)
        {
            if (h->master_bytes[os2 + tail + thread_rank()] != qbytes[qoffset[wid] + tail + thread_rank()]) // CODE FOR COPYING INTO MASTERBYTE
            {
                failed = true;
            }
        }
        // we cannot return threads until all the for loops have completed for a given string
        failed = tile.any(failed);
        if (!failed)
        {
            if (tile.thread_rank() == 0)
                results[wid] = h->value[results_h2];
            continue;
        }

    //////3RD HASH
    ///// essentially what were doing here is returning all the threads from the function that have found the key value pair from the first hash function itself
    /// when threads are returned the entire tile of threads will be returned as if one thread fails the full tile has failed)
    // all the threads that fail the first function ie the key,value pair is not foud move to the 2nd for loop to probe the next hash function given location
    try_hash3:
        os3 = tile.shfl(os3, 0);
        failed = false;
        i = tile.thread_rank();
        for (i = i << 2; i + 4 <= len; i += tile.size() << 2)
        {
            if (*(uint32_t *)(h->master_bytes + os3 + i) != *(uint32_t *)(qbytes + qoffset[wid] + i))
            {
                failed = true;
            }
            if (tile.any(failed))
                break;
        }
        // for trailing bytes(ie <4 (non ints))
        tail = (len & ~3); // last 4 byte "copyable" part
        if (tail + tile.thread_rank() < len)
        {
            if (h->master_bytes[os3 + tail + thread_rank()] != qbytes[qoffset[wid] + tail + thread_rank()]) // CODE FOR COPYING INTO MASTERBYTE
            {
                failed = true;
            }
        }
        // we cannot return threads until all the for loops have completed for a given string
        failed = tile.any(failed);
        if (!failed)
        {
            if (tile.thread_rank() == 0)
                results[wid] = h->value[results_h3];
            continue;
        }

        ////////////////////////////***OVERFLOW LOGIC***
        //
        ////// on going through all these three hash funcitons our required kay , value pair is still not found we finally move on to our overflow hash table
        ////// optimization: 1) if empty slot is encountered STOP check 2)ballot to check if found for early exit
        //
        /////////////CCCONTINUE FROM  HERE ////////////////////
        uint32_t osb;
        uint32_t overflow_slot;
        if (tile.thread_rank() == 0)
        {
            overflow_slot = ((results_h1 ^ results_h2 ^ results_h3) * 0x9e3779b9) % h->o_n; // this will give the address in the overflow hash table
        }
        // osb = tile.shfl(osb, 0); /// osb is the base offset value in the real master bytes array per tile
        overflow_slot = tile.shfl(overflow_slot, 0);
        for (int l = 0; l < 50; l++)
        {
            failed = false;
            if (tile.thread_rank() == 0)
            {
                osb = h->o_key[(overflow_slot + l) % h->o_n];
            }
            osb = tile.shfl(osb, 0);
            if (osb == -1)
                break;
            for (int i = tile.thread_rank(); i < len; i += tile.size())
            {
                if (h->master_bytes[osb + i] != qbytes[qoffset[wid] + i])
                {
                    failed = true;
                    break;
                }
            }
            failed = tile.any(failed);
            if (!failed)
            {
                if (tile.thread_rank() == 0)
                    results[wid] = h->o_value[(overflow_slot + l) % h->o_n];
                break;
            }
        }
        /*
       for (int l = 0; l < 50; l++)
       {
           uint32_t osb;
           if (tile.thread_rank() == 0)
           {
               osb = o_key[(overflow_slot + l) % o_n];
           }
           osb = tile.shfl(osb, 0);

           if (osb == (uint32_t)-1)
               return;

           bool failed = false;
           for (uint32_t i = tile.thread_rank(); i < len; i += tile.size())
           {
               if (master_bytes[osb + i] != qbytes[qoffset[wid] + i])
               {
                   failed = true;
               }
               if(tile.any(failed))
               {
                   break;
               }
           }

           if (!tile.any(failed))
           { // ← ONLY CORRECT CHECK
               if (tile.thread_rank() == 0)
                   results[wid] = o_value[(overflow_slot + l) % o_n];
               return;
           }
       }
       if (tile.thread_rank() == 0)
           results[wid] = -1;*/
    }
}

// Kernel: Delete Key from Hash Table
__device__ void delete_device(hashmap_engine *h, uint8_t *qbytes,
                              uint32_t *qoffset,
                              uint32_t length_qoffset,
                              uint32_t length_qbytes)
{

    // Placeholder for delete logic
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    uint32_t *qwords = reinterpret_cast<uint32_t *>(qbytes);
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint wid = tid / 4; wid < length_qoffset; wid += (blockDim.x * gridDim.x) / 4)
    {
        uint32_t start = qoffset[wid];
        uint32_t len = qoffset[wid + 1] - start;
        uint32_t posn = (start / 4) + tile.thread_rank();
        uint32_t results_h1;
        uint32_t results_h2;
        uint32_t results_h3;
        uint32_t os1;
        uint32_t os2;
        uint32_t os3;
        xh332(
            qbytes,
            tid, start, len, posn, wid,
            qoffset, qwords,
            results_h1,
            results_h2,
            results_h3,
            length_qbytes,
            length_qoffset);
        /// up till this poitn we have the posn of the values we need int the hashmap table in results_h1 , h2 and h3 per variablle only in lane 0
        if (tile.thread_rank() == 0)
        {
            os1 = h->key[results_h1];
            os2 = h->key[results_h2];
            os3 = h->key[results_h3];
        }

        // 1ST HASH
        //
        bool failed;
        os1 = tile.shfl(os1, 0); // putting in the values of os1 in all 4 threads of a tile
        failed = false;
        for (int i = tile.thread_rank(); i < len; i += tile.size())
        {
            if (tile.any(h->master_bytes[os1 + i] != qbytes[qoffset[wid] + i]))
            {
                failed = true; /// sets any and all tiles in whihc even one tile doesnt have a single matching byte check to true( ie failed)
            }
        }
        if (tile.thread_rank() == 0 && !failed)
        {
            h->key[results_h1] = -1;
        }

        // 2ND HASH
        //
        os2 = tile.shfl(os2, 0); // putting in the values of os1 in all 4 threads of a tile
        failed = false;
        for (int i = tile.thread_rank(); i < len; i += tile.size())
        {
            if (tile.any(h->master_bytes[os2 + i] != qbytes[qoffset[wid] + i]))
            {
                failed = true; /// sets any and all tiles in whihc even one tile doesnt have a single matching byte check to true( ie failed)
            }
        }
        if (tile.thread_rank() == 0 && !failed)
        {
            h->key[results_h2] = -1;
        }

        // 3RD HASH
        //
        os3 = tile.shfl(os3, 0); // putting in the values of os1 in all 4 threads of a tile
        failed = false;
        for (int i = tile.thread_rank(); i < len; i += tile.size())
        {
            if (tile.any(h->master_bytes[os3 + i] != qbytes[qoffset[wid] + i]))
            {
                failed = true; /// sets any and all tiles in whihc even one tile doesnt have a single matching byte check to true( ie failed)
            }
        }
        if (tile.thread_rank() == 0 && !failed)
        {
            h->key[results_h3] = -1;
        }
        //
        ///////CODE FOR CHECKING OVERFLOW HASHMAP
        //
        // if all 3 xxhash searches fail we do the following:
        //
        uint32_t osb;
        uint32_t overflow_slot;
        if (tile.thread_rank() == 0)
        {
            overflow_slot = ((results_h1 ^ results_h2 ^ results_h3) * 0x9e3779b9) % h->o_n; // this will give the address in the overflow hash table
        }
        // osb = tile.shfl(osb, 0); /// osb is the base offset value in the real master bytes array per tile
        overflow_slot = tile.shfl(overflow_slot, 0);
        // int failed;
        for (int l = 0; l < 50; l++)
        {
            failed = false;
            if (tile.thread_rank() == 0)
            {
                osb = h->o_key[(overflow_slot + l) % h->o_n];
            }
            osb = tile.shfl(osb, 0);
            if (osb == -1)
                break;
            for (int i = tile.thread_rank(); i < len; i += tile.size())
            {
                uint32_t mask = tile.ballot(h->master_bytes[osb + i] == qbytes[qoffset[wid] + i]);
                if (mask != 0xF) // will check if mask is set ie if all 4 threads in a tile finding matching bytes
                {
                    failed = true;
                    break;
                }
                /*if (i == len - 1)
                {
                    if (tile.thread_rank() == 0)
                        o_key[(overflow_slot + l) % o_n] = -1;
                    return;
                }*/
            }
            if (!failed)
            {
                if (tile.thread_rank() == 0)
                    h->o_key[(overflow_slot + l) % h->o_n] = -1;
                break;
            }
        }
    }
}

//
////LOOKUP KERNEL -> lookup_device
//
__global__ void lookup_kernel(hashmap_engine *h,
                              uint8_t *qbytes,
                              uint32_t *qoffset,
                              uint32_t length_qoffset,
                              uint32_t length_qbytes,
                              uint32_t *results)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    uint32_t *qwords = reinterpret_cast<uint32_t *>(qbytes);
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t wid = tid / 4;
    if (tid <= length_qoffset) ////////////////////////COMEPLETELY CUSTOM SETUP FOR APPLICATION 1
        qoffset[tid] = tid * 16;
    lookup_device(h, qbytes, qoffset, length_qoffset, length_qbytes, results);
}

//
////INSERT KERNEL -> insert_device
//
__global__ void insert_kernel(hashmap_engine *h,
                              uint32_t *words,
                              uint32_t *offset,
                              uint32_t *data,
                              uint32_t length_offset,
                              uint32_t length_bytes)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    // uint8_t *bytes = reinterpret_cast<uint8_t *>(words);
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ uint32_t lov;
    __shared__ uint32_t mbc;
    // uint32_t wid = tid / 4;
    if (tid <= length_offset) ////////////////////////COMEPLETELY CUSTOM SETUP FOR APPLICATION 1
        offset[tid] = tid * 16;
    if (threadIdx.x == 1)
    {
        mbc = h->master_byte_current;
        lov = h->last_offset_val;
    }
    __syncthreads();
    insert_device(h, words, lov, mbc, offset, data, length_offset, length_bytes);
}

//
////DELETE KERNEL -> delete_device
//
__global__ void delete_kernel(hashmap_engine *h,
                              uint8_t *qbytes,
                              uint32_t *qoffset,
                              uint32_t length_qoffset,
                              uint32_t length_qbytes)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    uint32_t *qwords = reinterpret_cast<uint32_t *>(qbytes);
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t wid = tid / 4;
    if (tid <= length_qoffset) ////////////////////////COMEPLETELY CUSTOM SETUP FOR APPLICATION 1
        qoffset[tid] = tid * 16;
    delete_device(h, qbytes, qoffset, length_qoffset, length_qbytes);
}
