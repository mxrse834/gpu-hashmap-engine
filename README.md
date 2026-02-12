# GPU HashMap Engine  
/*okay now ill  try to fix the errros u pointed out, lets see what im able to do . all the problems i dont list here assume that ive triedmy best to fix them , so just debug and see if ive gona abt it right or not , other i specify here are some i have doubts in pls clarify those then ill try to fix them . 1.4) this shudnt be a problem since im allocating in the init function even if its garbage for now , 1.5)  doesnt seem to be a problem im setting it in find_file(),1.7)check now ,2.2)ignore for now 3.1)YES i definitely want to add this stuff ,give me a elborate start on debugging in cuda (more liek error checking),3.2)done ,3.3)check this again now ,3.4)check again now ,4.1)ignore for now 4.2)yes thats my bad (just somne micro optimztion tho) 4.3)yes fragile but okay , 5.1)now? ,5.2,5.4)leave for now , 5.5) explain in detail what u think the problem is i dont see anythign off :( , 5.6)is there anyway to do like a whichever is firest to edit it shud edit and other can just not do it ( liek fastest wins)(not atomic in atomic everyone gest to make a change here i want only first)*/

/home/pnglinkpc/cuda/gpu-hashmap-engine/testfile1.bin
/home/pnglinkpc/cuda/gpu-hashmap-engine/testfile2.bin

OVERVIEW:
- Batch Insertions & Lookups of key-value pairs in parallel relying majorly on GPU

BUILD INSTRUCTIONS:


###PERMISSIBLE WARNINGS (per se, no really per se) :
1) warning #68-D: integer conversion resulted in a change of sign
                      if (atomicCAS(&o_key[idx], -1, words[wid]) == -1) ------> -1 is rounded to 2^32 -1 (made unsigned) as CAS only takes +ve values
2) src/hashmap.cu(288): warning #68-D: integer conversion resulted in a change of sign
                  if (osb == -1) --------------> same as above 


### APPLICATIONS AIMED TO ADD 
1) under development : FILE duplication checker 
2) general structural hashmap
3) -


## Hardware Constraints
Currently using a RTX 2070 Super for all benchmarks and as a baseline, please take performance limitations into consideration.
In particular: TPB and BPG definitions @src/hashmap.cu are as per my gpu (can change as per urs)

To maximize efficiency, im focusing on optimizing **4 core metrics**:

1. **Hashing Function** - Generate IDs as unique as possible with minimal collisions
2. **Index Calculation/Hash Table** - Efficient parallel data structure for storage and retrieval

3. **Collision Handling** - System to resolve hash conflicts efficiently (currently we are expermienting on a  variation of cuckoo hashing utlizing 3 different SEEDS

4. **Dynamic Resizing** - Automatically scale storage as key-value pairs increase   // NOT IMPLEMENTED YET WERE LIMITED TO four billion two hundred ninety-four million nine hundred sixty-seven thousand two hundred ninety-two (have fun reading that(sorry , here you go - 4294967292))


WHAT WE WANT TO ACHIVE AT THE END:

some things identified 
unlike cpus its very expensive to linearly probe on a gpu cuz the entire warp ahs to wait owing to SIMT

What the “ideal” GPU Hashmap might look like

Hash function: Multiply-shift, cheap & well-mixed.

Memory: Keys[] + Values[] arrays in global memory, aligned & padded.

Probe strategy: Warp-cooperative linear probing with block size = 32 slots (fits warp).

Atomics: Warp-wide atomicCAS batches.

Shared memory caching for hot regions.

Overflow stash for extreme collisions.

Bulk operations: Designed to insert/look up batches of keys at once (not single-key API).


>>for now we have decided on 3 diff strategies - namely one byte one string (CPU type comp), 2nd one is 4 bytes one string , third is is for very long strings  
  alternate thought - tweak this idea and launch only 2 threads (or add 2 kernel launch options in each one for greater than 16 byyte and for lesser this would avoid excessive branching and wasteful checks ?)


// STRATEGY 2
// 1 block 4096 bytes can be handled considerign att 1024 threads in a lbock are launched(we have 4 bytes being dealt with per thread)
// total bytes dealt with are 4096*n where n is no of blocks
// we have to employ grid stride loops
/*__global__ void xxhash(uint8_t *bytes, uint32_t *offset, uint32_t *cr)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint32_t *words = reinterpret_cast<const uint32_t *>(bytes);

    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x; // llest assume the launch odf 1024 threads per block
    // uint32_t warp_id = tid / warpSize;
    // uint32_t lane_id = tid % warpSize;
    //uint32_t lid = tid % 4;
    uint32_t wid = tid / 4;
    // NOTE:while shared memory may be fast it still is slower than the L1 cache and since data is consecutive it will most definitely hit L1,
    // i.e. we can conclude that shared mem only proveides a real advantage when teh same daata is used a high number of times

    if (wid >= 630)
        return; // hard coded boundary check defined by us to deal with no of strings = to no of offsets

    uint32_t start = offset[wid];
    uint32_t len = offset[wid + 1] - start;
    uint32_t posn = (start / 4) + tile.thread_rank();

    // int stride = blockDim.x * blockIdx.x;
    uint32_t mask = (len >= 16);
    uint32_t v[4];
    v[tile.thread_rank()] = g[tile.thread_rank()]; // initializing all 4 accumulators with their respective constants
    uint32_t res = (1 - mask) * (SEED + g[6]);
    uint32_t fin;
    uint32_t i = 0;
    // while(tile.all(i<len))
    for (; i + 16 <= len; i += 16)                                         // 0,16,32....so on
    {                                                                      // we are processign in grps of 16 so this is a check to make sure its within sentence range
        v[tile.thread_rank()] = round(v[tile.thread_rank()], words[posn]); /// given that a sentence(string) has more than 16 bytes this part will run and load in grps of 4
        posn += 4;
    }


    /*Step 3: Merge accumulators
    if length >= 16:
        acc = rotate_left(v1, 1) +
              rotate_left(v2, 7) +
              rotate_left(v3, 12) +
              rotate_left(v4, 18)
    */
    if (tile.any(mask) && tile.thread_rank() == 0)
    {
        res = inst(v[0], 1);
        res += tile.shfl(inst(v[1], 7), 1);
        res += tile.shfl(inst(v[2], 12), 2);
        res += tile.shfl(inst(v[3], 18), 3);
    }
    // now we can safely say we have all tiles having a particular value of res in their first thread which will hold
    // either the PRIME5*SEED value (if its length is less than 16 bytes long) or the accumulated value we have until step 3

    // this will read consecutive 4 bytes together that is its been converted from a char type array to a int type one
    // include all threads upto the closest multiple to 4 ( here all tid upto 16 )
    /*if(lane_id<8)
    {   v1=SEED + g[0] + g[1];
        v1+=words[(lane_id)*4]*g[1];
        v1+=inst(v1,13);
        v1*=g[0];
    }
    else if(lane_id<16)
    {
        v2=SEED + g[1];
        v2+=words[((lane_id-8)*4)+1]*g[1];
        v2+=inst(v2,13);
        v2*=g[0];
    }
    else if(lane_id<24)
    {
        v3=SEED;
        v3+=words[((lane_id-16)*4)+2]*g[1];
        v3+=inst(v3,13);
        v3*=g[0];
    }
    else if (lane_id<32)
    {
        v4=SEED-g[0];
        v4+=words[((lane_id-8)*4)+3]*g[1];
        v4+=inst(v4,13);
        v4*=g[0];
    }*/
    /*int n=length/16;
    if(length>16)
    {if(wid==0)
    {v[lid]=g[lid]; //were assuming seed=0 in the constants only
    for(int i=0;i<n;i+=4)
     {
     v[lid]+=words[lid+i]*g[1];    //{(0x9E3779B1+0x85EBCA77),0x85EBCA77,0,-0x9E3779B1,0xC2B2AE3D,0x27D4EB2F,0x165667B1};
     v[lid]+=inst(v[lid],13);
     v[lid]*=g[0];
     }
    }
    }
    if(lane_id==0)
    {res = inst(v[0],1);
     res+= __shfl_down_sync(0xffffffff,inst(v[1],7),1);
     res+= __shfl_down_sync(0Xffffffff,inst(v[2],12),2);
     res+= __shfl_down_sync(0Xffffffff,inst(v[3],18),3);
    }
    //now v1 of all first 8 threads stores the final accumulated values of all 8 words
    */

    /*Step 4: Process remaining bytes (<16)
    acc = acc + length

    while 4 bytes remain:
        k1 = word
        k1 = k1 * PRIME3
        k1 = rotate_left(k1, 17)
        k1 = k1 * PRIME4
        acc = acc ^ k1
        acc = rotate_left(acc, 17) * PRIME1 + PRIME4

    while 1 byte remains:
        k1 = byte * PRIME5
        acc = acc ^ k1
        acc = rotate_left(acc, 11) * PRIME1
        */
    if (tile.thread_rank() == 0)
    {
        res += len;
        uint32_t processed = (len / 16) * 16; // Bytes already processed
        uint32_t k1;
        uint32_t i = processed;
        for (i; (i + 4) <= len; i += 4)
        {
            k1 = words[(start + i)/4]; // will load consecutive bytes into con threads
            k1 *= g[4];
            k1 = inst(k1, 17);
            k1 *= g[5];
            res ^= k1;
            res = inst(res, 17) * -g[3] + g[5];
        }
        // i = (4 * i) - 4;
        while (i < len)
        {
            /*for(int offset= (length/4)*4;offset>0;offset>>=1)
            {
            k1 ^= __shfl_down_sync(0xffffffff,k1,offset);
            }*/
            k1 = bytes[start + i] * g[6];
            res ^= k1;
            res = inst(res, 11) * -g[3];
            i++;
        }

        /*
        Step 5: Final avalanche (mixing)
        acc = acc ^ (acc >> 15)
        acc = acc * PRIME2
        acc = acc ^ (acc >> 13)
        acc = acc * PRIME3
        acc = acc ^ (acc >> 16)
        */
        res ^= res >> 15;
        res *= g[1];
        res ^= res >> 13;
        res *= g[4];
        res ^= res >> 16;
        cr[wid] = res;
    }*/
}

///
Double buffering-load in one tile while processing another

///
HARDWARE UNDERSTANDING OF THE RTX 2070 SUPER:
   We have 40 Streaming multiprocessors(SMs) ,each SM can have upto 1024 resident thread(32warps)
   ,now the main part which is general to all turing architecture cards each SM has 4 hardware schedulers(each nominates 1 warp to run) , so by doing some basic math we see that 4*32=128 threads run (truly)parallely in a single clock per SM.but all 4 blocks in our case run concurrently .the SM can only see 1024 threads= 32 warps running it cannot differentiate between blocks.The shared memory and L1exist in teh same physical space hence cuda allows us to partition it as per our choice - these 3 choices are available: 
        | Shared Memory | L1 Cache |
        | ------------- | -------- |
        | **64 KB**     | 32 KB    |
        | **48 KB**     | 48 KB    |
        | **32 KB**     | 64 KB    |

    GENERAL INFO:
        | Component              | Count               |
        | ---------------------- | ------------------- |
        | Warp Schedulers        | **4**               |
        | Dispatch Units         | **8**               |
        | FP32 Cores             | **64**              |
        | INT32 Cores            | **64**              |
        | Tensor Cores           | **8**               |
        | Load/Store Units       | **8**               |
        | Special Function Units | **4**               |
        | Registers              | **65,536 (256 KB)** |
        | Shared Memory          | **Up to 64 KB**     |
        | L1 Cache               | **32 KB**           |
        | Texture Units          | **4**               |

///
PROBLEMS KNOWN
1) 75% thread underultilixzation in hashmap.cu
3) if i use shared memory for storing the bytes array in addition to the already stored offset   array ( whihc has a theoretical limit of 64 uint32_t offsets(256 bytes) per block) i will have to face a limit of 192 characters per string on avg ( where i use all 64 offsets)
To somewhat resolve this issue we can inc blocksize ( like instead od 256*160 maybe 512 * 80)
that we we get a larger shared memory and can accomodate for more inconsistent string sizes
4) master_offset_current should hold the next empty block and NOT THE LAST FILLED BLOCK
5) ALWAYS LAUNCH KERNEL IN A MULTIPLE OF 4 ELSE IT WILL FAIL(NEEDS TO BE DEALT WITH IN PROGRAM MAYBE PAD INPUT ?)
why ? in hashmap.cu for better compiler optimization we have two if condition (which will be run in parallel) as one uses threads%4 ==0 and one uses ==1 ( BUT A MAJOR ASSUMPTION IN THIS IS NUMBER OF THREADS IS EVEN)
6) upper limits of tids , wids are having alot of illegal memory access patterns
7) if last string in the burst being hashed is a single byte , program will fail
8) cannot write more than ~500,000 strings at once(shared memory limitations)
9) ***must always launch in multiples of 4 (threads must always  be in multiples of 4)***
10) in the atmoicCAS fucntion that is defined under a tid%4==0 condition
    we cant achive parallelism here as one thread writes a balue to glob al mem that the other one needs
11) whenever we give input were gonna have to add a final offset that is qeuqal to total size of bytes and all the condition that run must aassume total size - 1 this element will only be a padding measure    
TODO
1) initialize key array to -1 ->cudaMemset
2) try to use shared memory for every insertion burst ( use it for offsets  (ALSO U CAN ONLY USE IN __GLOBAL__ NOT IN __DEVICE__))
>>>>>>> setup single buffering code for shared memory in hashmap.cu
also use the manual method to inc shared memory from 48 to 64

3) now that weve declared everything inshared memory we need to convert the entire code in terms of their thread idx.x and not tid

4) make local_offset and local_bytes arrays to store the words that will be worked on in that iteration

5) SHIT! THERES SO MUCH PERFORMANCE IM LEAVING ON THE TABLE CAN JUST DO A TILE.SHFL ( TRY TO IMPLEMENT IMMEDIATELY IN INSERT KERNEL)

6) is it possible  to remove the master_offset array ?
7) optimize all the hash functions to do 4 thread comparisons at a time  , alr being implemented in overflow hash table 

8) Look into quotient-remainder probing as  replcament for linear probing

9) Can we presorty inseryions or lookups to have Better coalescing ? 

10) besides the offset why dont we add first m bytes of the actual value to check for a simpler check

11) sawp the tile.any() checks with ballots + __ffs or butterfly reduction ( check feasiblity since tile.any() has unecessary iterations)

***
Struct must exist on CPU first — for initialization.

Any pointers inside the struct must point to GPU memory before copying.

Copy the struct itself to GPU with cudaMalloc + cudaMemcpy
***