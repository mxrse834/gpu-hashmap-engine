//attempt to a 32 bit xxhash on gpu
//just to have an example to work along lets assume were hashing a string "abcd efgh ijkl mnop qrs"
//coop grps , shf_sync , ballot_sync



/*CPU PSEUDOCODE


Step 1:initialize counter
if length >= 16:
    v1 = seed + PRIME1 + PRIME2
    v2 = seed + PRIME2
    v3 = seed
    v4 = seed - PRIME1
else:
    acc = seed + PRIME5





Step 2: Process input in 16-byte chunks
for each 16-byte block in data:
    v1 = round(v1, word0)
    v2 = round(v2, word1)
    v3 = round(v3, word2)
    v4 = round(v4, word3)

where round(acc, input) =
    acc = acc + (input * PRIME2)
    acc = rotate_left(acc, 13)
    acc = acc * PRIME1



Step 3: Merge accumulators
if length >= 16:
    acc = rotate_left(v1, 1) +
          rotate_left(v2, 7) +
          rotate_left(v3, 12) +
          rotate_left(v4, 18)
else:
    acc = seed + PRIME5



Step 4: Process remaining bytes (<16)
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




Step 5: Final avalanche (mixing)
acc = acc ^ (acc >> 15)
acc = acc * PRIME2
acc = acc ^ (acc >> 13)
acc = acc * PRIME3
acc = acc ^ (acc >> 16)


*/


#include<iostream>
#include<cstdint>
#include<string>
using namespace std;
#define TPB 1024
#define SEED 0

__device__ __forceinline__ uint32_t inst(uint32_t x,int s)
{
    return __funnelshift_l(x,x,s);
}




__global__ void hash(void* value , int key,int length)
{   extern __shared__ int mb[];  //define mixing bowls in shared mem and hence theyll there for each block
int tid  = threadIdx.x + blockDim.x*blockIdx.x;
//defining the primes 
static constexpr uint32_t PRIME1 = 2654435761u; 
static constexpr uint32_t PRIME2 = 2246822519u; 
static constexpr uint32_t PRIME3 = 3266489917u; 
static constexpr uint32_t PRIME4 =  374761393u;
int lane_id=threadIdx.x%warpSize;
int warp_id = tid/warpSize;
int unit=(4+length-1)/4;
int res;
//each warp works on one character maybe ? ie we make 32 bit divisions(4byte divisions(which would mean 4 characters per division))
//(1bytes -> 2hex  character-> one char)
//finally we conclude one warp will work on 4 bytes
//okay now lefts make 32 but grps and process data in chunks
//so we will be using 5 warps here  
//we can launch f number of warps to do one operation that is the one done on the full set 
//we want one 'chararcter' that is 8 bits

//each thread deals with 4 bytes ?
// we can have v1 in thread0 , v2 in thread1 so on with v4 in thread3 
// we will need warp intrinsics across all 4 threads


const uint32_t* words = reinterpret_cast<const uint32_t*>(value); 
// this will read consecutive 4 bytes together that is its been converted from a char type array to a int type one 
if(tid<unit) // include all threads upto the closest multiple to 4 ( here all tid upto 16 ) 
{
if(lane_id<8) 
{   uint32_t v1=SEED + PRIME1 + PRIME2;
    v1+=words[lane_id]*PRIME2;
    v1+=inst(v1,13);
    v1*=PRIME1;
}

else if(lane_id<16) 
{
    uint32_t v2=SEED + PRIME2;
    v2+=words[lane_id]*PRIME2;
    v2+=inst(v2,13);
    v2*=PRIME1;
}
else if(lane_id<24) 
{
    uint32_t v3=SEED;
    v3+=words[lane_id]*PRIME2;
    v3+=inst(v3,13);
    v3*=PRIME1;
}
else if (lane_id<32) 
{
    uint32_t v4=SEED-PRIME1;
    v4+=words[lane_id]*PRIME2;
    v4+=inst(v4,13);
    v4*=PRIME1;
}
//done with step of v1,v2,v3,v4 for16 bytes (4byte each)
//now we combine them all
/*Step 3: Merge accumulators
if length >= 16:
    acc = rotate_left(v1, 1) +
          rotate_left(v2, 7) +
          rotate_left(v3, 12) +
          rotate_left(v4, 18)
else:
    acc = seed + PRIME5
*/
uint32_t v0=__shfl_down_sync(0xffffffff,v1)






 //now were accessing successive 4 byte grps by a single thread
// were gonna ccess in strides of 4 

} 
else // all threads beyond x
{
res = SEED + PRIME4;
}

}










int main()
{
 int len; 
 string val="abcdefghijklmnopqrs"; // value
 //cout << "Give input value to hash (non cryptographic)";
 //getline(cin,val);  
 int len=val.length(); // len stores length of value
 return 0;
}