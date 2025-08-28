//attempt to a 32 bit xxhash on gpu
//just to have an example to work along lets assume were hashing a string "abcd efgh ijkl mnop qrs"
//coop grps , shf_sync , ballot_sync
#include<iostream>
#include<cstdint>
#include<string>
using namespace std;
#define TPB 1024

__global__ void hash(string value , int key,int length)
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


if(tid<unit) // include all threads upto the closest multiple to 4 ( here all tid upto 16 ) 
{
value[tid];
/*if length >= 16:
    v1 = seed + PRIME1 + PRIME2
    v2 = seed + PRIME2
    v3 = seed
    v4 = seed - PRIME1
else:
    acc = seed + PRIME5
*/
} 
else ( all threads beyond x)
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