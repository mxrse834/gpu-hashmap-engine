#include<vector>
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#define TPB 1024
using namespace std;

// ==============================
// GPU HashMap Engine - Core File
// ==============================

// Kernel: Insert Key-Value Pair into Hash Table (out goal in this part is to generate a key for a specific value to get a key,value pair and store this in our dictionary)
__global__ void insert_kernel(int n, int *h_keys, int *h_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if(tid<n)
    {///h_keys[tid]=-1;//declare an array in global mem that can carry 8 elements initialize arr to -1
    int ptv;
    int base=h_values[tid]%n; // each thread has its own value of base and and empty ptv
    for(int i =0;i<n;i++)
    {
    int idx = (base+i)%n;  //  the for loop over "i" allows for linear probing
    ptv=atomicCAS(&h_keys[idx],-1,h_values[tid]);
    if(ptv==-1)
    return;   //very imp must use cudaDeviceSynchronize();to make sure all threads are on the same instructions and are done 
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
    while(i<n);*/ //same code as the above for loop less modular and easy to understand
    //okay now lets store the values we have in our keys lets copy to shared memory
    
}
}

// Kernel: Lookup Key in Hash Table
__global__ void lookup_kernel(int n,int l,int* vals,int *locn,int* h_keys) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // so the lookup kenerl with take some value and use the hashing function (%) on it and return a a place value where the lement may be stored or maybe have to linearly probe further
    if(tid<l) 
    {int base = vals[tid]%n; // thi will give the ele location without collision
     for(int i=0;i<n;i++)
     {if(vals[tid]==h_keys[(base + i)%n])
     {
      locn[tid]=(base +i)%n;
      return;
     }
     }

    }
}
// Kernel: Delete Key from Hash Table
__global__ void delete_kernel(int *keys, int *hash_table_keys, int *hash_table_values, int num_items) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Placeholder for delete logic
}

// Main Testing Function
int main() {
    std::cout << "GPU HashMap Engine - CUDA Kernel Skeleton Initialized" << std::endl;
    int h[] = {10, 20, 30, 40, 50, 60, 70, 80};  // sample values*/
    int n=8;
    string x;
    vector<int> temp;
    cout<<"Type in the elements u want to search for, when ur done type in 'eol' standing for end of list";
    int i=0;
    while(cin>>x && x!="eol")
    {try
    {
    int number=stoi(x);
    temp.push_back(number);
    }
    catch(const invalid_argument& e)
    {
        cerr << "Error is:" << e.what() << '\n';
    }
    i++;
    }
    int *g_k,*g_v,*locn,*g_temp;
    int *output=new int[i];
    int *h_temp=new int[i];
    size_t size=n*sizeof(int);
    size_t size1=i*sizeof(int);
    cudaMalloc(&g_k,size);
    cudaMalloc(&g_v,size);
    cudaMemset(g_k,0xFF,size);
    cudaMalloc(&locn,size1);
    cudaMemset(locn,0xFF,size1);
    cudaMalloc(&g_temp,size1);
    cudaMemcpy(g_v,h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(g_temp,temp.data(),size1,cudaMemcpyHostToDevice);
    //cudaMemset(locn,0xFF,size1);
    dim3 block(TPB);
    dim3 grid((TPB+n-1)/TPB);
    dim3 grid1((TPB+i-1)/TPB);
    insert_kernel<<<grid,block>>>(n,g_k,g_v);
    cudaDeviceSynchronize();
    lookup_kernel<<<grid1,block>>>(n,i,g_temp,locn,g_k);
    cudaDeviceSynchronize();
    cudaMemcpy(output,locn,size1,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_temp,g_temp,size1,cudaMemcpyDeviceToHost);
    for(int l =0;l<i;l++)
    {
    if(output[l] == -1)
    cout<<"The element " << h_temp[l]<< " is not a part of the hahs table";
    else
    cout<<h_temp[l]<<" was found at "<<output[l];
    }
    cudaFree(g_k);
    cudaFree(g_v);
    cudaFree(locn);
    cudaFree(g_temp);
    delete[] output;
    return 0;
}
