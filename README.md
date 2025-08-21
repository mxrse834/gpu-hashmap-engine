# GPU-Accelerated High-Performance Key-Value Store (GPU HashMap Engine)

## Project Overview

This is my first CUDA C++ application project and also my first project on GitHub! 🚀 

This project aims to achieve a highly efficient hashing engine using GPU parallelization. The system is designed as a core GPU data-structure engine for real-time analytics, caching layers, and infrastructure-scale workloads.

## Problem Statement

Building a high-performance GPU-accelerated in-memory Key-Value Store using CUDA C++ that supports:
- **Batch Insertions & Lookups** of key-value pairs in parallel
- **Aggregation Queries** (SUM, MIN, MAX) across key sets
- **Prefix Sum Range Queries**
- **Divergence-efficient filtering & Stream Compaction**
- **Warp & Shared Memory optimizations** for high throughput

## Repository Structure

```
src/           # Core CUDA code (kernels, device functions)
data/          # Example input datasets  
benchmarks/    # Nsight Compute results, profiling logs
docs/          # Design documents, architecture decisions
README.md      # Project overview (this file)
```

## Build Instructions

```bash
nvcc hashmap.cu -o hm
./hm
```

## Hardware Constraints

Current development is optimized for:
- **RTX 4070** and **RTX 2070 SUPER** GPUs
- Solutions must respect the capabilities and limits of these GPUs

---

## Implementation Journey

To maximize efficiency, I'm focusing on optimizing **4 core metrics**:

1. **Hashing Function** - Generate IDs as unique as possible with minimal collisions
2. **Index Calculation/Hash Table** - Efficient parallel data structure for storage and retrieval
3. **Collision Handling** - System to resolve hash conflicts efficiently
4. **Dynamic Resizing** - Automatically scale storage as key-value pairs increase



### Evolution Through Versions

The project follows an iterative approach from naive (Version I) to highly optimized (Version N).



## Version 1: Baseline Implementation

### Design Choices

**Hash Function:**
- Simple modulus: `value % table_size`

**Hash Table Structure:**  
- 1D array

**Collision Handling:**
- Linear probing

### Problems Identified

1. **Race Conditions**: Multiple threads accessing same hash buckets simultaneously cause data corruption
2. **SIMT Limitations**: Atomic operations lead to serialization and performance degradation  
3. **Poor Hash Distribution**: Modulo operator creates clustering with non-uniform data
4. **Inefficient Initialization**: Setting `h_keys[tid] = -1` inside kernels wastes SM resources

### Potential Solutions Explored

- **Thread Grouping**: Divide threads into groups working on separate hash buckets
- **Atomic Operations**: Use `atomicCAS` for exclusive access to hash table slots
- **Cooperative Groups**: Fine-grained synchronization within thread blocks and warps
- **Warp-level Communication**: Improve efficiency and reduce contention

---

## Version 2: Complete Working Implementation ✅

### Architecture Overview

**Core Data Structure:**
```cpp
int h[] = {10, 20, 30, 40, 50, 60, 70, 80};  // Hash table values
int n = 8;                                   // Fixed hash table size
```

**Memory Layout:**
- **Hash Keys Array** (`g_k`): Device array storing the actual hash table
- **Hash Values Array** (`g_v`): Device array with input values to be hashed
- **Search Array** (`g_temp`): Device array with user input search queries
- **Location Results** (`locn`): Device array storing found indices

### Key Improvements Implemented

#### 1. **Optimized Memory Initialization**
```cpp
cudaMemset(g_k, 0xFF, size);     // Hash table initialized to -1
cudaMemset(locn, 0xFF, size1);   // Results initialized to -1 (not found)
```
**Advantage**: Leverages GPU's DMA engines instead of wasting SM cycles

#### 2. **Race-Free Insertion Kernel**
```cpp
__global__ void insert_kernel(int n, int *h_keys, int *h_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if(tid < n) {
        int ptv;
        int base = h_values[tid] % n;
        for(int i = 0; i < n; i++) {
            int idx = (base + i) % n;
            ptv = atomicCAS(&h_keys[idx], -1, h_values[tid]);
            if(ptv == -1) 
                return;   // Successfully inserted, exit
        }
    }
}
```

**Key Features:**
- **Linear Probing**: `(base + i) % n` for collision resolution
- **Atomic Operations**: `atomicCAS` prevents race conditions
- **Early Exit**: Return immediately on successful insertion

#### 3. **Robust Lookup Kernel**
```cpp
__global__ void lookup_kernel(int n, int l, int* vals, int *locn, int* h_keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < l) {  // Check against search array size, not hash table size
        int base = vals[tid] % n;
        for(int i = 0; i < n; i++) {
            if(vals[tid] == h_keys[(base + i) % n]) {
                locn[tid] = (base + i) % n;
                return;
            }
        }
        // If not found, locn[tid] remains -1 (from cudaMemset)
    }
}
```

**Key Features:**
- **Proper Thread Bounds**: `tid < l` where `l` is search count, not hash table size
- **Linear Probing Search**: Same pattern as insertion for consistency
- **Not Found Handling**: Relies on pre-initialized -1 values

#### 4. **Dynamic User Input Processing**
```cpp
vector<int> temp;
string x;
int i = 0;
while(cin >> x && x != "eol") {
    try {
        int number = stoi(x);
        temp.push_back(number);
    }
    catch(const invalid_argument& e) {
        cerr << "Error is:" << e.what() << '\n';
    }
    i++;
}
```

**Features:**
- **Dynamic sizing**: Vector grows as user adds elements
- **Error handling**: Catches invalid input gracefully
- **Flexible termination**: "eol" to end input

#### 5. **Correct Memory Management**
```cpp
// Host allocations
int *output = new int[i];
int *h_temp = new int[i];

// Device allocations  
cudaMalloc(&g_k, size);
cudaMalloc(&g_v, size);
cudaMalloc(&locn, size1);
cudaMalloc(&g_temp, size1);

// Proper vector data copying
cudaMemcpy(g_temp, temp.data(), size1, cudaMemcpyHostToDevice);

// Complete cleanup
cudaFree(g_k); cudaFree(g_v); cudaFree(locn); cudaFree(g_temp);
delete[] output;  // Array deletion, not single object deletion
```

### CUDA Memory Operations Deep Dive

**cudaMemset Analysis:**
```cpp
cudaMemset(g_k, 0xFF, size);  // Sets each byte to 0xFF
```
- **Byte-wise limitation**: Can only store values 0-255
- **For -1 initialization**: `0xFF = 11111111` in binary = -1 for signed integers
- **DMA advantage**: Uses dedicated memory controllers, not SMs

**Vector vs Array Memory Transfer:**
```cpp
// ❌ WRONG - copies vector object metadata
cudaMemcpy(g_temp, &temp, size1, cudaMemcpyHostToDevice);

// ✅ CORRECT - copies actual data array
cudaMemcpy(g_temp, temp.data(), size1, cudaMemcpyHostToDevice);
```

### Debugging Journey

#### Error 1: SIGSEGV (Address Boundary Error)
**Cause**: 
```cpp
int *output;  // Uninitialized pointer
cudaMemcpy(output, locn, size1, cudaMemcpyDeviceToHost);  // CRASH!
```
**Solution**: 
```cpp
int *output = new int[i];  // Proper allocation
```

#### Error 2: Wrong Memory Transfer Pattern
**Cause**: Using `&temp` instead of `temp.data()` for vector
**Solution**: Understanding vector object vs vector data distinction

#### Error 3: Thread Bounds Mismatch
**Original Issue**: 
```cpp
if(tid < n)  // Wrong! n=8 but search array has i elements
```
**Solution**:
```cpp 
if(tid < l)  // Correct! l=i is actual search count
```

#### Error 4: Lookup Results Display Logic
**Final Implementation**:
```cpp
for(int l = 0; l < i; l++) {
    if(output[l] == -1)
        cout << "The element " << h_temp[l] << " is not a part of the hash table";
    else
        cout << h_temp[l] << " was found at " << output[l];
}
```

### Performance Test Results

**Test Input:**
20, 70, 90, 0


**Expected Hash Table State After Insertion:**
Index: 0  1  2  3  4  5  6  7
Value: 80 - 10 30 20 50 60 70
       ^              ^     ^
      (80%8=0)    (20%8=4) (70%8=6→7)


**Actual Output:**

20 was found at 4
70 was found at 7  
The element 90 is not a part of the hash table
The element 0 is not a part of the hash table


**Analysis**: Linear probing working correctly with proper not-found detection.

### Debugging Tools Integration

**compute-sanitizer Integration:**
We Run on Bash:
bash: "/opt/cuda/bin/compute-sanitizer ./hm"

OUTPUT:
========= COMPUTE-SANITIZER
GPU HashMap Engine - CUDA Kernel Skeleton Initialized
...
========= ERROR SUMMARY: 0 errors


**What compute-sanitizer detects:**
1. **Memory access errors** → Out-of-bounds access
2. **Race conditions** → Simultaneous memory modifications
3. **API errors** → CUDA runtime/driver misuse


## Current Status: Version 2 Complete 

### Working Features
- [x] **Batch Insertion**: Parallel hash table population using linear probing
- [x] **Collision Resolution**: Atomic `atomicCAS` operations prevent race conditions  
- [x] **Parallel Lookup**: Multi-threaded search with proper bounds checking
- [x] **Dynamic Input**: User can search for any number of elements
- [x] **Memory Efficiency**: DMA-based initialization, proper cleanup
- [x] **Error Handling**: Robust input validation and memory management
- [x] **Not Found Detection**: Proper handling of missing elements

### Performance Characteristics
- **Hash Function**: `value % table_size` (simple but collision-prone)
- **Collision Resolution**: Linear probing with atomic synchronization
- **Memory Access**: Coalesced for optimal bandwidth utilization  
- **Thread Efficiency**: Minimal warp divergence in insertion/lookup loops
- **Space Complexity**: O(n) hash table + O(search_count) auxiliary arrays

### Current Limitations
1. **Fixed Hash Table Size**: Currently hardcoded to 8 elements
2. **Simple Hash Function**: Modulo operation creates clustering  
3. **Linear Probing Overhead**: Performance degrades with high collision rates
4. **No Dynamic Resizing**: Cannot grow hash table at runtime

---





















"unsigned int active = __ballot_sync(0xffffffff, stillSearching);
if (!(active & (1 << lane_id))) break;
"
next topic :
shl_sync
also check if therea are any other methods and then write a lookup proegram ( thats phase 2)
and _fss

