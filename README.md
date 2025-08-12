# GPU-Accelerated High-Performance Key-Value Store (GPU HashMap Engine)

## Problem Statement
Building a high-performance GPU-accelerated in-memory Key-Value Store using CUDA C++.  
The system will support:
- Batch Insertions & Lookups of key-value pairs in parallel.
- Aggregation Queries (SUM, MIN, MAX) across key sets.
- Prefix Sum Range Queries.
- Divergence-efficient filtering & Stream Compaction.
- Warp & Shared Memory optimizations for high throughput.

This will serve as a core GPU data-structure engine, designed for real-time analytics, caching layers, and infra-scale workloads.

## Project Goals
1. Implement GPU-optimized batch insert and lookup operations.
2. Design collision handling using shared memory and warp-level strategies.
3. Support aggregation queries using warp-level reductions.
4. Efficiently handle divergent queries using stream compaction.
5. Structure code in a clean, modular, GitHub-ready format.

---

## Repo Structure
src/ # Core CUDA code (kernels, device functions)
data/ # Example input datasets
benchmarks/ # Nsight Compute results, profiling logs
docs/ # Design documents, architecture decisions
README.md # Project overview


## Roadmap
- Phase 1: Batch Insertions & Lookups (hashmap.cu)
- Phase 2: Shared Memory & Warp Optimizations
- Phase 3: Aggregation Queries
- Phase 4: Stream Compaction & Divergence Handling
- Phase 5: API Wrapping & GitHub Finalization



This is my first CUDA C++ application based project and also my first project on git :) this project aims to achieve a highly efficient hashing engine using a GPU. The program structure is as mentioned-
## Repo Structure
src/           # Core CUDA code (kernels, device functions)
data/          # Example input datasets
benchmarks/    # Nsight Compute results, profiling logs
docs/          # Design documents, architecture decisions
README.md      # Project overview

let us drive straight into the process-
At this point i somewhat feel that we need to narrow down on maximizing 3 metrics to make the code as efficient as possible- 
for my main file ( src/hashmap.cu). these 3 requirements are as follows - 
1)A hashing function-Hashing function is responsible(performs the work of reducing a large input element into an efficient "hash" ultimately to calculate the index to store this value in , it is a general algorithm that will generate as less collision as possible and be efficiently able to arrange a large dataset into hash table with minimal overlap of elements , hence we have a final aim of lets say "GENERATING IDs AS UNIQUE AS POSSIBLE". this function should be highly efficient and optimized for parallel computation, so our main task is to select a function that is most suitable to our usecase on a GPU

2)index calculation/hash table - after a efficient hashing function we must be able to index this hash effectively such that it will be easier to pull later on as well as be easier to store . as efficient of a data structure the table has for parallel draw , more efficient our kernel will be as indexing will be made simpler.

3)collision handling - finally the system we use to handle collisions is very important as this will determine how easy it will be to retrive an item later on 

4)dynamic resizing - we dont want to unnecessarily allocate a very big chunk of storage to it , it should dynamically increase as the number of key value pairs increases


Now for actual benchmarking purposes lets start with the simplest approach(the below will be represented in roman numerals starting from the most naive i) to the the most efficient n)-

i)Hash Function selected - Modulus(value % table_size)
  Hash Table - 1D array
  Collision Handling -Linear Probing
 Problems faced in this approach 
 i)since multiple elements have the same modulus n(where n is the table size)and they perform simultaneous read write functions multiple elements may read /write to a location in array resulting in corruption or overriding of data 
   solutions. any solutions we implement will either need more memory , more compute or will have higher time complexity. Currently were limited to hardware by a RTX 4070 and a RTX 2070 SUPER.
   Ideas- 
   1) Division of threads into groups that execute on multiple buckets/maybe different hash tables entirely ( the parameters for this division are to be decided)
   2) atmoicCAS
   3) explore cooperative groups 
   4) warp level scheduling


   ///working