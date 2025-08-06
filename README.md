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
