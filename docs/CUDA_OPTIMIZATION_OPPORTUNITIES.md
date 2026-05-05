# CUDA Optimization Opportunities

## Current State Analysis

After implementing graph data caching, CUDA performance improved 2-3x but is still slower than CPU. Here are additional optimization opportunities:

## 1. **Critical: Fix Kernel Bug** 🔴 HIGH PRIORITY

### Issue: Incorrect Atomic Operations
**Location**: `kernels/propagate_kernel.cu:107-112`

```cuda
// CURRENT (WRONG):
float old_val = atomicExch(&next_intensities[neighbor_intensity_idx], transmitted);
if (transmitted > old_val) {
    atomicExch(&next_intensities[neighbor_intensity_idx], transmitted);
} else {
    atomicExch(&next_intensities[neighbor_intensity_idx], old_val);
}
```

**Problem**: `atomicExch` doesn't return the old value correctly in this pattern. Multiple threads can race and overwrite each other.

**Fix**: Use `atomicMax` or proper compare-and-swap:
```cuda
// CORRECT:
atomicMax(&next_intensities[neighbor_intensity_idx], transmitted);
```

**Impact**: This bug likely causes the intensity value mismatch (CPU: 31.30 vs CUDA: 12.24). Fixing this may significantly improve correctness and potentially performance.

---

## 2. **Memory Access Optimizations** 🟡 MEDIUM PRIORITY

### 2.1 Use Shared Memory for Node Properties
**Current**: Each thread loads node properties from global memory
**Optimization**: Load node properties into shared memory per block

```cuda
__shared__ float shared_node_props[256 * 4];  // Per block
// Load once per block, reuse across threads
```

**Expected Impact**: 10-30% speedup for dense graphs

### 2.2 Coalesced Memory Access
**Current**: Memory access pattern `node_idx * num_angle_bins + angle_bin` may not be coalesced
**Optimization**: Reorganize data layout or use texture memory for read-only data

**Expected Impact**: 5-15% speedup

### 2.3 Use Texture Memory for Read-Only Data
**Current**: All data in global memory
**Optimization**: Bind `node_props`, `row_ptr`, `col_idx`, `edge_props` to texture memory

```cuda
texture<float, 1, cudaReadModeElementType> tex_node_props;
texture<unsigned int, 1, cudaReadModeElementType> tex_row_ptr;
```

**Expected Impact**: 10-20% speedup for memory-bound kernels

---

## 3. **Kernel Launch Optimizations** 🟡 MEDIUM PRIORITY

### 3.1 Optimize Block Size
**Current**: Fixed at 256 threads per block
**Optimization**: Profile different block sizes (128, 256, 512, 1024) and choose optimal

**Method**: Use CUDA occupancy calculator or empirical testing
**Expected Impact**: 5-20% speedup

### 3.2 Use Constant Memory for Parameters
**Current**: `k` and `min_intensity` passed as kernel arguments
**Optimization**: Use constant memory for frequently accessed parameters

```cuda
__constant__ float c_k;
__constant__ float c_min_intensity;
```

**Expected Impact**: 2-5% speedup

### 3.3 Reduce Synchronization Overhead
**Current**: Synchronize after each iteration
**Optimization**: Use CUDA streams for async execution, overlap computation with data transfer

**Expected Impact**: 10-30% speedup for batched queries

---

## 4. **Algorithm Optimizations** 🟢 LOW PRIORITY

### 4.1 Early Exit Optimization
**Current**: All threads process all neighbors
**Optimization**: Use warp-level primitives to exit early when intensity is too low

```cuda
if (current_intensity < min_intensity) {
    __syncthreads();
    return;  // Early exit for entire warp
}
```

**Expected Impact**: 5-15% speedup for sparse graphs

### 4.2 Warp-Level Reductions
**Current**: Each thread accumulates independently
**Optimization**: Use `__shfl_down_sync` for warp-level reductions

**Expected Impact**: 5-10% speedup

### 4.3 Kernel Fusion
**Current**: Separate init and propagate kernels
**Optimization**: Fuse initialization into first propagation iteration

**Expected Impact**: Eliminates one kernel launch (~0.1ms overhead)

---

## 5. **Memory Transfer Optimizations** 🟡 MEDIUM PRIORITY

### 5.1 Use Pinned Memory
**Current**: Using pageable memory for host data
**Optimization**: Use `cudaMallocHost` for pinned/page-locked memory

```rust
// In Rust, use pinned memory for host buffers
let mut pinned_node_props: Vec<f32> = Vec::with_capacity(...);
// Allocate with CUDA pinned memory API
```

**Expected Impact**: 20-50% faster host-to-device transfers

### 5.2 Zero Buffers on Device
**Current**: Copying zero vector from host to device
**Optimization**: Use `cudaMemset` or kernel to zero buffers on device

```cuda
cudaMemset(dev_next_intensities, 0, intensities_size);
```

**Expected Impact**: Eliminates host-to-device transfer eliminated (~0.05ms per iteration)

### 5.3 Asynchronous Memory Transfers
**Current**: Synchronous memory copies
**Optimization**: Use `cudaMemcpyAsync` with CUDA streams

**Expected Impact**: Overlap computation with transfers (10-30% speedup for batched queries)

---

## 6. **Advanced Optimizations** 🔵 ADVANCED

### 6.1 Multi-Query Batching
**Current**: Process one query at a time
**Optimization**: Batch multiple queries, process simultaneously

**Expected Impact**: Amortize overhead across queries (2-5x speedup for batch size 10+)

### 6.2 Dynamic Parallelism
**Current**: CPU controls iteration loop
**Optimization**: Use CUDA dynamic parallelism to control iterations on GPU

**Expected Impact**: Eliminate CPU-GPU synchronization overhead

### 6.3 Use CUDA Graphs
**Current**: Individual kernel launches
**Optimization**: Record kernel sequence as CUDA graph, replay efficiently

**Expected Impact**: 5-10% reduction in launch overhead

---

## 7. **Profiling-Based Optimizations** 📊 REQUIRED

### 7.1 Use NVIDIA Nsight Compute
**Tool**: Profile kernels to identify bottlenecks
**Metrics to check**:
- Memory throughput (GB/s)
- Compute utilization (%)
- Occupancy (%)
- Warp efficiency (%)

### 7.2 Use NVIDIA Nsight Systems
**Tool**: Profile entire application
**Metrics to check**:
- Kernel launch overhead
- Memory transfer time
- CPU-GPU synchronization time
- Context switching overhead

---

## Priority Ranking

### Immediate (Fix Bugs)
1. **Fix atomic operations bug** - Likely causing incorrect results
2. **Use `cudaMemset` instead of host-to-device zero copy** - Simple, immediate win

### High Impact (Easy Wins)
3. **Use pinned memory** - Significant transfer speedup
4. **Optimize block size** - Easy to test, good potential
5. **Use constant memory for parameters** - Simple change

### Medium Impact (Moderate Effort)
6. **Use shared memory for node properties** - Requires kernel rewrite
7. **Use texture memory** - Requires kernel rewrite
8. **Asynchronous memory transfers** - Requires stream management

### Advanced (High Effort)
9. **Multi-query batching** - Requires API changes
10. **Kernel fusion** - Requires algorithm changes
11. **CUDA graphs** - Requires significant refactoring

---

## Expected Combined Impact

If all optimizations are implemented:
- **Bug fixes**: Correctness + potential 10-20% speedup
- **Memory optimizations**: 30-50% speedup
- **Kernel optimizations**: 20-40% speedup
- **Batching**: 2-5x speedup for multiple queries

**Total potential**: 3-10x speedup, potentially making CUDA faster than CPU for graphs >10k nodes

---

## Implementation Order

1. **Week 1**: Fix atomic bug, use `cudaMemset`, add pinned memory
2. **Week 2**: Profile with Nsight, optimize block size, add constant memory
3. **Week 3**: Implement shared memory, texture memory
4. **Week 4**: Add async transfers, test batching

---

## Testing Strategy

After each optimization:
1. Verify correctness (results match CPU)
2. Benchmark performance (100, 1000, 10000 nodes)
3. Profile with Nsight Compute
4. Document speedup achieved

