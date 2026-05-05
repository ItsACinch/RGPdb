# CUDA Alternative Bindings Investigation

## Current Issue
`rustacuda` has context stack management issues that cause "IllegalAddress" errors during resource cleanup. The context is pushed onto a stack with `create_and_push`, but when resources (streams, buffers) are dropped, the context may have been popped, causing cleanup failures.

## Alternative CUDA Bindings for Rust

### 1. `cudarc` (CUDA Rust Core)
- **Status**: Active development, more modern API
- **Pros**: 
  - Better resource management
  - More Rust-idiomatic API
  - Active maintenance
- **Cons**: 
  - May require API changes
  - Different API surface
- **Migration effort**: Medium

### 2. `cuda-sys` (Low-level bindings)
- **Status**: Low-level FFI bindings
- **Pros**: 
  - Full control over resource management
  - No abstraction overhead
- **Cons**: 
  - More verbose
  - Manual resource management
  - More error-prone
- **Migration effort**: High

### 3. `cuda_runtime_sys` / `cudart-sys`
- **Status**: Runtime API bindings
- **Pros**: 
  - Simpler than Driver API
  - Better for simple use cases
- **Cons**: 
  - Less control
  - May not support all features we need
- **Migration effort**: Medium-High

## Recommendation

Given the scaling requirements, we should:
1. **Short-term**: Fix rustacuda context management (try resource reuse)
2. **Medium-term**: Evaluate `cudarc` as a replacement
3. **Long-term**: Consider `cuda-sys` if we need maximum control

## Next Steps

1. Try fixing rustacuda with resource reuse pattern
2. If that fails, prototype with `cudarc`
3. Benchmark both approaches
4. Choose the most stable solution

