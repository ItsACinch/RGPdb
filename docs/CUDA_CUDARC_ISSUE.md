# CUDA cudarc Implementation Issue

## Current Status

The migration from `rustacuda` to `cudarc` has been completed, but we're encountering `CUDA_ERROR_ILLEGAL_ADDRESS` errors when trying to allocate memory or create streams.

## Error Details

```
CUDA error (using CPU fallback): Memory allocation failed: Intensities: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")
```

This error occurs when:
1. Creating a new stream: `ctx.new_stream()`
2. Allocating device memory: `stream.alloc_zeros::<f32>(size)`

## Root Cause Analysis

The `CUDA_ERROR_ILLEGAL_ADDRESS` error typically indicates:
1. The CUDA context is not properly active/current
2. The context was destroyed or invalidated
3. Memory corruption or invalid pointer access

In `cudarc`, the `CudaContext::new()` creates a context, but it may not automatically make it current for operations. The safe API is supposed to handle this, but there might be an issue with how we're using it.

## Attempted Fixes

1. ✅ Fixed `load_module` to use `.into()` for `Ptx` type conversion
2. ✅ Removed double `Arc` wrapping (``CudaContextDriver::new()` already returns `Arc`)
3. ✅ Changed from creating new streams to using `default_stream()`
4. ❌ Still encountering `ILLEGAL_ADDRESS` errors

## Next Steps

1. **Investigate cudarc API usage**: Check if there's a way to explicitly activate the context or if we need to use a different initialization pattern
2. **Check cudarc examples**: Look for working examples of memory allocation and stream creation
3. **Consider alternative**: If `cudarc` continues to have issues, we might need to:
   - Use `cudarc`'s unsafe API directly
   - Try a different CUDA binding library
   - Fix the original `rustacuda` implementation with better resource management

## Current Implementation

- Context creation: `CudaContextDriver::new(0)` - returns `Arc<CudaContextDriver>`
- Stream creation: `ctx.default_stream()` - returns `Arc<CudaStream>`
- Memory allocation: `stream.alloc_zeros::<f32>(size)` - fails with `ILLEGAL_ADDRESS`

## Recommendation

The `cudarc` migration was intended to fix resource management issues, but we're encountering new issues. We should:
1. Investigate the `cudarc` API more thoroughly
2. Check if there are known issues or workarounds
3. Consider reverting to `rustacuda` with improved resource management if `cudarc` proves problematic

