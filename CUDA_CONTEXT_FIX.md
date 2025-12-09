# CUDA Context Management Fix

## Problem
When using `create_and_push`, the context is pushed onto a stack. When `CudaContext` is dropped, the context is popped, but streams and device buffers created with that context may still be trying to clean up, causing "IllegalAddress" errors.

## Solution: Explicit Context Stack Management

We need to ensure the context remains active (on the stack) when resources are being cleaned up. The key is to:
1. Keep the context on the stack throughout resource lifetime
2. Use explicit synchronization before cleanup
3. Ensure proper drop order

## Implementation Strategy

Since `rustacuda` doesn't provide explicit `push`/`pop` methods, we'll:
1. Keep the context alive by storing it in `CudaContext`
2. Ensure all CUDA operations happen while context is active
3. Use a scoped approach where context lifetime extends beyond all resource usage
4. Consider using `quick_init` for simpler initialization

