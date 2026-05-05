# CUDA Resource Management Fix Plan

## Problem Analysis

### Current Issues:
1. **Context Lifecycle**: Using `Context::create_and_push` creates a context that's pushed onto a stack. When `CudaContext` is dropped, the context is popped, but device buffers and streams may still be trying to clean up after the context is destroyed.

2. **Stream Cleanup**: Streams are created but may not be properly synchronized before the context is destroyed, causing "IllegalAddress" errors during cleanup.

3. **Device Buffer Cleanup**: Device buffers are dropped after the context might be invalid, causing "Failed to deallocate CUDA Device memory" errors.

4. **Resource Ordering**: Resources are cleaned up in the wrong order - context is destroyed before streams and buffers finish cleanup.

## Solution Strategy

### Phase 1: Fix Context Management
- **Change**: Use `Context::create` instead of `create_and_push` to have explicit control
- **Alternative**: Keep `create_and_push` but ensure context stays alive longer
- **Best Practice**: Use a scoped approach where context lifetime is explicitly managed

### Phase 2: Fix Stream Management
- **Change**: Ensure streams are synchronized before dropping
- **Change**: Use default stream (NULL) instead of creating new streams when possible
- **Change**: Explicitly drop streams before context cleanup

### Phase 3: Fix Device Buffer Management
- **Change**: Explicitly drop all device buffers before context cleanup
- **Change**: Ensure all memory operations complete before cleanup
- **Change**: Use explicit synchronization points

### Phase 4: Implement Proper Resource Cleanup Order
1. Synchronize all streams
2. Drop all device buffers
3. Drop all streams
4. Drop context (if using explicit management)

## Implementation Steps

### Step 1: Modify Context Creation
- Change from `create_and_push` to explicit context management
- Or ensure context lifetime extends beyond all resource usage

### Step 2: Add Explicit Synchronization
- Add `stream.synchronize()` before dropping streams
- Ensure all kernel launches complete before cleanup

### Step 3: Add Explicit Resource Cleanup
- Drop device buffers explicitly before context cleanup
- Use scoped blocks to control drop order

### Step 4: Test Resource Management
- Create minimal test case to verify cleanup order
- Test with single and multiple propagations
- Verify no memory leaks or crashes

### Step 5: Re-enable CUDA Implementation
- Uncomment CUDA code
- Test with benchmark
- Verify performance improvements

### Step 6: Performance Testing
- Run benchmark with fixed CUDA implementation
- Compare CPU vs CUDA performance
- Measure speedup on different graph sizes

## Expected Outcomes

1. **No Crashes**: CUDA operations complete without panics
2. **Proper Cleanup**: All resources are cleaned up in correct order
3. **Performance Gains**: CUDA shows measurable speedup over CPU
4. **Stability**: Multiple consecutive runs work without issues

## Risk Mitigation

- Keep CPU fallback as safety net
- Add error handling for CUDA operations
- Test incrementally (one fix at a time)
- Verify each step before proceeding

