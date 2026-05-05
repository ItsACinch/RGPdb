# CUDA Binding Migration Plan

## Current Status
- Using `rustacuda` 0.1.3
- CUDA kernels compile and execute successfully
- Resource cleanup crashes with "IllegalAddress" errors
- Issue: Context stack management in rustacuda

## Migration Options

### Option 1: Fix rustacuda (Current Approach)
**Status**: In progress
**Effort**: Low-Medium
**Risk**: Medium (may not be fixable)

**Attempts**:
- Reordered struct fields (fixed module unloading)
- Added explicit synchronization
- Ensured context lifetime extends beyond resources
- Still crashes during stream/buffer cleanup

**Next Steps**:
- Try resource reuse pattern
- Investigate rustacuda source code for workarounds
- Consider patching rustacuda locally if needed

### Option 2: Migrate to `cudarc`
**Status**: Not started
**Effort**: Medium-High
**Risk**: Low (more modern, better maintained)

**Pros**:
- More Rust-idiomatic API
- Better resource management
- Active development
- Used by major projects (mistralrs, luminal)

**Cons**:
- Requires rewriting CUDA module
- Different API surface
- Need to adapt kernel loading/launching

**Migration Steps**:
1. Add `cudarc` dependency
2. Rewrite `CudaContext` using cudarc API
3. Adapt kernel loading (PTX -> cudarc format)
4. Rewrite `propagate_light_cuda` using cudarc
5. Test and benchmark

### Option 3: Use `cuda-sys` (Low-level)
**Status**: Not started
**Effort**: High
**Risk**: High (more error-prone)

**Pros**:
- Full control
- No abstraction overhead
- Can implement exactly what we need

**Cons**:
- Very verbose
- Manual resource management
- More error-prone
- Longer development time

## Recommendation

**Phase 1 (Immediate)**: Continue trying to fix rustacuda
- Try resource reuse pattern
- Investigate context stack management
- Test with minimal reproducer

**Phase 2 (If Phase 1 fails)**: Migrate to `cudarc`
- More maintainable long-term
- Better resource management
- Active community support

**Phase 3 (If needed)**: Consider `cuda-sys` only if cudarc doesn't meet requirements

## Decision Criteria

Switch to cudarc if:
- rustacuda fixes don't work after 2-3 more attempts
- We need production-ready CUDA support
- Migration effort is acceptable (< 1 week)

Stick with rustacuda if:
- We find a workaround that works reliably
- Migration effort is too high
- rustacuda gets updated with fixes

