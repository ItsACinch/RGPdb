# CUDA Debugging Guide

## Environment Variables

### CUDA_LAUNCH_BLOCKING=1

Set this environment variable to make CUDA operations synchronous, which helps identify where errors occur:

**Windows (PowerShell):**
```powershell
$env:CUDA_LAUNCH_BLOCKING=1
cargo run --release --bin benchmark_cuda
```

**Windows (CMD):**
```cmd
set CUDA_LAUNCH_BLOCKING=1
cargo run --release --bin benchmark_cuda
```

**Linux/Mac:**
```bash
export CUDA_LAUNCH_BLOCKING=1
cargo run --release --bin benchmark_cuda
```

This will:
- Make all CUDA operations synchronous (slower but easier to debug)
- Report errors immediately when they occur
- Help identify which CUDA call is causing "illegal address" errors

## Common CUDA Error Causes

### CUDA_ERROR_ILLEGAL_ADDRESS

This error typically occurs due to:

1. **Incorrect kernel argument packing**: Arguments must be passed as pointers to the actual values
   - Device pointers: Pass pointer to the device pointer value
   - Scalar values: Pass pointer to the scalar value

2. **Uninitialized device memory**: Memory allocated but not initialized before use

3. **Invalid device pointers**: Pointers that are NULL or point to freed memory

4. **Context not active**: CUDA context must be active when allocating/using memory

5. **Memory alignment issues**: Some CUDA operations require specific alignment

## Debugging Steps

1. Enable `CUDA_LAUNCH_BLOCKING=1` to get immediate error reporting
2. Check CUDA error codes after each CUDA operation
3. Verify device memory allocations succeed before use
4. Ensure kernel arguments are properly packed
5. Check that CUDA context is active when operations are performed

## Kernel Argument Packing

For `cuLaunchKernel`, arguments must be packed correctly:

```c
// For device pointers (e.g., float* dev_ptr)
void* arg = &dev_ptr;  // Pointer to the device pointer value

// For scalar values (e.g., unsigned int num_nodes)
unsigned int num_nodes = 1000;
void* arg = &num_nodes;  // Pointer to the scalar value
```

The Rust FFI wrapper handles this automatically, but errors in argument packing will cause "illegal address" errors.
