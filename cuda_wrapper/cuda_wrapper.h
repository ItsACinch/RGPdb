#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    CUDA_WRAPPER_SUCCESS = 0,
    CUDA_WRAPPER_ERROR_INIT_FAILED = 1,
    CUDA_WRAPPER_ERROR_DEVICE_NOT_FOUND = 2,
    CUDA_WRAPPER_ERROR_MEMORY_ALLOCATION = 3,
    CUDA_WRAPPER_ERROR_MEMORY_COPY = 4,
    CUDA_WRAPPER_ERROR_MODULE_LOAD = 5,
    CUDA_WRAPPER_ERROR_FUNCTION_NOT_FOUND = 6,
    CUDA_WRAPPER_ERROR_KERNEL_LAUNCH = 7,
    CUDA_WRAPPER_ERROR_STREAM_SYNC = 8,
    CUDA_WRAPPER_ERROR_UNKNOWN = 99
} cuda_wrapper_error_t;

// CUDA context handle (opaque)
typedef void* cuda_context_t;

// Initialize CUDA and find NVIDIA GPU
// Returns device ID of NVIDIA GPU, or -1 if not found
int cuda_wrapper_find_nvidia_device(void);

// Create CUDA context for a specific device
cuda_wrapper_error_t cuda_wrapper_create_context(int device_id, cuda_context_t* out_context);

// Get device name
cuda_wrapper_error_t cuda_wrapper_get_device_name(cuda_context_t ctx, char* name, size_t name_len);

// Load PTX module
cuda_wrapper_error_t cuda_wrapper_load_module(cuda_context_t ctx, const char* ptx_string);

// Get kernel function
cuda_wrapper_error_t cuda_wrapper_get_function(cuda_context_t ctx, const char* function_name, void** out_function);

// Allocate device memory
cuda_wrapper_error_t cuda_wrapper_malloc(cuda_context_t ctx, size_t size_bytes, void** out_ptr);

// Free device memory
cuda_wrapper_error_t cuda_wrapper_free(cuda_context_t ctx, void* ptr);

// Copy host to device
cuda_wrapper_error_t cuda_wrapper_memcpy_htod(cuda_context_t ctx, void* dst, const void* src, size_t size_bytes);

// Copy device to host
cuda_wrapper_error_t cuda_wrapper_memcpy_dtoh(cuda_context_t ctx, void* dst, const void* src, size_t size_bytes);

// Set device memory to zero (faster than host-to-device copy)
cuda_wrapper_error_t cuda_wrapper_memset(cuda_context_t ctx, void* dst, int value, size_t size_bytes);

// Allocate pinned (page-locked) host memory
cuda_wrapper_error_t cuda_wrapper_malloc_host(cuda_context_t ctx, size_t size_bytes, void** out_ptr);

// Free pinned host memory
cuda_wrapper_error_t cuda_wrapper_free_host(cuda_context_t ctx, void* ptr);

// Launch kernel
cuda_wrapper_error_t cuda_wrapper_launch_kernel(
    cuda_context_t ctx,
    void* function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_mem_bytes,
    void** args,
    size_t num_args
);

// Synchronize
cuda_wrapper_error_t cuda_wrapper_synchronize(cuda_context_t ctx);

// Destroy context
void cuda_wrapper_destroy_context(cuda_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif // CUDA_WRAPPER_H

