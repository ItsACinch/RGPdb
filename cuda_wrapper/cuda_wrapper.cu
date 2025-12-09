#include "cuda_wrapper.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Internal context structure
typedef struct {
    int device_id;
    CUmodule module;
    CUcontext context;
    CUdevice device;
} cuda_context_internal_t;

int cuda_wrapper_find_nvidia_device(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return -1;
    }
    
    // Find first NVIDIA GPU
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            // Check if it's an NVIDIA device (all CUDA devices are NVIDIA)
            // But we can check compute capability to ensure it's a real GPU
            if (prop.major > 0) {
                return i;
            }
        }
    }
    
    return -1;
}

cuda_wrapper_error_t cuda_wrapper_create_context(int device_id, cuda_context_t* out_context) {
    if (out_context == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    cuda_context_internal_t* ctx = (cuda_context_internal_t*)malloc(sizeof(cuda_context_internal_t));
    if (ctx == NULL) {
        return CUDA_WRAPPER_ERROR_MEMORY_ALLOCATION;
    }
    
    memset(ctx, 0, sizeof(cuda_context_internal_t));
    ctx->device_id = device_id;
    
    // Initialize CUDA Driver API
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        free(ctx);
        return CUDA_WRAPPER_ERROR_INIT_FAILED;
    }
    
    // Get device
    result = cuDeviceGet(&ctx->device, device_id);
    if (result != CUDA_SUCCESS) {
        free(ctx);
        return CUDA_WRAPPER_ERROR_DEVICE_NOT_FOUND;
    }
    
    // Create context
    result = cuCtxCreate(&ctx->context, 0, ctx->device);
    if (result != CUDA_SUCCESS) {
        free(ctx);
        return CUDA_WRAPPER_ERROR_INIT_FAILED;
    }
    
    *out_context = (cuda_context_t)ctx;
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_get_device_name(cuda_context_t ctx, char* name, size_t name_len) {
    if (ctx == NULL || name == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    cuda_context_internal_t* internal = (cuda_context_internal_t*)ctx;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, internal->device_id);
    if (err != cudaSuccess) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    strncpy(name, prop.name, name_len - 1);
    name[name_len - 1] = '\0';
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_load_module(cuda_context_t ctx, const char* ptx_string) {
    if (ctx == NULL || ptx_string == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    cuda_context_internal_t* internal = (cuda_context_internal_t*)ctx;
    
    CUresult result = cuModuleLoadData(&internal->module, ptx_string);
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_MODULE_LOAD;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_get_function(cuda_context_t ctx, const char* function_name, void** out_function) {
    if (ctx == NULL || function_name == NULL || out_function == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    cuda_context_internal_t* internal = (cuda_context_internal_t*)ctx;
    
    CUfunction function;
    CUresult result = cuModuleGetFunction(&function, internal->module, function_name);
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_FUNCTION_NOT_FOUND;
    }
    
    *out_function = (void*)function;
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_malloc(cuda_context_t ctx, size_t size_bytes, void** out_ptr) {
    if (ctx == NULL || out_ptr == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    CUdeviceptr ptr;
    CUresult result = cuMemAlloc(&ptr, size_bytes);
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_MEMORY_ALLOCATION;
    }
    
    *out_ptr = (void*)ptr;
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_free(cuda_context_t ctx, void* ptr) {
    if (ctx == NULL || ptr == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    CUresult result = cuMemFree((CUdeviceptr)ptr);
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_MEMORY_ALLOCATION;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_memcpy_htod(cuda_context_t ctx, void* dst, const void* src, size_t size_bytes) {
    if (ctx == NULL || dst == NULL || src == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    CUresult result = cuMemcpyHtoD((CUdeviceptr)dst, src, size_bytes);
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_MEMORY_COPY;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_memcpy_dtoh(cuda_context_t ctx, void* dst, const void* src, size_t size_bytes) {
    if (ctx == NULL || dst == NULL || src == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    CUresult result = cuMemcpyDtoH(dst, (CUdeviceptr)src, size_bytes);
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_MEMORY_COPY;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_memset(cuda_context_t ctx, void* dst, int value, size_t size_bytes) {
    if (ctx == NULL || dst == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    // Use CUDA Runtime API for memset (simpler than Driver API)
    cudaError_t err = cudaMemset(dst, value, size_bytes);
    if (err != cudaSuccess) {
        return CUDA_WRAPPER_ERROR_MEMORY_COPY;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_malloc_host(cuda_context_t ctx, size_t size_bytes, void** out_ptr) {
    if (ctx == NULL || out_ptr == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    // Allocate pinned (page-locked) host memory
    cudaError_t err = cudaMallocHost(out_ptr, size_bytes);
    if (err != cudaSuccess) {
        return CUDA_WRAPPER_ERROR_MEMORY_ALLOCATION;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_free_host(cuda_context_t ctx, void* ptr) {
    if (ctx == NULL || ptr == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        return CUDA_WRAPPER_ERROR_MEMORY_ALLOCATION;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_launch_kernel(
    cuda_context_t ctx,
    void* function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_mem_bytes,
    void** args,
    size_t num_args
) {
    if (ctx == NULL || function == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    CUfunction func = (CUfunction)function;
    
    // Pack arguments for cuLaunchKernel
    // cuLaunchKernel expects arguments to be passed via void** where each pointer
    // points to the actual argument value (not the pointer itself)
    void* kernel_args[16];  // Max 16 arguments (should be enough)
    if (num_args > 16) {
        return CUDA_WRAPPER_ERROR_KERNEL_LAUNCH;
    }
    
    // Copy argument pointers (they point to the actual values)
    for (size_t i = 0; i < num_args; i++) {
        kernel_args[i] = args[i];
    }
    
    CUresult result = cuLaunchKernel(
        func,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        shared_mem_bytes,
        NULL,  // Use default stream
        kernel_args,
        NULL
    );
    
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_KERNEL_LAUNCH;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

cuda_wrapper_error_t cuda_wrapper_synchronize(cuda_context_t ctx) {
    if (ctx == NULL) {
        return CUDA_WRAPPER_ERROR_UNKNOWN;
    }
    
    CUresult result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        return CUDA_WRAPPER_ERROR_STREAM_SYNC;
    }
    
    return CUDA_WRAPPER_SUCCESS;
}

void cuda_wrapper_destroy_context(cuda_context_t ctx) {
    if (ctx == NULL) {
        return;
    }
    
    cuda_context_internal_t* internal = (cuda_context_internal_t*)ctx;
    
    if (internal->module) {
        cuModuleUnload(internal->module);
    }
    
    if (internal->context) {
        cuCtxDestroy(internal->context);
    }
    
    free(internal);
}

