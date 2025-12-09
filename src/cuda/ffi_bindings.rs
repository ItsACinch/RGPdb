// FFI bindings for CUDA wrapper

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CudaWrapperError {
    Success = 0,
    InitFailed = 1,
    DeviceNotFound = 2,
    MemoryAllocation = 3,
    MemoryCopy = 4,
    ModuleLoad = 5,
    FunctionNotFound = 6,
    KernelLaunch = 7,
    StreamSync = 8,
    Unknown = 99,
}

pub type CudaContext = *mut c_void;

#[link(name = "cuda_wrapper", kind = "static")]
extern "C" {
    pub fn cuda_wrapper_find_nvidia_device() -> c_int;
    pub fn cuda_wrapper_create_context(device_id: c_int, out_context: *mut CudaContext) -> CudaWrapperError;
    pub fn cuda_wrapper_get_device_name(ctx: CudaContext, name: *mut c_char, name_len: usize) -> CudaWrapperError;
    pub fn cuda_wrapper_load_module(ctx: CudaContext, ptx_string: *const c_char) -> CudaWrapperError;
    pub fn cuda_wrapper_get_function(ctx: CudaContext, function_name: *const c_char, out_function: *mut *mut c_void) -> CudaWrapperError;
    pub fn cuda_wrapper_malloc(ctx: CudaContext, size_bytes: usize, out_ptr: *mut *mut c_void) -> CudaWrapperError;
    pub fn cuda_wrapper_free(ctx: CudaContext, ptr: *mut c_void) -> CudaWrapperError;
    pub fn cuda_wrapper_memcpy_htod(ctx: CudaContext, dst: *mut c_void, src: *const c_void, size_bytes: usize) -> CudaWrapperError;
    pub fn cuda_wrapper_memcpy_dtoh(ctx: CudaContext, dst: *mut c_void, src: *const c_void, size_bytes: usize) -> CudaWrapperError;
    pub fn cuda_wrapper_memset(ctx: CudaContext, dst: *mut c_void, value: c_int, size_bytes: usize) -> CudaWrapperError;
    pub fn cuda_wrapper_malloc_host(ctx: CudaContext, size_bytes: usize, out_ptr: *mut *mut c_void) -> CudaWrapperError;
    pub fn cuda_wrapper_free_host(ctx: CudaContext, ptr: *mut c_void) -> CudaWrapperError;
    pub fn cuda_wrapper_launch_kernel(
        ctx: CudaContext,
        function: *mut c_void,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_mem_bytes: u32,
        args: *mut *mut c_void,
        num_args: usize,
    ) -> CudaWrapperError;
    pub fn cuda_wrapper_synchronize(ctx: CudaContext) -> CudaWrapperError;
    pub fn cuda_wrapper_destroy_context(ctx: CudaContext);
}

