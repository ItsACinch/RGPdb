/// Direct CUDA Runtime API implementation using FFI
/// 
/// This uses a C wrapper around CUDA Runtime API for maximum reliability
/// and control over resource management.

use crate::graph::Graph;
use crate::propagation::LightParams;
use crate::pvs::PVS;
use thiserror::Error;
use std::ffi::{CString, CStr};
use std::ptr;

use crate::cuda::ffi_bindings::{self, CudaWrapperError, CudaContext as FfiCudaContext};

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA not available (no GPU or CUDA toolkit not found)")]
    NotAvailable,
    #[error("CUDA runtime error: {0}")]
    Runtime(String),
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),
    #[error("Kernel launch failed: {0}")]
    KernelLaunch(String),
    #[error("Memory copy failed: {0}")]
    MemoryCopy(String),
    #[error("PTX compilation failed: {0}")]
    PtxCompilation(String),
    #[error("Kernels not loaded: {0}")]
    KernelsNotLoaded(String),
}

impl From<CudaWrapperError> for CudaError {
    fn from(err: CudaWrapperError) -> Self {
        match err {
            CudaWrapperError::Success => unreachable!(),
            CudaWrapperError::InitFailed => CudaError::Runtime("Failed to initialize CUDA".to_string()),
            CudaWrapperError::DeviceNotFound => CudaError::NotAvailable,
            CudaWrapperError::MemoryAllocation => CudaError::MemoryAllocation("Device memory allocation failed".to_string()),
            CudaWrapperError::MemoryCopy => CudaError::MemoryCopy("Memory copy failed".to_string()),
            CudaWrapperError::ModuleLoad => CudaError::PtxCompilation("Failed to load PTX module".to_string()),
            CudaWrapperError::FunctionNotFound => CudaError::KernelsNotLoaded("Function not found in module".to_string()),
            CudaWrapperError::KernelLaunch => CudaError::KernelLaunch("Kernel launch failed".to_string()),
            CudaWrapperError::StreamSync => CudaError::Runtime("Stream synchronization failed".to_string()),
            CudaWrapperError::Unknown => CudaError::Runtime("Unknown CUDA error".to_string()),
        }
    }
}

/// Cached graph data on GPU to avoid repeated memory transfers
struct CudaGraphCache {
    num_nodes: usize,
    num_edges: usize,
    dev_node_props: *mut std::ffi::c_void,
    dev_row_ptr: *mut std::ffi::c_void,
    dev_col_idx: *mut std::ffi::c_void,
    dev_edge_props: *mut std::ffi::c_void,
}

impl CudaGraphCache {
    fn new() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            dev_node_props: ptr::null_mut(),
            dev_row_ptr: ptr::null_mut(),
            dev_col_idx: ptr::null_mut(),
            dev_edge_props: ptr::null_mut(),
        }
    }
    
    fn is_valid_for(&self, num_nodes: usize, num_edges: usize) -> bool {
        self.num_nodes == num_nodes && self.num_edges == num_edges && 
        !self.dev_node_props.is_null()
    }
    
    fn cleanup(&mut self, ctx: FfiCudaContext) {
        unsafe {
            if !self.dev_node_props.is_null() {
                let _ = ffi_bindings::cuda_wrapper_free(ctx, self.dev_node_props);
                self.dev_node_props = ptr::null_mut();
            }
            if !self.dev_row_ptr.is_null() {
                let _ = ffi_bindings::cuda_wrapper_free(ctx, self.dev_row_ptr);
                self.dev_row_ptr = ptr::null_mut();
            }
            if !self.dev_col_idx.is_null() {
                let _ = ffi_bindings::cuda_wrapper_free(ctx, self.dev_col_idx);
                self.dev_col_idx = ptr::null_mut();
            }
            if !self.dev_edge_props.is_null() {
                let _ = ffi_bindings::cuda_wrapper_free(ctx, self.dev_edge_props);
                self.dev_edge_props = ptr::null_mut();
            }
        }
        self.num_nodes = 0;
        self.num_edges = 0;
    }
}

pub struct CudaContext {
    ctx: FfiCudaContext,
    init_kernel: Option<*mut std::ffi::c_void>,
    propagate_kernel: Option<*mut std::ffi::c_void>,
    graph_cache: CudaGraphCache,
}

impl CudaContext {
    pub fn new() -> Result<Self, CudaError> {
        // Find NVIDIA GPU
        let device_id = unsafe { ffi_bindings::cuda_wrapper_find_nvidia_device() };
        if device_id < 0 {
            return Err(CudaError::NotAvailable);
        }
        
        eprintln!("Found NVIDIA GPU at device {}", device_id);
        
        // Create context
        let mut ctx_ptr: FfiCudaContext = ptr::null_mut();
        let result = unsafe { ffi_bindings::cuda_wrapper_create_context(device_id, &mut ctx_ptr) };
        if result != CudaWrapperError::Success {
            return Err(result.into());
        }
        
        // Load PTX module
        let ptx_str = match crate::cuda::kernels::load_ptx() {
            Ok(s) => s,
            Err(_) => {
                unsafe { ffi_bindings::cuda_wrapper_destroy_context(ctx_ptr) };
                return Ok(Self {
                    ctx: ctx_ptr,
                    init_kernel: None,
                    propagate_kernel: None,
                    graph_cache: CudaGraphCache::new(),
                });
            }
        };
        
        let ptx_cstr = CString::new(ptx_str)
            .map_err(|e| CudaError::PtxCompilation(format!("Failed to create CString: {}", e)))?;
        
        let result = unsafe { ffi_bindings::cuda_wrapper_load_module(ctx_ptr, ptx_cstr.as_ptr()) };
        if result != CudaWrapperError::Success {
            unsafe { ffi_bindings::cuda_wrapper_destroy_context(ctx_ptr) };
            return Err(result.into());
        }
        
        // Get kernel functions
        let init_name = CString::new("init_propagation_kernel")
            .map_err(|e| CudaError::KernelsNotLoaded(format!("Function name: {}", e)))?;
        let mut init_func: *mut std::ffi::c_void = ptr::null_mut();
        let init_result = unsafe { ffi_bindings::cuda_wrapper_get_function(ctx_ptr, init_name.as_ptr(), &mut init_func) };
        
        let propagate_name = CString::new("propagate_iteration_kernel")
            .map_err(|e| CudaError::KernelsNotLoaded(format!("Function name: {}", e)))?;
        let mut propagate_func: *mut std::ffi::c_void = ptr::null_mut();
        let propagate_result = unsafe { ffi_bindings::cuda_wrapper_get_function(ctx_ptr, propagate_name.as_ptr(), &mut propagate_func) };
        
        if init_result != CudaWrapperError::Success || propagate_result != CudaWrapperError::Success {
            unsafe { ffi_bindings::cuda_wrapper_destroy_context(ctx_ptr) };
            return Ok(Self {
                ctx: ctx_ptr,
                init_kernel: None,
                propagate_kernel: None,
                graph_cache: CudaGraphCache::new(),
            });
        }
        
        Ok(Self {
            ctx: ctx_ptr,
            init_kernel: Some(init_func),
            propagate_kernel: Some(propagate_func),
            graph_cache: CudaGraphCache::new(),
        })
    }
    
    pub fn is_available() -> bool {
        unsafe { ffi_bindings::cuda_wrapper_find_nvidia_device() >= 0 }
    }
    
    pub fn kernels_loaded(&self) -> bool {
        self.init_kernel.is_some() && self.propagate_kernel.is_some()
    }
    
    pub fn device_name(&self) -> Result<String, CudaError> {
        let mut name_buf = vec![0u8; 256];
        let result = unsafe {
            ffi_bindings::cuda_wrapper_get_device_name(
                self.ctx,
                name_buf.as_mut_ptr() as *mut std::ffi::c_char,
                name_buf.len()
            )
        };
        
        if result != CudaWrapperError::Success {
            return Err(result.into());
        }
        
        let c_str = unsafe { CStr::from_ptr(name_buf.as_ptr() as *const std::ffi::c_char) };
        Ok(c_str.to_string_lossy().to_string())
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // Cleanup cached graph data
        self.graph_cache.cleanup(self.ctx);
        
        if !self.ctx.is_null() {
            unsafe { ffi_bindings::cuda_wrapper_destroy_context(self.ctx) };
        }
    }
}

/// Upload or update graph data on GPU
fn ensure_graph_cached(
    context: &mut CudaContext,
    graph: &Graph,
) -> Result<(), CudaError> {
    let num_nodes = graph.num_nodes();
    let num_edges = graph.num_edges();
    
    // Check if cache is valid
    if context.graph_cache.is_valid_for(num_nodes, num_edges) {
        return Ok(()); // Cache is valid, reuse it
    }
    
    // Cache is invalid or doesn't exist, upload graph data
    eprintln!("CUDA: Uploading graph data to GPU ({} nodes, {} edges)", num_nodes, num_edges);
    
    // Cleanup old cache
    context.graph_cache.cleanup(context.ctx);
    
    // Prepare host data - use pinned memory for faster transfers
    let node_props_size = num_nodes * 4 * std::mem::size_of::<f32>();
    let row_ptr_size = (num_nodes + 1) * std::mem::size_of::<u32>();
    let col_idx_size = num_edges * std::mem::size_of::<u32>();
    let edge_props_size = num_edges * 2 * std::mem::size_of::<f32>();
    
    // Allocate pinned host memory
    let mut pinned_node_props: *mut std::ffi::c_void = ptr::null_mut();
    let mut pinned_row_ptr: *mut std::ffi::c_void = ptr::null_mut();
    let mut pinned_col_idx: *mut std::ffi::c_void = ptr::null_mut();
    let mut pinned_edge_props: *mut std::ffi::c_void = ptr::null_mut();
    
    unsafe {
        let result = ffi_bindings::cuda_wrapper_malloc_host(context.ctx, node_props_size, &mut pinned_node_props);
        if result != CudaWrapperError::Success {
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc_host(context.ctx, row_ptr_size, &mut pinned_row_ptr);
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc_host(context.ctx, col_idx_size, &mut pinned_col_idx);
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc_host(context.ctx, edge_props_size, &mut pinned_edge_props);
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_col_idx);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
    }
    
    // Fill pinned memory with data
    unsafe {
        let node_props_ptr = pinned_node_props as *mut f32;
        let mut idx = 0;
        for props in graph.node_props() {
            let avg_luminance: f32 = props.directional_luminance.iter().sum::<f32>() / props.directional_luminance.len() as f32;
            *node_props_ptr.add(idx) = avg_luminance;
            *node_props_ptr.add(idx + 1) = props.reflection;
            *node_props_ptr.add(idx + 2) = props.refraction_index;
            *node_props_ptr.add(idx + 3) = props.default_angle_bin as f32;
            idx += 4;
        }
        
        let row_ptr_ptr = pinned_row_ptr as *mut u32;
        for (i, &val) in graph.row_ptr().iter().enumerate() {
            *row_ptr_ptr.add(i) = val as u32;
        }
        
        let col_idx_ptr = pinned_col_idx as *mut u32;
        for (i, &val) in graph.col_idx().iter().enumerate() {
            *col_idx_ptr.add(i) = val as u32;
        }
        
        let edge_props_ptr = pinned_edge_props as *mut f32;
        let mut idx = 0;
        for props in graph.edge_props() {
            *edge_props_ptr.add(idx) = props.attenuation;
            *edge_props_ptr.add(idx + 1) = props.angle_bin as f32;
            idx += 2;
        }
    }
    
    // Calculate sizes for device memory allocation
    let node_props_size = num_nodes * 4 * std::mem::size_of::<f32>();
    let row_ptr_size = (num_nodes + 1) * std::mem::size_of::<u32>();
    let col_idx_size = num_edges * std::mem::size_of::<u32>();
    let edge_props_size = num_edges * 2 * std::mem::size_of::<f32>();
    
    unsafe {
        // Allocate node properties
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, node_props_size, &mut context.graph_cache.dev_node_props);
        if result != CudaWrapperError::Success {
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        // Allocate CSR data
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, row_ptr_size, &mut context.graph_cache.dev_row_ptr);
        if result != CudaWrapperError::Success {
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, col_idx_size, &mut context.graph_cache.dev_col_idx);
        if result != CudaWrapperError::Success {
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, edge_props_size, &mut context.graph_cache.dev_edge_props);
        if result != CudaWrapperError::Success {
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        // Copy data to device (pinned memory enables faster transfers)
        let result = ffi_bindings::cuda_wrapper_memcpy_htod(
            context.ctx,
            context.graph_cache.dev_node_props,
            pinned_node_props,
            node_props_size
        );
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_col_idx);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_edge_props);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_memcpy_htod(
            context.ctx,
            context.graph_cache.dev_row_ptr,
            pinned_row_ptr,
            row_ptr_size
        );
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_col_idx);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_edge_props);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_memcpy_htod(
            context.ctx,
            context.graph_cache.dev_col_idx,
            pinned_col_idx,
            col_idx_size
        );
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_col_idx);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_edge_props);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_memcpy_htod(
            context.ctx,
            context.graph_cache.dev_edge_props,
            pinned_edge_props,
            edge_props_size
        );
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_col_idx);
            ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_edge_props);
            context.graph_cache.cleanup(context.ctx);
            return Err(result.into());
        }
        
        // Free pinned memory after copy (it's no longer needed)
        ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_node_props);
        ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_row_ptr);
        ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_col_idx);
        ffi_bindings::cuda_wrapper_free_host(context.ctx, pinned_edge_props);
    }
    
    // Update cache metadata
    context.graph_cache.num_nodes = num_nodes;
    context.graph_cache.num_edges = num_edges;
    
    Ok(())
}

pub fn propagate_light_cuda(
    context: &mut CudaContext,
    graph: &Graph,
    source: crate::graph::NodeId,
    initial_bin: crate::graph::AngleBin,
    params: LightParams,
    _pvs: Option<&PVS>,
) -> Result<Vec<f32>, CudaError> {
    // If kernels aren't loaded, fall back to CPU
    if !context.kernels_loaded() {
        return Ok(crate::propagation::propagate_light(graph, source, initial_bin, params));
    }
    
    // Ensure graph data is cached on GPU
    ensure_graph_cached(context, graph)?;
    
    let num_nodes = graph.num_nodes();
    let num_angle_bins = params.num_angle_bins;
    
    // Allocate device memory
    let mut dev_intensities: *mut std::ffi::c_void = ptr::null_mut();
    let mut dev_next_intensities: *mut std::ffi::c_void = ptr::null_mut();
    let mut dev_total_intensity: *mut std::ffi::c_void = ptr::null_mut();
    
    let intensities_size = (num_nodes * num_angle_bins * std::mem::size_of::<f32>()) as usize;
    let total_size = (num_nodes * std::mem::size_of::<f32>()) as usize;
    
    // Allocate working buffers (intensities) - these are per-query, not cached
    unsafe {
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, intensities_size, &mut dev_intensities);
        if result != CudaWrapperError::Success {
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, intensities_size, &mut dev_next_intensities);
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free(context.ctx, dev_intensities);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_malloc(context.ctx, total_size, &mut dev_total_intensity);
        if result != CudaWrapperError::Success {
            ffi_bindings::cuda_wrapper_free(context.ctx, dev_intensities);
            ffi_bindings::cuda_wrapper_free(context.ctx, dev_next_intensities);
            return Err(result.into());
        }
        
        // Initialize intensities to zero using cudaMemset (faster than host-to-device copy)
        let result = ffi_bindings::cuda_wrapper_memset(
            context.ctx,
            dev_intensities,
            0,
            intensities_size
        );
        if result != CudaWrapperError::Success {
            cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
            return Err(result.into());
        }
        
        let result = ffi_bindings::cuda_wrapper_memset(
            context.ctx,
            dev_total_intensity,
            0,
            total_size
        );
        if result != CudaWrapperError::Success {
            cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
            return Err(result.into());
        }
    }
    
    // Launch initialization kernel
    let block_size = 256u32;
    let grid_size = (num_nodes as u32 + block_size - 1) / block_size;
    
    let init_func = context.init_kernel.unwrap();
    
    // Pack kernel arguments correctly for cuLaunchKernel
    // Each argument must be a pointer to the actual value
    let mut num_nodes_u32 = num_nodes as u32;
    let mut num_angle_bins_u32 = num_angle_bins as u32;
    let mut source_u32 = source as u32;
    let mut initial_bin_u32 = initial_bin as u32;
    
    unsafe {
        // CUDA kernel arguments: pointers to device memory and scalar values
        // Use cached graph data
        let mut dev_node_props = context.graph_cache.dev_node_props;
        let mut dev_row_ptr = context.graph_cache.dev_row_ptr;
        let mut dev_col_idx = context.graph_cache.dev_col_idx;
        let mut dev_edge_props = context.graph_cache.dev_edge_props;
        
        let mut kernel_args: Vec<*mut std::ffi::c_void> = vec![
            &mut dev_intensities as *mut _ as *mut std::ffi::c_void,
            &mut dev_total_intensity as *mut _ as *mut std::ffi::c_void,
            &mut dev_node_props as *mut _ as *mut std::ffi::c_void,
            &mut num_nodes_u32 as *mut _ as *mut std::ffi::c_void,
            &mut num_angle_bins_u32 as *mut _ as *mut std::ffi::c_void,
            &mut source_u32 as *mut _ as *mut std::ffi::c_void,
            &mut initial_bin_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        
        let result = ffi_bindings::cuda_wrapper_launch_kernel(
            context.ctx,
            init_func,
            grid_size, 1, 1,
            block_size, 1, 1,
            0,
            kernel_args.as_mut_ptr(),
            kernel_args.len(),
        );
        if result != CudaWrapperError::Success {
            cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
            return Err(result.into());
        }
    }
    
    // Launch propagation iterations
    let propagate_func = context.propagate_kernel.unwrap();
    let total_pairs = num_nodes * num_angle_bins;
    let prop_block_size = 256u32;
    let prop_grid_size = (total_pairs as u32 + prop_block_size - 1) / prop_block_size;
    
    let mut use_first = true;
    let mut params_k = params.k;
    let mut params_min_intensity = params.min_intensity;
    
    for _depth in 0..params.max_depth {
        // Clear next buffer using cudaMemset (faster than host-to-device copy)
        unsafe {
            let next_buffer = if use_first { dev_next_intensities } else { dev_intensities };
            let result = ffi_bindings::cuda_wrapper_memset(
                context.ctx,
                next_buffer,
                0,
                intensities_size
            );
            if result != CudaWrapperError::Success {
                cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
                return Err(result.into());
            }
        }
        
        // Pack kernel arguments - CUDA expects pointers to the actual values
        // Create mutable variables to hold pointer values for kernel arguments
        let mut current_ptr = if use_first { dev_intensities } else { dev_next_intensities };
        let mut next_ptr = if use_first { dev_next_intensities } else { dev_intensities };
        
        unsafe {
            // Use cached graph data
            let mut dev_node_props = context.graph_cache.dev_node_props;
            let mut dev_row_ptr = context.graph_cache.dev_row_ptr;
            let mut dev_col_idx = context.graph_cache.dev_col_idx;
            let mut dev_edge_props = context.graph_cache.dev_edge_props;
            
            // For device pointers, pass pointer to the pointer value
            // For scalars, pass pointer to the scalar value
            let mut kernel_args: Vec<*mut std::ffi::c_void> = vec![
                &mut current_ptr as *mut _ as *mut std::ffi::c_void,
                &mut next_ptr as *mut _ as *mut std::ffi::c_void,
                &mut dev_total_intensity as *mut _ as *mut std::ffi::c_void,
                &mut dev_node_props as *mut _ as *mut std::ffi::c_void,
                &mut dev_row_ptr as *mut _ as *mut std::ffi::c_void,
                &mut dev_col_idx as *mut _ as *mut std::ffi::c_void,
                &mut dev_edge_props as *mut _ as *mut std::ffi::c_void,
                &mut num_nodes_u32 as *mut _ as *mut std::ffi::c_void,
                &mut num_angle_bins_u32 as *mut _ as *mut std::ffi::c_void,
                &mut params_k as *mut _ as *mut std::ffi::c_void,
                &mut params_min_intensity as *mut _ as *mut std::ffi::c_void,
            ];
            
            let result = ffi_bindings::cuda_wrapper_launch_kernel(
                context.ctx,
                propagate_func,
                prop_grid_size, 1, 1,
                prop_block_size, 1, 1,
                0,
                kernel_args.as_mut_ptr(),
                kernel_args.len(),
            );
            if result != CudaWrapperError::Success {
                cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
                return Err(result.into());
            }
        }
        
        use_first = !use_first;
    }
    
    // Synchronize
    unsafe {
        let result = ffi_bindings::cuda_wrapper_synchronize(context.ctx);
        if result != CudaWrapperError::Success {
            cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
            return Err(result.into());
        }
    }
    
    // Copy results back
    let mut result = vec![0.0f32; num_nodes];
    unsafe {
        let copy_result = ffi_bindings::cuda_wrapper_memcpy_dtoh(
            context.ctx,
            result.as_mut_ptr() as *mut std::ffi::c_void,
            dev_total_intensity,
            total_size
        );
        cleanup_device_memory(context.ctx, dev_intensities, dev_next_intensities, dev_total_intensity);
        
        if copy_result != CudaWrapperError::Success {
            return Err(copy_result.into());
        }
    }
    
    Ok(result)
}

unsafe fn cleanup_device_memory(ctx: FfiCudaContext, ptr1: *mut std::ffi::c_void, ptr2: *mut std::ffi::c_void, ptr3: *mut std::ffi::c_void) {
    if !ptr1.is_null() {
        let _ = ffi_bindings::cuda_wrapper_free(ctx, ptr1);
    }
    if !ptr2.is_null() {
        let _ = ffi_bindings::cuda_wrapper_free(ctx, ptr2);
    }
    if !ptr3.is_null() {
        let _ = ffi_bindings::cuda_wrapper_free(ctx, ptr3);
    }
}

