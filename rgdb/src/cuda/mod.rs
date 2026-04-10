/// CUDA GPU acceleration for RGDB
/// 
/// This module provides GPU-accelerated propagation kernels.
/// Requires CUDA toolkit and NVIDIA GPU.
/// 
/// MIGRATION NOTE: We're migrating from rustacuda to cudarc due to
/// context stack management issues in rustacuda. The cudarc implementation
/// provides better resource management and avoids cleanup crashes.

mod kernels;
mod cudarc_impl;
mod ffi_bindings;
mod direct_ffi_impl;

// Use direct FFI implementation (most reliable)
// For now, always try FFI first, fallback to cudarc if not available
pub use direct_ffi_impl::*;

// Legacy rustacuda implementation (kept for reference, not used)
// Commented out to avoid conflicts - can be enabled if needed
/*
mod rustacuda_impl {
use crate::graph::Graph;
use crate::propagation::LightParams;
use crate::pvs::PVS;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use rustacuda::launch;
use thiserror::Error;

/// CUDA context and device management
/// 
/// IMPORTANT: Drop order matters. Fields are dropped in reverse order:
/// 1. module (must be dropped first)
/// 2. _device
/// 3. _context (must be dropped last to keep context active during module cleanup)
/// 
/// This struct manages a persistent CUDA context that stays alive for the lifetime
/// of the application. The context is created with `create_and_push` which pushes it
/// onto CUDA's context stack. As long as this struct is alive, the context remains
/// active, allowing resources to be created and cleaned up safely.
pub struct CudaContext {
    module: Option<Module>,  // Drop first
    _device: Device,         // Drop second
    _context: Context,       // Drop last - keeps context active during module cleanup
}

impl CudaContext {
    /// Initialize CUDA context and load kernels
    pub fn new() -> Result<Self, CudaError> {
        // Initialize CUDA API
        rustacuda::init(CudaFlags::empty())
            .map_err(|e| CudaError::Runtime(format!("Failed to initialize CUDA: {}", e)))?;
        
        // Get the first device (device 0)
        // Note: rustacuda doesn't provide get_count, so we try to get device 0
        // and handle errors appropriately
        let device_id = 0;
        let device = Device::get_device(device_id)
            .map_err(|e| CudaError::Runtime(format!("Failed to get device {}: {}", device_id, e)))?;
        
        // Create context
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )
        .map_err(|e| CudaError::Runtime(format!("Failed to create context: {}", e)))?;
        
        // Try to load kernel module
        use std::ffi::CString;
        let module = match kernels::load_ptx() {
            Ok(ptx_str) => {
                match CString::new(ptx_str) {
                    Ok(ptx_cstring) => {
                        match Module::load_from_string(ptx_cstring.as_c_str()) {
                            Ok(m) => {
                                println!("CUDA kernels loaded successfully");
                                Some(m)
                            }
                            Err(e) => {
                                println!("Warning: Failed to load CUDA kernels: {}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        println!("Warning: Failed to create CString from PTX: {}", e);
                        None
                    }
                }
            }
            Err(_) => {
                println!("Warning: PTX file not found, using CPU fallback");
                None
            }
        };
        
        Ok(Self {
            module,      // Drop first
            _device: device,  // Drop second  
            _context: context,  // Drop last - keeps context active during cleanup
        })
    }
    
    /// Check if CUDA is available
    pub fn is_available() -> bool {
        rustacuda::init(CudaFlags::empty()).is_ok()
    }
    
    /// Check if kernels are loaded
    pub fn kernels_loaded(&self) -> bool {
        self.module.is_some()
    }
    
    /// Get device name
    pub fn device_name(&self) -> Result<String, CudaError> {
        Ok(format!("CUDA Device 0"))
    }
}

/// Errors for CUDA operations
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

/// GPU-accelerated propagation using CUDA
/// 
/// This implementation:
/// 1. Copies graph data to GPU memory
/// 2. Launches propagation kernels
/// 3. Copies results back to CPU
/// 
/// CRITICAL: Context must remain active throughout this function's execution
/// and during resource cleanup. The context is stored in CudaContext and
/// must outlive all resources created here.
pub fn propagate_light_cuda(
    context: &CudaContext,
    graph: &Graph,
    source: crate::graph::NodeId,
    initial_bin: crate::graph::AngleBin,
    params: LightParams,
    _pvs: Option<&PVS>,
) -> Result<Vec<f32>, CudaError> {
    // If kernels aren't loaded, fall back to CPU
    if context.module.is_none() {
        return Ok(crate::propagation::propagate_light(graph, source, initial_bin, params));
    }
    
    // CRITICAL: Ensure context is active before creating any resources
    // The context from CudaContext should already be active (pushed by create_and_push)
    // but we need to ensure it stays active during resource cleanup
    
    // GPU path: kernels are loaded, proceed with CUDA execution on GPU
    
    let num_nodes = graph.num_nodes();
    let num_angle_bins = params.num_angle_bins;
    let module = context.module.as_ref().unwrap();
    
    // Allocate and copy intensities to GPU (per node, per angle bin)
    let mut host_intensities = vec![0.0f32; num_nodes * num_angle_bins];
    let mut dev_intensities = DeviceBuffer::from_slice(&host_intensities)
        .map_err(|e| CudaError::MemoryAllocation(format!("Intensities allocation: {}", e)))?;
    
    // Allocate next iteration intensities
    let mut dev_next_intensities = DeviceBuffer::from_slice(&vec![0.0f32; num_nodes * num_angle_bins])
        .map_err(|e| CudaError::MemoryAllocation(format!("Next intensities allocation: {}", e)))?;
    
    // Allocate total intensity output buffer on GPU
    let mut dev_total_intensity = DeviceBuffer::from_slice(&vec![0.0f32; num_nodes])
        .map_err(|e| CudaError::MemoryAllocation(format!("Total intensity allocation: {}", e)))?;
    
    // Prepare and copy node properties to GPU
    // Use average luminance from directional_luminance array
    let mut host_node_props = Vec::with_capacity(num_nodes * 4);
    for props in graph.node_props() {
        // Calculate average luminance from directional_luminance
        let avg_luminance: f32 = props.directional_luminance.iter().sum::<f32>() / props.directional_luminance.len() as f32;
        host_node_props.push(avg_luminance);
        host_node_props.push(props.reflection);
        host_node_props.push(props.refraction_index);
        host_node_props.push(props.default_angle_bin as f32);
    }
    let mut dev_node_props = DeviceBuffer::from_slice(&host_node_props)
        .map_err(|e| CudaError::MemoryAllocation(format!("Node props allocation: {}", e)))?;
    
    // Copy CSR structure to GPU
    let host_row_ptr: Vec<u32> = graph.row_ptr().iter().map(|&x| x as u32).collect();
    let mut dev_row_ptr = DeviceBuffer::from_slice(&host_row_ptr)
        .map_err(|e| CudaError::MemoryAllocation(format!("Row ptr allocation: {}", e)))?;
    
    let host_col_idx: Vec<u32> = graph.col_idx().to_vec();
    let mut dev_col_idx = DeviceBuffer::from_slice(&host_col_idx)
        .map_err(|e| CudaError::MemoryAllocation(format!("Col idx allocation: {}", e)))?;
    
    // Copy edge properties to GPU
    let mut host_edge_props = Vec::with_capacity(graph.num_edges() * 2);
    for props in graph.edge_props() {
        host_edge_props.push(props.attenuation);
        host_edge_props.push(props.angle_bin as f32);
    }
    let mut dev_edge_props = DeviceBuffer::from_slice(&host_edge_props)
        .map_err(|e| CudaError::MemoryAllocation(format!("Edge props allocation: {}", e)))?;
    
    // Create stream for async operations
    // CRITICAL: Must synchronize before stream is dropped
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
        .map_err(|e| CudaError::Runtime(format!("Failed to create stream: {}", e)))?;
    
    // Get kernel functions
    use std::ffi::CString;
    let init_func_name = CString::new("init_propagation_kernel")
        .map_err(|e| CudaError::KernelsNotLoaded(format!("Function name: {}", e)))?;
    let init_func = module.get_function(init_func_name.as_c_str())
        .map_err(|e| CudaError::KernelsNotLoaded(format!("init_propagation_kernel: {}", e)))?;
    
    let propagate_func_name = CString::new("propagate_iteration_kernel")
        .map_err(|e| CudaError::KernelsNotLoaded(format!("Function name: {}", e)))?;
    let propagate_func = module.get_function(propagate_func_name.as_c_str())
        .map_err(|e| CudaError::KernelsNotLoaded(format!("propagate_iteration_kernel: {}", e)))?;
    
    // Launch initialization kernel
    let block_size = 256u32;
    let grid_size = (num_nodes as u32 + block_size - 1) / block_size;
    
    unsafe {
        launch!(init_func<<<grid_size, block_size, 0, stream>>>(
            dev_intensities.as_device_ptr(),
            dev_total_intensity.as_device_ptr(),
            dev_node_props.as_device_ptr(),
            num_nodes as u32,
            num_angle_bins as u32,
            source,
            initial_bin
        ))
        .map_err(|e| CudaError::KernelLaunch(format!("Init kernel: {}", e)))?;
    }
    
    // Launch propagation iterations
    let total_pairs = num_nodes * num_angle_bins;
    let prop_block_size = 256u32;
    let prop_grid_size = (total_pairs as u32 + prop_block_size - 1) / prop_block_size;
    
    // Use references for ping-pong buffers
    let mut use_first = true;
    
    for depth in 0..params.max_depth {
        // Clear next buffer before each iteration by creating a new zero buffer
        // In production, you'd use cudaMemset, but for now we'll rely on the kernel
        // to handle this properly
        
        // Use mutable references for ping-pong buffers
        let (current, next) = if use_first {
            (&mut dev_intensities, &mut dev_next_intensities)
        } else {
            (&mut dev_next_intensities, &mut dev_intensities)
        };
        
        unsafe {
            launch!(propagate_func<<<prop_grid_size, prop_block_size, 0, stream>>>(
                current.as_device_ptr(),
                next.as_device_ptr(),
                dev_total_intensity.as_device_ptr(),
                dev_node_props.as_device_ptr(),
                dev_row_ptr.as_device_ptr(),
                dev_col_idx.as_device_ptr(),
                dev_edge_props.as_device_ptr(),
                num_nodes as u32,
                num_angle_bins as u32,
                params.k,
                params.min_intensity
            ))
            .map_err(|e| CudaError::KernelLaunch(format!("Propagation kernel depth {}: {}", depth, e)))?;
        }
        
        // Swap buffers for next iteration
        use_first = !use_first;
    }
    
    // CRITICAL: Synchronize stream before copying results
    // This ensures all kernel launches complete before we access results
    stream.synchronize()
        .map_err(|e| CudaError::Runtime(format!("Stream sync failed: {}", e)))?;
    
    // Copy results back to host BEFORE any resources are dropped
    let mut result = vec![0.0f32; num_nodes];
    dev_total_intensity.copy_to(&mut result)
        .map_err(|e| CudaError::MemoryCopy(format!("Result copy: {}", e)))?;
    
    // CRITICAL: Final synchronization to ensure copy completes
    // All operations must complete while context is still active
    stream.synchronize()
        .map_err(|e| CudaError::Runtime(format!("Final sync failed: {}", e)))?;
    
    // RESOURCE CLEANUP:
    // When this function returns, Rust will drop resources in reverse order:
    // 1. stream (dropped last)
    // 2. dev_edge_props, dev_col_idx, dev_row_ptr, dev_node_props, dev_total_intensity, dev_next_intensities, dev_intensities
    //
    // The context MUST remain active during this cleanup. Since context is stored
    // in CudaContext (passed as &CudaContext), it should remain alive. However,
    // rustacuda's context stack management may cause issues if the context was
    // popped from the stack.
    //
    // WORKAROUND: We've synchronized the stream twice to ensure all operations
    // complete. The context in CudaContext should keep it active, but if cleanup
    // still fails, we may need to switch to a different CUDA binding library.
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_availability() {
        let available = CudaContext::is_available();
        if available {
            println!("CUDA is available");
        } else {
            println!("CUDA is not available (this is OK if no GPU present)");
        }
    }
    
    #[test]
    fn test_cuda_context_creation() {
        match CudaContext::new() {
            Ok(ctx) => {
                println!("CUDA context created successfully");
                println!("Kernels loaded: {}", ctx.kernels_loaded());
                if let Ok(name) = ctx.device_name() {
                    println!("Device: {}", name);
                }
            }
            Err(e) => {
                println!("CUDA context creation failed: {} (expected if no GPU)", e);
            }
        }
    }
    
    #[test]
    fn test_cuda_memory_operations() {
        if let Ok(ctx) = CudaContext::new() {
            let host_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
            match DeviceBuffer::from_slice(&host_data) {
                Ok(dev_data) => {
                    let mut result = vec![0.0f32; host_data.len()];
                    match dev_data.copy_to(&mut result) {
                        Ok(_) => {
                            assert_eq!(result, host_data);
                            println!("GPU memory operations work correctly");
                        }
                        Err(e) => println!("Failed to copy from device: {}", e),
                    }
                }
                Err(e) => println!("Failed to copy to device: {}", e),
            }
        }
    }
}
*/ // End of rustacuda_impl module (commented out)
