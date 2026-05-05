/// CUDA implementation using cudarc (modern CUDA bindings)
/// 
/// This replaces the rustacuda implementation to fix context management issues.
/// cudarc provides better resource management and avoids context stack problems.

use crate::graph::Graph;
use crate::propagation::LightParams;
use crate::pvs::PVS;
use thiserror::Error;
use std::sync::Arc;

use cudarc::driver::safe::{CudaContext as CudaContextDriver, CudaStream, CudaModule, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::driver::result::DriverError;

/// CUDA context using cudarc
pub struct CudaContext {
    ctx: Arc<CudaContextDriver>,
    module: Option<Arc<CudaModule>>,
    init_kernel: Option<CudaFunction>,
    propagate_kernel: Option<CudaFunction>,
}

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

impl From<DriverError> for CudaError {
    fn from(e: DriverError) -> Self {
        CudaError::Runtime(format!("{}", e))
    }
}

impl CudaContext {
    /// Find the first NVIDIA GPU device and return its context
    fn find_nvidia_device() -> Result<Arc<CudaContextDriver>, CudaError> {
        // Try to enumerate devices - cudarc doesn't expose device_count directly,
        // so we'll try devices 0-7 (most systems won't have more than 8 GPUs)
        for device_id in 0..8 {
            if let Ok(ctx) = CudaContextDriver::new(device_id) {
                // Check if this is an NVIDIA device by checking the name
                if let Ok(name) = ctx.name() {
                    if name.to_uppercase().contains("NVIDIA") {
                        eprintln!("Found NVIDIA GPU at device {}: {}", device_id, name);
                        return Ok(ctx);
                    }
                    eprintln!("Device {}: {} (not NVIDIA, skipping)", device_id, name);
                }
            }
        }
        
        // If no NVIDIA device found, fall back to device 0
        eprintln!("No NVIDIA GPU found, falling back to device 0");
        CudaContextDriver::new(0)
            .map_err(|e| CudaError::Runtime(format!("Failed to get CUDA device 0: {}", e)))
    }
    
    /// Initialize CUDA context and load kernels using cudarc
    pub fn new() -> Result<Self, CudaError> {
        // Find the NVIDIA GPU device and get its context
        let ctx = Self::find_nvidia_device()?;
        
        // Load PTX kernels
        let ptx_str = match crate::cuda::kernels::load_ptx() {
            Ok(s) => s,
            Err(_) => {
                return Ok(Self {
                    ctx,
                    module: None,
                    init_kernel: None,
                    propagate_kernel: None,
                });
            }
        };
        
        // Load PTX module (convert String to Ptx type)
        let module = ctx.load_module(ptx_str.into())
            .map_err(|e| CudaError::PtxCompilation(format!("Failed to load PTX: {}", e)))?;
        
        // Get kernel functions from module
        let init_kernel = module.load_function("init_propagation_kernel")
            .ok();
        
        let propagate_kernel = module.load_function("propagate_iteration_kernel")
            .ok();
        
        if init_kernel.is_none() || propagate_kernel.is_none() {
            return Ok(Self {
                ctx,
                module: Some(module),
                init_kernel: None,
                propagate_kernel: None,
            });
        }
        
        Ok(Self {
            ctx,
            module: Some(module),
            init_kernel,
            propagate_kernel,
        })
    }
    
    /// Check if CUDA is available
    pub fn is_available() -> bool {
        // Try to find any CUDA device (check devices 0-7)
        for device_id in 0..8 {
            if CudaContextDriver::new(device_id).is_ok() {
                return true;
            }
        }
        false
    }
    
    /// Check if kernels are loaded
    pub fn kernels_loaded(&self) -> bool {
        self.init_kernel.is_some() && self.propagate_kernel.is_some()
    }
    
    /// Get device name
    pub fn device_name(&self) -> Result<String, CudaError> {
        self.ctx.name()
            .map_err(|e| CudaError::Runtime(format!("Failed to get device name: {}", e)))
    }
}

/// GPU-accelerated propagation using cudarc
pub fn propagate_light_cuda(
    context: &CudaContext,
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
    
    let num_nodes = graph.num_nodes();
    let num_angle_bins = params.num_angle_bins;
    
    // Get kernel functions
    let init_func = context.init_kernel.as_ref().unwrap();
    let propagate_func = context.propagate_kernel.as_ref().unwrap();
    
    // Use the default stream (more reliable than creating a new stream)
    let stream = context.ctx.default_stream();
    
    // Allocate device memory using cudarc stream
    // NOTE: The CUDA_ERROR_ILLEGAL_ADDRESS suggests the context might not be active.
    // This is a known issue with cudarc - the context needs to be properly initialized.
    let dev_intensities = stream
        .alloc_zeros::<f32>(num_nodes * num_angle_bins)
        .map_err(|e| CudaError::MemoryAllocation(format!("Intensities: {}", e)))?;
    
    let dev_next_intensities = stream
        .alloc_zeros::<f32>(num_nodes * num_angle_bins)
        .map_err(|e| CudaError::MemoryAllocation(format!("Next intensities: {}", e)))?;
    
    let dev_total_intensity = stream
        .alloc_zeros::<f32>(num_nodes)
        .map_err(|e| CudaError::MemoryAllocation(format!("Total intensity: {}", e)))?;
    
    // Prepare and copy node properties
    let mut host_node_props = Vec::with_capacity(num_nodes * 4);
    for props in graph.node_props() {
        let avg_luminance: f32 = props.directional_luminance.iter().sum::<f32>() / props.directional_luminance.len() as f32;
        host_node_props.push(avg_luminance);
        host_node_props.push(props.reflection);
        host_node_props.push(props.refraction_index);
        host_node_props.push(props.default_angle_bin as f32);
    }
    let dev_node_props = stream
        .clone_htod(&host_node_props)
        .map_err(|e| CudaError::MemoryAllocation(format!("Node props: {}", e)))?;
    
    // Copy CSR structure
    let host_row_ptr: Vec<u32> = graph.row_ptr().iter().map(|&x| x as u32).collect();
    let dev_row_ptr = stream
        .clone_htod(&host_row_ptr)
        .map_err(|e| CudaError::MemoryAllocation(format!("Row ptr: {}", e)))?;
    
    let host_col_idx: Vec<u32> = graph.col_idx().to_vec();
    let dev_col_idx = stream
        .clone_htod(&host_col_idx)
        .map_err(|e| CudaError::MemoryAllocation(format!("Col idx: {}", e)))?;
    
    // Copy edge properties
    let mut host_edge_props = Vec::with_capacity(graph.num_edges() * 2);
    for props in graph.edge_props() {
        host_edge_props.push(props.attenuation);
        host_edge_props.push(props.angle_bin as f32);
    }
    let dev_edge_props = stream
        .clone_htod(&host_edge_props)
        .map_err(|e| CudaError::MemoryAllocation(format!("Edge props: {}", e)))?;
    
    // Launch initialization kernel using launch_builder pattern
    let block_size = 256u32;
    let grid_size = (num_nodes as u32 + block_size - 1) / block_size;
    
    unsafe {
        let mut launch_args = stream.launch_builder(init_func);
        launch_args
            .arg(&dev_intensities)
            .arg(&dev_total_intensity)
            .arg(&dev_node_props)
            .arg(&(num_nodes as u32))
            .arg(&(num_angle_bins as u32))
            .arg(&source)
            .arg(&initial_bin)
            .launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| CudaError::KernelLaunch(format!("Init kernel: {}", e)))?;
    }
    
    // Launch propagation iterations
    let total_pairs = num_nodes * num_angle_bins;
    let prop_block_size = 256u32;
    let prop_grid_size = (total_pairs as u32 + prop_block_size - 1) / prop_block_size;
    
    let mut use_first = true;
    for _depth in 0..params.max_depth {
        let (current, next) = if use_first {
            (&dev_intensities, &dev_next_intensities)
        } else {
            (&dev_next_intensities, &dev_intensities)
        };
        
        unsafe {
            let mut launch_args = stream.launch_builder(propagate_func);
            launch_args
                .arg(current)
                .arg(next)
                .arg(&dev_total_intensity)
                .arg(&dev_node_props)
                .arg(&dev_row_ptr)
                .arg(&dev_col_idx)
                .arg(&dev_edge_props)
                .arg(&(num_nodes as u32))
                .arg(&(num_angle_bins as u32))
                .arg(&params.k)
                .arg(&params.min_intensity)
                .launch(LaunchConfig {
                    grid_dim: (prop_grid_size, 1, 1),
                    block_dim: (prop_block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudaError::KernelLaunch(format!("Propagation kernel: {}", e)))?;
        }
        
        use_first = !use_first;
    }
    
    // Synchronize stream
    stream
        .synchronize()
        .map_err(|e| CudaError::Runtime(format!("Stream sync failed: {}", e)))?;
    
    // Copy results back
    let result = stream
        .clone_dtoh(&dev_total_intensity)
        .map_err(|e| CudaError::MemoryCopy(format!("Result copy: {}", e)))?;
    
    Ok(result)
}
