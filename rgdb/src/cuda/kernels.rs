/// CUDA kernel loading
/// 
/// Loads PTX files compiled from CUDA kernel sources.

use std::path::Path;

/// Try to load PTX from file
pub fn load_ptx() -> Result<String, std::io::Error> {
    // Try multiple possible paths
    let paths = [
        "kernels/propagate_kernel.ptx",
        "../kernels/propagate_kernel.ptx",
        "./kernels/propagate_kernel.ptx",
    ];
    
    for path in &paths {
        if Path::new(path).exists() {
            return std::fs::read_to_string(path);
        }
    }
    
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("PTX file not found. Tried: {:?}. Compile kernels/propagate_kernel.cu with: nvcc -ptx propagate_kernel.cu -o propagate_kernel.ptx", paths)
    ))
}

