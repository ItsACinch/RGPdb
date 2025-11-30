// Build script to compile CUDA kernels

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/propagate_kernel.cu");
    
    // Check if nvcc is available
    let nvcc_output = Command::new("nvcc")
        .arg("--version")
        .output();
    
    if nvcc_output.is_err() {
        println!("cargo:warning=nvcc not found, CUDA kernels will not be compiled");
        println!("cargo:warning=Install CUDA toolkit and ensure nvcc is in PATH");
        return;
    }
    
    // Compile CUDA kernel to PTX
    let kernel_dir = PathBuf::from("kernels");
    let cu_file = kernel_dir.join("propagate_kernel.cu");
    let ptx_file = kernel_dir.join("propagate_kernel.ptx");
    
    if !cu_file.exists() {
        println!("cargo:warning=CUDA kernel source not found: {:?}", cu_file);
        return;
    }
    
    let output = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cu_file)
        .arg("-arch=sm_75") // Support compute capability 7.5+
        .output();
    
    match output {
        Ok(result) => {
            if result.status.success() {
                println!("cargo:warning=CUDA kernel compiled successfully: {:?}", ptx_file);
            } else {
                let stderr = String::from_utf8_lossy(&result.stderr);
                println!("cargo:warning=Failed to compile CUDA kernel: {}", stderr);
                println!("cargo:warning=You may need to set up Visual Studio build tools");
            }
        }
        Err(e) => {
            println!("cargo:warning=Failed to run nvcc: {}", e);
            println!("cargo:warning=CUDA kernel compilation skipped");
        }
    }
}

