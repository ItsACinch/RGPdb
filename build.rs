// Build script to compile CUDA kernels

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::fs;
use std::io::Write;

fn main() {
    println!("cargo:rerun-if-changed=kernels/propagate_kernel.cu");
    println!("cargo:rerun-if-changed=cuda_wrapper/cuda_wrapper.cu");
    println!("cargo:rerun-if-changed=cuda_wrapper/cuda_wrapper.h");
    
    // Compile CUDA kernel to PTX
    // NOTE: CUDA compilation failure is non-fatal - CPU fallback is always available
    let kernel_dir = PathBuf::from("kernels");
    let cu_file = kernel_dir.join("propagate_kernel.cu");
    let ptx_file = kernel_dir.join("propagate_kernel.ptx");
    
    if !cu_file.exists() {
        println!("cargo:warning=rgdb@0.1.0: CUDA kernel source not found: {:?}", cu_file);
        println!("cargo:warning=rgdb@0.1.0: CUDA acceleration will not be available");
        return;
    }
    
    // On Windows, nvcc needs Visual Studio environment
    #[cfg(target_os = "windows")]
    {
        compile_cuda_windows(&cu_file, &ptx_file);
        compile_cuda_wrapper_windows();
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        compile_cuda_unix(&cu_file, &ptx_file);
        compile_cuda_wrapper_unix();
    }
    
    // Link CUDA libraries
    link_cuda_libraries();
}

#[cfg(target_os = "windows")]
fn compile_cuda_windows(cu_file: &std::path::Path, ptx_file: &std::path::Path) {
    // Try to find vcvars64.bat for Visual Studio
    // Check Professional first (most common for development), then Enterprise, Community, BuildTools
    let vs_paths = vec![
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ];
    
    let mut vcvars_path: Option<String> = None;
    for path in vs_paths {
        if std::path::Path::new(path).exists() {
            vcvars_path = Some(path.to_string());
            break;
        }
    }
    
    // Also try PowerShell search as fallback
    if vcvars_path.is_none() {
        let search_output = Command::new("powershell")
            .arg("-Command")
            .arg(r#"Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName"#)
            .output();
        
        if let Ok(output) = search_output {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() && std::path::Path::new(&path_str).exists() {
                vcvars_path = Some(path_str);
            }
        }
    }
    
    if let Some(vcvars) = vcvars_path {
        // Create a temporary batch file to set up environment and compile
        // This ensures environment variables persist for nvcc
        let workspace_dir = env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| env::current_dir().unwrap().to_string_lossy().to_string());
        
        let temp_batch = std::env::temp_dir().join(format!("rgdb_build_{}.bat", std::process::id()));
        let batch_content = format!(
            r#"@echo off
setlocal EnableDelayedExpansion
call "{}"
cd /d "{}"
nvcc -ptx -o "{}" "{}" -arch=sm_75
if errorlevel 1 exit /b 1
"#,
            vcvars,
            workspace_dir,
            ptx_file.display(),
            cu_file.display()
        );
        
        // Write batch file
        if let Ok(mut file) = fs::File::create(&temp_batch) {
            if file.write_all(batch_content.as_bytes()).is_ok() {
                let output = Command::new("cmd")
                    .arg("/C")
                    .arg(&temp_batch)
                    .output();
                
                // Clean up temp file
                let _ = fs::remove_file(&temp_batch);
                
                match output {
                    Ok(result) => {
                        if result.status.success() {
                            println!("cargo:warning=rgdb@0.1.0: CUDA kernel compiled successfully");
                        } else {
                            // Non-fatal: CUDA compilation failure doesn't prevent build
                            let stderr = String::from_utf8_lossy(&result.stderr);
                            let stdout = String::from_utf8_lossy(&result.stdout);
                            
                            // Only show detailed errors in verbose mode
                            if env::var("RUSTC_LOG").is_ok() || env::var("CARGO_VERBOSE").is_ok() {
                                if !stderr.is_empty() {
                                    eprintln!("CUDA compilation stderr: {}", stderr);
                                }
                                if !stdout.is_empty() {
                                    eprintln!("CUDA compilation stdout: {}", stdout);
                                }
                            }
                            
                            println!("cargo:warning=rgdb@0.1.0: CUDA kernel compilation failed (non-fatal)");
                            println!("cargo:warning=rgdb@0.1.0: CPU fallback will be used - this is expected if CUDA toolkit is not configured");
                        }
                    }
                    Err(e) => {
                        println!("cargo:warning=rgdb@0.1.0: Failed to run nvcc: {} (non-fatal)", e);
                        println!("cargo:warning=rgdb@0.1.0: CPU fallback will be used");
                    }
                }
                return;
            }
        }
    }
    
    // Fallback: Try direct nvcc call (might work if environment is already set)
    let output = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cu_file)
        .arg("-arch=sm_75")
        .output();
    
    match output {
        Ok(result) => {
            if result.status.success() {
                println!("cargo:warning=rgdb@0.1.0: CUDA kernel compiled successfully");
            } else {
                // Non-fatal: CUDA compilation failure doesn't prevent build
                let stderr = String::from_utf8_lossy(&result.stderr);
                println!("cargo:warning=rgdb@0.1.0: CUDA kernel compilation failed (non-fatal)");
                if env::var("RUSTC_LOG").is_ok() || env::var("CARGO_VERBOSE").is_ok() {
                    if !stderr.is_empty() {
                        eprintln!("CUDA compilation error: {}", stderr);
                    }
                }
                println!("cargo:warning=rgdb@0.1.0: CPU fallback will be used - this is expected if CUDA toolkit is not configured");
            }
        }
        Err(_) => {
            println!("cargo:warning=rgdb@0.1.0: nvcc not found in PATH (non-fatal)");
            println!("cargo:warning=rgdb@0.1.0: CPU fallback will be used - this is expected if CUDA toolkit is not installed");
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn compile_cuda_unix(cu_file: &std::path::Path, ptx_file: &std::path::Path) {
    let output = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cu_file)
        .arg("-arch=sm_75")
        .output();
    
    match output {
        Ok(result) => {
            if result.status.success() {
                println!("cargo:warning=rgdb@0.1.0: CUDA kernel compiled successfully");
            } else {
                // Non-fatal: CUDA compilation failure doesn't prevent build
                let stderr = String::from_utf8_lossy(&result.stderr);
                println!("cargo:warning=rgdb@0.1.0: CUDA kernel compilation failed (non-fatal)");
                if env::var("RUSTC_LOG").is_ok() || env::var("CARGO_VERBOSE").is_ok() {
                    if !stderr.is_empty() {
                        eprintln!("CUDA compilation error: {}", stderr);
                    }
                }
                println!("cargo:warning=rgdb@0.1.0: CPU fallback will be used");
            }
        }
        Err(_) => {
            println!("cargo:warning=rgdb@0.1.0: nvcc not found (non-fatal)");
            println!("cargo:warning=rgdb@0.1.0: CPU fallback will be used - this is expected if CUDA toolkit is not installed");
        }
    }
}

#[cfg(target_os = "windows")]
fn compile_cuda_wrapper_windows() {
    let wrapper_dir = PathBuf::from("cuda_wrapper");
    let wrapper_cu = wrapper_dir.join("cuda_wrapper.cu");
    
    if !wrapper_cu.exists() {
        println!("cargo:warning=rgdb@0.1.0: CUDA wrapper source not found: {:?}", wrapper_cu);
        return;
    }
    
    // Find CUDA toolkit path
    let cuda_paths = vec![
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    ];
    
    let mut cuda_path: Option<String> = None;
    for path in cuda_paths {
        if std::path::Path::new(path).exists() {
            cuda_path = Some(path.to_string());
            break;
        }
    }
    
    let cuda_path = match cuda_path {
        Some(p) => p,
        None => {
            println!("cargo:warning=rgdb@0.1.0: CUDA toolkit not found, CUDA wrapper will not be compiled");
            return;
        }
    };
    
    // Find Visual Studio vcvars64.bat
    let vs_paths = vec![
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ];
    
    let mut vcvars_path: Option<String> = None;
    for path in vs_paths {
        if std::path::Path::new(path).exists() {
            vcvars_path = Some(path.to_string());
            break;
        }
    }
    
    let vcvars_path = match vcvars_path {
        Some(p) => p,
        None => {
            println!("cargo:warning=rgdb@0.1.0: Visual Studio not found, CUDA wrapper will not be compiled");
            return;
        }
    };
    
    // Create batch file to compile CUDA wrapper as static library
    let out_dir = env::var("OUT_DIR").unwrap();
    let batch_file = PathBuf::from(&out_dir).join("compile_cuda_wrapper.bat");
    let mut batch = fs::File::create(&batch_file).expect("Failed to create batch file");
    
    let lib_file = PathBuf::from(&out_dir).join("cuda_wrapper.lib");
    let obj_file = PathBuf::from(&out_dir).join("cuda_wrapper.obj");
    
    writeln!(batch, "@echo off").unwrap();
    writeln!(batch, "call \"{}\"", vcvars_path).unwrap();
    writeln!(batch, "set CUDA_PATH={}", cuda_path).unwrap();
    writeln!(batch, "set PATH=%CUDA_PATH%\\bin;%PATH%").unwrap();
    // Compile CUDA code to object file
    writeln!(batch, "\"{}\\bin\\nvcc.exe\" -c -o \"{}\" \"{}\" -arch=sm_75",
        cuda_path,
        obj_file.to_string_lossy(),
        wrapper_cu.to_string_lossy()
    ).unwrap();
    // Create static library using lib.exe
    writeln!(batch, "lib.exe /OUT:\"{}\" \"{}\"",
        lib_file.to_string_lossy(),
        obj_file.to_string_lossy()
    ).unwrap();
    
    drop(batch);
    
    // Execute batch file
    let output = Command::new("cmd")
        .args(&["/C", batch_file.to_string_lossy().as_ref()])
        .output();
    
    match output {
        Ok(output) => {
            if output.status.success() {
                println!("cargo:warning=rgdb@0.1.0: CUDA wrapper compiled successfully");
                println!("cargo:rustc-link-search=native={}", out_dir);
                println!("cargo:rustc-link-lib=static=cuda_wrapper");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("cargo:warning=rgdb@0.1.0: CUDA wrapper compilation failed");
                if env::var("RUSTC_LOG").is_ok() || env::var("CARGO_VERBOSE").is_ok() {
                    if !stderr.is_empty() {
                        eprintln!("CUDA wrapper compilation stderr: {}", stderr);
                    }
                    if !stdout.is_empty() {
                        eprintln!("CUDA wrapper compilation stdout: {}", stdout);
                    }
                }
            }
        }
        Err(e) => {
            println!("cargo:warning=rgdb@0.1.0: Failed to execute CUDA wrapper compilation: {}", e);
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn compile_cuda_wrapper_unix() {
    // TODO: Implement Unix compilation
    println!("cargo:warning=rgdb@0.1.0: CUDA wrapper compilation for Unix not yet implemented");
}

fn link_cuda_libraries() {
    // Try to find CUDA library path
    #[cfg(target_os = "windows")]
    {
        let cuda_paths = vec![
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64",
        ];
        
        for path in cuda_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
                break;
            }
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    
    // Link CUDA libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
}
