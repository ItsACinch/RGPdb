// CUDA kernel for light propagation in RGDB
// Compile with: nvcc -ptx propagate_kernel.cu -o propagate_kernel.ptx

#include <cuda_runtime.h>
#include <math.h>

// Angular distance computation (matches Rust implementation)
__device__ unsigned char angular_distance(unsigned char b1, unsigned char b2, unsigned int num_bins) {
    int diff = abs((int)b1 - (int)b2);
    int wrapped = (int)num_bins - diff;
    return (unsigned char)(diff < wrapped ? diff : wrapped);
}

// Refraction factor computation
__device__ float refraction_factor(
    unsigned char bin_in,
    unsigned char bin_out,
    float n,
    float k,
    unsigned int num_bins
) {
    float delta = (float)angular_distance(bin_in, bin_out, num_bins);
    float b = (float)num_bins;
    float x = (delta / b) * (delta / b);
    return expf(-k * n * x);
}

// Initialization kernel - sets source node intensity
extern "C" __global__ void init_propagation_kernel(
    float* intensities,           // [num_nodes * num_angle_bins]
    float* total_intensity,        // [num_nodes]
    const float* node_props,       // [num_nodes * 4]
    unsigned int num_nodes,
    unsigned int num_angle_bins,
    unsigned int source,
    unsigned int initial_bin
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    // Initialize source node
    if (node_idx == source) {
        float luminance = node_props[node_idx * 4];
        int intensity_idx = node_idx * num_angle_bins + initial_bin;
        float init_intensity = fmaxf(luminance, 1.0f);
        intensities[intensity_idx] = init_intensity;
        atomicAdd(&total_intensity[node_idx], init_intensity);
    }
}

// Propagation kernel - one iteration of light propagation
extern "C" __global__ void propagate_iteration_kernel(
    float* intensities,           // [num_nodes * num_angle_bins] - current intensities
    float* next_intensities,      // [num_nodes * num_angle_bins] - next iteration
    float* total_intensity,        // [num_nodes] - accumulated total
    const float* node_props,       // [num_nodes * 4]
    const unsigned int* row_ptr,  // [num_nodes + 1]
    const unsigned int* col_idx,  // [num_edges]
    const float* edge_props,       // [num_edges * 2]
    unsigned int num_nodes,
    unsigned int num_angle_bins,
    float k,
    float min_intensity
) {
    // Each thread processes one (node, angle_bin) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_nodes * num_angle_bins;
    
    if (idx >= total_pairs) return;
    
    int node_idx = idx / num_angle_bins;
    int angle_bin = idx % num_angle_bins;
    
    float current_intensity = intensities[idx];
    if (current_intensity < min_intensity) return;
    
    // Get node properties
    float reflection = node_props[node_idx * 4 + 1];
    float refraction_index = node_props[node_idx * 4 + 2];
    float reflected = current_intensity * reflection;
    
    // Process neighbors
    unsigned int edge_start = row_ptr[node_idx];
    unsigned int edge_end = row_ptr[node_idx + 1];
    
    for (unsigned int e = edge_start; e < edge_end; e++) {
        unsigned int neighbor = col_idx[e];
        float attenuation = edge_props[e * 2];
        unsigned char edge_angle_bin = (unsigned char)edge_props[e * 2 + 1];
        
        // Compute refraction
        float rho = refraction_factor(
            (unsigned char)angle_bin,
            edge_angle_bin,
            refraction_index,
            k,
            num_angle_bins
        );
        
        // Compute transmitted intensity
        float transmitted = reflected * (1.0f - attenuation) * rho;
        
        if (transmitted >= min_intensity) {
            int neighbor_intensity_idx = neighbor * num_angle_bins + edge_angle_bin;
            
            // Use atomic max to handle concurrent updates
            // atomicMax is the correct operation for finding maximum value atomically
            atomicMax(&next_intensities[neighbor_intensity_idx], transmitted);
            
            // Accumulate total intensity
            atomicAdd(&total_intensity[neighbor], transmitted);
        }
    }
}

// Reduction kernel - sum intensities across angle bins for each node
extern "C" __global__ void reduce_intensities_kernel(
    const float* intensities,     // [num_nodes * num_angle_bins]
    float* total_intensity,        // [num_nodes] - output
    unsigned int num_nodes,
    unsigned int num_angle_bins
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    float sum = 0.0f;
    for (int b = 0; b < num_angle_bins; b++) {
        sum += intensities[node_idx * num_angle_bins + b];
    }
    
    total_intensity[node_idx] = sum;
}

