#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <string.h>  
#include <math.h>    
#include <cstdint>  
 
typedef struct {
    float real;
    float imag;
} Complex;

__global__ void initialize_state(Complex* state, int num_qubits) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0) {
        state[0] = {1.0, 0.0};
    } else if(idx < (1 << num_qubits)) {
        state[idx] = {0.0, 0.0};
    }
}

__global__ void apply_h(Complex* state, int qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return;
    
    int mask = 1 << qubit;
    int i0 = idx & ~mask;
    int i1 = idx | mask;
    
    if((idx & mask) == 0) {
        Complex v0 = state[i0];
        Complex v1 = state[i1];
        state[i0] = { (v0.real + v1.real)/sqrtf(2.0), (v0.imag + v1.imag)/sqrtf(2.0) };
        state[i1] = { (v0.real - v1.real)/sqrtf(2.0), (v0.imag - v1.imag)/sqrtf(2.0) };
    }
}

__global__ void apply_cx(Complex* state, int control, int target, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return;
    
    int control_mask = 1 << control;
    int target_mask = 1 << target;
    
    if((idx & control_mask) && !(idx & target_mask)) {
        int paired = idx ^ target_mask;
        Complex temp = state[idx];
        state[idx] = state[paired];
        state[paired] = temp;
    }
}
__global__ void apply_x(Complex* state, int num_qubits, int target_qubit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long size = 1ULL << num_qubits;

    if (idx >= size / 2) return;

    // Compute the two indices that differ in target_qubit bit
    unsigned long long bit_mask = 1ULL << target_qubit;

    // Index of the pair to swap with
    unsigned long long idx1 = ((idx >> target_qubit) << (target_qubit + 1)) | (idx & (bit_mask - 1));
    unsigned long long idx2 = idx1 | bit_mask;

    // Swap amplitudes at idx1 and idx2
    Complex temp = state[idx1];
    state[idx1] = state[idx2];
    state[idx2] = temp;
}

// Kernel to calculate probabilities
__global__ void calculate_probabilities(Complex* state, float* probs, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        probs[idx] = state[idx].real * state[idx].real + state[idx].imag * state[idx].imag;
    }
}

// Kernel for measurement sampling
__global__ void measure_kernel(float* cum_probs, int size, int* results, int num_shots, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_shots) return;

    curandState_t state;
    curand_init(seed, idx, 0, &state);
    float r = curand_uniform(&state);

    // Binary search for the outcome
    int low = 0, high = size;
    while(low < high) {
        int mid = low + (high - low) / 2;
        if(cum_probs[mid] < r) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    results[idx] = low < size ? low : size - 1;
}
__global__ void apply_y(Complex* state, int target_qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    int mask = 1 << target_qubit;
    int i0 = idx & ~mask;
    int i1 = idx | mask;

    if ((idx & mask) == 0) {
        Complex v0 = state[i0];
        Complex v1 = state[i1];

        state[i0].real = v1.imag;
        state[i0].imag = -v1.real;
        state[i1].real = -v0.imag;
        state[i1].imag = v0.real;
    }
}

__global__ void apply_z(Complex* state, int target_qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if (idx & (1 << target_qubit)) {
        state[idx].real *= -1;
        state[idx].imag *= -1;
    }
}

__global__ void apply_s(Complex* state, int target_qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if (idx & (1 << target_qubit)) {
        float real = state[idx].real;
        float imag = state[idx].imag;
        state[idx].real = -imag;
        state[idx].imag = real;
    }
}

__global__ void apply_sdg(Complex* state, int target_qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if (idx & (1 << target_qubit)) {
        float real = state[idx].real;
        float imag = state[idx].imag;
        state[idx].real = imag;
        state[idx].imag = -real;
    }
}

__global__ void apply_t(Complex* state, int target_qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if (idx & (1 << target_qubit)) {
        float cos_theta = M_SQRT1_2; // cos(π/4)
        float sin_theta = M_SQRT1_2; // sin(π/4)
        float real = state[idx].real * cos_theta - state[idx].imag * sin_theta;
        float imag = state[idx].real * sin_theta + state[idx].imag * cos_theta;
        state[idx].real = real;
        state[idx].imag = imag;
    }
}

__global__ void apply_tdg(Complex* state, int target_qubit, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if (idx & (1 << target_qubit)) {
        float cos_theta = M_SQRT1_2;
        float sin_theta = -M_SQRT1_2;
        float real = state[idx].real * cos_theta - state[idx].imag * sin_theta;
        float imag = state[idx].real * sin_theta + state[idx].imag * cos_theta;
        state[idx].real = real;
        state[idx].imag = imag;
    }
}

__global__ void apply_cz(Complex* state, int control, int target, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    if ((idx & (1 << control)) && (idx & (1 << target))) {
        state[idx].real *= -1;
        state[idx].imag *= -1;
    }
}

__global__ void apply_swap(Complex* state, int qubit_a, int qubit_b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    int a_bit = (idx >> qubit_a) & 1;
    int b_bit = (idx >> qubit_b) & 1;
    int swapped_idx = idx;
    swapped_idx &= ~((1 << qubit_a) | (1 << qubit_b));
    swapped_idx |= (a_bit << qubit_b) | (b_bit << qubit_a);

    if (swapped_idx > idx) {
        Complex temp = state[idx];
        state[idx] = state[swapped_idx];
        state[swapped_idx] = temp;
    }
}

extern "C" {
    void* create_state(int num_qubits) {
        Complex* d_state;
        cudaMalloc(&d_state, (1 << num_qubits) * sizeof(Complex));
        int threads = 256;
        int blocks = ((1 << num_qubits) + threads - 1) / threads;
        initialize_state<<<blocks, threads>>>(d_state, num_qubits);
        return d_state;
    }
    
    void apply_gate(void* state, const char* gate, int qubit, int target, int num_qubits) {
        int size = 1 << num_qubits;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        if (strcmp(gate, "h") == 0) {
            apply_h<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "cx") == 0) {
            apply_cx<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, target, size);
        } else if (strcmp(gate, "x") == 0) {
            apply_x<<<blocks, threads>>>(static_cast<Complex*>(state), num_qubits, qubit); 
        } else if (strcmp(gate, "y") == 0) {
            apply_y<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "z") == 0) {
            apply_z<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "s") == 0) {
            apply_s<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "sdg") == 0) {
            apply_sdg<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "t") == 0) {
            apply_t<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "tdg") == 0) {
            apply_tdg<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, size);
        } else if (strcmp(gate, "cz") == 0) {
            apply_cz<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, target, size);
        } else if (strcmp(gate, "swap") == 0) {
            apply_swap<<<blocks, threads>>>(static_cast<Complex*>(state), qubit, target, size);
        }
        cudaDeviceSynchronize();
    }
    
    void measure(void* state, int* results, int num_shots, int num_qubits) {
        int size = 1 << num_qubits;
        Complex* d_state = static_cast<Complex*>(state);
        
        // Allocate device memory for probabilities
        float *d_probs, *d_cum_probs;
        cudaMalloc(&d_probs, size * sizeof(float));
        cudaMalloc(&d_cum_probs, size * sizeof(float));
    
        // Calculate probabilities
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        calculate_probabilities<<<grid, block>>>(d_state, d_probs, size);
    
        // Calculate cumulative probabilities using Thrust
        thrust::inclusive_scan(thrust::device, d_probs, d_probs + size, d_cum_probs);
    
        // Normalize cumulative probabilities
        float total;
        cudaMemcpy(&total, d_cum_probs + size - 1, sizeof(float), cudaMemcpyDeviceToHost);
        thrust::transform(thrust::device, d_cum_probs, d_cum_probs + size, d_cum_probs,
            [=] __device__ (float x) { return x / total; });
    
        // Allocate device memory for results
        int* d_results;
        cudaMalloc(&d_results, num_shots * sizeof(int));
    
        // Generate random samples
        measure_kernel<<<(num_shots + 255)/256, 256>>>(d_cum_probs, size, d_results, num_shots, time(0));
    
        // Copy results back to host
         cudaMemcpy(results, d_results, num_shots * sizeof(int), cudaMemcpyDeviceToHost);
    
        // Cleanup
        cudaFree(d_probs);
        cudaFree(d_cum_probs);
        cudaFree(d_results);
    }
}