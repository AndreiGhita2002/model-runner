#include <cuda.h>
#include <torch/extension.h>

// Reads the current 'clock64' value and writes it to the buffer.
__global__ void start_timer_kernel(long long int* buffer) {
    // We only have one thread, but this is the safe way to do it
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        buffer[0] = clock64();
    }
}

// Reads the start time from the buffer, gets the current 'clock64' time,
// computes the difference, and writes the elapsed time back to the buffer.
__global__ void end_timer_kernel(long long int* buffer) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long int start_time = buffer[0];
        long long int end_time = clock64();

        // Write to buffer
        buffer[0] = end_time - start_time;
    }
}


/*
C++ launcher functions.
*/

void start_timer_launcher(torch::Tensor buffer) {
    // Get the raw data pointer from the PyTorch tensor
    long long int* buffer_ptr = buffer.data_ptr<long long int>();

    // Launch the kernel with 1 block and 1 thread
    start_timer_kernel<<<1, 1>>>(buffer_ptr);
}

void end_timer_launcher(torch::Tensor buffer) {
    long long int* buffer_ptr = buffer.data_ptr<long long int>();

    // Launch the kernel with 1 block and 1 thread
    end_timer_kernel<<<1, 1>>>(buffer_ptr);
}