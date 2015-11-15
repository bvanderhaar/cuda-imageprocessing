#include <stdio.h>
#include <stdlib.h>

/*
 * In CUDA it is necessary to define block sizes
 * The grid of data that will be worked on is divided into blocks
 */
#define BLOCK_SIZE 512

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void cu_dotProduct(long long *distance_array_d,
                              long long *force_array_d,
                              long long *result_array_d, long long max) {
  long long x;
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (x < max) {
    result_array_d[x] = distance_array_d[x] * force_array_d[x];
  }
}

__global__ void cu_gen_force_array(long long *force_array_d, long long max) {
  long long x, half_vectors;
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  half_vectors = max / 2;
  if (x < half_vectors) {
    force_array_d[x] = x + 1;
  } else {
    force_array_d[x] = half_vectors + (half_vectors - x);
  }
}

__global__ void cu_gen_distance_array(long long *distance_array_d,
                                      long long max) {
  long long x;
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  distance_array_d[x] = (x + 1) % 10;
  if (distance_array_d[x] == 0) {
    distance_array_d[x] = 10;
  }
}

// Called from driver program.  Handles running GPU calculation
extern "C" void gpu_dotProduct(long long *result_array, long long num_vectors) {
  long long *distance_array_d;
  long long *force_array_d;
  long long *result_array_d;

  // allocate space in the device
  cudaMalloc((void **)&distance_array_d, sizeof(long long) * num_vectors);
  cudaMalloc((void **)&force_array_d, sizeof(long long) * num_vectors);
  cudaMalloc((void **)&result_array_d, sizeof(long long) * num_vectors);

  // set execution configuration
  dim3 dimblock(BLOCK_SIZE);
  dim3 dimgrid(ceil((long double)num_vectors / BLOCK_SIZE));

  cu_gen_force_array<<<dimgrid, dimblock>>>(force_array_d, num_vectors);
  cu_gen_distance_array<<<dimgrid, dimblock>>>(distance_array_d, num_vectors);
  cu_dotProduct<<<dimgrid, dimblock>>>(distance_array_d, force_array_d,
                                       result_array_d, num_vectors);
  // transfer results back to host
  cudaMemcpy(result_array, result_array_d, sizeof(long long) * num_vectors,
             cudaMemcpyDeviceToHost);

  // release the memory on the GPU
  cudaFree(distance_array_d);
  cudaFree(force_array_d);
  cudaFree(result_array_d);
}
