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

__global__ void cu_sobel(int *source_array_d, int *result_array_d,
                         int source_row_size, int source_size) {
  int x, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8, sum_0, sum_1;
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  // edge of matrix has zeros.  don't process
  bool top = x < source_row_size;
  bool bottom = x > (source_size - source_row_size);
  bool left_edge = (x % source_row_size) == 0;
  bool right_edge = (x % (source_row_size + 1)) == 0;
  if (top == false && bottom == false && left_edge == false &&
      right_edge == false) {
    x_0 = source_array_d[x - row_size - 1];
    x_1 = source_array_d[x - row_size];
    x_2 = source_array_d[x - row_size + 1];
    x_3 = source_array_d[x - 1];
    x_5 = source_array_d[x + 1];
    x_6 = source_array_d[x + row_size - 1];
    x_7 = source_array_d[x + row_size];
    x_8 = source_array_d[x + row_size + 1];
    sum_0 = (x_0 + (2 * x_1) + x_2) - (x_6 + (2 * x_7) + x_8);
    sum_1 = (x_2 + (2 * x_5) + x_8) - (x_0 + (2 * x_3) + x_6);
    result_array_d[x] = (sum_0 + sum_1);
  }
}

// Called from driver program.  Handles running GPU calculation
extern "C" void gpu_sobel(int *source_array, int *result_array,
                          int dest_row_size, int dest_column_size) {
  // while confusing, source is 2 cols + 2 rows larger than dest for 0 padding
  int dest_size = dest_row_size * dest_column_size;
  int source_size = (dest_row_size + 2) * (dest_column_size + 2);
  int source_row_size = dest_row_size + 2;

  // allocate space in the device
  cudaMalloc((void **)&source_array_d, sizeof(int) * source_size);
  cudaMalloc((void **)&result_array_d, sizeof(int) * dest_size);

  cudaMemcpy(source_array, source_array_d, sizeof(int) * source_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(result_array, result_array_d, sizeof(int) * dest_size,
             cudaMemcpyHostToDevice);

  // set execution configuration
  dim3 dimblock(BLOCK_SIZE);
  dim3 dimgrid(ceil((double)dest_size / BLOCK_SIZE));

  cu_sobel<<<dimgrid, dimblock>>>(source_array_d, result_array_d, source_row_size, source_size);
  // transfer results back to host
  cudaMemcpy(result_array, result_array_d, sizeof(int) * dest_size,
             cudaMemcpyDeviceToHost);

  // release the memory on the GPU
  cudaFree(source_array_d);
  cudaFree(result_array_d);
}
