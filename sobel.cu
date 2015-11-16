#include <stdio.h>
#include <stdlib.h>

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

__global__ void cu_sobel(int *l_source_array_d, int *l_result_array_d, int rows,
                         int column_size) {
  int x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8, sum_0, sum_1;
  // x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  // int index_source = col * grid_width + row;

  // edge of matrix has zeros.  don't process
  bool top = (row == 0);
  bool bottom = (row == (rows - 1));
  bool left_edge = (col == 0);
  bool right_edge = (col == (column_size - 1));
  if (top == false && bottom == false && left_edge == false &&
      right_edge == false) {
    x_0 = l_source_array_d[(col - 1) * grid_width + (row - 1)];
    x_1 = l_source_array_d[(col)*grid_width + (row - 1)];
    x_2 = l_source_array_d[(col + 1) * grid_width + (row - 1)];
    x_3 = l_source_array_d[(col - 1) * grid_width + (row)];
    x_5 = l_source_array_d[(col + 1) * grid_width + (row)];
    x_6 = l_source_array_d[(col - 1) * grid_width + (row + 1)];
    x_7 = l_source_array_d[(col)*grid_width + (row + 1)];
    x_8 = l_source_array_d[(col + 1) * grid_width + (row + 1)];
    sum_0 = (x_0 + (2 * x_1) + x_2) - (x_6 + (2 * x_7) + x_8);
    sum_1 = (x_2 + (2 * x_5) + x_8) - (x_0 + (2 * x_3) + x_6);
    // write new data onto smaller matrix
    l_result_array_d[(col - 1) * (grid_width - 2) + (row - 1)] = sum_0 + sum_1;
  }
}

// Called from driver program.  Handles running GPU calculation
extern "C" void gpu_sobel(int **source_array, int **result_array, int src_rows,
                          int src_column_size) {
  int row, col;
  int num_elements_x = src_column_size;
  int num_elements_y = src_rows;
  int num_bytes_source = src_column_size * src_rows * sizeof(int);

  // linear-ize source array
  int *l_source_array = 0;
  l_source_array = (int *)malloc(num_bytes_source);
  for (row = 0; row < src_rows; row++) {
    for (col = 0; col < src_column_size; col++) {
      l_source_array[row * src_column_size + col] = source_array[row][col];
    }
  }
  int *source_array_d = 0;
  cudaMalloc((void **)&source_array_d, num_bytes_source);
  cudaMemcpy(l_source_array, source_array_d, num_bytes_source,
             cudaMemcpyHostToDevice);

  int result_column_size = src_column_size - 2;
  int result_row_size = src_rows - 2;
  int num_bytes_result = result_column_size * result_column_size * sizeof(int);
  int *l_result_array = 0;
  int *l_result_array_d = 0;
  l_result_array = (int *)malloc(num_bytes_result);
  cudaMalloc((void **)&l_result_array_d, num_bytes_result);
  cudaMemcpy(l_result_array, l_result_array_d, num_bytes_result,
             cudaMemcpyHostToDevice);

  // create two dimensional 4x4 thread blocks
  dim3 block_size;
  block_size.x = 4;
  block_size.y = 4;

  // configure a two dimensional grid as well
  dim3 grid_size;
  grid_size.x = num_elements_x / block_size.x;
  grid_size.y = num_elements_y / block_size.y;

  // grid_size & block_size are passed as arguments to the triple chevrons as
  // usual
  cu_sobel<<<grid_size, block_size>>>(l_source_array_d, l_result_array_d,
                                      src_rows, src_column_size);

  // transfer results back to host
  cudaMemcpy(l_result_array, l_result_array_d, num_bytes_result,
             cudaMemcpyDeviceToHost);

  // de-linearize result array
  for (row = 0; row < result_row_size; row++) {
    result_array[row] = (int *)malloc(result_column_size * sizeof(int));
    for (col = 0; col < result_column_size; col++) {
      result_array[row][col] = l_result_array[row * result_column_size + col];
    }
  }

  // release the memory on the GPU
  cudaFree(source_array_d);
  cudaFree(result_array_d);
}
