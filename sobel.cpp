/*
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc simple.c simple.cu
 */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

extern "C" void gpu_dotProduct(long long *result_array, long long num_vectors);

void printarray(long long *array, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
}

int main(int argc, char *argv[]) {
  long long num_vectors = 0;
  long long pu_dotproduct_result = 0;
  long long i = 0;
  if (argc < 2) {
    std::cout << "Usage: cuda-musclework num_vectors" << std::endl;
  }
  num_vectors = atoll(argv[1]);
  std::cout << "Using num_vectors: " << num_vectors << std::endl;
  long long *result = (long long *)malloc(num_vectors * sizeof(long long));

  // Call the function that will call the GPU function
  clock_t gpu_start = clock();
  gpu_dotProduct(result, num_vectors);
  for (i = 0; i < num_vectors; i++) {
    pu_dotproduct_result = pu_dotproduct_result + result[i];
  }
  clock_t gpu_stop = clock();
  double elapsed_gpu = double(gpu_stop - gpu_start) / (CLOCKS_PER_SEC / 1000);

  std::cout << "GPU Calc Total: " << pu_dotproduct_result << std::endl;
  std::cout << "GPU Time Taken (msec): " << elapsed_gpu << std::endl;

  return 0;
}
