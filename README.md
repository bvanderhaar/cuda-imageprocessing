## Abstract
An bitmap image processor implementing a psuedo-sobel algorithm.  Both CUDA and CPU versions are compared for performance.

## High Level Design
A bitmap image is put in a 2D array.  The array is then linearized.  In the CPU version, each row is processed in order.  In the GPU version, block size is set to column size; grid size to number of rows.

## Implementation
For the serial program, Clang/LLVM 3.6 is used with level 2 compiler optimizations enabled. The parallel GPU program is compiled with Nvidia's CUDA compiler driver V7.5 that wraps gcc; c++11 also enabled with the Nvidia CUDA compiler so the program could use C++ features in code compiled for CUDA.

## Testing Methodology
A Macbook Pro with a Core i5 at 2.7GHz and 8GB RAM is used for the serial program. GVSU's Seawolf with a Tesla K40c is used for the GPU version of the program.
The C++ time API is used to record method execution of the Sobel processing only.  Since both programs execute over a linear array - any linear-izing or de-linearizing is also not included in the time calculation.

## Discussion



## Conclusion
