## Abstract
An bitmap image processor implementing a psuedo-sobel algorithm.  Both CUDA and CPU versions are compared for performance.

## High Level Design
A bitmap image is put in a 2D array.  The array is then linearized.  In the CPU version, each row is processed in order.  In the GPU version, block size is set to column size; grid size to number of rows.

## Implementation
For the serial program, Clang/LLVM 3.6 is used with level 2 compiler optimizations enabled on Macbook OS. The GNU C compiler is used on Seawolf.  The parallel GPU program is compiled with Nvidia's CUDA compiler driver V7.5 that wraps gcc; c++11 also enabled with the Nvidia CUDA compiler so the program could use C++ features in code compiled for CUDA.

## Testing Methodology
A Macbook Pro with a Core i5 at 2.7GHz and 8GB RAM and Seawolf's Xeon CPU E3-1231 v3 is used for the serial program. GVSU's Seawolf with a Tesla K40c is used for the GPU version of the program.
The C++ time API is used to record method execution of the Sobel processing only.  Since both programs execute over a linear array - any linear-izing or de-linearizing is also not included in the time calculation.  Both programs also read and write from the filesystem; this time is ignored in the performance comparison.

## Raw data
<img src="https://raw.githubusercontent.com/bvanderhaar/cuda-imageprocessing/master/raw-data.png">

## Discussion
<img src="https://raw.githubusercontent.com/bvanderhaar/cuda-imageprocessing/master/sobel-running-time.png">

As you can see from the chart, smaller sizes do not benefit from GPU acceleration.  Nor do the smaller sizes benefit from the Xeon's server-class CPU.  But, the largest size (2772 x 2703) benefits significantly from the GPU acceleration and the largest size benefits from Xeon's server-class CPU.

Processing on the macbook is surprisingly orders of magnitude faster than the server-class CPU compared as well as the GPU.  The GPU is somewhat expected due to the design of the GPU.  But the slowness on the server CPU is harder to explain.  Is the Macbook's Clang compiler and OS more optimized?  Is it due to the later generation Broadwell architecture versus the Haswell architecure of the Xeon?  More testing is needed to determine these answers.

## Conclusion
GPU acceleration is beneficial with large data sets and the nature of the problem to be solved.  GPU's design is influenced by video and image processing - this experiment proves the success of this design.
