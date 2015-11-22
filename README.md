## Abstract
An bitmap image processor implementing a psuedo-sobel algorithm.  Both CUDA and CPU versions are compared for performance.

## High Level Design
A bitmap image is put in a 2D array.  The array is then linearized.  In the CPU version, each row is processed in order.  In the GPU version, block size is set to column size; grid size to number of rows.


## Implementation



## Testing Methodology



## Discussion



## Conclusion
