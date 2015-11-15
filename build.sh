g++ sobel-cpu.cpp -o image-filter
chmod +x image-filter
if hash nvcc 2>/dev/null; then
  nvcc sobel.cpp sobel.cu
fi
