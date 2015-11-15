g++ sobel-cpu.cpp -o cpu-image-filter
chmod +x image-filter
if hash nvcc 2>/dev/null; then
  nvcc sobel.cpp sobel.cu -o gpu-image-filter
fi
