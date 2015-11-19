if hash clang-format 2>/dev/null; then
  clang-format -i sobel.cpp sobel.cu sobel-cpu.cpp sobel-cpu-linear.cpp
fi
# g++ -std=c++1y -pedantic sobel-cpu.cpp -o cpu-image-filter
g++ -std=c++1y -pedantic sobel-cpu-linear.cpp -o image-filter
chmod +x cpu-image-filter
if hash nvcc 2>/dev/null; then
  nvcc --std=c++11 sobel.cpp sobel.cu -o gpu-image-filter
  chmod +x gpu-image-filter
fi
