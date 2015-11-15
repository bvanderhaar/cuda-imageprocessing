if hash clang-format 2>/dev/null; then
  clang-format -i sobel.cpp sobel.cu sobel-cpu.cpp
fi
g++ -std=c++1y -pedantic sobel-cpu.cpp -o cpu-image-filter
chmod +x cpu-image-filter
if hash nvcc 2>/dev/null; then
  nvcc -std=c++0x sobel.cpp sobel.cu -o gpu-image-filter
fi
