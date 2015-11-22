#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
using namespace std;

#pragma pack(1)
typedef struct {
  char id[2];
  int file_size;
  int reserved;
  int offset;
} header_type;

#pragma pack(1)
typedef struct {
  int header_size;
  int width;
  int height;
  unsigned short int color_planes;
  unsigned short int color_depth;
  unsigned int compression;
  int image_size;
  int xresolution;
  int yresolution;
  int num_colors;
  int num_important_colors;
} information_type;

void sobel(int *l_source_array, int *l_result_array, int rows, int column_size,
           int row, int col) {
  int x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8, sum_0, sum_1, sum;
  bool top = (row == 0);
  bool bottom = (row == (rows - 1));
  bool left_edge = (col == 0);
  bool right_edge = (col == (column_size - 1));
  if (top == false && bottom == false && left_edge == false &&
      right_edge == false) {
    printf("row: %i col: %i \n", row, col);
    x_0 = l_source_array[(row - 1) * column_size + (col - 1)];
    x_1 = l_source_array[(row - 1) * column_size + (col)];
    x_2 = l_source_array[(row - 1) * column_size + (col + 1)];
    x_3 = l_source_array[(row)*column_size + (col - 1)];
    x_5 = l_source_array[(row)*column_size + (col + 1)];
    x_6 = l_source_array[(row + 1) * column_size + (col - 1)];
    x_7 = l_source_array[(row + 1) * column_size + (col)];
    x_8 = l_source_array[(row + 1) * column_size + (col + 1)];
    sum_0 = (x_0 + (2 * x_1) + x_2) - (x_6 + (2 * x_7) + x_8);
    sum_1 = (x_2 + (2 * x_5) + x_8) - (x_0 + (2 * x_3) + x_6);
    sum = sum_0 + sum_1;
    if (sum > 20) {
      sum = 255
    } else {
      sum = 0;
    }
    // write new data onto smaller matrix
    l_result_array[((row - 1) * (column_size - 2)) + (col - 1)] = sum;
  }
}

int main(int argc, char *argv[]) {
  header_type header;
  information_type information;
  string imageFileName, newImageFileName;
  unsigned char tempData[3];
  int row, col, row_bytes, padding;
  // prepare files
  imageFileName = argv[1];
  ifstream imageFile;
  imageFile.open(imageFileName.c_str(), ios::binary);
  if (!imageFile) {
    std::cerr << "file not found" << std::endl;
    exit(-1);
  }
  // std::cout << "New imagefile name? ";
  newImageFileName = argv[2];
  ofstream newImageFile;
  newImageFile.open(newImageFileName.c_str(), ios::binary);

  // read file header
  imageFile.read((char *)&header, sizeof(header_type));
  if (header.id[0] != 'B' || header.id[1] != 'M') {
    cerr << "Does not appear to be a .bmp file.  Goodbye." << endl;
    exit(-1);
  }
  // read/compute image information
  imageFile.read((char *)&information, sizeof(information_type));
  row_bytes = information.width * 3;
  padding = row_bytes % 4;
  if (padding)
    padding = 4 - padding;

  // extract image data, initialize vectors
  int src_rows = information.height + 2;
  int src_column_size = information.width + 2;
  int src_size = src_rows * src_column_size;
  int **data = (int **)malloc(src_size * sizeof(int *));
  for (row = 1; row <= information.height; row++) {
    data[row] = (int *)malloc(src_column_size * sizeof(int));
    for (col = 0; col <= information.width; col++) {
      if (col == 0) {
        data[row][0] = 0;
      } else {
        imageFile.read((char *)tempData, 3 * sizeof(unsigned char));
        data[row][col] = ((int)tempData[0]);
      }
    }
    // pad last column
    data[row][col + 1] = 0;
    if (padding)
      imageFile.read((char *)tempData, padding * sizeof(unsigned char));
  }

  // pad first & last row
  int last_row = src_rows - 1;
  data[0] = (int *)malloc(src_column_size * sizeof(int));
  data[last_row] = (int *)malloc(src_column_size * sizeof(int));
  for (col = 0; col < src_column_size; col++) {
    data[0][col] = 0;
    data[src_rows - 1][col] = 0;
  }
  std::cout << imageFileName << ": " << information.width << " x "
            << information.height << std::endl;

  // linear-ize source array
  int dest_size = information.width * information.height;
  int *l_source_array = 0;
  l_source_array = new int[src_size];
  for (row = 0; row < src_rows; row++) {
    for (col = 0; col < src_column_size; col++) {
      l_source_array[row * src_column_size + col] = data[row][col];
    }
  }

  int result_rows = information.height;
  int result_column_size = information.width;
  int *l_result_array = 0;
  l_result_array = new int[dest_size];
  for (row = 1; row < (information.height + 1); row++) {
    for (col = 1; col < (information.width + 1); col++) {
      sobel(l_source_array, l_result_array, src_rows, src_column_size, row,
            col);
    }
  }

  // de-linearize result array
  int **newData = (int **)malloc(dest_size * sizeof(int *));
  for (row = 0; row < result_rows; row++) {
    newData[row] = new int[result_column_size];
    for (col = 0; col < result_column_size; col++) {
      newData[row][col] = l_result_array[(row * result_column_size) + col];
    }
  }

  // write header to new image file
  newImageFile.write((char *)&header, sizeof(header_type));
  newImageFile.write((char *)&information, sizeof(information_type));

  // write new image data to new image file
  for (row = 0; row < information.height; row++) {
    for (col = 0; col < information.width; col++) {
      tempData[0] = (unsigned char)newData[row][col];
      tempData[1] = (unsigned char)newData[row][col];
      tempData[2] = (unsigned char)newData[row][col];
      newImageFile.write((char *)tempData, 3 * sizeof(unsigned char));
    }
    if (padding) {
      tempData[0] = 0;
      tempData[1] = 0;
      tempData[2] = 0;
      newImageFile.write((char *)tempData, padding * sizeof(unsigned char));
    }
  }
  std::cout << newImageFileName << " done." << std::endl;
  imageFile.close();
  newImageFile.close();

  return 0;
}
