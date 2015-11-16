// readWrite-bmp.cc
//
// extracts pixel data from user-specified .bmp file
// inserts data back into new .bmp file
//
// gw

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
//#include <omp.h>
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

int main(int argc, char *argv[]) {
  header_type header;
  information_type information;
  string imageFileName, newImageFileName;
  unsigned char tempData[3];
  int row, col, row_bytes, padding;
  // prepare files
  cout << "Original imagefile? ";
  cin >> imageFileName;
  ifstream imageFile;
  imageFile.open(imageFileName.c_str(), ios::binary);
  if (!imageFile) {
    cerr << "file not found" << endl;
    exit(-1);
  }
  cout << "New imagefile name? ";
  cin >> newImageFileName;
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
  int rows = information.height + 2;
  int column_size = information.width + 2;
  int src_size = rows * column_size;
  int **data = (int **)malloc(src_size * sizeof(int *));
  // extract image data, initialize vectors
  for (row = 1; row <= information.height; row++) {
    data[row] = (int *)malloc(column_size * sizeof(int));
    for (col = 0; col <= information.width; col++) {
      if (col == 0) {
        data[row][0] = 0;
      } else {
        imageFile.read((char *)tempData, 3 * sizeof(unsigned char));
        data[row][col] = ((int)tempData[0]);
      }
    }
    std::cout << "done processing row " << row << std::endl;
    // pad last column
    data[row][col + 1] = 0;
    if (padding)
      imageFile.read((char *)tempData, padding * sizeof(unsigned char));
  }

  std::cout << "done processing main image" << std::endl;

  // pad first & last row
  int last_row = rows - 1;
  data[0] = (int *)malloc(column_size * sizeof(int));
  data[last_row] = (int *)malloc(column_size * sizeof(int));
  for (col = 0; col < column_size; col++) {
    data[0][col] = 0;
    data[rows - 1][col] = 0;
  }
  std::cout << imageFileName << ": " << information.width << " x "
            << information.height << std::endl;

  int dest_size = information.width * information.height;
  int **newData = (int **)malloc(dest_size * sizeof(int *));
  int x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8, sum_0, sum_1;
  for (row = 1; row < (information.height + 1); row++) {
    newData[row - 1] = (int *)malloc(information.width * sizeof(int));
    for (col = 1; col < (information.width + 1); col++) {
      // std::cout << "row " << row << " col " << col << std::endl;
      bool top = (row == 0);
      bool bottom = (row == (rows - 1));
      bool left_edge = (col == 0);
      bool right_edge = (col == (column_size - 1));
      if (top == false && bottom == false && left_edge == false &&
          right_edge == false) {
        // newData[row].push_back(data[row][col]);
        x_0 = data[row - 1][col - 1];
        x_1 = data[row - 1][col];
        x_2 = data[row - 1][col + 1];
        x_3 = data[row][col - 1];
        x_5 = data[row][col + 1];
        x_6 = data[row + 1][col - 1];
        x_7 = data[row + 1][col];
        x_8 = data[row + 1][col + 1];
        sum_0 = (x_0 + (2 * x_1) + x_2) - (x_6 + (2 * x_7) + x_8);
        sum_1 = (x_2 + (2 * x_5) + x_8) - (x_0 + (2 * x_3) + x_6);

        // write new data onto smaller matrix
        newData[row - 1][col - 1] = sum_0 + sum_1;
      }
    }
  }

  std::cout << "finished iterating old data" << std::endl;

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
  cout << newImageFileName << " done." << endl;
  imageFile.close();
  newImageFile.close();

  return 0;
}
