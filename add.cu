#include <iostream>
#include <math.h>
#include <stdio.h>
// Kernel function to add the elements of two arrays

#define RAND_NUM 10.0

__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

__global__ void matrix_multiply(float *a, float *b, float *c, int width){
  int row = threadIdx.x;
  int col = threadIdx.y;
  float sum = 0.0f;
  for(int i = 0; i < width; i++){
    sum += a[row * width + i] + b[i * width + col];
  }
  c[row * width + col] = sum;
}

void generateRandomArray(int size, float *array) {
    int i;
    for (i = 0; i < size; i++) {
        array[i] = (float)rand() / RAND_NUM * 100.0f; // Generate random floats between 0 and 100
    }
}

void printFloatPointer(float* array, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
        std::cout << "a" << " ";
    }
    std::cout << std::endl;
}

void matrix_mult(int width, int height){
  int wid = width;
  int hei = height;
  width *= sizeof(float);
  height *= sizeof(float);

  //CPU variables
  float *c_a, *c_b, *c_c;
  //GPU variables
  float *g_a, *g_b, *g_c;

  printf("Hello");

  c_a = (float*)malloc(width);
  c_b = (float*)malloc(height * width);
  c_c = (float*)malloc(height);
  generateRandomArray(wid, c_a);
  generateRandomArray(wid * hei, c_b);
  printFloatPointer(c_a, wid);
  printFloatPointer(c_b, wid * hei);

  printf("Hello_2");

  //allocate mem on gpu
  cudaMalloc(&g_a, width);
  cudaMalloc(&g_b, width * height);
  cudaMalloc(&g_c, height);

  cudaMemcpy(g_a, c_a, width, cudaMemcpyHostToDevice);
  cudaMemcpy(g_b, c_b, width * height, cudaMemcpyHostToDevice);

  matrix_multiply<<<1, height>>>(g_a, g_b, g_c, wid);

  cudaMemcpy(g_c, c_c, height, cudaMemcpyDeviceToHost);

  cudaFree(g_a);
  cudaFree(g_b);
  cudaFree(g_c);

  printFloatPointer(c_c, hei);

  free(c_a);
  free(c_b);
  free(c_c);
}

int main(void)
{
  matrix_mult(10, 10); 
  return 0;
}