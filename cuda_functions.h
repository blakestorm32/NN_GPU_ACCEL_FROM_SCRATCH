#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <cuda_runtime.h>

// Declare your functions here
float computeMSE(float* d_predictions, float* d_targets, int size);
void matrixMultiplyCuda(float* A, float* B, float* C, int M, int N, int K);
void launchSigmoidActivation(float* d_input, float* d_output, int size);
void backpropagation(float* d_input, float* d_hidden_output, float* d_output, float* d_targets,
                     float* d_hidden_weights, float* d_output_weights,
                     float* d_hidden_biases, float* d_output_biases,
                     int input_size, int hidden_size, int output_size, float learning_rate);

#endif // CUDA_FUNCTIONS_H