//
// Created by gil on 09/11/17.
//
#include <cmath>
#include "distances.h"

double vectorVectorNorm(double* A, double* B, int p, int cols) {

    double normResult = 0;

    for (int i = 0; i < cols; ++i) {
        normResult += pow(fabs(A[i] - B[i]), p);
    }

    normResult = pow(normResult, (double) 1 / p);

    return normResult;
}

void matrixMatrixNorm(double** A, double** B, int p, int rows, int cols, double* result) {

    for (int i = 0; i < rows; ++i) {
        result[i] = vectorVectorNorm(A[i], B[i], p, cols);
    }
}


void matrixVectorNorm(double** A, double* B, int p, int rows, int cols, double* result) {

    for (int i = 0; i < rows; ++i) {
        result[i] = vectorVectorNorm(A[i], B, p, cols);
    }
}
