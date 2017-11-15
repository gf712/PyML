//
// Created by Gil Ferreira Hoben on 06/11/17.
//
#include <Python.h>
#include "flatArrays.h"

#ifndef MATHS_LINEARALGEBRAMODULE_H
#define MATHS_LINEARALGEBRAMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

void matrixVectorDotProduct(double** A, double* v, int ASize, int VSize, double* result);
double vectorSum(const double* array, int rows);
void matrixTranspose(double** X, double** result, int rows, int cols, int block_size);
void vectorSubtract(const double* u, const double* v, int size, double* result);
void vectorDivide(double* X, int n, int size);
double vectorMean(const double* array, int size);
void matrixMean(double **array, int cols, int rows, int axis, double* result);
void flatMatrixPower(flatArray *A, int p);
void flatArraySubtract(flatArray *A, flatArray *B, flatArray *result);
void leastSquares(flatArray *X, flatArray *y, double *theta);


#ifdef __cplusplus
}
#endif

#endif //SRC_LINEARALGEBRAMODULE_H
