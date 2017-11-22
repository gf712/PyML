//
// Created by Gil Ferreira Hoben on 09/11/17.
//

#ifndef METRICS_DISTANCES_H
#define METRICS_DISTANCES_H


double vectorVectorNorm(double* A, double* B, int p, int cols);
void matrixMatrixNorm(double** A, double** B, int p, int rows, int cols, double* result);
void matrixVectorNorm(double** A, double* B, int p, int rows, int cols, double* result);


#endif //METRICS_DISTANCES_H
