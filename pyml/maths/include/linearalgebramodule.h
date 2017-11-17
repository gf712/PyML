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

void leastSquares(flatArray *X, flatArray *y, double *theta);
flatArray *covariance(flatArray *X);
flatArray *jacobiEigenDecomposition(flatArray *S, double tolerance, int maxIterations);

#ifdef __cplusplus
}
#endif

#endif //SRC_LINEARALGEBRAMODULE_H
