//
// Created by Gil Ferreira Hoben on 06/11/17.
//
#include <Python.h>
#include "flatArrays.h"

#ifndef MATHS_LINEARALGEBRAMODULE_H
#define MATHS_LINEARALGEBRAMODULE_H

template <typename T>
void leastSquares(flatArray<T>& X, flatArray<T>& y, T *theta);

template <typename T>
flatArray<T>* covariance(flatArray<T> *X);

template <typename T>
flatArray<T>* jacobiEigenDecomposition(flatArray<T> *S, double tolerance, int maxIterations);

template <typename T>
double determinant(flatArray<T>* array);

#endif //SRC_LINEARALGEBRAMODULE_H
