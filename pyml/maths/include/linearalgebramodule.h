//
// Created by Gil Ferreira Hoben on 06/11/17.
//
#include <Python.h>

#ifndef SRC_LINEARALGEBRAMODULE_H
#define SRC_LINEARALGEBRAMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

void pypyMatrixVectorDotProduct(PyObject* A, PyObject* v, int ASize, int VSize, double* result);
void ccMatrixVectorDotProduct(double** X, const double * w, double* prediction, int rows, int cols);
void cPyVectorSubtract(const double* prediction, PyObject* y, double* loss, int rows);
double cVectorSum(const double* array, int rows);
void cVectorDivide(double* X, int n, int size);
void pyTranspose(PyObject* X, double** result, int rows, int cols);
void vector_power(PyObject* A, int pPower, int ASize, double* result);
void pyCVectorSubtract(PyObject* u, PyObject* v, int ASize, double* result);
double pyVectorSum(PyObject* u, int size);
void pypyMatrixMatrixProduct(PyObject* A, PyObject* B, int ASize, int BSize, double** result);
void pyLeastSquares(PyObject* X, PyObject* y, double* theta, int n, int m);

#ifdef __cplusplus
}
#endif

#endif //SRC_LINEARALGEBRAMODULE_H
