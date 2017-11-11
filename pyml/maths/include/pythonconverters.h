//
// Created by gil on 07/11/17.
//
#include <Python.h>
#include "flatArrays.h"

#ifndef MATHS_PYTHONCONVERTERS_H
#define MATHS_PYTHONCONVERTERS_H

#ifdef __cplusplus
extern "C" {
#endif

PyObject* Convert_1DArray(double* array, int size);
PyObject* Convert_1DArrayInt(long* array, int size);
PyObject* Convert_2DArray(double** array, int rows, int cols);
void convertPy_1DArray(PyObject* array, double* result, int size);
void convertPy_2DArray(PyObject* array, double** result, int rows, int cols);
void convertPy2D_flat2DArray(PyObject *array, double *result, int rows, int cols);
PyObject* ConvertFlat2DArray_2DPy(flat2DArrays* array);

#ifdef __cplusplus
}
#endif

#endif //MATHS_PYTHONCONVERTERS_H
