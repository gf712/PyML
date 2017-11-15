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
void convertPy_1DArray(PyObject* array, double* result, int size);
void convertPy_2DArray(PyObject* array, double** result, int rows, int cols);
void convertPy_flatArray(PyObject *array, flatArray *result);
PyObject* ConvertFlatArray_PyList(flatArray *array, const char pyType[5]);

#ifdef __cplusplus
}
#endif

#endif //MATHS_PYTHONCONVERTERS_H
