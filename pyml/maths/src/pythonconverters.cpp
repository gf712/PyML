//
// Created by Gil Ferreira Hobe on 07/11/17.
//
#include <Python.h>
#include "pythonconverters.h"
#include <iostream>


PyObject* Convert_1DArray(double* array, int size) {

    // converts a C++ 1D array to a python list

    PyObject *pylist;
    PyObject *item;
    int i;

    pylist = PyList_New(size);

    if (pylist != nullptr) {

        for (i=0; i < size; i++) {
            item = PyFloat_FromDouble(array[i]);
            PyList_SET_ITEM(pylist, i, item);

        }
    }

    return pylist;
}


PyObject* Convert_1DArrayInt(long* array, int size) {

    // converts a C++ 1D array to a python list

    PyObject *pylist;
    PyObject *item;
    int i;

    pylist = PyList_New(size);

    if (pylist != nullptr) {

        for (i=0; i < size; i++) {
            item = PyLong_FromLong(array[i]);
            PyList_SET_ITEM(pylist, i, item);

        }
    }

    return pylist;
}


PyObject* Convert_2DArray(double** array, int rows, int cols) {

    // converts a C++ 2D array to a python list

    PyObject* twoDResult;
    PyObject* row;
    PyObject* item;

    twoDResult = PyList_New(rows);

    if (twoDResult != nullptr) {

        for (int j = 0; j < rows; ++j) {
            row = PyList_New(cols);

            if (row != nullptr) {

                for (int k = 0; k < cols; ++k) {
                    item = PyFloat_FromDouble(array[j][k]);
                    PyList_SET_ITEM(row, k, item);
                }

                PyList_SET_ITEM(twoDResult, j, row);
            }


        }

    }

    return twoDResult;
}


PyObject* ConvertFlat2DArray_2DPy(double* array, int rows, int cols) {

    // converts a 1D C++ array representation of a 2D array to a python list of lists

    PyObject* twoDResult;
    PyObject* row;
    PyObject* item;

    // internal representation of the array
    int n = 0;

    twoDResult = PyList_New(rows);

    if (twoDResult != nullptr) {

        for (int j = 0; j < rows; ++j) {
            row = PyList_New(cols);

            if (row != nullptr) {

                for (int k = 0; k < cols; ++k) {
                    item = PyFloat_FromDouble(array[n]);
                    PyList_SET_ITEM(row, k, item);
                    n++;
                }

                PyList_SET_ITEM(twoDResult, j, row);
            }


        }

    }

    return twoDResult;
}


void convertPy_1DArray(PyObject* array, double* result, int size) {

    // converts a python list to a C++ 1D array

    // iterate through python list and populate C++ array
    for (int i = 0; i < size; ++i) {
        result[i] = PyFloat_AsDouble(PyList_GET_ITEM(array, i));
    }
}


void convertPy_2DArray(PyObject* array, double** result, int rows, int cols) {

    // converts a python list to a C++ 2D array

    // row is a python object that will point to the current row

    PyObject* row;

    // iterate through python list and populate C++ array
    for (int i = 0; i < rows; ++i) {

        row = PyList_GET_ITEM(array, i);

        if (cols != PyList_GET_SIZE(row)) {
            std::string error1 = "Size of row ";
            std::string error2 = " is ";
            std::string error3 = " but expected row of size ";
            int length = PyList_GET_SIZE(row);
            std::string resultE;
            resultE = error1 + std::to_string(i) + error2 + std::to_string(length) + error3 + std::to_string(cols);
            PyErr_SetString(PyExc_ValueError, resultE.c_str());
        }

        for (int j = 0; j < cols; ++j) {
            result[i][j] = PyFloat_AsDouble(PyList_GET_ITEM(row, j));
        }

    }
}


void convertPy2D_flat2DArray(PyObject *array, double *result, int rows, int cols) {

    // converts a python list of lists to a C++ 1D array representation of a 2D array

    // row is a python object that will point to the current row

    PyObject* row;
    // n is the position in the flat matrix
    int n = 0;

    // iterate through python list and populate C++ array
    for (int i = 0; i < rows; ++i) {

        row = PyList_GET_ITEM(array, i);

        if (cols != PyList_GET_SIZE(row)) {
            std::string error1 = "Size of row ";
            std::string error2 = " is ";
            std::string error3 = " but expected row of size ";
            int length = PyList_GET_SIZE(row);
            std::string resultE;
            resultE = error1 + std::to_string(i) + error2 + std::to_string(length) + error3 + std::to_string(cols);
            PyErr_SetString(PyExc_ValueError, resultE.c_str());
        }

        for (int j = 0; j < cols; ++j) {
            result[n] = PyFloat_AsDouble(PyList_GET_ITEM(row, j));
            n++;
        }
    }
}
