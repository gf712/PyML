//
// Created by Gil Ferreira Hoben on 07/11/17.
//

#ifndef PYTHONCONVERTERS_H
#define PYTHONCONVERTERS_H

#include <Python.h>
#include <iostream>
#include <typeinfo>
#include "flatArrays.h"
#include "arrayInitialisers.h"


template <typename T>
inline PyObject* Convert_1DArray(T* array, int size) {

    // converts a C++ 1D array to a python list

    PyObject *pylist = nullptr;
    PyObject *item = nullptr;
    int i;

    pylist = PyList_New(size);

    if (pylist != nullptr) {

        for (i=0; i < size; i++) {
            if (typeid(T) == typeid(double)) {
                item = PyFloat_FromDouble(array[i]);
            }
            else if (typeid(T) == typeid(int)) {
                item = PyLong_FromLong(array[i]);
            }

            PyList_SET_ITEM(pylist, i, item);

        }
    }

    return pylist;
}


inline PyObject* Convert_1DArrayInt(long* array, int size) {

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


inline PyObject* Convert_2DArray(double** array, int rows, int cols) {

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


template<typename T>
inline PyObject* ConvertFlatArray_PyList(flatArray<T> *array, const char pyType[5]) {

    // converts a 1D C++ array representation of a 2D array to a python list of lists

    PyObject* result;
    PyObject* row;
    PyObject* item;

    // internal representation of the array
    int n = 0;

    if (array->getRows() > 1) {
        // if it's a matrix create list that will have N lists
        result = PyList_New(array->getRows());
    }
    else {
        // if it's a vector create a single list with M elements
        result = PyList_New(array->getCols());
    }

    if (result != nullptr) {
        if (array->getRows() > 1) {

            if (strcmp(pyType, "int") == 0) {

                for (int j = 0; j < array->getRows(); ++j) {

                    // if it's a matrix (instead of a vector) return a list of lists
                    row = PyList_New(array->getCols());

                    if (row != nullptr) {

                        for (int k = 0; k < array->getCols(); ++k) {
                            item = PyLong_FromDouble(array->getNElement(n));
                            PyList_SET_ITEM(row, k, item);
                            n++;
                        }

                        PyList_SET_ITEM(result, j, row);
                    }
                }
            }

            else {
                for (int j = 0; j < array->getRows(); ++j) {
                    // if it's a matrix (instead of a vector) return a list of lists
                    row = PyList_New(array->getCols());

                    if (row != nullptr) {

                        for (int k = 0; k < array->getCols(); ++k) {
                            item = PyFloat_FromDouble(array->getNElement(n));
                            PyList_SET_ITEM(row, k, item);
                            n++;
                        }

                        PyList_SET_ITEM(result, j, row);
                    }
                }
            }
        }

        else {

            if (strcmp(pyType, "int") == 0) {
                for (int k = 0; k < array->getCols(); ++k) {
                    item = PyLong_FromDouble(array->getNElement(k));
                    PyList_SET_ITEM(result, k, item);
                }
            }

            else {
                for (int k = 0; k < array->getCols(); ++k) {
                    item = PyFloat_FromDouble(array->getNElement(k));
                    PyList_SET_ITEM(result, k, item);
                }
            }
        }
    }

    return result;
}


template<typename T>
inline T* convertPy_1DArray(PyObject *array, int size) {
    // converts a python list to a C++ 1D array
    T* result;

    result = new T[size];

    // iterate through python list and populate C++ array
    for (int i = 0; i < size; ++i) {

        if (typeid(T) == typeid(double)) {
            result[i] = PyFloat_AsDouble(PyList_GET_ITEM(array, i));
        }

        else if (typeid(T) == typeid(int)) {
            result[i] = PyLong_AsLong(PyList_GET_ITEM(array, i));
        }
    }

    return result;
}


inline void convertPy_2DArray(PyObject* array, double** result, int rows, int cols) {

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


template <typename T>
flatArray<T>* convertPy_flatArray(PyObject *array, int rows, int cols) {

    // converts a python list/list of lists to a C++ 1D array representation of a 1D/2D array

    // row is a python object that will point to the current row
    flatArray<T>* result = nullptr;

    result = emptyArray<T>(rows, cols);

    PyObject* row;
    // n is the position in the flat matrix
    int n = 0;

    // iterate through python list and populate C++ array
    for (int i = 0; i < rows; ++i) {

        if (rows > 1) {
            row = PyList_GET_ITEM(array, i);
        }
        else {
            row = array;
        }

        if (cols != PyList_GET_SIZE(row)) {
            std::string error1 = "Size of row ";
            std::string error2 = " is ";
            std::string error3 = " but expected row of size ";
            auto length = static_cast<int>(PyList_GET_SIZE(row));
            std::string resultE;
            resultE = error1 + std::to_string(i) + error2 + std::to_string(length) + error3 + std::to_string(result->getCols());
            PyErr_SetString(PyExc_ValueError, resultE.c_str());
        }

        for (int j = 0; j < cols; ++j) {
            result->setNElement(PyFloat_AsDouble(PyList_GET_ITEM(row, j)), n);
            n++;
        }
    }

    return result;
}

#endif
