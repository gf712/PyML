//
// Created by Gil Ferreira Hobe on 07/11/17.
//
#include <Python.h>
#include "pythonconverters.h"


PyObject* Convert_1DArray(double array[], int length) {

    PyObject *pylist;
    PyObject *item;
    int i;

    pylist = PyList_New(length);

    if (pylist != nullptr) {

        for (i=0; i<length; i++) {
            item = PyFloat_FromDouble(array[i]);
            PyList_SET_ITEM(pylist, i, item);

        }

    }

    return pylist;
}


PyObject* Convert_2DArray(double** array, int rows, int cols) {
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