//
// Created by gil on 10/11/17.
//
#include "pythonconverters.h"
#include "flatArrays.h"
#include <Python.h>

void flat2DArrays::readFromPythonList(PyObject *pyList) {
    // read in array from Python list
    rows = static_cast<int>(PyList_GET_SIZE(pyList));
    cols = static_cast<int>(PyList_GET_SIZE(PyList_GET_ITEM(pyList, 0)));

    // memory allocation of array
    array = new double [rows * cols];

    convertPy2D_flat2DArray(pyList, array, rows, cols);
}

int flat2DArrays::getRows() {
    return flat2DArrays::rows;
}

int flat2DArrays::getCols() {
    return flat2DArrays::cols;
}

void flat2DArrays::setRows(int rows) {
    flat2DArrays::rows = rows;
}

void flat2DArrays::setCols(int cols) {
    flat2DArrays::cols = cols;
}

double *flat2DArrays::getArray() {
    return flat2DArrays::array;
}

double flat2DArrays::getElement(int row, int col) {
    return array[cols*row+col%cols];
}

void flat2DArrays::setElement(double value, int row, int col) {
    array[cols*row+col%cols] = value;
}

void flat2DArrays::startEmptyArray(int rows, int cols) {
    flat2DArrays::rows = rows;
    flat2DArrays::cols = cols;
    array = new double [rows * cols];
}

flat2DArrays* flat2DArrays::transpose() {
    // faster transpose method
    auto result = new flat2DArrays;

    result->startEmptyArray(cols, rows);

    for (int n = 0; n < rows * cols; ++n) {
        int row = n / rows;
        int column = n % rows * cols;

        result->setNElement(getNElement(row + column), n);
    }

    return result;
}

double flat2DArrays::getNElement(int n) {
    return array[n];
}

void flat2DArrays::setNElement(double value, int n) {
    array[n] = value;
}

flat2DArrays::flat2DArrays() = default;
