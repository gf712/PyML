//
// Created by gil on 10/11/17.
//
#include "pythonconverters.h"

void flatArray::readFromPythonList(PyObject *pyList) {

    // read in array from Python list
    // check if it's a matrix or a vector
    if (PyFloat_Check(PyList_GET_ITEM(pyList, 0)) || PyLong_Check(PyList_GET_ITEM(pyList, 0))) {
        cols = static_cast<int>(PyList_GET_SIZE(pyList));
        rows = 1;
    }
    else {
        rows = static_cast<int>(PyList_GET_SIZE(pyList));
        cols = static_cast<int>(PyList_GET_SIZE(PyList_GET_ITEM(pyList, 0)));
    }
    // memory allocation of array
    startEmptyArray(rows, cols);

    convertPy_flatArray(pyList, this);
}

int flatArray::getRows() {
    return flatArray::rows;
}

int flatArray::getCols() {
    return flatArray::cols;
}

double *flatArray::getArray() {
    return flatArray::array;
}

double flatArray::getElement(int row, int col) {
    return array[cols*row+col%cols];
}

void flatArray::setElement(double value, int row, int col) {
    array[cols*row+col%cols] = value;
}

void flatArray::startEmptyArray(int rows, int cols) {
    flatArray::rows = rows;
    flatArray::cols = cols;
    flatArray::size = rows * cols;
    array = new double [size];
}

flatArray* flatArray::transpose() {
    // faster transpose method
    auto result = new flatArray;

    result->startEmptyArray(cols, rows);

    for (int n = 0; n < rows * cols; ++n) {
        int column = n / rows;
        int row = n % rows * cols;

        result->setNElement(array[row + column], n);
        }
    return result;
}

double flatArray::getNElement(int n) {
    return array[n];
}

void flatArray::setNElement(double value, int n) {
    array[n] = value;
}

int flatArray::getSize() {
    return size;
}

double flatArray::sum() {
    double result = 0;
    for (int n = 0; n < size; ++n) {
        result += array[n];
    };
}

double *flatArray::getRow(int i) {
    auto *row = new double [cols];
    int n = 0;

    for (int j = i * cols; j < (i + 1) * cols; ++j) {
        row[n] = array[j];
        n++;
    }

    return row;
}

double *flatArray::getCol(int j) {
    auto *column = new double [rows];
    int n = 0;

    for (int i = j; i < size; i+=cols) {
        column[n] = array[i];
        n++;
    }

    return column;
}

void flatArray::setRow(double *row, int i) {

    int n = 0;

    for (int j = i * cols; j < (i + 1) * cols; ++j) {
        array[j] = row[n];
        n++;
    }
}
