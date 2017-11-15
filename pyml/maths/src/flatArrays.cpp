//
// Created by Gil Ferreira Hoben on 10/11/17.
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

flatArray *flatArray::dot(flatArray *other) {

    if (other->getRows() > 1) {
        // matrix matrix multiplication
        if (cols != other->getRows()) {
            PyErr_SetString(PyExc_ValueError, "Number of columns in A must be the same as the number of rows in B");
            return nullptr;
        }

        auto result = new flatArray;
        result->startEmptyArray(rows, other->getCols());

        int rRows = result->getRows();
        int rCols = result->getCols();
        int N = getCols();
        int M = other->getCols();
        int n, i, j, k, posA, posB;
        double eResult;
        double *otherArray = other->getArray();

        for (i = 0; i < rRows; ++i) {
            for (j = 0; j < rCols; ++j) {
                posA = i * N;
                posB = j;
                eResult = 0;
                n = i * M + j;
                for (k = 0; k < N; ++k) {
                    eResult += array[posA] * otherArray[posB];

                    posA++;
                    posB += M;
                }

                result->setNElement(eResult, n);
            }
        }

        return result;
    }

    else if (other->getRows() == 1) {
        // matrix vector multiplication
        if (cols != other->getCols()){
            PyErr_SetString(PyExc_ValueError, "A and v must have the same size");
            return nullptr;
        }

        auto result = new flatArray;
        result->startEmptyArray(1, rows);
        double *v = other->getArray();

        int n = 0;
        for (int i = 0; i < rows; ++i) {
            double row_result  = 0;
            for (int j = 0; j < cols; ++j) {
                row_result += array[n] * v[j];
                n++;
            }
            result->setNElement(row_result, i);
        }

        return result;

    }

    else {
        // ERROR
        return nullptr;
    }
}

flatArray *flatArray::subtract(flatArray *other) {

    auto result = new flatArray;

    if (rows != other->getRows()) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have matching number of rows.");
        return nullptr;
    }
    if (cols != other->getCols()) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have matching number of columns.");
        return nullptr;
    }

    result->startEmptyArray(rows, cols);

    double *B = other->getArray();

    for (int n = 0; n < size; ++n) {
        result->setNElement(array[n] - B[n], n);
    }

    return result;

}

flatArray *flatArray::power(int p) {

    auto result = new flatArray;

    result->startEmptyArray(rows, cols);

    for (int n = 0; n < size; ++n) {
        result->setNElement(pow(array[n], p), n);
    }

    return result;
}

flatArray *flatArray::divide(double m) {

    auto result = new flatArray;

    result->startEmptyArray(rows, cols);

    for (int n = 0; n < size; ++n) {
        result->setNElement(array[n] / m, n);
    }

    return result;
}

void flatArray::setRows(int r) {
    rows = r;
}

void flatArray::setCols(int c) {
    cols = c;
}