/**
 *  @file    flatArrays.cpp
 *  @author  Gil Ferreira Hoben (gf712)
 *  @date    10/11/2017
 *  @version 0.1
 *
 *  @brief Flat representation of 2D arrays
 *
 *  @section DESCRIPTION
 *
 *  This class represents 2D arrays with a 1D array
 *  This improves computational as less time is spent
 *  following pointers of pointers
 *
 */
#include "pythonconverters.h"
#include "../../utils/include/exceptionClasses.h"


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


void flatArray::startEmptyArray(int rows, int cols) {
    flatArray::rows = rows;
    flatArray::cols = cols;
    size = rows * cols;
    array = nullptr;
    array = new double [size];
}


void flatArray::identity(int n) {

    startEmptyArray(n, n);

    int row=0;

    for (int i = 0; i < size; ++i) {
        if (i == row) {
            array[i] = 1;
            row += cols + 1;
        }
        else {
            array[i] = 0;
        }
    }
}


void flatArray::constArray(int rows_, int cols_, int c) {

    startEmptyArray(rows_, cols_);

    for (int i = 0; i < size; ++i) {
        array[i] = c;
    }

}


void flatArray::zeroArray(int rows_, int cols_) {

    constArray(rows_, cols_, 0);
}


void flatArray::oneArray(int rows_, int cols_) {

    constArray(rows_, cols_, 1);
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

    return result;
}

double *flatArray::getRow(int i) {

    double *row = nullptr;

    row = new double [cols];
    int n = 0;

    for (int j = i * cols; j < (i + 1) * cols; ++j) {
        row[n] = array[j];
        n++;
    }

    return row;
}

double *flatArray::getCol(int j) {

    double *column = nullptr;

    column = new double [rows];
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

void flatArray::setCol(double *column, int j) {

    int n = 0;
    for (int k = j; k < size; k+=cols) {
        array[k] = column[n];
        n++;
    }
}

flatArray *flatArray::dot(flatArray *other) {

    if (other->getRows() > 1) {
        // matrix matrix multiplication
        if (cols != other->getRows()) {
            throw flatArrayColumnMismatch(this, other);
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
            throw flatArrayColumnMismatch(this, other);
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
        throw flatArrayDimensionMismatch(this, other);
    }
}

flatArray *flatArray::subtract(flatArray *other) {

    auto result = new flatArray;

    result->startEmptyArray(rows, cols);

    if (other->getRows() == 1 && rows > 1) {
        // other is a vector and this is a matrix

        // if square matrix will prioritise column wise subtraction (consider transposing matrix
        // if this is not what you want)
        if (other->getCols() == rows) {
            // number of rows match number of dimensions of vector

            double *B = other->getRow(0);

            int n = 0;
            for (int i = 0; i < rows; ++i) {

                for (int j = 0; j < cols; ++j) {

                    // subtract each row by the ith element of other vector
                    result->setNElement(array[n] - B[i], n);

                    n++;
                }
            }

            delete [] B;
        }

        else if (other->getCols() == cols){
            double *B = other->getRow(0);

            int n = 0;
            for (int i = 0; i < rows; ++i) {

                for (int j = 0; j < cols; ++j) {

                    // subtract each column by the ith element of other vector
                    result->setNElement(array[n] - B[j], n);

                    n++;
                }
            }

            delete [] B;
        }

        else {
            throw flatArrayDimensionMismatch(this, other);
        }
    }

    else if (other->getRows() == 1 && other->getCols() == 1) {
        // other is a scalar represented as an array
        double B = other->getNElement(0);

        for (int i = 0; i < size; ++i) {
            result->setNElement(array[i] - B, i);
        }

    }

    else if (other->getSize() == size) {
        // both are matrices or vectors (with same size)

        if (rows != other->getRows()) {
            throw flatArrayRowMismatch(this, other);
        }
        if (cols != other->getCols()) {
            throw flatArrayColumnMismatch(this, other);
        }

        double *B = other->getArray();

        for (int n = 0; n < size; ++n) {
            result->setNElement(array[n] - B[n], n);
        }
    }

    else {
        throw flatArrayDimensionMismatch(this, other);
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
    // update size
    size = rows * cols;
}

void flatArray::setCols(int c) {
    cols = c;
    // update size
    size = rows * cols;
}

flatArray *flatArray::multiply(flatArray *other) {
    auto result = new flatArray;

    result->startEmptyArray(rows, cols);

    for (int n = 0; n < size; ++n) {
        result->setNElement(array[n] * other->getNElement(n), n);
    }

    return result;
}

flatArray *flatArray::nlog(double base) {
    auto result = new flatArray;


    result->startEmptyArray(rows, cols);

    for (int n = 0; n < size; ++n) {
        result->setNElement(log(array[n]) / log(base), n);
    }

    return result;
}

flatArray *flatArray::mean(int axis) {

    auto result = new flatArray;

    if (rows == 1) {
        // vector
        double rowResult = 0;

        result->startEmptyArray(1, 1);

        for (int i = 0; i < cols; ++i) {
            rowResult += array[i];
        }

        rowResult /= (double) cols;

        result->setNElement(rowResult, 0);
    }

    else if (rows > 1) {
        // matrix
        if (axis == 0) {
            // mean of each column
            double colResult;

            result->startEmptyArray(1 , cols);

            for (int i = 0; i < cols; ++i) {

                colResult = 0;
                double *colArray = getCol(i);

                for (int j = 0; j < rows; ++j) {
                    colResult += colArray[j];
                }

                colResult /= (double) rows;

                result->setNElement(colResult, i);

                delete [] colArray;
            }
        }

        else {
            // mean of each row
            double rowResult;

            result->startEmptyArray(1, rows);

            for (int i = 0; i < rows; ++i) {

                rowResult = 0;
                double *rowArray = getRow(i);

                for (int j = 0; j < cols; ++j) {
                    rowResult += rowArray[j];
                }

                rowResult /= (double) cols;

                result->setNElement(rowResult, i);

                delete [] rowArray;
            }
        }
    }

    return result;
}

flatArray *flatArray::std(int degreesOfFreedom, int axis) {

    auto result = new flatArray;

    flatArray *arrayVar = var(degreesOfFreedom, axis);

    if (rows == 1) {
        // vector
        result->startEmptyArray(1, 1);

        double arrayMean_i = arrayVar->getNElement(0);

        result->setNElement(pow(arrayMean_i, 0.5), 0);
    }

    else if (rows > 1) {
        // matrix
        if (axis == 0) {
            // std of each column
            result->startEmptyArray(1, cols);

            for (int i = 0; i < cols; ++i) {

                result->setNElement(pow(arrayVar->getNElement(i), 0.5), i);
            }

        }

        else {
            // std of each row
            result->startEmptyArray(1, rows);

            for (int i = 0; i < rows; ++i) {

                result->setNElement(pow(arrayVar->getNElement(i), 0.5), i);

            }
        }
    }

    delete arrayVar;

    return result;
}


flatArray *flatArray::var(int degreesOfFreedom, int axis) {

    auto result = new flatArray;

    flatArray *arrayMean = mean(axis);

    if (rows == 1) {
        // vector
        double rowResult = 0;

        result->startEmptyArray(1, 1);

        double arrayMean_i = arrayMean->getNElement(0);

        for (int i = 0; i < size; ++i) {
            rowResult += pow(array[i] - arrayMean_i, 2);
        }

        rowResult /= (double) (size - degreesOfFreedom);

        result->setNElement(rowResult, 0);
    }

    else if (rows > 1) {
        // matrix
        if (axis == 0) {
            // std of each column
            double colResult;

            result->startEmptyArray(1, cols);

            for (int i = 0; i < cols; ++i) {

                colResult = 0;
                double colMean = arrayMean->getNElement(i);
                double *colArray = getCol(i);

                for (int j = 0; j < rows; ++j) {
                    colResult += pow(colArray[j] - colMean, 2);
                }

                colResult /= (double) (rows - degreesOfFreedom);

                result->setNElement(colResult, i);

                delete [] colArray;
            }
        }

        else {
            // std of each row
            double rowResult;

            result->startEmptyArray(1, rows);

            for (int i = 0; i < rows; ++i) {

                rowResult = 0;
                double rowMean = arrayMean->getNElement(i);
                double *rowArray = getRow(i);

                for (int j = 0; j < cols; ++j) {
                    rowResult += pow(rowArray[j] - rowMean, 2);
                }

                rowResult /= (double) (cols - degreesOfFreedom);

                result->setNElement(rowResult, i);

                delete [] rowArray;
            }
        }
    }

    delete arrayMean;

    return result;
}


double *flatArray::getRowSlice(int i, int start, int end) {

    double *row = getRow(i);
    double *result = nullptr;

    result = new double [end - start];

    int n = 0;
    for (int j = start; j < end; ++j) {

        result[n] = row[j];

        n++;
    }

    delete [] row;

    return result;
}

double *flatArray::getColSlice(int j, int start, int end) {

    double *col = getCol(j);
    double *result = nullptr;

    result = new double[end - start];

    int n = 0;
    for (int i = start; i < end; ++i) {

        result[n] = col[i];

        n++;
    }

    delete [] col;

    return result;
}

double *flatArray::diagonal() {

    double *result = nullptr;

    result = new double [rows];

    for (int i = 0; i < rows; ++i) {
        result[i] = array[i + i * rows];
    }

    return result;
}
