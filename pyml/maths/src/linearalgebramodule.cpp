#include <flatArrays.h>
#include <Python.h>
#include "linearalgebramodule.h"
#include <omp.h>
#include <iostream>

// Handle errors
// static PyObject *algebraError;

double dotProduct(const double* u, const double* v, int size) {

    double result = 0;

    for (int i = 0; i < size; ++i) {

        result += v[i] * u[i];
    }

    return result;
}


void matrixVectorDotProduct(double** A, double* v, int ASize, int VSize, double* result) {

    for (int i = 0; i < ASize ; ++i) {

        result[i] = dotProduct(A[i], v, VSize);

    }

}


void matrixMatrixProduct(double** A, double** B, int rows, int cols, double** result) {

    // it's easier to just transpose B
    double** other = nullptr;

    other = new double *[rows];
    for (int i = 0; i < rows; ++i) {
        other[i] = new double [cols];
    }

    matrixTranspose(B, other, cols, rows, 16);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            result[i][j] = dotProduct(A[i], other[j], cols);
        }
    }

    for (int i = 0; i < rows; ++i) {
        delete other[i];
    }

    delete [] other;

}

void flatMatrixMatrixProduct(flatArray *A, flatArray *B, flatArray *result) {

    int rRows = result->getRows();
    int rCols = result->getCols();
    int N = A->getCols();
    int M = B->getCols();
    int n, i, j, k, posA, posB;
    double eResult;

    {
    #pragma omp parallel for default(shared) private (i, j, k, n, posA, posB, eResult) schedule(static) collapse(2)
        for (i = 0; i < rRows; ++i) {
            for (j = 0; j < rCols; ++j) {
                posA = i * N;
                posB = j;
                eResult = 0;
                n = i * M + j;
                for (k = 0; k < N; ++k) {
                    eResult += A->getNElement(posA) * B->getNElement(posB);

                    posA++;
                    posB += M;
                }

                result->setNElement(eResult, n);
            }
        }
    }
}


void flatMatrixVectorDotProduct(flatArray *X, flatArray *V, flatArray *result) {
    int n = 0;
    for (int i = 0; i < X->getRows(); ++i) {
        double row_result  = 0;
        for (int j = 0; j < X->getCols(); ++j) {
            row_result += X->getNElement(n) * V->getNElement(j);
            n++;
        }
        result->setNElement(row_result, i);
    }
}


void flatMatrixPower(flatArray *A, int p) {
    for (int n = 0; n < A->getSize(); ++n) {
        A->setNElement(pow(A->getNElement(n), p), n);
    }
}


void vectorSubtract(const double* u, const double* v, int size, double* result) {

    for (int i = 0; i < size; ++i) {

        result[i] = u[i] - v[i];

    }
}

void flatArraySubtract(flatArray *A, flatArray *B, flatArray *result) {

    for (int n = 0; n < A->getSize(); ++n) {
        result->setNElement(A->getNElement(n) - B->getNElement(n), n);
    }
}

double vectorSum(const double* array, int rows) {

    double result=0;

    for (int i = 0; i < rows; ++i) {
        result += array[i];
    }

    return result;
}


void vectorDivide(double* X, int n, int size) {
    for (int i = 0; i < size; ++i) {
        X[i] /= (double)n;
    }
}


void matrixTranspose(double** X, double** result, int rows, int cols, int block_size) {

//    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = X[i][j];
        }
    }
}


void maximumSearch(double* vector, int size, int i, double* result) {
    // store result in result array
    // result[0] is the maximum value and result[1] is the position of the maximum value
    result[0] = vector[i];
    result[1] = i;

    for (int k=i+1; k<size; k++) {

        if (abs(vector[i]) > result[0]) {

            // if this element is larger than the previous maximum store result
            result[0] = vector[k];
            result[1] = k;

        }
    }
}


void swapRows(flatArray *A, int row1, int row2) {

    double *aRow1 = A->getRow(row1);
    double *aRow2 = A->getRow(row2);

    A->setRow(aRow2, row1);
    A->setRow(aRow1, row2);

    delete [] aRow1;
    delete [] aRow2;

}


void gaussianElimination(flatArray *A, double *result) {

    // based on https://martin-thoma.com/images/2013/05/Gaussian-elimination.png

    int n = A->getRows();

    double *maxResult = nullptr;
    maxResult = new double[2];
    double c;
    double *rowI = nullptr;
    double *rowK = nullptr;

    for (int i = 0; i < n; ++i) {
        // Search for maximum in this column
        maximumSearch(A->getRow(i), n, i, maxResult);

        // Swap maximum row with current row (column by column)
        swapRows(A, (int) maxResult[1], i);

        rowI = A->getRow(i);
        // Make all rows below this one 0 in current column
        for (int k = i + 1; k < n; k++) {

            rowK = A->getRow(k);

            c = -rowK[i] / rowI[i];

            for (int j = i; j < n + 1; j++) {

                if (i == j) {

                    rowK[j] = 0;

                } else {

                    rowK[j] += c * rowI[j];

                }

                A->setRow(rowK, k);
            }

            A->setRow(rowI, i);
        }
    }

    // A is the U matrix

    for (int i = n - 1; i >= 0; i--) {

        result[i] = A->getElement(i,n) / A->getElement(i, i);

        for (int j = i - 1; j >= 0; j--) {
            A->setElement(A->getElement(j, n) - A->getElement(j, i) * result[i], j, n);
        }
    }

    delete [] maxResult;
    delete [] rowI;
    delete [] rowK;
}


void leastSquares(flatArray *X, flatArray *y, double *theta) {

    // variable declaration
    auto XTX = new flatArray;
    auto A = new flatArray;
    auto right = new flatArray;
    int m = X->getCols();
    int n;
    int XTXn=0;

    // memory allocation
    XTX->startEmptyArray(m, m);
    A->startEmptyArray(m, m+1);
    right->startEmptyArray(m, 1);

    // start algorithm

    // first transpose X and get XT
    flatArray *XT = X->transpose();

    // XT is a m by n matrix
    // an m by n matrix multiplied by a n by m matrix results in a m by m matrix
    // so XTX is a m by m matrix
    flatMatrixMatrixProduct(XT, X, XTX);

    // XT is a m by n matrix
    // y is a n dimensional vector
    // the result is a m dimensional vector called right (since it fits to the right of the A matrix)
    flatMatrixVectorDotProduct(XT, y, right);

    // fill in A which is a m by m + 1 matrix
    n = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            A->setNElement(XTX->getNElement(XTXn),n);
            n++;
            XTXn++;
        }
        A->setNElement(right->getNElement(i), n);
        n++;
    }

    // now we can perform gaussian elimination to get an approximation of theta
    gaussianElimination(A, theta);

    // and free up memory
    delete XTX;
    delete XT;
    delete right;
    delete A;
}


double vectorMean(const double* array, int size) {

    double result = 0;

    for (int i = 0; i < size; ++i) {
        result += array[i];
    }

    return result / static_cast<double>(size);

}


void matrixMean(double** array, int cols, int rows, int axis, double* result) {

    if (axis == 0) {
        // mean of each column

        for (int i = 0; i < cols; ++i) {
            result[i] = 0;
            for (int j = 0; j < rows; ++j) {
                result[i] += array[j][i];
            }
            result[i] /= rows;
        }
    }

    else {
        // mean of each row
        for (int i = 0; i < rows; ++i) {
            result[i] = vectorMean(array[i], cols);
        }
    }
}
