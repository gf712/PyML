//
// Created by Gil Ferreira Hoben on 07/11/17.
//


#include <flatArrays.h>
#include <Python.h>
#include "linearalgebramodule.h"
#include <iostream>

// Handle errors
// static PyObject *algebraError;


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
    auto A = new flatArray;
    int m = X->getCols();
    int n;
    int XTXn=0;

    // memory allocation
    A->startEmptyArray(m, m+1);

    // start algorithm

    // first transpose X and get XT
    flatArray *XT = X->transpose();

    // XT is a m by n matrix
    // an m by n matrix multiplied by a n by m matrix results in a m by m matrix
    // so XTX is a m by m matrix
    flatArray *XTX = XT->dot(X);

    // XT is a m by n matrix
    // y is a n dimensional vector
    // the result is a m dimensional vector called right (since it fits to the right of the A matrix)
    flatArray *right = XT->dot(y);

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
