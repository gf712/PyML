//
// Created by Gil Ferreira Hoben on 07/11/17.
//

#include <flatArrays.h>
#include "linearalgebramodule.h"

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


flatArray *covariance(flatArray *X) {

    // variable declaration
    auto covMatrix = new flatArray;
    int cols, rows;
    flatArray *XVar = nullptr;
    auto vecProd = new flatArray;

    // get number of cols and rows (quicker than calling getter all the time)
    cols = X->getCols();
    rows = X->getRows();

    // initialise covariance matrix
    // if X is a n by m matrix
    // the covariance matrix is m by m
    covMatrix->startEmptyArray(cols, cols);

    // get mean and variance of each column
    flatArray *XVecMean = X->mean(0);
    XVar = X->var(0, 0);

    vecProd->startEmptyArray(1, rows);

    for (int i = 0; i < cols; ++i) {
        // the diagonal of the covariance matrix is the column wise variance of X
        covMatrix->setNElement(XVar->getNElement(i), i + i * cols);

        // get ith vector
        double *Vec1 = X->getCol(i);

        // only need to calculate the upper triangle
        for (int j = cols - 1; j > i; --j) {

            // get jth vector
            double *Vec2 = X->getCol(j);

            for (int k = 0; k < rows; ++k) {
                vecProd->setNElement(Vec1[k] * Vec2[k], k);
            }

            // cov(X, Y) = E(X*Y) - E(X)*E(Y)
            // where X is the ith vector and Y the jth vector
            flatArray *vecProdMean = vecProd->mean(0);
            double result = vecProdMean->getNElement(0) - XVecMean->getNElement(i) * XVecMean->getNElement(j);

            // set S(i, j) = S(j, i) = cov(i, j)
            covMatrix->setNElement(result, i * cols + j);
            covMatrix->setNElement(result, j * cols + i);

            delete [] Vec2;
            delete vecProdMean;
        }

        delete [] Vec1;
    }

    // memory dellocation
    delete XVecMean;
    delete XVar;
    delete vecProd;

    return covMatrix;
}

void *maxElementOffDiag(flatArray *S, double result[3]) {

    int n = S->getCols();

    result[0] = 0;
    result[1] = 0;
    result[2] = 0;

    for (int k = 0; k < n; ++k) {

        double *row = S->getRowSlice(k, k + 1, n);

        int j = 0;
        for (int i = k + 1; i < n; ++i) {

            if (fabs(row[j]) >= result[0]) {
                result[0] = fabs(row[j]);
                result[1] = k;
                result[2] = i;
            }
            j++;
        }

        delete [] row;
    }

    return result;
}


flatArray *jacobiEigenDecomposition(flatArray *S, double tolerance, int maxIterations) {

    // Implementation of the Jacobi rotation algorithm
    // https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
    // http://www.southampton.ac.uk/~feeg6002/lecturenotes/feeg6002_numerical_methods08.pdf
    //
    // if S is a n by n matrix, then:
    // return a flatMatrix (n + 1 by n) with the first row corresponding to the eigenvalues (n-dimensional vector)
    // and below are the eigenvector (n by n matrix)

    int l, k;
    double s, c, t, y, temp, diff, phi;

    auto E = new flatArray;
    auto result = new flatArray;

    double *maxValues = nullptr;
    maxValues = new double[3];

    // number of rows
    int n = S->getRows();

    // set max iterations to 5 * n ** 2 if this value is not set
    if (maxIterations == 0) {
        maxIterations = static_cast<int>( 5 * pow(n, 2));
    }

    // initialise e, E, ind and changed
    // memory allocation
    result->startEmptyArray(n + 1, n);

    // initialise values
    E->identity(n);

    int iteration = 0;

    while (iteration < maxIterations) {

        //  get max values off the diagonal
        maxElementOffDiag(S, maxValues);
        k = (int) maxValues[1];
        l = (int) maxValues[2];

        if (maxValues[0] < tolerance) {
            // found convergence
            break;
        }

        // Jacobi rotation
        diff = S->getNElement(l * n + l) - S->getNElement(k * n + k);

        if (fabs(S->getNElement(k * n + l)) < fabs(diff) * 1.0e-40) {
            y = S->getNElement(k * n + l) / diff;
        }

        else {
            phi = diff / (2 * S->getNElement(k * n + l));
            y = 1.0 / (fabs(phi) + sqrt(pow(phi, 2) + 1.0));
            if (phi < 0) {
                y = -y;
            }
        }

        c = 1.0 / sqrt(pow(y, 2) + 1);
        s = y * c;
        t = s / (1 + c);

        temp = S->getNElement(k * n + l);

        S->setNElement(0, k * n + l);
        S->setNElement(S->getNElement(k * n + k) - y * temp, k * n + k);
        S->setNElement(S->getNElement(l * n + l) + y * temp, l * n + l);

        for (int i = 0; i < k; ++i) {
            // temp = a[i,k]
            // a[i,k] = temp - s*(a[i,l] + tau*temp)
            // a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
            temp = S->getNElement(i * n + k);
            S->setNElement(temp - s*(S->getNElement(i * n + l) + t * temp), i * n + k);
            S->setNElement(S->getNElement(i * n + l) + s * (temp - t * S->getNElement(i * n + l)), i * n + l);
        }

        for (int i = k + 1; i < l; ++i) {
            //  temp = a[k,i]
            // a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            // a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
            temp = S->getNElement(k * n + i);
            S->setNElement(temp - s * (S->getNElement(i * n + l) + t * S->getNElement(k * n + i)), k * n + i);
            S->setNElement(S->getNElement(i * n + l) + s * (temp - t * S->getNElement(i * n + l)), i * n + l);

        }

        for (int i = l + 1; i < n; ++i) {
            // temp = a[k,i]
            // a[k,i] = temp - s*(a[l,i] + tau*temp)
            // a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
            temp = S->getNElement(k * n + i);
            S->setNElement(temp - s * (S->getNElement(n * l + i) + t * temp), k * n + i);
            S->setNElement(S->getNElement(l * n + i) + s * (temp - t * S->getNElement(l * n + i)), l * n + i);
        }

        for (int i = 0; i < n; ++i) {
            // temp = p[i,k]
            // p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            // p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
            temp = E->getNElement(i * n + k);
            E->setNElement(temp - s * (E->getNElement(i * n + l) + t * E->getNElement(i * n + k)), i * n + k);
            E->setNElement(E->getNElement(i * n + l) + s * (temp - t * E->getNElement(i * n + l)), i * n + l);
        }

        iteration++;
    }

    // the diagonal of S has the eigenvalues
    double* diag = S->diagonal();
    result->setRow(diag, 0);

    for (int j = 1; j < n + 1; ++j) {
        double *row = E->getRow(j -1);
        result->setRow(row, j);
        delete [] row;
    }

    // memory deallocation
    delete E;
    delete [] diag;
    delete [] maxValues;

    return result;
}
