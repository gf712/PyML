//
// Created by Gil Ferreira Hoben on 07/11/17.
//

#include <flatArrays.h>
#include "flatArrays.cpp"
#include "linearalgebramodule.h"
#include "exceptionClasses.h"

// Handle errors
// static PyObject *algebraError;


inline void maximumSearch(double* vector, int size, int i, double* result) {
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


template <typename T>
inline void swapRows(flatArray<T> *A, int row1, int row2) {

    T *aRow1 = A->getRow(row1);
    T *aRow2 = A->getRow(row2);

    A->setRow(aRow2, row1);
    A->setRow(aRow1, row2);

    delete [] aRow1;
    delete [] aRow2;

}

template <typename T>
void gaussianElimination(flatArray<T> *A, T *result) {

    // based on https://martin-thoma.com/images/2013/05/Gaussian-elimination.png

    int n = A->getRows();

    T *maxResult = nullptr;
    maxResult = new T[2];
    T c;
    T *rowI = nullptr;
    T *rowK = nullptr;

    for (int i = 0; i < n; ++i) {
        // Search for maximum in this column
        maximumSearch(A->getRow(i), n, i, maxResult);

        if (maxResult[0] == 0) {
            throw singularMatrixException();
        }

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


template <typename T>
void leastSquares(flatArray<T> &X, flatArray<T> &y, T *theta) {

    // variable declaration
    flatArray<T>* A = nullptr;
    flatArray<T>* XT = nullptr;
    flatArray<T>* XTX = nullptr;
    flatArray<T>* right = nullptr;

    int m = X.getCols();
    int n;
    int XTXn=0;

    // memory allocation
    A = emptyArray<T>(m, m+1);

    // start algorithm

    // first transpose X and get XT
    XT = X.transpose();

    // XT is a m by n matrix
    // an m by n matrix multiplied by a n by m matrix results in a m by m matrix
    // so XTX is a m by m matrix
    XTX = XT->dot(X);

//    if (XTX->det() == 0) {
//        throw singularMatrixException();
//    }

    // XT is a m by n matrix
    // y is a n dimensional vector
    // the result is a m dimensional vector called right (since it fits to the right of the A matrix)
    right = XT->dot(y);

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
    try {
        gaussianElimination(A, theta);
    }
    catch (singularMatrixException &e) {
        throw;
    }

    // and free up memory
    delete XTX;
    delete XT;
    delete right;
    delete A;
}


template <typename T>
flatArray<T>* covariance(flatArray<T> *X) {

    // variable declaration
    flatArray<T>* covMatrix = nullptr;
    flatArray<T>* vecProd = nullptr;
    flatArray<T>* XVar = nullptr;
    flatArray<T>* XVecMean = nullptr;

    int cols, rows;
    T result;

    // get number of cols and rows (quicker than calling getter all the time)
    cols = X->getCols();
    rows = X->getRows();

    // initialise covariance matrix
    // if X is a n by m matrix
    // the covariance matrix is m by m
    covMatrix = emptyArray<T>(cols, cols);

    // get mean and variance of each column
    XVecMean = X->mean(0);
    XVar = X->var(0, 0);

    vecProd = emptyArray<T>(1, rows);

    for (int i = 0; i < cols; ++i) {
        // the diagonal of the covariance matrix is the column wise variance of X
        covMatrix->setNElement(XVar->getNElement(i), i + i * cols);

        T *Vec1 = nullptr;

        // get ith vector
        Vec1 = X->getCol(i);

        // only need to calculate the upper triangle
        for (int j = cols - 1; j > i; --j) {

            T *Vec2 = nullptr;
            flatArray<T>* vecProdMean = nullptr;

            // get jth vector
            Vec2 = X->getCol(j);

            for (int k = 0; k < rows; ++k) {
                vecProd->setNElement(Vec1[k] * Vec2[k], k);
            }

            // cov(X, Y) = E(X*Y) - E(X)*E(Y)
            // where X is the ith vector and Y the jth vector
            vecProdMean = vecProd->mean(0);

            result = vecProdMean->getNElement(0) - XVecMean->getNElement(i) * XVecMean->getNElement(j);

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

template <typename T>
void *maxElementOffDiag(flatArray<T> *S, T result[3]) {

    int n = S->getCols();

    result[0] = 0;
    result[1] = 0;
    result[2] = 0;

    for (int k = 0; k < n; ++k) {

        T *row = nullptr;

        row = S->getRowSlice(k, k + 1, n);

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

template <typename T>
flatArray<T>* jacobiEigenDecomposition(flatArray<T> *S, double tolerance, int maxIterations) {

    // Implementation of the Jacobi rotation algorithm
    // https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
    // http://www.southampton.ac.uk/~feeg6002/lecturenotes/feeg6002_numerical_methods08.pdf
    //
    // if S is a n by n matrix, then:
    // return a flatMatrix (n + 1 by n) with the first row corresponding to the eigenvalues (n-dimensional vector)
    // and below are the eigenvector (n by n matrix)

    int l, k;
    double s, c, t, y, temp, diff, phi;

    flatArray<T>* E = nullptr;
    flatArray<T>* result = nullptr;

    T *maxValues = nullptr;
    maxValues = new T[3];

    // number of rows
    int n = S->getRows();

    // set max iterations to 5 * n ** 2 if this value is not set
    if (maxIterations == 0) {
        maxIterations = static_cast<int>( 5 * pow(n, 2));
    }

    // initialise e, E, ind and changed
    // memory allocation
    result = emptyArray<T>(n + 1, n);

    // initialise values
    E = identity<T>(n);

    int iteration = 0;

    while (iteration < maxIterations) {

        //  get max values off the diagonal
        maxElementOffDiag(S, maxValues);
        k = static_cast<int>(maxValues[1]);
        l = static_cast<int>(maxValues[2]);

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
    auto diag = S->diagonal();

    result->setRow(diag, 0);

    for (int j = 1; j < n + 1; ++j) {
        auto *row = E->getRow(j -1);
        result->setRow(row, j);
        delete [] row;
    }

    // memory deallocation
    delete E;
    delete [] diag;
    delete [] maxValues;

    return result;
}

template <typename T>
inline flatArray<T>* signChart(int rows, int cols) {

    flatArray<T>* result = nullptr;

    result = emptyArray<T>(rows, cols);

    for (int i = 0; i < result->getSize(); ++i) {

        if (i % 2 == 0) {
            result->setNElement(1, i);
        }

        else {
            result->setNElement(-1 , i);
        }
    }

    return result;
}


template <typename T>
double determinant(flatArray<T>* array) {

    int rows = array->getRows();
    int cols = array->getCols();
    T determinantResult = 0;


    if (rows == 2) {
        // base case is the determinant of a 2 by 2 matrix
        determinantResult = array->getNElement(0) * array->getNElement(3) - array->getNElement(1) * array->getNElement(2);
    }

    else {
        flatArray<T>* M = nullptr;
        flatArray<T>* C = nullptr;
        flatArray<T>* signs = nullptr;

        // find determinant of minors and store in matrix M
        M = emptyArray<T>(rows, cols);

        int m = 0;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {

                flatArray<T> *M_i = nullptr;

                M_i = emptyArray<T>(rows - 1, cols - 1);

                // populate M_i with all rows except i, and all cols except j
                int n = 0;

                for (int k = 0; k < rows; ++k) {
                    for (int l = 0; l < cols; ++l) {

                        if (k != i && l != j) {
                            M_i->setNElement(array->getNElement(k * cols + l), n);
                            n++;
                        }
                    }
                }

                M->setNElement(determinant(M_i), m);

                m++;
                delete M_i;
            }
        }

        signs = signChart<T>(rows, cols);

        C = (*M) * (*signs);

        T* row = array->getRow(0);

        for (int i = 0; i < rows; ++i) {
            determinantResult += row[i] * C->getNElement(i);
        }

        delete [] row;
        delete signs;
        delete M;
        delete C;
    }

    return determinantResult;
}