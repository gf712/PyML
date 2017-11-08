#include <Python.h>
#include "linearalgebramodule.h"

// Handle errors
// static PyObject *algebraError;

// Define vector dot product
double pypyDotProduct(PyObject* u, PyObject* v, int size) {

    double result = 0;

    for (int i = 0; i < size; ++i) {

        PyObject *v_item = PyList_GetItem(v, i);
        PyObject *u_item = PyList_GetItem(u, i);

        double pUItem = PyFloat_AsDouble(u_item);
        double pVItem = PyFloat_AsDouble(v_item);

        result += pUItem * pVItem;
    }

    return result;
}

double pyCDotProduct(PyObject* u, const double* v, int size) {

    double result = 0;

    for (int i = 0; i < size; ++i) {

        PyObject *u_item = PyList_GetItem(u, i);

        double pUItem = PyFloat_AsDouble(u_item);

        result += pUItem * v[i];
    }

    return result;
}

double cPyDotProduct(double* u, PyObject* v, int size) {

    double result = 0;

    for (int i = 0; i < size; ++i) {

        PyObject *v_item = PyList_GetItem(v, i);

        double pVItem = PyFloat_AsDouble(v_item);

        result += pVItem * u[i];
    }

    return result;
}

double ccDotProduct(const double* u, const double* v, int size) {

    double result = 0;

    for (int i = 0; i < size; ++i) {

        result += v[i] * u[i];
    }

    return result;
}


void pypyMatrixVectorDotProduct(PyObject* A, PyObject* v, int ASize, int VSize, double* result) {

    for (int i = 0; i < ASize ; ++i) {

        PyObject *A_item = PyList_GetItem(A, i);

        result[i] = pypyDotProduct(A_item, v, VSize);
    }

}


void pypyMatrixMatrixProduct(PyObject* A, PyObject* B, int rows, int cols, double** result) {

    // it's easier to just transpose B
    double** other = nullptr;

    other = new double *[rows];
    for (int i = 0; i < rows; ++i) {
        other[i] = new double [cols];
    }

    pyTranspose(B, other, cols, rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            result[i][j] = pyCDotProduct(PyList_GetItem(A, i), other[j], cols);
        }
    }

    for (int i = 0; i < rows; ++i) {
        delete other[i];
    }

    delete [] other;

}


void ccMatrixVectorDotProduct(double** X, const double * w, double* prediction, int rows, int cols) {

    for (int i = 0; i < rows; ++i) {
        prediction[i] = 0;
        for (int j = 0; j < cols; ++j) {
            prediction[i] += X[i][j] * w[j];
        }
    }
}

//double * matrix_matrix_dot_product(PyObject *A, PyObject *B, int ASize, int BSize) {
//
//
//
//}

void vector_power(PyObject* A, int pPower, int ASize, double* result) {

    int i;

    for (i = 0; i < ASize; ++i) {

        PyObject *A_item = PyList_GetItem(A, i);
        double pElement = PyFloat_AsDouble(A_item);

        result[i] = pow(pElement, pPower);

    }

}


void pyCVectorSubtract(PyObject* u, PyObject* v, int ASize, double* result) {

    int i;

    for (i = 0; i < ASize; ++i) {

        PyObject *uItem = PyList_GetItem(u, i);
        PyObject *vItem = PyList_GetItem(v, i);

        double pUElement = PyFloat_AsDouble(uItem);
        double pVElement = PyFloat_AsDouble(vItem);

        result[i] = pUElement - pVElement;

    }
}

void cPyVectorSubtract(const double* prediction, PyObject* y, double* loss, int rows) {

    for (int i = 0; i < rows; ++i) {
        loss[i] = prediction[i] - PyFloat_AsDouble(PyList_GetItem(y, i));
    }

}


double pyVectorSum(PyObject* u, int size) {

    double sum_result = 0;
    int i;

    for (i = 0; i < size; ++i) {

        PyObject *uItem = PyList_GetItem(u, i);

        double pUElement = PyFloat_AsDouble(uItem);

        sum_result += pUElement;

    }

    return sum_result;

}

double cVectorSum(const double* array, int rows) {

    double result=0;

    for (int i = 0; i < rows; ++i) {
        result += array[i];
    }

    return result;
}


void cVectorDivide(double* X, int n, int size) {
    for (int i = 0; i < size; ++i) {
        X[i] /= (double)n;
    }
}


void pyTranspose(PyObject* X, double** result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        PyObject* row = PyList_GetItem(X, i);
        for (int j = 0; j < cols; ++j) {
            result[j][i] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
}

void cTranspose(double** X, double** result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = X[i][j];
        }
    }
}


void cPyMatrixMatrixProduct(double** A, PyObject* B, int rows, int cols, double** result) {

    // it's easier to just transpose B
    double** other = nullptr;

    other = new double *[rows];
    for (int i = 0; i < rows; ++i) {
        other[i] = new double [cols];
    }

    pyTranspose(B, other, cols, rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            result[i][j] = ccDotProduct(A[i], other[j], cols);
        }
    }

    for (int i = 0; i < rows; ++i) {
        delete other[i];
    }

    delete [] other;

}

void cPyMatrixVectorDotProduct(double** A, PyObject* v, int ASize, int VSize, double* result) {

    for (int i = 0; i < ASize ; ++i) {

        result[i] = cPyDotProduct(A[i], v, VSize);
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


void swapRows(double** A, int row1, int row2, int size) {

    for (int j = 0; j < size; ++j) {

        double tmp = A[row1][j];
        A[row1][j] = A[row2][j];
        A[row2][j] = tmp;
        
    }

}


void gaussianElimination(double** A, int n, double* result) {

    double* maxResult = nullptr;
    maxResult = new double [2];

    for (int i = 0; i < n; ++i) {
        // Search for maximum in this column
        maximumSearch(A[i], n, i, maxResult);

        // Swap maximum row with current row (column by column)
        swapRows(A, (int) maxResult[1], i, n+1);

        // Make all rows below this one 0 in current column
        for (int k=i+1; k<n; k++) {

            double c = -A[k][i]/A[i][i];

            for (int j=i; j<n+1; j++) {

                if (i==j) {

                    A[k][j] = 0;
                }

                else {

                    A[k][j] += c * A[i][j];

                }
            }
        }
    }

    for (int i=n-1; i>=0; i--) {
        result[i] = A[i][n]/A[i][i];
        for (int k=i-1;k>=0; k--) {
            A[k][n] -= A[k][i] * result[i];
        }
    }

    delete [] maxResult;
}


void pyLeastSquares(PyObject* X, PyObject* y, double* theta, int n, int m) {

    // variable declaration
    double** XTX = nullptr;
    double** A = nullptr;
    double** XT = nullptr;
    double* right = nullptr;

    // memory allocation
    XTX = new double *[m];
    for (int i = 0; i < m; ++i) {
        XTX[i] = new double [m];
    }

    A = new double *[m];
    for (int i = 0; i < m; ++i) {
        A[i] = new double [m+1];
    }

    XT = new double *[m];
    for (int i = 0; i < m; ++i) {
        XT[i] = new double [n];
    }

    right = new double [m];

    // start algorithm

    // first transpose X and get XT
    pyTranspose(X, XT, n, m);

    // XT is a m by n matrix
    // an m by n matrix multiplied by a n by m matrix results in a m by m matrix
    // so XTX is a m by m matrix
    cPyMatrixMatrixProduct(XT, X, m, n, XTX);

    // XT is a m by n matrix
    // y is a n dimensional vector
    // the result is a m dimensional vector called right (since it fits to the right of the A matrix)
    cPyMatrixVectorDotProduct(XT, y, m, n, right);


    // fill in A which is a m by m + 1 matrix
    for (int i = 0; i < m; ++i) {
        // fill in XTX into A
        for (int j = 0; j < m; ++j) {
            A[i][j] = XTX[i][j];
        }
        A[i][m] = right[i];
    }


    // now we can perform gaussian elimination to get an approximation of theta
    gaussianElimination(A, m, theta);

    // and free up memory
    for (int i = 0; i < m; ++i) {
        delete [] A[i];
    }
    delete [] A;

    for (int i = 0; i < m; ++i) {
        delete [] XTX[i];
    }
    delete [] XTX;

    for (int i = 0; i < m; ++i) {
        delete [] XT[i];
    }
    delete [] XT;

    delete [] right;

}