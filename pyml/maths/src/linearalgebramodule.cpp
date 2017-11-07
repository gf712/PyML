#include <Python.h>
#include "linearalgebramodule.h"

// Handle errors
// static PyObject *algebraError;

// Define vector dot product
double vector_dot_product(PyObject* u, PyObject* v, int size) {

    double result = 0;
    int i;

    for (i = 0; i < size; ++i) {

        PyObject *v_item = PyList_GetItem(v, i);
        PyObject *u_item = PyList_GetItem(u, i);

        double pUItem = PyFloat_AsDouble(u_item);
        double pVItem = PyFloat_AsDouble(v_item);

        result += pUItem * pVItem;
    }

    return result;
}

void pypyMatrixVectorDotProduct(PyObject* A, PyObject* v, int ASize, int VSize, double* result) {

    int i;

    for (i = 0; i < ASize ; ++i) {

        PyObject *A_item = PyList_GetItem(A, i);

        result[i] = vector_dot_product(A_item, v, VSize);
    }

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


PyObject *Convert_1DArray(double array[], int length) {

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

    for (int j = 0; j < rows; ++j) {
        row = PyList_New(cols);
        for (int k = 0; k < cols; ++k) {
            item = PyFloat_FromDouble(array[j][k]);
            PyList_SET_ITEM(row, k, item);
        }
        PyList_SET_ITEM(twoDResult, j, row);
    }

    return twoDResult;
}


void pyTranspose(PyObject* X, double** result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        PyObject* row = PyList_GetItem(X, i);
        for (int j = 0; j < cols; ++j) {
            result[j][i] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
}