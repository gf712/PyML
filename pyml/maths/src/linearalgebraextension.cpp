//
// Created by Gil Ferreira Hoben on 07/11/17.
//
#include <Python.h>
#include <linearalgebramodule.h>
#include "pythonconverters.h"
#include <iostream>

static PyObject* dot_product(PyObject* self, PyObject *args) {

    // variable instantiation
    // A is a list of lists (matrix)
    // V is a list (vector)
    auto A = new flatArray;
    auto V = new flatArray;
    auto result = new flatArray;

    // pointers to python lists
    PyObject * pAArray;
    PyObject * pVVector;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pAArray, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    A->readFromPythonList(pAArray);
    V->readFromPythonList(pVVector);
    result->startEmptyArray(1, A->getRows());

    if (A->getCols() != V->getCols()){
        PyErr_SetString(PyExc_ValueError, "A and v must have the same size");
        return nullptr;
    }


    // calculate dot product
    flatMatrixVectorDotProduct(A, V, result);

    // convert result to python list
    PyObject* result_py_list = ConvertFlat2DArray_2DPy(result);

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    delete result;
    delete A;
    delete V;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* power(PyObject* self, PyObject *args) {

    // variable declaration
    auto A = new flatArray;
    int p;

    // pointers to python lists
    PyObject * pAArray;


    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pAArray, &p)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return nullptr;
    }

    // read in python list
    A->readFromPythonList(pAArray);

    // calculate the power elementwise
    flatMatrixPower(A, p);

    // convert vector to python list
    PyObject* result_py_list = ConvertFlat2DArray_2DPy(A);

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // deallocate memory
    delete A;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* subtract(PyObject* self, PyObject *args) {

    // variable instantiation
    auto A = new flatArray;
    auto B = new flatArray;
    auto result = new flatArray;

    PyObject *pA;
    PyObject *pB;
    PyObject *result_py_list;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pA, &PyList_Type, &pB)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    // get python lists
    A->readFromPythonList(pA);
    B->readFromPythonList(pB);

    // memory allocation
    result->startEmptyArray(A->getRows(), A->getCols());

    // calculation
    flatArraySubtract(A, B, result);

    result_py_list = ConvertFlat2DArray_2DPy(result);

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // memory deallocation
    delete result;
    delete A;
    delete B;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* sum(PyObject* self, PyObject *args) {

    // sum of all elements in a matrix/vector

    // variable declaration
    auto A = new flatArray;
    PyObject *pAArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pAArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return nullptr;
    }

    // use PyList_Size to get size of vector
    A->readFromPythonList(pAArray);

    double result = A->sum();

    PyObject *FinalResult = Py_BuildValue("d", result);

    return FinalResult;
}

static PyObject* pyTranspose(PyObject* self, PyObject *args) {

    // declarations
//    auto result = new flatArray;
    auto A = new flatArray;
//    int block_size;
    PyObject* pyResult;
    PyObject* pArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list!");
        return nullptr;
    }

//    if (block_size <= 0) {
//        block_size = 1;
//    }

    A->readFromPythonList(pArray);

    flatArray *result = A->transpose();

    pyResult = ConvertFlat2DArray_2DPy(result);

    PyObject* FinalResult = Py_BuildValue("O", pyResult);

//    PyObject* FinalResult = Py_BuildValue("i", 0);

    delete result;
    delete A;

    Py_DECREF(pyResult);

    return FinalResult;
}


static PyObject* matrix_product(PyObject* self, PyObject *args) {

    // variable declaration
    int rowsA, rowsB, colsA, colsB;
    PyObject *pAArray;
    PyObject *pBArray;
    double** result = nullptr;
    double** A = nullptr;
    double** B = nullptr;
    PyObject *result_py_list;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pAArray, &PyList_Type, &pBArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    if (!PyList_Check(PyList_GetItem(pAArray, 0))) {
        PyErr_SetString(PyExc_TypeError, "Expected A to be a list of lists!");
        return nullptr;
    }

    if (!PyList_Check(PyList_GetItem(pBArray, 0))) {
        PyErr_SetString(PyExc_TypeError, "Expected B to be a list of lists!");
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    rowsA = static_cast<int>(PyList_Size(pAArray));
    colsA = static_cast<int>(PyList_Size(PyList_GetItem(pAArray, 0)));
    rowsB = static_cast<int>(PyList_Size(pBArray));
    colsB = static_cast<int>(PyList_Size(PyList_GetItem(pBArray, 0)));

    if (rowsA != colsB){
        PyErr_SetString(PyExc_ValueError, "Number of rows in A must be the same as the number of columns in B");
        return nullptr;
    }

    if (colsA != rowsB){
        PyErr_SetString(PyExc_ValueError, "Number of columns in A must be the same as the number of rows in B");
        return nullptr;
    }

    // memory allocation
    result = new double *[rowsA];

    for (int i = 0; i < rowsA; ++i) {
        result[i] = new double [rowsA];
    }
    A = new double *[rowsA];

    for (int i = 0; i < rowsA; ++i) {
        A[i] = new double [colsA];
    }
    B = new double *[rowsB];

    for (int i = 0; i < rowsB; ++i) {
        B[i] = new double [colsB];
    }

    // convert python lists of lists to 2D C++ arrays
    convertPy_2DArray(pAArray, A, rowsA, colsA);
    convertPy_2DArray(pBArray, B, rowsB, colsB);

    matrixMatrixProduct(A, B, rowsA, colsA, result);

    result_py_list = Convert_2DArray(result, rowsA, rowsA);

    // memory deallocation
    for (int i = 0; i < rowsA; ++i) {
        delete [] result[i];
    }
    delete [] result;

    for (int i = 0; i < rowsA; ++i) {
        delete [] A[i];
    }
    delete [] A;

    for (int i = 0; i < rowsB; ++i) {
        delete [] B[i];
    }
    delete [] B;

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* least_squares(PyObject* self, PyObject *args) {

    // variable declaration
    int m, n, ySize;
    PyObject *pX;
    PyObject *py;
    double** X = nullptr;
    double* y = nullptr;
    double* theta = nullptr;
    PyObject *result_py_list;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pX, &PyList_Type, &py)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    if (!PyList_Check(PyList_GetItem(pX, 0))) {
        PyErr_SetString(PyExc_TypeError, "Expected A to be a list of lists!");
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    n = static_cast<int>(PyList_Size(pX));
    m = static_cast<int>(PyList_Size(PyList_GetItem(pX, 0)));
    ySize = static_cast<int>(PyList_Size(py));

    // sanity check
    if (n != ySize){
        PyErr_SetString(PyExc_ValueError, "Number of rows of X must be the same as the number of training examples");
        return nullptr;
    }

    // memory allocation
    theta = new double [m];

    X = new double *[n];
    for (int j = 0; j < n; ++j) {
        X[j] = new double [m];
    }
    y = new double [n];

    convertPy_2DArray(pX, X, n, m);
    convertPy_1DArray(py, y, n);

    leastSquares(X, y, theta, n, m);

    result_py_list = Convert_1DArray(theta, m);

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // memory deallocation
    delete [] theta;
    for (int j = 0; j < n; ++j) {
        delete [] X[j];
    }
    delete [] X;
    delete [] y;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* mean(PyObject* self, PyObject *args) {

    // variable declaration
    int axis;
    PyObject *pX;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pX, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and one integer!");
        return nullptr;
    }

    if (PyFloat_Check(PyList_GetItem(pX, 0))) {
        // if the first element is a float we assume this is a 1D array/vector

        // variable declaration
        double result = 0;
        double* X = nullptr;
        int cols;

        // get size of array
        cols = static_cast<int> PyList_GET_SIZE(pX);

        // memory allocation
        X = new double[cols];

        convertPy_1DArray(pX, X, cols);

        result = vectorMean(X, cols);

        PyObject *FinalResult = Py_BuildValue("d", result);

        // memory deallocation
        delete [] X;

        return FinalResult;
    }
    else if (PyList_Check(PyList_GET_ITEM(pX, 0))){
        // if it is a list we assume this is a 2D array/matrix

        // variable declaration
        int rows, cols;
        double* result = nullptr;
        double **X = nullptr;
        PyObject *result_py_list;

        // get size of array
        rows = static_cast<int> PyList_GET_SIZE(pX);
        cols = static_cast<int> PyList_GET_SIZE(PyList_GET_ITEM(pX, 0));

        // memory allocation
        X = new double *[rows];
        for (int i = 0; i < rows; ++i) {
            X[i] = new double [cols];
        }

        if (axis == 0) {
            result = new double[cols];
        }
        else {
            result = new double[rows];
        }

        convertPy_2DArray(pX, X, rows, cols);

        matrixMean(X, cols, rows, axis, result);

        if (axis == 0) {
            result_py_list = Convert_1DArray(result, cols);
        }
        else {
            result_py_list = Convert_1DArray(result, rows);
        }

        PyObject *FinalResult = Py_BuildValue("O", result_py_list);

        // memory deallocation
        for (int j = 0; j < rows; ++j) {
            delete [] X[j];
        }
        delete [] X;

        delete [] result;

        Py_DECREF(result_py_list);

        return FinalResult;

    }
    else {
        PyErr_SetString(PyExc_TypeError, "Can only handle 2D float arrays at the moment!");
        return nullptr;
    }
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.3");
}


static PyMethodDef linearAlgebraMethods[] = {
        // Python name    C function              argument representation  description
        {"dot_product",   dot_product,            METH_VARARGS,            "Calculate the dot product of two vectors"},
        {"matrix_product",matrix_product,         METH_VARARGS,            "Calculate the product of two matrices"},
        {"power",         power,                  METH_VARARGS,            "Calculate element wise power"},
        {"subtract",      subtract,               METH_VARARGS,            "Calculate element wise subtraction"},
        {"sum",           sum,                    METH_VARARGS,            "Calculate the total sum of a vector"},
        {"transpose",     pyTranspose,            METH_VARARGS,            "Transpose a 2D matrix"},
        {"least_squares", least_squares,          METH_VARARGS,            "Perform least squares"},
        {"Cmean",         mean,                   METH_VARARGS,            "Numpy style array mean"},
        {"version",       (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef linearAlgebraModule = {
        PyModuleDef_HEAD_INIT,
        "linearAlgebra", // module name
        "Collection of linear algebra functions in C to be used in Python", // documentation of module
        -1, // global state
        linearAlgebraMethods // method defs
};


PyMODINIT_FUNC PyInit_Clinear_algebra(void) {
    return PyModule_Create(&linearAlgebraModule);
}