//
// Created by Gil Ferreira Hoben on 07/11/17.
//
#include <Python.h>
#include <linearalgebramodule.h>
#include "pythonconverters.h"

static PyObject* dot_product(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfA;
    int sizeOfV;

    // pointers to python lists
    PyObject *pAArray;
    PyObject *pVVector;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pAArray, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    sizeOfA = PyList_Size(pAArray);
    sizeOfV = PyList_Size(pVVector);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    if (sizeOfV == 0){
        PyErr_SetString(PyExc_ValueError, "Argument V is empty");
        return nullptr;
    }

    double *result = nullptr;
    result = new double[sizeOfA];

    PyObject *result_py_list;


    pypyMatrixVectorDotProduct(pAArray, pVVector, sizeOfA, sizeOfV, result);

    result_py_list = Convert_1DArray(result, sizeOfA);

    free(result);

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* power(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfA;

    // pointers to python lists
    PyObject *pAArray;
    int pPower;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pAArray, &pPower)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return nullptr;
    }

    // use PyList_Size to get size of vector
    sizeOfA = PyList_Size(pAArray);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }


    if (pPower < 0) {
        PyErr_SetString(PyExc_ValueError, "Power must be greater than 0.");
        return nullptr;
    }

    double *result;
    result = new double[sizeOfA];
    PyObject *result_py_list;

    vector_power(pAArray, pPower, sizeOfA, result);

    result_py_list = Convert_1DArray(result, sizeOfA);

    delete [] result;

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* subtract(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfU;
    int sizeOfV;

    // pointers to python lists
    PyObject *pUVector;
    PyObject *pVVector;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pUVector, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    sizeOfU = PyList_Size(pUVector);
    sizeOfV = PyList_Size(pVVector);

    if (sizeOfU == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    if (sizeOfV == 0){
        PyErr_SetString(PyExc_ValueError, "Argument V is empty");
        return nullptr;
    }

    double* result = nullptr;
    result = new double[sizeOfU];

    PyObject *result_py_list;

    if (sizeOfU != sizeOfV) {
        PyErr_SetString(PyExc_ValueError, "Expected two lists of the same size.");
        return nullptr;
    }

    pyCVectorSubtract(pUVector, pVVector, sizeOfU, result);

    result_py_list = Convert_1DArray(result, sizeOfU);

    delete [] result;

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* sum(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfA;

    // pointers to python lists
    PyObject *pAArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pAArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return NULL;
    }

    // use PyList_Size to get size of vector
    sizeOfA = PyList_Size(pAArray);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    double result;

    result = pyVectorSum(pAArray, sizeOfA);

    PyObject *FinalResult = Py_BuildValue("d", result);

    return FinalResult;
}

static PyObject* transpose(PyObject* self, PyObject *args) {

    // declarations
    double** result;
    int cols, rows;
    PyObject* pyResult;


    // pointers to python lists
    PyObject* pArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list!");
        return nullptr;
    }

    // use PyList_Size to get dimensions of array
    rows = PyList_Size(pArray);
    cols = PyList_Size(PyList_GetItem(pArray, 0));


    if (cols == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    // allocate memory for result
    result = new double *[cols];
    for (int i = 0; i < cols; ++i) {
        result[i] = new double[rows];
    }

    pyTranspose(pArray, result, rows, cols);

    pyResult = Convert_2DArray(result, cols, rows);

    PyObject* FinalResult = Py_BuildValue("O", pyResult);

    for (int i = 0; i < cols; ++i) {
        delete [] result[i];
    }
    delete [] result;

    Py_DECREF(pyResult);

    return FinalResult;
}


static PyObject* matrix_product(PyObject* self, PyObject *args) {

    int rowsA, rowsB, colsA, colsB;

    // pointers to python lists
    PyObject *pAArray;
    PyObject *pBArray;

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
    rowsA = PyList_Size(pAArray);
    colsA = PyList_Size(PyList_GetItem(pAArray, 0));
    rowsB = PyList_Size(pBArray);
    colsB = PyList_Size(PyList_GetItem(pBArray, 0));

    if (rowsA != colsB){
        PyErr_SetString(PyExc_ValueError, "Number of rows in A must be the same as the number of columns in B");
        return nullptr;
    }

    if (colsA != rowsB){
        PyErr_SetString(PyExc_ValueError, "Number of columns in A must be the same as the number of rows in B");
        return nullptr;
    }

    double** result = nullptr;

    result = new double *[rowsA];

    for (int i = 0; i < rowsA; ++i) {
        result[i] = new double [rowsA];
    }

    PyObject *result_py_list;

    pypyMatrixMatrixProduct(pAArray, pBArray, rowsA, colsA, result);

    result_py_list = Convert_2DArray(result, rowsA, colsB);

    for (int i = 0; i < rowsA; ++i) {
        delete [] result[i];
    }
    delete [] result;

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;

}


static PyObject* least_squares(PyObject* self, PyObject *args) {

    int m, n, ySize;

    // pointers to python lists
    PyObject *X;
    PyObject *y;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &X, &PyList_Type, &y)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    if (!PyList_Check(PyList_GetItem(X, 0))) {
        PyErr_SetString(PyExc_TypeError, "Expected A to be a list of lists!");
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    n = PyList_Size(X);
    m = PyList_Size(PyList_GetItem(X, 0));
    ySize = PyList_Size(y);

    // sanity check
    if (n != ySize){
        PyErr_SetString(PyExc_ValueError, "Number of rows of X must be the same as the number of training examples");
        return nullptr;
    }

    double* theta = nullptr;

    theta = new double [m];

    PyObject *result_py_list;

    pyLeastSquares(X, y, theta, n, m);



    result_py_list = Convert_1DArray(theta, m);

    delete [] theta;

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
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
        {"transpose",     transpose,              METH_VARARGS,            "Transpose a 2D matrix"},
        {"least_squares", least_squares,           METH_VARARGS,            "Perform least squares"},
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