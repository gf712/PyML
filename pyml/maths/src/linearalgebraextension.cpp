//
// Created by Gil Ferreira Hoben on 07/11/17.
//
// This file provides the Python API for
// the linear algebra code written in C++


#include <Python.h>
#include <linearalgebramodule.h>
#include "linearalgebramodule.cpp"
#include "../../utils/include/exceptionClasses.h"
#include "arrayInitialisers.cpp"


static PyObject* dot_product(PyObject* self, PyObject *args) {

    flatArray<double>* A = nullptr;
    flatArray<double>* V = nullptr;
    flatArray<double>* result = nullptr;

    // pointers to python lists
    PyObject* pAArray;
    PyObject* pVVector;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pAArray, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    A = readFromPythonList<double>(pAArray);
    V = readFromPythonList<double>(pVVector);

    // calculate dot product
    try {
        result = A->dot(V);
    }
    catch (flatArrayDimensionMismatchException<double> &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    catch (flatArrayColumnMismatchException<double> &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }

    // convert result to python list
    PyObject* result_py_list = ConvertFlatArray_PyList(result, "float");

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
    flatArray<double>* A = nullptr;
    flatArray<double>* result = nullptr;
    int p;

    // pointers to python lists
    PyObject * pAArray;


    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pAArray, &p)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return nullptr;
    }

    // read in python list
    A = readFromPythonList<double>(pAArray);

    // calculate the power elementwise
    flatArray *result = A->power(p);

    // convert vector to python list
    PyObject* result_py_list = ConvertFlatArray_PyList(result, "float");

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // deallocate memory
    delete A;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* subtract(PyObject* self, PyObject *args) {

    // variable instantiation
    flatArray<double>* A = nullptr;
    flatArray<double>* B = nullptr;
    flatArray<double>* result = nullptr;

    PyObject *pA;
    PyObject *pB;
    PyObject *result_py_list;
    PyObject *FinalResult;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pA, &PyList_Type, &pB)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }

    // get python lists
    A = readFromPythonList<double>(pA);
    B = readFromPythonList<double>(pB);

    // subtraction
    try {
        result = A->subtract(B);
    }

    catch (flatArrayDimensionMismatchException<double> &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    catch (flatArrayColumnMismatchException<double> &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    catch (flatArrayRowMismatchException<double> &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    result_py_list = ConvertFlatArray_PyList(result, "float");
    FinalResult = Py_BuildValue("O", result_py_list);

//    FinalResult = Py_BuildValue("i", 0);

    // memory deallocation
    delete result;
    delete A;
    delete B;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* divide(PyObject* self, PyObject *args) {

    // variable declaration
    flatArray<double>* A = nullptr;
    flatArray<double>* result = nullptr;
    double n;

    // pointers to python lists
    PyObject * pAArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!d", &PyList_Type, &pAArray, &n)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and a float!");
        return nullptr;
    }

    // read in python list
    A = readFromPythonList<double>(pAArray);

    // calculate elementwise division by n
    try {
        result = A->divide(n);
    }
    catch (flatArrayZeroDivisionError &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    // convert vector to python list
    PyObject* result_py_list = ConvertFlatArray_PyList(result, "float");

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // deallocate memory
    delete A;
    delete result;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* sum(PyObject* self, PyObject *args) {

    // sum of all elements in a matrix/vector

    // variable declaration
    flatArray<double>* A = nullptr;
    PyObject *pAArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pAArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return nullptr;
    }

    // use PyList_Size to get size of vector
    A = readFromPythonList<double>(pAArray);

    double result = A->sum();

    PyObject *FinalResult = Py_BuildValue("d", result);

    return FinalResult;
}


static PyObject* pyTranspose(PyObject* self, PyObject *args) {

    // declarations
    flatArray<double>* A = nullptr;
    flatArray<double>* result = nullptr;

    PyObject* pyResult = nullptr;
    PyObject* pArray = nullptr;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list!");
        return nullptr;
    }

    A = readFromPythonList<double>(pArray);

    result = A->transpose();

    pyResult = ConvertFlatArray_PyList(result, "float");

    PyObject* FinalResult = Py_BuildValue("O", pyResult);

    delete result;
    delete A;

    Py_DECREF(pyResult);

    return FinalResult;
}


static PyObject* least_squares(PyObject* self, PyObject *args) {

    // variable declaration
    flatArray<double>* X = nullptr;
    flatArray<double>* y = nullptr;

    PyObject *pX;
    PyObject *py;
    PyObject *result_py_list;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pX, &PyList_Type, &py)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return nullptr;
    }


    // read in python lists
    X = readFromPythonList<double>(pX);
    y =readFromPythonList<double>(py);


    // sanity check
    if (X->getRows() != y->getCols()){
        PyErr_SetString(PyExc_ValueError, "Number of rows of X must be the same as the number of training examples");
        return nullptr;
    }

    // memory allocation of theta
    auto theta = new double [X->getCols()];

    // get theta estimate using least squares
    leastSquares<double>(X, y, theta);
    

    result_py_list = Convert_1DArray(theta, X->getCols());

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // memory deallocation
    delete theta;
    delete X;
    delete y;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* mean(PyObject* self, PyObject *args) {

    // variable declaration
    int axis;

    flatArray<double>* X = nullptr;
    flatArray<double>* result = nullptr;

    PyObject *pX = nullptr;
    PyObject *FinalResult = nullptr;


    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pX, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and one integer!");
        return nullptr;
    }

    X = readFromPythonList<double>(pX);

    try {
        result = X->mean(axis);
    }
    catch (flatArrayUnknownAxis &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    if (X->getRows() == 1) {

        FinalResult = Py_BuildValue("d", result->getNElement(0));

    }

    else {

        PyObject *result_py_list = ConvertFlatArray_PyList(result, "float");

        FinalResult = Py_BuildValue("O", result_py_list);

        Py_DECREF(result_py_list);
    }

    delete X;
    delete result;

    return FinalResult;
}


static PyObject* standardDeviation(PyObject* self, PyObject *args) {

    // variable declaration
    int axis, degreesOfFreedom;
    flatArray<double>* X = nullptr;
    flatArray<double>* result = nullptr;

    PyObject *FinalResult = nullptr;
    PyObject *pX = nullptr;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &pX, &degreesOfFreedom, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and one integer!");
        return nullptr;
    }

    X = readFromPythonList<double>(pX);

    result = X->std(degreesOfFreedom, axis);

    if (X->getRows() == 1) {

        FinalResult = Py_BuildValue("d", result->getNElement(0));

    }

    else {

        PyObject *result_py_list = ConvertFlatArray_PyList(result, "float");

        FinalResult = Py_BuildValue("O", result_py_list);

        Py_DECREF(result_py_list);
    }

    delete X;
    delete result;

    return FinalResult;
}


static PyObject* variance(PyObject* self, PyObject *args) {

    // variable declaration
    int axis, degreesOfFreedom;
    flatArray<double>* X = nullptr;
    flatArray<double>* result = nullptr;

    PyObject *pX = nullptr;
    PyObject *FinalResult = nullptr;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &pX, &degreesOfFreedom, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and two integers!");
        return nullptr;
    }

    X = readFromPythonList<double>(pX);

    result = X->var(degreesOfFreedom, axis);

    if (X->getRows() == 1) {

        FinalResult = Py_BuildValue("d", result->getNElement(0));

    }

    else {

        PyObject *result_py_list = ConvertFlatArray_PyList(result, "float");

        FinalResult = Py_BuildValue("O", result_py_list);

        Py_DECREF(result_py_list);
    }

    delete X;
    delete result;

    return FinalResult;
}


static PyObject* cov(PyObject* self, PyObject *args) {

    // variable declaration
    flatArray<double>* X = nullptr;
    flatArray<double>* result = nullptr;

    PyObject *pX = nullptr;
    PyObject *FinalResult = nullptr;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &pX)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list!");
        return nullptr;
    }

    X = readFromPythonList<double>(pX);

    result = covariance<double>(X);

    PyObject *result_py_list = ConvertFlatArray_PyList(result, "float");

    FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    delete X;
    delete result;

    return FinalResult;
}


static PyObject* eigenSolve(PyObject* self, PyObject *args) {

    // variable declaration
    int maxIterations;
    double tolerance;

    flatArray<double>* X = nullptr;
    flatArray<double>* eigFArray = nullptr;
    flatArray<double>* result = nullptr;

    PyObject *pX = nullptr;
    PyObject *FinalResult = nullptr;
    PyObject *eigV = nullptr;
    PyObject *eigE = nullptr;


    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!di", &PyList_Type, &pX, &tolerance, &maxIterations)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list, a float and an integer!");
        return nullptr;
    }

    X = readFromPythonList<double>(pX);

    result = jacobiEigenDecomposition<double>(X, tolerance, maxIterations);

    eigV = Convert_1DArray(result->getRow(0), result->getCols());

    eigFArray = emptyArray<double>(result->getCols(), result->getCols());

    for (int i = 1; i < result->getRows(); ++i) {
        double *rowResult = result->getRow(i);
        eigFArray->setRow(rowResult, i - 1);
        delete [] rowResult;
    }

    eigE = ConvertFlatArray_PyList(eigFArray, "float");

    FinalResult = Py_BuildValue("OO", eigV, eigE);

    Py_DECREF(eigE);
    Py_DECREF(eigV);

    delete X;
    delete result;
    delete eigFArray;

    return FinalResult;
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.3");
}


static PyMethodDef linearAlgebraMethods[] = {
        // Python name    C function              argument representation  description
        {"dot_product",   dot_product,            METH_VARARGS,            "Calculate the dot product of two vectors"},
        {"power",         power,                  METH_VARARGS,            "Calculate element wise power"},
        {"subtract",      subtract,               METH_VARARGS,            "Calculate element wise subtraction"},
        {"divide",        divide,                 METH_VARARGS,            "Calculate element wise division"},
        {"sum",           sum,                    METH_VARARGS,            "Calculate the total sum of a vector"},
        {"transpose",     pyTranspose,            METH_VARARGS,            "Transpose a 2D matrix"},
        {"least_squares", least_squares,          METH_VARARGS,            "Perform least squares"},
        {"Cmean",         mean,                   METH_VARARGS,            "Numpy style array mean"},
        {"Cstd",          standardDeviation,      METH_VARARGS,            "Numpy style array standard deviation"},
        {"Cvariance",     variance,               METH_VARARGS,            "Numpy style array variance"},
        {"Ccovariance",   cov,                    METH_VARARGS,            "Calculate covariance matrix"},
        {"eigen_solve",   eigenSolve,             METH_VARARGS,            "Eigendecomposition of symmetric matrix"},
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
