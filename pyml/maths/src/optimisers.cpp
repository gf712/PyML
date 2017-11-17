//
// Created by Gil Ferreira Hoben on 16/11/17.
//
#include <Python.h>
#include "pythonconverters.h"
#include "optimisers.h"

static PyObject *gradient_descent(PyObject *self, PyObject *args) {

    // variable declaration
    int m, n, maxIterations, iterations;
    double epsilon, learningRate;
    auto costArray = new flatArray;
    auto X = new flatArray;
    auto y = new flatArray;
    auto theta = new flatArray;
    char *predType;

    PyObject* ptheta;
    PyObject* pX;
    PyObject* py;
    PyObject* pyCostArray;
    PyObject* pyTheta;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!O!idds", &PyList_Type, &pX, &PyList_Type, &ptheta, &PyList_Type, &py, &maxIterations, &epsilon, &learningRate, &predType)) {
        PyErr_SetString(PyExc_TypeError, "Check arguments!");
        return nullptr;
    }

    // read python lists
    X->readFromPythonList(pX);
    y->readFromPythonList(py);
    theta->readFromPythonList(ptheta);

    n = X->getRows();
    m = X->getCols();

    if (PyList_Size(ptheta) != m) {
        PyErr_SetString(PyExc_ValueError, "Theta should be the same size as the number of features.");
        return nullptr;
    }

    if (m > n) {
        PyErr_SetString(PyExc_ValueError, "More features than training examples!");
        return nullptr;
    }

    // memory allocation
    costArray->startEmptyArray(1, maxIterations);

    // gradient descent
    iterations = gradientDescent(X, y, theta, maxIterations, epsilon, learningRate, costArray, predType);

    // costArray only needs #iterations columns
    costArray->setCols(iterations);

    // convert cost array and theta to lists
    pyCostArray = ConvertFlatArray_PyList(costArray, "float");
    pyTheta = ConvertFlatArray_PyList(theta, "float");

    PyObject* FinalResult = Py_BuildValue("OOi", pyTheta, pyCostArray, iterations);

    // memory deallocation
    delete costArray;
    delete theta;
    delete y;
    delete X;

    Py_DECREF(pyCostArray);
    Py_DECREF(pyTheta);

    return FinalResult;
}

static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}

static PyMethodDef optimisersMethods[] = {
        // Python name       C function              argument representation  description
        {"gradient_descent", gradient_descent,       METH_VARARGS,            "Gradient Descent"},
        {"version",          (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr,            nullptr,                0,                       nullptr}
};


static struct PyModuleDef optimisersModule = {
        PyModuleDef_HEAD_INIT,
        "gradientDescentModule", // module name
        "Collection of linear algebra functions in C to be used in Python", // documentation of module
        -1, // global state
        optimisersMethods // method defs
};

PyMODINIT_FUNC PyInit_optimisers(void) {
    return PyModule_Create(&optimisersModule);
}
