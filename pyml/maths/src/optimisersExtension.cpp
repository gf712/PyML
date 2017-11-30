//
// Created by Gil Ferreira Hoben on 16/11/17.
//
#include <Python.h>
#include "pythonconverters.h"
#include "optimisersExtension.h"
#include "arrayInitialisers.cpp"
#include "optimisers.cpp"


static PyObject *GD(PyObject *self, PyObject *args) {

    // variable declaration
    int m, n, maxIterations, iterations, batchSize, seed;
    double epsilon, learningRate, alpha, fudge_factor;
    flatArray<double>* costArray = nullptr;
    flatArray<double>* X = nullptr;
    flatArray<double>* y = nullptr;
    flatArray<double>* theta = nullptr;
    char* predType;
    char* method;

    PyObject* ptheta;
    PyObject* pX;
    PyObject* py;
    PyObject* pyCostArray;
    PyObject* pyTheta;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!O!iidddssid", &PyList_Type, &pX, &PyList_Type, &ptheta, &PyList_Type, &py,
                         &batchSize, &maxIterations, &epsilon, &learningRate, &alpha, &predType, &method, &seed,
                         &fudge_factor)) {
        PyErr_SetString(PyExc_TypeError, "Check arguments!");
        return nullptr;
    }

    // read python lists
    X = readFromPythonList<double>(pX);
    y = readFromPythonList<double>(py);
    theta = readFromPythonList<double>(ptheta);

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


    // memory allocation of costArray
    if (batchSize > 0 && batchSize < n) {
        auto batchIterations = static_cast<int>(std::floor(n / batchSize));

        if (n % batchSize == 0) {
            costArray = emptyArray<double>(1, maxIterations * batchIterations);
        }

        else {
            costArray = emptyArray<double>(1, maxIterations * (batchIterations + 1));
        }
    }

    else {
        costArray = emptyArray<double>(1, maxIterations);
    }

    // gradient descent
    iterations = gradientDescent<double>(X, y, theta, maxIterations, epsilon, learningRate, alpha, costArray, predType,
                                         batchSize, seed, method, fudge_factor);

    // costArray only needs #iterations columns
    if (batchSize > 0 && batchSize < n) {
        auto batchIterations = static_cast<int>(std::floor(n / batchSize));

        if (n % batchSize == 0) {
            costArray->setCols(iterations * batchIterations);
        }

        else {
            costArray->setCols(iterations * (batchIterations + 1));
        }
    }

    else {
        costArray->setCols(iterations + 1);
    }


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
    return Py_BuildValue("s", "Version 0.2.1");
}

static PyMethodDef optimisersMethods[] = {
        // Python name       C function              argument representation  description
        {"gradient_descent", GD,       METH_VARARGS,            "Gradient Descent"},
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

    PyObject *m;

    m = PyModule_Create(&optimisersModule);

    if (m == nullptr)
        return nullptr;

    return m;
}
