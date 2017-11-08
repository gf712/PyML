#include <Python.h>
#include <iostream>
#include "linearalgebramodule.h"
#include "pythonconverters.h"


void predict(PyObject* X, PyObject* w, double* prediction, int rows, int cols) {
    pypyMatrixVectorDotProduct(X, w, rows, cols, prediction);
}


void power(const double* array, int n, int rows, double* result) {

    for (int i = 0; i < rows; ++i) {
        result[i] = pow(array[i], n);
    }
}


double cost(double* loss, int rows){
    double result[rows];
    power(loss, 2, rows, result);
    return cVectorSum(result, rows) / (2 * rows);
}



void gradientCalculation(double** X, double* loss, double* gradients, int rows, int cols) {

    ccMatrixVectorDotProduct(X, loss, gradients, rows, cols);
    cVectorDivide(gradients, cols, rows);

}


void updateWeights(PyObject* theta, const double* gradients, double learningRate, int size) {

    for (int i = 0; i < size; ++i) {
        double theta_i = PyFloat_AsDouble(PyList_GetItem(theta, i));
        theta_i -= gradients[i] * learningRate;
        PyList_SetItem(theta, i, PyFloat_FromDouble(theta_i));
    }
}


double calculateCost(PyObject* X, PyObject* theta, double* prediction, PyObject* y, double* loss, int n, int m) {
    // first prediction
    predict(X, theta, prediction, n, m);

    // calculate initial cost and store result
    cPyVectorSubtract(prediction, y, loss, n);
    return cost(loss, n);
}


int gradientDescent(PyObject* X, PyObject* y, PyObject* theta, int maxIteration, double epsilon, double learningRate, int n, int m, double* costArray) {

    // variable initialisation
    double JOld;
    double JNew;
    double** X_pyTranspose;
    int iteration = 0;
    double e = 1000;
    double* prediction = nullptr;
    double* loss = nullptr;
    double* gradients = nullptr;

    // memory allocation
    X_pyTranspose = new double *[m];
    for (int i = 0; i < m; ++i) {
        X_pyTranspose[i] = new double [n];
    }

    prediction = new double[n];
    loss = new double[n];
    gradients = new double[m];

    // X pyTranspose (m by n matrix)
    // X is a n by m matrix
    pyTranspose(X, X_pyTranspose, n, m);

    JNew = calculateCost(X, theta, prediction, y, loss, n, m);
    costArray[iteration] = JNew;

    // gradient descent
    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // calculate gradient
        gradientCalculation(X_pyTranspose, loss, gradients, m, n);

        // update coefficients
        updateWeights(theta, gradients, learningRate, m);

        // calculate cost for new weights
        JNew = calculateCost(X, theta, prediction, y, loss, n, m);
        e = JOld - JNew;
        costArray[iteration+1] = JNew;

        iteration += 1;
    }

    // free up memory
    for (int i = 0; i < m; ++i) {
        delete [] X_pyTranspose[i];
    }

    delete [] prediction;
    delete [] X_pyTranspose;
    delete [] loss;
    delete [] gradients;


    // return number of iterations needed to reach convergence
    return iteration;
}


static PyObject *gradient_descent(PyObject *self, PyObject *args) {

    // variable declaration
    int m;
    int n;
    PyObject* theta;
    PyObject* X;
    PyObject* y;
    int maxIterations;
    double epsilon;
    double learningRate;
    int iterations;
    double* costArray = nullptr;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!O!idd", &PyList_Type, &X, &PyList_Type, &theta, &PyList_Type, &y, &maxIterations, &epsilon, &learningRate)) {
        PyErr_SetString(PyExc_TypeError, "Check arguments!");
        return nullptr;
    }

    n = (int) PyList_Size(X);
    m = (int) PyList_Size(PyList_GetItem(X, 0));

    if (PyList_Size(theta) != m) {
        PyErr_SetString(PyExc_ValueError, "Theta should be the same size as the number of features.");
        return nullptr;
    }

    if (m > n) {
        PyErr_SetString(PyExc_ValueError, "More features than training examples!");
        return nullptr;
    }

    costArray = new double[maxIterations];
    PyObject* pyCostArray;

    iterations = gradientDescent(X, y, theta, maxIterations, epsilon, learningRate, n, m, costArray);

    pyCostArray = Convert_1DArray(costArray, iterations);

    delete [] costArray;

    PyObject* FinalResult = Py_BuildValue("OOi", theta, pyCostArray, iterations);

    Py_DECREF(pyCostArray);

    return FinalResult;
}

static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}

static PyMethodDef gradientDescentMethods[] = {
        // Python name       C function              argument representation  description
        {"gradient_descent", gradient_descent,       METH_VARARGS,            "Gradient Descent"},
        {"version",          (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef gradientDescentModule = {
        PyModuleDef_HEAD_INIT,
        "gradientDescentModule", // module name
        "Collection of linear algebra functions in C to be used in Python", // documentation of module
        -1, // global state
        gradientDescentMethods // method defs
};

PyMODINIT_FUNC PyInit_gradient_descent(void) {
    return PyModule_Create(&gradientDescentModule);
}
