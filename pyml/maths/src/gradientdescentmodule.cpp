#include <Python.h>
#include <iostream>
#include "linearalgebramodule.h"
#include "pythonconverters.h"


void predict(double** X, double* w, double* prediction, int rows, int cols) {
    matrixVectorDotProduct(X, w, rows, cols, prediction);
}


void power(const double* array, int n, int rows, double* result) {

    for (int i = 0; i < rows; ++i) {
        result[i] = pow(array[i], n);
    }
}


double cost(double* loss, int rows){
    double* result;
    result = new double[rows];
    power(loss, 2, rows, result);
    return vectorSum(result, rows) / (2 * rows);
}



void gradientCalculation(double** X, double* loss, double* gradients, int rows, int cols) {

    matrixVectorDotProduct(X, loss, rows, cols, gradients);
    vectorDivide(gradients, cols, rows);

}


void updateWeights(double* theta, const double* gradients, double learningRate, int size) {

    for (int i = 0; i < size; ++i) {
        theta[i] -= gradients[i] * learningRate;
    }
}


double calculateCost(double** X, double* theta, double* prediction, double* y, double* loss, int n, int m) {
    // first prediction
    predict(X, theta, prediction, n, m);

    // calculate initial cost and store result
    vectorSubtract(prediction, y, n, loss);
    return cost(loss, n);
}


int gradientDescent(double** X, double* y, double* theta, int maxIteration, double epsilon, double learningRate, int n, int m, double* costArray) {

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
    matrixTranspose(X, X_pyTranspose, n, m, 16);

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
    PyObject* ptheta;
    PyObject* pX;
    PyObject* py;
    int maxIterations;
    double epsilon;
    double learningRate;
    int iterations;
    double* costArray = nullptr;
    double** X = nullptr;
    double* y = nullptr;
    double* theta = nullptr;
    PyObject* pyCostArray;
    PyObject* pyTheta;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!O!idd", &PyList_Type, &pX, &PyList_Type, &ptheta, &PyList_Type, &py, &maxIterations, &epsilon, &learningRate)) {
        PyErr_SetString(PyExc_TypeError, "Check arguments!");
        return nullptr;
    }

    n = (int) PyList_Size(pX);
    m = (int) PyList_Size(PyList_GetItem(pX, 0));

    if (PyList_Size(ptheta) != m) {
        PyErr_SetString(PyExc_ValueError, "Theta should be the same size as the number of features.");
        return nullptr;
    }

    if (m > n) {
        PyErr_SetString(PyExc_ValueError, "More features than training examples!");
        return nullptr;
    }

    // memory allocation
    X = new double *[n];
    for (int j = 0; j < n; ++j) {
        X[j] = new double [m];
    }
    y = new double [n];
    theta = new double [m];

    costArray = new double[maxIterations];

    // convert python lists to C++ arrays
    convertPy_1DArray(py, y, n);
    convertPy_1DArray(ptheta, theta, m);
    convertPy_2DArray(pX, X, n, m);


    // gradient descent
    iterations = gradientDescent(X, y, theta, maxIterations, epsilon, learningRate, n, m, costArray);

    // convert cost array and theta to lists
    pyCostArray = Convert_1DArray(costArray, iterations);
    pyTheta = Convert_1DArray(theta, m);

    PyObject* FinalResult = Py_BuildValue("OOi", pyTheta, pyCostArray, iterations);

    // memory deallocation
    delete [] costArray;
    delete [] theta;
    delete [] y;
    for (int i = 0; i < n; ++i) {
        delete [] X[i];
    }
    delete [] X;

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
