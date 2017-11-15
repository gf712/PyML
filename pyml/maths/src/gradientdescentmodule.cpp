#include <Python.h>
#include <iostream>
#include "linearalgebramodule.h"
#include "pythonconverters.h"


void sigmoid(flatArray *scores) {
    for (int i = 0; i < scores->getSize(); ++i) {
        scores->setNElement(1 / (1 + exp(-scores->getNElement(i))), i);
    }
}


flatArray *predict(flatArray *X, flatArray *w) {

    return X->dot(w);
}


double logLikelihood(flatArray *scores, flatArray *y) {

    double result = 0;

    for (int i = 0; i < y->getSize(); ++i) {
        result += y->getNElement(i) * scores->getNElement(i) - log(1 + exp(scores->getNElement(i)));
    }

    return result;
}


double cost(flatArray* loss){
    flatArray *result = loss->power(2);
    double costResult = result->sum() / (2 * result->getCols());

    delete result;

    return costResult;
}



flatArray *gradientCalculation(flatArray *X, flatArray *loss) {

    flatArray *gradients = X->dot(loss);

    flatArray* result = gradients->divide(X->getCols());

    delete gradients;

    return result;
}


void updateWeights(flatArray *theta, flatArray *gradients, double learningRate, int size) {

    for (int i = 0; i < size; ++i) {
        theta->setNElement(theta->getNElement(i) - gradients->getNElement(i) * learningRate,i);
    }
}


double calculateCost(flatArray *X, flatArray *theta, flatArray *y, char *predType) {

    double result;
    flatArray *prediction = predict(X, theta);

    if (strcmp(predType, "logit") == 0) {

        result = logLikelihood(prediction, y);

    }

    else {

        // calculate initial cost and store result
        flatArray *loss = prediction->subtract(y);
        result = cost(loss);

        delete loss;

    }

    delete prediction;

    return result;
}


int gradientDescent(flatArray *X, flatArray *y, flatArray *theta, int maxIteration, double epsilon, double learningRate, flatArray* costArray, char *predType) {

    // variable declaration
    double JOld;
    double JNew;
    int iteration = 0;
    double e = 1000;
    int m = X->getCols();
    flatArray *XT;


    // X pyTranspose (m by n matrix)
    // X is a n by m matrix
    XT = X->transpose();

    JNew = calculateCost(X, theta, y, predType);
    costArray->setNElement(JNew, iteration);

    // gradient descent
    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // calculate gradient
        flatArray *h = predict(X, theta);

        if (strcmp(predType, "logit") == 0) {
            sigmoid(h);
        }

        flatArray *error = h->subtract(y);

        flatArray *gradients = gradientCalculation(XT, error);

        // update coefficients
        updateWeights(theta, gradients, learningRate, m);

        // calculate cost for new weights
        JNew = calculateCost(X, theta, y, predType);

        e = fabs(JOld) - fabs(JNew);

        costArray->setNElement(JNew, iteration+1);

        iteration++;

        delete h;
        delete error;
        delete gradients;
    }

    // free up memory
    delete XT;

    // return number of iterations needed to reach convergence
    return iteration;
}


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

static PyMethodDef gradientDescentMethods[] = {
        // Python name       C function              argument representation  description
        {"gradient_descent", gradient_descent,       METH_VARARGS,            "Gradient Descent"},
        {"version",          (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr,            nullptr,                0,                       nullptr}
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
