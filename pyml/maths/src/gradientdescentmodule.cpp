#include <Python.h>
#include <iostream>


void pyDotProduct(PyObject* X, PyObject* w, double* prediction, int rows, int cols) {

    for (int i = 0; i < rows; ++i) {
        PyObject* row = PyList_GetItem(X, i);
        prediction[i] = 0;
        for (int j = 0; j < cols; ++j) {
            double w_j = PyFloat_AsDouble(PyList_GetItem(w, j));
            prediction[i] += PyFloat_AsDouble(PyList_GetItem(row, j)) * w_j;
        }
    }
}


void predict(PyObject* X, PyObject* w, double* prediction, int rows, int cols) {
    pyDotProduct(X, w, prediction, rows, cols);
}


void dotProduct(double** X, const double * w, double* prediction, int rows, int cols) {

    for (int i = 0; i < rows; ++i) {
        prediction[i] = 0;
        for (int j = 0; j < cols; ++j) {
            prediction[i] += X[i][j] * w[j];
        }
    }
}


void subtract(const double* prediction, PyObject* y, double* loss, int rows) {

    for (int i = 0; i < rows; ++i) {
        loss[i] = prediction[i] - PyFloat_AsDouble(PyList_GetItem(y, i));
    }

}


double sum(const double* array, int rows) {

    double result=0;

    for (int i = 0; i < rows; ++i) {
        result += array[i];
    }

    return result;
}


void power(const double* array, int n, int rows, double* result) {

    for (int i = 0; i < rows; ++i) {
        result[i] = pow(array[i], n);
    }
}

void transpose(PyObject* X, double** result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        PyObject* row = PyList_GetItem(X, i);
        for (int j = 0; j < cols; ++j) {
            result[j][i] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
}


double cost(double* loss, int rows){
    double result[rows];
    power(loss, 2, rows, result);
    return sum(result, rows) / (2 * rows);
}


void divide(double* X, int n, int size) {
    for (int i = 0; i < size; ++i) {
        X[i] /= (double)n;
    }
}


void gradientCalculation(double** X, double* loss, double* gradients, int rows, int cols) {

    dotProduct(X, loss, gradients, rows, cols);
    divide(gradients, cols, rows);

}


void updateWeights(PyObject* theta, const double* gradients, double learningRate, int size) {

    for (int i = 0; i < size; ++i) {
        double theta_i = PyFloat_AsDouble(PyList_GetItem(theta, i));
        theta_i -= gradients[i] * learningRate;
        PyList_SetItem(theta, i, PyFloat_FromDouble(theta_i));
    }
}


int gradientDescent(PyObject* X, PyObject* y, PyObject* theta, int maxIteration, double epsilon, double learningRate, int n, int m, double* costArray) {

    // variable initialisation
    double JOld;
    double JNew;
    double** X_transpose;
    int iteration = 0;
    double e = 1000;
    double* prediction = nullptr;
    double* loss = nullptr;
    double* gradients = nullptr;

    // memory allocation
    X_transpose = new double *[m];
    for (int i = 0; i < m; ++i) {
        X_transpose[i] = new double [n];
    }

    prediction = new double[n];
    loss = new double[n];
    gradients = new double[m];


    // X transpose (m by n matrix)
    // X is a n by m matrix
    transpose(X, X_transpose, n, m);

//    // make sure that X_transpose is correct
//    if (PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(X, 5), 50)) != X_transpose[5][50])
//        PyErr_SetString(PyExc_ValueError, "Error!!!!!!!!");

    // first prediction
    predict(X, theta, prediction, n, m);

    // calculate initial cost and store result
    subtract(prediction, y, loss, n);
    JNew = cost(loss, n);
    costArray[iteration] = JNew;

    // gradient descent
    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // calculate gradient
        gradientCalculation(X_transpose, loss, gradients, m, n);

        // update coefficients
        updateWeights(theta, gradients, learningRate, m);

        // calculate cost for new weights
        predict(X, theta, prediction, n, m);
        subtract(prediction, y, loss, n);
        JNew = cost(loss, n);
        e = JOld - JNew;

        costArray[iteration+1] = JNew;


        iteration += 1;
    }

    // free up memory
    for (int i = 0; i < m; ++i) {
        delete [] X_transpose[i];
    }

    delete [] prediction;
    delete [] X_transpose;
    delete [] loss;
    delete [] gradients;

    return iteration;
}


PyObject* Convert_1DArray(double* array, int length) {

    PyObject* pylist;
    PyObject* item;
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

PyMODINIT_FUNC PyInit_gradientDescentModule(void) {
    return PyModule_Create(&gradientDescentModule);
}
