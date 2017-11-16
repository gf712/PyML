//
// Created by gil on 09/11/17.
//

#include <Python.h>
#include "pythonconverters.h"
#include "distances.h"

static PyObject* norm(PyObject* self, PyObject *args) {

    // variable instantiation
    // A is a list of lists (matrix)
    // u is a list (vector)
    int colsA, rowsA, colsB, rowsB;
    int p;

    // pointers to python lists
    PyObject* pA;
    PyObject* pB;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!O!i", &PyList_Type, &pA, &PyList_Type, &pB, &p)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists and one integer!");
        return nullptr;
    }

    if (PyList_Check(PyList_GET_ITEM(pA, 0))) {
        rowsA = static_cast<int>(PyList_GET_SIZE(pA));
        colsA = static_cast<int>(PyList_GET_SIZE(PyList_GET_ITEM(pA, 0)));
    }
    else {
        rowsA = 0;
        colsA = static_cast<int>(PyList_GET_SIZE(pA));
    }


    if (PyList_Check(PyList_GET_ITEM(pB, 0))) {
        rowsB = static_cast<int>(PyList_GET_SIZE(pB));
        colsB = static_cast<int>(PyList_GET_SIZE(PyList_GET_ITEM(pB, 0)));
    }

    else {
        rowsB = 0;
        colsB = static_cast<int>(PyList_GET_SIZE(pB));
    }

    if (p == 0) {
        PyErr_SetString(PyExc_TypeError, "P cannot be 0!");
        return nullptr;
    }

    if (rowsA == 0){
        if (rowsB == 0) {
            // int this case it's the norm of two vectors
            // variable declaration
            double* A = nullptr;
            double* B = nullptr;
            double result;
            // memory allocation
            A = new double [colsA];
            B = new double [colsB];

            convertPy_1DArray(pA, A, colsA);
            convertPy_1DArray(pB, B, colsB);

            result = vectorVectorNorm(A, B, p, colsA);

            PyObject *FinalResult = Py_BuildValue("d", result);

            // memory deallocation
            delete [] A;
            delete [] B;

            return FinalResult;
        }
        else {
            // if B is not a 1D array cannot perform norm calculation
            PyErr_SetString(PyExc_ValueError, "If A is a vector, B must be a vector too.");
            return nullptr;
        }
    }
    // if A is a matrix
    else if (rowsA > 0) {
        // if B is a vector
        if (rowsB == 0 && colsA == colsB) {
            double** A = nullptr;
            double* B = nullptr;
            double* result = nullptr;
            PyObject* pylistResult = nullptr;
            PyObject* FinalResult = nullptr;

            // memory allocation
            A = new double *[rowsA];
            for (int i = 0; i < rowsA; ++i) {
                A[i] = new double [colsA];
            }
            B = new double [colsB];
            result = new double [rowsA];

            // convert python to C++
            convertPy_2DArray(pA, A, rowsA, colsA);
            convertPy_1DArray(pB, B, colsB);

            matrixVectorNorm(A, B, p, rowsA, colsA, result);

            pylistResult = Convert_1DArray(result, rowsA);

            FinalResult = Py_BuildValue("O", pylistResult);

            // memory deallocation
            for (int i = 0; i < rowsA; ++i) {
                delete [] A[i];
            }
            delete [] A;
            delete [] B;
            delete [] result;

            Py_DECREF(pylistResult);

            return FinalResult;
        }

        else if (rowsB == 0 && colsA != colsB) {
            PyErr_SetString(PyExc_TypeError, "Number of columns of A must match number of columns of B!");
            return nullptr;
        }

        else {
            double** A = nullptr;
            double** B = nullptr;
            double* result = nullptr;
            PyObject* pylistResult = nullptr;
            PyObject* FinalResult = nullptr;

            // memory allocation
            A = new double *[rowsA];
            for (int i = 0; i < rowsA; ++i) {
                A[i] = new double [colsA];
            }
            B = new double *[rowsB];
            for (int i = 0; i < rowsB; ++i) {
                B[i] = new double[colsB];
            }
            result = new double [rowsA];

            // convert python to C++
            convertPy_2DArray(pA, A, rowsA, colsA);
            convertPy_2DArray(pB, B, rowsB, colsB);

            matrixMatrixNorm(A, B, p, rowsA, colsA, result);

            pylistResult = Convert_1DArray(result, rowsA);

            FinalResult = Py_BuildValue("O", pylistResult);

            // memory deallocation
            for (int i = 0; i < rowsA; ++i) {
                delete [] A[i];
            }
            delete [] A;
            for (int i = 0; i < rowsB; ++i) {
                delete [] B[i];
            }
            delete [] B;
            delete [] result;

            Py_DECREF(pylistResult);

            return FinalResult;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Not sure how you got here! Getting you back to safety!");
        return nullptr;
    }
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}


static PyMethodDef distanceMetricsMethods[] = {
        // Python name    C function              argument representation  description
        {"norm",          norm,                   METH_VARARGS,            "Calculate the norm between to matrices and/or vectors"},
        {"version",       (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef distanceMetricsModule = {
        PyModuleDef_HEAD_INIT,
        "distanceMetrics", // module name
        "Collection of linear algebra functions in C to be used in Python", // documentation of module
        -1, // global state
        distanceMetricsMethods // method defs
};


PyMODINIT_FUNC PyInit_CMetrics(void) {
    return PyModule_Create(&distanceMetricsModule);
}
