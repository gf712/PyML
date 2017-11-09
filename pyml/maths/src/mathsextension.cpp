//
// Created by Gil Ferreira Hoben on 09/11/17.
//
#include <Python.h>
#include "maths.h"
#include "pythonconverters.h"


static PyObject* quick_sort(PyObject* self, PyObject *args) {

    // variable instantiation
    int size;
    double* A;

    // pointers to python lists
    PyObject* pA;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &pA)) {
        PyErr_SetString(PyExc_TypeError, "Expected one lists!");
        return nullptr;
    }

    size = static_cast<int>(PyList_GET_SIZE(pA));

    A = new double [size];

    convertPy_1DArray(pA, A, size);

    // calculate dot product
    quicksort(A, 0, size);
//    quickSort(A, 0, size - 1);
    // convert result to python list
    PyObject* result_py_list = Convert_1DArray(A, size);

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // free up memory
    delete [] A;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}


static PyMethodDef CMathsMethods[] = {
        // Python name    C function              argument representation  description
        {"quick_sort",    quick_sort,             METH_VARARGS,            "Quick sort algorithm"},
        {"version",       (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef CMathsModule = {
        PyModuleDef_HEAD_INIT,
        "CMaths", // module name
        "Collection of maths functions in C to be used in Python", // documentation of module
        -1, // global state
        CMathsMethods // method defs
};


PyMODINIT_FUNC PyInit_CMaths(void) {
    return PyModule_Create(&CMathsModule);
}
