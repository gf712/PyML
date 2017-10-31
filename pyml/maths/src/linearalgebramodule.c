#include <Python.h>

// Handle errors
// static PyObject *algebraError;

// Define vector dot product
int vector_dot_product(PyObject* u, PyObject* v, int size) {

    int result = 0;
    int i;

    for (i = 0; i < size; ++i) {

        PyObject *v_item = PyList_GetItem(v, i);
        PyObject *u_item = PyList_GetItem(u, i);

        double pUItem = PyFloat_AsDouble(u_item);
        double pVItem = PyFloat_AsDouble(v_item);
        result += pUItem * pVItem;
    }

    return result;
}

// read in Python objects
static PyObject* dot_product(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfU;
    int sizeOfV;
    double result = 0.0;

    PyObject *pUVector;
    PyObject *pVVector;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pUVector, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return NULL;
    }

    sizeOfU = PyList_Size(pUVector);
    sizeOfV = PyList_Size(pVVector);

    if (sizeOfU == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return NULL;
    }

    if (sizeOfV == 0){
        PyErr_SetString(PyExc_ValueError, "Argument V is empty");
        return NULL;
    }

    if (sizeOfU != sizeOfV) {
        PyErr_SetString(PyExc_ValueError, "Expected two lists of the same size.");
        return NULL;
    }

    result = vector_dot_product(pUVector, pVVector, sizeOfU);

    return Py_BuildValue("d", result);
}

static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}

static PyMethodDef linearAlgebraMethods[] = {
        // Python name    C function              argument representation  description
        {"dot_product",   dot_product,            METH_VARARGS,            "Calculated the dot product of two vectors"},
        {"version",       (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef linearAlgebraModule = {
    PyModuleDef_HEAD_INIT,
    "linearAlgebra", // module name
    "Collection of linear algebra functions in C to be used in Python", // documentation of module
    -1, // global state
    linearAlgebraMethods // method defs
};


PyMODINIT_FUNC PyInit_linearAlgebraModule(void) {
    return PyModule_Create(&linearAlgebraModule);
}
