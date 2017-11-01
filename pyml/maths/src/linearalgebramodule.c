#include <Python.h>
#include <stdio.h>
#include <math.h>

// Handle errors
// static PyObject *algebraError;

// Define vector dot product
double vector_dot_product(PyObject* u, PyObject* v, int size) {

    double result = 0;
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

double * matrix_vector_dot_product(PyObject* A, PyObject* v, int ASize, int VSize) {

    double* row_result = malloc(sizeof(double) * ASize);
    int i;

    #pragma omp parallel for
    for (i = 0; i < ASize ; ++i) {


        PyObject *A_item = PyList_GetItem(A, i);

        row_result[i] = vector_dot_product(A_item, v, VSize);
    }

    return row_result;
}

//double * matrix_matrix_dot_product(PyObject *A, PyObject *B, int ASize, int BSize) {
//
//
//
//}

double * vector_power(PyObject* A, int pPower, int ASize) {

    double* power_result = malloc(sizeof(double) * ASize);
    int i;

    for (i = 0; i < ASize; ++i) {

        PyObject *A_item = PyList_GetItem(A, i);
        double pElement = PyFloat_AsDouble(A_item);

        power_result[i] = pow(pElement, pPower);

    }

    return power_result;

}


double * vector_subtract(PyObject* u, PyObject* v, int ASize) {

    double* subtract_result = malloc(sizeof(double) * ASize);
    int i;

    for (i = 0; i < ASize; ++i) {

        PyObject *uItem = PyList_GetItem(u, i);
        PyObject *vItem = PyList_GetItem(v, i);

        double pUElement = PyFloat_AsDouble(uItem);
        double pVElement = PyFloat_AsDouble(vItem);

        subtract_result[i] = pUElement - pVElement;

    }

    return subtract_result;

}


PyObject *Convert_1DArray(double array[], int length) {

    PyObject *pylist;
    PyObject *item;
    int i;

    pylist = PyList_New(length);

    if (pylist != NULL) {

        for (i=0; i<length; i++) {
            item = PyFloat_FromDouble(array[i]);
            PyList_SET_ITEM(pylist, i, item);

        }

    }

    return pylist;
}


// Define dot product using python lists
static PyObject* dot_product(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfA;
    int sizeOfV;

    // pointers to python lists
    PyObject *pAArray;
    PyObject *pVVector;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pAArray, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return NULL;
    }

    // use PyList_Size to get size of vectors
    sizeOfA = PyList_Size(pAArray);
    sizeOfV = PyList_Size(pVVector);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return NULL;
    }

    if (sizeOfV == 0){
        PyErr_SetString(PyExc_ValueError, "Argument V is empty");
        return NULL;
    }

    double *result;
    PyObject *result_py_list;

//    if (sizeOfU != sizeOfV) {
//        PyErr_SetString(PyExc_ValueError, "Expected two lists of the same size.");
//        return NULL;
//    }

    result = matrix_vector_dot_product(pAArray, pVVector, sizeOfA, sizeOfV);

    result_py_list = Convert_1DArray(result, sizeOfA);

    free(result);

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* power(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfA;

    // pointers to python lists
    PyObject *pAArray;
    PyObject *pPower;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pAArray, &pPower)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return NULL;
    }

    // use PyList_Size to get size of vector
    sizeOfA = PyList_Size(pAArray);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return NULL;
    }


    if (pPower < 0) {
        PyErr_SetString(PyExc_ValueError, "Power must be greater than 0.");
    }

    double *result;
    PyObject *result_py_list;

    result = vector_power(pAArray, pPower, sizeOfA);

    result_py_list = Convert_1DArray(result, sizeOfA);

    free(result);

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* subtract(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfU;
    int sizeOfV;

    // pointers to python lists
    PyObject *pUVector;
    PyObject *pVVector;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &pUVector, &PyList_Type, &pVVector)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists!");
        return NULL;
    }

    // use PyList_Size to get size of vectors
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

    double *result;
    PyObject *result_py_list;

    if (sizeOfU != sizeOfV) {
        PyErr_SetString(PyExc_ValueError, "Expected two lists of the same size.");
        return NULL;
    }

    result = vector_subtract(pUVector, pVVector, sizeOfU);

    result_py_list = Convert_1DArray(result, sizeOfU);

    free(result);

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}

static PyMethodDef linearAlgebraMethods[] = {
        // Python name    C function              argument representation  description
        {"dot_product",   dot_product,            METH_VARARGS,            "Calculate the dot product of two vectors"},
        {"power",         power,                  METH_VARARGS,            "Calculate element wise power"},
        {"subtract",      subtract,               METH_VARARGS,            "Calculate element wise subtraction"},
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
