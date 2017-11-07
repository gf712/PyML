#include <Python.h>
#include "linearalgebramodule.h"

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

void pypyMatrixVectorDotProduct(PyObject* A, PyObject* v, int ASize, int VSize, double* result) {

    int i;

    for (i = 0; i < ASize ; ++i) {

        PyObject *A_item = PyList_GetItem(A, i);

        result[i] = vector_dot_product(A_item, v, VSize);
    }

}

void ccMatrixVectorDotProduct(double** X, const double * w, double* prediction, int rows, int cols) {

    for (int i = 0; i < rows; ++i) {
        prediction[i] = 0;
        for (int j = 0; j < cols; ++j) {
            prediction[i] += X[i][j] * w[j];
        }
    }
}

//double * matrix_matrix_dot_product(PyObject *A, PyObject *B, int ASize, int BSize) {
//
//
//
//}

void vector_power(PyObject* A, int pPower, int ASize, double* result) {

    int i;

    for (i = 0; i < ASize; ++i) {

        PyObject *A_item = PyList_GetItem(A, i);
        double pElement = PyFloat_AsDouble(A_item);

        result[i] = pow(pElement, pPower);

    }

}


void pyCVectorSubtract(PyObject* u, PyObject* v, int ASize, double* result) {

    int i;

    for (i = 0; i < ASize; ++i) {

        PyObject *uItem = PyList_GetItem(u, i);
        PyObject *vItem = PyList_GetItem(v, i);

        double pUElement = PyFloat_AsDouble(uItem);
        double pVElement = PyFloat_AsDouble(vItem);

        result[i] = pUElement - pVElement;

    }
}

void cPyVectorSubtract(const double* prediction, PyObject* y, double* loss, int rows) {

    for (int i = 0; i < rows; ++i) {
        loss[i] = prediction[i] - PyFloat_AsDouble(PyList_GetItem(y, i));
    }

}


double pyVectorSum(PyObject* u, int size) {

    double sum_result = 0;
    int i;

    for (i = 0; i < size; ++i) {

        PyObject *uItem = PyList_GetItem(u, i);

        double pUElement = PyFloat_AsDouble(uItem);

        sum_result += pUElement;

    }

    return sum_result;

}

double cVectorSum(const double* array, int rows) {

    double result=0;

    for (int i = 0; i < rows; ++i) {
        result += array[i];
    }

    return result;
}


void cVectorDivide(double* X, int n, int size) {
    for (int i = 0; i < size; ++i) {
        X[i] /= (double)n;
    }
}


PyObject *Convert_1DArray(double array[], int length) {

    PyObject *pylist;
    PyObject *item;
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


PyObject* Convert_2DArray(double** array, int rows, int cols) {
    PyObject* twoDResult;
    PyObject* row;
    PyObject* item;

    twoDResult = PyList_New(rows);

    for (int j = 0; j < rows; ++j) {
        row = PyList_New(cols);
        for (int k = 0; k < cols; ++k) {
            item = PyFloat_FromDouble(array[j][k]);
            PyList_SET_ITEM(row, k, item);
        }
        PyList_SET_ITEM(twoDResult, j, row);
    }

    return twoDResult;
}


void pyTranspose(PyObject* X, double** result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        PyObject* row = PyList_GetItem(X, i);
        for (int j = 0; j < cols; ++j) {
            result[j][i] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
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
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    sizeOfA = PyList_Size(pAArray);
    sizeOfV = PyList_Size(pVVector);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    if (sizeOfV == 0){
        PyErr_SetString(PyExc_ValueError, "Argument V is empty");
        return nullptr;
    }

    double *result = nullptr;
    result = new double[sizeOfA];

    PyObject *result_py_list;


    pypyMatrixVectorDotProduct(pAArray, pVVector, sizeOfA, sizeOfV, result);

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
    int pPower;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pAArray, &pPower)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return nullptr;
    }

    // use PyList_Size to get size of vector
    sizeOfA = PyList_Size(pAArray);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }


    if (pPower < 0) {
        PyErr_SetString(PyExc_ValueError, "Power must be greater than 0.");
        return nullptr;
    }

    double *result;
    result = new double[sizeOfA];
    PyObject *result_py_list;

    vector_power(pAArray, pPower, sizeOfA, result);

    result_py_list = Convert_1DArray(result, sizeOfA);

    delete [] result;

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
        return nullptr;
    }

    // use PyList_Size to get size of vectors
    sizeOfU = PyList_Size(pUVector);
    sizeOfV = PyList_Size(pVVector);

    if (sizeOfU == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    if (sizeOfV == 0){
        PyErr_SetString(PyExc_ValueError, "Argument V is empty");
        return nullptr;
    }

    double* result = nullptr;
    result = new double[sizeOfU];

    PyObject *result_py_list;

    if (sizeOfU != sizeOfV) {
        PyErr_SetString(PyExc_ValueError, "Expected two lists of the same size.");
        return nullptr;
    }

    pyCVectorSubtract(pUVector, pVVector, sizeOfU, result);

    result_py_list = Convert_1DArray(result, sizeOfU);

    delete [] result;

    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* sum(PyObject* self, PyObject *args) {

    // prepare to handle python objects (two lists U and V), and U and V items and then n
    int sizeOfA;

    // pointers to python lists
    PyObject *pAArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pAArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list and an integer!");
        return NULL;
    }

    // use PyList_Size to get size of vector
    sizeOfA = PyList_Size(pAArray);

    if (sizeOfA == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    double result;

    result = pyVectorSum(pAArray, sizeOfA);

    PyObject *FinalResult = Py_BuildValue("d", result);

    return FinalResult;
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.2");
}


static PyObject* transpose(PyObject* self, PyObject *args) {

    // declarations
    double** result;
    int cols, rows;
    PyObject* pyResult;


    // pointers to python lists
    PyObject* pArray;

    // return error if we don't get all the arguments
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &pArray)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list!");
        return nullptr;
    }

    // use PyList_Size to get dimensions of array
    cols = PyList_Size(pArray);
    rows = PyList_Size(PyList_GetItem(pArray, 0));


    if (cols == 0){
        PyErr_SetString(PyExc_ValueError, "Argument U is empty");
        return nullptr;
    }

    // allocate memory for result
    result = new double *[cols];
    for (int i = 0; i < cols; ++i) {
        result[i] = new double[rows];
    }

    pyTranspose(pArray, result, rows, cols);

    pyResult = Convert_2DArray(result, cols, rows);

    PyObject* FinalResult = Py_BuildValue("O", pyResult);

    for (int i = 0; i < cols; ++i) {
        delete [] result[i];
    }
    delete [] result;

    Py_DECREF(pyResult);

    return FinalResult;
}

static PyMethodDef linearAlgebraMethods[] = {
        // Python name    C function              argument representation  description
        {"dot_product",   dot_product,            METH_VARARGS,            "Calculate the dot product of two vectors"},
        {"power",         power,                  METH_VARARGS,            "Calculate element wise power"},
        {"subtract",      subtract,               METH_VARARGS,            "Calculate element wise subtraction"},
        {"sum",           sum,                    METH_VARARGS,            "Calculate the total sum of a vector"},
        {"transpose",     transpose,              METH_VARARGS,            "Transpose a 2D matrix"},
        {"version",       (PyCFunction)version,   METH_NOARGS,             "Returns version."},
        {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef linearAlgebraModule = {
    PyModuleDef_HEAD_INIT,
    "linearAlgebra", // module name
    "Collection of linear algebra functions in C to be used in Python", // documentation of module
    -1, // global state
    linearAlgebraMethods // method defs
};


PyMODINIT_FUNC PyInit_Clinear_algebra(void) {
    return PyModule_Create(&linearAlgebraModule);
}
