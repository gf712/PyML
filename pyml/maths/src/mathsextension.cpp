//
// Created by Gil Ferreira Hoben on 09/11/17.
//
#include <Python.h>
#include "maths.h"
#include "pythonconverters.h"
#include "flatArrays.h"


static PyObject* quick_sort(PyObject* self, PyObject *args) {

    // variable instantiation
    auto A = new flatArray;
    auto order = new flatArray;
    int axis;

    // pointers to python lists
    PyObject* pA;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pA, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected one lists and an integer!");
        return nullptr;
    }

    // read in Python list
    A->readFromPythonList(pA);

    if (A->getRows() == 1) {

        order->startEmptyArray(A->getRows(), A->getCols());

        // if A is a vector
        auto *orderArray = new double[A->getSize()];
        double *array = A->getRow(0);


        for (int i = 0; i < A->getSize(); ++i) {
            orderArray[i] = i;
        }

        quicksort(array, orderArray, 0, A->getSize());

        order->setRow(orderArray, 0);
        A->setRow(array, 0);
    }

    else {
        // if A is a matrix
        if (axis == 1) {

            order->startEmptyArray(A->getRows(), A->getCols());


            // if axis is 1 return row wise argsort
            for (int i = 0; i < A->getRows(); ++i) {

                auto *orderArray = new double[A->getCols()];
                double *array = A->getRow(i);

                for (int j = 0; j < A->getCols(); ++j) {
                    orderArray[j] = j;
                }

                quicksort(array, orderArray, 0, A->getCols());

                order->setRow(orderArray, i);
                A->setRow(array, i);

            }
        }
        else if (axis == 0) {

            order->startEmptyArray(A->getCols(), A->getRows());


            // if axis is 0 return column wise argsort
            for (int i = 0; i < A->getCols(); ++i) {

                auto *orderArray = new double[A->getRows()];
                double *array = A->getCol(i);

                for (int j = 0; j < A->getRows(); ++j) {
                    orderArray[j] = j;
                }

                quicksort(array, orderArray, 0, A->getRows());

                order->setRow(orderArray, i);
                A->setCol(array, i);
            }
        }
        else {
            // ERROR
            PyErr_SetString(PyExc_TypeError, "Expected axis value to be 0 or 1");
            return nullptr;
        }
    }


    // convert result to python list
    PyObject* result_py_list = ConvertFlatArray_PyList(A, "float");
//    PyObject* result_py_list = Convert_1DArray(array, A->getCols());

    PyObject* order_py_list = ConvertFlatArray_PyList(order, "int");

    // build python object
    PyObject *FinalResult = Py_BuildValue("OO", result_py_list, order_py_list);

    // free up memory
    delete A;
    delete order;

    Py_DECREF(result_py_list);
    Py_DECREF(order_py_list);

    return FinalResult;
}


static PyObject* Cargmax(PyObject* self, PyObject *args) {

    // variable instantiation
    auto A = new flatArray;
    auto resultList = new flatArray;
    int axis;

    // pointers to python lists
    PyObject* pA;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pA, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected one lists and an integer!");
        return nullptr;
    }

    // read in Python list
    A->readFromPythonList(pA);

    if (A->getRows() == 1) {

        // if A is a vector
        double *array = A->getRow(0);

        resultList->startEmptyArray(1, 1);

        resultList->setNElement(argmax(array, A->getCols()), 0);

        delete [] array;
    }

    else {
        // if A is a matrix
        if (axis == 1) {

            resultList->startEmptyArray(1, A->getRows());

            double *array = nullptr;

            // if axis is 1 return row wise argmax
            for (int i = 0; i < A->getRows(); ++i) {

                array = A->getRow(i);

                resultList->setNElement(argmax(array, A->getCols()), i);

            }

            delete [] array;

        }
        else if (axis == 0) {

            resultList->startEmptyArray(1, A->getCols());

            double *array = nullptr;

            // if axis is 1 return column wise argmax
            for (int i = 0; i < A->getCols(); ++i) {

                array = A->getCol(i);

                resultList->setNElement(argmax(array, A->getRows()), i);

            }

            delete [] array;

        }
        else {
            // ERROR
            PyErr_SetString(PyExc_TypeError, "Expected axis value to be 0 or 1");
            return nullptr;
        }
    }


    // convert result to python list
    PyObject* result_py_list = ConvertFlatArray_PyList(resultList, "int");

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // free up memory
    delete A;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* Cargmin(PyObject* self, PyObject *args) {

    // variable instantiation
    auto A = new flatArray;
    auto resultList = new flatArray;
    int axis;

    // pointers to python lists
    PyObject* pA;

    // return error if we don't get all the arguments
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &pA, &axis)) {
        PyErr_SetString(PyExc_TypeError, "Expected one lists and an integer!");
        return nullptr;
    }

    // read in Python list
    A->readFromPythonList(pA);

    if (A->getRows() == 1) {

        // if A is a vector
        double *array = A->getRow(0);

        resultList->startEmptyArray(1, 1);

        resultList->setNElement(argmin(array, A->getCols()), 0);

        delete [] array;
    }

    else {
        // if A is a matrix
        if (axis == 1) {

            resultList->startEmptyArray(1, A->getRows());

            double *array = nullptr;

            // if axis is 1 return row wise argmax
            for (int i = 0; i < A->getRows(); ++i) {

                array = A->getRow(i);

                resultList->setNElement(argmin(array, A->getCols()), i);

            }

            delete [] array;

        }
        else if (axis == 0) {

            resultList->startEmptyArray(1, A->getCols());

            double *array = nullptr;

            // if axis is 1 return column wise argmax
            for (int i = 0; i < A->getCols(); ++i) {

                array = A->getCol(i);

                resultList->setNElement(argmin(array, A->getRows()), i);

            }

            delete [] array;

        }
        else {
            // ERROR
            PyErr_SetString(PyExc_TypeError, "Expected axis value to be 0 or 1");
            return nullptr;
        }
    }


    // convert result to python list
    PyObject* result_py_list = ConvertFlatArray_PyList(resultList, "int");

    // build python object
    PyObject *FinalResult = Py_BuildValue("O", result_py_list);

    // free up memory
    delete A;

    Py_DECREF(result_py_list);

    return FinalResult;
}


static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 0.1");
}


static PyMethodDef CMathsMethods[] = {
        // Python name    C function              argument representation  description
        {"quick_sort",    quick_sort,             METH_VARARGS,            "Quick sort algorithm"},
        {"Cargmin",       Cargmin,                METH_VARARGS,            "Returns index of minimum value"},
        {"Cargmax",       Cargmax,                METH_VARARGS,            "Returns index of maximum value"},
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
