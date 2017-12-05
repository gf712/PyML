//
// Created by gil on 22/11/17.
//

#include "pythonconverters.h"
#include "arrayInitialisers.h"

template <typename T>
flatArray<T>* readFromPythonList(PyObject *pyList) {

    // read in array from Python list
    // check if it's a matrix or a vector
    int cols, rows;
    flatArray<T>* result = nullptr;

    if (PyFloat_Check(PyList_GET_ITEM(pyList, 0)) || PyLong_Check(PyList_GET_ITEM(pyList, 0))) {
        cols = static_cast<int>(PyList_GET_SIZE(pyList));
        rows = 1;
    }
    else {
        rows = static_cast<int>(PyList_GET_SIZE(pyList));
        cols = static_cast<int>(PyList_GET_SIZE(PyList_GET_ITEM(pyList, 0)));
    }

    result = convertPy_flatArray <T> (pyList, rows, cols);

    return result;
}


template <typename T>
flatArray<T>* emptyArray(int rows, int cols) {
    flatArray<T>* result = nullptr;
    T* array = nullptr;
    int size = rows * cols;

    array = new T [size];

    result = new flatArray <T> (array, rows, cols);

    delete [] array;

    return result;
}


template <typename T>
inline flatArray<T>* identity(int n) {

    int row=0;
    int size = n * n;
    flatArray<T>* result = nullptr;
    T* array = nullptr;


    array = new T [size];

    for (int i = 0; i < size; ++i) {
        if (i == row) {
            array[i] = 1;
            row += n + 1;
        }
        else {
            array[i] = 0;
        }
    }

    result = new flatArray <T> (array, n, n);

    delete [] array;

    return result;
}


template <typename T>
inline flatArray<T>* constArray(int rows, int cols, double c) {

    T* array = nullptr;
    flatArray<T>* result = nullptr;
    int size = rows * cols;

    // convert c to type T
    c = static_cast<T>(c);

    array = new T [size];

    for (int i = 0; i < size; ++i) {
        array[i] = c;
    }

    result = new flatArray <T> (array, rows, cols);

    delete [] array;

    return result;
}


template <typename T>
inline flatArray<T>* zeroArray(int rows, int cols) {

    return constArray <T> (rows, cols, 0);
}


template <typename T>
inline flatArray<T>* oneArray(int rows, int cols) {

    return constArray <T> (rows, cols, 1);
}