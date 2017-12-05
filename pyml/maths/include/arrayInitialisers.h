//
// Created by gil on 22/11/17.
//

#ifndef PYML_ARRAYINITIALISERS_H
#define PYML_ARRAYINITIALISERS_H

#include <Python.h>

// initialisers
template <typename T>
flatArray<T>* readFromPythonList(PyObject *pyList);

template <typename T>
flatArray<T>* emptyArray(int rows, int cols);

template <typename T>
inline flatArray<T>* identity(int n);

template <typename T>
inline flatArray<T>* zeroArray(int rows, int cols);

template <typename T>
inline flatArray<T>* oneArray(int rows, int cols);

template <typename T>
inline flatArray<T>* constArray(int rows, int cols, double c);

#endif //PYML_ARRAYINITIALISERS_H
