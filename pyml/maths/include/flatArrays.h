//
// Created by gil on 10/11/17.
//
#include <Python.h>


#ifndef PYML_DEV_FLATARRAYS_H
#define PYML_DEV_FLATARRAYS_H

#ifdef __cplusplus
extern "C" {
#endif

class flatArray {
private:
    int rows;
    int cols;
    int size;
    double *array;

public:
    flatArray() {
        array = nullptr;
        rows = 0;
        cols = 0;
        size = 0;
    };
    ~flatArray() {
        delete array;
    };
    void readFromPythonList(PyObject *pyList);
    void startEmptyArray(int rows_, int cols_);
    int getRows();
    int getCols();
    double* getArray();
    double getElement(int row, int col);
    void setElement(double value, int row, int col);
    flatArray* transpose();
    double getNElement(int n);
    void setNElement(double value, int n);
    int getSize();
    double sum();
    double* getRow(int i);
    double* getCol(int j);
    void setRow(double *row, int i);
};

#ifdef __cplusplus
}
#endif

#endif //PYML_DEV_FLATARRAYS_H
