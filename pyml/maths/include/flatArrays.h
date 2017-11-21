/**
 *  @file    flatArrays.h
 *  @author  Gil Ferreira Hoben (gf712)
 *  @date    10/11/2017
 *  @version 0.1
 *
 *  @brief Flat representation of 2D arrays
 *
 *  @section DESCRIPTION
 *
 *  This class represents 2D arrays with a 1D array
 *  This improves computational as less time is spent
 *  following pointers of pointers
 *
 */

#include <Python.h>

#ifndef PYML_FLATARRAYS_H
#define PYML_FLATARRAYS_H

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
        delete [] array;
    };

    // initialisers
    void readFromPythonList(PyObject *pyList);
    void startEmptyArray(int rows_, int cols_);
    void identity(int n);
    void zeroArray(int rows_, int cols_);
    void oneArray(int rows_, int cols_);
    void constArray(int rows_, int cols_, int c);

    // GETTERS/SETTERS
    // column and row size
    int getRows();
    int getCols();
    void setRows(int r);
    void setCols(int c);

    // matrix size
    int getSize();

    // get array
    double* getArray();

    // get array element by row and column
    double getElement(int row, int col);
    void setElement(double value, int row, int col);

    // set array element by row and column
    double getNElement(int n);
    void setNElement(double value, int n);

    // row and column
    double* getRow(int i);
    double* getCol(int j);
    void setRow(double *row, int i);
    void setCol(double *row, int j);

    // row and column slices
    double *getRowSlice(int i, int start, int end);
    double *getColSlice(int j, int start, int end);

    // MATRIX MANIPULATION/LINEAR ALGEBRA
    flatArray* transpose();
    double sum();
    flatArray *dot(flatArray *other);
    flatArray *subtract(flatArray *other);
    flatArray *power(int p);
    flatArray *divide(double m);
    flatArray *multiply(flatArray *other);
    flatArray *nlog(double base);
    flatArray *mean(int axis);
    flatArray *std(int degreesOfFreedom, int axis);
    flatArray *var(int degreesOfFreedom, int axis);
    double *diagonal();
};

#ifdef __cplusplus
}
#endif

#endif //PYML_FLATARRAYS_H
