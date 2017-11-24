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


template <class T>
class flatArray {
private:
    T* array;
    int rows;
    int cols;
    int size;

public:

    flatArray(T* array, int rows, int cols) {
        flatArray::rows = rows;
        flatArray::cols = cols;
        flatArray::size = rows * cols;
        flatArray::array = new T [size];

        for (int i = 0; i < size; ++i) {
            flatArray::array[i] = array[i];
        }
    }

    ~flatArray() {
        delete [] array;
    };

    // GETTERS/SETTERS
    // column and row size
    int getRows();
    int getCols();
    void setRows(int r);
    void setCols(int c);


    // matrix size
    int getSize();

    // get array
    T* getArray();

    // get array element by row and column
    T getElement(int row, int col);
    void setElement(T value, int row, int col);

    // set array element by row and column
    T getNElement(int n);
    void setNElement(T value, int n);

    // row and column
    T* getRow(int i);
    T* getCol(int j);
    void setRow(T *row, int i);
    void setCol(T *row, int j);

    // row and column slices
    T *getRowSlice(int i, int start, int end);
    T *getColSlice(int j, int start, int end);

    // MATRIX MANIPULATION/LINEAR ALGEBRA
    flatArray* transpose();
    T sum();
    flatArray *dot(flatArray *other);
    flatArray *subtract(flatArray *other);
    flatArray *power(double p);
    flatArray *divide(double m);
    flatArray *multiply(flatArray *other);
    flatArray *nlog(double base);
    flatArray *mean(int axis);
    flatArray *std(int degreesOfFreedom, int axis);
    flatArray *var(int degreesOfFreedom, int axis);
    T *diagonal();
};


#endif //PYML_FLATARRAYS_H
