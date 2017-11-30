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
    T* array = nullptr;
    int rows;
    int cols;
    int size;

public:

    // constructor
    flatArray(T* array, int rows, int cols) {
        flatArray::rows = rows;
        flatArray::cols = cols;
        flatArray::size = rows * cols;
        flatArray::array = new T [size];

        for (int i = 0; i < size; ++i) {
            flatArray::array[i] = array[i];
        }
    }

    // destructor
    ~flatArray() {
        delete [] array;
    };

    // Copy constructor
    flatArray(const flatArray& source) {

        // because row, cols and size are not pointers, we can shallow copy
        rows = source.rows;
        cols = source.cols;
        size = source.size;

        // array is a pointer, so we need to deep copy it if it is non-null
        // allocate memory for our copy
        array = new T[size];

        // do the copy
        if (source.array != nullptr) {
            for (int i=0; i < size; ++i)
                array[i] = source.array[i];
        }
        else {
            array = nullptr;
        }
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
    flatArray<T>* transpose();
    T sum();
    flatArray<T>* dot(flatArray *other);
    flatArray<T>* subtract(flatArray *other, int replace=0);
    flatArray<T>* add(flatArray *other, int replace=0);
    flatArray<T>* power(double p, int replace=0);
    flatArray<T>* divide(flatArray *other, int replace=0);
    flatArray<T>* multiply(flatArray *other, int replace=0);
    flatArray<T>* nlog(double base);
    flatArray<T>* mean(int axis);
    flatArray<T>* std(int degreesOfFreedom, int axis);
    flatArray<T>* var(int degreesOfFreedom, int axis);
    T* diagonal();
    double det();
    flatArray<T>* invertSign(int replace=0);
};


#endif //PYML_FLATARRAYS_H
