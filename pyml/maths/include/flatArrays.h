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

// forward declaration of exceptions to avoid infinite recursion of header files
template<class T>
class flatArrayDimensionMismatchException;

template <class T>
class flatArrayColumnMismatchException;

template <class T>
class flatArrayRowMismatchException;

template <class T>
class flatArrayOutOfBoundsException;

template <class T>
class flatArrayOutOfBoundsRowException;

template <class T>
class flatArrayOutOfBoundsColumnException;

class flatArrayZeroDivisionError;
class flatArrayUnknownAxis;

template <class T>
class flatArray {
private:
    T* array = nullptr;
    int rows;
    int cols;
    int size;

public:

    // constructor
    flatArray(T* const array, int rows, int cols) {
        flatArray::rows = rows;
        flatArray::cols = cols;
        flatArray::size = rows * cols;
        flatArray::array = new T [size];

        // creates a copy of input array and stores in flatArray::array
        for (int i = 0; i < size; ++i) {
            flatArray::array[i] = array[i];
        }
    }

    // destructor
    ~flatArray() {
        delete [] array;
    }

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
    }

    // overloading +, -, / and * operators
    flatArray<T>* operator+(const flatArray<T>& other) {return add(other);}
    flatArray<T>* operator+(const T other) {return add(other);}

    flatArray<T>* operator-(const flatArray<T>& other) {return subtract(other);}
    flatArray<T>* operator-(const T other) {return subtract(other);}

    flatArray<T>* operator*(const flatArray<T>& other) {return multiply(other);}
    flatArray<T>* operator*(const T other) {return multiply(other);}

    flatArray<T>* operator/(const flatArray<T>& other) {return divide(other);}
    flatArray<T>* operator/(const T other) {return divide(other);}

    // overloading compound +, -, / and * operators
    flatArray<T>& operator+=(const flatArray<T>& other) {return *add(other, 1);}
    flatArray<T>& operator+=(const T other) {return *add(other, 1);}

    flatArray<T>& operator-=(const flatArray<T>& other) {return *subtract(other, 1);}
    flatArray<T>& operator-=(const T other) {return *subtract(other, 1);}

    flatArray<T>& operator*=(const flatArray<T>& other) {return *multiply(other, 1);}
    flatArray<T>& operator*=(const T other) {return *multiply(other, 1);}

    flatArray<T>& operator/=(const flatArray<T>& other) {return *divide(other, 1);}
    flatArray<T>& operator/=(const T other) {return *divide(other, 1);}

    // overloading of -> operator to vectorise operations
    // each function must have format f(other, inplace
//    void operator->(T (f(T, int)));

    // function to run vectorised operations
//    flatArray<T>* run();

    // simplifying element getters and setters with [] operator
    T& operator[](int n) {return array[n];}
    const T&operator[](int n) const { return array[n];}

    // GETTERS/SETTERS
    // column and row size
    int getRows()const {return rows;}
    int getCols()const {return cols;}
    void setRows(int r) {rows = r;}
    void setCols(int c) {cols = c;}

    // matrix size
    int getSize()const {return size;};

    // get array
    T * getArray()const {return array;};

    // get array element by row and column
    T getElement(int row, int col) {return array[row * cols + col];}
    void setElement(T value, int row, int col) {array[row * cols + col] = value;}

    // set array element by element position inD array
    T getNElement(int n)const {return array[n];}
    void setNElement(T value, int n)const { array[n] = value;}

    // row and column
    T * getRow(int i)const {

        if (i > rows) {
            throw flatArrayOutOfBoundsRowException<T>(*this, i);
        }

        T *row = nullptr;

        row = new T [cols];
        int n = 0;

        for (int j = i * cols; j < (i + 1) * cols; ++j) {
            row[n] = array[j];
            n++;
        }

        return row;
    }

    T* getCol(int j){

        if (j > cols) {
            throw flatArrayOutOfBoundsColumnException<T>(*this, j);
        }

        T *column = nullptr;

        column = new T [rows];
        int n = 0;

        for (int i = j; i < size; i+=cols) {
            column[n] = array[i];
            n++;
        }

        return column;
    }

    void setRow(const T *row, int i) {

        if (i > rows) {
            throw flatArrayOutOfBoundsRowException<T>(*this, i);
        }

        int n = 0;

        for (int j = i * cols; j < (i + 1) * cols; ++j) {
            array[j] = row[n];
            n++;
        }
    }

    void setCol(const T *column, int j) {

        if (j > cols) {
            throw flatArrayOutOfBoundsColumnException<T>(*this, j);
        }

        int n = 0;
        for (int k = j; k < size; k+=cols) {
            array[k] = column[n];
            n++;
        }
    }

    // row and column slices
    T *getRowSlice(int i, int start, int end);
    T *getColSlice(int j, int start, int end);

    // MATRIX MANIPULATION/LINEAR ALGEBRA
    flatArray<T>* transpose();
    T sum();
    flatArray<T>* dot(const flatArray& other);

    flatArray<T>* subtract(const flatArray& other, int replace=0);
    flatArray<T>* subtract(T other, int replace=0);

    flatArray<T>* add(const flatArray& other, int replace=0);
    flatArray<T>* add(T other, int replace=0);

    flatArray<T>* divide(const flatArray& other, int replace=0);
    flatArray<T>* divide(T other, int replace=0);

    flatArray<T>* multiply(const flatArray& other, int replace=0);
    flatArray<T>* multiply(T other, int replace=0);

    flatArray<T>* power(double p, int replace=0);

    flatArray<T>* nlog(double base, int replace=0);
    flatArray<T>* mean(int axis);
    flatArray<T>* std(int degreesOfFreedom, int axis);
    flatArray<T>* var(int degreesOfFreedom, int axis);
    T* diagonal();
    double det();
    flatArray<T>& invertSign(int replace=0);
    };

#endif //PYML_FLATARRAYS_H
