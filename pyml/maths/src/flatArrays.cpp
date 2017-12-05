/**  @file    flatArrays.cpp
 *  @author  Gil Ferreira Hoben (gf712)
 *  @date    10/11/2017
 *  @version 0.1
 *
 *  @brief Flat representation of 2D arrays
 *
 *  @section DESCRIPTION
 *
 *  This class represents 2D arrays with a 1D array
 *  This improves computation speed as less time is spent
 *  following pointers of pointers
 *
 */
#include "pythonconverters.h"
#include "linearalgebramodule.h"

template <class T>
int flatArray<T>::getRows() {
    return flatArray::rows;
}

template <class T>
int flatArray<T>::getCols() {
    return flatArray::cols;
}

template <class T>
void flatArray<T>::setRows(int r) {
    rows = r;
    // update size
    size = rows * cols;
}

template <class T>
void flatArray<T>::setCols(int c) {
    cols = c;
    // update size
    size = rows * cols;
}

template <class T>
T* flatArray<T>::getArray() {
    return flatArray::array;
}

template <class T>
int flatArray<T>::getSize() {
    return size;
}

template <class T>
T flatArray<T>::getNElement(int n) {
    if (n > size) {
        throw flatArrayOutOfBoundsException<T>(this, n);
    }
    return array[n];
}

template <class T>
void flatArray<T>::setNElement(T value, int n) {
    if (n > size) {
        throw flatArrayOutOfBoundsException<T>(this, n);
    }
    array[n] = value;
}

template <class T>
void flatArray<T>::setElement(T value, int row, int col) {
    int n = cols * row + col;

    setNElement(value, n);
}

template <class T>
T flatArray<T>::getElement(int row, int col) {
    int n = cols * row + col;

    return getNElement(n);
}


template <class T>
flatArray<T>* flatArray<T>::transpose() {
    // faster transpose method
    flatArray* result = nullptr;

    result = emptyArray <T> (cols, rows);

    for (int n = 0; n < rows * cols; ++n) {

        int column = n / rows;
        int row = n % rows * cols;

        result->setNElement(array[row + column], n);

        }

    return result;
}

template <class T>
T flatArray<T>::sum() {

    T result = 0;

    for (int n = 0; n < size; ++n) {
        result += array[n];
    };

    return result;
}

template <class T>
T* flatArray<T>::getRow(int i) {

    if (i > rows) {
        throw flatArrayOutOfBoundsRowException<T>(this, i);
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

template <class T>
T *flatArray<T>::getCol(int j) {

    if (j > cols) {
        throw flatArrayOutOfBoundsColumnException<T>(this, j);
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

template <class T>
void flatArray<T>::setRow(T *row, int i) {

    if (i > rows) {
        throw flatArrayOutOfBoundsRowException<T>(this, i);
    }

    int n = 0;

    for (int j = i * cols; j < (i + 1) * cols; ++j) {
        array[j] = row[n];
        n++;
    }
}

template <class T>
void flatArray<T>::setCol(T *column, int j) {

    if (j > cols) {
        throw flatArrayOutOfBoundsColumnException<T>(this, j);
    }

    int n = 0;
    for (int k = j; k < size; k+=cols) {
        array[k] = column[n];
        n++;
    }
}

template <class T>
flatArray<T>* flatArray<T>::dot(flatArray* other) {

    flatArray<T>* result = nullptr;

    if (other->getRows() > 1) {
        // matrix matrix multiplication
        if (cols != other->getRows()) {
            throw flatArrayColumnMismatchException<T>(this, other);
        }

        result = emptyArray<T>(rows, other->getCols());

        int rRows = result->getRows();
        int rCols = result->getCols();
        int N = getCols();
        int M = other->getCols();
        int n, i, j, k, posA, posB;
        T eResult;
        T *otherArray = other->getArray();

        for (i = 0; i < rRows; ++i) {
            for (j = 0; j < rCols; ++j) {
                posA = i * N;
                posB = j;
                eResult = 0;
                n = i * M + j;
                for (k = 0; k < N; ++k) {
                    eResult += array[posA] * otherArray[posB];

                    posA++;
                    posB += M;
                }

                result->setNElement(eResult, n);
            }
        }

        return result;
    }

    else if (other->getRows() == 1) {
        // matrix/vector vector multiplication
        if (cols != other->getCols()){
            throw flatArrayDimensionMismatchException<T>(this, other);
        }

        result = emptyArray<T>(1, rows);
        T *v = other->getArray();

        int n = 0;
        for (int i = 0; i < rows; ++i) {
            T row_result  = 0;
            for (int j = 0; j < cols; ++j) {
                row_result += array[n] * v[j];
                n++;
            }
            result->setNElement(row_result, i);
        }

        return result;
    }

    else {
        throw flatArrayDimensionMismatchException<T>(this, other);
    }
}


// this is a template for elementwise operations
template <typename T>
void elementwiseTemplate(flatArray<T>* self, flatArray<T>* other, T (f(T,T)), T* result) {

    int rows = self->getRows();
    int cols = self->getCols();
    int size = self->getSize();
    T* array = self->getArray();

    if (other->getCols() > 1 && other->getRows() == 1 && rows > 1) {
        // other is a vector and self is a matrix

        // if square matrix will prioritise column wise operation f (consider transposing matrix
        // if this is not what you want)
        if (rows == other->getCols()) {

            // number of rows match number of dimensions of vector
            T *B = other->getRow(0);
            int n = 0;
            for (int i = 0; i < rows; ++i) {

                for (int j = 0; j < cols; ++j) {

                    // operation f on each row by the ith element of other vector
                    result[n] = f(array[n], B[i]);

                    n++;
                }
            }

            delete [] B;
        }

        else if (cols == other.getCols()){

            // number of columns of self match number of dimensions of vector
            T *B = other.getRow(0);

            int n = 0;
            for (int i = 0; i < rows; ++i) {

                for (int j = 0; j < cols; ++j) {

                    // operation f on each each column by the ith element of other vector
                    result[n] = f(array[n], B[j]);

                    n++;
                }
            }

            delete [] B;
        }

        else {
            throw flatArrayDimensionMismatchException<T>(self, other);
        }
    }

    else if (other.getSize() == size) {
        // both are matrices or vectors (with same size)

        if (rows != other.getRows() || cols != other.getCols()) {
            throw flatArrayDimensionMismatchException<T>(self, other);
        }

        T *B = other.getArray();

        for (int n = 0; n < size; ++n) {
            result[n] = f(array[n], B[n]);
        }
    }

    else if (other.getCols() == 1) {
        // other is a scalar represented as an array

        T B = other.getNElement(0);

        for (int n = 0; n < size; ++n) {
            result[n] = f(array[n], B);
        }

    }

    else {
        throw flatArrayDimensionMismatchException<T>(self, other);
    }
}


template <typename T>
flatArray<T>* elementwiseHelper(flatArray<T>* self, const flatArray<T> &other, T (f(T, T)), int replace) {

    if (replace == 0) {

        int rows = self->getRows();
        int cols = self->getCols();
        auto* newArray = new T[rows * cols];

        elementwiseTemplate<T>(*self, other, f, newArray);

        auto result = new flatArray<T>(newArray, rows, cols);

        delete newArray;

        return result;
    }

    else {
        elementwiseTemplate<T>(*self, other, f, self->getArray());
        return self;
    }
}


template <typename T>
flatArray<T>* scalarElementwiseHelper(flatArray<T>* self, const T other, T (f(T, T)), int replace) {

    if (replace == 0) {

        int rows = self->getRows();
        int cols = self->getCols();
        auto* newArray = new T[rows * cols];

        scalarElementwiseTemplate<T>(*self, other, f, newArray);

        auto result = new flatArray<T>(newArray, rows, cols);

        delete newArray;

        return result;
    }

    else {
        scalarElementwiseTemplate<T>(*self, other, f, self->getArray());
        return self;
    }
}


template <class T>
flatArray<T>* flatArray<T>::add(const flatArray<T> &other, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (a + b); };

    return elementwiseHelper<T>(this, other, f, replace);
}


template <class T>
flatArray<T> *flatArray<T>::add(T other, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (a + b); };

    return scalarElementwiseHelper<T>(this, other, f, replace);
}



template <class T>
flatArray<T>* flatArray<T>::subtract(const flatArray<T> &other, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (a - b); };

    return elementwiseHelper<T>(this, other, f, replace);
}


template <class T>
flatArray<T> *flatArray<T>::subtract(T other, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (a - b); };

    return scalarElementwiseHelper<T>(this, other, f, replace);
}


template <class T>
flatArray<T>* flatArray<T>::divide(const flatArray<T> &other, int replace) {

    T (*f)(T, T)=[](T a, T b) { if (b != 0) {return (a / b);} else {throw flatArrayZeroDivisionError();} };

    return elementwiseHelper<T>(this, other, f, replace);
}

template <class T>
flatArray<T> *flatArray<T>::divide(const T other, int replace) {

    T (*f)(T, T)=[](T a, T b) { if (b != 0) {return (a / b);} else {throw flatArrayZeroDivisionError();} };

    return scalarElementwiseHelper<T>(this, other, f, replace);
}


template <class T>
flatArray<T>* flatArray<T>::multiply(const flatArray<T> &other, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (a * b); };

    return elementwiseHelper<T>(this, other, f, replace);

}


template <class T>
flatArray<T> *flatArray<T>::multiply(T other, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (a * b); };

    return scalarElementwiseHelper<T>(this, other, f, replace);
}


template <class T>
flatArray<T>* flatArray<T>::power(double p, int replace) {

    T (*f)(T, T)=[](T a, T b) { return (pow(a, b)); };

    return scalarElementwiseHelper<T>(this, p, f, replace);
}


template <class T>
flatArray<T>* flatArray<T>::nlog(double base, int replace) {


    T (*f)(T, T)=[](T a, T b) { return (log(a) / log(b)); };

    return scalarElementwiseHelper<T>(this, base, f, replace);
}

template <class T>
flatArray<T>* flatArray<T>::mean(int axis) {

    flatArray<T>* result = nullptr;

    if (rows == 1) {
        // vector
        T rowResult = 0;

        result = emptyArray<T>(1, 1);

        for (int i = 0; i < cols; ++i) {
            rowResult += array[i];
        }

        rowResult /= static_cast<T>(cols);

        result->setNElement(rowResult, 0);
    }

    else if (rows > 1) {
        // matrix
        if (axis == 0) {
            // mean of each column
            T colResult;

            result = emptyArray<T>(1 , cols);

            for (int i = 0; i < cols; ++i) {

                colResult = 0;
                T *colArray = getCol(i);

                for (int j = 0; j < rows; ++j) {
                    colResult += colArray[j];
                }

                colResult /= static_cast<T>(rows);

                result->setNElement(colResult, i);

                delete [] colArray;
            }
        }

        else if (axis == 1) {
            // mean of each row
            T rowResult;

            result = emptyArray<T>(1, rows);

            for (int i = 0; i < rows; ++i) {

                rowResult = 0;
                T *rowArray = getRow(i);

                for (int j = 0; j < cols; ++j) {
                    rowResult += rowArray[j];
                }

                rowResult /= static_cast<T>(cols);

                result->setNElement(rowResult, i);

                delete [] rowArray;
            }
        }

        else {
            throw flatArrayUnknownAxis(axis);
        }

    }

    return result;
}

template <class T>
flatArray<T>* flatArray<T>::std(int degreesOfFreedom, int axis) {

    flatArray* result = nullptr;
    flatArray *arrayVar = nullptr;

    arrayVar = var(degreesOfFreedom, axis);

    if (rows == 1) {
        // vector
        result = emptyArray<T>(1, 1);

        T arrayMean_i = arrayVar->getNElement(0);

        result->setNElement(pow(arrayMean_i, 0.5), 0);
    }

    else if (rows > 1) {
        // matrix
        if (axis == 0) {
            // std of each column
            result = emptyArray<T>(1, cols);

            for (int i = 0; i < cols; ++i) {

                result->setNElement(pow(arrayVar->getNElement(i), 0.5), i);
            }

        }

        else {
            // std of each row
            result = emptyArray<T>(1, rows);

            for (int i = 0; i < rows; ++i) {

                result->setNElement(pow(arrayVar->getNElement(i), 0.5), i);

            }
        }
    }

    delete arrayVar;

    return result;
}

template <class T>
flatArray<T>* flatArray<T>::var(int degreesOfFreedom, int axis) {

    flatArray<T>* result = nullptr;
    flatArray<T>* arrayMean = nullptr;

    arrayMean = mean(axis);

    if (rows == 1) {
        // vector
        T rowResult = 0;

        result = emptyArray<T>(1, 1);

        T arrayMean_i = arrayMean->getNElement(0);

        for (int i = 0; i < size; ++i) {
            rowResult += pow(array[i] - arrayMean_i, 2);
        }

        rowResult /= static_cast<T>(size - degreesOfFreedom);

        result->setNElement(rowResult, 0);
    }

    else if (rows > 1) {
        // matrix
        if (axis == 0) {
            // std of each column
            T colResult;

            result = emptyArray<T>(1, cols);

            for (int i = 0; i < cols; ++i) {

                colResult = 0;
                T colMean = arrayMean->getNElement(i);
                T *colArray = getCol(i);

                for (int j = 0; j < rows; ++j) {
                    colResult += pow(colArray[j] - colMean, 2);
                }

                colResult /= static_cast<T>(rows - degreesOfFreedom);

                result->setNElement(colResult, i);

                delete [] colArray;
            }
        }

        else if (axis == 1) {
            // std of each row
            T rowResult;

            result = emptyArray<T>(1, rows);

            for (int i = 0; i < rows; ++i) {

                rowResult = 0;
                T rowMean = arrayMean->getNElement(i);
                T *rowArray = getRow(i);

                for (int j = 0; j < cols; ++j) {
                    rowResult += pow(rowArray[j] - rowMean, 2);
                }

                rowResult /=  static_cast<T>(cols - degreesOfFreedom);

                result->setNElement(rowResult, i);

                delete [] rowArray;
            }
        }

        else {
            throw flatArrayUnknownAxis(axis);
        }
    }

    delete arrayMean;

    return result;
}

template <class T>
T* flatArray<T>::getRowSlice(int i, int start, int end) {

    T* result = nullptr;
    T* row = nullptr;

    row = getRow(i);

    if (end > getCols()) {
        arrayOutOfBoundsException(getRows(), end);
    }

    result = new T [end - start];

    int n = 0;
    for (int j = start; j < end; ++j) {

        result[n] = row[j];

        n++;
    }

    delete [] row;

    return result;
}

template <class T>
T* flatArray<T>::getColSlice(int j, int start, int end) {

    T* result = nullptr;
    T* col = nullptr;

    col = getCol(j);

    if (end > getCols()) {
        arrayOutOfBoundsException(getCols(), end);
    }

    result = new T[end - start];

    int n = 0;
    for (int i = start; i < end; ++i) {

        result[n] = col[i];

        n++;
    }

    delete [] col;

    return result;
}

template <class T>
T* flatArray<T>::diagonal() {

    T *result = nullptr;

    result = new T [rows];

    for (int i = 0; i < rows; ++i) {
        result[i] = array[i + i * rows];
    }

    return result;
}

template <typename T>
double flatArray<T>::det() {
    return determinant(this);
}


template <typename T>
flatArray<T>& flatArray<T>::invertSign(int replace) {

    if (replace == 0) {
        auto newArray = new double[size];

        for (int i = 0; i < size; ++i) {
            newArray[i] = -array[i];
        }

        flatArray<T> result = flatArray<T>(newArray, rows, cols);

        delete newArray;

        return result;
    }
    else {
        for (int i = 0; i < size; ++i) {
            array[i] = -array[i];
        }
        return *this;
    }
}
