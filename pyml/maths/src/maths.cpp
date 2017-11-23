//
// Created by Gil Ferreira Hoben on 07/11/17.
//
#include "maths.h"

template <typename T>
inline void swap(T& a, T& b)
{
    T t = a;
    a = b;
    b = t;
}


inline int factorial(int size) {
    int fact=1;

    for(int i=2; i<=size; i++) {
        fact *= i;
    }

    return fact;
}


inline void permutations(double* array, double** result, int size) {

    int fact=factorial(size);

    for(int i=0;i<fact;i++) {
        int j = i % (size-1);

        swap(array[j], array[j+1]);

        for (int k = 0; k < size; ++k) {
            result[i][k] = array[k];
        }

    }

}


template <typename T>
inline int partition(T* array, int* order, int low, int high) {
    double pivot = array[low];
    int i = low;

    for (int j = i + 1; j < high; ++j) {
        if (array[j] <= pivot) {
            i++;
            swap(array[i], array[j]);
            swap(order[i], order[j]);
        }
    }
    swap(array[i], array[low]);
    swap(order[i], order[low]);
    return i;
}


template <typename T>
    void quicksort(T* array, int* order, int low, int high) {
    if (low < high) {
        int p = partition(array, order, low, high);
        quicksort(array, order, low, p);
        quicksort(array, order, p + 1, high);
    }
}


inline int argmax(const double *array, int size) {

    int result = 0;
    double max = array[result];

    for (int i = 1; i < size; ++i) {
        if (array[i] > max) {
            result = i;
            max = array[i];
        }
    }

    return result;
}


inline int argmin(const double *array, int size) {

    int result = 0;
    double min = array[result];

    for (int i = 1; i < size; ++i) {
        if (array[i] < min) {
            result = i;
            min = array[i];
        }
    }

    return result;
}