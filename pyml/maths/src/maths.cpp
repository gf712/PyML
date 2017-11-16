//
// Created by Gil Ferreira Hoben on 07/11/17.
//
#include <flatArrays.h>
#include "maths.h"


void swapDouble(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}

void swapLong(long* a, long* b)
{
    long t = *a;
    *a = *b;
    *b = t;
}

int factorial(int size) {
    int fact=1;

    for(int i=2; i<=size; i++) {
        fact *= i;
    }

    return fact;
}

void permutations(double* array, double** result, int size) {

    int fact=factorial(size);

    for(int i=0;i<fact;i++) {
        int j = i % (size-1);

        swapDouble(&array[j], &array[j+1]);

        for (int k = 0; k < size; ++k) {
            result[i][k] = array[k];
        }

    }

}


int partition(double* array, double* order, int low, int high) {
    double pivot = array[low];
    int i = low;

    for (int j = i + 1; j < high; ++j) {
        if (array[j] <= pivot) {
            i++;
            swapDouble(&array[i], &array[j]);
            swapDouble(&order[i], &order[j]);
        }
    }
    swapDouble(&array[i], &array[low]);
    swapDouble(&order[i], &order[low]);
    return i;
}


void quicksort(double* array, double* order, int low, int high) {
    if (low < high) {
        int p = partition(array, order, low, high);
        quicksort(array, order, low, p);
        quicksort(array, order, p + 1, high);
    }
}
