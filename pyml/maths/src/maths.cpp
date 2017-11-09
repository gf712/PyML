//
// Created by Gil Ferreira Hoben on 07/11/17.
//
#include "maths.h"


void swap(double * a, double * b)
{
    double t = *a;
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

        swap(&array[j], &array[j+1]);

        for (int k = 0; k < size; ++k) {
            result[i][k] = array[k];
        }

    }

}


int partition(double* array, int low, int high) {
    double pivot = array[low];
    int i = low;

    for (int j = i + 1; j < high; ++j) {
        if (array[j] <= pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i], &array[low]);
    return i;
}


void quicksort(double* array, int low, int high) {
    if (low < high) {
        int p = partition(array, low, high);
        quicksort(array, low, p);
        quicksort(array, p + 1, high);
    }
}


//void quickSort(vector<int>& A, int p,int q)
//{
//    int r;
//    if(p<q)
//    {
//        r=partition(A, p,q);
//        quickSort(A,p,r);
//        quickSort(A,r+1,q);
//    }
//}
//
//
//int partition(vector<int>& A, int p,int q)
//{
//    int x= A[p];
//    int i=p;
//    int j;
//
//    for(j=p+1; j<q; j++)
//    {
//        if(A[j]<=x)
//        {
//            i=i+1;
//            swap(A[i],A[j]);
//        }
//
//    }
//
//    swap(A[i],A[p]);
//    return i;
//}