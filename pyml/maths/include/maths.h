//
// Created by Gil Ferreira Hoben on 07/11/17.
//

#ifndef MATHS_MATHS_H
#define MATHS_MATHS_H

template <typename T>
inline T MIN(T a, T b);

template <typename T>
inline T MAX(T a, T b);

inline void shuffle(int* rNums, int size);

template <typename T>
inline void swap(T& a, T& b);

void permutations(double* array, double** result, int size);

template <typename T>
void quicksort(T* array, int* order, int low, int high);

int argmax(const double *array, int size);
int argmin(const double *array, int size);

#endif //MATHS_MATHS_H
