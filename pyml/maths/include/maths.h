//
// Created by Gil Ferreira Hoben on 07/11/17.
//

#ifndef MATHS_MATHS_H
#define MATHS_MATHS_H

#ifdef __cplusplus
extern "C" {
#endif

void permutations(double* array, double** result, int size);
void quicksort(double* array, double* order, int low, int high);
int argmax(const double *array, int size);
int argmin(const double *array, int size);

#ifdef __cplusplus
}
#endif

#endif //MATHS_MATHS_H
