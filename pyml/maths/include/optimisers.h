//
// Created by gil on 16/11/17.
//


#ifndef PYML_GRADIENTDESCENT_H
#define PYML_GRADIENTDESCENT_H

template <typename T>
int gradientDescent(flatArray<T> *X, flatArray<T> *y, flatArray<T> *theta, int maxIteration, double epsilon, double learningRate, flatArray<T>* costArray, char *predType);


#endif //PYML_GRADIENTDESCENT_H
