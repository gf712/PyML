//
// Created by gil on 16/11/17.
//


#ifndef PYML_GRADIENTDESCENT_H
#define PYML_GRADIENTDESCENT_H

#ifdef __cplusplus
extern "C" {
#endif

int gradientDescent(flatArray *X, flatArray *y, flatArray *theta, int maxIteration, double epsilon, double learningRate, flatArray* costArray, char *predType);

#ifdef __cplusplus
}
#endif

#endif //PYML_DEV_GRADIENTDESCENT_H
