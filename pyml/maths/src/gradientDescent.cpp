//
// Created by Gil Ferreira Hoben on 16/11/17.
//

#include "flatArrays.h"
#include "flatArrays.cpp"
#include "optimisers.h"


template <typename T>
void sigmoid(flatArray<T> *scores) {
    for (int i = 0; i < scores->getSize(); ++i) {
        scores->setNElement(1 / (1 + exp(-scores->getNElement(i))), i);
    }
}


template <typename T>
flatArray<T> *predict(flatArray<T> *X, flatArray<T> *w) {

    return X->dot(w);
}


template <typename T>
T logLikelihood(flatArray<T> *scores, flatArray<T> *y) {

    T result = 0;

    for (int i = 0; i < y->getSize(); ++i) {
        result += y->getNElement(i) * scores->getNElement(i) - log(1 + exp(scores->getNElement(i)));
    }

    return result;
}


template <typename T>
T cost(flatArray<T>* loss){
    flatArray<T>* result = nullptr;

    result = loss->power(2);

    T costResult = result->sum() / (2 * result->getCols());

    delete result;

    return costResult;
}



template <typename T>
flatArray<T> *gradientCalculation(flatArray<T> *X, flatArray<T> *loss) {

    flatArray<T>* gradients = X->dot(loss);

    flatArray<T>* result = gradients->divide(X->getCols());

    delete gradients;

    return result;
}


template <typename T>
void updateWeights(flatArray<T> *theta, flatArray<T> *gradients, double learningRate, int size) {

    for (int i = 0; i < size; ++i) {
        theta->setNElement(theta->getNElement(i) - gradients->getNElement(i) * learningRate,i);
    }
}


template <typename T>
T calculateCost(flatArray<T>* X, flatArray<T>* theta, flatArray<T> *y, char *predType) {

    T result;
    flatArray<T>* prediction = nullptr;

    prediction = predict(X, theta);

    if (strcmp(predType, "logit") == 0) {

        result = logLikelihood(prediction, y);

    }

    else {

        // calculate initial cost and store result
        flatArray<T> *loss = prediction->subtract(y);
        result = cost(loss);

        delete loss;

    }

    delete prediction;

    return result;
}


template <typename T>
int gradientDescent(flatArray<T> *X, flatArray<T> *y, flatArray<T> *theta, int maxIteration, double epsilon, double learningRate, flatArray<T>* costArray, char *predType) {

    // variable declaration
    T JOld;
    T JNew;

    int iteration = 0;
    double e = 1000;

    int m = X->getCols();

    flatArray<T> *XT = nullptr;


    // X pyTranspose (m by n matrix)
    // X is a n by m matrix
    XT = X->transpose();

    JNew = calculateCost(X, theta, y, predType);
    costArray->setNElement(JNew, iteration);

    // gradient descent
    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // calculate gradient
        flatArray<T> *h = predict(X, theta);

        if (strcmp(predType, "logit") == 0) {
            sigmoid(h);
        }

        flatArray<T> *error = h->subtract(y);

        flatArray<T> *gradients = gradientCalculation(XT, error);

        // update coefficients
        updateWeights(theta, gradients, learningRate, m);

        // calculate cost for new weights
        JNew = calculateCost(X, theta, y, predType);

        e = fabs(JOld) - fabs(JNew);

        costArray->setNElement(JNew, iteration+1);

        iteration++;

        delete h;
        delete error;
        delete gradients;
    }

    // free up memory
    delete XT;

    // return number of iterations needed to reach convergence
    return iteration;
}