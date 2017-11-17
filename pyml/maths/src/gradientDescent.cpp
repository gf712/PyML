//
// Created by Gil Ferreira Hoben on 16/11/17.
//

#include "flatArrays.h"
#include "optimisers.h"

void sigmoid(flatArray *scores) {
    for (int i = 0; i < scores->getSize(); ++i) {
        scores->setNElement(1 / (1 + exp(-scores->getNElement(i))), i);
    }
}


flatArray *predict(flatArray *X, flatArray *w) {

    return X->dot(w);
}


double logLikelihood(flatArray *scores, flatArray *y) {

    double result = 0;

    for (int i = 0; i < y->getSize(); ++i) {
        result += y->getNElement(i) * scores->getNElement(i) - log(1 + exp(scores->getNElement(i)));
    }

    return result;
}


double cost(flatArray* loss){
    flatArray *result = loss->power(2);
    double costResult = result->sum() / (2 * result->getCols());

    delete result;

    return costResult;
}



flatArray *gradientCalculation(flatArray *X, flatArray *loss) {

    flatArray *gradients = X->dot(loss);

    flatArray* result = gradients->divide(X->getCols());

    delete gradients;

    return result;
}


void updateWeights(flatArray *theta, flatArray *gradients, double learningRate, int size) {

    for (int i = 0; i < size; ++i) {
        theta->setNElement(theta->getNElement(i) - gradients->getNElement(i) * learningRate,i);
    }
}


double calculateCost(flatArray *X, flatArray *theta, flatArray *y, char *predType) {

    double result;
    flatArray *prediction = predict(X, theta);

    if (strcmp(predType, "logit") == 0) {

        result = logLikelihood(prediction, y);

    }

    else {

        // calculate initial cost and store result
        flatArray *loss = prediction->subtract(y);
        result = cost(loss);

        delete loss;

    }

    delete prediction;

    return result;
}


int gradientDescent(flatArray *X, flatArray *y, flatArray *theta, int maxIteration, double epsilon, double learningRate, flatArray* costArray, char *predType) {

    // variable declaration
    double JOld;
    double JNew;
    int iteration = 0;
    double e = 1000;
    int m = X->getCols();
    flatArray *XT;


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
        flatArray *h = predict(X, theta);

        if (strcmp(predType, "logit") == 0) {
            sigmoid(h);
        }

        flatArray *error = h->subtract(y);

        flatArray *gradients = gradientCalculation(XT, error);

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