//
// Created by Gil Ferreira Hoben on 16/11/17.
//

#include <algorithm>
#include "flatArrays.h"
#include "flatArrays.cpp"
#include "optimisers.h"
#include "maths.h"
#include "maths.cpp"


template <typename T>
inline void sigmoid(flatArray<T> *scores) {
    for (int i = 0; i < scores->getSize(); ++i) {
        scores->setNElement(1 / (1 + exp(-scores->getNElement(i))), i);
    }
}


template <typename T>
inline flatArray<T> *predict(flatArray<T> *X, flatArray<T> *w) {

    return X->dot(w);
}


template <typename T>
inline T logLikelihood(flatArray<T> *scores, flatArray<T> *y) {

    T result = 0;

    for (int i = 0; i < y->getSize(); ++i) {
        result += y->getNElement(i) * scores->getNElement(i) - log(1 + exp(scores->getNElement(i)));
    }

    return result;
}


template <typename T>
inline T cost(flatArray<T>* loss){
    flatArray<T>* result = nullptr;

    result = loss->power(2);

    T costResult = result->sum() / (2 * result->getCols());

    delete result;

    return costResult;
}


template <typename T>
inline T calculateCost(flatArray<T>* X, flatArray<T>* theta, flatArray<T> *y, char *predType) {

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
inline void updateWeights(flatArray<T>* X, flatArray<T>* y, flatArray<T>* theta, flatArray<T>* XT, flatArray<T>* nu,
                          double alpha, double learningRate, int m, int n, char* predType) {

    flatArray<T>* h = predict<T>(X, theta);

    if (strcmp(predType, "logit") == 0) {
        sigmoid(h);
    }

    // calculate the gradient
    flatArray<T>* error = h->subtract(y);

    flatArray<T>* result = XT->dot(error);

    flatArray<T>* gradients = result->divide(n);

    // update coefficients
    for (int i = 0; i < m; ++i) {
        T nu_i = nu->getNElement(i) * alpha + gradients->getNElement(i) * learningRate;
        theta->setNElement(theta->getNElement(i) - nu_i, i);
        nu->setNElement(nu_i, i);
    }

    delete error;
    delete gradients;
    delete h;
    delete result;
}


template <typename T>
void batchGradientDescent(flatArray<T>* X, flatArray<T>* y, flatArray<T>* theta, flatArray<T>* XT,
                          flatArray<T>* costArray, flatArray<T>* nu, double e, double epsilon,
                          int maxIteration, char* predType, double alpha,
                          double learningRate, int m, int n, int& iteration) {

    // calculate gradient using the whole dataset
    T JOld;
    T JNew;

    JNew = calculateCost(X, theta, y, predType);
    costArray->setNElement(JNew, iteration);


    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // update weights
        updateWeights<T>(X, y, theta, XT, nu, alpha, learningRate, m, n, predType);

        // calculate cost for new weights
        JNew = calculateCost<T>(X, theta, y, predType);

        e = fabs(JOld) - fabs(JNew);

        costArray->setNElement(JNew, iteration+1);

        iteration++;
    }
}


template <typename T>
void getBatches(flatArray<T>* X, flatArray<T>* y, flatArray<T>* XT, flatArray<T>* XNew, flatArray<T>* yNew,
                flatArray<T>* XTNew, int* rNums, int batchSize, int batchNumber, int n) {
    int t=0;
    for (int i = batchSize * batchNumber; i < MIN(batchSize * (batchNumber + 1), n); ++i) {
        T* rowX = X->getRow(rNums[i]);
        T* rowXT = XT->getCol(rNums[i]);

        XNew->setRow(rowX, t);
        yNew->setNElement(y->getNElement(rNums[i]), t);
        XTNew->setCol(rowXT, t);

        delete rowX;
        delete rowXT;

        t++;
    }
}


template <typename T>
void minibatchGradientDescent(flatArray<T>* X, flatArray<T>* y, flatArray<T>* theta, flatArray<T>* XT,
                              flatArray<T>* costArray, flatArray<T>* nu, double e, double epsilon,
                              int maxIteration, char* predType, double alpha,
                              double learningRate, int m, int n, int batchSize, int& iteration) {

    // calculate gradient using mini batch (where 1 <= batch_size < m)
    T JOld;
    T JNew;

    JNew = calculateCost(X, theta, y, predType);
    costArray->setNElement(JNew, iteration);

    auto batchIterations = static_cast<int>(floor(n / batchSize));

    int k = 0;

    while (fabs(e) >= epsilon and iteration < maxIteration) {

        JOld = JNew;

        // reshuffle data
        auto rNums = new int[n];

        for (int i = 0; i < n; ++i) {
            rNums[i] = i;
        }

        shuffle(rNums, n);

        int batchNumber = 0;
        flatArray<T>* XNew = nullptr;
        flatArray<T>* yNew = nullptr;
        flatArray<T>* XTNew = nullptr;

        XNew = emptyArray<T>(batchSize, m);
        yNew = emptyArray<T>(1, batchSize);
        XTNew = emptyArray<T>(m, batchSize);

        for (int i = 0; i < batchIterations; ++i) {


            getBatches<T>(X, y, XT, XNew, yNew, XTNew, rNums, batchSize, batchNumber, n);

            // update weights using this batch
            updateWeights<T>(XNew, yNew, theta, XTNew, nu, alpha, learningRate, m, n, predType);

            // calculate overall cost
            JNew = calculateCost<T>(X, theta, y, predType);

            costArray->setNElement(JNew, k);
            batchNumber++;
            k++;
        }

        delete XNew;
        delete yNew;
        delete XTNew;

        int remainder = n % batchSize;

        if (remainder != 0) {
            // in this case we need to perform update with last batch (where this batch < batchSize)
            flatArray<T>* XNewi = nullptr;
            flatArray<T>* yNewi = nullptr;
            flatArray<T>* XTNewi = nullptr;

            XNewi = emptyArray<T>(remainder, m);
            yNewi = emptyArray<T>(1, remainder);
            XTNewi = emptyArray<T>(m, remainder);

            getBatches<T>(X, y, XT, XNewi, yNewi, XTNewi, rNums, batchSize, batchNumber, n);

            // update weights using this batch
            updateWeights<T>(XNewi, yNewi, theta, XTNewi, nu, alpha, learningRate, m, n, predType);

            // calculate overall cost
            JNew = calculateCost<T>(X, theta, y, predType);

            costArray->setNElement(JNew, k);

            delete XNewi;
            delete yNewi;
            delete XTNewi;
            k++;
        }

        e = fabs(JOld) - fabs(JNew);

        iteration++;
    }
}


template <typename T>
int gradientDescent(flatArray<T> *X, flatArray<T> *y, flatArray<T> *theta, int maxIteration, double epsilon,
                    double learningRate, double alpha, flatArray<T>* costArray, char *predType, int batchSize,
                    int seed) {

    // set random variables
    srand(static_cast<unsigned int>(seed));

    // variable declaration
    int iteration = 0;
    double e = epsilon * 2;

    int m = X->getCols();
    int n = X->getRows();

    flatArray<T>* XT = nullptr;
    flatArray<T>* nu = nullptr;

    // X pyTranspose (m by n matrix)
    // X is a n by m matrix
    XT = X->transpose();

    // initialise nu (when using momentum) as an empty array with same dimensions as theta (m dimensional vector)
    nu = zeroArray<T>(1, theta->getCols());

    // decide which type of gradient descent to perform (batch or mini batch gradient descent)
    if (batchSize <= 0) {
        // batch gradient descent
        batchGradientDescent(X, y, theta, XT, costArray, nu, e, epsilon, maxIteration,
                             predType, alpha, learningRate, m, n, iteration);
    }

    else if (batchSize > 0 && batchSize < X->getRows()) {
        // mini batch gradient descent (if batch size = 1 it's the equivalent of stochastic gradient descent)
        minibatchGradientDescent(X, y, theta, XT, costArray, nu, e, epsilon, maxIteration, predType,
                                 alpha, learningRate, m, n, batchSize, iteration);
    }

    else if (batchSize >= X->getRows()) {
        // batch_size > number of examples, default to batch gradient descent
        batchGradientDescent(X, y, theta, XT, costArray, nu, e, epsilon, maxIteration,
                             predType, alpha, learningRate, m, n, iteration);
    }

    else {
        PyErr_SetString(PyExc_ValueError, std::to_string(batchSize).c_str());
    }

    // free up memory
    delete XT;
    delete nu;

    // return number of iterations needed to reach convergence
    return iteration;
}