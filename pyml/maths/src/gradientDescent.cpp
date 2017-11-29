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
inline flatArray<T> *predict(flatArray<T> *X, flatArray<T> *w, char predType[10]) {

    flatArray<T>* h;

    h = X->dot(w);

    // if using classification, calculate sigmoid(h)
    if (strcmp(predType, "logit") == 0) {
        sigmoid(h);
    }

    return h;
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

    loss->power(2, 1);

    T costResult = loss->sum() / (2 * loss->getCols());

    return costResult;
}


template <typename T>
inline T calculateCost(flatArray<T>* X, flatArray<T>* theta, flatArray<T> *y, char predType[10]) {

    T result;
    flatArray<T>* prediction = nullptr;
    char empty[0];

    prediction = predict(X, theta, empty);

    if (strcmp(predType, "logit") == 0) {
        result = logLikelihood(prediction, y);
    }

    else {

        // calculate initial cost and store result
        prediction->subtract(y, 1);
        result = cost(prediction);
    }

    delete prediction;

    return result;
}


template <typename T>
inline void updateWeights(flatArray<T>* X, flatArray<T>* y, flatArray<T>* theta, flatArray<T>* XT, flatArray<T>* nu,
                          double gamma, double learningRate, int m, flatArray<T>* n, char predType[10], char method[10],
                          flatArray<T>* epsilon, flatArray<T>* G) {

    // variable declaration
    flatArray<T>* error = nullptr;
    flatArray<T>* updateTerm = nullptr;

    if (strcmp(method, "normal") == 0) {

    // #######################################################
    //                 Vanilla gradient term
    // #######################################################
    //
    //                   updateTerm = ∇J(θ)
    //
        // get error for this step
        error = predict<T>(X, theta, predType)->subtract(y, 1);

        // calculate updateTerm = gradient
        updateTerm = XT->dot(error)->divide(n, 1);
    }

    else if (strcmp(method, "nesterov") == 0) {

    // #######################################################
    //                 Nesterov gradient term
    // #######################################################
    //
    //             updateTerm = ∇J(θ − γ · v[t-1])
    //

        // copy theta
        flatArray<T>* tempTheta = theta;

        // approximate next position of parameters (θ − γ · v[t-1])
        for (int i = 0; i < m; ++i) {
            T nu_i = nu->getNElement(i) * gamma;
            tempTheta->setNElement(theta->getNElement(i) - nu_i, i);
        }

        // calculate the gradient with new theta
        error = predict<T>(X, tempTheta, predType)->subtract(y, 1);

        updateTerm = XT->dot(error)->divide(n, 1);

    }

    else if (strcmp(method, "adagrad") == 0) {

    // #######################################################
    //                 Adagrad gradient term
    // #######################################################
    //
    //                   g[t] = ∇J(θ[t])
    //
    //                  G[t] = G[t - 1] + g[t] ** 2
    //
    //         updateTerm = g[t] / ((G[t] ** 0.5 + e)
    //
        flatArray<T>* g = nullptr;
        flatArray<T>* g_2 = nullptr;
        flatArray<T>* G_i = nullptr;

        // get predictions for this step
        error = predict<T>(X, theta, predType)->subtract(y, 1);

        g = XT->dot(error)->divide(n, 1); // g = ∇J(θ[t])

        g_2 = g->power(2, 0); // g_2 = g[t] ** 2

        G->add(g_2, 1); // G += g_2

        G_i = G->power(0.5, 0); // G_i = G ** 0.5
        G_i->add(epsilon, 1); // G_i += e

        updateTerm = g->divide(G_i, 0); // updateTerm = g / G_i

        delete g;
        delete G_i;
        delete g_2;
    }

    else if (strcmp(method, "adadelta") == 0) {

    }

    else {
        PyErr_SetString(PyExc_ValueError, method);
        return;
    }

    // #######################################################
    //             Update theta with updateTerm (v)
    // #######################################################
    //
    //            v[t] = γ · v[t-1] + η · updateTerm
    //
    //                  θ[t] = θ[t-1] − v[t]
    //

    if (updateTerm == nullptr) {
        PyErr_SetString(PyExc_ValueError, "ERROR");
        return;
    }

    for (int i = 0; i < m; ++i) {
        T nu_i = nu->getNElement(i) * gamma + updateTerm->getNElement(i) * learningRate;
        theta->setNElement(theta->getNElement(i) - nu_i, i);
        nu->setNElement(nu_i, i);
    }

    delete error;
    delete updateTerm;
}


template <typename T>
void batchGradientDescent(flatArray<T>* X, flatArray<T>* y, flatArray<T>* theta, flatArray<T>* XT,
                          flatArray<T>* costArray, flatArray<T>* nu, double e, double epsilon,
                          int maxIteration, char predType[10], double alpha,
                          double learningRate, int m, flatArray<T>* n, int& iteration, char method[10],
                          flatArray<T>* fudgeFactor) {

    // calculate gradient using the whole dataset
    T JOld;
    T JNew;
    auto* G = zeroArray<T>(1, m);

    JNew = calculateCost(X, theta, y, predType);
    costArray->setNElement(JNew, iteration);


    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // update weights
        updateWeights<T>(X, y, theta, XT, nu, alpha, learningRate, m, n, predType, method, fudgeFactor, G);

        // calculate cost for new weights
        JNew = calculateCost<T>(X, theta, y, predType);

        e = fabs(JOld) - fabs(JNew);

        costArray->setNElement(JNew, iteration+1);

        iteration++;
    }

    delete G;
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
                              int maxIteration, char predType[10], double alpha,
                              double learningRate, int m, flatArray<T>* n, int batchSize, int& iteration,
                              char method[10], flatArray<T>* fudgeFactor) {

    // calculate gradient using mini batch (where 1 <= batch_size < m)
    T JOld;
    T JNew;
    auto nScalar = static_cast<int>(n->getNElement(0));
    auto* G = zeroArray<T>(1, m);

    JNew = calculateCost(X, theta, y, predType);
    costArray->setNElement(JNew, iteration);

    auto batchIterations = static_cast<int>(floor(nScalar / batchSize));

    int k = 0;

    while (fabs(e) >= epsilon and iteration < maxIteration) {

        JOld = JNew;

        // reshuffle data
        auto rNums = new int[nScalar];

        for (int i = 0; i < nScalar; ++i) {
            rNums[i] = i;
        }

        shuffle(rNums, nScalar);
        int batchNumber = 0;
        flatArray<T>* XNew = nullptr;
        flatArray<T>* yNew = nullptr;
        flatArray<T>* XTNew = nullptr;

        XNew = emptyArray<T>(batchSize, m);
        yNew = emptyArray<T>(1, batchSize);
        XTNew = emptyArray<T>(m, batchSize);

        for (int i = 0; i < batchIterations; ++i) {


            getBatches<T>(X, y, XT, XNew, yNew, XTNew, rNums, batchSize, batchNumber, nScalar);

            // update weights using this batch
            updateWeights<T>(XNew, yNew, theta, XTNew, nu, alpha, learningRate, m, n, predType, method, fudgeFactor, G);

            // calculate overall cost
            JNew = calculateCost<T>(X, theta, y, predType);

            costArray->setNElement(JNew, k);
            batchNumber++;
            k++;
        }

        delete XNew;
        delete yNew;
        delete XTNew;

        int remainder = nScalar % batchSize;

        if (remainder != 0) {
            // in this case we need to perform update with last batch (where this batch < batchSize)
            flatArray<T>* XNewi = nullptr;
            flatArray<T>* yNewi = nullptr;
            flatArray<T>* XTNewi = nullptr;

            XNewi = emptyArray<T>(remainder, m);
            yNewi = emptyArray<T>(1, remainder);
            XTNewi = emptyArray<T>(m, remainder);

            getBatches<T>(X, y, XT, XNewi, yNewi, XTNewi, rNums, batchSize, batchNumber, nScalar);

            // update weights using this batch
            updateWeights<T>(XNewi, yNewi, theta, XTNewi, nu, alpha, learningRate, m, n, predType, method, fudgeFactor, G);

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

    delete G;
}


template <typename T>
int gradientDescent(flatArray<T> *X, flatArray<T> *y, flatArray<T> *theta, int maxIteration, T epsilon,
                    T learningRate, T alpha, flatArray<T>* costArray, char predType[10], int batchSize,
                    int seed, char method[10], T fudge_factor) {

    // set random variables
    srand(static_cast<unsigned int>(seed));

    // variable declaration
    int iteration = 0;
    double e = epsilon * 2;

    int m = X->getCols();

    auto * nArray = new T[1];
    nArray[0] = static_cast<T>(X->getRows());

    auto * fArray = new T[1];
    fArray[0] = fudge_factor;

    auto* n = new flatArray<T>(nArray, 1, 1);
    auto* fudgeFactor = new flatArray<T>(fArray, 1, 1);

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
                             predType, alpha, learningRate, m, n, iteration, method, fudgeFactor);
    }

    else if (batchSize > 0 && batchSize < X->getRows()) {
        // mini batch gradient descent (if batch size = 1 it's the equivalent of stochastic gradient descent)
        minibatchGradientDescent(X, y, theta, XT, costArray, nu, e, epsilon, maxIteration, predType,
                                 alpha, learningRate, m, n, batchSize, iteration, method, fudgeFactor);
    }

    else if (batchSize >= X->getRows()) {
        // batch_size > number of examples, default to batch gradient descent
        batchGradientDescent(X, y, theta, XT, costArray, nu, e, epsilon, maxIteration,
                             predType, alpha, learningRate, m, n, iteration, method, fudgeFactor);
    }

    else {
        PyErr_SetString(PyExc_ValueError, std::to_string(batchSize).c_str());
    }

    // free up memory
    delete XT;
    delete nu;
    delete n;
    delete [] nArray;
    delete fudgeFactor;
    delete [] fArray;

    // return number of iterations needed to reach convergence
    return iteration;
}