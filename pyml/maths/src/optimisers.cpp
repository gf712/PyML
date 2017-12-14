//
// Created by Gil Ferreira Hoben on 16/11/17.
//

#include <algorithm>
#include "flatArrays.h"
#include "flatArrays.cpp"
#include "optimisersExtension.h"
#include "maths.h"
#include "maths.cpp"


template <typename T>
inline void sigmoid(flatArray<T> *scores) {
    for (int i = 0; i < scores->getSize(); ++i) {
        scores->setNElement(1 / (1 + exp(-scores->getNElement(i))), i);
    }
}


template <typename T>
inline void predict(flatArray<T> &X, flatArray<T> &w, char predType[10], flatArray<T> *result) {

    *result = *X.dot(w);

    // if using classification, calculate sigmoid(h)
    if (strcmp(predType, "logit") == 0) {
        sigmoid(result);
    }
}

template <typename T>
inline T logLikelihood(flatArray<T> &scores, flatArray<T> &y) {

    T result = 0;

    for (int i = 0; i < y.getSize(); ++i) {
        result += y[i] * scores[i] - log(1 + exp(scores[i]));
    }

    return result;
}


template <typename T>
inline T cost(flatArray<T>& loss){

    loss.power(2, 1);

    T costResult = loss.sum() / (2 * loss.getCols());

    return costResult;
}


template <typename T>
inline T calculateCost(flatArray<T>& X, flatArray<T>& theta, flatArray<T> &y,
                       flatArray<T>* prediction, char predType[10]) {

    T result;
    char empty[0];

    predict<T>(X, theta, empty, prediction);


    if (strcmp(predType, "logit") == 0) {
        result = logLikelihood(*prediction, y);
    }

    else {

        // calculate initial cost and store result
        *prediction-=y;
        result = cost(*prediction);
    }

    return result;
}


template <typename T>
inline void updateWeights(flatArray<T>& X, flatArray<T>& y, flatArray<T>* theta, flatArray<T>& XT, flatArray<T>* nu,
                          flatArray<T>* error, double gamma, double learningRate, int m, T n, char predType[10],
                          char method[10], T epsilon, flatArray<T>* G, int iteration) {

    // variable declaration
    flatArray<T>* updateTerm = nullptr;

    if (strcmp(method, "normal") == 0) {

    // #######################################################
    //                 Vanilla gradient term
    // #######################################################
    //
    //                   updateTerm = ∇J(θ)
    //
        // get error for this step
        predict<T>(X, *theta, predType, error);
        *error -= y;

        // calculate updateTerm = gradient
        updateTerm = XT.dot(*error);
        *updateTerm /= n;
    }

    else if (strcmp(method, "nesterov") == 0) {

    // #######################################################
    //                 Nesterov gradient term
    // #######################################################
    //
    //             updateTerm = ∇J(θ − γ · v[t-1])
    //

        // copy theta
        auto* tempTheta = new flatArray<T>(*theta);

        // approximate next position of parameters (θ − γ · v[t-1])
        for (int i = 0; i < m; ++i) {
            T nu_i = nu->getNElement(i) * gamma;
            tempTheta->setNElement((*theta)[i] - nu_i, i);
        }

        // calculate the gradient with new theta
        predict<T>(X, *tempTheta, predType, error);
        *error -= y;

        updateTerm = XT.dot(*error);
        *updateTerm /= n;

        delete tempTheta;

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
        predict<T>(X, *theta, predType, error);
        (*error) -= y;

        g = XT.dot(*error);
        (*g) /= n; // g = ∇J(θ[t])

        g_2 = g->power(2, 0); // g_2 = g[t] ** 2

        (*G) += (*g_2); // G += g_2

        G_i = G->power(0.5, 0); // G_i = G ** 0.5
        (*G_i) += epsilon; // G_i += e

        updateTerm = g->divide(*G_i, 0); // updateTerm = g / G_i

        delete g;
        delete G_i;
        delete g_2;
    }

    else if (strcmp(method, "adadelta") == 0) {
    // #######################################################
    //                 Adadelta gradient term
    // #######################################################
    //
    //            RMS[∆θ][t] = (E[∆θ**2] + e) ** .5
    //
    //          ∆θt= (-RMS[∆θ[t-1]] / RMS[g[t]]) · g[t]
    //
    // Note: In this implementation the negative sign is ignored,
    // since all update terms are subtracted from theta anyway
    //

        flatArray<T>* g = nullptr;
        flatArray<T>* g_2 = nullptr;
        flatArray<T>* E_prev = nullptr;
        flatArray<T>* E_i = nullptr;

        // previous mean
        // (E[g[t-1]**2] + e) ** .5
        E_prev = (*G) + epsilon;
        E_prev->power(0.5, 1);

        // get predictions for this step
        predict<T>(X, *theta, predType, error);
        *error -= y;

        g = XT.dot(*error);
        *g /= n; // g = ∇J(θ[t])

        g_2 = g->power(2, 0); // g_2 = g[t] ** 2

        // online mean:
        //
        // delta = (x - mean) / n
        // mean += delta

        *g_2 -= *G;

        // new mean
        for (int j = 0; j < m; ++j) {
            T temp = g_2->getNElement(j) / static_cast<T>(iteration + 1);
            G->setNElement(G->getNElement(j) + temp, j); // G += E[∆θ**2]
        }

        E_i = *G + epsilon;
        E_i->power(0.5, 1); // RMS[g[t]]

        updateTerm = *E_prev / *E_i;

        *updateTerm *= *g;// g[t] * (RMS[∆θ[t-1]] / RMS[g[t]])

        delete g;
        delete g_2;
        delete E_prev;
        delete E_i;
    }

    else if (strcmp(method, "rmsprop") == 0) {

        // #######################################################
        //                 RMSprop gradient update
        // #######################################################
        //
        // RMS[∆θ][t] = ((γ · E[g[t-1]]**2] + (1 - γ) · E[g[t]]**2] )  + e) ** .5
        //
        //         θ[t + 1] = θ[t] - (η / RMS[∆θ][t]) · g[t]
        //
        // In this case we skip the generic update since the momentum
        // is already used in the update term

        flatArray<T>* g = nullptr;
        flatArray<T>* g_2 = nullptr;

        // previous mean
        // E[g[t-1]**2]
        auto* E_prev = new flatArray<T>(*G);

        // get predictions for this step
        predict<T>(X, *theta, predType, error);
        *error -= y;

        g = XT.dot(*error);
        *g /= n; // g = ∇J(θ[t])

        g_2 = g->power(2, 0); // g_2 = g[t] ** 2

        // online mean:
        //
        // delta = (x - mean) / n
        // mean += delta
        //
        // This avoids recalculating the mean after each iteration

        *g_2 -= *G;
        T temp;

        // new mean and gradient update
        for (int j = 0; j < m; ++j) {
            temp = g_2->getNElement(j) / static_cast<T>(iteration + 1);
            G->setNElement(G->getNElement(j) + temp, j); // G += E[∆θ**2]
            T update = learningRate / pow((gamma * E_prev->getNElement(j) + (1 - gamma) * G->getNElement(j)) + epsilon, 0.5);
            theta->setNElement(theta->getNElement(j) - update * g->getNElement(j), j);
        }

        delete g;
        delete g_2;
        delete E_prev;

        // skip generic theta update
        goto END;
    }

    else {
        PyErr_SetString(PyExc_ValueError, method);
        return;
    }

    // #######################################################
    //             Update theta with updateTerm
    // #######################################################
    //
    //          v[t] = γ · v[t-1] + η · updateTerm
    //
    //                 θ[t + 1] = θ[t-1] − v[t]
    //


    if (updateTerm == nullptr) {
        PyErr_SetString(PyExc_ValueError, "ERROR");
        return;
    }

    for (int i = 0; i < m; ++i) {
        // to switch off momentum set gamma to 0
        // to switch off learning rate set learningRate to 1
        T nu_i = nu->getNElement(i) * gamma + updateTerm->getNElement(i) * learningRate;
        theta->setNElement(theta->getNElement(i) - nu_i, i);
        nu->setNElement(nu_i, i);
    }

    END:
    delete updateTerm;
}


template <typename T>
void batchGradientDescent(flatArray<T>& X, flatArray<T>& y, flatArray<T>* theta, flatArray<T>& XT,
                          flatArray<T>* costArray, flatArray<T>* nu, double e, double epsilon,
                          int maxIteration, char predType[10], double alpha,
                          double learningRate, int m, T n, int& iteration, char method[10],
                          T fudgeFactor) {

    // calculate gradient using the whole dataset
    T JOld;
    T JNew;
    auto* G = zeroArray<T>(1, m);
    auto* prediction = emptyArray<T>(1, n);
    auto* error = emptyArray<T>(1, n);

    JNew = calculateCost(X, *theta, y, prediction, predType);
    costArray->setNElement(JNew, iteration);


    while (fabs(e) >= epsilon and iteration < maxIteration) {

        // update J
        JOld = JNew;

        // update weights
        updateWeights<T>(X, y, theta, XT, nu, error, alpha, learningRate, m, n, predType, method, fudgeFactor, G,
                         iteration);

//        PyErr_SetString(PyExc_ValueError, std::to_string(y.getNElement(0)).c_str());

        // calculate cost for new weights
        JNew = calculateCost<T>(X, *theta, y, prediction, predType);

        e = fabs(JOld) - fabs(JNew);

        costArray->setNElement(JNew, iteration+1);

        iteration++;
    }

    delete G;
    delete prediction;
    delete error;
}


template <typename T>
void getBatches(flatArray<T>& X, flatArray<T>& y, flatArray<T>& XT, flatArray<T>* XNew, flatArray<T>* yNew,
                flatArray<T>* XTNew, int* rNums, int batchSize, int batchNumber, int n) {

    int t=0;
    for (int i = batchSize * batchNumber; i < MIN(batchSize * (batchNumber + 1), n); ++i) {
        T* rowX = X.getRow(rNums[i]);
        T* rowXT = XT.getCol(rNums[i]);

        XNew->setRow(rowX, t);
        yNew->setNElement(y[rNums[i]], t);
        XTNew->setCol(rowXT, t);

        delete rowX;
        delete rowXT;

        t++;
    }
}


template <typename T>
void minibatchGradientDescent(flatArray<T>& X, flatArray<T>& y, flatArray<T>* theta, flatArray<T>& XT,
                              flatArray<T>* costArray, flatArray<T>* nu, double e, double epsilon,
                              int maxIteration, char predType[10], double alpha,
                              double learningRate, int m, T n, int batchSize, int& iteration,
                              char method[10], T fudgeFactor) {

    // calculate gradient using mini batch (where 1 <= batch_size < m)
    int remainder = static_cast<int>(n) % batchSize;

    T JOld;
    T JNew;
    auto* G = zeroArray<T>(1, m);
    auto* prediction = emptyArray<T>(1, n);
    auto* batchError = emptyArray<T>(1, batchSize);
    auto* batchErrorRemainder = emptyArray<T>(1, remainder);

    JNew = calculateCost(X, *theta, y, prediction, predType);
    costArray->setNElement(JNew, iteration);

    auto batchIterations = static_cast<int>(floor(n / batchSize));

    int k = 0;

    while (fabs(e) >= epsilon and iteration < maxIteration) {

        JOld = JNew;

        // reshuffle data
        auto rNums = new int[static_cast<int>(n)];

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
            updateWeights<T>(*XNew, *yNew, theta, *XTNew, nu, batchError, alpha, learningRate, m, batchSize, predType,
                             method, fudgeFactor, G, iteration);

            // calculate overall cost
            JNew = calculateCost<T>(X, *theta, y, prediction, predType);

            costArray->setNElement(JNew, k);
            batchNumber++;
            k++;
        }

        delete XNew;
        delete yNew;
        delete XTNew;

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
            updateWeights<T>(*XNewi, *yNewi, theta, *XTNewi, nu, batchErrorRemainder, alpha, learningRate, m, remainder,
                             predType, method, fudgeFactor, G, iteration);

            // calculate overall cost
            JNew = calculateCost<T>(X, *theta, y, prediction, predType);

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
    delete prediction;
    delete batchError;
    delete batchErrorRemainder;
}


template <typename T>
int gradientDescent(flatArray<T>& X, flatArray<T> &y, flatArray<T> *theta, int maxIteration, T epsilon,
                    T learningRate, T alpha, flatArray<T>* costArray, char predType[10], int batchSize,
                    int seed, char method[10], T fudge_factor) {

    // set random variables
    srand(static_cast<unsigned int>(seed));

    // variable declaration
    int iteration = 0;
    double e = epsilon * 2;

    int m = X.getCols();

    auto n = static_cast<T>(X.getRows());

    flatArray<T>* XT = nullptr;
    flatArray<T>* nu = nullptr;

    // X pyTranspose (m by n matrix)
    // X is a n by m matrix
    XT = X.transpose();

    // initialise nu (when using momentum) as an empty array with same dimensions as theta (m dimensional vector)
    nu = zeroArray<T>(1, theta->getCols());

    // decide which type of gradient descent to perform (batch or mini batch gradient descent)
    if (batchSize <= 0) {
        // batch gradient descent
        batchGradientDescent(X, y, theta, *XT, costArray, nu, e, epsilon, maxIteration,
                             predType, alpha, learningRate, m, n, iteration, method, fudge_factor);
    }

    else if (batchSize > 0 && batchSize < X.getRows()) {
        // mini batch gradient descent (if batch size = 1 it's the equivalent of stochastic gradient descent)
        minibatchGradientDescent(X, y, theta, *XT, costArray, nu, e, epsilon, maxIteration, predType,
                                 alpha, learningRate, m, n, batchSize, iteration, method, fudge_factor);
    }

    else if (batchSize >= X.getRows()) {
        // batch_size > number of examples, default to batch gradient descent
        batchGradientDescent(X, y, theta, *XT, costArray, nu, e, epsilon, maxIteration,
                             predType, alpha, learningRate, m, n, iteration, method, fudge_factor);
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

