from pyml.linear_models.base import LinearBase
from pyml.base import Classifier
from pyml.maths import dot_product, sigmoid, argmax, softmax, transpose
from pyml.metrics.scores import accuracy
import random
import math


class LogisticRegression(LinearBase, Classifier):
    def __init__(self, seed=None, bias=True, learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000, alpha=0.0,
                 batch_size=0, method='normal', fudge_factor=10e-8):
        """
        Linear regression implementation.

        Args:
            bias (bool): whether or not to add a bias (column of 1s) if it
                isn't already present.
            solver (str): use 'OLS' (ordinary least squares) or
                'gradient_descent'.
            learning_rate (float): learning rate of gradient descent.
            epsilon (float): early stopping criterium for gradient descent.
                If the difference in loss of two consecutive iterations is
                    less than delta the algorithm stops.
            max_iterations (int): maximum number of gradients descent
                iterations.
            alpha (float): momentum of gradient descent.
            batch_size (int): batch size to perform batch gradients descent.
                Set to one to perform stochastic gradient descent.
            method (str): gradient descent method.
                - 'normal'
                - 'nesterov'
                - 'adagrad'
                - 'adadelta'
                - 'rmsprop'
            fudge_factor (float): fudge factor for Adagrad/Adadelta/RMSprop to
                prevent zero divisions.
            seed (int or NoneType): set random seed.

        Examples:
            >>> from pyml.linear_models import LogisticRegression
            >>> from pyml.datasets import gaussian
            >>> from pyml.preprocessing import train_test_split
            >>> X, y = gaussian(labels=2, sigma=0.2, seed=1970)
            >>> X_train, y_train, X_test, y_test = train_test_split(X, y, train_split=0.8, seed=1970)
            >>> lr = LogisticRegression(seed=1970)
            >>> _ = lr.train(X_train, y_train)
            >>> lr.cost[0]
            -106.11158912690777
            >>> lr.iterations
            1623
            >>> lr.coefficients
            [-1.1576475345638408, 0.1437129269620468, 2.4464052394504856]
            >>> lr.score(X_test, y_test)
            0.975
        """

        LinearBase.__init__(self, learning_rate=learning_rate, epsilon=epsilon,
                            max_iterations=max_iterations, alpha=alpha,
                            batch_size=batch_size, method=method, seed=seed,
                            _type='logit', fudge_factor=fudge_factor)
        Classifier.__init__(self)

        self._bias = bias
        self._coefficients = list()

    def _train(self, X, y=None):

        """
        Train a linear regression model.

        Args:
            X (list): list of lists with each row corresponding to a
                datapoint's features.
            y (list): list of labels.
        """

        self.X = X
        self.y = y

        self._n_features = len(X[0])
        self._n_classes = len(set(y))

        if self._n_classes == 2:
            theta = self._initiate_weights(bias=self._bias)
            self._coefficients, self._cost, self._iterations = \
                self._gradient_descent(self.X, self.y, theta=theta)

        else:
            # multiclass prediction
            # let's train individual binary classifiers
            self._one_vs_rest()

    def _predict(self, X):

        """
        Predict X with trained model

        Args:
            X (list): list of lists with each row corresponding to a
                datapoint's features.

        Returns:
            list: list of class predictions.
        """

        if self.n_classes > 2:
            # class label corresponds to argmax of each row of scores
            return argmax(self.predict_proba(X), axis=1)
        else:
            return [int(round(x)) for x in self.predict_proba(X)]

    def _predict_proba(self, X):

        """
        Predict probability of X belonging to class y with trained model.

        Args:
            X: list of lists with each row corresponding to a datapoint's
                features.

        Returns:
            list: list of probabilities.
        """

        if (self._bias and len(X[0]) == self._n_features + 1) or not \
                self._bias:

            if self.n_classes > 2:
                scores = transpose([dot_product(X, coef) for coef in
                                    self.coefficients])
                return softmax(scores)

            else:
                return sigmoid(dot_product(X, self.coefficients))

        elif self._bias and len(X[0]) == self._n_features:

            if self.n_classes > 2:
                scores = transpose([dot_product([[1] + row for row in X], coef)
                                    for coef in self.coefficients])
                return softmax(scores)

            else:
                return sigmoid(dot_product([[1] + row for row in X],
                                           self.coefficients))

        else:
            raise NotImplementedError("This part of the code has not been "
                                      "explored yet, returning to safety...")

    def _score(self, X, y_true, scorer='accuracy'):
        """
        Model scoring.

        Args:
            X:
            y_true (list): list of lists with each row corresponding to a
                datapoint's features
            scorer (str): scorer name.
                Currently only 'accuracy' is supported.

        Returns:
            float: score

        """

        if scorer == 'accuracy':
            return accuracy(self.predict(X), y_true)
        else:
            raise NotImplementedError("Unknown scorer!")

    @property
    def seed(self):
        """
        int: returns seed.
        """
        return self._seed

    @property
    def coefficients(self):
        """
        list: returns the learnt model coefficients.
         """
        return self._coefficients

    @property
    def cost(self):
        """
        float: returns the cost of each iteration of gradient descent.
        """
        return self._cost

    @property
    def iterations(self):
        """
        int: returns the number of iterations of gradient descent to reach
            stopping criterium.
        """
        return self._iterations

    @property
    def n_classes(self):
        """
        int: returns the number of classes.
        """
        return self._n_classes

    def _one_vs_rest(self):

        """
        One vs rest classification

        """

        self._cost = []
        self._iterations = []

        first = True
        for x in range(self._n_classes):
            # relabel classes
            y_i = [1 if y_ == x else 0 for y_ in self.y]

            # initiate coefficients
            if first:
                theta = self._initiate_weights(bias=self._bias)
                first = False
            else:
                theta = [random.gauss(0, 1) for x in
                         range(self._n_features + 1)]

            _coefficients_i, cost_i, iterations_i = \
                self._gradient_descent(self.X, y_i, theta=theta)

            # keep coefficients of each model
            self._coefficients.append(_coefficients_i)
            self._cost.append(cost_i)
            self._iterations.append(iterations_i)
