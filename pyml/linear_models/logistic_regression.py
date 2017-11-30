from pyml.linear_models.base import LinearBase
from pyml.base import Classifier
from pyml.maths import dot_product, sigmoid, power, argmax
from pyml.metrics.scores import accuracy
import random
import math


class LogisticRegression(LinearBase, Classifier):
    def __init__(self, seed=None, bias=True, learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000, alpha=0.0,
                 batch_size=0, method='normal', fudge_factor=10e-8):
        """
        Logistic regression implementation

        :type seed: None or int
        :type bias: bool
        :type learning_rate: float
        :type epsilon: float
        :type max_iterations: int
        :type alpha: float
        :type batch_size: int
        :type method: str
        :type fudge_factor: float

        :param seed: random seed
        :param bias: whether or not to add a bias (column of 1s) if it isn't already present
        :param learning_rate: learning rate for gradient descent
        :param epsilon: early stopping parameter for gradient descent
        :param max_iterations: early stopping parameter for gradient descent
        :param alpha: momentum parameter for gradient descent
        :param batch_size: batch size, if it is set to zero or a number larger than training examples it will
                           default to batch gradient descent
        :param method: method to run gradient descent.
                        - "normal": vanilla GD (gradient descent)
                        - "nesterov": nesterov method for GD
                        - "adagrad": adagrad method for GD
                        - "adadelta": adadelta method for GD
                        - "rmsprop": rmsprop method for GD
        :param fudge_factor: fudge factor for Adagrad/Adadelta/RMSprop to avoid zero divisions

        Example:
        --------

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

        LinearBase.__init__(self, learning_rate=learning_rate, epsilon=epsilon, max_iterations=max_iterations,
                            alpha=alpha, batch_size=batch_size, method=method, seed=seed, _type='logit',
                            fudge_factor=fudge_factor)
        Classifier.__init__(self)

        self._bias = bias
        self._coefficients = list()

    def _train(self, X, y=None):

        """
        Train a logistic regression model

        :type X: list
        :type y: list

        :param X: list of lists with each row corresponding to a datapoint's features
        :param y: list of targets

        :rtype: object
        :return: self
        """
        self.X = X
        self.y = y

        self._n_features = len(X[0])
        self._n_classes = len(set(y))

        if self._n_classes == 2:
            theta = self._initiate_weights(bias=self._bias)
            self._coefficients, self._cost, self._iterations = self._gradient_descent(self.X, self.y, theta=theta)

        else:

            # multiclass prediction
            # let's train individual binary classifiers
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
                    theta = [random.gauss(0, 1) for x in range(self._n_features + 1)]

                _coefficients_i, cost_i, iterations_i = self._gradient_descent(self.X, y_i, theta=theta)

                # keep coefficients of each model
                self._coefficients.append(_coefficients_i)
                self._cost.append(cost_i)
                self._iterations.append(iterations_i)

    def _predict(self, X):

        """
        Predict class of each entry in X with trained model

        :type X: list

        :param X: list of lists with each row corresponding to a datapoint's features

        :rtype: list
        :return: list of predictions
        """

        if self.n_classes > 2:
            # class label corresponds to argmax of each row of scores
            return argmax(self.predict_proba(X), axis=1)
        else:
            return [int(round(x)) for x in self.predict_proba(X)]

    def _predict_proba(self, X):

        """
        Predict probability of the class of each entry in X with trained model

        :type X: list

        :param X: list of lists with each row corresponding to a datapoint's features

        :rtype: list
        :return: list of prediction probabilities
        """

        if (self._bias and len(X[0]) == self._n_features + 1) or not self._bias:

            if self.n_classes > 2:
                scores = [dot_product(X, coef) for coef in self.coefficients]
                return [softmax([scores[i][x] for i in range(self.n_classes)]) for x in range(len(scores[0]))]

            else:
                return sigmoid(dot_product(X, self.coefficients))

        elif self._bias and len(X[0]) == self._n_features:

            if self.n_classes > 2:
                scores = [dot_product([[1] + row for row in X], coef) for coef in self.coefficients]
                return [softmax([scores[i][x] for i in range(self.n_classes)]) for x in range(len(scores[0]))]

            else:
                return sigmoid(dot_product([[1] + row for row in X], self.coefficients))

        else:
            raise ValueError("Something went wrong.")

    def _score(self, X, y_true, scorer='accuracy'):
        """
        Model scoring

        :type X: list
        :type y_true: list
        :type scorer: str

        :param X: list of lists with each row corresponding to a datapoint's features
        :param y_true: list with
        :param scorer: scorer name (currently only accuracy)

        :rtype float
        :return: score
        """

        if scorer == 'accuracy':
            return accuracy(self.predict(X), y_true)
        else:
            raise ValueError("Unknown scorer")

    @property
    def seed(self):
        """
        Random seed
        :getter: returns seed used
        :type: int
        """
        return self._seed

    @property
    def coefficients(self):
        """
        Model coefficients
        :getter: returns the learnt model coefficients
        :type: list
        """
        return self._coefficients

    @property
    def cost(self):
        """
        Cost returned by cost function
        :getter: returns the cost of each iteration of gradient descent or 'NaN' for OLS
        :type: list or str
        """
        return self._cost

    @property
    def iterations(self):
        """
        Number of gradient descent iterations
        :getter: returns the nunmber of iterations of gradient descent to reach stopping criterium
        :type: int
        """
        return self._iterations

    @property
    def n_classes(self):
        """
        Number of classes
        :getter: returns the number of classes determined by the number of unique targets
        :type: int
        """
        return self._n_classes


def softmax(u):
    """
    Computes softmax of a vector u

    :param u:
    :return:
    """

    z_exp = [math.exp(u_i) for u_i in u]
    sum_z_exp = sum(z_exp)
    return [i / sum_z_exp for i in z_exp]
