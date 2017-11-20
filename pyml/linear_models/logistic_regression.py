from pyml.linear_models.base import LinearBase
from pyml.base import Classifier
from pyml.maths import dot_product, sigmoid
from pyml.maths.optimisers import gradient_descent
from pyml.metrics.scores import accuracy
from pyml.utils import set_seed


class LogisticRegression(LinearBase, Classifier):
    def __init__(self, seed=None, bias=True, learning_rate=0.01, epsilon=0.01, max_iterations=10000):
        """
        Logistic regression implementation

        :type seed: None or int
        :type bias: bool
        :type learning_rate: float
        :type epsilon: float
        :type max_iterations: int

        :param seed: random seed
        :param bias: whether or not to add a bias (column of 1s) if it isn't already present
        :param learning_rate: learning rate for gradient descent
        :param epsilon: early stopping parameter of gradient descent
        :param max_iterations: early stopping parameter of gradient descent

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

        LinearBase.__init__(self)
        Classifier.__init__(self)

        self._seed = set_seed(seed)
        self.bias = bias
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self._learning_rate = learning_rate

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

        self._initiate_weights(bias=self.bias)
        self._coefficients, self._cost, self._iterations = gradient_descent(self.X, self.coefficients, self.y,
                                                                            self.max_iterations, self.epsilon,
                                                                            self._learning_rate, 'logit')

    def _predict(self, X):

        """
        Predict class of each entry in X with trained model

        :type X: list

        :param X: list of lists with each row corresponding to a datapoint's features

        :rtype: list
        :return: list of predictions
        """

        return [int(round(x)) for x in self.predict_proba(X)]

    def _predict_proba(self, X):

        """
        Predict probability of the class of each entry in X with trained model

        :type X: list

        :param X: list of lists with each row corresponding to a datapoint's features

        :rtype: list
        :return: list of prediction probabilities
        """

        if self.bias and len(X[0]) == self._n_features + 1:
            return sigmoid(dot_product(X, self.coefficients))
        elif self.bias and len(X[0]) == self._n_features:
            return sigmoid(dot_product([[1] + row for row in X], self._coefficients))
        elif not self.bias:
            return sigmoid(dot_product(X, self.coefficients))
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
