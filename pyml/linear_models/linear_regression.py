from pyml.linear_models.base import LinearBase
from pyml.maths import dot_product, least_squares
from pyml.maths.optimisers import gradient_descent
from pyml.metrics.scores import mean_squared_error, mean_absolute_error
from pyml.utils import set_seed


class LinearRegression(LinearBase):
    def __init__(self, seed=None, bias=True, solver='OLS', learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000):
        """
        Linear regression implementation

        :type seed: None or int
        :type bias: bool
        :type solver: str
        :type learning_rate: float
        :type epsilon: float
        :type max_iterations: int

        :param seed: random seed
        :param bias: whether or not to add a bias (column of 1s) if it isn't already present
        :param solver: use 'OLS' (ordinary least squares) or 'gradient_descent'
        :param learning_rate: learning rate for gradient descent
        :param epsilon: early stopping parameter for gradient descent
        :param max_iterations: early stopping parameter for gradient descent


        Example:
        --------
        >>> from pyml.linear_models import LinearRegression
        >>> from pyml.datasets import regression
        >>> X, y = regression(seed=1970)
        >>> lr = LinearRegression(solver='OLS', bias=True)
        >>> _ = lr.train(X, y)
        >>> lr.coefficients
        [0.3011617891659273, 0.9428803588636959]
        """

        LinearBase.__init__(self)

        self._seed = set_seed(seed)
        self.bias = bias
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self._learning_rate = learning_rate
        if solver in ['OLS', 'gradient_descent']:
            self._solver = solver

    def _train(self, X, y=None):

        """
        Train a linear regression model

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

        if self._solver == 'gradient_descent':
            self._initiate_weights(bias=self.bias)
            self._coefficients, self._cost, self._iterations = gradient_descent(self.X, self.coefficients, self.y,
                                                                                self.max_iterations, self.epsilon,
                                                                                self._learning_rate, 'rgrs')
        else:
            if self.bias:
                self.X = [[1] + row for row in self.X]
            self._cost = 'NaN'
            self._iterations = 'NaN'
            self._coefficients = least_squares(self.X, self.y)

    def _predict(self, X):

        """
        Predict X with trained model

        :type X: list

        :param X: list of lists with each row corresponding to a datapoint's features

        :rtype: list
        :return: list of predictions
        """

        if self.bias and len(X[0]) == self._n_features + 1:
            return dot_product(X, self.coefficients)
        elif self.bias and len(X[0]) == self._n_features:
            return dot_product([[1] + row for row in X], self._coefficients)
        elif not self.bias:
            return dot_product(X, self.coefficients)
        else:
            raise ValueError("Something went wrong.")

    def _score(self, X, y_true, scorer='mean_squared_error'):
        """
        Model scoring

        :type X: list
        :type y_true: list
        :type scorer: str

        :param X: list of lists with each row corresponding to a datapoint's features
        :param y_true: list with
        :param scorer: scorer name (either 'mean_squared_error' or 'mean_absolute_error'

        :rtype float
        :return: score
        """
        if scorer == 'mean_squared_error':
            return mean_squared_error(self.predict(X), y_true)
        elif scorer == 'mean_absolute_error':
            return mean_absolute_error(self.predict(X), y_true)

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
