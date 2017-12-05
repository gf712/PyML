from pyml.linear_models.base import LinearBase
from pyml.maths import dot_product, least_squares
from pyml.metrics.scores import mean_squared_error, mean_absolute_error


class LinearRegression(LinearBase):
    def __init__(self, seed=None, bias=True, solver='OLS', learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000, alpha=0.0, batch_size=0,
                 method='normal', fudge_factor=10e-8):
        """
        Linear regression implementation

        :type seed: None or int
        :type bias: bool
        :type solver: str
        :type learning_rate: float
        :type epsilon: float
        :type max_iterations: int
        :type alpha: float
        :type batch_size: int
        :type method: str
        :type fudge_factor: float

        :param seed: random seed
        :param bias: whether or not to add a bias (column of 1s) if it isn't already present
        :param solver: use 'OLS' (ordinary least squares) or 'gradient_descent'
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
        >>> from pyml.linear_models import LinearRegression
        >>> from pyml.datasets import regression
        >>> X, y = regression(seed=1970)
        >>> lr = LinearRegression(solver='OLS', bias=True)
        >>> _ = lr.train(X, y)
        >>> lr.coefficients
        [0.3011617891659273, 0.9428803588636959]
        """

        LinearBase.__init__(self, learning_rate=learning_rate, epsilon=epsilon, max_iterations=max_iterations,
                            alpha=alpha, batch_size=batch_size, method=method, seed=seed, _type='regressor',
                            fudge_factor=fudge_factor)

        self.bias = bias
        if solver in ['OLS', 'gradient_descent']:
            self._solver = solver
        else:
            raise ValueError("Unknown solver!")

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
            theta = self._initiate_weights(bias=self.bias)
            self._coefficients, self._cost, self._iterations = self._gradient_descent(self.X, self.y, theta=theta)
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
            raise NotImplementedError("This part of the code has not been explored yet, "
                                      "returning to safety...")

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
        if scorer == 'mean_squared_error' or scorer == 'mse':
            return mean_squared_error(self.predict(X), y_true)
        elif scorer == 'mean_absolute_error' or scorer == 'mae':
            return mean_absolute_error(self.predict(X), y_true)
        else:
            raise NotImplementedError("Unknown scorer!")

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
