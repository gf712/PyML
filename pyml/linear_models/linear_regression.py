from pyml.linear_models.base import LinearBase
from pyml.maths import dot_product, least_squares
from pyml.metrics.scores import mean_squared_error, mean_absolute_error


class LinearRegression(LinearBase):

    def __init__(self, bias=True, solver='OLS', learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000, alpha=0.0, batch_size=0,
                 method='normal', fudge_factor=10e-8, seed=None):

        """

        Linear regression implementation.

        Args:
            bias (bool): whether or not to add a bias (column of 1s) if it
                isn't already present
            solver (str): use 'OLS' (ordinary least squares) or
                'gradient_descent'
            learning_rate (float): learning rate of gradient descent.
            epsilon (float): early stopping criterium for gradient descent.
                If the difference in loss of two consecutive iterations is
                less than delta the algorithm stops.
            max_iterations (int): maximum number of gradients descent
                iterations.
            alpha (float): momentum of gradient descent
            batch_size (int): batch size to perform batch gradients descent.
                Set to one to perform stochastic gradient descent
            method (str): gradient descent method.
                - 'normal'
                - 'nesterov'
                - 'adagrad'
                - 'adadelta'
                - 'rmsprop'
            fudge_factor (float): fudge factor for Adagrad/Adadelta/RMSprop to
                prevent zero divisions.
            seed (int or NoneType): set random seed

        Examples:
            >>> from pyml.linear_models import LinearRegression
            >>> from pyml.datasets import regression
            >>> X, y = regression(seed=1970)
            >>> lr = LinearRegression(solver='OLS', bias=True)
            >>> _ = lr.train(X, y)
            >>> lr.coefficients
            [0.3011617891659273, 0.9428803588636959]
        """

        LinearBase.__init__(self, learning_rate=learning_rate, epsilon=epsilon,
                            max_iterations=max_iterations, alpha=alpha,
                            batch_size=batch_size, method=method, seed=seed,
                            _type='regressor', fudge_factor=fudge_factor)

        self.bias = bias
        if solver in ['OLS', 'gradient_descent']:
            self._solver = solver
        else:
            raise ValueError("Unknown solver!")

    def _train(self, X, y=None):

        """
        Train a linear regression model.

        Args:
            X (list): list of lists with each row corresponding to a
                datapoint's features
            y (list): list of labels
        """

        self.X = X
        self.y = y

        self._n_features = len(X[0])

        if self._solver == 'gradient_descent':
            theta = self._initiate_weights(bias=self.bias)
            self._coefficients, self._cost, self._iterations = \
                self._gradient_descent(self.X, self.y, theta=theta)
        else:
            if self.bias:
                self.X = [[1] + row for row in self.X]
            self._cost = 'NaN'
            self._iterations = 'NaN'
            self._coefficients = least_squares(self.X, self.y)

    def _predict(self, X):

        """
        Predict X with trained model

        Args:
            X (list): list of lists with each row corresponding to a
                datapoint's features

        Returns:
            list: list of predictions
        """

        if self.bias and len(X[0]) == self._n_features + 1:
            return dot_product(X, self.coefficients)
        elif self.bias and len(X[0]) == self._n_features:
            return dot_product([[1] + row for row in X], self._coefficients)
        elif not self.bias:
            return dot_product(X, self.coefficients)
        else:
            raise NotImplementedError("This part of the code has not been "
                                      "explored yet, returning to safety...")

    def _score(self, X, y_true, scorer='mean_squared_error'):

        """
        Model scoring.

        Args:
            X (list): list of lists with each row corresponding to a
                datapoint's features
            y_true (list): list of lists with each row corresponding to a
                datapoint's features
            scorer (str): scorer name (either 'mean_squared_error' or
                'mean_absolute_error'

        Returns:
            float: score

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
        float: returns the cost of each iteration of gradient descent or 'NaN'
            for OLS.
        """
        return self._cost

    @property
    def iterations(self):
        """
        int: returns the number of iterations of gradient descent to reach
            stopping criterium.
        """
        return self._iterations
