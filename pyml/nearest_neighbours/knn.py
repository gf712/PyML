from pyml.nearest_neighbours.base import KNNBase
from pyml.base import Predictor, Classifier
from pyml.maths import max_occurence, mean
from pyml.metrics.scores import mean_squared_error, mean_absolute_error


class KNNClassifier(KNNBase, Classifier):
    def __init__(self, n=3, norm='l1'):
        """
        KNN classifier.

        Args:
            n (int): number of neighbours
            norm (int or str): norm to use in distance calculation

        Examples:
            >>> from pyml.datasets import gaussian
            >>> from pyml.preprocessing import train_test_split
            >>> from pyml.nearest_neighbours import KNNRegressor
            >>> X, y = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
            >>> X_train, y_train, X_test, y_test = train_test_split(X, y, train_split=0.95, seed=1970)
            >>> classifier = KNNClassifier(n=5)
            >>> _ = classifier.train(X=X_train, y=y_train)
            >>> print(classifier.score(X=X_test, y_true=y_test))
            1.0
        """

        KNNBase.__init__(self)
        self.n = n
        self.norm = norm

    def _predict(self, X):
        """
        Prediction using a majority classifier.

        Args:
            X (list): list of lists with each row corresponding to a datapoint's features.

        Returns:
            list: list of predictions
        """

        self._find_neighbours(X)
        return [max_occurence(row) for row in self._neighbours]

    def _score(self, X, y_true):
        """
        Calculates the model accuracy.
        Args:
            X (list): list of lists with each row corresponding to a datapoint's features.
            y_true (list): list of lists with each row corresponding to a datapoint's features.

        Returns:
            float: model accuracy.

        """

        predictions = self.predict(X)

        return sum([1 if pred_i == y_i else 0 for pred_i, y_i in zip(predictions, y_true)]) / len(X)


class KNNRegressor(KNNBase, Predictor):

    def __init__(self, n=3, norm='l1'):
        """
        KNN regressor.

        Args:
            n (int): number of neighbours
            norm (int or str): norm to use in distance calculation

        Examples:
            >>> from pyml.datasets import regression
            >>> from pyml.preprocessing import train_test_split
            >>> from pyml.nearest_neighbours import KNNRegressor
            >>> X, y = regression(100, seed=1970)
            >>> X_train, y_train, X_test, y_test = train_test_split(X, y, train_split=0.8, seed=1970)
            >>> regressor = KNNRegressor(n=5)
            >>> _ = regressor.train(X=X_train, y=y_train)
            >>> print(regressor.score(X=X_test, y_true=y_test, scorer='mse'))
            1.5470835956432736
        """

        KNNBase.__init__(self)
        self.n = n
        self.norm = norm

    def _predict(self, X):
        """
        Prediction using a the mean of neighbours.

        Args:
            X (list): list of lists with each row corresponding to a datapoint's features.

        Returns:
            list: list of predictions
        """

        self._find_neighbours(X)
        return [mean(row) for row in self._neighbours]

    def _score(self, X, y_true, scorer='mean_squared_error'):
        """
        Model score.

        Args:
            X (list): list of lists with each row corresponding to a datapoint's features.
            y_true (list): list of lists with each row corresponding to a datapoint's features.
            scorer (str): scorer name (either 'mean_squared_error' or 'mean_absolute_error'

        Returns:
            float: model score.

        """

        if scorer in ['mean_squared_error', 'mse']:
            return mean_squared_error(self.predict(X), y_true)
        elif scorer in ['mean_absolute_error', 'mae']:
            return mean_absolute_error(self.predict(X), y_true)
