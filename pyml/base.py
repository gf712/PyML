class BaseLearner:
    def __init__(self):
        pass

    def train(self, X, y=None):
        self._train(X, y)

    def predict(self, X):
        return self._predict(X)

    def score(self, X, y_true):
        return self._score(self.predict(X), y_true)
