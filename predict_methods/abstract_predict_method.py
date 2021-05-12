

class AbstractPredictMethod(object):
    def __init__(self, context_dim, decision_dim):
        """
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        self.context_dim = context_dim
        self.decision_dim = decision_dim

    def fit(self, x, y, w):
        """
        :param x: PyTorch tensor of contexts, should be shape (n, context_dim)
        :param y: PyTorch tensor of associated costs, should be shape
            (n, decision_dim)
        :param w: PyTorch tensor of weights, should be shape (n, 1)
        :return: None
        """
        raise NotImplementedError()

    def predict(self, x):
        """
        :param x: PyTorch tensor of contexts, should be shape (n, context_dim)
        :return: PyTorch tensor of predicted costs, should be shape
            (n, decision_dim)
        """
        raise NotImplementedError()
