import torch

from predict_methods.abstract_predict_method import AbstractPredictMethod


class LBFGSPredictMethod(AbstractPredictMethod):
    def __init__(self, context_dim, decision_dim, model_class, model_args):
        """
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        AbstractPredictMethod.__init__(self, context_dim, decision_dim)
        self.model_class = model_class
        self.model_args = model_args
        self.model = None

    def fit(self, x, y, w):
        """
        :param x: PyTorch tensor of contexts, should be shape (n, context_dim)
        :param y: PyTorch tensor of associated costs, should be shape
            (n, decision_dim)
        :param w: PyTorch tensor of weights, should be shape (n, 1)
        :return: None
        """
        self.model = self.model_class(**self.model_args)
        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            y_hat = self.model(x)
            assert y_hat.shape == y.shape
            weighted_loss = (w * ((y_hat - y) ** 2)).sum(1).mean()
            weighted_loss.backward()
            return weighted_loss

        optimizer.step(closure)

    def predict(self, x):
        """
        :param x: PyTorch tensor of contexts, should be shape (n, context_dim)
        :return: PyTorch tensor of predicted costs, should be shape
            (n, decision_dim)
        """
        y_hat = self.model(x)
        return y_hat.detach()
