from predict_methods.abstract_predict_method import AbstractPredictMethod


class AbstractSensitivityMethod(object):
    def __init__(self, lp_solver, context_dim, decision_dim):
        """
        :param lp_solver: LPSolver object containing constraints for problem
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        self.lp_solver = lp_solver
        self.context_dim = context_dim
        self.decision_dim = decision_dim

    def calc_sensitivity(self, x, predict_model, epsilon):
        """
        :param x: PyTorch tensor of contexts, should be shape (n, context_dim)
        :param predict_model: should be instance of AbstractPredictMethod
            used for computing predicted cost vectors
        :param epsilon: radius to use for Wasserstein ball
        :return: PyTorch tensor of sensitivity values, shape should be (n, 1)
        """
        n = x.shape[0]
        assert isinstance(predict_model, AbstractPredictMethod)
        y_hat = predict_model.predict(x)
        assert y_hat.shape == (n, self.decision_dim)
        s = self._calc_sensitivity_internal(x, y_hat, epsilon)
        assert s.shape == (n, 1) or s.shape == (n, self.decision_dim)
        return s

    def _calc_sensitivity_internal(self, x, y_hat, epsilon):
        """
        :param x: PyTorch tensor of contexts, should be shape (n, context_dim)
        :param y_hat: PyTorch tensor of predicted costs,
            should be shape (n, decision_dim)
        :param epsilon: radius to use for Wasserstein ball
        :return: PyTorch tensor of sensitivity values, shape should be (n, 1)
        """
        raise NotImplementedError()
