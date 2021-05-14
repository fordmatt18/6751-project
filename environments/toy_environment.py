import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment


def softplus(x):
    return (torch.log(1 + torch.exp(-torch.abs(x)))
            + torch.max(x, torch.zeros_like(x)))


class ToyEnvironment(AbstractEnvironment):
    def __init__(self):
        AbstractEnvironment.__init__(self)
        self.context_dim = 3
        self.decision_dim = 5

        self.theta = torch.FloatTensor([1.0, -1.0, 2.0])
        self.y_0 = torch.FloatTensor([-0.5, -0.25, 0.0, 0.25, 0.50])
        self.y_sigma = 0.5

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        x = torch.randn(n, self.context_dim)
        h = softplus(x @ self.theta)
        y_mean = h.reshape(-1, 1) + self.y_0.reshape(1, -1)
        y = y_mean + torch.randn(y_mean.shape) * self.y_sigma
        return x, y

    def get_constraints(self):
        """
        :return: constraints for given environment
        """
        a_ub = np.eye(self.decision_dim)
        b_ub = np.ones(self.decision_dim)
        return {"A_eq": None, "b_eq": None, "A_ub": a_ub, "b_ub": b_ub}

    def get_context_dim(self):
        """
        :return: context_dim for given environment
        """
        return self.context_dim

    def get_decision_dim(self):
        """
        :return: decision_dim for given environment
        """
        return self.decision_dim
