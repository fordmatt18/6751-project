import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment


class RandomResourceConstraintEnvironment(AbstractEnvironment):
    def __init__(self, context_dim, num_products, num_resources):
        AbstractEnvironment.__init__(self)
        self.context_dim = context_dim
        self.decision_dim = num_products
        self.num_constraints = num_resources

        self.a = np.random.randn(num_resources, num_products) ** 2
        self.b = np.random.randn(num_resources) ** 2
        w_0 = torch.randn(num_products, self.context_dim,
                                 self.context_dim)
        self.w = -1.0 * (w_0.permute(0, 2, 1) @ w_0)
        self.x_0 = torch.randn(self.context_dim)
        self.sigma = .03

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        x = torch.randn(n, self.context_dim) ** 2
        y_mean = self.compute_oracle_mean_y(x)
        y = y_mean + self.sigma * torch.randn(n, self.decision_dim)
        return x, y

    def get_constraints(self):
        """
        :return: constraints for given environment
        """
        return {"A_eq": None, "b_eq": None, "A_ub": self.a, "b_ub": self.b}

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

    def compute_oracle_mean_y(self, x):
        x_0 = x - self.x_0.view(1, self.context_dim)
        return torch.einsum("ijk,nj,nk->ni", self.w, x_0, x_0)
