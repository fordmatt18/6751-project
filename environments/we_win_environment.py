import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment

class WeWinEnvironment(AbstractEnvironment):
    def __init__(self, context_dim=3, decision_dim=10, num_constraints=5,
                 poly_deg=5):
        AbstractEnvironment.__init__(self)
        self.context_dim = context_dim
        self.decision_dim = decision_dim
        self.num_constraints = num_constraints

        self.a = np.random.randn(num_constraints, decision_dim) ** 2 + 0.1
        self.b = np.random.randn(num_constraints) ** 2 + 0.1
        self.w =  torch.randn(self.context_dim, self.decision_dim)
        self.x_mean = 5 * torch.ones(context_dim)

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        x = torch.randn(n, self.context_dim) + self.x_mean
        y = torch.diag((torch.norm(x - self.x_mean, p=2, dim=1) < self.context_dim/5)).float() @ (x @ self.w)  
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
      return x @ self.w
