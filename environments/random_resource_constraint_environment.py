import math

import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment
from sklearn.preprocessing import PolynomialFeatures


def softplus(x):
    return (torch.log(1 + torch.exp(-torch.abs(x)))
            + torch.max(x, torch.zeros_like(x)))


class RandomResourceConstraintEnvironment(AbstractEnvironment):
    def __init__(self, context_dim=3, num_products=10, num_resources=5,
                 poly_deg=2):
        AbstractEnvironment.__init__(self)
        self.context_dim = context_dim
        self.decision_dim = num_products
        self.num_constraints = num_resources

        self.a = np.random.randn(num_resources, num_products) ** 2 + 0.1
        self.b = np.random.randn(num_resources) ** 2 + 0.1
        self.poly = PolynomialFeatures(degree=poly_deg, include_bias=False,
                                       interaction_only=True)
        self.poly_dim = self._calc_poly_dim(context_dim, poly_deg)
        # state = torch.random.get_rng_state()
        # torch.manual_seed(10)
        self.w = torch.randn(self.poly_dim, self.decision_dim)
        # torch.random.set_rng_state(state)
        # w_0 = torch.randn(num_products, self.context_dim, self.context_dim)
        # self.w = -1.0 * (w_0.permute(0, 2, 1) @ w_0)
        # self.x_0 = torch.randn(self.context_dim)
        self.max_rel_noise = 0.25
        self.target_mean_y = None
        self.offset = 0
        if self.target_mean_y is not None:
            self._set_offset(self.target_mean_y)

    def _calc_poly_dim(self, context_dim, poly_deg):
        max_d = min(context_dim, poly_deg)
        return sum(math.comb(context_dim, d) for d in range(1, max_d+1))

    def _set_offset(self, target_mean_y, num_sample=100000):
        x = torch.randn(num_sample, self.context_dim)
        mean_y = self.compute_oracle_mean_y(x)
        self.offset = target_mean_y - float(mean_y.mean())

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        x = torch.randn(n, self.context_dim)
        y_mean = self.compute_oracle_mean_y(x)
        # print(y_mean[:20])
        # print(y_mean.mean())
        sig = self.max_rel_noise
        noise = (1 - sig) + 2.0 * sig * torch.rand(n, self.decision_dim)
        y = y_mean * noise
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
        poly_x = torch.from_numpy(self.poly.fit_transform(x)).float()
        return -1.0 * softplus(poly_x @ self.w) + self.offset
