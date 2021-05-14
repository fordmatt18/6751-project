from itertools import product

import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment
from sklearn.preprocessing import PolynomialFeatures


class ShortestPathEnvironment(AbstractEnvironment):
    def __init__(self):
        AbstractEnvironment.__init__(self)
        self.num_x = 5
        self.num_y = 5
        # use same w, parameters from which y is generated,
        # every time without effecting other randomness in pytorch
        state = torch.random.get_rng_state()
        torch.manual_seed(10)
        self.w = torch.randn(40, 31)
        torch.random.set_rng_state(state)
        # polynomial transformation of context variables used to generate y
        self.poly = PolynomialFeatures(degree=5, include_bias=False,
                                       interaction_only=True)
        self.a = self._build_a()
        self.b = self._build_b()
        self.context_dim = 5
        self.decision_dim = self.a.shape[1]

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        x = torch.randn(n, self.context_dim)
        m = self.decision_dim
        poly_x = torch.from_numpy(self.poly.fit_transform(x)).float()
        noise = 0.75 + 0.5 * torch.rand(m)
        y = (poly_x @ self.w.T + 3) @ torch.diag(noise)
        return x, y

    def _encode_coord(self, x, y):
        return self.num_y * x + y

    def _build_a(self):
        # calculate constraint matrix A which encodes flow constraints
        # construct set of edges
        edge_nodes = []
        for x, y in product(range(self.num_x), range(self.num_y)):
            if x < self.num_x - 1:
                # add edge to next x
                source = self._encode_coord(x, y)
                dest = self._encode_coord(x + 1, y)
                edge_nodes.append((source, dest))
            if y < self.num_y - 1:
                # add edge to next y
                source = self._encode_coord(x, y)
                dest = self._encode_coord(x, y + 1)
                edge_nodes.append((source, dest))

        a = np.zeros((25, 40))
        for node, edge in product(range(self.num_x * self.num_y),
                                  range(len(edge_nodes))):
            source, dest = edge_nodes[edge]
            if node == source:
                a[node, edge] = -1.0
            elif node == dest:
                a[node, edge] = 1.0

        return a

    def _build_b(self):
        b = torch.zeros(self.num_x * self.num_y)
        b[0] = 1.0
        b[-1] = -1.0
        return b

    def get_constraints(self):
        """
        :return: constraints for given environment
        """
        return {"A_eq": self.a, "b_eq": self.b, "A_ub": None, "b_ub": None}

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
