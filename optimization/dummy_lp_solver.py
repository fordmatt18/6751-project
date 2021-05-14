import torch
from torch.nn.functional import one_hot


class DummyLPSolver(object):
    def __init__(self, constraints, context_dim, decision_dim, batch_size=None):
        """
        :param constraints: general constraints on feasible Z for LP
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        self.constraints = constraints
        self.context_dim = context_dim
        self.decision_dim = decision_dim
        self.batch_size = batch_size

    def solve_lp(self, y):
        """
        :param y: batch of cost vectors, should be of shape (n, decision_dim)
        :return: corresponding batch of solutions to the LP, should be of
            shape (n, decision_dim)
        """
        return (y < 0) * 1.0

    def all_feasible(self, z):
        """
        :param z: PyTorch tensor of decisions, of shape (n, decision_dim)
        :return: True if all decisions in tensor z are feasible, else false
        """
        min_val = float(torch.min(z))
        max_val = float(torch.max(z))
        return (min_val >= 0.0) and (max_val <= 1.0)

    def get_constraints(self):
        return self.constraints
