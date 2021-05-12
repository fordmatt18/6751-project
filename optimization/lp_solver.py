

class LPSolver(object):
    def __init__(self, constraints, context_dim, decision_dim):
        """
        :param constraints: general constraints on feasible Z for LP
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        self.constraints = constraints
        self.context_dim = context_dim
        self.decision_dim = decision_dim

    def solve_lp(self, y):
        """
        :param y: batch of cost vectors, should be of shape (n, decision_dim)
        :return: corresponding batch of solutions to the LP, should be of
            shape (n, decision_dim)
        """
        # TODO: need to implement this!
        raise NotImplementedError()

    def all_feasible(self, z):
        """
        :param z: PyTorch tensor of decisions, of shape (n, decision_dim)
        :return: True if all decisions in tensor z are feasible, else false
        """
        # TODO: need to implement this!
        raise NotImplementedError()

    def get_constraints(self):
        return self.constraints
