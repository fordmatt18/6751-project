
class AbstractBenchmark(object):
    def __init__(self, lp_solver, context_dim, decision_dim):
        """
        :param lp_solver: instance of LPSolver, which can be used to solve
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
            the underlying decision problem
        """
        self.lp_solver = lp_solver
        self.context_dim = context_dim
        self.decision_dim = decision_dim

    def fit(self, x, y):
        """
        fit the given End-to-end method given observed data
        :param x: Pytorch batch of contexts, shape should be (n, context_dim)
        :param y: Pytorch batch of observed costs, shape should be
            (n, decision_dim)
        :return: None
        """
        raise NotImplementedError()

    def decide(self, x):
        """
        given batch of contexts, make corresponding decisions using fit model
        :param x: Pytorch batch of contexts, shape should be (n, context_dim)
        :return: Pytorch batch of corresponding decisions, shape should be
            (n, decision_dim)
        """
        raise NotImplementedError()
