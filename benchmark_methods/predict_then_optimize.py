import torch

from benchmark_methods.abstract_benchmark import AbstractBenchmark


class PredictThenOptimize(AbstractBenchmark):
    def __init__(self, lp_solver, context_dim, decision_dim, predict_class,
                 predict_args):
        self.predict_model = predict_class(
            context_dim=context_dim, decision_dim=decision_dim, **predict_args)
        AbstractBenchmark.__init__(self, lp_solver, context_dim, decision_dim)

    def fit(self, x, y):
        n = x.shape[0]
        assert x.shape == (n, self.context_dim)
        assert y.shape == (n, self.decision_dim)
        self.predict_model.fit(x, y, torch.ones(n, 1))

    def decide(self, x):
        y_hat = self.predict_model.predict(x)
        return self.lp_solver.solve_lp(y_hat)
