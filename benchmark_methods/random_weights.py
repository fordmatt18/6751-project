import torch

from benchmark_methods.abstract_benchmark import AbstractBenchmark


class RandomWeights(AbstractBenchmark):
    def __init__(self, lp_solver, context_dim, decision_dim, num_weights,
                 predict_class, predict_args, multi=True):
        self.num_weights = num_weights
        self.predict_args = predict_args
        self.predict_class = predict_class
        self.predict_model = None
        self.w = None
        self.multi = multi
        AbstractBenchmark.__init__(self, lp_solver, context_dim, decision_dim)

    def fit(self, x, y):
        n = x.shape[0]
        assert x.shape == (n, self.context_dim)
        assert y.shape == (n, self.decision_dim)

        best_performance = float("inf")
        for _ in range(self.num_weights):
            if self.multi:
                w = torch.randn(n, self.decision_dim) ** 2
            else:
                w = torch.randn(n, 1) ** 2
            predict_model = self.predict_class(
                context_dim=self.context_dim, decision_dim=self.decision_dim,
                **self.predict_args)
            predict_model.fit(x, y, w)
            z = self._decide_internal(x, predict_model)
            train_pv = float((z * y).sum(1).mean(0))
            if train_pv < best_performance:
                self.w = w
                self.predict_model = predict_model
                best_performance = train_pv

    def decide(self, x):
        return self._decide_internal(x, self.predict_model)

    def _decide_internal(self, x, predict_model):
        y_hat = predict_model.predict(x)
        return self.lp_solver.solve_lp(y_hat)
