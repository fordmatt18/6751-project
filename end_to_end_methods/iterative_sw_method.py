import torch

from end_to_end_methods.abstract_end_to_end import AbstractEndToEnd


class IterativeSWMethod(AbstractEndToEnd):
    def __init__(self, lp_solver, context_dim, decision_dim, num_iter,
                 sensitivity_class, sensitivity_args, weighting_class,
                 weighting_args, predict_class, predict_args):
        self.num_iter = num_iter
        sensitivity_method = sensitivity_class(
            lp_solver=lp_solver, context_dim=context_dim,
            decision_dim=decision_dim, **sensitivity_args)
        self.weighting_method = weighting_class(
            sensitivity_method=sensitivity_method, **weighting_args)
        self.predict_model = predict_class(
            context_dim=context_dim, decision_dim=decision_dim, **predict_args)
        self.w = None
        AbstractEndToEnd.__init__(self, lp_solver, context_dim, decision_dim)

    def fit(self, x, y):
        n = x.shape[0]
        assert x.shape == (n, self.context_dim)
        assert y.shape == (n, self.decision_dim)

        # initialize weights uniformly
        w = torch.ones(n, 1)

        for _ in range(self.num_iter):
            # fit prediction model using previous weights
            self.predict_model.fit(x, y, w)

            # re-compute weights using prediction solution
            w = self.weighting_method.calc_weights(x, self.predict_model)

        # fit final prediction model, and save final weights
        self.w = w
        self.predict_model.fit(x, y, self.w)

    def decide(self, x):
        y_hat = self.predict_model.predict(x)
        return self.lp_solver.solve_lp(y_hat)
