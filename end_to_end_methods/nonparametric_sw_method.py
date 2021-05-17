import torch

from end_to_end_methods.abstract_end_to_end import AbstractEndToEnd


class NonparametricSWMethod(AbstractEndToEnd):
    def __init__(self, lp_solver, context_dim, decision_dim,
                 sensitivity_class, sensitivity_args, weighting_class,
                 weighting_args, predict_class, predict_args,
                 flexible_predict_class, flexible_predict_args):
        sensitivity_method = sensitivity_class(
            lp_solver=lp_solver, context_dim=context_dim,
            decision_dim=decision_dim, **sensitivity_args)
        self.weighting_method = weighting_class(
            sensitivity_method=sensitivity_method, **weighting_args)
        self.predict_model = predict_class(
            context_dim=context_dim, decision_dim=decision_dim, **predict_args)
        self.flexible_model = flexible_predict_class(
            context_dim=context_dim, decision_dim=decision_dim,
            **flexible_predict_args)
        self.w = None
        AbstractEndToEnd.__init__(self, lp_solver, context_dim, decision_dim)

    def fit(self, x, y):
        n = x.shape[0]
        assert x.shape == (n, self.context_dim)
        assert y.shape == (n, self.decision_dim)

        # fit flexible prediction model
        w_uniform = torch.ones(n, 1)
        self.flexible_model.fit(x, y, w_uniform)

        # compute weights using prediction solution
        w = self.weighting_method.calc_weights(x, self.flexible_model)

        # fit final prediction model, and save final weights
        self.w = w
        self.predict_model.fit(x, y, self.w)

    def decide(self, x):
        y_hat = self.predict_model.predict(x)
        return self.lp_solver.solve_lp(y_hat)
