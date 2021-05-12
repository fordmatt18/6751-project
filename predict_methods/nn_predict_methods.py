import torch.nn as nn

from predict_methods.lbfgs_predict_method import LBFGSPredictMethod


class LinearModel(nn.Module):
    def __init__(self, context_dim, decision_dim):
        nn.Module.__init__(self)
        self.linear = nn.Linear(context_dim, decision_dim)

    def forward(self, x):
        return self.linear(x)


class FlexibleModel(nn.Module):
    def __init__(self, context_dim, decision_dim, hidden_dims=None):
        nn.Module.__init__(self)
        default_hidden_dims = (200, 200)
        dim_list = (list(hidden_dims) if hidden_dims is not None
                    else default_hidden_dims)
        layer_list = []
        prev_dim = context_dim
        assert(len(dim_list) > 0)
        for dim in dim_list:
            layer_list.append(nn.Linear(prev_dim, dim))
            layer_list.append(nn.GELU())
            prev_dim = dim
        layer_list.append(nn.Linear(dim_list[-1], decision_dim))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class LinearPredictMethod(LBFGSPredictMethod):
    def __init__(self, context_dim, decision_dim):
        """
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        model_class = LinearModel
        model_args = {"context_dim": context_dim, "decision_dim": decision_dim}
        LBFGSPredictMethod.__init__(self, context_dim, decision_dim,
                                    model_class, model_args)


class FlexiblePredictMethod(LBFGSPredictMethod):
    def __init__(self, context_dim, decision_dim, hidden_dims=None):
        """
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        :param hidden_dims: tuple of hidden dim dimensions (if not provided,
            uses a default value set in FlexibleModel definition)
        """
        model_class = FlexibleModel
        model_args = {"context_dim": context_dim, "decision_dim": decision_dim,
                      "hidden_dims": hidden_dims}
        LBFGSPredictMethod.__init__(self, context_dim, decision_dim,
                                    model_class, model_args)


