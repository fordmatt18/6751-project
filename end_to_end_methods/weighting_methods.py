from functools import partial

import numpy as np
import torch
from scipy.stats import chi2

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class AbstractWeightingMethod(object):
    def __init__(self, sensitivity_method):
        assert isinstance(sensitivity_method, AbstractSensitivityMethod)
        self.sensitivity_method = sensitivity_method

    def calc_weights(self, x, predict_model):
        weights = self._calc_weights_internal(x, predict_model)
        assert (weights >= 0).all()
        if weights.sum() == 0:
            return torch.ones(weights.shape)
        else:
            return weights

    def _calc_weights_internal(self, x, predict_model):
        raise NotImplementedError()


class SingleEpsilonWeightingMethod(AbstractWeightingMethod):
    def __init__(self, sensitivity_method, epsilon):
        AbstractWeightingMethod.__init__(self, sensitivity_method)
        self.epsilon = epsilon

    def _calc_weights_internal(self, x, predict_model):
        return self.sensitivity_method.calc_sensitivity(
            x=x, predict_model=predict_model, epsilon=self.epsilon)


class MultipleEpsilonWeightingMethod(AbstractWeightingMethod):
    def __init__(self, sensitivity_method, epsilon_list):
        AbstractWeightingMethod.__init__(self, sensitivity_method)
        self.sensitivity_method = sensitivity_method
        self.epsilon_list = epsilon_list

    def _calc_weights_internal(self, x, predict_model):
        s_list = [self.sensitivity_method.calc_sensitivity_scipy(x, predict_model, e_)
                  for e_ in self.epsilon_list]
        return torch.cat(s_list, dim=1).mean(1, keepdim=True)


class ChiSquaredWeightingMethod(MultipleEpsilonWeightingMethod):
    def __init__(self, sensitivity_method, n, df, scale):
        q_vals = np.linspace(0, 1, n+2)[1:-1]
        epsilon_list = list(chi2.ppf(q_vals, df=df, scale=scale))
        MultipleEpsilonWeightingMethod.__init__(self, sensitivity_method,
                                                epsilon_list)
