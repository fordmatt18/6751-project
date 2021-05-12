from benchmark_methods.predict_then_optimize import PredictThenOptimize
from end_to_end_methods.iterative_sensitivity_method import \
    IterativeSensitivityMethod
from end_to_end_methods.weighting_methods import SingleEpsilonWeightingMethod
from environments.toy_environment import ToyEnvironment
from predict_methods.nn_predict_methods import LinearPredictMethod, \
    FlexiblePredictMethod
from sensitivity_methods.fixed_decision_sensitivity_method import \
    FixedDecisionSensitivityMethod
from utils.hyperparameter_optimization import HyperparameterPlaceholder

method_list = [
    {
        "name": "FixedDecisionLinear",
        "placeholder_options": {
            "p": [1, 2, float("inf")],
        },
        "class": IterativeSensitivityMethod,
        "args": {
            "num_iter": 2,
            "sensitivity_class": FixedDecisionSensitivityMethod,
            "sensitivity_args": {
                "p": HyperparameterPlaceholder("p"),
            },
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": 1.0,
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
    {
        "name": "FixedDecisionFlexible",
        "placeholder_options": {
            "p": [1, 2, float("inf")],
        },
        "class": IterativeSensitivityMethod,
        "args": {
            "num_iter": 2,
            "sensitivity_class": FixedDecisionSensitivityMethod,
            "sensitivity_args": {
                "p": HyperparameterPlaceholder("p"),
            },
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": 1.0,
            },
            "predict_class": FlexiblePredictMethod,
            "predict_args": {},
        },
    },
]

benchmark_list = [
    {
        "name": "PredictThenOptimizeLinear",
        "class": PredictThenOptimize,
        "args": {
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
    {
        "name": "PredictThenOptimizeFlexible",
        "class": PredictThenOptimize,
        "args": {
            "predict_class": FlexiblePredictMethod,
            "predict_args": {},
        },
    },
]

# n_range = [10000, 5000, 2000, 1000, 500, 200, 100]
n_range = [1000, 100]
num_test = 100000
num_reps = 32
num_procs = 1


toy_setup = {
    "setup_name": "toy_setup",
    "environment": {
        "class": ToyEnvironment,
        "args": {}
    },
    "n_range": n_range,
    "num_test": num_test,
    "verbose": True,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "methods": method_list,
    "benchmark_methods": benchmark_list,
}
