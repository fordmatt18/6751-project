from benchmark_methods.predict_then_optimize import PredictThenOptimize
from benchmark_methods.spo_plus import SPOPlus
from end_to_end_methods.nonparametric_sw_method import \
    NonparametricSWMethod
from end_to_end_methods.iterative_sw_method import \
    IterativeSWMethod
from end_to_end_methods.weighting_methods import SingleEpsilonWeightingMethod, \
    ChiSquaredWeightingMethod
from environments.random_resource_constraint_environment import \
    RandomResourceConstraintEnvironment
from environments.shortest_path_environment import ShortestPathEnvironment
from environments.toy_environment import ToyEnvironment
from predict_methods.nn_predict_methods import LinearPredictMethod, \
    FlexiblePredictMethod
from sensitivity_methods.fixed_decision_sensitivity_method import \
    FixedDecisionSensitivityMethod
from sensitivity_methods.maximum_suboptimality_method import \
    MaximumSuboptimalitySensitivityMethod
from sensitivity_methods.variable_decision_method import \
    VariableDecisionSensitivityMethod
from utils.hyperparameter_optimization import HyperparameterPlaceholder

method_list = [
    {
        "name": "VariableDecisionLinear",
        "placeholder_options": {
            "epsilon": [1e-1, 1e0, 1e1],
            "num_iter": [2, 3, 5]
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": VariableDecisionSensitivityMethod,
            "sensitivity_args": {
                "batch_size": 10,
            },
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": HyperparameterPlaceholder("epsilon"),
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
    {
        "name": "MaximumSuboptimalityLinear",
        "placeholder_options": {
            "epsilon-scale": [1.0, 5.0],
            "num_iter": [2, 3, 5]
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": MaximumSuboptimalitySensitivityMethod,
            "sensitivity_args": {
                "batch_size": 10,
            },
            "weighting_class": ChiSquaredWeightingMethod,
            "weighting_args": {
                "n": 5,
                "scale": HyperparameterPlaceholder("epsilon-scale"),
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
    {
        "name": "FixedDecisionLinear",
        "placeholder_options": {
            "p": [1, 2, float("inf")],
            "num_iter": [2, 3, 5]
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
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
    {
        "name": "SPO+",
        "class": SPOPlus,
        "args": {
            "lmbda": 0.0,
        }
    }
]

# n_range = [10000, 5000, 2000, 1000, 500, 200, 100]
# n_range = [1000, 100]
n_range = [1000]
num_test = 10000
num_reps = 32
num_procs = 1
batch_size = 1


toy_setup = {
    "setup_name": "toy_setup",
    "environment": {
        "class": RandomResourceConstraintEnvironment,
        "args": {}
    },
    "n_range": n_range,
    "num_test": num_test,
    "verbose": True,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "methods": method_list,
    "benchmark_methods": benchmark_list,
    "batch_size": batch_size,
}
