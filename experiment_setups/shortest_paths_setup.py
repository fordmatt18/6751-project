from benchmark_methods.predict_then_optimize import PredictThenOptimize
from benchmark_methods.random_weights import RandomWeights
from benchmark_methods.spo_plus import SPOPlus
from end_to_end_methods.iterative_sw_method import \
    IterativeSWMethod
from end_to_end_methods.nonparametric_sw_method import NonparametricSWMethod
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
from sensitivity_methods.multi_fixed_decision_sensitivity_method import \
    MultiFixedDecisionSensitivityMethod
from sensitivity_methods.multi_maximum_suboptimality_method import \
    MultiMaximumSuboptimalitySensitivityMethod
from sensitivity_methods.multi_variable_decision_method import \
    MultiVariableDecisionSensitivityMethod
from sensitivity_methods.variable_decision_method import \
    VariableDecisionSensitivityMethod
from utils.hyperparameter_optimization import HyperparameterPlaceholder, \
    GlobalSetupVal


method_list = [
    {
        "name": "VariableDecisionLinearIter",
        "placeholder_options": {
            "epsilon": [1e0, 1e1, 5e1],
            "num_iter": [2, 3],
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": VariableDecisionSensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
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
        "name": "VariableDecisionLinearNP",
        "placeholder_options": {
            "epsilon": [1e0, 1e1, 5e1],
        },
        "class": NonparametricSWMethod,
        "args": {
            "sensitivity_class": VariableDecisionSensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
            },
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": HyperparameterPlaceholder("epsilon"),
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
            "flexible_predict_class": FlexiblePredictMethod,
            "flexible_predict_args": {},
        },
    },
    {
        "name": "MultiVariableDecisionLinearIter",
        "placeholder_options": {
            "epsilon": [1e0, 1e1, 5e1],
            "num_iter": [2, 3],
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": MultiVariableDecisionSensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
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
        "name": "MultiVariableDecisionLinearNP",
        "placeholder_options": {
            "epsilon": [1e0, 1e1, 5e1],
        },
        "class": NonparametricSWMethod,
        "args": {
            "sensitivity_class": MultiVariableDecisionSensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
            },
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": HyperparameterPlaceholder("epsilon"),
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
            "flexible_predict_class": FlexiblePredictMethod,
            "flexible_predict_args": {},
        },
    },
    {
        "name": "MaximumSuboptimalityLinearIter",
        "placeholder_options": {
            "epsilon-scale": [5.0, 10.0],
            "num_iter": [2, 3],
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": MaximumSuboptimalitySensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
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
        "name": "MaximumSuboptimalityLinearNP",
        "placeholder_options": {
            "epsilon-scale": [5.0, 10.0],
        },
        "class": NonparametricSWMethod,
        "args": {
            "sensitivity_class": MaximumSuboptimalitySensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
            },
            "weighting_class": ChiSquaredWeightingMethod,
            "weighting_args": {
                "n": 5,
                "scale": HyperparameterPlaceholder("epsilon-scale"),
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
            "flexible_predict_class": FlexiblePredictMethod,
            "flexible_predict_args": {},
        },
    },
    {
        "name": "MultiMaximumSuboptimalityLinearIter",
        "placeholder_options": {
            "epsilon-scale": [5.0, 10.0],
            "num_iter": [2, 3],
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": MultiMaximumSuboptimalitySensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
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
        "name": "MultiMaximumSuboptimalityLinearNP",
        "placeholder_options": {
            "epsilon-scale": [5.0, 10.0],
        },
        "class": NonparametricSWMethod,
        "args": {
            "sensitivity_class": MultiMaximumSuboptimalitySensitivityMethod,
            "sensitivity_args": {
                "batch_size": GlobalSetupVal("batch_size"),
            },
            "weighting_class": ChiSquaredWeightingMethod,
            "weighting_args": {
                "n": 5,
                "scale": HyperparameterPlaceholder("epsilon-scale"),
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
            "flexible_predict_class": FlexiblePredictMethod,
            "flexible_predict_args": {},
        },
    },
    {
        "name": "FixedDecisionLinearIter",
        "placeholder_options": {
            "p": [1, 2, float("inf")],
            "num_iter": [2, 3],
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
    {
        "name": "FixedDecisionLinearNP",
        "placeholder_options": {
            "p": [1, 2, float("inf")],
        },
        "class": NonparametricSWMethod,
        "args": {
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
            "flexible_predict_class": FlexiblePredictMethod,
            "flexible_predict_args": {},
        },
    },
    {
        "name": "MultiFixedDecisionLinearIter",
        "placeholder_options": {
            "num_iter": [2, 3],
        },
        "class": IterativeSWMethod,
        "args": {
            "num_iter": HyperparameterPlaceholder("num_iter"),
            "sensitivity_class": MultiFixedDecisionSensitivityMethod,
            "sensitivity_args": {},
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": 1.0,
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
    {
        "name": "MultiFixedDecisionLinearNP",
        "placeholder_options": {},
        "class": NonparametricSWMethod,
        "args": {
            "sensitivity_class": MultiFixedDecisionSensitivityMethod,
            "sensitivity_args": {},
            "weighting_class": SingleEpsilonWeightingMethod,
            "weighting_args": {
                "epsilon": 1.0,
            },
            "predict_class": LinearPredictMethod,
            "predict_args": {},
            "flexible_predict_class": FlexiblePredictMethod,
            "flexible_predict_args": {},
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
            "lmbda": 0.1,
        }
    },
    {
        "name": "RandomWeights",
        "class": RandomWeights,
        "args": {
            "num_weights": 1000,
            "multi": False,
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
    {
        "name": "RandomWeightsMulti",
        "class": RandomWeights,
        "args": {
            "num_weights": 1000,
            "multi": True,
            "predict_class": LinearPredictMethod,
            "predict_args": {},
        },
    },
]

n_range = [1000, 500, 200, 100, 50]
num_test = 10000
num_reps = 32
num_procs = 1
batch_size = 5


shortest_paths_setup = {
    "setup_name": "shortest_paths_setup_rest",
    "environment": {
        "class": ShortestPathEnvironment,
        "args": {}
    },
    "n_range": n_range,
    "num_test": num_test,
    "verbose": False,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "methods": method_list,
    "benchmark_methods": benchmark_list,
    "batch_size": batch_size,
}
