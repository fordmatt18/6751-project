# ORIE-6751 Final Project

Code repo for running experiments in ORIE 6751 final project. 

Title: End-to-end Stochastic Contextual Linear Optimization by Retargeting

Authors: Andrew Bennett and Matthew Ford

Abstract: We investigate a novel approach for doing end-to-end stochastic
contextual linear optimization by retargeting the (weighted) population that
the prediction model is trained on, taking into account the downstream
decision task. We propose multiple approaches for doing this by computing
weights based on the sensitivity of the downstream task to incorrect
prediction for each data point. We show that the calculation of these weights
can be reduced to solving some tractable LPs, and that these approaches result
 in very competitive empirical decision-making performance.

All experiments were run using Python 3.8.5, using environment summarized by
requirements.txt file. In order to run main experiments, execute the script
run_experiments.py. Variables at the top this script can be edited to change
where results should be saved, which experiments should be run, etc.

Certain experiment parameters which may be important for
computational performance but shouldn't change
results (e.g. number of parallel processes used) can be modified by editing
the experiment setup files, which for our main experiments are:
- experiment_setups/random_resource_setup.py 
- experiment_setups/shortest_paths.setup.py.

After running experiments, plots as in the paper can be created by executing
plot_results.py. The variables at the top of this script will need to be
edited according to where results are saved, which experiment result file(s)
plots are to be created from, desired title of plots etc.