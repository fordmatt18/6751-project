import json
import os
from collections import defaultdict
from multiprocessing import Queue, Process

import numpy as np

from experiment_setups.toy_setups import toy_setup
from utils.hyperparameter_optimization import iterate_placeholder_values, \
    fill_placeholders, fill_global_values
from optimization.lp_solver import LPSolver
# from optimization.dummy_lp_solver import DummyLPSolver as LPSolver

setup_list = [toy_setup]
save_dir = "results_toy"


def main():
    for setup in setup_list:
        run_experiment(setup)


def run_experiment(setup):
    results = []

    n_range = sorted(setup["n_range"], reverse=True)
    num_procs = setup["num_procs"]
    num_reps = setup["num_reps"]
    num_jobs = len(n_range) * num_reps

    if num_procs == 1:
        # run jobs sequentially
        for n in n_range:
            for rep_i in range(setup["num_reps"]):
                results.extend(do_job(setup, n, rep_i))
    else:
        # run jobs in separate processes using queue'd system
        jobs_queue = Queue()
        results_queue = Queue()

        for n in n_range:
            for rep_i in range(setup["num_reps"]):
                jobs_queue.put((setup, n, rep_i))

        procs = []
        for i in range(num_procs):
            p = Process(target=run_jobs_loop, args=(jobs_queue, results_queue))
            procs.append(p)
            jobs_queue.put("STOP")
            p.start()

        num_done = 0
        while num_done < num_jobs:
            results.extend(results_queue.get())
            num_done += 1
        for p in procs:
            p.join()

    # build aggregate results
    aggregate_results = build_aggregate_results(results)
    print(json.dumps(aggregate_results, sort_keys=True, indent=2))

    save_path = os.path.join(save_dir, "%s_results.json" % setup["setup_name"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, "w") as f:
        output = {"results": results, "setup": setup,
                  "aggregate_results": aggregate_results}
        json.dump(output, f, default=lambda c_: c_.__name__,
                  indent=2, sort_keys=True)


def run_jobs_loop(jobs_queue, results_queue):
    for job_args in iter(jobs_queue.get, "STOP"):
        results = do_job(*job_args)
        results_queue.put(results)


def do_job(setup, n, rep_i):
    results = []
    print("setting up scenario for %s setup (n=%d, rep=%d)"
          % (setup["setup_name"], n, rep_i))

    # set up environment, and sample data
    env = setup["environment"]["class"](**setup["environment"]["args"])
    x_train, y_train = env.sample_data(n)
    x_test, y_test = env.sample_data(setup["num_test"])
    verbose = setup["verbose"]
    constraints = env.get_constraints()
    context_dim = env.get_context_dim()
    decision_dim = env.get_decision_dim()
    lp_solver = LPSolver(constraints=constraints, context_dim=context_dim,
                         decision_dim=decision_dim,
                         batch_size=setup["batch_size"])

    # estimate oracle policy value on test data
    z_oracle = lp_solver.solve_lp(y_test)
    oracle_policy_val = float((z_oracle * y_test).sum(1).mean(0))

    if verbose:
        print("")
        print("oracle policy value estimate:", oracle_policy_val)

    # iterate over end-to-end methods
    if verbose:
        print("end-to-end results:")
        print("")
    for method in setup["methods"]:
        placeholder_options = method["placeholder_options"]
        for placeholder_values in iterate_placeholder_values(
                placeholder_options):

            method = fill_global_values(method, setup)
            method = fill_placeholders(method, placeholder_values)

            if verbose:
                print("running %s method under %s setup"
                      " (n=%d, rep=%d)" % (method["name"],
                                           setup["setup_name"], n, rep_i))
                if placeholder_options:
                    print("using placeholder values: %r" %  placeholder_values)

            # set up and fit method
            policy = method["class"](
                lp_solver=lp_solver, context_dim=context_dim,
                decision_dim=decision_dim, **method["args"])
            policy.fit(x_train, y_train)

            # compute performance of method on test data
            z_test = policy.decide(x_test)
            assert lp_solver.all_feasible(z_test)
            pv_estimate = float((z_test * y_test).sum(1).mean(0))

            row = {
                "record_kind": "end-to-end",
                "n": n,
                "pv_estimate": pv_estimate,
                "method": method["name"],
                "oracle_pv": oracle_policy_val,
                "regret": pv_estimate - oracle_policy_val,
                "placeholders": placeholder_values,
                "rep": rep_i,
            }
            results.append(row)
            if verbose:
                print(json.dumps(row, sort_keys=True, indent=2))
                print("")

    # iterate over benchmark methods
    if verbose:
        print("benchmark results:")
        print("")
    for benchmark in setup["benchmark_methods"]:
        policy = benchmark["class"](
            lp_solver=lp_solver, context_dim=context_dim,
            decision_dim=decision_dim, **benchmark["args"])
        policy.fit(x_train, y_train)
        z_test = policy.decide(x_test)
        assert lp_solver.all_feasible(z_test)
        pv_estimate = float((z_test * y_test).sum(1).mean(0))

        row = {
            "record_kind": "benchmark",
            "n": n,
            "pv_estimate": pv_estimate,
            "method": benchmark["name"],
            "oracle_pv": oracle_policy_val,
            "regret": pv_estimate - oracle_policy_val,
            "rep": rep_i,
        }
        results.append(row)
        if verbose:
            print(json.dumps(row, sort_keys=True, indent=2))
            print("")

    return results


def build_aggregate_results(results):
    results_list_collection = defaultdict(list)

    # put together lists of results for each method
    for row in results:
        if row["record_kind"] == "end-to-end":
            # make end-to-end method key
            method_key = row["method"]
            options = row["placeholders"]
            if options:
                options_key = "__".join(["%s=%r" % (k, v)
                                         for k, v in sorted(options.items())])
                method_key = method_key + "__" + options_key

        elif row["record_kind"] == "benchmark":
            method_key = row["method"]
        else:
            raise ValueError("Invalid Record Kind: %s" % row["record_kind"])

        key = (row["n"], method_key)
        results_list_collection[key].append(row["regret"])

    # compute aggregate statistics
    aggregate_results = {}
    for key, results_list in sorted(results_list_collection.items()):
        n, method_key = key
        results_key = "%05d___%s" % (n, method_key)
        regret_array = np.array(results_list)
        mean_regret = float(regret_array.mean())
        std_regret = float(regret_array.std())
        aggregate_results[results_key] = {
            "mean_regret": mean_regret,
            "std_regret": std_regret,
        }
    return aggregate_results


if __name__ == "__main__":
    main()