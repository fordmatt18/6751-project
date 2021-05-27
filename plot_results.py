import json
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

import pandas as pd
import seaborn as sns


path_list = ["results_random_resource/random_resource_setup_results.json"]
baseline = "PredictThenOptimizeLinear"
scenario_name = "Random Resource Allocation"
save_path = "results_random_resource/random_resource_results.pdf"
save_path_bar = "results_random_resource/random_resource_results_bar.pdf"

method_prefix_map = {
    "Fixed": "Fixed Decision",
    "Variable": "Variable Decision",
    "Random": "Random Weights",
    "Maximum": "Maximum Suboptimality",
    "BEST-GLOBAL": "Best Sensitivity",
    "SPO+": "SPO+",
    baseline: "Predict Then Optimize",
}

prefix_order = ["BEST-GLOBAL", "Fixed", "Variable", "Maximum", "Random", "SPO+"]
method_order = [method_prefix_map[p_] for p_ in prefix_order]


def main(path_list):
    regret_dict = defaultdict(float)
    best_performance_dict = defaultdict(lambda: float("inf"))
    baseline_regret_dict = {}
    results = []
    for path in path_list:
        with open(path) as f:
            results.extend(json.load(f)["results"])
    for row in results:
        method = row["method"]
        for prefix, name in method_prefix_map.items():
            if method.startswith(prefix) or method.startswith("Multi" + prefix):
                key = row["n"], row["rep"]
                regret = row["regret"]
                performance = row["train_pv"]
                if prefix == baseline:
                    baseline_regret_dict[key] = regret
                elif performance < best_performance_dict[(name, key)]:
                    best_performance_dict[(name, key)] = performance
                    regret_dict[(name, key)] = regret

    comp_rows = []
    for n in sorted(set(n_ for n_, _ in baseline_regret_dict.keys())):
        print("n=%d" % n)
        comp_PTO = [regret_dict[("Best Sensitivity", (n, i))]
                    < baseline_regret_dict[(n, i)]
                    for i in range(32)]
        # print(comp_PTO)
        ours_vs_pto = sum(comp_PTO) / 32
        print("percentage ours better than PTO", ours_vs_pto)
        comp_SPO = [regret_dict[("Best Sensitivity", (n, i))]
                    < regret_dict[("SPO+", (n, i))]
                    for i in range(32)]
        ours_vs_spo = sum(comp_SPO) / 32
        print("percentage ours better than SPO+", ours_vs_spo)
        comp_SPO_PTO = [regret_dict[("SPO+", (n, i))]
                        < baseline_regret_dict[(n, i)]
                        for i in range(32)]
        spo_vs_pto = sum(comp_SPO_PTO) / 32
        print("percentage SPO+ better than PTO", spo_vs_pto)
        print("")
        comp_rows.append({"n": n, "perc": ours_vs_pto,
                          "Comparison": "Ours vs Predict Then Optimize"})
        comp_rows.append({"n": n, "perc": ours_vs_spo,
                          "Comparison": "Ours vs SPO+"})
        comp_rows.append({"n": n, "perc": spo_vs_pto,
                          "Comparison": "SPO+ vs Predict Then Optimize"})

    comp_df = pd.DataFrame(comp_rows)
    fig, ax = plt.subplots()
    plot = sns.barplot(x="n", y="perc", hue="Comparison", data=comp_df)
    ax.set_title("%s Scenario" % scenario_name, fontsize=18)
    ax.set_xlabel("Training Set Size", fontsize=16)
    ax.set_ylabel("Fraction regret(A) < regret(B)", fontsize=16)
    plot.legend(prop={'size': 12}, loc="upper right",
                bbox_to_anchor=(0.975, 0.30),
                framealpha=1.0)
    fig.savefig(save_path_bar, bbox_inches="tight")
    # fig.savefig(save_path_bar)

    plot_rows = []
    for (name, key), regret in regret_dict.items():
        n, rep = key
        baseline_regret = baseline_regret_dict[key]
        rrr = (baseline_regret - regret) / baseline_regret * 100.0
        row = {"n": n, "rep": rep, "rel_regret_reduction": rrr,
               "Method": name}
        plot_rows.append(row)

    plot_df = pd.DataFrame(plot_rows)

    fig, ax = plt.subplots()
    # ax.set(xscale="log")

    plot = sns.lineplot(x="n", y="rel_regret_reduction", hue="Method",
                        style="Method", data=plot_df, ci=95,
                        hue_order=method_order, style_order=method_order)

    ax.set_title("%s Scenario" % scenario_name, fontsize=18)
    ax.set_xlabel("Training Set Size", fontsize=16)
    ax.set_ylabel("Relative Regret Reduction (%)", fontsize=16)
    # if f_name.split("_")[0] == "quadratic":
    #     ax.set_ylim(-0.15, 0.15)
    # else:
    #     ax.set_ylim(-0.05, 0.05)
    # ax.set_ylim(-25.0, 100.0)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    # plot.legend(prop={'size': 26}, loc="upper right",
    #             bbox_to_anchor=(0.275, 1.00))
    # fig.subplots_adjust(bottom=0.16, right=0.86)
    fig.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
        main(path_list)
