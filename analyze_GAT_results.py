import re
import os
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind, f_oneway  # for independent runs comparison

folder_path = "./GNN_Results/Classifier_Results/BBC"
dataset_name = "BBC"
num_class = 5


def parse_runs(text, filename, num_class=2):
    """
    Extract per-run metrics from the log text, handling any number of classes.

    Args:
        text: str, log file content
        filename: str, name of the log file
        num_class: int, number of classes in F1 per class
    Returns:
        list of dicts with per-run metrics
    """
    method, window, *_, type_graph, _ = filename.split("_")
    runs = []

    # Split blocks by config
    blocks = re.split(r"TRAINING MODELS SETTING", text)

    for block in blocks[1:]:  # skip first
        # Extract config
        layers = re.search(r"#LAYERS:\s*(\d+)", block)
        hidden = re.search(r"HIDDEN DIM:\s*(\d+)", block)
        layers = int(layers.group(1)) if layers else None
        hidden = int(hidden.group(1)) if hidden else None

        # Find all runs
        run_matches = re.findall(r"----------- Run #(\d+)-----------(.*?)Training \+ Testing time", block, re.DOTALL)

        for run_num, run_text in run_matches:
            run_data = {"file": filename,
                        "method": method,
                        "window": window,
                        "type_graph": type_graph,
                        "layers": layers,
                        "hidden_dim": hidden,
                        "run": int(run_num)}

            # Extract metrics
            acc = re.search(r"Acc Test:\s*tensor\(([\d\.]+)\)", run_text)
            f1_macro = re.search(r"F1-macro Test:\s*tensor\(([\d\.]+)\)", run_text)
            f1_classes = re.search(r"F1 for each class:\s*tensor\(\[([\d\.,\s]+)\]\)", run_text)
            train_time = re.search(r"Training time:\s*([\d\.]+)", run_text)
            total_time = re.search(r"Training \+ Testing time:\s*([\d\.]+)", run_text)

            if acc: run_data["acc"] = float(acc.group(1))
            if f1_macro: run_data["f1_macro"] = float(f1_macro.group(1))

            # Handle any number of classes
            if f1_classes:
                vals = list(map(float, f1_classes.group(1).split(',')))
                for i in range(num_class):
                    key = f"f1_class_{i}"
                    run_data[key] = vals[i] if i < len(vals) else None

            if train_time: run_data["train_time"] = float(train_time.group(1))
            if total_time: run_data["total_time"] = float(total_time.group(1))

            runs.append(run_data)

    return runs


def parse_blocks(text, filename, num_class=2):
    blocks = re.findall(r"\*{48}\n(.*?)\n\*{48}", text, re.DOTALL)
    method, window, *_, type_graph, _ = filename.split("_")
    results = []

    for block in blocks:
        data = {"file": filename,
                "method": method,
                "window": window,
                "type_graph": type_graph,
                }

        def extract(pattern):
            match = re.search(pattern, block)
            return match.groups() if match else None

        layers = extract(r"N_LAYERS:\s*(\d+)")
        hidden = extract(r"HIDDEN DIM_FEATURES:\s*(\d+)")
        acc = extract(r"Test Acc:\s*([\d\.]+)\s*-- std:\s*([\d\.]+)")
        f1 = extract(r"Test F1-macro:\s*([\d\.]+)\s*-- std:\s*([\d\.]+)")
        f1_class = extract(r"Test F1 per class:\s*\[([\d\.\s]+)\]")
        train_time = extract(r"Training time:\s*([\d\.]+)\s*-- std:\s*([\d\.]+)")
        total_time = extract(r"Total time:\s*([\d\.]+)\s*-- std:\s*([\d\.]+)")

        if layers:
            data["layers"] = int(layers[0])
        if hidden:
            data["hidden_dim"] = int(hidden[0])
        if acc:
            data["acc_mean"] = float(acc[0])
            data["acc_std"] = float(acc[1])
        if f1:
            data["f1_mean"] = float(f1[0])
            data["f1_std"] = float(f1[1])

        # Generalized multi-class handling
        if f1_class:
            vals = list(map(float, f1_class[0].split()))
            for i in range(num_class):
                key = f"f1_class_{i}"
                data[key] = vals[i] if i < len(vals) else None

        if train_time:
            data["train_time_mean"] = float(train_time[0])
            data["train_time_std"] = float(train_time[1])
        if total_time:
            data["total_time_mean"] = float(total_time[0])
            data["total_time_std"] = float(total_time[1])

        results.append(data)

    return results

all_results = []

# Loop through files
for file in os.listdir(folder_path):
    if file.endswith("unified.txt"):
        with open(os.path.join(folder_path, file), "r") as f:
            text = f.read()
            all_results.extend(parse_blocks(text, file, num_class))

# Convert to DataFrame
df = pd.DataFrame(all_results)
# Save
df.to_csv(f"./GNN_Analyses/{dataset_name}/merged_results.csv", index=False)
print(df.head())

# # for AVG
# ## Wilcoxon
# from scipy.stats import wilcoxon
# wilcoxon(group1, group2)
# ## Confidence intervals
# import math
# def ci(mean, std, n=5):
#     margin = 1.96 * (std / math.sqrt(n))
#     return (mean - margin, mean + margin)
# ci1 = ci(0.744, 0.006)
# ci2 = ci(0.755, 0.008)
# print(ci1, ci2) # If intervals don’t overlap → likely significant difference
# ## Cohen's - effect size
# import numpy as np
# def cohens_d(a, b):
#     return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2) # between 0 and 1
# df.groupby("hidden_dim")["f1_mean"].mean()
# df.groupby("layers")["f1_mean"].mean()

all_runs = []

# Parse all files
for file in os.listdir(folder_path):
    if file.endswith("unified.txt"):
        with open(os.path.join(folder_path, file), "r") as f:
            text = f.read()
            all_runs.extend(parse_runs(text, file, num_class))

# Convert to DataFrame
df = pd.DataFrame(all_runs)
df.to_csv(f"./GNN_Analyses/{dataset_name}/per_run_results.csv", index=False)
print("Saved per-run results.")
print(df.head())

# ------------------------------------------------
# Compute ANOVA and Tukey
# ------------------------------------------------
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

def run_tukey_df(df, config_cols, metric):
    """
    Run Tukey HSD and return a clean DataFrame.
    Supports multi-column configs.
    """
    data = df[config_cols + [metric]].dropna().copy()

    # Create combined group label
    data["group"] = data[config_cols].astype(str).agg("_".join, axis=1)

    tukey = pairwise_tukeyhsd(
        endog=data[metric],
        groups=data["group"],
        alpha=0.05
    )

    # Convert to DataFrame
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    return tukey_df

from scipy.stats import f_oneway

def compute_anova_by_config(df, config_cols, cols_to_compare, alpha=0.05):
    results = []
    tukey_results = []

    grouped = df.groupby(config_cols)

    for metric in cols_to_compare:
        groups = []
        group_labels = []

        for name, group in grouped:
            values = group[metric].dropna()
            if len(values) > 0:
                groups.append(values)
                group_labels.append(
                    "_".join(map(str, name)) if isinstance(name, tuple) else name
                )

        if len(groups) < 2:
            continue

        stat, p_val = f_oneway(*groups)

        is_significant = p_val < alpha

        results.append({
            "config": "+".join(config_cols),
            "metric": metric,
            "p_value": p_val,
            "significant": is_significant,
            "num_groups": len(groups),
            "groups": group_labels
        })

        # 🔥 NEW: run Tukey if significant
        if is_significant:
            tukey_df = run_tukey_df(df, config_cols, metric)
            tukey_df["config"] = "+".join(config_cols)
            tukey_df["metric"] = metric
            tukey_results.append(tukey_df)

    return pd.DataFrame(results), tukey_results


def get_valid_config_combinations(config_cols):
    return [
        list(combo)
        for r in range(1, len(config_cols) + 1)
        for combo in combinations(config_cols, r)
    ]

# df: per-run results DataFrame
config_cols = ["method", "type_graph", "layers", "hidden_dim"]  # columns identifying configs
cols_to_compare = ["acc", "f1_macro", "train_time"] #, "f1_class_0", "f1_class_1"]  # metrics

all_anova = []
all_tukey = []

for cfg in get_valid_config_combinations(config_cols):
    anova_res, tukey_res = compute_anova_by_config(df, cfg, cols_to_compare)

    all_anova.append(anova_res)

    if tukey_res:
        all_tukey.extend(tukey_res)

anova_df = pd.concat(all_anova, ignore_index=True)

if all_tukey:
    tukey_df = pd.concat(all_tukey, ignore_index=True)
else:
    tukey_df = pd.DataFrame()

# Save
anova_df.to_csv(f"./GNN_Analyses/{dataset_name}/anova_results.csv", index=False)
tukey_df.to_csv(f"./GNN_Analyses/{dataset_name}/tukey_results.csv", index=False)

print("ANOVA results:")
print(anova_df.head())

print("\nTukey results:")
print(tukey_df.head())

# p-adj → corrected p-value
# reject = True → significant difference
# meandiff → how much better/worse
