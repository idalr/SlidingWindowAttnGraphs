import argparse
import re
import os
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind, f_oneway, ttest_rel  # for independent runs comparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

FILENAME_CONVENTION = {
    "method": 0,
    "window": 1,
    "type_graph": -2
} # as in GAT result file name, separated by "_", e.g., NoTemp_w30_0.813_2GAT_max_unified


def parse_filename(filename, convention):
    parts = filename.split("_")
    return {k: parts[i] for k, i in convention.items()}


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
    info = parse_filename(filename, FILENAME_CONVENTION)
    method = info["method"]
    window = info["window"]
    type_graph = info["type_graph"]

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


# ------------------------------------------------
# Compute ANOVA and Tukey
# ------------------------------------------------

def get_valid_config_combinations(config_cols):
    return [
        list(combo)
        for r in range(1, len(config_cols) + 1)
        for combo in combinations(config_cols, r)
    ]


def compute_tukey(df, config_cols, metric):
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

        # run Tukey if significant
        if is_significant:
            tukey_df = compute_tukey(df, config_cols, metric)
            tukey_df["config"] = "+".join(config_cols)
            tukey_df["metric"] = metric
            tukey_results.append(tukey_df)

    return pd.DataFrame(results), tukey_results


def build_config_label(df, analyze_configs):
    return df[analyze_configs].astype(str).agg("_".join, axis=1)


def compute_baselines_ttest(baselines_path, results_df, analyze_configs, use_paired=False, alpha=0.05):
    baselines = pd.read_csv(baselines_path, usecols=["Model", "Test score"])
    baselines['window'] = baselines['Model'].apply(lambda x: x.split('_')[2]) #######################################################
    baselines_df = baselines.rename(columns={"Test score": "acc"}).drop(columns=['Model'])

    rest_configs = analyze_configs.copy()
    rest_configs = [c for c in rest_configs if c != "window"] #########################
    results_df["config"] = build_config_label(results_df, rest_configs)

    results = []
    for method in results_df["window"].unique():

        base_scores = baselines_df[baselines_df["window"] == method]["acc"].values
        if len(base_scores) == 0:
            print(f"{method} → no baseline found, skipping")
            continue

        res_model_df = results_df[results_df["window"] == method]

        print(f"{method}\nAverage baseline accuracy: {np.mean(base_scores)}")
        print(f"Average GAT accuracy: {np.mean(res_model_df['acc'])}")

        for config in res_model_df["config"].unique():

            res_model_scores = res_model_df[res_model_df["config"] == config]["acc"].values

            # safety check
            n = min(len(res_model_scores), len(base_scores))
            res_model_scores = res_model_scores[:n]
            base_scores = base_scores[:n]

            if use_paired:
                stat, p_val = ttest_rel(base_scores, res_model_scores)
            else: # t-test (independent; safer unless runs are aligned)
                stat, p_val = ttest_ind(base_scores, res_model_scores, equal_var=False)
            is_significant = p_val < alpha

            results.append({
                "method": method,
                "config": config,
                "p_value": p_val,
                "significant": is_significant,
            })

    # multiple comparison correction per model (optional but recommended)
    final = []

    for model in set(r["method"] for r in results):
        model_res = [r for r in results if r["method"] == model]
        p_vals = [r["p_value"] for r in model_res]

        _, p_corr, _, _ = multipletests(p_vals, method="holm")

        for r, pc in zip(model_res, p_corr):
            r["p_value"] = pc
            final.append(r)

    return pd.DataFrame(final)


def parse_dataset_name(folder_path):
    valid_datasets = {"AX", "arXiv", "BBC", "HND-windows", "GR", "GovReports"} ######################################################

    # match dataset token as a standalone path/token chunk
    match = re.search(r'(?<![A-Za-z0-9])(AX|arXiv|BBC|HND-windows|GR|GovReports)(?![A-Za-z0-9])', folder_path) #####################################

    if not match:
        raise ValueError(
            f"Could not extract dataset name from path: {folder_path!r}. "
            f"Expected one of: {sorted(valid_datasets)}"
        )

    dataset_name = match.group(1)

    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name extracted: {dataset_name}")

    if dataset_name == "BBC":
        num_class = 11
    elif dataset_name == "AX" or dataset_name == "arXiv":
        num_class = 11
    else:
        num_class = 2

    return dataset_name, num_class


def main_run(results_path, analyze_configs, analyze_cols, baselines_path, save_files=True):

    dataset_name, num_class = parse_dataset_name(results_path)
    print(f"...Analyzing dataset: {dataset_name}")

    all_runs = []
    # Parse all files
    for file in os.listdir(results_path):
        if file.endswith("unified.txt"):
            with open(os.path.join(results_path, file), "r") as f:
                text = f.read()
                all_runs.extend(parse_runs(text, file, num_class))
    # Convert to DataFrame
    df_runs = pd.DataFrame(all_runs)

    if save_files:
        df_runs.to_csv(os.path.join(results_path, dataset_name + '_GAT_results.csv'), index=False)
        print("Saved parsed GNN results to file.")

    if baselines_path is not None:
        ttest_baselines_df = compute_baselines_ttest(baselines_path, df_runs, analyze_configs)
        if save_files:
            if not os.path.exists("GNN_Analyses"):
                os.mkdir("GNN_Analyses")
            ttest_baselines_df.to_csv(f"./GNN_Analyses/{dataset_name}_baselines_ttest.csv", index=False)
            print("Saved baselines ttest to file.")


    if analyze_configs is not None:

        all_anova = []
        all_tukey = []
        for cfg in get_valid_config_combinations(analyze_configs):
            anova_res, tukey_res = compute_anova_by_config(df_runs, cfg, analyze_cols)
            all_anova.append(anova_res)
            # if any significant, extend tukey_res
            if tukey_res:
                all_tukey.extend(tukey_res)
        anova_df = pd.concat(all_anova, ignore_index=True)

        # build tukey_df
        if all_tukey:
            tukey_df = pd.concat(all_tukey, ignore_index=True)
        else:
            tukey_df = pd.DataFrame()

        # save analyses to files
        if not os.path.exists("GNN_Analyses"):
            os.mkdir("GNN_Analyses")
        anova_df.to_csv(f"./GNN_Analyses/{dataset_name}_anova_results.csv", index=False)
        tukey_df.to_csv(f"./GNN_Analyses/{dataset_name}_tukey_results.csv", index=False) if tukey_df is not None else None
        print("Saved ANOVA to file.")
        print("Saved Tukey to file.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="path to the input directory",
    )
    arg_parser.add_argument(
        "--analyze_configs",
        nargs="+",
        type=str,
        default=None,
        help="configs for comparing results: ['method', 'type_graph', 'layers', 'hidden_dim']",
    )
    arg_parser.add_argument(
        "--analyze_cols",
        nargs="+",
        type=str,
        default=None,
        help="targeted metrics for comparing results: ['acc', 'f1_macro', 'train_time']",
    )
    arg_parser.add_argument(
        "--baselines_path",
        type=str,
        default=None,
        help="path to the baseline MHA csv",
    )
    arg_parser.add_argument(
        "--save_files",
        action="store_true",
        help="add to save the parsed result file",
    )
    args = arg_parser.parse_args()

    # parse analyse_configs
    if args.analyze_configs is None:
        analyze_configs = None
    elif len(args.analyze_configs) == 1:
        analyze_configs = args.analyze_configs[0]
    else:
        analyze_configs = args.analyze_configs

    # parse analyze_cols
    if args.analyze_cols is None:
        analyze_cols = None
    elif len(args.analyze_cols) == 1:
        analyze_cols = args.analyze_cols[0]
    else:
        analyze_cols = args.analyze_cols

    main_run(**vars(args))

# p-adj → corrected p-value
# reject = True → significant difference
# meandiff → how much better/worse
