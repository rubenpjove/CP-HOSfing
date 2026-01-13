import sys
import warnings
import logging
import os
import shutil
from typing import Dict, List, Any
import absl
import absl.logging
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from exps.utils.io_utils import setup_logging, load_config_and_params
from exps.predictors.src.cphos.data import load_dataset, preprocess_dataset
from exps.predictors.src.cphos.models import MLP, seed_everything
from exps.predictors.src.cphos.train import (
    train_one_fold, apply_feature_selection
)
from exps.predictors.src.cphos.infer import load_model_bundle
from exps.utils.results_export import export_experiment_results
from exps.confpred.src.cp_experiment import run_lwcp_one_shot
from exps.confpred.src.utils import setup_single_file_logger, split_and_rename_artifacts
from exps.confpred.src.plotting import (
    plot_alpha_boxplots,
    plot_aggregate_lines,
    plot_ab_comparison,
    METRIC_KEYS,
    LEVELS
)
from exps.confpred.src.aggregation import (
    aggregate_per_alpha,
    aggregate_cross_alpha,
    aggregate_cross_baseline
)
from exps.data_split.src import load_pre_split_dataset


logging.captureWarnings(True)
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)


def load_best_hyperparameters(level: str, artifacts_path: str, logger: logging.Logger) -> tuple:
    """Load best hyperparameters from model artifacts.
    
    Args:
        level: Level name (family, major, or leaf)
        artifacts_path: Path to artifacts directory
        logger: Logger instance
        
    Returns:
        Tuple of (best_params dict, bundle dict)
    """
    logger.info(f"Loading best hyperparameters for {level} level from artifacts...")
    
    bundle = load_model_bundle(level, artifacts_path, logger)
    
    best_params = bundle.get("best_params", {})
    if not best_params:
        raise ValueError(f"No best_params found in artifacts for {level} level")
    
    logger.info(f"Loaded best hyperparameters: {best_params}")
    
    return best_params, bundle


def main():
    resolved_path = sys.argv[2]
    config, input_params = load_config_and_params(resolved_path)

    out_dir = config.get("paths", {}).get("out", ".")
    logger = setup_logging(level=logging.INFO, to_file=True, log_dir=out_dir, to_console=False)
    # Reconfigure to write all logs into a single file named after the logging level
    setup_single_file_logger(logger, out_dir)
    # Flush all handlers to ensure file is created and first message is written immediately
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    logger.info(f"Loading configuration from: {resolved_path}")

    # Seed from params (fallback to 2026)
    seed = int(input_params.get("seed", 2026))
    seed_everything(seed)
    logger.info(f"Random seed set to: {seed}")

    # GPU availability and configuration
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU acceleration available: {gpu_count} GPU(s) detected")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("No GPU available - using CPU")

    # Load pre-split training data and preprocessing metadata
    dataset_caltest_path = input_params.get("dataset_caltest_path")
    maps_path = input_params.get("maps_path")
    if not dataset_caltest_path or not maps_path:
        raise KeyError("Both 'dataset_caltest_path' and 'maps_path' must be defined in input params")

    df, maps = load_pre_split_dataset(logger, dataset_caltest_path, maps_path)
    logger.info(f"Loaded pre-split dataset ({df.shape[0]} rows, {df.shape[1]} columns)")

    # Optional: use only a fraction of the dataset for faster debugging runs
    # Supports input_params["dataset_frac"] in (0,1]; if >1 is passed, it is interpreted as percent
    try:
        frac = None
        if "dataset_frac" in input_params:
            f = float(input_params.get("dataset_frac", 1.0))
            # If a value > 1.0 is provided, interpret it as a percent
            if f > 1.0:
                f = min(100.0, max(0.1, f)) / 100.0
            frac = min(1.0, max(0.001, f))
        if frac is not None and frac < 1.0:
            n_before = len(df)
            df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
            logger.info(f"Using dataset_frac={frac:.3f} (~{frac*100:.1f}%). Rows: {n_before} -> {len(df)}")
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Dataset subsampling skipped due to error: {e}")

    # Debug-mode safety: remove rare classes that would break stratified splits
    if bool(input_params.get("debug", False)):
        try:
            min_count = int(input_params.get("debug_min_class_count", 2))
            if "leaf_idx" in df.columns and min_count > 1:
                counts = df.loc[df["leaf_idx"].notna(), "leaf_idx"].value_counts()
                keep_mask = df["leaf_idx"].isna() | df["leaf_idx"].map(lambda v: counts.get(v, 0) >= min_count)
                before_n = len(df)
                df = df.loc[keep_mask].copy()
                removed_n = before_n - len(df)
                logger.info(
                    f"[debug] Filtered rare leaf classes (min_count={min_count}). "
                    f"Rows: {before_n} -> {len(df)} (removed {removed_n})"
                )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[debug] Rare-class filtering skipped due to error: {e}")

    # Prepare per-level bundles (hyperparameters and feature lists)
    artifacts_path = input_params.get("artifacts_path", out_dir)
    bundle_per_level = {}
    for level in ["family", "major", "leaf"]:
        bundle_per_level[level] = load_model_bundle(level, artifacts_path, logger)

    # Determine alpha sweep and runs
    method = input_params.get("method", None)
    methods = input_params.get("methods", None)
    # List of methods to aggregate outputs for (default to both if not specified)
    if methods is None:
        methods_to_aggregate = [method] if method is not None else ["LwCP", "LoUPCP"]
    else:
        methods_to_aggregate = list(methods) if isinstance(methods, (list, tuple)) else [str(methods)]
    # The one to execute (training/inference) remains single for now; default "LwCP"
    if method is None:
        method = "LwCP"
    
    # Validate method configuration
    valid_methods = {"LwCP", "LoUPCP"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method for execution: {method}. Must be one of {valid_methods}")
    invalid_aggregate = [m for m in methods_to_aggregate if m not in valid_methods]
    if invalid_aggregate:
        raise ValueError(f"Invalid methods in aggregation list: {invalid_aggregate}. Must be subset of {valid_methods}")
    if method not in methods_to_aggregate:
        logger.warning(
            f"Method '{method}' is being executed but not in aggregation list {methods_to_aggregate}. "
            f"Results for this method may not be aggregated."
        )
    logger.info(f"Method configuration: execute={method}, aggregate={methods_to_aggregate}")
    
    alphas = input_params.get("alphas", None)
    if alphas is None or (isinstance(alphas, (list, tuple)) and len(alphas) == 0):
        # Fallback to a single default alpha
        alphas = [0.10]
    if not isinstance(alphas, (list, tuple)):
        alphas = [float(alphas)]
    num_runs = int(input_params.get("num_runs", 1))

    # Per-alpha/run execution (single method execution)
    for alpha in alphas:
        method_exec_dir = os.path.join(out_dir, f"method_{method}")
        os.makedirs(method_exec_dir, exist_ok=True)
        # Convert to float for consistent directory naming (0 -> 0.0, not "0")
        alpha_float = float(alpha)
        alpha_dir_exec = os.path.join(method_exec_dir, f"alpha_{alpha_float}")
        os.makedirs(alpha_dir_exec, exist_ok=True)

        # Run multiple times per alpha
        for run_idx in range(1, num_runs + 1):
            run_dir = os.path.join(alpha_dir_exec, f"run_{run_idx}")
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"Alpha={alpha} Run={run_idx} started -> {run_dir}")

            # Build run-specific params
            input_params_run = dict(input_params)
            input_params_run["alpha"] = alpha_float
            input_params_run["seed"] = int(seed + run_idx)

            if method == "LwCP":
                # Construct LoUP-CP directory if LoUPCP is in methods_to_aggregate
                out_dir_LoUPCP = None
                if "LoUPCP" in methods_to_aggregate:
                    method_LoUPCP_exec_dir = os.path.join(out_dir, f"method_LoUPCP")
                    os.makedirs(method_LoUPCP_exec_dir, exist_ok=True)
                    alpha_dir_LoUPCP_exec = os.path.join(method_LoUPCP_exec_dir, f"alpha_{alpha_float}")
                    os.makedirs(alpha_dir_LoUPCP_exec, exist_ok=True)
                    out_dir_LoUPCP = os.path.join(alpha_dir_LoUPCP_exec, f"run_{run_idx}")
                    os.makedirs(out_dir_LoUPCP, exist_ok=True)
                    
                    # Validate that out_dir_LoUPCP structure matches expected aggregation paths
                    expected_base = os.path.join(out_dir, f"method_LoUPCP", f"alpha_{alpha_float}")
                    if not out_dir_LoUPCP.startswith(expected_base):
                        logger.warning(
                            f"out_dir_LoUPCP structure may not match aggregation expectations: "
                            f"expected to start with {expected_base}, got {out_dir_LoUPCP}"
                        )
                    else:
                        logger.debug(f"Validated out_dir_LoUPCP structure: {out_dir_LoUPCP}")
                
                run_lwcp_one_shot(
                    logger=logger,
                    df=df,
                    maps=maps,
                    input_params=input_params_run,
                    out_dir=run_dir,
                    bundle_per_level=bundle_per_level,
                    out_dir_LoUPCP=out_dir_LoUPCP,
                )
            else:
                raise ValueError(f"Unsupported method for execution: {method}")

            # Split and rename artifacts into method-specific trees with full-trace filenames
            split_and_rename_artifacts(
                run_dir, out_dir, alpha_float, run_idx, method, methods_to_aggregate, logger
            )

    # Aggregations and plots per method
    levels = LEVELS
    metric_keys = METRIC_KEYS

    for m in methods_to_aggregate:
        method_dir = os.path.join(out_dir, f"method_{m}")
        os.makedirs(method_dir, exist_ok=True)

        # Per-alpha aggregation for this method
        for alpha in alphas:
            # Convert to float for consistent directory naming
            alpha_float = float(alpha)
            # Aggregate per alpha
            aggregate_per_alpha(
                out_dir=out_dir,
                baseline=m,
                alpha=alpha_float,
                num_runs=num_runs,
                levels=levels,
                metric_keys=metric_keys,
                logger=logger
            )
            
            # Per-alpha plotting (boxplots) for this method
            alpha_dir = os.path.join(method_dir, f"alpha_{alpha_float}")
            alpha_agg_dir = os.path.join(alpha_dir, f"{m}_alpha_{alpha_float}_aggregated")
            if os.path.exists(alpha_agg_dir):
                try:
                    plot_alpha_boxplots(alpha_dir=alpha_agg_dir)
                except (IOError, OSError, ValueError) as e:
                    logger.warning(f"[{m}] Per-alpha plotting failed for alpha={alpha}: {e}")

        # Cross-alpha aggregation for this method
        aggregate_cross_alpha(
            out_dir=out_dir,
            baseline=m,
            alphas=alphas,
            levels=levels,
            metric_keys=metric_keys,
            logger=logger
        )
        
        # Cross-alpha plotting
        bl_agg_dir = os.path.join(method_dir, f"{m}_aggregated")
        cross_csv = os.path.join(bl_agg_dir, "metrics_across_alphas.csv")
        if os.path.exists(cross_csv):
            try:
                plot_aggregate_lines(out_dir=bl_agg_dir, cross_alpha_csv=cross_csv)
            except (IOError, OSError, ValueError) as e:
                logger.warning(f"[{m}] Aggregate plotting failed: {e}")

    # Cross-method aggregation
    aggregate_cross_baseline(
        out_dir=out_dir,
        baselines=methods_to_aggregate,
        logger=logger
    )
    
    # Cross-method plotting
    methods_dir = os.path.join(out_dir, "methods_aggregated")
    combined_csv = os.path.join(methods_dir, "metrics_across_alphas_by_method.csv")
    if os.path.exists(combined_csv):
        try:
            plot_ab_comparison(ab_dir=methods_dir, combined_csv=combined_csv)
        except (IOError, OSError, ValueError) as e:
            logger.warning(f"Methods aggregated plotting failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("main").exception("Fatal error in main")
        raise
