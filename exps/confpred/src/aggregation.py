import os
import logging
import pandas as pd
from typing import List, Dict, Any


def aggregate_per_alpha(
    out_dir: str,
    baseline: str,
    alpha: Any,
    num_runs: int,
    levels: List[str],
    metric_keys: List[str],
    logger: logging.Logger,
    input_dir: str = None
) -> None:
    """Collects metrics from runs for a given baseline/alpha, aggregates per level, saves CSV files.
    
    Args:
        out_dir: Root output directory (where to write aggregations)
        baseline: Baseline name (LwCP or LoUPCP)
        alpha: Alpha value (can be string or numeric)
        num_runs: Number of runs to aggregate
        levels: List of hierarchy levels (family, major, leaf)
        metric_keys: List of metric keys to aggregate
        logger: Logger instance
        input_dir: Root input directory (where to read run results from). If None, uses out_dir.
    """
    # Use input_dir if provided, otherwise use out_dir for both reading and writing
    read_dir = input_dir if input_dir is not None else out_dir
    
    # Normalize alpha to float for consistent directory naming
    alpha_float = float(alpha)
    # Try both alpha_0 and alpha_0.0 for backward compatibility when alpha is 0
    alpha_dir_variants = []
    if alpha_float == 0.0:
        alpha_dir_variants = [
            os.path.join(read_dir, f"method_{baseline}", "alpha_0"),
            os.path.join(read_dir, f"method_{baseline}", "alpha_0.0")
        ]
    else:
        alpha_dir_variants = [
            os.path.join(read_dir, f"method_{baseline}", f"alpha_{alpha_float}")
        ]
    
    # Find the actual alpha directory that exists
    source_alpha_dir = None
    for variant in alpha_dir_variants:
        if os.path.exists(variant):
            source_alpha_dir = variant
            break
    
    if source_alpha_dir is None:
        logger.warning(
            f"[{baseline}] Alpha directory not found for alpha={alpha}. "
            f"Checked: {alpha_dir_variants}"
        )
        return
    
    # Destination aggregation location for this method/alpha (always in output_dir)
    method_dir = os.path.join(out_dir, f"method_{baseline}")
    os.makedirs(method_dir, exist_ok=True)
    # Use alpha_float for consistent directory naming (0 -> 0.0, not "0")
    alpha_dir = os.path.join(method_dir, f"alpha_{alpha_float}")
    os.makedirs(alpha_dir, exist_ok=True)
    alpha_agg_dir = os.path.join(alpha_dir, f"{baseline}_alpha_{alpha_float}_aggregated")
    os.makedirs(alpha_agg_dir, exist_ok=True)

    for level in levels:
        rows = []
        for run_idx in range(1, num_runs + 1):
            run_dir = os.path.join(source_alpha_dir, f"run_{run_idx}")
            
            # Check if run directory exists
            if not os.path.exists(run_dir):
                logger.warning(
                    f"[{baseline}] Run directory not found for alpha={alpha}, run={run_idx}, level={level}: {run_dir}"
                )
                continue
            
            # Try multiple filename variants (alpha_0 vs alpha_0.0, and original alpha value)
            csv_candidates = []
            json_candidates = []
            
            # New full-trace filenames with alpha_float (for consistency)
            csv_candidates.append(os.path.join(run_dir, f"{baseline}_alpha_{alpha_float}_run_{run_idx}_cp_{level}_summary.csv"))
            json_candidates.append(os.path.join(run_dir, f"{baseline}_alpha_{alpha_float}_run_{run_idx}_cp_{level}_summary.json"))
            
            # New full-trace filenames with original alpha value (for backward compatibility)
            if str(alpha) != str(alpha_float):
                csv_candidates.append(os.path.join(run_dir, f"{baseline}_alpha_{alpha}_run_{run_idx}_cp_{level}_summary.csv"))
                json_candidates.append(os.path.join(run_dir, f"{baseline}_alpha_{alpha}_run_{run_idx}_cp_{level}_summary.json"))
            
            # Backward-compatible legacy names
            csv_legacy = os.path.join(run_dir, f"cp_{level}_summary.csv") if baseline == "LwCP" else os.path.join(run_dir, f"b_cp_{level}_summary.csv")
            json_legacy = os.path.join(run_dir, f"cp_{level}_summary.json") if baseline == "LwCP" else os.path.join(run_dir, f"b_cp_{level}_summary.json")
            csv_candidates.append(csv_legacy)
            json_candidates.append(json_legacy)

            # Prefer CSV if it already contains metric columns
            csv_found = False
            for csv_path in csv_candidates:
                if os.path.exists(csv_path):
                    try:
                        df_row = pd.read_csv(csv_path)
                        if any(k in df_row.columns for k in metric_keys):
                            df_row["run"] = run_idx
                            rows.append(df_row)
                            csv_found = True
                            break
                    except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError) as e:
                        logger.debug(
                            f"[{baseline}] Failed to read CSV {csv_path} for alpha={alpha}, run={run_idx}, level={level}: {e}"
                        )
                        continue
            
            if csv_found:
                continue

            # JSON read
            json_found = False
            for json_path in json_candidates:
                if os.path.exists(json_path):
                    try:
                        j = pd.read_json(json_path, typ="series")
                        metrics = j.get("metrics", {})
                        # Only proceed if metrics dict exists and is not empty
                        if metrics and isinstance(metrics, dict):
                            rec = {k: metrics.get(k, float("nan")) for k in metric_keys}
                            rec.update({
                                "level": level,
                                "alpha": float(alpha),
                                "run": run_idx,
                            })
                            rows.append(pd.DataFrame([rec]))
                            json_found = True
                            break
                    except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError, ValueError) as e:
                        logger.debug(
                            f"[{baseline}] Failed to read JSON {json_path} for alpha={alpha}, run={run_idx}, level={level}: {e}"
                        )
                        continue
            
            if json_found:
                continue

            # If neither provided usable metrics, log and skip this run
            logger.warning(
                f"[{baseline}] No usable metrics found for alpha={alpha}, run={run_idx}, level={level}. "
                f"Checked: CSV ({', '.join(csv_candidates)}), JSON ({', '.join(json_candidates)})"
            )
            continue

        if rows:
            try:
                df_all = pd.concat(rows, ignore_index=True)
                runs_metrics_csv = os.path.join(alpha_agg_dir, f"runs_metrics_{level}.csv")
                df_all.to_csv(runs_metrics_csv, index=False)
                agg = {}
                for key in metric_keys:
                    if key in df_all.columns:
                        agg[f"{key}_mean"] = float(df_all[key].mean())
                        agg[f"{key}_std"] = float(df_all[key].std(ddof=1)) if len(df_all[key]) > 1 else float("nan")
                agg_df = pd.DataFrame([agg])
                out_csv = os.path.join(alpha_agg_dir, f"aggregated_results_{level}.csv")
                agg_df.to_csv(out_csv, index=False)
                logger.info(f"[{baseline}] Saved aggregated metrics for alpha={alpha}, level={level} -> {out_csv}")
            except (IOError, OSError, ValueError, KeyError) as e:
                logger.error(
                    f"[{baseline}] Failed to aggregate metrics for alpha={alpha}, level={level}: {e}"
                )
        else:
            logger.warning(
                f"[{baseline}] No data collected for alpha={alpha}, level={level}. "
                f"Skipping aggregation for this level."
            )


def aggregate_cross_alpha(
    out_dir: str,
    baseline: str,
    alphas: List[Any],
    levels: List[str],
    metric_keys: List[str],
    logger: logging.Logger
) -> None:
    """Aggregates metrics across alphas for a baseline, creates cross-alpha CSV.
    
    Args:
        out_dir: Root output directory
        baseline: Baseline name (LwCP or LoUPCP)
        alphas: List of alpha values
        levels: List of hierarchy levels (family, major, leaf)
        metric_keys: List of metric keys to aggregate
        logger: Logger instance
    """
    method_dir = os.path.join(out_dir, f"method_{baseline}")
    cross_rows = []
    for alpha in alphas:
        # Convert to float for consistent directory naming
        alpha_float = float(alpha)
        alpha_dir = os.path.join(method_dir, f"alpha_{alpha_float}", f"{baseline}_alpha_{alpha_float}_aggregated")
        for level in levels:
            agg_path = os.path.join(alpha_dir, f"aggregated_results_{level}.csv")
            if os.path.exists(agg_path):
                try:
                    agg_df = pd.read_csv(agg_path)
                except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError) as e:
                    logger.warning(
                        f"[{baseline}] Failed to read aggregated results for alpha={alpha}, level={level}: {e}"
                    )
                    continue
                for key in metric_keys:
                    mean_col = f"{key}_mean"
                    std_col = f"{key}_std"
                    if mean_col in agg_df.columns:
                        try:
                            cross_rows.append({
                                "alpha": float(alpha),
                                "level": level,
                                "metric": key,
                                "mean": float(agg_df.iloc[0][mean_col]),
                                "std": float(agg_df.iloc[0].get(std_col, float("nan"))),
                            })
                        except (ValueError, KeyError, IndexError) as e:
                            logger.warning(
                                f"[{baseline}] Failed to extract metric {key} for alpha={alpha}, level={level}: {e}"
                            )
            else:
                logger.debug(
                    f"[{baseline}] Aggregated results file not found for alpha={alpha}, level={level}: {agg_path}"
                )
    if cross_rows:
        try:
            cross_df = pd.DataFrame(cross_rows)
            bl_agg_dir = os.path.join(method_dir, f"{baseline}_aggregated")
            os.makedirs(bl_agg_dir, exist_ok=True)
            cross_csv = os.path.join(bl_agg_dir, "metrics_across_alphas.csv")
            cross_df.to_csv(cross_csv, index=False)
            logger.info(f"[{baseline}] Saved cross-alpha metrics -> {cross_csv}")
        except (IOError, OSError, ValueError) as e:
            logger.error(f"[{baseline}] Failed to save cross-alpha metrics: {e}")
    else:
        logger.warning(f"[{baseline}] No cross-alpha data collected. Skipping cross-alpha aggregation.")


def aggregate_cross_baseline(
    out_dir: str,
    baselines: List[str],
    logger: logging.Logger
) -> None:
    """Combines metrics from multiple methods into methods_aggregated directory.
    
    Args:
        out_dir: Root output directory
        baselines: List of method names to aggregate (e.g., ["LwCP", "LoUPCP"])
        logger: Logger instance
    """
    # Build combined cross-method CSV at root out dir
    combined_rows = []
    for bl in baselines:
        method_dir = os.path.join(out_dir, f"method_{bl}")
        cross_csv = os.path.join(method_dir, f"{bl}_aggregated", "metrics_across_alphas.csv")
        if os.path.exists(cross_csv):
            try:
                df_bl = pd.read_csv(cross_csv)
                df_bl.insert(0, "method", bl)
                combined_rows.append(df_bl)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError) as e:
                logger.warning(f"Failed to read cross CSV for method {bl}: {e}. File: {cross_csv}")
        else:
            logger.debug(f"Cross-alpha CSV not found for method {bl}: {cross_csv}")
    if combined_rows:
        try:
            df_combined = pd.concat(combined_rows, ignore_index=True)
            methods_dir = os.path.join(out_dir, "methods_aggregated")
            os.makedirs(methods_dir, exist_ok=True)
            combined_csv = os.path.join(methods_dir, "metrics_across_alphas_by_method.csv")
            df_combined.to_csv(combined_csv, index=False)
            logger.info(f"Saved combined cross-method metrics -> {combined_csv}")
        except (IOError, OSError, ValueError) as e:
            logger.error(f"Failed to save combined cross-method metrics: {e}")
    else:
        logger.warning("No cross-method data collected. Skipping combined aggregation.")

