import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Dict, List


METRIC_KEYS = [
    "coverage",
    "set_size_mean",
    "set_size_median",
    "empty_rate",
    "singleton_rate",
    "top1_acc_non_empty",
    "base_top1_acc",
    "hir",
]
LEVELS = ["family", "major", "leaf"]


def format_metric_name(metric: str) -> str:
    """Format metric name for display in plots (remove underscores, capitalize properly).
    
    Args:
        metric: Raw metric name (e.g., "singleton_rate", "hir")
        
    Returns:
        Formatted metric name (e.g., "Singleton Rate", "HIR")
    """
    metric_format_map = {
        "singleton_rate": "Singleton Rate",
        "empty_rate": "Empty Set Rate",
        "set_size_mean": "Mean Set Size",
        "set_size_median": "Median Set Size",
        "hir": "HIR",
        "coverage": "Coverage",
        "top1_acc_non_empty": "Top-1 Accuracy (Non-Empty)",
        "base_top1_acc": "Base Top-1 Accuracy",
    }
    return metric_format_map.get(metric, metric.replace("_", " ").title())


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_alpha_boxplots(alpha_dir: str):
    """Create boxplots across runs for each metric and level in a given alpha dir.

    Expects files: alpha_dir/runs_metrics_{level}.csv
    Saves to: alpha_dir/plots/{metric}_{level}_boxplot.{png,svg}
    """
    plots_dir = os.path.join(alpha_dir, "plots")
    _ensure_dir(plots_dir)

    for level in LEVELS:
        csv_path = os.path.join(alpha_dir, f"runs_metrics_{level}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        # For each metric, plot a boxplot across runs
        for metric in METRIC_KEYS:
            if metric not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            # Boxplot of per-run values
            ax.boxplot([df[metric].dropna().values], labels=[level])
            formatted_metric = format_metric_name(metric)
            ax.set_title(f"{formatted_metric} — {level}")
            ax.set_ylabel(formatted_metric)
            ax.grid(True, linestyle=":", alpha=0.6)
            png_path = os.path.join(plots_dir, f"{metric}_{level}_boxplot.png")
            svg_path = os.path.join(plots_dir, f"{metric}_{level}_boxplot.svg")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.savefig(svg_path)
            plt.close(fig)


def plot_aggregate_lines(out_dir: str, cross_alpha_csv: str):
    """Create line plots of mean±std vs alpha for each metric, with one line per level.

    Expects: out_dir/metrics_across_alphas.csv with columns: alpha, level, metric, mean, std
    Saves to: out_dir/plots/{metric}_mean_std_by_level.{png,svg}
    """
    try:
        df = pd.read_csv(cross_alpha_csv)
    except Exception:
        return
    plots_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plots_dir)

    # Ensure sorted alphas for consistent lines
    if "alpha" in df.columns:
        # We'll iterate per metric
        for metric in METRIC_KEYS:
            sub = df[df["metric"] == metric].copy()
            if sub.empty:
                continue
            # Pivot to have rows=alpha, columns=level, values=mean/std handled separately
            alphas_sorted = sorted(sub["alpha"].unique())
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for level in LEVELS:
                lvl = sub[sub["level"] == level]
                if lvl.empty:
                    continue
                # Align by alpha ordering
                lvl_sorted = lvl.set_index("alpha").reindex(alphas_sorted).reset_index()
                x = lvl_sorted["alpha"].values
                y = lvl_sorted["mean"].values
                s = lvl_sorted["std"].values if "std" in lvl_sorted.columns else None
                ax.plot(x, y, marker="o", label=level)
                if s is not None and not pd.isna(s).all():
                    y1 = y - s
                    y2 = y + s
                    ax.fill_between(x, y1, y2, alpha=0.15)
            formatted_metric = format_metric_name(metric)
            ax.set_title(formatted_metric)
            ax.set_xlabel("alpha")
            ax.set_ylabel(formatted_metric)
            # Use logarithmic scale for set_size_mean to handle large values at alpha=0
            if metric == "set_size_mean":
                ax.set_yscale('symlog', linthresh=1.0)  # Symmetric log scale handles both small and large values
                # Use ScalarFormatter to display regular numbers instead of base-10 notation
                ax.yaxis.set_major_formatter(ScalarFormatter())
                # Set custom y-axis ticks: fine granularity in 0-5 range, good coverage of upper range
                ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50])
                # Set y-axis to start at 0.5 instead of 0.0
                ax.set_ylim(bottom=0.5)
            # Add reference line for coverage plots
            if metric == "coverage":
                ref_x = np.array(alphas_sorted)
                ref_y = 1 - ref_x
                ax.plot(ref_x, ref_y, color='black', linestyle='-', linewidth=0.8, 
                       label='1-α', alpha=0.7)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(title="level")
            png_path = os.path.join(plots_dir, f"{metric}_mean_std_by_level.png")
            svg_path = os.path.join(plots_dir, f"{metric}_mean_std_by_level.svg")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.savefig(svg_path)
            plt.close(fig)


def plot_ab_comparison(ab_dir: str, combined_csv: str):
    """Generate method comparison plots: per metric, overlay 3 levels (colors) and 2 methods (linestyles).
    
    Expects: combined_csv with columns: method, alpha, level, metric, mean, std
    Saves to: ab_dir/plots/{methods}_{metric}_mean_std_across_levels.{png,svg}
    where {methods} is the methods joined by "-" (e.g., "LwCP-LoUPCP")
    
    Args:
        ab_dir: Directory for methods aggregated results (plots subdirectory will be created)
        combined_csv: Path to metrics_across_alphas_by_method.csv
    """
    try:
        df_combined = pd.read_csv(combined_csv)
    except Exception as e:
        return
    
    plots_dir = os.path.join(ab_dir, "plots")
    _ensure_dir(plots_dir)
    
    # Ensure numeric alpha for sorting
    df_combined["alpha"] = df_combined["alpha"].astype(float)
    # Handle both "baseline" and "method" column names for backward compatibility
    method_col = "method" if "method" in df_combined.columns else "baseline"
    methods_present = sorted(df_combined[method_col].dropna().unique().tolist())
    # Create method prefix for filenames (e.g., "LwCP-LoUPCP")
    methods_prefix = "-".join(methods_present)
    # Method colors; line styles encode levels
    method_color_map = {"LwCP": "#1f77b4", "LoUPCP": "#2ca02c"}
    level_linestyle_map = {"family": "-", "major": "--", "leaf": ":"}
    
    def format_method_label(method: str) -> str:
        """Format method name for display."""
        if method == "LwCP":
            return "Lw-CP"
        elif method == "LoUPCP":
            return "LoUP-CP"
        return f"Method {method}"
    
    for metric in METRIC_KEYS:
        plt.figure(figsize=(9, 5))
        plotted_any = False

        # Special handling for HIR: one line per method (no per-level breakdown)
        if metric == "hir":
            for m in methods_present:
                df_m = df_combined[(df_combined["metric"] == metric) & (df_combined[method_col] == m)].copy()
                if df_m.empty:
                    continue
                df_m = df_m.sort_values("alpha")
                # If multiple levels exist, aggregate across levels for this method/alpha
                df_m = (
                    df_m.groupby("alpha", as_index=False)
                    .agg({"mean": "mean", "std": "mean"})
                    .sort_values("alpha")
                )
                alphas = df_m["alpha"].values
                means = df_m["mean"].values
                stds = df_m["std"].values
                color = method_color_map.get(m, "#1f77b4")
                label = format_method_label(m)
                plt.plot(alphas, means, linestyle="-", color=color, label=label)
                lower = np.where(np.isfinite(stds), means - stds, np.nan)
                upper = np.where(np.isfinite(stds), means + stds, np.nan)
                plt.fill_between(alphas, lower, upper, alpha=0.15, color=color)
                plotted_any = True
            if not plotted_any:
                plt.close()
                continue
            # Add reference line for coverage plots
            if metric == "coverage":
                alphas_for_ref = sorted(df_combined[df_combined["metric"] == metric]["alpha"].unique())
                if len(alphas_for_ref) > 0:
                    ref_x = np.array(alphas_for_ref)
                    ref_y = 1 - ref_x
                    plt.plot(ref_x, ref_y, color='black', linestyle='-', linewidth=0.8, 
                            label='1-α', alpha=0.7)
            formatted_metric = format_metric_name(metric)
            plt.xlabel("alpha")
            plt.ylabel(formatted_metric)
            plt.title(formatted_metric)
            plt.legend(ncol=1)
            plt.grid(True, linestyle=":", linewidth=0.5)
            base_name = f"{methods_prefix}_{metric}_mean_std_across_levels"
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, base_name + ".svg"))
            plt.savefig(os.path.join(plots_dir, base_name + ".png"), dpi=150)
            plt.close()
            continue

        # Default: per-level lines with per-method styles (same for all metrics including coverage)
        for m in methods_present:
            color = method_color_map.get(m, "#1f77b4")
            for level in LEVELS:
                df_m = df_combined[
                    (df_combined["level"] == level)
                    & (df_combined["metric"] == metric)
                    & (df_combined[method_col] == m)
                ].copy()
                if df_m.empty:
                    continue
                df_m = df_m.sort_values("alpha")
                alphas = df_m["alpha"].values
                means = df_m["mean"].values
                stds = df_m["std"].values
                method_label = format_method_label(m)
                label = f"{level} ({method_label})"
                linestyle = level_linestyle_map.get(level, "-")
                plt.plot(alphas, means, linestyle=linestyle, color=color, label=label)
                lower = np.where(np.isfinite(stds), means - stds, np.nan)
                upper = np.where(np.isfinite(stds), means + stds, np.nan)
                plt.fill_between(alphas, lower, upper, alpha=0.15, color=color)
                plotted_any = True
        
        # Add reference line for coverage plots (but don't include it in legend)
        if metric == "coverage":
            alphas_for_ref = sorted(df_combined[df_combined["metric"] == metric]["alpha"].unique())
            if len(alphas_for_ref) > 0:
                ref_x = np.array(alphas_for_ref)
                ref_y = 1 - ref_x
                plt.plot(ref_x, ref_y, color='black', linestyle='-', linewidth=0.8, 
                        label=None, alpha=0.7)  # label=None to exclude from legend
        
        if not plotted_any:
            plt.close()
            continue
        
        formatted_metric = format_metric_name(metric)
        plt.xlabel("alpha")
        plt.ylabel(formatted_metric)
        # Use logarithmic scale for set_size_mean to handle large values at alpha=0
        if metric == "set_size_mean":
            plt.yscale('symlog', linthresh=1.0)  # Symmetric log scale handles both small and large values
            # Use ScalarFormatter to display regular numbers instead of base-10 notation
            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
            # Set custom y-axis ticks: fine granularity in 0-5 range, good coverage of upper range
            plt.yticks([0, 0.5, 1, 2, 5, 10, 20, 50])
            # Set y-axis to start at 0.5 instead of 0.0
            plt.ylim(bottom=0.5)
        plt.title(formatted_metric)
        # Use same legend format for all metrics (including coverage, but excluding HIR which has special handling)
        plt.legend(ncol=2)
        
        plt.grid(True, linestyle=":", linewidth=0.5)
        base_name = f"{methods_prefix}_{metric}_mean_std_across_levels"
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, base_name + ".svg"))
        plt.savefig(os.path.join(plots_dir, base_name + ".png"), dpi=150)
        plt.close()
