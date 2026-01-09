import sys
import os
import logging
import glob
import re
import argparse
import shutil
from typing import List, Set, Tuple, Dict

from exps.utils.io_utils import setup_logging
from exps.confpred.src.aggregation import (
    aggregate_per_alpha,
    aggregate_cross_alpha,
    aggregate_cross_baseline
)
from exps.confpred.src.plotting import (
    plot_alpha_boxplots,
    plot_aggregate_lines,
    plot_ab_comparison,
    METRIC_KEYS,
    LEVELS
)


def discover_methods_and_alphas(out_dir: str) -> Tuple[List[str], Dict[str, List[float]]]:
    """Discover methods and alphas from directory structure.
    
    Args:
        out_dir: Root output directory
        
    Returns:
        Tuple of (list of method names, dict mapping method to list of alpha values)
    """
    methods = []
    method_alphas = {}
    
    # Look for method_LwCP, method_LoUPCP directories
    for method_dir in glob.glob(os.path.join(out_dir, "method_*")):
        method_name = os.path.basename(method_dir).replace("method_", "")
        if method_name in ["LwCP", "LoUPCP"]:
            methods.append(method_name)
            
            # Discover alphas in this method directory
            alphas = []
            for alpha_dir in glob.glob(os.path.join(method_dir, "alpha_*")):
                alpha_name = os.path.basename(alpha_dir).replace("alpha_", "")
                # Skip aggregated directories
                if "aggregated" in alpha_name:
                    continue
                try:
                    # Try to parse as float
                    alpha_val = float(alpha_name)
                    alphas.append(alpha_val)
                except ValueError:
                    # Handle special case: alpha_0 vs alpha_0.0
                    if alpha_name == "0":
                        alphas.append(0.0)
                    else:
                        continue
            
            # Remove duplicates and sort
            alphas = sorted(list(set(alphas)))
            method_alphas[method_name] = alphas
    
    return sorted(methods), method_alphas


def discover_num_runs(out_dir: str, method: str, alpha: float) -> int:
    """Discover the number of runs for a given method/alpha by counting run directories.
    
    Args:
        out_dir: Root output directory
        method: Method name
        alpha: Alpha value
        
    Returns:
        Maximum run index found (or 0 if none found)
    """
    # Try both alpha_0 and alpha_0.0 for backward compatibility
    alpha_dir_variants = []
    if alpha == 0.0:
        alpha_dir_variants = [
            os.path.join(out_dir, f"method_{method}", "alpha_0"),
            os.path.join(out_dir, f"method_{method}", "alpha_0.0")
        ]
    else:
        alpha_dir_variants = [
            os.path.join(out_dir, f"method_{method}", f"alpha_{alpha}")
        ]
    
    max_run = 0
    for variant in alpha_dir_variants:
        if os.path.exists(variant):
            # Find all run_* directories
            for run_dir in glob.glob(os.path.join(variant, "run_*")):
                run_name = os.path.basename(run_dir)
                match = re.match(r"run_(\d+)", run_name)
                if match:
                    run_idx = int(match.group(1))
                    max_run = max(max_run, run_idx)
            break
    
    return max_run


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate aggregations and plots from existing experiment output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python regenerate_plots.py /path/to/output/directory
  python regenerate_plots.py /path/to/output/directory --debug
  python regenerate_plots.py /path/to/output/directory --mode metrics
  python regenerate_plots.py /path/to/output/directory --mode plots
  python regenerate_plots.py /path/to/output/directory --output /path/to/new/output
  python regenerate_plots.py /path/to/output/directory --mode both --output /path/to/new/output
  python regenerate_plots.py --help
        """
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path to the root output directory of a previous experiment execution"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging (show DEBUG level messages)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["metrics", "plots", "both"],
        default="both",
        help="What to generate: 'metrics' (aggregations only), 'plots' (plots only), or 'both' (default: both)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for regenerated aggregations and plots (default: same as input directory)"
    )
    
    args = parser.parse_args()
    
    input_dir = args.output_directory
    output_dir = args.output if args.output else input_dir
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist and is different from input
    if output_dir != input_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging (console only, no file logging)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level, to_file=False, log_dir=None, to_console=True)
    
    mode = args.mode
    generate_metrics = mode in ["metrics", "both"]
    generate_plots = mode in ["plots", "both"]
    
    logger.info(f"Input directory (reading from): {input_dir}")
    logger.info(f"Output directory (writing to): {output_dir}")
    logger.info(f"Mode: {mode} (metrics={generate_metrics}, plots={generate_plots})")
    if args.debug:
        logger.debug("Debug logging enabled")
    
    # Discover methods and alphas from input directory structure
    methods, method_alphas = discover_methods_and_alphas(input_dir)
    
    if not methods:
        logger.warning("No methods found in input directory. Nothing to regenerate.")
        return
    
    logger.info(f"Discovered methods: {methods}")
    for m, alphas in method_alphas.items():
        logger.info(f"  Method {m}: {len(alphas)} alphas found")
    
    # Constants
    levels = LEVELS
    metric_keys = METRIC_KEYS
    
    # Per-method processing
    for m in methods:
        alphas = method_alphas.get(m, [])
        if not alphas:
            logger.warning(f"[{m}] No alphas found. Skipping method {m}.")
            continue
        
        method_dir = os.path.join(output_dir, f"method_{m}")
        os.makedirs(method_dir, exist_ok=True)
        
        # Per-alpha aggregation for this method
        for alpha in alphas:
            # Discover number of runs for this alpha (from input directory)
            num_runs = discover_num_runs(input_dir, m, alpha)
            if num_runs == 0:
                logger.warning(
                    f"[{m}] No runs found for alpha={alpha}. Skipping."
                )
                continue
            
            logger.info(f"[{m}] Processing alpha={alpha} with {num_runs} runs")
            
            # Aggregate per alpha (if metrics mode)
            # Note: aggregate_per_alpha reads from input_dir but writes to output_dir
            if generate_metrics:
                aggregate_per_alpha(
                    out_dir=output_dir,
                    baseline=m,
                    alpha=alpha,
                    num_runs=num_runs,
                    levels=levels,
                    metric_keys=metric_keys,
                    logger=logger,
                    input_dir=input_dir if input_dir != output_dir else None
                )
            
            # Per-alpha plotting (boxplots) (if plots mode)
            if generate_plots:
                # Check both input and output directories for aggregated files
                alpha_dir_output = os.path.join(method_dir, f"alpha_{alpha}")
                alpha_agg_dir_output = os.path.join(alpha_dir_output, f"{m}_alpha_{alpha}_aggregated")
                
                alpha_dir_input = os.path.join(input_dir, f"method_{m}", f"alpha_{alpha}")
                alpha_agg_dir_input = os.path.join(alpha_dir_input, f"{m}_alpha_{alpha}_aggregated")
                
                # Use output directory if it exists, otherwise try input directory
                alpha_agg_dir = None
                if os.path.exists(alpha_agg_dir_output):
                    alpha_agg_dir = alpha_agg_dir_output
                elif os.path.exists(alpha_agg_dir_input):
                    # Found in input directory - use it for reading, but create output structure for plots
                    alpha_agg_dir = alpha_agg_dir_input
                    # Create output directory structure
                    os.makedirs(alpha_agg_dir_output, exist_ok=True)
                    # Copy runs_metrics CSV files to output directory if they don't exist there
                    for level in levels:
                        runs_metrics_input = os.path.join(alpha_agg_dir_input, f"runs_metrics_{level}.csv")
                        runs_metrics_output = os.path.join(alpha_agg_dir_output, f"runs_metrics_{level}.csv")
                        if os.path.exists(runs_metrics_input) and not os.path.exists(runs_metrics_output):
                            shutil.copy2(runs_metrics_input, runs_metrics_output)
                    # Now use output directory for plotting
                    alpha_agg_dir = alpha_agg_dir_output
                
                if alpha_agg_dir and os.path.exists(alpha_agg_dir):
                    try:
                        plot_alpha_boxplots(alpha_dir=alpha_agg_dir)
                    except (IOError, OSError, ValueError) as e:
                        logger.warning(f"[{m}] Per-alpha plotting failed for alpha={alpha}: {e}")
                else:
                    logger.debug(f"[{m}] No aggregated data found for alpha={alpha} (checked: {alpha_agg_dir_output}, {alpha_agg_dir_input})")
        
        # Cross-alpha aggregation for this method (if metrics mode)
        if generate_metrics:
            aggregate_cross_alpha(
                out_dir=output_dir,
                baseline=m,
                alphas=alphas,
                levels=levels,
                metric_keys=metric_keys,
                logger=logger
            )
        
        # Cross-alpha plotting (if plots mode)
        if generate_plots:
            # Check both input and output directories for aggregated files
            bl_agg_dir_output = os.path.join(method_dir, f"{m}_aggregated")
            cross_csv_output = os.path.join(bl_agg_dir_output, "metrics_across_alphas.csv")
            
            bl_agg_dir_input = os.path.join(input_dir, f"method_{m}", f"{m}_aggregated")
            cross_csv_input = os.path.join(bl_agg_dir_input, "metrics_across_alphas.csv")
            
            # Use output directory if it exists, otherwise try input directory
            cross_csv = None
            bl_agg_dir = None
            if os.path.exists(cross_csv_output):
                cross_csv = cross_csv_output
                bl_agg_dir = bl_agg_dir_output
            elif os.path.exists(cross_csv_input):
                cross_csv = cross_csv_input
                bl_agg_dir = bl_agg_dir_output  # Write plots to output directory
                os.makedirs(bl_agg_dir_output, exist_ok=True)
            
            if cross_csv and os.path.exists(cross_csv):
                try:
                    plot_aggregate_lines(out_dir=bl_agg_dir, cross_alpha_csv=cross_csv)
                except (IOError, OSError, ValueError) as e:
                    logger.warning(f"[{m}] Aggregate plotting failed: {e}")
            else:
                logger.debug(f"[{m}] No cross-alpha metrics found (checked: {cross_csv_output}, {cross_csv_input})")
    
    # Cross-method aggregation (if metrics mode)
    if len(methods) > 1:
        if generate_metrics:
            aggregate_cross_baseline(
                out_dir=output_dir,
                baselines=methods,
                logger=logger
            )
        
        # Cross-method plotting (if plots mode)
        if generate_plots:
            # Check both input and output directories for aggregated files
            methods_dir_output = os.path.join(output_dir, "methods_aggregated")
            combined_csv_output = os.path.join(methods_dir_output, "metrics_across_alphas_by_method.csv")
            
            methods_dir_input = os.path.join(input_dir, "methods_aggregated")
            combined_csv_input = os.path.join(methods_dir_input, "metrics_across_alphas_by_method.csv")
            
            # Use output directory if it exists, otherwise try input directory
            combined_csv = None
            methods_dir = None
            if os.path.exists(combined_csv_output):
                combined_csv = combined_csv_output
                methods_dir = methods_dir_output
            elif os.path.exists(combined_csv_input):
                combined_csv = combined_csv_input
                methods_dir = methods_dir_output  # Write plots to output directory
                os.makedirs(methods_dir_output, exist_ok=True)
            
            if combined_csv and os.path.exists(combined_csv):
                try:
                    plot_ab_comparison(ab_dir=methods_dir, combined_csv=combined_csv)
                except (IOError, OSError, ValueError) as e:
                    logger.warning(f"Methods aggregated plotting failed: {e}")
            else:
                logger.debug(f"No methods aggregated metrics found (checked: {combined_csv_output}, {combined_csv_input})")
    
    logger.info("Regeneration complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("regenerate_plots").exception("Fatal error in regenerate_plots")
        raise

