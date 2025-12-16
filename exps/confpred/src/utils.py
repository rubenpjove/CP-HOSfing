"""Shared utilities for AB baseline experiment."""
import os
import logging
import shutil
from typing import Dict, Tuple
import pandas as pd
import numpy as np


def derive_mode_mapping(pairs_df: pd.DataFrame, key_col: str, val_col: str) -> Dict[int, int]:
    """Derive mode-based mapping with deterministic tie-breaking (smallest code).
    
    Args:
        pairs_df: DataFrame with pairs of (key, value)
        key_col: Column name for keys
        val_col: Column name for values
        
    Returns:
        Dictionary mapping keys to their most frequent value (with deterministic tie-breaking)
    """
    if pairs_df.empty:
        return {}
    counts = (
        pairs_df
        .groupby([key_col, val_col])
        .size()
        .reset_index(name="cnt")
        .sort_values([key_col, "cnt", val_col], ascending=[True, False, True])
    )
    best = counts.drop_duplicates(subset=[key_col])[[key_col, val_col]]
    return {int(k): int(v) for k, v in zip(best[key_col].values, best[val_col].values)}


def build_hierarchy_mappings(df: pd.DataFrame, logger: logging.Logger) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Build hierarchy mappings from dataframe using mode vote.
    
    Args:
        df: DataFrame with leaf_idx, major_idx, family_idx columns
        logger: Logger instance
        
    Returns:
        Tuple of (leaf_to_major, major_to_family, leaf_to_family) mappings
    """
    # Derive leaf->major mapping from data using mode vote
    leaf_major_pairs = df.loc[df["leaf_idx"].notna() & df["major_idx"].notna(), 
                              ["leaf_idx", "major_idx"]].astype("int64")
    if not leaf_major_pairs.empty:
        leaf_to_major = derive_mode_mapping(leaf_major_pairs, "leaf_idx", "major_idx")
    else:
        leaf_to_major = {}
        logger.warning("No leaf->major pairs found in dataframe")
    
    # Derive major->family mapping from data using mode vote
    major_family_pairs = df.loc[df["major_idx"].notna() & df["family_idx"].notna(), 
                                ["major_idx", "family_idx"]].astype("int64")
    if not major_family_pairs.empty:
        major_to_family = derive_mode_mapping(major_family_pairs, "major_idx", "family_idx")
    else:
        major_to_family = {}
        logger.warning("No major->family pairs found in dataframe")
    
    # Derive leaf->family mapping (direct or via major)
    leaf_family_pairs = df.loc[df["leaf_idx"].notna() & df["family_idx"].notna(), 
                              ["leaf_idx", "family_idx"]].astype("int64")
    if not leaf_family_pairs.empty:
        leaf_to_family = derive_mode_mapping(leaf_family_pairs, "leaf_idx", "family_idx")
    else:
        # Fall back to leaf->major->family chain where major exists
        leaf_to_family = {}
        for leaf_gid, major_gid in leaf_to_major.items():
            family_gid = major_to_family.get(major_gid, None)
            if family_gid is not None:
                leaf_to_family[leaf_gid] = family_gid
        logger.info("Built leaf->family mapping from leaf->major->family chain")
    
    return leaf_to_major, major_to_family, leaf_to_family


def setup_single_file_logger(logger: logging.Logger, out_dir: str) -> None:
    """Configure logger to write to a single file named after the logging level.
    
    Args:
        logger: Logger instance to configure
        out_dir: Directory for log file
    """
    try:
        # Remove existing file handlers to avoid duplicate writes
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)

        level_to_name = {
            logging.DEBUG: "debug",
            logging.INFO: "info",
            logging.WARNING: "warn",
            logging.ERROR: "error",
            logging.CRITICAL: "critical",
        }
        current_level_name = level_to_name.get(logger.level, "info")
        level_log_path = os.path.join(out_dir, f"{current_level_name}.log")
        file_handler = logging.FileHandler(level_log_path, encoding="utf-8")
        # Mirror formatter used by setup_logging (human-readable UTC)
        import datetime as dt
        from datetime import timezone as tz
        formatter = logging.Formatter(fmt="%(asctime)sZ [%(levelname)s] - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        formatter.converter = lambda *args: dt.datetime.now(tz.utc).timetuple()
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to setup single file logger: {e}")


def copy_artifact_file(src: str, dst: str, logger: logging.Logger) -> bool:
    """Copy a single artifact file, logging any errors.
    
    Args:
        src: Source file path
        dst: Destination file path
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Normalize paths to handle symlinks and relative paths
        src_normalized = os.path.abspath(os.path.normpath(src))
        dst_normalized = os.path.abspath(os.path.normpath(dst))
        
        # Skip copy if source and destination are the same file
        if src_normalized == dst_normalized:
            logger.debug(f"Skipping copy: source and destination are the same file: {src}")
            return True
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    except (OSError, IOError, shutil.Error) as e:
        logger.warning(f"Failed to copy {src} to {dst}: {e}")
        return False


def split_and_rename_artifacts(run_dir: str, out_dir: str, alpha: float, run_idx: int, 
                                baseline: str, baselines_to_aggregate: list, logger: logging.Logger) -> None:
    """Rename and organize baseline A artifacts into baseline-specific trees with full-trace filenames.
    
    Note: Baseline B files are now generated directly in their correct location, so this function
    only handles baseline A file renaming and organization.
    
    Args:
        run_dir: Source run directory (where baseline A files are generated)
        out_dir: Root output directory
        alpha: Alpha value
        run_idx: Run index
        baseline: Baseline being executed (A or B)
        baselines_to_aggregate: List of baselines to aggregate
        logger: Logger instance
    """
    # Ensure alpha is float for consistent formatting (0 -> 0.0, not "0")
    alpha = float(alpha)
    levels = ["family", "major", "leaf"]
    
    # Only process baseline A files (baseline B files are already in correct location)
    if "A" not in baselines_to_aggregate:
        return
    
    run_dir_A = os.path.join(out_dir, f"baseline_A", f"alpha_{alpha}", f"run_{run_idx}")
    os.makedirs(run_dir_A, exist_ok=True)
    
    # Rename baseline A summary files (from cp_* to A_alpha_*_run_*_cp_*)
    for level in levels:
        for ext in ("csv", "json"):
            src = os.path.join(run_dir, f"cp_{level}_summary.{ext}")
            if os.path.exists(src):
                dst = os.path.join(run_dir_A, f"A_alpha_{alpha}_run_{run_idx}_cp_{level}_summary.{ext}")
                # Check if source and destination are the same before copying
                src_normalized = os.path.abspath(os.path.normpath(src))
                dst_normalized = os.path.abspath(os.path.normpath(dst))
                if src_normalized != dst_normalized:
                    # Only copy and remove if they're different files
                    if copy_artifact_file(src, dst, logger):
                        try:
                            os.remove(src)
                        except OSError:
                            pass

    # Rename baseline A test samples details file
    src_new_details = os.path.join(run_dir, f"A_alpha_{alpha}_run_{run_idx}_test_samples.csv")
    src_legacy_details = os.path.join(run_dir, "run_details.csv")
    dst_details = os.path.join(run_dir_A, f"A_alpha_{alpha}_run_{run_idx}_test_samples.csv")

    if os.path.exists(src_new_details):
        # Check if source and destination are the same before copying
        src_normalized = os.path.abspath(os.path.normpath(src_new_details))
        dst_normalized = os.path.abspath(os.path.normpath(dst_details))
        if src_normalized != dst_normalized:
            # Only copy and remove if they're different files
            if copy_artifact_file(src_new_details, dst_details, logger):
                try:
                    os.remove(src_new_details)
                except OSError:
                    pass
    elif os.path.exists(src_legacy_details):
        # Check if source and destination are the same before copying
        src_normalized = os.path.abspath(os.path.normpath(src_legacy_details))
        dst_normalized = os.path.abspath(os.path.normpath(dst_details))
        if src_normalized != dst_normalized:
            # Only copy and remove if they're different files
            if copy_artifact_file(src_legacy_details, dst_details, logger):
                try:
                    os.remove(src_legacy_details)
                except OSError:
                    pass

