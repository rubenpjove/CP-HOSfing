import os
import json
import math
import logging
import time
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold

from exps.predictors.src.cphos.train import (
    _pd_na_to_np_nan,
    _astype_str,
    _make_ohe,
    apply_feature_selection,
    train_one_fold,
)
from exps.predictors.src.cphos.infer import predict_proba
from exps.predictors.src.cphos.models import MLP
from exps.confpred.common.projection import (
    build_vocab_maps_for_B,
    project_leaf_masks_upward,
    evaluate_sets_from_masks,
)
from exps.confpred.src.constants import DEFAULT_VAL_SIZE, DEFAULT_VARIANCE_THRESHOLD
from exps.confpred.src.utils import derive_mode_mapping, build_hierarchy_mappings


def ensure_ancestor_consistency(df: pd.DataFrame, maps: Dict[str, Dict[int, int]], logger: logging.Logger) -> pd.DataFrame:
    """Ensure family_idx and major_idx are deterministic ancestors of leaf_idx using maps.

    If inconsistencies are detected or columns are missing, derive them from leaf via maps.
    """
    out = df.copy()

    # Ensure nullable integer types are preserved
    for col in ["family_idx", "major_idx", "leaf_idx"]:
        if col in out.columns:
            out[col] = out[col].astype("Int64")

    # Derive leaf->major mapping from data (rows with both defined)
    leaf_major_pairs = out.loc[out["leaf_idx"].notna() & out["major_idx"].notna(), ["leaf_idx", "major_idx"]].astype("int64")
    leaf_to_major = derive_mode_mapping(leaf_major_pairs, "leaf_idx", "major_idx")

    # Derive major->family mapping from data (rows with both defined)
    major_family_pairs = out.loc[out["major_idx"].notna() & out["family_idx"].notna(), ["major_idx", "family_idx"]].astype("int64")
    major_to_family = derive_mode_mapping(major_family_pairs, "major_idx", "family_idx")

    # Apply mappings to repair inconsistencies, counting changes
    repairs_major = 0
    if len(leaf_to_major) > 0 and "major_idx" in out.columns and "leaf_idx" in out.columns:
        mask_leaf_def = out["leaf_idx"].notna()
        # Proposed majors from mapping when available; otherwise keep current
        proposed_major = out.loc[mask_leaf_def, "leaf_idx"].astype("int64").map(leaf_to_major)
        cur_major = out.loc[mask_leaf_def, "major_idx"]
        # Where proposed exists and differs from current (or current is NA), set to proposed
        need_update = proposed_major.notna() & (cur_major.isna() | (cur_major.astype("Int64") != proposed_major.astype("Int64")))
        repairs_major = int(need_update.sum())
        out.loc[mask_leaf_def & need_update, "major_idx"] = proposed_major[need_update].astype("Int64")
        out["major_idx"] = out["major_idx"].astype("Int64")

    repairs_family = 0
    if len(major_to_family) > 0 and "family_idx" in out.columns and "major_idx" in out.columns:
        mask_major_def = out["major_idx"].notna()
        proposed_family = out.loc[mask_major_def, "major_idx"].astype("int64").map(major_to_family)
        cur_family = out.loc[mask_major_def, "family_idx"]
        need_update_fam = proposed_family.notna() & (cur_family.isna() | (cur_family.astype("Int64") != proposed_family.astype("Int64")))
        repairs_family = int(need_update_fam.sum())
        out.loc[mask_major_def & need_update_fam, "family_idx"] = proposed_family[need_update_fam].astype("Int64")
        out["family_idx"] = out["family_idx"].astype("Int64")

    # Validate determinism after repair; log warnings if inconsistencies remain
    def count_inconsistencies(df_in: pd.DataFrame, child: str, parent: str) -> int:
        sub = df_in.loc[df_in[child].notna() & df_in[parent].notna(), [child, parent]].astype("int64")
        if sub.empty:
            return 0
        grp = sub.groupby(child)[parent].nunique()
        return int((grp > 1).sum())

    inc_leaf_major = count_inconsistencies(out, "leaf_idx", "major_idx") if {"leaf_idx", "major_idx"}.issubset(out.columns) else 0
    inc_major_fam = count_inconsistencies(out, "major_idx", "family_idx") if {"major_idx", "family_idx"}.issubset(out.columns) else 0

    if repairs_major > 0 or repairs_family > 0:
        logger.warning(
            f"Ancestor repair applied: major updates={repairs_major}, family updates={repairs_family}"
        )
    if inc_leaf_major > 0 or inc_major_fam > 0:
        logger.warning(
            f"Residual hierarchy inconsistencies: leaf→major keys with >1 major={inc_leaf_major}, major→family keys with >1 family={inc_major_fam}"
        )
    else:
        logger.info("Hierarchy columns are consistent after validation/repair")

    return out


def perform_two_stage_stratified_split(
    df: pd.DataFrame,
    leaf_col: str,
    proportions: Dict[str, float],
    seed: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified Shuffle Split to obtain Cal/Test indices with stratification by leaf.

    Uses StratifiedShuffleSplit to split data into calibration and test sets
    with the specified proportions, stratified by leaf class.
    Returns (idx_cal, idx_test).
    """
    rng = np.random.RandomState(seed)

    p_cal = float(proportions.get("cal", 0.5))
    p_test = float(proportions.get("test", 0.5))
    if not math.isclose(p_cal + p_test, 1.0, rel_tol=1e-6):
        raise ValueError("split_proportions must sum to 1.0")

    y_leaf = df[leaf_col].astype("Int64")
    defined_mask = y_leaf.notna().values
    idx_all = np.arange(df.shape[0])
    idx_defined = idx_all[defined_mask]
    idx_undef = idx_all[~defined_mask]

    n_total = idx_all.size
    n_test_target = int(round(p_test * n_total))
    n_cal_target = n_total - n_test_target

    # Split defined rows using StratifiedShuffleSplit
    test_defined_idx: np.ndarray
    cal_defined_idx: np.ndarray
    if idx_defined.size > 0:
        y_leaf_defined = y_leaf[defined_mask].astype("int64").values
        
        # Check if we have enough samples per class for stratified split
        unique_classes, class_counts = np.unique(y_leaf_defined, return_counts=True)
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            # Fallback: if any class has < 2 samples, use random split instead
            logger.warning(
                f"Some leaf classes have < 2 samples (min={min_class_count}). "
                "Using random split instead of stratified split."
            )
            rng.shuffle(idx_defined)
            n_test_def = int(round(p_test * idx_defined.size))
            test_defined_idx = idx_defined[:n_test_def]
            cal_defined_idx = idx_defined[n_test_def:]
        else:
            # Use StratifiedShuffleSplit
            test_size = min(max(p_test, 1e-9), 1 - 1e-9)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            split = next(sss.split(np.zeros_like(y_leaf_defined), y_leaf_defined))
            test_defined_idx = idx_defined[split[1]]
            cal_defined_idx = idx_defined[split[0]]
    else:
        test_defined_idx = np.array([], dtype=int)
        cal_defined_idx = np.array([], dtype=int)

    # Distribute undefined rows proportionally
    n_test_def = int(test_defined_idx.size)
    n_cal_def = int(cal_defined_idx.size)
    n_def_total = n_test_def + n_cal_def
    
    if n_def_total > 0:
        # Proportion of undefined rows to assign to test
        p_test_actual = n_test_def / n_def_total
    else:
        # If no defined rows, split undefined 50/50
        p_test_actual = 0.5
    
    rng.shuffle(idx_undef)
    n_test_undef = int(round(p_test_actual * idx_undef.size))
    test_undef_idx = idx_undef[:n_test_undef]
    cal_undef_idx = idx_undef[n_test_undef:]

    idx_test = np.concatenate([test_defined_idx, test_undef_idx])
    idx_cal = np.concatenate([cal_defined_idx, cal_undef_idx])

    logger.info(
        f"Split sizes (all rows): Cal={idx_cal.size}, Test={idx_test.size}"
    )
    return idx_cal, idx_test


def build_level_preprocessor(
    train_df: pd.DataFrame, 
    categorical_cols: List[str], 
    numerical_cols: List[str],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
) -> Tuple[ColumnTransformer, SimpleImputer, StandardScaler, VarianceThreshold, np.ndarray]:
    """Build and fit preprocessing pipeline for a level.
    
    Args:
        train_df: Training DataFrame
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        variance_threshold: Threshold for variance filter
        
    Returns:
        Tuple of (pre_enc, imp, scl, vt, X_train_vt)
    """
    valid_categorical_cols = [c for c in categorical_cols if train_df[c].notna().any() and c in train_df.columns]

    cat_nonbyte_pipeline = Pipeline(steps=[
        ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
        ("impute", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
        ("to_str", FunctionTransformer(_astype_str, validate=False)),
        ("encode", _make_ohe()),
    ])

    num_pipeline = Pipeline(steps=[
        ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
    ])

    pre_enc = ColumnTransformer(
        transformers=[
            ("cat_nonbyte", cat_nonbyte_pipeline, valid_categorical_cols),
            ("num", num_pipeline, numerical_cols),
        ],
        remainder="drop",
    )

    X_train_enc = pre_enc.fit_transform(train_df)

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_train_s = scl.fit_transform(imp.fit_transform(X_train_enc))

    vt = VarianceThreshold(threshold=variance_threshold)
    X_train_vt = vt.fit_transform(X_train_s)

    return pre_enc, imp, scl, vt, X_train_vt


def transform_features(pre_enc, imp, scl, vt, df: pd.DataFrame):
    # Guard: ensure transformers were already fitted on Train
    assert hasattr(pre_enc, "transformers_"), "Pre-encoder not fitted on Train"
    assert hasattr(imp, "statistics_"), "Imputer not fitted on Train"
    assert hasattr(scl, "mean_"), "Scaler not fitted on Train"
    assert hasattr(vt, "variances_"), "VarianceThreshold not fitted on Train"
    X_enc = pre_enc.transform(df)
    X_s = scl.transform(imp.transform(X_enc))
    X_vt = vt.transform(X_s)
    return X_vt


def train_level_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict,
    input_params: Dict,
    seed: int,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor], np.ndarray]:
    """Train a model for a single level.
    
    Args:
        X_train: Training features
        y_train: Training labels (local indices)
        best_params: Best hyperparameters
        input_params: Input parameters dict
        seed: Random seed
        logger: Logger instance
        
    Returns:
        Tuple of (model, state_dict, unique_classes)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create internal validation split for early stopping
    val_size = input_params.get("val_size", DEFAULT_VAL_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    state, metrics, best_epoch = train_one_fold(
        X_tr,
        y_tr,
        X_val,
        y_val,
        best_params,
        device,
        verbose=False,
        rng_seed=seed,
        use_amp=input_params.get("use_amp", True),
        num_workers=input_params.get("num_workers", 4),
    )

    in_dim = X_train.shape[1]
    out_dim = int(np.unique(y_train).size)
    model = MLP(
        in_dim,
        best_params["hidden_dims"],
        out_dim,
        dropout=best_params["dropout"],
        use_bn=best_params.get("use_bn", True),
    ).to(device)

    if input_params.get("use_multi_gpu", True) and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Align state dict keys for DP/Non-DP
    if isinstance(model, torch.nn.DataParallel):
        if state and not any(key.startswith("module.") for key in state.keys()):
            state = {f"module.{k}": v for k, v in state.items()}
    else:
        if state and any(key.startswith("module.") for key in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()

    unique_classes = np.unique(y_train)
    return model, state, unique_classes


def predict_proba_level(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def calibrate_split_cp(true_label_probs: np.ndarray, alpha: float) -> Tuple[float, int, int, float]:
    """Compute right-continuous empirical quantile and related stats.

    Returns (q_hat, m, j, step) where j = ceil((m+1)*(1-alpha)) clamped to [1, m],
    and step = 1/(m+1).
    """
    # Nonconformity scores = 1 - p(true)
    scores = 1.0 - true_label_probs
    m = int(scores.shape[0])
    if m <= 0:
        return float("nan"), 0, 0, float("nan")
    step = 1.0 / float(m + 1)
    j = int(math.ceil((m + 1) * (1.0 - float(alpha))))
    j = max(1, min(j, m))
    q_hat = np.partition(scores, j - 1)[j - 1]
    return float(q_hat), m, j, float(step)


def evaluate_cp_sets(
    probs: np.ndarray,
    q_hat: float,
    y_true_local: np.ndarray,
) -> Dict[str, float]:
    """Compute CP evaluation metrics per level.

    - Coverage: mean( indicator{ true in set } )
    - Efficiency: mean/median set size
    - Empty-set rate
    - Singleton rate
    - Optional: Top-1 accuracy when non-empty
    """
    if probs.shape[0] == 0:
        return {
            "coverage": float("nan"),
            "set_size_mean": float("nan"),
            "set_size_median": float("nan"),
            "empty_rate": float("nan"),
            "singleton_rate": float("nan"),
            "top1_acc_non_empty": float("nan"),
        }

    # Nonconformity scores for all labels
    scores_all = 1.0 - probs
    keep_mask = scores_all <= q_hat  # right-continuous: s <= q_hat

    set_sizes = keep_mask.sum(axis=1).astype("int64")
    empty_rate = float(np.mean(set_sizes == 0))
    singleton_rate = float(np.mean(set_sizes == 1))
    set_size_mean = float(np.mean(set_sizes))
    set_size_median = float(np.median(set_sizes))

    # Coverage
    rows = np.arange(probs.shape[0])
    covered = keep_mask[rows, y_true_local]
    coverage = float(np.mean(covered))

    # Top-1 conditional on non-empty using argmax restricted to kept set
    masked_probs = np.where(keep_mask, probs, -np.inf)
    top1 = masked_probs.argmax(axis=1)
    non_empty = set_sizes > 0
    top1_acc_non_empty = float(np.mean((top1[non_empty] == y_true_local[non_empty]))) if np.any(non_empty) else float("nan")

    return {
        "coverage": coverage,
        "set_size_mean": set_size_mean,
        "set_size_median": set_size_median,
        "empty_rate": empty_rate,
        "singleton_rate": singleton_rate,
        "top1_acc_non_empty": top1_acc_non_empty,
    }


def _build_children_maps(
    leaf_to_major: Dict[int, int],
    major_to_family: Dict[int, int],
) -> Tuple[Dict[int, set], Dict[int, set]]:
    """Build downward adjacency lists for hierarchy checks."""
    major_children: Dict[int, set] = {}
    for leaf_gid, major_gid in leaf_to_major.items():
        major_children.setdefault(int(major_gid), set()).add(int(leaf_gid))
    family_children: Dict[int, set] = {}
    for major_gid, family_gid in major_to_family.items():
        family_children.setdefault(int(family_gid), set()).add(int(major_gid))
    return major_children, family_children


def compute_hir_rate(
    family_sets: Dict[int, set],
    major_sets: Dict[int, set],
    leaf_sets: Dict[int, set],
    leaf_to_major: Dict[int, int],
    major_to_family: Dict[int, int],
) -> float:
    """Compute Hierarchical Inconsistency Rate over provided per-sample sets."""
    major_children, family_children = _build_children_maps(leaf_to_major, major_to_family)
    sample_ids = set(family_sets.keys()) & set(major_sets.keys()) & set(leaf_sets.keys())
    if not sample_ids:
        return float("nan")

    violations = 0
    for sid in sample_ids:
        fam_set = family_sets.get(sid, set())
        maj_set = major_sets.get(sid, set())
        leaf_set = leaf_sets.get(sid, set())

        orphan_leaf = any(
            (leaf_gid in leaf_to_major) and (leaf_to_major[leaf_gid] not in maj_set)
            for leaf_gid in leaf_set
        )
        orphan_major = any(
            (maj_gid in major_to_family) and (major_to_family[maj_gid] not in fam_set)
            for maj_gid in maj_set
        )

        dead_family = any(
            (children := family_children.get(fam_gid, set()))
            and children.isdisjoint(maj_set)
            for fam_gid in fam_set
        )
        dead_major = any(
            (children := major_children.get(maj_gid, set()))
            and children.isdisjoint(leaf_set)
            for maj_gid in maj_set
        )

        violation = orphan_leaf or orphan_major or dead_family or dead_major
        violations += int(violation)

    return float(violations / len(sample_ids))


def save_cp_artifacts(out_dir: str, level: str, summary: Dict):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"cp_{level}_summary.json")
    csv_path = os.path.join(out_dir, f"cp_{level}_summary.csv")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Also save CSV for a flat view of top-level fields if possible
    flat = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
    pd.DataFrame([flat]).to_csv(csv_path, index=False)


def save_split_summary(out_dir: str, split_info: Dict):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cp_split_summary.json")
    with open(json_path, "w") as f:
        json.dump(split_info, f, indent=2)


def save_loupcp_cp_artifacts(out_dir_target: str, level: str, summary: Dict, alpha: float, run_idx: int):
    """Save LoUP-CP CP artifacts.
    
    Args:
        out_dir_target: Output directory
        level: Level name
        summary: Summary dictionary
        alpha: Alpha value
        run_idx: Run index
    """
    os.makedirs(out_dir_target, exist_ok=True)
    # Use same format as Lw-CP: LoUPCP_alpha_{alpha}_run_{run_idx}_cp_{level}_summary.{ext}
    json_path = os.path.join(out_dir_target, f"LoUPCP_alpha_{alpha}_run_{run_idx}_cp_{level}_summary.json")
    csv_path = os.path.join(out_dir_target, f"LoUPCP_alpha_{alpha}_run_{run_idx}_cp_{level}_summary.csv")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    flat = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
    pd.DataFrame([flat]).to_csv(csv_path, index=False)


def run_loupcp(
    df: pd.DataFrame,
    maps: Dict,
    input_params: Dict,
    out_dir: str,
    leaf_vocab_global: np.ndarray,
    major_vocab_global: np.ndarray,
    family_vocab_global: np.ndarray,
    probs_test_leaf: np.ndarray,
    q_hat_leaf: float,
    m_leaf: int,
    idx_test_leaf_defined: np.ndarray,
    idx_test_leaf_seen: np.ndarray,
    y_test_leaf_global_aligned: np.ndarray,
    alpha_single: float,
    base_pred_labels_from_LwCP: Dict[str, Dict[int, str]],
    logger: logging.Logger,
    out_dir_LoUPCP: Optional[str] = None,
) -> None:
    """Run LoUP-CP: Leaf-only CP with upward closure.
    
    Args:
        df: DataFrame with data
        maps: Mapping dictionaries
        input_params: Input parameters
        out_dir: Output directory for Lw-CP
        leaf_vocab_global: Leaf vocabulary (global indices)
        major_vocab_global: Major vocabulary (global indices)
        family_vocab_global: Family vocabulary (global indices)
        probs_test_leaf: Leaf test probabilities
        q_hat_leaf: Leaf quantile
        m_leaf: Leaf calibration set size
        idx_test_leaf_defined: Test leaf defined indices
        idx_test_leaf_seen: Test leaf seen indices
        y_test_leaf_global_aligned: Test leaf global labels (aligned)
        alpha_single: Alpha value
        base_pred_labels_from_LwCP: Base predictions from Lw-CP
        logger: Logger instance
        out_dir_LoUPCP: Optional output directory for LoUP-CP (defaults to out_dir if None)
    """
    logger.info("=" * 80)
    logger.info("Processing LoUP-CP: Leaf-only CP with upward projection")
    logger.info("=" * 80)
    
    # Check if we have all required data
    if leaf_vocab_global is None or probs_test_leaf is None or q_hat_leaf is None or idx_test_leaf_seen is None:
        logger.warning("LoUP-CP requires leaf level data. Skipping LoUP-CP.")
        return
    elif m_leaf == 0:
        logger.warning("LoUP-CP requires non-empty leaf calibration set. Skipping LoUP-CP.")
        return
    
    # Save LoUP-CP files to the provided out_dir_LoUPCP (or fallback to out_dir if not provided)
    if out_dir_LoUPCP is None:
        out_dir_LoUPCP = out_dir
    os.makedirs(out_dir_LoUPCP, exist_ok=True)
    
    # Extract run_idx from path (e.g., .../run_1 -> 1)
    norm_out = os.path.normpath(out_dir)
    run_idx = 1  # default
    try:
        path_parts = norm_out.split(os.sep)
        for part in path_parts:
            if part.startswith("run_") and len(part) > 4:
                try:
                    run_idx = int(part[4:])
                    break
                except ValueError:
                    pass
    except Exception:
        logger.warning(f"Could not extract run_idx from path {norm_out}, using default {run_idx}")
    
    # Build global hierarchy maps from dataframe
    logger.info("Building global hierarchy maps from dataframe...")
    leaf_to_major_global, major_to_family_global, leaf_to_family_global = build_hierarchy_mappings(df, logger)
    
    # Build incidence matrices for projection
    logger.info("Building incidence matrices for upward projection...")
    A_major, A_family, maj_g2l, fam_g2l = build_vocab_maps_for_B(
        leaf_vocab_global,
        major_vocab_global,
        family_vocab_global,
        leaf_to_major_global,
        major_to_family_global,
        leaf_to_family_global,
        logger,
    )
    
    # LoUP-CP quantile: reuse Lw-CP's leaf quantile
    q_hat_LoUPCP = q_hat_leaf
    logger.info(f"LoUP-CP using leaf quantile: q_hat_LoUPCP = {q_hat_LoUPCP:.6f} (reused from Lw-CP leaf alpha={alpha_single})")
    
    # Test-time hook for LoUP-CP: reuse Lw-CP's test tensors
    # Filter test samples to closed-set at leaf level (same as A)
    # Note: probs_test_leaf and y_test_leaf_global_aligned are already aligned from A's processing
    y_test_leaf_global = y_test_leaf_global_aligned
    
    # Build leaf local mapping for LoUP-CP
    # Use integer keys to match the types in labels
    leaf_g2l_LoUPCP = {int(g): i for i, g in enumerate(leaf_vocab_global)}
    test_leaf_seen_mask = np.array([int(g) in leaf_g2l_LoUPCP for g in y_test_leaf_global])
    
    # Apply leaf closed-set filter
    probs_test_leaf_subset = probs_test_leaf[test_leaf_seen_mask]
    y_test_leaf_global_subset = y_test_leaf_global[test_leaf_seen_mask]
    idx_test_leaf_subset = idx_test_leaf_seen[test_leaf_seen_mask]
    
    # Also get major and family labels for these filtered samples
    y_test_major_global = df.iloc[idx_test_leaf_subset]["major_idx"].astype("Int64").values
    y_test_family_global = df.iloc[idx_test_leaf_subset]["family_idx"].astype("Int64").values
    
    logger.info(
        f"LoUP-CP test filtering: {len(y_test_leaf_global)} total, "
        f"{test_leaf_seen_mask.sum()} with leaf in vocab"
    )
    
    # Build leaf keep masks for LoUP-CP
    scores_leaf = 1.0 - probs_test_leaf_subset
    keep_leaf_bool = (scores_leaf <= q_hat_LoUPCP)  # [N, m_leaf]
    
    # Project upward
    keep_major_bool, keep_family_bool = project_leaf_masks_upward(
        keep_leaf_bool, A_major, A_family
    )
    
    # Map true labels to local indices aligned with Lw-CP matrices
    # Leaf level - convert to int for dictionary lookup consistency
    y_leaf_local = np.array([leaf_g2l_LoUPCP[int(g)] for g in y_test_leaf_global_subset], dtype="int64")
    
    # Major level: filter to samples where major is in vocab
    maj_in_vocab_mask = np.array([g is not None and not pd.isna(g) and int(g) in maj_g2l 
                                 for g in y_test_major_global])
    y_test_major_global_filtered = y_test_major_global[maj_in_vocab_mask]
    y_major_local = np.array([maj_g2l[int(g)] for g in y_test_major_global_filtered], dtype="int64")
    keep_major_bool_filtered = keep_major_bool[maj_in_vocab_mask]
    
    # Family level: filter to samples where family is in vocab
    fam_in_vocab_mask = np.array([g is not None and not pd.isna(g) and int(g) in fam_g2l 
                                 for g in y_test_family_global])
    y_test_family_global_filtered = y_test_family_global[fam_in_vocab_mask]
    y_family_local = np.array([fam_g2l[int(g)] for g in y_test_family_global_filtered], dtype="int64")
    keep_family_bool_filtered = keep_family_bool[fam_in_vocab_mask]
    
    logger.info(
        f"LoUP-CP per-level test filtering: "
        f"major: {maj_in_vocab_mask.sum()}/{len(maj_in_vocab_mask)} in vocab, "
        f"family: {fam_in_vocab_mask.sum()}/{len(fam_in_vocab_mask)} in vocab"
    )
    
    # Evaluate LoUP-CP
    logger.info("LoUP-CP: Reusing calibration from Lw-CP (no separate calibration time)")
    test_start_time_LoUPCP = time.time()
    metrics_LoUPCP_leaf = evaluate_sets_from_masks(keep_leaf_bool, y_leaf_local)
    metrics_LoUPCP_major = evaluate_sets_from_masks(keep_major_bool_filtered, y_major_local)
    metrics_LoUPCP_family = evaluate_sets_from_masks(keep_family_bool_filtered, y_family_local)
    test_time_LoUPCP = time.time() - test_start_time_LoUPCP
    logger.info(f"LoUP-CP testing: num_test_samples={len(y_leaf_local)}, time={test_time_LoUPCP:.4f}s")
    
    # Compute top1_acc_non_empty and base_top1_acc at leaf level
    # Initialize variables for use in major/family calculations
    base_top1_leaf = None
    top1_leaf = None
    base_top1_leaf_global = None
    top1_leaf_global = None
    
    if keep_leaf_bool.shape[0] > 0:
        # Base top1: argmax from all probs (unmasked)
        base_top1_leaf = probs_test_leaf_subset.argmax(axis=1)
        metrics_LoUPCP_leaf["base_top1_acc"] = float(np.mean(base_top1_leaf == y_leaf_local))
        
        # Map to global indices for use in major/family calculations
        base_top1_leaf_global = np.array([leaf_vocab_global[base_top1_leaf[i]] for i in range(len(base_top1_leaf))])
        
        # Top1 within set: argmax from masked probs (only classes in set)
        masked_probs_leaf = np.where(keep_leaf_bool, probs_test_leaf_subset, -np.inf)
        top1_leaf = masked_probs_leaf.argmax(axis=1)
        non_empty_leaf = keep_leaf_bool.sum(axis=1) > 0
        if np.any(non_empty_leaf):
            metrics_LoUPCP_leaf["top1_acc_non_empty"] = float(np.mean((top1_leaf[non_empty_leaf] == y_leaf_local[non_empty_leaf])))
        else:
            metrics_LoUPCP_leaf["top1_acc_non_empty"] = float("nan")
        
        # Map to global indices for use in major/family calculations
        top1_leaf_global = np.array([leaf_vocab_global[top1_leaf[i]] for i in range(len(top1_leaf))])
    else:
        metrics_LoUPCP_leaf["top1_acc_non_empty"] = float("nan")
        metrics_LoUPCP_leaf["base_top1_acc"] = float("nan")
    
    # Compute top1_acc_non_empty and base_top1_acc for major level
    # Use leaf argmax predictions mapped up to major via hierarchy
    if keep_major_bool_filtered.shape[0] > 0 and base_top1_leaf_global is not None:
        # Base top1: leaf argmax mapped to major
        base_top1_major_global = np.array([
            leaf_to_major_global.get(int(g), None) if g is not None else None
            for g in base_top1_leaf_global
        ])
        # Filter to samples where major is in vocab and mapped prediction is valid
        maj_valid_mask = (maj_in_vocab_mask & 
                         np.array([g is not None and int(g) in maj_g2l for g in base_top1_major_global]))
        if np.any(maj_valid_mask):
            base_top1_major_local = np.array([maj_g2l[int(g)] for g in base_top1_major_global[maj_valid_mask]], dtype="int64")
            # Align mask to compressed y_major_local (which is filtered by maj_in_vocab_mask)
            y_major_local_valid = y_major_local[maj_valid_mask[maj_in_vocab_mask]]
            metrics_LoUPCP_major["base_top1_acc"] = float(np.mean(base_top1_major_local == y_major_local_valid))
        else:
            metrics_LoUPCP_major["base_top1_acc"] = float("nan")
        
        # Top1 within set: use leaf top1 (within set) mapped to major
        if top1_leaf_global is not None:
            top1_major_global = np.array([
                leaf_to_major_global.get(int(g), None) if g is not None else None
                for g in top1_leaf_global
            ])
            # Filter to samples where major is in vocab and prediction is valid and in set
            maj_valid_mask = (maj_in_vocab_mask & 
                             np.array([g is not None and int(g) in maj_g2l for g in top1_major_global]))
            maj_in_set_mask = maj_valid_mask & (keep_major_bool.sum(axis=1) > 0)
            if np.any(maj_in_set_mask):
                top1_major_local = np.array([maj_g2l[int(g)] for g in top1_major_global[maj_in_set_mask]], dtype="int64")
                # Align mask to compressed y_major_local (filtered by maj_in_vocab_mask)
                y_major_local_in_set = y_major_local[maj_in_set_mask[maj_in_vocab_mask]]
                metrics_LoUPCP_major["top1_acc_non_empty"] = float(np.mean(top1_major_local == y_major_local_in_set))
            else:
                metrics_LoUPCP_major["top1_acc_non_empty"] = float("nan")
        else:
            metrics_LoUPCP_major["top1_acc_non_empty"] = float("nan")
    else:
        metrics_LoUPCP_major["top1_acc_non_empty"] = float("nan")
        metrics_LoUPCP_major["base_top1_acc"] = float("nan")
    
    # Compute top1_acc_non_empty and base_top1_acc for family level
    # Use leaf argmax predictions mapped up to family via hierarchy
    if keep_family_bool_filtered.shape[0] > 0 and base_top1_leaf_global is not None:
        # Base top1: leaf argmax mapped to family
        base_top1_family_global = np.array([
            leaf_to_family_global.get(int(g), None) if g is not None else None
            for g in base_top1_leaf_global
        ])
        # Filter to samples where family is in vocab and mapped prediction is valid
        fam_valid_mask = (fam_in_vocab_mask & 
                         np.array([g is not None and int(g) in fam_g2l for g in base_top1_family_global]))
        if np.any(fam_valid_mask):
            base_top1_family_local = np.array([fam_g2l[int(g)] for g in base_top1_family_global[fam_valid_mask]], dtype="int64")
            # Align mask to compressed y_family_local (which is filtered by fam_in_vocab_mask)
            y_family_local_valid = y_family_local[fam_valid_mask[fam_in_vocab_mask]]
            metrics_LoUPCP_family["base_top1_acc"] = float(np.mean(base_top1_family_local == y_family_local_valid))
        else:
            metrics_LoUPCP_family["base_top1_acc"] = float("nan")
        
        # Top1 within set: use leaf top1 (within set) mapped to family
        if top1_leaf_global is not None:
            top1_family_global = np.array([
                leaf_to_family_global.get(int(g), None) if g is not None else None
                for g in top1_leaf_global
            ])
            # Filter to samples where family is in vocab and prediction is valid and in set
            fam_valid_mask = (fam_in_vocab_mask & 
                             np.array([g is not None and int(g) in fam_g2l for g in top1_family_global]))
            fam_in_set_mask = fam_valid_mask & (keep_family_bool.sum(axis=1) > 0)
            if np.any(fam_in_set_mask):
                top1_family_local = np.array([fam_g2l[int(g)] for g in top1_family_global[fam_in_set_mask]], dtype="int64")
                # Align mask to compressed y_family_local (filtered by fam_in_vocab_mask)
                y_family_local_in_set = y_family_local[fam_in_set_mask[fam_in_vocab_mask]]
                metrics_LoUPCP_family["top1_acc_non_empty"] = float(np.mean(top1_family_local == y_family_local_in_set))
            else:
                metrics_LoUPCP_family["top1_acc_non_empty"] = float("nan")
        else:
            metrics_LoUPCP_family["top1_acc_non_empty"] = float("nan")
    else:
        metrics_LoUPCP_family["top1_acc_non_empty"] = float("nan")
        metrics_LoUPCP_family["base_top1_acc"] = float("nan")

    # Compute HIR for LoUP-CP using projected sets
    hir_family_sets: Dict[int, set] = {}
    hir_major_sets: Dict[int, set] = {}
    hir_leaf_sets: Dict[int, set] = {}
    for i, global_idx in enumerate(idx_test_leaf_subset):
        hir_leaf_sets[int(global_idx)] = set(
            int(leaf_vocab_global[j]) for j in np.where(keep_leaf_bool[i])[0]
        )
        hir_major_sets[int(global_idx)] = set(
            int(major_vocab_global[j]) for j in np.where(keep_major_bool[i])[0]
        )
        hir_family_sets[int(global_idx)] = set(
            int(family_vocab_global[j]) for j in np.where(keep_family_bool[i])[0]
        )

    hir_rate_LoUPCP = compute_hir_rate(
        hir_family_sets,
        hir_major_sets,
        hir_leaf_sets,
        leaf_to_major_global,
        major_to_family_global,
    )
    metrics_LoUPCP_family["hir"] = hir_rate_LoUPCP
    
    # Prepare labels for artifacts (same order as A)
    idx2id_map_leaf = {v: k for k, v in maps["leaf_id2idx"].items()}
    class_labels_leaf = [str(idx2id_map_leaf.get(int(gidx), str(int(gidx)))) for gidx in leaf_vocab_global]
    
    idx2id_map_major = {v: k for k, v in maps["major_id2idx"].items()}
    class_labels_major = [str(idx2id_map_major.get(int(gidx), str(int(gidx)))) for gidx in major_vocab_global]
    
    idx2id_map_family = {v: k for k, v in maps["family_id2idx"].items()}
    class_labels_family = [str(idx2id_map_family.get(int(gidx), str(int(gidx)))) for gidx in family_vocab_global]
    
    # Save LoUP-CP artifacts
    # Leaf summary
    summary_LoUPCP_leaf = {
        "level": "leaf",
        "alpha_LoUPCP": alpha_single,  # Note: reused Lw-CP leaf alpha
        "m_cal": m_leaf,
        "q_hat_leaf_LoUPCP": q_hat_LoUPCP,
        "metrics": metrics_LoUPCP_leaf,
        "num_test_defined": int(len(idx_test_leaf_defined)),
        "num_test_used": int(y_leaf_local.shape[0]),
        "class_labels": class_labels_leaf,
    }
    save_loupcp_cp_artifacts(out_dir_LoUPCP, "leaf", summary_LoUPCP_leaf, alpha_single, run_idx)
    logger.info(f"Saved LoUP-CP CP summary for leaf -> {out_dir_LoUPCP}/LoUPCP_alpha_{alpha_single}_run_{run_idx}_cp_leaf_summary.json")
    
    # Major summary
    summary_LoUPCP_major = {
        "level": "major",
        "alpha_LoUPCP": alpha_single,
        "m_cal": m_leaf,
        "q_hat_leaf_LoUPCP": q_hat_LoUPCP,
        "metrics": metrics_LoUPCP_major,
        "num_test_defined": int(len(idx_test_leaf_defined)),
        "num_test_used": int(y_major_local.shape[0]),
        "class_labels": class_labels_major,
    }
    save_loupcp_cp_artifacts(out_dir_LoUPCP, "major", summary_LoUPCP_major, alpha_single, run_idx)
    logger.info(f"Saved LoUP-CP CP summary for major -> {out_dir_LoUPCP}/LoUPCP_alpha_{alpha_single}_run_{run_idx}_cp_major_summary.json")
    
    # Family summary
    summary_LoUPCP_family = {
        "level": "family",
        "alpha_LoUPCP": alpha_single,
        "m_cal": m_leaf,
        "q_hat_leaf_LoUPCP": q_hat_LoUPCP,
        "metrics": metrics_LoUPCP_family,
        "num_test_defined": int(len(idx_test_leaf_defined)),
        "num_test_used": int(y_family_local.shape[0]),
        "class_labels": class_labels_family,
    }
    save_loupcp_cp_artifacts(out_dir_LoUPCP, "family", summary_LoUPCP_family, alpha_single, run_idx)
    logger.info(f"Saved LoUP-CP CP summary for family -> {out_dir_LoUPCP}/LoUPCP_alpha_{alpha_single}_run_{run_idx}_cp_family_summary.json")
    
    # Build per-sample details for LoUP-CP
    # Leaf predictions (argmax from masked probs)
    if keep_leaf_bool.shape[0] > 0:
        masked_probs_leaf = np.where(keep_leaf_bool, probs_test_leaf_subset, -np.inf)
        top1_leaf_local = masked_probs_leaf.argmax(axis=1)
        # Map leaf local predictions to global indices
        top1_leaf_global = np.array([leaf_vocab_global[top1_leaf_local[i]] for i in range(len(top1_leaf_local))])
        
        # Map leaf predictions to major and family using hierarchy
        top1_major_global = np.array([
            leaf_to_major_global.get(int(g), None) if g is not None else None
            for g in top1_leaf_global
        ])
        top1_family_global = np.array([
            leaf_to_family_global.get(int(g), None) if g is not None else None
            for g in top1_leaf_global
        ])
        
        # Get true labels for test samples
        y_test_leaf_global_full = df.iloc[idx_test_leaf_subset]["leaf_idx"].astype("Int64").values
        y_test_major_global_full = df.iloc[idx_test_leaf_subset]["major_idx"].astype("Int64").values
        y_test_family_global_full = df.iloc[idx_test_leaf_subset]["family_idx"].astype("Int64").values
        
        # Build sets from masks (convert boolean to label strings)
        def mask_to_label_set(mask_row, vocab_global, idx2id_map):
            """Convert a boolean mask row to a pipe-separated label string."""
            kept_indices = np.where(mask_row)[0]
            if len(kept_indices) == 0:
                return ""
            kept_global = [vocab_global[i] for i in kept_indices]
            kept_labels = [str(idx2id_map.get(int(g), str(int(g)))) for g in kept_global]
            return "|".join(kept_labels)
        
        # Build per-level details for LoUP-CP
        levels = ["family", "major", "leaf"]
        per_level_details_LoUPCP = {"family": {}, "major": {}, "leaf": {}}
        for i, global_row_idx in enumerate(idx_test_leaf_subset):
            # Leaf level
            # Use Lw-CP's leaf model prediction for pred_leaf
            leaf_pred_label = base_pred_labels_from_LwCP.get("leaf", {}).get(int(global_row_idx), "")
            leaf_true_label = str(idx2id_map_leaf.get(int(y_test_leaf_global_full[i]), str(int(y_test_leaf_global_full[i])))) if pd.notna(y_test_leaf_global_full[i]) else ""
            leaf_set = mask_to_label_set(keep_leaf_bool[i], leaf_vocab_global, idx2id_map_leaf)
            per_level_details_LoUPCP["leaf"][int(global_row_idx)] = {
                "true_leaf": leaf_true_label,
                "pred_leaf": leaf_pred_label,
                "set_leaf": leaf_set,
            }
            
            # Major level (use full upward closure for all test samples)
            if pd.notna(y_test_major_global_full[i]):
                # Use Lw-CP's major model prediction for pred_major
                major_pred_label = base_pred_labels_from_LwCP.get("major", {}).get(int(global_row_idx), "")
                major_true_label = str(idx2id_map_major.get(int(y_test_major_global_full[i]), str(int(y_test_major_global_full[i]))))
                # Use full keep_major_bool (not filtered) for all test samples
                major_set = mask_to_label_set(keep_major_bool[i], major_vocab_global, idx2id_map_major)
                per_level_details_LoUPCP["major"][int(global_row_idx)] = {
                    "true_major": major_true_label,
                    "pred_major": major_pred_label,
                    "set_major": major_set,
                }
            
            # Family level (use full upward closure for all test samples)
            if pd.notna(y_test_family_global_full[i]):
                # Use Lw-CP's family model prediction for pred_family
                family_pred_label = base_pred_labels_from_LwCP.get("family", {}).get(int(global_row_idx), "")
                family_true_label = str(idx2id_map_family.get(int(y_test_family_global_full[i]), str(int(y_test_family_global_full[i]))))
                # Use full keep_family_bool (not filtered) for all test samples
                family_set = mask_to_label_set(keep_family_bool[i], family_vocab_global, idx2id_map_family)
                per_level_details_LoUPCP["family"][int(global_row_idx)] = {
                    "true_family": family_true_label,
                    "pred_family": family_pred_label,
                    "set_family": family_set,
                }
        
        # Merge per-level details into a single DataFrame and save for LoUP-CP
        # Use only test set indices that have at least one level prediction
        test_indices_with_predictions_LoUPCP = set()
        for lvl in levels:
            test_indices_with_predictions_LoUPCP.update(per_level_details_LoUPCP[lvl].keys())
        test_indices_sorted_LoUPCP = sorted([int(i) for i in test_indices_with_predictions_LoUPCP])
        rows_LoUPCP = []
        for idx in test_indices_sorted_LoUPCP:
            # Only include rows that are in the test set
            if idx not in idx_test_leaf_defined:
                continue
            row = {"index": idx}
            for lvl in levels:
                det = per_level_details_LoUPCP[lvl].get(idx, {})
                row.update({
                    f"true_{lvl}": det.get(f"true_{lvl}", ""),
                    f"pred_{lvl}": det.get(f"pred_{lvl}", ""),
                    f"set_{lvl}": det.get(f"set_{lvl}", ""),
                })
            rows_LoUPCP.append(row)
        if rows_LoUPCP:
            details_df_LoUPCP = pd.DataFrame(rows_LoUPCP).set_index("index").sort_index()
            details_csv_LoUPCP = os.path.join(out_dir_LoUPCP, f"LoUPCP_alpha_{alpha_single}_run_{run_idx}_test_samples.csv")
            details_df_LoUPCP.to_csv(details_csv_LoUPCP)
            logger.info(f"Saved per-sample run details for LoUP-CP (test set only) -> {details_csv_LoUPCP}")
    
    logger.info("LoUP-CP one-shot CP experiment completed")


def run_lwcp_one_shot(
    logger: logging.Logger,
    df: pd.DataFrame,
    maps: Dict,
    input_params: Dict,
    out_dir: str,
    bundle_per_level: Dict[str, Dict],
    out_dir_LoUPCP: Optional[str] = None,
):
    seed = int(input_params.get("seed", 42))

    # Read method configuration
    methods = input_params.get("methods", ["LwCP"])
    if isinstance(methods, str):
        methods = [methods]
    compute_LoUPCP = "LoUPCP" in methods
    if compute_LoUPCP:
        logger.info("LoUP-CP enabled: will compute leaf-only CP with upward projection")

    # Ensure ancestor consistency before any split
    df = ensure_ancestor_consistency(df, maps, logger)

    # Build hierarchy mappings for HIR and LoUP-CP
    leaf_to_major_global, major_to_family_global, leaf_to_family_global = build_hierarchy_mappings(df, logger)

    # Build masks for defined labels (handle token-based undefined where available)
    if "family_id" in df.columns:
        mask_family_defined = df["family_id"].notna().values
    else:
        mask_family_defined = df["family_idx"].notna().values

    if "major_id" in df.columns:
        mask_major_defined = (df["major_id"].notna() & (df["major_id"] != "<MUnk>")).values
    else:
        mask_major_defined = df["major_idx"].notna().values

    if "leaf_id" in df.columns:
        mask_leaf_defined = (df["leaf_id"].notna() & (df["leaf_id"] != "<mUnk>")).values
    else:
        mask_leaf_defined = df["leaf_idx"].notna().values

    # Per-run split: stratified by leaf; create distinct Cal/Test per run
    split_props = input_params.get("split_proportions", {"cal": 0.5, "test": 0.5})
    idx_cal, idx_test = perform_two_stage_stratified_split(
        df, "leaf_idx", split_props, seed, logger
    )

    def per_class_counts(indexes: np.ndarray):
        sub = df.iloc[indexes]
        vals, cnts = np.unique(sub["leaf_idx"].dropna().astype("int64").values, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    def defined_counts(mask: np.ndarray, indexes: np.ndarray) -> Dict[str, int]:
        total = int(indexes.size)
        defined = int(mask[indexes].sum())
        return {"defined": defined, "undefined": int(total - defined), "total": total}

    split_info = {
        "sizes": {
            "cal": int(idx_cal.size),
            "test": int(idx_test.size),
        },
        "defined_counts": {
            "family": {
                "cal": defined_counts(mask_family_defined, idx_cal),
                "test": defined_counts(mask_family_defined, idx_test),
            },
            "major": {
                "cal": defined_counts(mask_major_defined, idx_cal),
                "test": defined_counts(mask_major_defined, idx_test),
            },
            "leaf": {
                "cal": defined_counts(mask_leaf_defined, idx_cal),
                "test": defined_counts(mask_leaf_defined, idx_test),
            },
        },
        "leaf_counts": {
            "cal": per_class_counts(idx_cal),
            "test": per_class_counts(idx_test),
        },
    }
    save_split_summary(out_dir, split_info)
    logger.info("Saved split summary")

    # Per-level processing with a single alpha shared across levels
    levels = ["family", "major", "leaf"]
    alpha_single = float(input_params.get("alpha", 0.10))

    # Collect per-sample details per level, then merge after loop
    per_level_details: Dict[str, Dict[int, Dict[str, str]]] = {lvl: {} for lvl in levels}
    per_level_sets_global: Dict[str, Dict[int, set]] = {lvl: {} for lvl in levels}
    
    # For LoUP-CP: collect vocabs and leaf model artifacts
    family_vocab_global = None
    major_vocab_global = None
    leaf_vocab_global = None
    probs_test_leaf = None
    probs_cal_leaf = None
    model_leaf = None
    idx_test_leaf_defined = None
    idx_test_leaf_seen = None
    idx_cal_leaf_seen = None
    y_test_leaf_global_aligned = None
    y_cal_leaf_global_aligned = None
    q_hat_leaf = None
    m_leaf = None

    for level in levels:
        logger.info("=" * 80)
        logger.info(f"Processing level: {level} (pretrained model)")
        logger.info("=" * 80)

        # Level-specific mask
        if level == "family":
            mask_defined = mask_family_defined
        elif level == "major":
            mask_defined = mask_major_defined
        else:
            mask_defined = mask_leaf_defined

        # Cal/Test indices for defined labels only
        idx_cal_k = idx_cal[mask_defined[idx_cal]]
        idx_test_k = idx_test[mask_defined[idx_test]]
        if idx_cal_k.size == 0 or idx_test_k.size == 0:
            logger.warning(f"No cal/test data for level={level}; skipping.")
            continue

        # Load pretrained bundle (model + preprocessing)
        bundle = bundle_per_level[level]
        feature_cols = list(bundle.get("categorical_features", [])) + list(bundle.get("numerical_features", []))
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Compute class ordering (global indices) to align with model outputs
        classes_global = bundle.get("classes_")
        if classes_global is None:
            logger.warning(
                f"classes_ not found in bundle for {level} level. "
                "Falling back to dataset classes, which may cause misalignment with model outputs. "
                "Please retrain the model to regenerate artifacts with classes_."
            )
            # Use all defined labels in the dataset to avoid truncating classes not present in this split
            classes_global = np.unique(df.loc[df[f"{level}_idx"].notna(), f"{level}_idx"].astype("int64").values)
        else:
            logger.info(f"Using classes_ from bundle for {level} level: {len(classes_global)} classes")
        class_to_local = {c: i for i, c in enumerate(classes_global)}

        # Calibration data (seen classes only)
        y_cal_global = df.iloc[idx_cal_k][f"{level}_idx"].astype("int64").values
        cal_seen_mask = np.array([g in class_to_local for g in y_cal_global])
        idx_cal_seen = idx_cal_k[cal_seen_mask]
        if idx_cal_seen.size == 0:
            raise ValueError(f"Calibration set empty for level='{level}' after filtering to seen classes.")
        probs_cal = predict_proba(bundle, df.iloc[idx_cal_seen][feature_cols])
        # Validate probability dimensions match expected classes
        if probs_cal.shape[1] != len(classes_global):
            raise ValueError(
                f"Probability dimension mismatch for {level} level: "
                f"model outputs {probs_cal.shape[1]} classes, but classes_global has {len(classes_global)} classes. "
                "This indicates a misalignment between model training and inference. "
                "Please retrain the model to ensure consistency."
            )
        y_cal_local = np.array([class_to_local[g] for g in y_cal_global[cal_seen_mask]], dtype="int64")

        # Test data (seen classes only)
        y_test_global = df.iloc[idx_test_k][f"{level}_idx"].astype("int64").values
        test_seen_mask = np.array([g in class_to_local for g in y_test_global])
        idx_test_seen = idx_test_k[test_seen_mask]
        probs_test = predict_proba(bundle, df.iloc[idx_test_seen][feature_cols]) if idx_test_seen.size > 0 else np.empty((0, len(classes_global)))
        # Validate probability dimensions match expected classes
        if idx_test_seen.size > 0 and probs_test.shape[1] != len(classes_global):
            raise ValueError(
                f"Probability dimension mismatch for {level} level (test): "
                f"model outputs {probs_test.shape[1]} classes, but classes_global has {len(classes_global)} classes. "
                "This indicates a misalignment between model training and inference. "
                "Please retrain the model to ensure consistency."
            )
        y_test_local = np.array([class_to_local[g] for g in y_test_global[test_seen_mask]], dtype="int64")

        # Calibration quantile
        rows_cal = np.arange(y_cal_local.shape[0])
        true_probs_cal = probs_cal[rows_cal, y_cal_local]
        alpha_k = alpha_single
        cal_start_time = time.time()
        q_hat, m_k, j_k, step_k = calibrate_split_cp(true_probs_cal, alpha_k)
        cal_time = time.time() - cal_start_time
        logger.info(
            f"Calibration: level={level}, m_k={m_k}, j={j_k}, step={step_k:.6f}, q_hat={q_hat:.6f}, time={cal_time:.4f}s"
        )

        # LoUP-CP capture (leaf only)
        if compute_LoUPCP and level == "leaf":
            q_hat_leaf = q_hat
            m_leaf = m_k
            probs_test_leaf = probs_test
            probs_cal_leaf = probs_cal
            model_leaf = bundle["model"]
            idx_test_leaf_defined = idx_test_k
            idx_test_leaf_seen = idx_test_seen
            idx_cal_leaf_seen = idx_cal_seen
            y_test_leaf_global_aligned = y_test_global[test_seen_mask]
            y_cal_leaf_global_aligned = y_cal_global[cal_seen_mask]
            leaf_vocab_global = classes_global
        elif compute_LoUPCP and level == "major":
            major_vocab_global = classes_global
        elif compute_LoUPCP and level == "family":
            family_vocab_global = classes_global

        # Warn on tiny calibration set
        if m_k == 0:
            raise ValueError(
                f"Calibration set is empty for level='{level}' when using pretrained model."
            )
        if m_k < 5:
            logger.warning(
                f"Very small calibration set for level={level}: m_k={m_k} (quantile_step={step_k:.6f}). Results may be unstable."
            )

        # Evaluate on Test
        test_start_time = time.time()
        metrics = evaluate_cp_sets(probs_test, q_hat, y_test_local)
        if probs_test.shape[0] > 0:
            base_top1 = probs_test.argmax(axis=1)
            metrics["base_top1_acc"] = float(np.mean(base_top1 == y_test_local))
        else:
            metrics["base_top1_acc"] = float("nan")
        test_time = time.time() - test_start_time
        logger.info(
            f"Testing: level={level}, num_test_samples={probs_test.shape[0]}, time={test_time:.4f}s"
        )

        # Prepare labels for artifact (order aligns with classes_global / model outputs)
        idx2id_map = {v: k for k, v in maps[f"{level}_id2idx"].items()}
        class_labels = [str(idx2id_map.get(int(gidx), str(int(gidx)))) for gidx in classes_global]
        # Guard: if model outputs more classes than we have labels, pad with index-based labels to avoid KeyError
        if probs_test.shape[1] > len(class_labels):
            for extra in range(len(class_labels), probs_test.shape[1]):
                class_labels.append(str(extra))

        summary = {
            "level": level,
            "alpha": alpha_k,
            "m_k": m_k,
            "j": j_k,
            "quantile_step": step_k,
            "q_hat": q_hat,
            "metrics": metrics,
            "num_test_defined": int(idx_test_k.size),
            "num_test_used": int(y_test_local.shape[0]),
            "class_labels": class_labels,
        }
        save_cp_artifacts(out_dir, level, summary)
        logger.info(f"Saved CP summary for {level}")

        # Build per-sample details for this level (test set only, seen classes)
        local_to_label = {i: class_labels[i] for i in range(len(class_labels))}
        scores_all = 1.0 - probs_test
        keep_mask = scores_all <= q_hat
        sets_as_labels: List[str] = []
        base_preds_as_labels: List[str] = []
        for row in range(probs_test.shape[0]):
            kept_local = np.where(keep_mask[row])[0].tolist()
            kept_labels = [local_to_label[i] for i in kept_local]
            sets_as_labels.append("|".join(kept_labels))
            base_pred_local = int(probs_test[row].argmax())
            base_preds_as_labels.append(local_to_label[base_pred_local])

            # Track kept sets in global IDs for HIR
            kept_global = [int(classes_global[i]) for i in kept_local]
            per_level_sets_global[level][int(idx_test_seen[row])] = set(kept_global)

        true_labels_as_strings = [str(idx2id_map.get(int(g), str(int(g)))) for g in y_test_global[test_seen_mask]]

        for i, global_row_idx in enumerate(idx_test_seen):
            per_level_details[level][int(global_row_idx)] = {
                f"true_{level}": true_labels_as_strings[i],
                f"pred_{level}": base_preds_as_labels[i],
                f"set_{level}": sets_as_labels[i],
            }

    # Merge per-level details into a single DataFrame and save
    # Use only test set indices that have at least one level prediction
    if idx_test.size > 0:
        # Collect all test indices that have predictions for at least one level
        test_indices_with_predictions = set()
        for lvl in levels:
            test_indices_with_predictions.update(per_level_details[lvl].keys())
        test_indices_sorted = sorted([int(i) for i in test_indices_with_predictions])
        rows = []
        for idx in test_indices_sorted:
            # Only include rows that are in the test set
            if idx not in idx_test:
                continue
            row = {"index": idx}
            for lvl in levels:
                det = per_level_details[lvl].get(idx, {})
                row.update({
                    f"true_{lvl}": det.get(f"true_{lvl}", ""),
                    f"pred_{lvl}": det.get(f"pred_{lvl}", ""),
                    f"set_{lvl}": det.get(f"set_{lvl}", ""),
                })
            rows.append(row)
        details_df = pd.DataFrame(rows).set_index("index").sort_index()
        # Extract run_idx from path (e.g., .../run_1 -> 1)
        norm_out = os.path.normpath(out_dir)
        run_idx = 1
        try:
            for part in norm_out.split(os.sep):
                if part.startswith("run_") and len(part) > 4:
                    run_idx = int(part[4:])
                    break
        except Exception:
            pass
        details_csv = os.path.join(out_dir, f"LwCP_alpha_{alpha_single}_run_{run_idx}_test_samples.csv")
        details_df.to_csv(details_csv)
        logger.info(f"Saved per-sample run details (test set only) -> {details_csv}")

    # Compute HIR across levels (family-major-leaf) using kept sets
    hir_rate = compute_hir_rate(
        per_level_sets_global["family"],
        per_level_sets_global["major"],
        per_level_sets_global["leaf"],
        leaf_to_major_global,
        major_to_family_global,
    )
    if not math.isnan(hir_rate):
        try:
            # Update family summary to include HIR
            fam_json = os.path.join(out_dir, "cp_family_summary.json")
            if os.path.exists(fam_json):
                with open(fam_json, "r") as f:
                    fam_summary = json.load(f)
                fam_summary.setdefault("metrics", {})["hir"] = hir_rate
                save_cp_artifacts(out_dir, "family", fam_summary)
                logger.info(f"Added HIR={hir_rate:.6f} to family summary")
            else:
                logger.warning("Family summary not found; HIR not persisted")
        except Exception as e:
            logger.warning(f"Failed to persist HIR metric: {e}")
    else:
        logger.warning("HIR could not be computed (insufficient overlapping samples)")

    logger.info("Lw-CP one-shot CP experiment completed")
    
    # Build mapping of Lw-CP base predictions per level for reuse in LoUP-CP run_details
    base_pred_labels_from_LwCP: Dict[str, Dict[int, str]] = {
        lvl: {idx: det.get(f"pred_{lvl}", "") for idx, det in per_level_details[lvl].items()} for lvl in levels
    }
    
    # LoUP-CP: Leaf-only CP with upward closure
    if compute_LoUPCP:
        run_loupcp(
            df=df,
            maps=maps,
            input_params=input_params,
            out_dir=out_dir,
            leaf_vocab_global=leaf_vocab_global,
            major_vocab_global=major_vocab_global,
            family_vocab_global=family_vocab_global,
            probs_test_leaf=probs_test_leaf,
            q_hat_leaf=q_hat_leaf,
            m_leaf=m_leaf,
            idx_test_leaf_defined=idx_test_leaf_defined,
            idx_test_leaf_seen=idx_test_leaf_seen,
            y_test_leaf_global_aligned=y_test_leaf_global_aligned,
            alpha_single=alpha_single,
            base_pred_labels_from_LwCP=base_pred_labels_from_LwCP,
            logger=logger,
            out_dir_LoUPCP=out_dir_LoUPCP,
        )


