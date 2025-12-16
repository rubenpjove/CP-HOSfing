import itertools
import logging
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, 
    mutual_info_classif, f_classif, chi2, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

from exps.predictors.src.cphos.models import MLP, make_dataloaders, compute_class_weights, seed_everything
from exps.utils.io_utils import log_model_info, log_training_progress


def _astype_str(X):
    """Cast incoming array/dataframe to string dtype for safe categorical encoding."""
    return X.astype(str)

def _pd_na_to_np_nan(X):
    """Replace pandas NA with numpy nan to keep sklearn happy on object dtypes."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.replace({pd.NA: np.nan})
    # Fallback for numpy arrays or other inputs
    try:
        return np.where(pd.isna(X), np.nan, X)
    except Exception:
        return X
def _make_ohe():
    """Create a version-compatible OneHotEncoder using sparse_output (new) or sparse (old)."""
    try:
        # Newer sklearn versions (>=1.2+) use sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    except TypeError:
        # Older sklearn versions use sparse
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)


def fit_transform_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.15, seed: int = 42):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y)
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(X_tr))
    X_va = scl.transform(imp.transform(X_va))
    return X_tr, X_va, y_tr, y_va, imp, scl


def expand_grid(param_grid: Dict[str, List[Any]], max_samples: int = None, random_state: int = 42):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    configs = [dict(zip(keys, vals)) for vals in combos]
    if (max_samples is not None) and (len(configs) > max_samples):
        import random as _random
        _random.seed(random_state)
        configs = _random.sample(configs, max_samples)
    return configs


def stratified_cv_splits(y: np.ndarray, desired_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    _, counts = np.unique(y, return_counts=True)
    max_splits = int(counts.min())
    n_splits = max(2, min(desired_splits, max_splits))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_feature_selection_methods(X_tr: np.ndarray, X_va: np.ndarray, y_tr: np.ndarray, 
                                     methods: List[str] = None, random_state: int = 42) -> Dict[str, Any]:
    """
    Evaluate different feature selection methods to find the best one.
    
    Args:
        X_tr: Training features
        X_va: Validation features  
        y_tr: Training labels
        methods: List of methods to evaluate
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with method performance results
    """
    if methods is None:
        methods = ["mutual_info", "f_classif", "rfe", "rf"]
    
    results = {}
    n_samples, n_features_total = X_tr.shape
    
    # Determine feature counts to test
    feature_counts = [
        min(20, n_features_total // 4),
        min(30, n_features_total // 3), 
        min(50, n_features_total // 2),
        min(100, n_features_total * 2 // 3)
    ]
    feature_counts = [f for f in feature_counts if f < n_features_total and f > 5]
    
    for method in methods:
        results[method] = {}
        for n_features in feature_counts:
            try:
                X_tr_sel, X_va_sel, selector, _ = apply_feature_selection(
                    X_tr, X_va, y_tr, method=method, n_features=n_features, random_state=random_state
                )
                
                # Quick evaluation with a simple classifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import f1_score
                
                rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1)
                rf.fit(X_tr_sel, y_tr)
                y_pred = rf.predict(X_va_sel)
                f1 = f1_score(y_tr, y_pred, average='macro')
                
                results[method][n_features] = {
                    'f1_score': f1,
                    'n_features_selected': X_tr_sel.shape[1],
                    'selector': selector
                }
                
            except Exception as e:
                results[method][n_features] = {'error': str(e)}
    
    return results


def select_best_feature_selection(X_tr: np.ndarray, X_va: np.ndarray, y_tr: np.ndarray, 
                                 random_state: int = 42) -> Tuple[str, int, Any]:
    """
    Automatically select the best feature selection method and parameters.
    
    Args:
        X_tr: Training features
        X_va: Validation features  
        y_tr: Training labels
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (best_method, best_n_features, best_selector)
    """
    # Evaluate different methods
    results = evaluate_feature_selection_methods(X_tr, X_va, y_tr, random_state=random_state)
    
    best_score = -1
    best_method = "mutual_info"
    best_n_features = 30
    best_selector = None
    
    # Find the best combination
    for method, method_results in results.items():
        for n_features, result in method_results.items():
            if 'f1_score' in result and result['f1_score'] > best_score:
                best_score = result['f1_score']
                best_method = method
                best_n_features = n_features
                best_selector = result.get('selector')
    
    # If no good results, fall back to mutual_info with 30 features
    if best_selector is None:
        X_tr_sel, X_va_sel, best_selector, _ = apply_feature_selection(
            X_tr, X_va, y_tr, method="mutual_info", n_features=30, random_state=random_state
        )
    
    return best_method, best_n_features, best_selector


def select_optimal_features(X_tr: np.ndarray, X_va: np.ndarray, y_tr: np.ndarray, 
                           method: str = "mutual_info", random_state: int = 42) -> int:
    """
    Automatically select the optimal number of features using multiple strategies.
    
    Args:
        X_tr: Training features
        X_va: Validation features  
        y_tr: Training labels
        method: Feature selection method
        random_state: Random state for reproducibility
        
    Returns:
        Optimal number of features to select
    """
    n_samples, n_features_total = X_tr.shape
    
    # Strategy 1: Percentile-based selection with validation
    # Test different percentiles and pick the one with best validation performance
    percentiles = [0.1, 0.2, 0.3, 0.5, 0.7]  # 10%, 20%, 30%, 50%, 70% of features
    feature_counts = [max(10, int(n_features_total * p)) for p in percentiles]
    feature_counts = [f for f in feature_counts if f < n_features_total and f > 5]
    
    best_score = -1
    best_n_features = min(50, n_features_total // 2)  # Default fallback
    
    for n_features in feature_counts:
        try:
            # Quick evaluation with a simple classifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import f1_score
            
            X_tr_sel, X_va_sel, _, _ = apply_feature_selection(
                X_tr, X_va, y_tr, method=method, n_features=n_features, random_state=random_state
            )
            
            # Quick evaluation
            rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1)
            rf.fit(X_tr_sel, y_tr)
            y_pred = rf.predict(X_va_sel)
            f1 = f1_score(y_tr, y_pred, average='macro')
            
            if f1 > best_score:
                best_score = f1
                best_n_features = n_features
                
        except Exception as e:
            continue
    
    return best_n_features


def select_optimal_features_statistical(X_tr: np.ndarray, y_tr: np.ndarray, 
                                       method: str = "mutual_info", 
                                       alpha: float = 0.05) -> int:
    """
    Select optimal number of features using statistical significance thresholds.
    
    Args:
        X_tr: Training features
        y_tr: Training labels
        method: Feature selection method
        alpha: Statistical significance threshold
        
    Returns:
        Number of statistically significant features
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    
    n_samples, n_features_total = X_tr.shape
    
    try:
        if method == "mutual_info":
            # For mutual information, we can't use p-values directly
            # Instead, use a threshold based on the distribution of scores
            selector = SelectKBest(score_func=mutual_info_classif, k='all')
            selector.fit(X_tr, y_tr)
            scores = selector.scores_
            
            # Use 75th percentile as threshold (top 25% of features)
            threshold = np.percentile(scores, 75)
            n_features = np.sum(scores >= threshold)
            
        elif method == "f_classif":
            # Use F-test with p-value threshold
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X_tr, y_tr)
            scores, p_values = f_classif(X_tr, y_tr)
            
            # Select features with p-value < alpha
            n_features = np.sum(p_values < alpha)
            
        else:
            # Fallback to percentile-based selection
            n_features = max(10, n_features_total // 4)
            
        # Ensure reasonable bounds
        n_features = max(10, min(n_features, n_features_total // 2))
        return n_features
        
    except Exception as e:
        # Fallback to conservative selection
        return max(10, n_features_total // 4)


def apply_feature_selection(X_tr: np.ndarray, X_va: np.ndarray, y_tr: np.ndarray, 
                          method: str = "mutual_info", n_features: int = None, 
                          random_state: int = 42, feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """
    Apply feature selection to training and validation data.
    
    Args:
        X_tr: Training features
        X_va: Validation features  
        y_tr: Training labels
        method: Feature selection method ('mutual_info', 'f_classif', 'chi2', 'rfe', 'lasso', 'rf')
        n_features: Number of features to select (auto if None)
        random_state: Random state for reproducibility
        feature_names: List of feature names for logging (optional)
        
    Returns:
        Tuple of (X_tr_selected, X_va_selected, selector, info_dict)
    """
    import time
    start_time = time.time()
    n_samples, n_features_total = X_tr.shape
    
    # Handle string "None" values that might come from YAML configuration
    if n_features == "None" or n_features == "none":
        n_features = None
    elif isinstance(n_features, str) and n_features.isdigit():
        n_features = int(n_features)
    
    # Handle case where n_features is None (use all features)
    if n_features is None:
        # Return all features without selection
        elapsed_time = time.time() - start_time
        info_dict = {
            'method': 'none',
            'n_features_requested': None,
            'n_features_selected': n_features_total,
            'n_features_original': n_features_total,
            'selected_indices': list(range(n_features_total)),
            'selected_feature_names': feature_names if feature_names else [],
            'elapsed_time': elapsed_time,
            'selector_type': 'NoSelection'
        }
        return X_tr, X_va, None, info_dict
    
    # Auto-determine number of features if not specified (this shouldn't happen now)
    if n_features is None:
        # Conservative approach: select 20-50% of features based on sample size
        if n_samples < 1000:
            n_features = min(20, n_features_total // 2)
        elif n_samples < 10000:
            n_features = min(50, n_features_total // 3)
        else:
            n_features = min(100, n_features_total // 4)
    
    # Ensure we don't select more features than available
    n_features = min(n_features, n_features_total)
    
    try:
        if method == "mutual_info":
            # Mutual information - good for non-linear relationships
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            
        elif method == "f_classif":
            # F-test - good for linear relationships
            selector = SelectKBest(score_func=f_classif, k=n_features)
            
        elif method == "chi2":
            # Chi-squared - good for categorical features
            # Ensure non-negative values for chi2
            X_tr_nonneg = np.maximum(X_tr, 0)
            X_va_nonneg = np.maximum(X_va, 0)
            selector = SelectKBest(score_func=chi2, k=n_features)
            X_tr_selected = selector.fit_transform(X_tr_nonneg, y_tr)
            X_va_selected = selector.transform(X_va_nonneg)
            return X_tr_selected, X_va_selected, selector
            
        elif method == "rfe":
            # Recursive Feature Elimination with Random Forest
            estimator = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
            
        elif method == "lasso":
            # Lasso-based feature selection
            lasso = LassoCV(cv=3, random_state=random_state, max_iter=1000)
            selector = SelectFromModel(lasso, threshold='median')
            
        elif method == "rf":
            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
            selector = SelectFromModel(rf, max_features=n_features)
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit and transform
        X_tr_selected = selector.fit_transform(X_tr, y_tr)
        X_va_selected = selector.transform(X_va)
        
        # Collect information about selected features
        elapsed_time = time.time() - start_time
        
        # Get selected feature indices
        if hasattr(selector, 'get_support'):
            selected_indices = selector.get_support(indices=True)
        else:
            # For some selectors, we need to infer from the shape change
            selected_indices = list(range(X_tr_selected.shape[1]))
        
        # Get selected feature names if available
        selected_feature_names = []
        if feature_names is not None and len(selected_indices) <= len(feature_names):
            selected_feature_names = [feature_names[i] for i in selected_indices if i < len(feature_names)]
        
        info_dict = {
            'method': method,
            'n_features_requested': n_features,
            'n_features_selected': X_tr_selected.shape[1],
            'n_features_original': X_tr.shape[1],
            'selected_indices': selected_indices.tolist() if hasattr(selected_indices, 'tolist') else selected_indices,
            'selected_feature_names': selected_feature_names,
            'elapsed_time': elapsed_time,
            'selector_type': type(selector).__name__
        }
        
        return X_tr_selected, X_va_selected, selector, info_dict
        
    except Exception as e:
        # Fallback to variance threshold if other methods fail
        print(f"Feature selection method {method} failed: {e}. Using VarianceThreshold as fallback.")
        selector = VarianceThreshold(threshold=1e-8)
        X_tr_selected = selector.fit_transform(X_tr)
        X_va_selected = selector.transform(X_va)
        
        elapsed_time = time.time() - start_time
        
        # Get selected feature indices for fallback
        if hasattr(selector, 'get_support'):
            selected_indices = selector.get_support(indices=True)
        else:
            selected_indices = list(range(X_tr_selected.shape[1]))
        
        # Get selected feature names if available
        selected_feature_names = []
        if feature_names is not None and len(selected_indices) <= len(feature_names):
            selected_feature_names = [feature_names[i] for i in selected_indices if i < len(feature_names)]
        
        info_dict = {
            'method': 'VarianceThreshold_fallback',
            'n_features_requested': n_features,
            'n_features_selected': X_tr_selected.shape[1],
            'n_features_original': X_tr.shape[1],
            'selected_indices': selected_indices.tolist() if hasattr(selected_indices, 'tolist') else selected_indices,
            'selected_feature_names': selected_feature_names,
            'elapsed_time': elapsed_time,
            'selector_type': 'VarianceThreshold',
            'error': str(e)
        }
        
        return X_tr_selected, X_va_selected, selector, info_dict


def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    params: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
    rng_seed: int = 42,
    fold_info: Dict[str, Any] = None,
    use_amp: bool = True,
    num_workers: int = 4,
):
    fold_logger = logging.getLogger(f"fold_{id(X_tr)}")

    # Convert string parameters to proper types (YAML parsing issue)
    if isinstance(params.get("lr"), str):
        params["lr"] = float(params["lr"])
    if isinstance(params.get("weight_decay"), str):
        params["weight_decay"] = float(params["weight_decay"])
    if isinstance(params.get("batch_size"), str):
        params["batch_size"] = int(params["batch_size"])
    if isinstance(params.get("max_epochs"), str):
        params["max_epochs"] = int(params["max_epochs"])
    if isinstance(params.get("patience"), str):
        params["patience"] = int(params["patience"])
    if isinstance(params.get("grad_clip"), str):
        params["grad_clip"] = float(params["grad_clip"])
    if isinstance(params.get("dropout"), str):
        params["dropout"] = float(params["dropout"])
    
    # Convert boolean parameters
    if isinstance(params.get("use_bn"), str):
        params["use_bn"] = params["use_bn"].lower() in ['true', '1', 'yes']
    if isinstance(params.get("use_undersample"), str):
        params["use_undersample"] = params["use_undersample"].lower() in ['true', '1', 'yes']
    if isinstance(params.get("use_smote"), str):
        params["use_smote"] = params["use_smote"].lower() in ['true', '1', 'yes']
    if isinstance(params.get("use_onecycle"), str):
        params["use_onecycle"] = params["use_onecycle"].lower() in ['true', '1', 'yes']

    # Log initial data information
    if fold_info:
        fold_logger.info(f"=== FOLD {fold_info.get('fold_number', '?')} DATA TRANSFORMATION LOG ===")
        fold_logger.info(f"Initial data shape: {fold_info.get('initial_shape', 'Unknown')}")
        fold_logger.info(f"Initial features - Categorical: {fold_info.get('initial_categorical', 'Unknown')}, Numerical: {fold_info.get('initial_numerical', 'Unknown')}")
        
        # Log class distribution before any transformations
        if 'initial_class_counts' in fold_info:
            fold_logger.info("Initial class distribution:")
            for class_id, count in fold_info['initial_class_counts'].items():
                fold_logger.info(f"  Class {class_id}: {count} samples")

    in_dim = X_tr.shape[1]
    unique_classes = np.unique(np.concatenate([y_tr, y_va]))
    class_to_local = {c: i for i, c in enumerate(unique_classes)}
    
    # Log class distribution after encoding but before any sampling
    fold_logger.info(f"After encoding - Class distribution:")
    for class_id in unique_classes:
        tr_count = np.sum(y_tr == class_id)
        va_count = np.sum(y_va == class_id)
        fold_logger.info(f"  Class {class_id}: {tr_count} train + {va_count} val = {tr_count + va_count} total")
    
    y_tr = np.vectorize(class_to_local.get)(y_tr)
    y_va = np.vectorize(class_to_local.get)(y_va)
    out_dim = int(unique_classes.size)
    fold_logger.info(f"Training fold: input_dim={in_dim}, output_dim={out_dim}")

    model = MLP(in_dim, params["hidden_dims"], out_dim,
                dropout=params["dropout"], use_bn=params.get("use_bn", True)).to(device)

    log_model_info(fold_logger, model, f"Fold Model")

    # Log class distribution before sampling
    classes_arr, counts_arr = np.unique(y_tr, return_counts=True)
    fold_logger.info(f"Before sampling - Training class distribution:")
    for c, cnt in zip(classes_arr, counts_arr):
        fold_logger.info(f"  Class {c}: {cnt} samples")
    fold_logger.info(f"Total training samples before sampling: {len(y_tr)}")

    # Optional class rebalancing: undersample extreme majority classes, then SMOTE with safe k
    if IMBLEARN_AVAILABLE:
        try:
            # Compute class distribution on local labels
            min_count = int(counts_arr.min()) if counts_arr.size > 0 else 0
            if min_count > 0 and params.get("use_undersample", True):
                # Cap majority classes to a multiple of the minority count
                max_ratio = params.get("undersample_max_ratio", 10)
                target_min = min_count
                sampling_strategy = {}
                cap = max_ratio * target_min
                for c, cnt in zip(classes_arr, counts_arr):
                    sampling_strategy[int(c)] = int(min(cnt, cap))
                
                fold_logger.info(f"Applying undersampling with max_ratio={max_ratio}, target_min={target_min}")
                fold_logger.info(f"Undersampling strategy: {sampling_strategy}")
                
                rus = RandomUnderSampler(random_state=rng_seed, sampling_strategy=sampling_strategy)
                X_tr_before_under = X_tr.shape[0]
                X_tr, y_tr = rus.fit_resample(X_tr, y_tr)
                X_tr_after_under = X_tr.shape[0]
                fold_logger.info(f"Undersampling: {X_tr_before_under} -> {X_tr_after_under} samples (removed {X_tr_before_under - X_tr_after_under})")
                
                # Log class distribution after undersampling
                classes_after_under, counts_after_under = np.unique(y_tr, return_counts=True)
                fold_logger.info(f"After undersampling - Class distribution:")
                for c, cnt in zip(classes_after_under, counts_after_under):
                    fold_logger.info(f"  Class {c}: {cnt} samples")
                    
            if params.get("use_smote", True):
                # Recompute after potential undersampling
                _, counts_arr2 = np.unique(y_tr, return_counts=True)
                safe_k = max(1, min(5, int(counts_arr2.min()) - 1)) if counts_arr2.size > 0 else 1
                if safe_k >= 1 and int(counts_arr2.min()) >= 2:
                    fold_logger.info(f"Applying SMOTE with k_neighbors={safe_k}")
                    X_tr_df = pd.DataFrame(X_tr)
                    y_tr_s = pd.Series(y_tr)
                    
                    X_tr_before_smote = X_tr.shape[0]
                    oversampler = SMOTE(random_state=rng_seed, k_neighbors=safe_k)
                    X_resampled, y_resampled = oversampler.fit_resample(X_tr_df, y_tr_s)
                    X_tr = X_resampled.values
                    y_tr = y_resampled.values
                    X_tr_after_smote = X_tr.shape[0]
                    fold_logger.info(f"SMOTE: {X_tr_before_smote} -> {X_tr_after_smote} samples (added {X_tr_after_smote - X_tr_before_smote})")
                    
                    # Log class distribution after SMOTE
                    classes_after_smote, counts_after_smote = np.unique(y_tr, return_counts=True)
                    fold_logger.info(f"After SMOTE - Class distribution:")
                    for c, cnt in zip(classes_after_smote, counts_after_smote):
                        fold_logger.info(f"  Class {c}: {cnt} samples")
                else:
                    fold_logger.info(f"SMOTE skipped: safe_k={safe_k}, min_count={counts_arr2.min() if counts_arr2.size > 0 else 0}")
        except Exception as e:
            fold_logger.warning(f"Rebalancing (undersample/SMOTE) failed or was skipped: {e}")
    
    fold_logger.info(f"Final training samples: {len(y_tr)}")
    fold_logger.info(f"Final validation samples: {len(y_va)}")

    class_weights = compute_class_weights(y_tr).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = None
    if params.get("use_onecycle", False):
        steps_per_epoch = math.ceil(len(y_tr) / params["batch_size"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=params["lr"], epochs=params["max_epochs"], steps_per_epoch=steps_per_epoch
        )
        fold_logger.info(f"Using OneCycleLR scheduler with {steps_per_epoch} steps per epoch")

    # Mixed precision training setup
    scaler = None
    if use_amp and device.type == 'cuda':
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        fold_logger.info("Using automatic mixed precision (AMP) training")
    else:
        fold_logger.info("Using full precision training")

    dl_tr, dl_va = make_dataloaders(X_tr, y_tr, X_va, y_va, batch_size=params["batch_size"], num_workers=num_workers)
    fold_logger.info(f"Data loaders created: train_batches={len(dl_tr)}, val_batches={len(dl_va)}")

    best_state, best_f1, best_acc = None, -1.0, -1.0
    best_epoch = -1
    patience = params["patience"]
    since_improve = 0

    fold_logger.info(f"Starting training for {params['max_epochs']} epochs with patience={patience}")

    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                # Mixed precision training
                with autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                # Full precision training
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.get("grad_clip", 1.0))
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            train_batches += 1

        model.eval()
        preds, gts = [], []
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                if scaler is not None:
                    # Mixed precision validation
                    with autocast():
                        logits = model(xb)
                        loss = criterion(logits, yb.to(device))
                else:
                    # Full precision validation
                    logits = model(xb)
                    loss = criterion(logits, yb.to(device))
                yhat = logits.argmax(dim=1).cpu().numpy()
                preds.append(yhat)
                gts.append(yb.numpy())
                val_loss += loss.item()
                val_batches += 1

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(gts)
        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

        metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": f1,
            "val_acc": acc
        }

        if verbose or epoch % 10 == 0:
            log_training_progress(fold_logger, epoch, params["max_epochs"], metrics, "  ")

        improved = (f1 > best_f1) or (math.isclose(f1, best_f1) and acc > best_acc)
        if improved:
            best_f1, best_acc = f1, acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since_improve = 0
            fold_logger.info(f"  New best at epoch {epoch}: f1={f1:.4f}, acc={acc:.4f}")
        else:
            since_improve += 1

        if since_improve >= patience:
            fold_logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    fold_logger.info(f"Fold completed: best_epoch={best_epoch}, best_f1={best_f1:.4f}, best_acc={best_acc:.4f}")
    return best_state, {"f1_macro": best_f1, "acc": best_acc}, best_epoch


def evaluate_param_set(
    data: pd.DataFrame,
    y: np.ndarray,
    categorical_cols: List[str],
    numerical_cols: List[str],
    params: Dict[str, Any],
    cv_splits: int = 5,
    verbose: bool = False,
    random_state: int = 42,
    feature_selection_method: str = "mutual_info",
    n_features: int = None,
    use_amp: bool = True,
    num_workers: int = 4,
) -> Dict[str, Any]:
    # Handle string "None" values that might come from YAML configuration
    if n_features == "None" or n_features == "none":
        n_features = None
    elif isinstance(n_features, str) and n_features.isdigit():
        n_features = int(n_features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        fold_logger = logging.getLogger("evaluate_param_set")
        fold_logger.info(f"Using GPU device: {device} ({torch.cuda.get_device_name(device.index)})")
        fold_logger.info(f"GPU memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.1f} GB")
    skf = stratified_cv_splits(y, desired_splits=cv_splits, random_state=random_state)

    fold_metrics: List[Dict[str, float]] = []
    best_epochs: List[int] = []
    feature_cols = list(categorical_cols) + list(numerical_cols)
    # Drop categorical columns that are entirely missing across the provided data
    valid_categorical_cols = [c for c in categorical_cols if data[c].notna().any()]
    # Split categorical columns into byte-like and non-byte for encoding strategies
    byte_cols = [c for c in valid_categorical_cols if c.endswith('_byte_') or '_byte_' in c]
    nonbyte_cols = [c for c in valid_categorical_cols if c not in byte_cols]
    for fold, (tr_idx, va_idx) in enumerate(skf.split(data[feature_cols].values, y), start=1):
        Xtr_df, ytr = data.iloc[tr_idx][feature_cols], y[tr_idx]
        Xva_df, yva = data.iloc[va_idx][feature_cols], y[va_idx]
        
        # Collect fold information for logging
        fold_info = {
            'fold_number': fold,
            'initial_shape': data.shape,
            'initial_categorical': len(categorical_cols),
            'initial_numerical': len(numerical_cols),
            'initial_class_counts': dict(zip(*np.unique(y, return_counts=True)))
        }

        # Two pipelines: OneHot for non-byte categoricals; Ordinal for byte features
        cat_nonbyte_pipeline = Pipeline(steps=[
            ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
            ("impute", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
            ("to_str", FunctionTransformer(_astype_str, validate=False)),
            ("encode", _make_ohe()),
        ])

        cat_byte_pipeline = Pipeline(steps=[
            ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
            ("impute", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
            ("to_str", FunctionTransformer(_astype_str, validate=False)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        num_pipeline = Pipeline(steps=[
            ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
        ])

        pre_enc = ColumnTransformer(
            transformers=[
                ("cat_nonbyte", cat_nonbyte_pipeline, nonbyte_cols),
                ("cat_byte", cat_byte_pipeline, byte_cols),
                ("num", num_pipeline, numerical_cols),
            ],
            remainder="drop",
        )

        # Log encoding information
        fold_logger = logging.getLogger(f"fold_{fold}")
        fold_logger.info(f"=== FOLD {fold} ENCODING TRANSFORMATION ===")
        fold_logger.info(f"Input features - Categorical: {len(nonbyte_cols)} non-byte + {len(byte_cols)} byte, Numerical: {len(numerical_cols)}")
        fold_logger.info(f"Training samples: {len(Xtr_df)}, Validation samples: {len(Xva_df)}")
        
        Xtr_enc = pre_enc.fit_transform(Xtr_df)
        Xva_enc = pre_enc.transform(Xva_df)
        
        fold_logger.info(f"After encoding - Features: {Xtr_enc.shape[1]}")
        fold_logger.info(f"Encoding applied: OneHot for {len(nonbyte_cols)} categorical, Ordinal for {len(byte_cols)} byte, Pass-through for {len(numerical_cols)} numerical")

        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        Xtr_s = scl.fit_transform(imp.fit_transform(Xtr_enc))
        Xva_s = scl.transform(imp.transform(Xva_enc))
        
        fold_logger.info(f"After imputation and scaling - Features: {Xtr_s.shape[1]}")

        vt = VarianceThreshold(threshold=1e-8)
        Xtr_vt = vt.fit_transform(Xtr_s)
        Xva_vt = vt.transform(Xva_s)
        
        removed_features = Xtr_s.shape[1] - Xtr_vt.shape[1]
        fold_logger.info(f"After variance threshold - Features: {Xtr_vt.shape[1]} (removed {removed_features} low-variance features)")

        # Apply feature selection within CV fold
        fold_logger.info(f"=== FOLD {fold} FEATURE SELECTION ===")
        
        # Create feature names for logging (if available)
        feature_names = None
        if hasattr(data, 'columns'):
            # Try to get feature names from the original data
            try:
                # Get the feature names after encoding
                if hasattr(pre_enc, 'get_feature_names_out'):
                    feature_names = pre_enc.get_feature_names_out().tolist()
                elif hasattr(pre_enc, 'get_feature_names'):
                    feature_names = pre_enc.get_feature_names().tolist()
            except:
                pass
        
        # Handle feature selection based on n_features parameter
        if n_features is None:
            # Use all features (no feature selection)
            fold_logger.info("Using all features (no feature selection)")
            Xtr, Xva = Xtr_vt, Xva_vt
            feature_selector = None
            fs_info = {
                'method': 'none',
                'n_features_requested': None,
                'n_features_selected': Xtr_vt.shape[1],
                'n_features_original': Xtr_vt.shape[1],
                'selected_indices': list(range(Xtr_vt.shape[1])),
                'selected_feature_names': feature_names if feature_names else [],
                'elapsed_time': 0.0,
                'selector_type': 'NoSelection'
            }
        elif n_features == "auto":
            # Automatically select optimal number of features using validation
            fold_logger.info("Automatically selecting optimal number of features using validation...")
            optimal_n_features = select_optimal_features(Xtr_vt, Xva_vt, ytr, method=feature_selection_method, random_state=random_state)
            fold_logger.info(f"Selected optimal number of features: {optimal_n_features}")
            Xtr, Xva, feature_selector, fs_info = apply_feature_selection(
                Xtr_vt, Xva_vt, ytr, 
                method=feature_selection_method, 
                n_features=optimal_n_features, 
                random_state=random_state,
                feature_names=feature_names
            )
        elif n_features == "statistical":
            # Select features based on statistical significance
            fold_logger.info("Selecting features based on statistical significance...")
            optimal_n_features = select_optimal_features_statistical(Xtr_vt, ytr, method=feature_selection_method)
            fold_logger.info(f"Selected {optimal_n_features} statistically significant features")
            Xtr, Xva, feature_selector, fs_info = apply_feature_selection(
                Xtr_vt, Xva_vt, ytr, 
                method=feature_selection_method, 
                n_features=optimal_n_features, 
                random_state=random_state,
                feature_names=feature_names
            )
        else:
            # Apply feature selection with specified method and number of features
            fold_logger.info(f"Using specified method: {feature_selection_method} with {n_features} features")
            Xtr, Xva, feature_selector, fs_info = apply_feature_selection(
                Xtr_vt, Xva_vt, ytr, 
                method=feature_selection_method, 
                n_features=n_features, 
                random_state=random_state,
                feature_names=feature_names
            )
        
        # Log detailed feature selection information
        fold_logger.info(f"Feature selection completed in {fs_info['elapsed_time']:.3f} seconds")
        fold_logger.info(f"Selected {fs_info['n_features_selected']} features from {fs_info['n_features_original']} original features")
        fold_logger.info(f"Feature selection method: {fs_info['selector_type']}")
        
        # Log selected feature indices (first 20 and last 5 for brevity)
        selected_indices = fs_info['selected_indices']
        if len(selected_indices) <= 25:
            fold_logger.info(f"Selected feature indices: {selected_indices}")
        else:
            fold_logger.info(f"Selected feature indices: {selected_indices[:20]} ... {selected_indices[-5:]}")
        
        # Log selected feature names if available
        if fs_info['selected_feature_names']:
            selected_names = fs_info['selected_feature_names']
            if len(selected_names) <= 10:
                fold_logger.info(f"Selected feature names: {selected_names}")
            else:
                fold_logger.info(f"Selected feature names: {selected_names[:10]} ... {selected_names[-5:]}")
        else:
            fold_logger.info("Feature names not available for selected features")

        state, metrics, best_epoch = train_one_fold(Xtr, ytr, Xva, yva, params, device, verbose=False, rng_seed=random_state, fold_info=fold_info, use_amp=use_amp, num_workers=num_workers)
        fold_metrics.append(metrics)
        best_epochs.append(best_epoch)
        if verbose:
            print(f"    fold {fold}: f1={metrics['f1_macro']:.4f} acc={metrics['acc']:.4f} epoch={best_epoch}")

    mean_f1 = float(np.mean([m["f1_macro"] for m in fold_metrics]))
    std_f1 = float(np.std([m["f1_macro"] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0
    mean_acc = float(np.mean([m["acc"] for m in fold_metrics]))
    mean_epoch = int(np.mean(best_epochs))

    out = dict(mean_f1=mean_f1, std_f1=std_f1, mean_acc=mean_acc, mean_epoch=mean_epoch,
               params=params, per_fold=fold_metrics)
    return out


def select_best(config_results: List[Dict[str, Any]]):
    def keyfn(r):
        size = sum(r["params"]["hidden_dims"])
        return (r["mean_f1"], r["mean_acc"], -size)
    return max(config_results, key=keyfn)


def train_level_torch(
    df: pd.DataFrame,
    maps: Dict[str, Any],
    target_col: str,
    param_grid: Dict[str, List[Any]] = None,
    granularity: str = None,
    cv_splits: int = 5,
    max_configs: int = None,
    random_state: int = 42,
    verbose: bool = True,
    feature_selection_method: str = "mutual_info",
    n_features: int = None,
    custom_hyperparameters: Dict[str, Dict[str, List[Any]]] = None,
    use_amp: bool = True,
    num_workers: int = 4,
    use_multi_gpu: bool = True,
):
    level_logger = logging.getLogger(f"level_{target_col}")
    level_logger.info(f"=== Training {target_col.upper()} Level Model ===")

    # Determine granularity from target_col if not provided
    if granularity is None:
        if target_col.startswith("family_"):
            granularity = "family"
        elif target_col.startswith("major_"):
            granularity = "major"
        elif target_col.startswith("leaf_"):
            granularity = "leaf"
        else:
            granularity = "unknown"
    
    level_logger.info(f"Model granularity: {granularity}")

    # Define granularity-specific parameter grids
    def get_granularity_param_grid(granularity: str, custom_hyperparameters: Dict[str, Dict[str, List[Any]]] = None) -> Dict[str, List[Any]]:
        """Get parameter grid based on model granularity."""
        
        # Check if custom hyperparameters are provided for this granularity
        if custom_hyperparameters and granularity in custom_hyperparameters:
            level_logger.info(f"Using custom hyperparameters for {granularity} level")
            return custom_hyperparameters[granularity]
        
        # Base parameter grid (used as fallback)
        base_grid = {
            "hidden_dims": [
                [64, 32],
                [128, 64],
                [256, 128],
                [512, 256],
                [1024, 512],
                [256, 128, 64],
                [512, 256, 128],
                [256, 128, 64, 32],
            ],
            "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "lr": [1e-4, 3e-4, 1e-3, 3e-3],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
            "batch_size": [8, 32, 64, 128, 256, 512],
            "max_epochs": [100, 150, 200, 250, 300],
            "patience": [20, 25, 30],
            "use_bn": [True],
            "use_onecycle": [False, True],
            "grad_clip": [0.5, 1.0, 2.0],
            "use_smote": [True, False],
            "use_undersample": [True, False],
            "n_features": [None, "auto", "statistical"],
        }
        
        if granularity == "family":
            # Family level: fewer classes, simpler models
            return {
                "hidden_dims": [
                    [64, 32],
                    [128, 64],
                    [256, 128],
                    [512, 256],
                    [1024, 512],
                    [256, 128, 64],
                    [512, 256, 128],
                    [256, 128, 64, 32],
                ],
                "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "lr": [1e-4, 3e-4, 1e-3, 3e-3],
                "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                "batch_size": [64, 128, 256],
                "max_epochs": [100, 150, 200, 250, 300],
                "patience": [20, 25, 30],
                "use_bn": [True],
                "use_onecycle": [True],
                "grad_clip": [0.5, 1.0, 2.0],
                "use_smote": [False],
                "use_undersample": [True, False],
                "n_features": [None],
            }
        elif granularity == "major":
            # Major level: medium complexity
            return {
                "hidden_dims": [
                    [64, 32],
                    [128, 64],
                    [256, 128],
                    [512, 256],
                    [1024, 512],
                    [256, 128, 64],
                    [512, 256, 128],
                    [256, 128, 64, 32],
                ],
                "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "lr": [1e-4, 3e-4, 1e-3, 3e-3],
                "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                "batch_size": [32, 64, 128, 256],
                "max_epochs": [100, 150, 200, 250, 300],
                "patience": [20, 25, 30],
                "use_bn": [True],
                "use_onecycle": [True],
                "grad_clip": [0.5, 1.0, 2.0],
                "use_smote": [True, False],
                "use_undersample": [True, False],
                "n_features": [None],
                # "n_features": [None, "auto", "statistical"],
            }
        elif granularity == "leaf":
            # Leaf level: most classes, most complex models
            return {
                "hidden_dims": [
                    [64, 32],
                    [128, 64],
                    [256, 128],
                    [512, 256],
                    [1024, 512],
                    [256, 128, 64],
                    [512, 256, 128],
                    [256, 128, 64, 32],
                ],
                "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "lr": [1e-4, 3e-4, 1e-3, 3e-3],
                "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                "batch_size": [32, 64, 128, 256],
                "max_epochs": [100, 150, 200, 250, 300],
                "patience": [20, 25, 30],
                "use_bn": [True],
                "use_onecycle": [True],
                "grad_clip": [0.5, 1.0, 2.0],
                "use_smote": [True, False],
                "use_undersample": [True, False],
                "n_features": [None],
                # "n_features": [None, "auto", "statistical"],
            }
        else:
            # Unknown granularity, use base grid
            return base_grid

    # Use granularity-specific parameter grid if none provided
    if param_grid is None:
        param_grid = get_granularity_param_grid(granularity, custom_hyperparameters)
        level_logger.info(f"Using granularity-specific parameter grid for {granularity} level")
    else:
        level_logger.info(f"Using provided parameter grid for {granularity} level")

    seed_everything(random_state)
    data = df.copy()
    data = data.loc[data[target_col].notna()].copy()

    y_int = data[target_col].astype("int64")
    classes, counts = np.unique(y_int.values, return_counts=True)
    single_instance_classes = classes[counts < 2]
    rows_to_remove = int(np.sum(counts[counts < 2]))
    if rows_to_remove > 0:
        level_logger.warning(
            f"Filtering {len(single_instance_classes)} classes with < 2 samples for {target_col}. Total samples removed: {rows_to_remove}"
        )
        counts_map = dict(zip(classes, counts))
        keep_mask = y_int.map(lambda c: counts_map.get(c, 0) >= 2)
        data = data.loc[keep_mask].copy()

    y = data[target_col].astype("int64").values

    categorical_cols = [c for c in maps["config"].get("categorical_features", []) if c in data.columns]
    numerical_cols = [c for c in maps["config"].get("numerical_features", []) if c in data.columns]
    feature_cols = list(categorical_cols) + list(numerical_cols)

    level_logger.info(f"Data prepared: {data.shape[0]} samples, {len(feature_cols)} features")
    level_logger.info(f"Target classes: {len(np.unique(y))} unique values")
    level_logger.info(f"Feature selection method: {feature_selection_method}")
    if n_features is not None:
        level_logger.info(f"Target number of features: {n_features}")
    else:
        level_logger.info("Number of features will be determined automatically")

    configs = expand_grid(param_grid, max_samples=max_configs, random_state=random_state)
    level_logger.info(f"Evaluating {len(configs)} configurations with {cv_splits}-fold CV")

    results = []
    for i, p in enumerate(configs, start=1):
        t0 = time.time()
        level_logger.info(f"Config {i}/{len(configs)}: {p}")

        # Extract feature selection parameters from grid search
        config_n_features = p.pop('n_features', n_features)  # Remove from params to avoid passing to model
        config_feature_selection_method = feature_selection_method  # Use the specified method
        
        res = evaluate_param_set(data[feature_cols], y, categorical_cols, numerical_cols, p, cv_splits=cv_splits, verbose=False, random_state=random_state, feature_selection_method=config_feature_selection_method, n_features=config_n_features, use_amp=use_amp, num_workers=num_workers)
        res["time_s"] = time.time() - t0
        results.append(res)

        level_logger.info(f"  Config {i} results: f1={res['mean_f1']:.4f}±{res['std_f1']:.4f}, "
                          f"acc={res['mean_acc']:.4f}, epoch≈{res['mean_epoch']}, time={res['time_s']:.1f}s")

    best = select_best(results)
    level_logger.info(f"Best configuration selected: f1={best['mean_f1']:.4f}±{best['std_f1']:.4f}, "
                      f"acc={best['mean_acc']:.4f}, params={best['params']}")

    level_logger.info("Refitting final model on full dataset")
    # Final train/val split on the filtered data
    X_full = data[feature_cols]
    X_tr_df, X_va_df, y_tr, y_va = train_test_split(X_full, y, test_size=0.12, random_state=random_state, stratify=y)
    
    # Log final model training information
    level_logger.info(f"=== FINAL MODEL TRAINING ===")
    level_logger.info(f"Final split - Training: {len(X_tr_df)} samples, Validation: {len(X_va_df)} samples")
    level_logger.info(f"Final class distribution - Training:")
    for class_id, count in zip(*np.unique(y_tr, return_counts=True)):
        level_logger.info(f"  Class {class_id}: {count} samples")
    level_logger.info(f"Final class distribution - Validation:")
    for class_id, count in zip(*np.unique(y_va, return_counts=True)):
        level_logger.info(f"  Class {class_id}: {count} samples")

    # Final-fit: skip fully-missing categorical columns and use split pipelines
    valid_categorical_cols = [c for c in categorical_cols if data[c].notna().any()]
    # byte_cols = [c for c in valid_categorical_cols if c.endswith('_byte_') or '_byte_' in c]
    nonbyte_cols = valid_categorical_cols

    cat_nonbyte_pipeline = Pipeline(steps=[
        ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
        ("impute", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
        ("to_str", FunctionTransformer(_astype_str, validate=False)),
        ("encode", _make_ohe()),
    ])

    # cat_byte_pipeline = Pipeline(steps=[
    #     ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
    #     ("impute", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
    #     ("to_str", FunctionTransformer(_astype_str, validate=False)),
    #     ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    # ])

    num_pipeline = Pipeline(steps=[
        ("na2nan", FunctionTransformer(_pd_na_to_np_nan, validate=False)),
    ])
    pre_enc = ColumnTransformer(
        transformers=[
            ("cat_nonbyte", cat_nonbyte_pipeline, nonbyte_cols),
            # ("cat_byte", cat_byte_pipeline, byte_cols),
            ("num", num_pipeline, numerical_cols),
        ],
        remainder="drop",
    )
    level_logger.info(f"Final encoding - Input features: Categorical: {len(nonbyte_cols)}, Numerical: {len(numerical_cols)}")
    
    X_tr_enc = pre_enc.fit_transform(X_tr_df)
    X_va_enc = pre_enc.transform(X_va_df)
    
    level_logger.info(f"After encoding - Features: {X_tr_enc.shape[1]}")
    level_logger.info(f"Encoding applied: OneHot for {len(nonbyte_cols)} categorical, Pass-through for {len(numerical_cols)} numerical")

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_tr_s = scl.fit_transform(imp.fit_transform(X_tr_enc))
    X_va_s = scl.transform(imp.transform(X_va_enc))
    
    level_logger.info(f"After imputation and scaling - Features: {X_tr_s.shape[1]}")

    vt = VarianceThreshold(threshold=1e-8)
    X_tr_vt = vt.fit_transform(X_tr_s)
    X_va_vt = vt.transform(X_va_s)
    
    removed_features = X_tr_s.shape[1] - X_tr_vt.shape[1]
    level_logger.info(f"After variance threshold - Features: {X_tr_vt.shape[1]} (removed {removed_features} low-variance features)")
    
    # Apply feature selection to final model
    level_logger.info(f"=== FINAL FEATURE SELECTION ===")
    
    # Create feature names for logging (if available)
    feature_names = None
    try:
        # Get the feature names after encoding
        if hasattr(pre_enc, 'get_feature_names_out'):
            feature_names = pre_enc.get_feature_names_out().tolist()
        elif hasattr(pre_enc, 'get_feature_names'):
            feature_names = pre_enc.get_feature_names().tolist()
    except:
        pass
    
    # Use the best configuration's feature selection parameters
    best_n_features = best["params"].get('n_features', n_features)
    best_feature_selection_method = feature_selection_method
    
    # Handle string "None" values that might come from YAML configuration
    if best_n_features == "None" or best_n_features == "none":
        best_n_features = None
    
    if best_n_features is None:
        # Use all features (no feature selection)
        level_logger.info("Using all features for final model (no feature selection)")
        X_tr, X_va = X_tr_vt, X_va_vt
        final_feature_selector = None
        fs_info = {
            'method': 'none',
            'n_features_requested': None,
            'n_features_selected': X_tr_vt.shape[1],
            'n_features_original': X_tr_vt.shape[1],
            'selected_indices': list(range(X_tr_vt.shape[1])),
            'selected_feature_names': feature_names if feature_names else [],
            'elapsed_time': 0.0,
            'selector_type': 'NoSelection'
        }
    elif best_n_features == "auto":
        # Automatically select optimal number of features for final model using validation
        level_logger.info("Automatically selecting optimal number of features for final model using validation...")
        optimal_n_features = select_optimal_features(X_tr_vt, X_va_vt, y_tr, method=best_feature_selection_method, random_state=random_state)
        level_logger.info(f"Selected optimal number of features for final model: {optimal_n_features}")
        X_tr, X_va, final_feature_selector, fs_info = apply_feature_selection(
            X_tr_vt, X_va_vt, y_tr, 
            method=best_feature_selection_method, 
            n_features=optimal_n_features, 
            random_state=random_state,
            feature_names=feature_names
        )
    elif best_n_features == "statistical":
        # Select features based on statistical significance for final model
        level_logger.info("Selecting features based on statistical significance for final model...")
        optimal_n_features = select_optimal_features_statistical(X_tr_vt, y_tr, method=best_feature_selection_method)
        level_logger.info(f"Selected {optimal_n_features} statistically significant features for final model")
        X_tr, X_va, final_feature_selector, fs_info = apply_feature_selection(
            X_tr_vt, X_va_vt, y_tr, 
            method=best_feature_selection_method, 
            n_features=optimal_n_features, 
            random_state=random_state,
            feature_names=feature_names
        )
    else:
        level_logger.info(f"Using best configuration's feature selection: {best_feature_selection_method} with {best_n_features} features")
        X_tr, X_va, final_feature_selector, fs_info = apply_feature_selection(
            X_tr_vt, X_va_vt, y_tr, 
            method=best_feature_selection_method, 
            n_features=best_n_features, 
            random_state=random_state,
            feature_names=feature_names
        )
    
    # Log detailed feature selection information
    level_logger.info(f"Final feature selection completed in {fs_info['elapsed_time']:.3f} seconds")
    level_logger.info(f"Selected {fs_info['n_features_selected']} features from {fs_info['n_features_original']} original features")
    level_logger.info(f"Final feature selection method: {fs_info['selector_type']}")
    
    # Log selected feature indices (first 20 and last 5 for brevity)
    selected_indices = fs_info['selected_indices']
    if len(selected_indices) <= 25:
        level_logger.info(f"Final selected feature indices: {selected_indices}")
    else:
        level_logger.info(f"Final selected feature indices: {selected_indices[:20]} ... {selected_indices[-5:]}")
    
    # Log selected feature names if available
    if fs_info['selected_feature_names']:
        selected_names = fs_info['selected_feature_names']
        if len(selected_names) <= 10:
            level_logger.info(f"Final selected feature names: {selected_names}")
        else:
            level_logger.info(f"Final selected feature names: {selected_names[:10]} ... {selected_names[-5:]}")
    else:
        level_logger.info("Feature names not available for final selected features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_logger.info(f"Using device: {device}")

    # Establish local class mapping identical to train_one_fold's behavior
    unique_classes = np.unique(np.concatenate([y_tr, y_va]))
    state, metrics, best_epoch = train_one_fold(
        X_tr, y_tr, X_va, y_va, best["params"], device, verbose=False, rng_seed=random_state, use_amp=use_amp, num_workers=num_workers
    )

    in_dim = X_tr.shape[1]
    out_dim = int(unique_classes.size)
    final_model = MLP(in_dim, best["params"]["hidden_dims"], out_dim,
                      dropout=best["params"]["dropout"], use_bn=best["params"].get("use_bn", True)).to(device)
    
    # Multi-GPU support
    if use_multi_gpu and device.type == 'cuda' and torch.cuda.device_count() > 1:
        final_model = torch.nn.DataParallel(final_model)
        level_logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    # Handle state_dict loading for DataParallel models
    if isinstance(final_model, torch.nn.DataParallel):
        # If the saved state_dict doesn't have 'module.' prefix, add it
        if state and not any(key.startswith('module.') for key in state.keys()):
            state = {f'module.{k}': v for k, v in state.items()}
            level_logger.info("Adjusted state_dict keys for DataParallel model")
    else:
        # If the model is not DataParallel but state_dict has 'module.' prefix, remove it
        if state and any(key.startswith('module.') for key in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
            level_logger.info("Removed 'module.' prefix from state_dict keys for single GPU model")
    
    final_model.load_state_dict(state)
    final_model.eval()

    log_model_info(level_logger, final_model, f"Final {target_col} Model")
    

    # Map y_va to local label space for correct evaluation
    class_to_local = {c: i for i, c in enumerate(unique_classes)}
    y_va_local = np.vectorize(class_to_local.get)(y_va)

    with torch.no_grad():
        Xv = torch.from_numpy(X_va).float().to(device)
        logits = final_model(Xv)
        y_hat = logits.argmax(dim=1).cpu().numpy()
    final_metrics = dict(
        acc=accuracy_score(y_va_local, y_hat),
        f1_micro=f1_score(y_va_local, y_hat, average="micro"),
        f1_macro=f1_score(y_va_local, y_hat, average="macro"),
        f1_weighted=f1_score(y_va_local, y_hat, average="weighted"),
        prec_micro=precision_score(y_va_local, y_hat, average="micro", zero_division=0),
        prec_macro=precision_score(y_va_local, y_hat, average="macro", zero_division=0),
        prec_weighted=precision_score(y_va_local, y_hat, average="weighted", zero_division=0),
        rec_micro=recall_score(y_va_local, y_hat, average="micro", zero_division=0),
        rec_macro=recall_score(y_va_local, y_hat, average="macro", zero_division=0),
        rec_weighted=recall_score(y_va_local, y_hat, average="weighted", zero_division=0),
        report=classification_report(y_va_local, y_hat, digits=3)
    )

    level_logger.info(f"Final holdout metrics: f1={final_metrics['f1_macro']:.4f}, acc={final_metrics['acc']:.4f}")

    # === Persist evaluation artifacts (metrics and confusion matrices) ===
    try:
        import pandas as _pd
        # Build mapping from local index -> global idx -> id string
        if target_col.startswith("family_"):
            id2idx_map = maps["family_id2idx"]
        elif target_col.startswith("major_"):
            id2idx_map = maps["major_id2idx"]
        else:
            id2idx_map = maps["leaf_id2idx"]
        idx2id_map = {v: k for k, v in id2idx_map.items()}
        class_labels = [idx2id_map.get(int(gidx), str(int(gidx))) for gidx in unique_classes]

        def _save_cm_png(cm, labels, png_path, title):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            n = len(labels)
            fig_w = min(16, max(6, n * 0.35))
            fig_h = min(16, max(5, n * 0.35))
            fig, ax = _plt.subplots(figsize=(fig_w, fig_h))
            
            # Normalize confusion matrix to percentages (0-1 range)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
            
            im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Proportion')
            ax.set(
                xticks=range(n), yticks=range(n),
                xticklabels=labels, yticklabels=labels,
                ylabel="True label", xlabel="Predicted label", title=title,
            )
            _plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
            _plt.setp(ax.get_yticklabels(), rotation=0, va="center")
            ax.grid(False)
            fig.tight_layout()
            fig.savefig(png_path, dpi=200)
            _plt.close(fig)

        # Metrics CSVs (aggregate)
        metrics_all_df = _pd.DataFrame([
            {
                "metric": "accuracy", "value": final_metrics["acc"],
            },
            {"metric": "precision_micro", "value": final_metrics["prec_micro"]},
            {"metric": "precision_macro", "value": final_metrics["prec_macro"]},
            {"metric": "precision_weighted", "value": final_metrics["prec_weighted"]},
            {"metric": "recall_micro", "value": final_metrics["rec_micro"]},
            {"metric": "recall_macro", "value": final_metrics["rec_macro"]},
            {"metric": "recall_weighted", "value": final_metrics["rec_weighted"]},
            {"metric": "f1_micro", "value": final_metrics["f1_micro"]},
            {"metric": "f1_macro", "value": final_metrics["f1_macro"]},
            {"metric": "f1_weighted", "value": final_metrics["f1_weighted"]},
        ])

        # Per-label metrics for ALL classes
        per_label_prec = precision_score(y_va_local, y_hat, average=None, labels=list(range(len(class_labels))), zero_division=0)
        per_label_rec = recall_score(y_va_local, y_hat, average=None, labels=list(range(len(class_labels))), zero_division=0)
        per_label_f1 = f1_score(y_va_local, y_hat, average=None, labels=list(range(len(class_labels))))
        per_label_support = _pd.Series(y_va_local).value_counts().reindex(range(len(class_labels)), fill_value=0).values
        per_label_df_all = _pd.DataFrame({
            "label": class_labels,
            "precision": per_label_prec,
            "recall": per_label_rec,
            "f1": per_label_f1,
            "support": per_label_support,
        })
        # Write combined CSV for ALL
        metrics_all_df.to_csv(f"{target_col}_metrics_all.csv", index=False)
        per_label_df_all.to_csv(f"{target_col}_metrics_all_per_label.csv", index=False)

        # Identify OTHER classes by label name containing 'OTHER'
        other_local_indices = {i for i, name in enumerate(class_labels) if "OTHER" in str(name)}
        mask_no_other = _pd.Series([i not in other_local_indices for i in y_va_local]).values
        if mask_no_other.any():
            # Filter out rows where TRUE label is OTHER
            y_true_no_other = y_va_local[mask_no_other]
            y_pred_no_other = y_hat[mask_no_other]

            # Additionally drop rows whose PREDICTED label is OTHER so the reduced
            # class space is consistent on both sides
            kept_local_indices = [i for i in range(len(class_labels)) if i not in other_local_indices]
            _pred_is_kept = _pd.Series(y_pred_no_other).isin(kept_local_indices).values
            y_true_no_other_filt = y_true_no_other[_pred_is_kept]
            y_pred_no_other_filt = y_pred_no_other[_pred_is_kept]

            if y_true_no_other_filt.size > 0:
                metrics_no_other = dict(
                    acc=accuracy_score(y_true_no_other_filt, y_pred_no_other_filt),
                    prec_micro=precision_score(y_true_no_other_filt, y_pred_no_other_filt, average="micro", zero_division=0),
                    prec_macro=precision_score(y_true_no_other_filt, y_pred_no_other_filt, average="macro", zero_division=0),
                    prec_weighted=precision_score(y_true_no_other_filt, y_pred_no_other_filt, average="weighted", zero_division=0),
                    rec_micro=recall_score(y_true_no_other_filt, y_pred_no_other_filt, average="micro", zero_division=0),
                    rec_macro=recall_score(y_true_no_other_filt, y_pred_no_other_filt, average="macro", zero_division=0),
                    rec_weighted=recall_score(y_true_no_other_filt, y_pred_no_other_filt, average="weighted", zero_division=0),
                    f1_micro=f1_score(y_true_no_other_filt, y_pred_no_other_filt, average="micro"),
                    f1_macro=f1_score(y_true_no_other_filt, y_pred_no_other_filt, average="macro"),
                    f1_weighted=f1_score(y_true_no_other_filt, y_pred_no_other_filt, average="weighted"),
                )
                metrics_no_other_df = _pd.DataFrame([
                    {"metric": "accuracy", "value": metrics_no_other["acc"]},
                    {"metric": "precision_micro", "value": metrics_no_other["prec_micro"]},
                    {"metric": "precision_macro", "value": metrics_no_other["prec_macro"]},
                    {"metric": "precision_weighted", "value": metrics_no_other["prec_weighted"]},
                    {"metric": "recall_micro", "value": metrics_no_other["rec_micro"]},
                    {"metric": "recall_macro", "value": metrics_no_other["rec_macro"]},
                    {"metric": "recall_weighted", "value": metrics_no_other["rec_weighted"]},
                    {"metric": "f1_micro", "value": metrics_no_other["f1_micro"]},
                    {"metric": "f1_macro", "value": metrics_no_other["f1_macro"]},
                    {"metric": "f1_weighted", "value": metrics_no_other["f1_weighted"]},
                ])
                metrics_no_other_df.to_csv(f"{target_col}_metrics_no_other.csv", index=False)

                # Per-label metrics for NO_OTHER subset
                per_label_prec_no = precision_score(
                    y_true_no_other_filt, y_pred_no_other_filt, average=None, labels=kept_local_indices, zero_division=0
                )
                per_label_rec_no = recall_score(
                    y_true_no_other_filt, y_pred_no_other_filt, average=None, labels=kept_local_indices, zero_division=0
                )
                per_label_f1_no = f1_score(
                    y_true_no_other_filt, y_pred_no_other_filt, average=None, labels=kept_local_indices
                )
                per_label_support_no = _pd.Series(y_true_no_other_filt).value_counts().reindex(kept_local_indices, fill_value=0).values
                kept_labels = [class_labels[i] for i in kept_local_indices]
                per_label_df_no = _pd.DataFrame({
                    "label": kept_labels,
                    "precision": per_label_prec_no,
                    "recall": per_label_rec_no,
                    "f1": per_label_f1_no,
                    "support": per_label_support_no,
                })
                per_label_df_no.to_csv(f"{target_col}_metrics_no_other_per_label.csv", index=False)

        # Confusion matrices
        cm_all = confusion_matrix(y_va_local, y_hat, labels=list(range(len(class_labels))))
        cm_all_df = _pd.DataFrame(cm_all, index=class_labels, columns=class_labels)
        cm_all_df.to_csv(f"{target_col}_confusion_all.csv")
        _save_cm_png(cm_all, class_labels, f"{target_col}_confusion_all.png", f"{target_col}: Confusion (ALL)")

        if mask_no_other.any():
            # Determine kept local class indices and labels
            kept_local_indices = [i for i in range(len(class_labels)) if i not in other_local_indices]
            kept_labels = [class_labels[i] for i in kept_local_indices]
            # Prepare filtered true/pred arrays for the reduced space
            y_true_no_other = y_va_local[mask_no_other]
            y_pred_no_other = y_hat[mask_no_other]
            _pred_is_kept = _pd.Series(y_pred_no_other).isin(kept_local_indices).values
            y_true_no_other_filt = y_true_no_other[_pred_is_kept]
            y_pred_no_other_filt = y_pred_no_other[_pred_is_kept]
            if y_true_no_other_filt.size > 0:
                # Map to a compact range for confusion_matrix labels
                local_to_compact = {loc: j for j, loc in enumerate(kept_local_indices)}
                y_true_compact = np.array([local_to_compact[i] for i in y_true_no_other_filt])
                y_pred_compact = np.array([local_to_compact[i] for i in y_pred_no_other_filt])
                cm_no_other = confusion_matrix(y_true_compact, y_pred_compact, labels=list(range(len(kept_labels))))
                cm_no_other_df = _pd.DataFrame(cm_no_other, index=kept_labels, columns=kept_labels)
                cm_no_other_df.to_csv(f"{target_col}_confusion_no_other.csv")
                _save_cm_png(cm_no_other, kept_labels, f"{target_col}_confusion_no_other.png", f"{target_col}: Confusion (NO_OTHER)")
    except Exception as _e:
        level_logger.warning(f"Failed to write metrics/confusion matrices for {target_col}: {_e}")

    bundle = dict(
        model=final_model,
        encoder=pre_enc,
        imputer=imp,
        scaler=scl,
        var_selector=vt,
        feature_selector=final_feature_selector,
        categorical_features=categorical_cols,
        numerical_features=numerical_cols,
        best_params=best["params"],
        cv_summary=dict(mean_f1=best["mean_f1"], mean_acc=best["mean_acc"],
                        std_f1=best["std_f1"], mean_epoch=best["mean_epoch"]),
        holdout_metrics=final_metrics,
        target_col=target_col,
        classes_=unique_classes,
        feature_selection_method=feature_selection_method,
        n_features_selected=X_tr.shape[1],
        in_dim=X_tr.shape[1],
        out_dim=len(unique_classes),
    )

    level_logger.info(f"=== {target_col.upper()} Level Training Complete ===")
    return bundle


def train_all_torch_models(
    encoded_df: pd.DataFrame,
    maps: Dict[str, Any],
    random_state: int = 42,
    param_grid: Dict[str, List[Any]] = None,
    cv_splits: int = 5,
    max_configs: int = 16,
    feature_selection_method: str = "mutual_info",
    n_features: int = None,
    use_granularity_specific_grids: bool = True,
    models_to_train: List[str] = None,
    custom_hyperparameters: Dict[str, Dict[str, List[Any]]] = None,
    use_amp: bool = True,
    num_workers: int = 4,
    use_multi_gpu: bool = True,
):
    training_logger = logging.getLogger("training")
    training_logger.info("=== Starting Hierarchical Model Training ===")
    
    # Set default models to train if not specified
    if models_to_train is None:
        models_to_train = ["family", "major", "leaf"]
    
    training_logger.info(f"Training models: {models_to_train}")

    if use_granularity_specific_grids:
        training_logger.info("Using granularity-specific parameter grids")
    else:
        training_logger.info(f"Using shared parameter grid: {len(param_grid)} parameters, max_configs={max_configs}")
    training_logger.info(f"CV splits: {cv_splits}, random_state: {random_state}")

    seed_everything(random_state)
    # Use schemas from maps; do not rely on numeric-only pick
    categorical_cols = [c for c in maps["config"].get("categorical_features", []) if c in encoded_df.columns]
    numerical_cols = [c for c in maps["config"].get("numerical_features", []) if c in encoded_df.columns]
    X_cols = list(categorical_cols) + list(numerical_cols)
    training_logger.info(f"Selected {len(X_cols)} feature columns ({len(categorical_cols)} categorical, {len(numerical_cols)} numerical)")

    def invert_map(id2idx: dict):
        return {idx: id_ for id_, idx in id2idx.items()}
    y_spaces = {
        "family": invert_map(maps["family_id2idx"]),
        "major": invert_map(maps["major_id2idx"]),
        "leaf": invert_map(maps["leaf_id2idx"]),
    }

    training_logger.info(f"Class spaces: family={len(y_spaces['family'])}, "
                         f"major={len(y_spaces['major'])}, leaf={len(y_spaces['leaf'])}")

    models: Dict[str, Any] = {}
    
    # Train only the specified models
    if "family" in models_to_train:
        training_logger.info("Starting FAMILY level training")
        models["family"] = train_level_torch(
            encoded_df, maps=maps, target_col="family_idx",
            param_grid=param_grid if not use_granularity_specific_grids else None,
            granularity="family", cv_splits=cv_splits, max_configs=max_configs,
            random_state=random_state, verbose=True,
            feature_selection_method=feature_selection_method, n_features=n_features,
            custom_hyperparameters=custom_hyperparameters,
            use_amp=use_amp, num_workers=num_workers, use_multi_gpu=use_multi_gpu
        )
    else:
        training_logger.info("Skipping FAMILY level training")

    if "major" in models_to_train:
        training_logger.info("Starting MAJOR level training")
        models["major"] = train_level_torch(
            encoded_df, maps=maps, target_col="major_idx",
            param_grid=param_grid if not use_granularity_specific_grids else None,
            granularity="major", cv_splits=cv_splits, max_configs=max_configs,
            random_state=random_state, verbose=True,
            feature_selection_method=feature_selection_method, n_features=n_features,
            custom_hyperparameters=custom_hyperparameters,
            use_amp=use_amp, num_workers=num_workers, use_multi_gpu=use_multi_gpu
        )
    else:
        training_logger.info("Skipping MAJOR level training")

    if "leaf" in models_to_train:
        training_logger.info("Starting LEAF level training")
        models["leaf"] = train_level_torch(
            encoded_df, maps=maps, target_col="leaf_idx",
            param_grid=param_grid if not use_granularity_specific_grids else None,
            granularity="leaf", cv_splits=cv_splits, max_configs=max_configs,
            random_state=random_state, verbose=True,
            feature_selection_method=feature_selection_method, n_features=n_features,
            custom_hyperparameters=custom_hyperparameters,
            use_amp=use_amp, num_workers=num_workers, use_multi_gpu=use_multi_gpu
        )
    else:
        training_logger.info("Skipping LEAF level training")

    training_logger.info("=== Hierarchical Model Training Complete ===")
    return models, y_spaces, X_cols


