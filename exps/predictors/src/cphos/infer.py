from typing import Any, Dict
import logging
import joblib
import numpy as np
import torch
import pandas as pd
from .models import MLP


def predict_ids(bundle: Dict[str, Any], X_df: pd.DataFrame) -> np.ndarray:
    feature_cols = list(bundle.get("categorical_features", [])) + list(bundle.get("numerical_features", []))
    X_df = X_df[feature_cols]
    X = bundle["encoder"].transform(X_df)
    X = bundle["imputer"].transform(X)
    X = bundle["scaler"].transform(X)
    X = bundle["var_selector"].transform(X)
    model = bundle["model"]
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        yhat_local = logits.argmax(dim=1).cpu().numpy()
    classes_ = bundle.get("classes_", None)
    if classes_ is not None:
        return np.array([classes_[i] for i in yhat_local])
    return yhat_local


def predict_proba(bundle: Dict[str, Any], X_df: pd.DataFrame) -> np.ndarray:
    feature_cols = list(bundle.get("categorical_features", [])) + list(bundle.get("numerical_features", []))
    X_df = X_df[feature_cols]
    X = bundle["encoder"].transform(X_df)
    X = bundle["imputer"].transform(X)
    X = bundle["scaler"].transform(X)
    X = bundle["var_selector"].transform(X)
    model = bundle["model"]
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def load_model_bundle(level: str, out_dir: str, logger: logging.Logger):
    """Load a model bundle from disk for a given level."""
    import os
    logger.info(f"Loading {level} model bundle...")
    
    # Load model state
    model_state_path = os.path.join(out_dir, f"{level}_mlp_state.pt")
    try:
        model_state = torch.load(model_state_path, map_location='cpu')
        logger.info(f"Loaded model state from: {model_state_path}")
    except Exception as e:
        logger.error(f"Failed to load model state from {model_state_path}: {e}")
        raise
    
    # Load preprocessing artifacts
    preproc_path = os.path.join(out_dir, f"{level}_preproc.joblib")
    try:
        preproc = joblib.load(preproc_path)
        logger.info(f"Loaded preprocessing artifacts from: {preproc_path}")
    except Exception as e:
        logger.error(f"Failed to load preprocessing artifacts from {preproc_path}: {e}")
        raise
    
    # Reconstruct the model architecture
    encoder = preproc.get("encoder")
    if encoder is None:
        raise ValueError(f"Encoder not found in {preproc_path}")
    
    # Input dimension - first try to get from preproc, otherwise infer from model state
    in_dim = preproc.get("in_dim")
    if in_dim is not None:
        logger.info(f"Got input dimension from preproc: {in_dim}")
    else:
        # Fall back to n_features_selected from preproc
        in_dim = preproc.get("n_features_selected")
        if in_dim is not None:
            logger.info(f"Got input dimension from n_features_selected: {in_dim}")
        
    # If still not available, try to infer from model state
    if in_dim is None and model_state:
        # Find the first linear layer in the network
        for key in sorted(model_state.keys()):
            if 'weight' in key and 'net.0' in key:  # First linear layer is net.0
                in_dim = model_state[key].shape[1]
                logger.info(f"Inferred input dimension from model state: {in_dim}")
                break
    
    if in_dim is None:
        raise ValueError("Could not determine input dimension. Check saved model artifacts.")
    
    # Get model parameters
    best_params = preproc.get("best_params", {})
    hidden_dims = best_params.get("hidden_dims", [128, 64])
    dropout = best_params.get("dropout", 0.1)
    use_bn = best_params.get("use_bn", True)
    
    # Output dimension - first try to get from preproc, otherwise infer from model state
    out_dim = preproc.get("out_dim")
    if out_dim is not None:
        logger.info(f"Got output dimension from preproc: {out_dim}")
    else:
        # Infer from model state by finding the last layer (highest numbered net layer)
        if model_state:
            # Find all net layer keys with weights
            net_weight_keys = [key for key in model_state.keys() if 'net.' in key and 'weight' in key]
            if net_weight_keys:
                # Extract layer numbers and find the max
                layer_numbers = []
                for key in net_weight_keys:
                    # Extract number from key like "net.0.weight" -> 0
                    parts = key.split('.')
                    if len(parts) >= 2 and parts[0] == 'net':
                        try:
                            layer_num = int(parts[1])
                            layer_numbers.append(layer_num)
                        except ValueError:
                            continue
                
                if layer_numbers:
                    # Find the last layer (highest number) and get its output dimension
                    max_layer = max(layer_numbers)
                    last_key = f"net.{max_layer}.weight"
                    if last_key in model_state:
                        out_dim = model_state[last_key].shape[0]
                        logger.info(f"Inferred output dimension from model state: {out_dim} (layer {max_layer})")
    
    if out_dim is None:
        raise ValueError("Cannot infer output dimension from model state. Check saved model artifacts.")
    
    # Create model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = MLP(in_dim, hidden_dims, out_dim, dropout=dropout, use_bn=use_bn).to(device)
    
    # Load state dict
    # Handle potential DataParallel wrapper
    try:
        model.load_state_dict(model_state)
    except RuntimeError as e:
        if "module." in str(e):
            # Try removing module prefix
            unwrapped_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            model.load_state_dict(unwrapped_state)
            logger.info("Removed 'module.' prefix from state dict")
        else:
            raise e
    
    model.eval()
    
    # Build bundle
    classes_ = preproc.get("classes_")
    if classes_ is None:
        logger.warning(
            f"classes_ not found in preproc for {level} level. "
            "This may cause alignment issues with model outputs. "
            "Please retrain the model to regenerate artifacts with classes_."
        )
    
    bundle = {
        "model": model,
        "encoder": encoder,
        "imputer": preproc.get("imputer"),
        "scaler": preproc.get("scaler"),
        "var_selector": preproc.get("var_selector"),
        "feature_selector": preproc.get("feature_selector"),
        "categorical_features": preproc.get("categorical_features", []),
        "numerical_features": preproc.get("numerical_features", []),
        "best_params": best_params,
        "target_col": f"{level}_idx",
        "classes_": classes_,
        "feature_selection_method": preproc.get("feature_selection_method"),
        "n_features_selected": preproc.get("n_features_selected"),
    }
    
    logger.info(f"Successfully loaded {level} model bundle")
    return bundle

