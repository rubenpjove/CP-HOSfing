import os
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from exps.utils.os_tree_funcs import export_label_hierarchy_artifacts


def _save_cm_png(cm: np.ndarray, labels: Sequence[str], png_path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    fig_w = min(16, max(6, n * 0.35))
    fig_h = min(16, max(5, n * 0.35))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=labels, yticklabels=labels,
        ylabel="True label", xlabel="Predicted label", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def export_experiment_results(
    *,
    logger: logging.Logger,
    out_dir: str,
    # Trees/labels
    df: Optional[pd.DataFrame] = None,
    label_cols: Optional[List[str]] = None,
    # Metrics/confusions
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence[str]] = None,
    # Optional: exclude labels containing this substring from NO_OTHER views
    other_substring: str = "OTHER",
    # Optional: per-sample outputs
    sample_ids: Optional[Sequence[Any]] = None,
    y_proba: Optional[np.ndarray] = None,
    # Namespacing for filenames
    artifact_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified export function for experiments.

    - If df+label_cols provided: exports label hierarchy artifacts.
    - If y_true+y_pred(+class_labels) provided: exports aggregate and per-label metrics,
      confusion matrices (CSV + PNG), and returns a metrics dict.
    - If sample_ids provided with predictions/probabilities: writes per-sample CSV.

    Returns a dict containing computed metrics (if any) and file paths created.
    """
    os.makedirs(out_dir, exist_ok=True)
    created: Dict[str, Any] = {"files": []}

    # 1) Label hierarchy artifacts
    if df is not None and label_cols:
        try:
            export_label_hierarchy_artifacts(df, out_dir, cols=label_cols)
            logger.info(f"Exported label hierarchy artifacts for {label_cols} to {out_dir}")
        except Exception as e:
            logger.warning(f"Failed exporting label hierarchy artifacts: {e}")

    # 2) Metrics + confusions
    metrics_dict: Optional[Dict[str, float]] = None
    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Compute aggregate metrics
        metrics_dict = dict(
            acc=accuracy_score(y_true, y_pred),
            f1_micro=f1_score(y_true, y_pred, average="micro", zero_division=0),
            f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
            prec_micro=precision_score(y_true, y_pred, average="micro", zero_division=0),
            prec_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            prec_weighted=precision_score(y_true, y_pred, average="weighted", zero_division=0),
            rec_micro=recall_score(y_true, y_pred, average="micro", zero_division=0),
            rec_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
            rec_weighted=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        )

        # Log concise summary
        logger.info(
            f"Metrics: acc={metrics_dict['acc']:.4f} f1_macro={metrics_dict['f1_macro']:.4f} f1_weighted={metrics_dict['f1_weighted']:.4f}"
        )

        # Prepare labels for confusion/CSV
        n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1 if class_labels is None else len(class_labels)
        labels_idx = list(range(n_classes))
        labels_names = [str(i) for i in labels_idx] if class_labels is None else list(class_labels)

        prefix = (artifact_prefix or "results").rstrip("_")

        # Aggregate metrics CSV
        metrics_all_df = pd.DataFrame([
            {"metric": "accuracy", "value": metrics_dict["acc"]},
            {"metric": "precision_micro", "value": metrics_dict["prec_micro"]},
            {"metric": "precision_macro", "value": metrics_dict["prec_macro"]},
            {"metric": "precision_weighted", "value": metrics_dict["prec_weighted"]},
            {"metric": "recall_micro", "value": metrics_dict["rec_micro"]},
            {"metric": "recall_macro", "value": metrics_dict["rec_macro"]},
            {"metric": "recall_weighted", "value": metrics_dict["rec_weighted"]},
            {"metric": "f1_micro", "value": metrics_dict["f1_micro"]},
            {"metric": "f1_macro", "value": metrics_dict["f1_macro"]},
            {"metric": "f1_weighted", "value": metrics_dict["f1_weighted"]},
        ])
        p_metrics_all = os.path.join(out_dir, f"{prefix}_metrics_all.csv")
        metrics_all_df.to_csv(p_metrics_all, index=False)
        created["files"].append(p_metrics_all)

        # Per-label metrics (ALL)
        per_label_prec = precision_score(y_true, y_pred, average=None, labels=labels_idx, zero_division=0)
        per_label_rec = recall_score(y_true, y_pred, average=None, labels=labels_idx, zero_division=0)
        per_label_f1 = f1_score(y_true, y_pred, average=None, labels=labels_idx, zero_division=0)
        per_label_support = pd.Series(y_true).value_counts().reindex(labels_idx, fill_value=0).values
        per_label_df_all = pd.DataFrame({
            "label": labels_names,
            "precision": per_label_prec,
            "recall": per_label_rec,
            "f1": per_label_f1,
            "support": per_label_support,
        })
        p_pl_all = os.path.join(out_dir, f"{prefix}_metrics_all_per_label.csv")
        per_label_df_all.to_csv(p_pl_all, index=False)
        created["files"].append(p_pl_all)

        # Confusion (ALL)
        cm_all = confusion_matrix(y_true, y_pred, labels=labels_idx)
        cm_all_df = pd.DataFrame(cm_all, index=labels_names, columns=labels_names)
        p_cm_csv = os.path.join(out_dir, f"{prefix}_confusion_all.csv")
        cm_all_df.to_csv(p_cm_csv)
        created["files"].append(p_cm_csv)
        p_cm_png = os.path.join(out_dir, f"{prefix}_confusion_all.png")
        _save_cm_png(cm_all, labels_names, p_cm_png, f"{prefix}: Confusion (ALL)")
        created["files"].append(p_cm_png)

        # NO_OTHER views if applicable
        other_indices = {i for i, name in enumerate(labels_names) if other_substring and other_substring in str(name)}
        mask_no_other = pd.Series(y_true).apply(lambda i: i not in other_indices).values
        if mask_no_other.any():
            y_true_no_other = y_true[mask_no_other]
            y_pred_no_other = y_pred[mask_no_other]
            kept_indices = [i for i in labels_idx if i not in other_indices]

            # Filter to kept indices only on predictions as well
            pred_is_kept = pd.Series(y_pred_no_other).isin(kept_indices).values
            y_true_no_other_filt = y_true_no_other[pred_is_kept]
            y_pred_no_other_filt = y_pred_no_other[pred_is_kept]

            if y_true_no_other_filt.size > 0:
                metrics_no_other = dict(
                    acc=accuracy_score(y_true_no_other_filt, y_pred_no_other_filt),
                    prec_micro=precision_score(y_true_no_other_filt, y_pred_no_other_filt, average="micro", zero_division=0),
                    prec_macro=precision_score(y_true_no_other_filt, y_pred_no_other_filt, average="macro", zero_division=0),
                    prec_weighted=precision_score(y_true_no_other_filt, y_pred_no_other_filt, average="weighted", zero_division=0),
                    rec_micro=recall_score(y_true_no_other_filt, y_pred_no_other_filt, average="micro", zero_division=0),
                    rec_macro=recall_score(y_true_no_other_filt, y_pred_no_other_filt, average="macro", zero_division=0),
                    rec_weighted=recall_score(y_true_no_other_filt, y_pred_no_other_filt, average="weighted", zero_division=0),
                    f1_micro=f1_score(y_true_no_other_filt, y_pred_no_other_filt, average="micro", zero_division=0),
                    f1_macro=f1_score(y_true_no_other_filt, y_pred_no_other_filt, average="macro", zero_division=0),
                    f1_weighted=f1_score(y_true_no_other_filt, y_pred_no_other_filt, average="weighted", zero_division=0),
                )
                metrics_no_other_df = pd.DataFrame([
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
                p_metrics_no_other = os.path.join(out_dir, f"{prefix}_metrics_no_other.csv")
                metrics_no_other_df.to_csv(p_metrics_no_other, index=False)
                created["files"].append(p_metrics_no_other)

                # Per-label for NO_OTHER
                kept_labels = [labels_names[i] for i in kept_indices]
                per_label_prec_no = precision_score(
                    y_true_no_other_filt, y_pred_no_other_filt, average=None, labels=kept_indices, zero_division=0
                )
                per_label_rec_no = recall_score(
                    y_true_no_other_filt, y_pred_no_other_filt, average=None, labels=kept_indices, zero_division=0
                )
                per_label_f1_no = f1_score(
                    y_true_no_other_filt, y_pred_no_other_filt, average=None, labels=kept_indices, zero_division=0
                )
                per_label_support_no = pd.Series(y_true_no_other_filt).value_counts().reindex(kept_indices, fill_value=0).values
                per_label_df_no = pd.DataFrame({
                    "label": kept_labels,
                    "precision": per_label_prec_no,
                    "recall": per_label_rec_no,
                    "f1": per_label_f1_no,
                    "support": per_label_support_no,
                })
                p_pl_no_other = os.path.join(out_dir, f"{prefix}_metrics_no_other_per_label.csv")
                per_label_df_no.to_csv(p_pl_no_other, index=False)
                created["files"].append(p_pl_no_other)

                # Confusion (NO_OTHER)
                local_to_compact = {loc: j for j, loc in enumerate(kept_indices)}
                y_true_compact = np.array([local_to_compact[i] for i in y_true_no_other_filt])
                y_pred_compact = np.array([local_to_compact[i] for i in y_pred_no_other_filt])
                cm_no_other = confusion_matrix(y_true_compact, y_pred_compact, labels=list(range(len(kept_indices))))
                p_cm_no_csv = os.path.join(out_dir, f"{prefix}_confusion_no_other.csv")
                cm_no_df = pd.DataFrame(cm_no_other, index=[labels_names[i] for i in kept_indices], columns=[labels_names[i] for i in kept_indices])
                cm_no_df.to_csv(p_cm_no_csv)
                created["files"].append(p_cm_no_csv)
                p_cm_no_png = os.path.join(out_dir, f"{prefix}_confusion_no_other.png")
                _save_cm_png(cm_no_other, [labels_names[i] for i in kept_indices], p_cm_no_png, f"{prefix}: Confusion (NO_OTHER)")
                created["files"].append(p_cm_no_png)

        created["metrics"] = metrics_dict

    # 3) Per-sample outputs
    if sample_ids is not None and y_pred is not None:
        df_rows = {
            "sample_id": list(sample_ids),
            "y_pred": [int(p) for p in y_pred],
        }
        if y_true is not None:
            df_rows["y_true"] = [int(t) for t in y_true]
        if y_proba is not None:
            df_rows["y_pred_proba_max"] = np.max(y_proba, axis=1)
        per_sample_df = pd.DataFrame(df_rows)
        p_samples = os.path.join(out_dir, f"{(artifact_prefix or 'results').rstrip('_')}_predictions.csv")
        per_sample_df.to_csv(p_samples, index=False)
        created["files"].append(p_samples)

    return created


