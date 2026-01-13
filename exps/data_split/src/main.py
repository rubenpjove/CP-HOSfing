import sys
import warnings
import logging
import joblib
import random
import json
from pathlib import Path
import absl
import absl.logging

from exps.utils.io_utils import setup_logging, load_config_and_params
from exps.predictors.src.cphos.data import load_dataset, prepare_raw_dataset, preprocess_dataset
from exps.predictors.src.cphos.models import seed_everything


logging.captureWarnings(True)
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)


def _split_dataset(df, config, logger):
    train_frac = float(config.get("dataset_train_frac", 0.8))
    if not 0 < train_frac < 1:
        raise ValueError(f"dataset_train_frac must be in (0, 1); got {train_frac}")

    if len(df) < 2:
        raise ValueError("Dataset must contain at least two samples before splitting")

    strat_col = "leaf_id"
    if strat_col not in df.columns:
        raise KeyError(f"Required column '{strat_col}' not found in preprocessed dataset")

    min_caltest = int(config.get("caltest_min_per_class", 0) or 0)
    if min_caltest < 0:
        raise ValueError("caltest_min_per_class must be non-negative")

    seed = int(config.get("seed", 42))
    rng = random.Random(seed)

    groups = df.groupby(strat_col, dropna=False).indices
    target_train = max(1, min(len(df) - 1, int(round(len(df) * train_frac))))

    train_assignments = {}
    cal_assignments = {}
    cal_requirements = {}
    train_total = 0
    cal_total = 0

    for leaf, idxs in groups.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)

        if min_caltest > 0 and n < min_caltest:
            raise ValueError(
                f"Leaf '{leaf}' has only {n} samples, fewer than the required {min_caltest} cal/test samples"
            )

        required_cal = min_caltest if min_caltest > 0 else (1 if n > 1 else 0)
        required_cal = min(required_cal, n)
        cal_requirements[leaf] = required_cal

        if n == 1:
            if required_cal >= 1:
                train_chunk = []
                cal_chunk = idxs
            else:
                if train_total < target_train:
                    train_chunk = idxs
                    cal_chunk = []
                else:
                    train_chunk = []
                    cal_chunk = idxs

            train_assignments[leaf] = train_chunk
            cal_assignments[leaf] = cal_chunk
            train_total += len(train_chunk)
            cal_total += len(cal_chunk)
            continue

        train_count = int(round(n * train_frac))
        train_count = max(1, min(train_count, n - 1))

        max_train_allowed = n - required_cal
        train_count = min(train_count, max_train_allowed)
        train_count = max(0, train_count)

        cal_count = n - train_count
        if cal_count < required_cal:
            deficit = required_cal - cal_count
            shift = min(deficit, train_count)
            train_count -= shift
            cal_count = n - train_count

        train_chunk = idxs[:train_count]
        cal_chunk = idxs[train_count:]

        train_assignments[leaf] = train_chunk
        cal_assignments[leaf] = cal_chunk
        train_total += len(train_chunk)
        cal_total += len(cal_chunk)

    if train_total == 0 and cal_total > 0:
        moved = False
        for leaf, cal_chunk in cal_assignments.items():
            required = cal_requirements.get(leaf, 0)
            if len(cal_chunk) > required:
                sample = cal_chunk.pop()
                train_assignments.setdefault(leaf, []).append(sample)
                moved = True
                train_total += 1
                cal_total -= 1
                break
        if not moved:
            raise RuntimeError("Unable to create a non-empty training split without violating cal/test requirements")

    if cal_total == 0 and train_total > 0:
        moved = False
        for leaf, train_chunk in train_assignments.items():
            if train_chunk:
                sample = train_chunk.pop()
                cal_assignments.setdefault(leaf, []).append(sample)
                moved = True
                train_total -= 1
                cal_total += 1
                break
        if not moved:
            raise RuntimeError("Unable to create a non-empty calibration/test split")

    train_indices = [idx for chunk in train_assignments.values() for idx in chunk]
    caltest_indices = [idx for chunk in cal_assignments.values() for idx in chunk]

    df_train = df.iloc[train_indices].copy()
    df_caltest = df.iloc[caltest_indices].copy()

    logger.info(
        "Stratified dataset split across %d leaves: %d training samples, %d calibration/test samples "
        "(min cal/test per leaf: %d)",
        len(groups),
        len(df_train),
        len(df_caltest),
        min_caltest,
    )
    return df_train, df_caltest


def _persist_indices_mapping(df_train, df_caltest, config, logger):
    indices_path = config.get("dataset_indices_map_path")
    if not indices_path:
        logger.warning("Skipping index mapping export; no output path configured or derivable")
        return

    missing_original_idx = (
        "original_index" not in df_train.columns or "original_index" not in df_caltest.columns
    )
    if missing_original_idx:
        logger.warning("Cannot export index mapping; 'original_index' column missing from splits")
        return

    mapping = {
        "train": [int(idx) for idx in df_train["original_index"].tolist()],
        "caltest": [int(idx) for idx in df_caltest["original_index"].tolist()],
    }

    output_path = Path(indices_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=4)
    logger.info(f"Index mapping exported to: {output_path}")


def _export(df_train, df_caltest, maps, input_params, logger):
    df_train.to_csv(input_params.get("dataset_train_path"), index=False)
    df_caltest.to_csv(input_params.get("dataset_caltest_path"), index=False)
    joblib.dump(maps, input_params.get("maps_path"))
    _persist_indices_mapping(df_train, df_caltest, input_params, logger)
    logger.info(f"Dataset splits exported to: {input_params.get('dataset_train_path')} and {input_params.get('dataset_caltest_path')}")
    logger.info(f"Maps exported to: {input_params.get('maps_path')}")


def main():
	resolved_path = sys.argv[2]
	config, input_params = load_config_and_params(resolved_path)

	out_dir = config.get("paths", {}).get("out", ".")
	logger = setup_logging(level=logging.INFO, to_file=True, log_dir=out_dir, to_console=False)
	logger.info(f"Loading configuration from: {resolved_path}")

	# Seed from params (fallback to 2026)
	seed = int(input_params.get("seed", 2026))
	seed_everything(seed)
	logger.info(f"Random seed set to: {seed}")

	# Load dataset
	df = load_dataset(logger, input_params["dataset_path"])

	# Prepare raw dataset (normalize labels, column names, etc.)
	skip_raw_preparation = input_params.get("skip_raw_preparation", False)
	if not skip_raw_preparation:
		df = prepare_raw_dataset(logger, df)
	else:
		logger.info("Skipping raw dataset preparation (skip_raw_preparation=True)")

	# Preprocess dataset (feature encoding, label hierarchy)
	df, maps = preprocess_dataset(logger, df)

	# Split dataset into training and calibration/test sets
	df_train, df_caltest = _split_dataset(df, input_params, logger)

	# Export dataset splits
	_export(df_train, df_caltest, maps, input_params, logger)


if __name__ == "__main__":
	try:
		main()
	except Exception:
		logging.getLogger("main").exception("Fatal error in main")
		raise
