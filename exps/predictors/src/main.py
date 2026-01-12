import sys
import warnings
import logging
import joblib
import absl
import absl.logging

from exps.utils.io_utils import setup_logging, load_config_and_params
from exps.predictors.src.cphos.train import train_all_torch_models
from exps.predictors.src.cphos.models import seed_everything
from exps.predictors.src.cphos.infer import predict_ids, predict_proba
from exps.utils.results_export import export_experiment_results
from exps.data_split.src import load_pre_split_dataset


logging.captureWarnings(True)
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)



def main():
	
	resolved_path = sys.argv[2]
	config, input_params = load_config_and_params(resolved_path)

	out_dir = config.get("paths", {}).get("out", ".")
	logger = setup_logging(level=logging.INFO, to_file=True, log_dir=out_dir, to_console=False)
	logger.info(f"Loading configuration from: {resolved_path}")

	# Seed from params (fallback to 42)
	seed = int(input_params.get("seed", 42))
	seed_everything(seed)
	logger.info(f"Random seed set to: {seed}")

	# GPU availability and configuration
	import torch
	if torch.cuda.is_available():
		gpu_count = torch.cuda.device_count()
		logger.info(f"GPU acceleration available: {gpu_count} GPU(s) detected")
		for i in range(gpu_count):
			gpu_name = torch.cuda.get_device_name(i)
			gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
			logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
		
		# Log GPU configuration from params
		use_amp = input_params.get("use_amp", True)
		num_workers = input_params.get("num_workers", 4)
		use_multi_gpu = input_params.get("use_multi_gpu", True)
		logger.info(f"GPU configuration: AMP={use_amp}, num_workers={num_workers}, multi_gpu={use_multi_gpu}")
	else:
		logger.warning("No GPU available - falling back to CPU training")
		logger.info("GPU configuration: AMP=False, num_workers=0, multi_gpu=False")

	# Load pre-split training data and preprocessing metadata
	dataset_train_path = input_params.get("dataset_train_path")
	maps_path = input_params.get("maps_path")
	if not dataset_train_path or not maps_path:
		raise KeyError("Both 'dataset_train_path' and 'maps_path' must be defined in input params")

	df, maps = load_pre_split_dataset(logger, dataset_train_path, maps_path)

	# Get training parameters from input parameters
	models_to_train = input_params.get("models_to_train", ["family", "major", "leaf"])
	
	# === Hierarchical label exploration outputs (pre-CV) ===
	try:
		# Export label hierarchy artifacts via unified exporter
		label_cols = ["family_id", "major_id", "leaf_id"]
		export_experiment_results(logger=logger, out_dir=out_dir, df=df, label_cols=label_cols)
	except Exception as e:
		logger.warning(f"Failed to generate hierarchical label visualizations: {e}")
	
	logger.info(f"Training models: {models_to_train}")
	cv_splits = input_params.get("cv_splits", 5)
	max_configs = input_params.get("max_configs", 16)
	custom_hyperparameters = input_params.get("hyperparameters", None)
	
	logger.info(f"CV splits: {cv_splits}")
	logger.info(f"Max configs: {max_configs}")
	if custom_hyperparameters:
		logger.info(f"Using custom hyperparameters for: {list(custom_hyperparameters.keys())}")
	
	# Train all models with granularity-specific hyperparameter grids
	models, y_spaces, _ = train_all_torch_models(
		df, maps, random_state=seed, cv_splits=cv_splits, max_configs=max_configs,
		use_granularity_specific_grids=True, models_to_train=models_to_train,
		custom_hyperparameters=custom_hyperparameters,
		use_amp=input_params.get("use_amp", True),
		num_workers=input_params.get("num_workers", 4),
		use_multi_gpu=input_params.get("use_multi_gpu", True)
		# df, maps, random_state=seed, cv_splits=5, max_configs=1
	)

	# Log final results
	logger.info("=== Final Model Results ===")
	for level, bundle in models.items():
		logger.info(f"{level.upper()} Model:")
		logger.info(f"  Best params: {bundle['best_params']}")
		logger.info(
			f"  CV summary: f1={bundle['cv_summary']['mean_f1']:.4f}±{bundle['cv_summary']['std_f1']:.4f}, "
			f"acc={bundle['cv_summary']['mean_acc']:.4f}"
		)
		logger.info(
			f"  Holdout metrics: f1={bundle['holdout_metrics']['f1_macro']:.4f}, "
			f"acc={bundle['holdout_metrics']['acc']:.4f}"
		)

	# Demo predictions - use the first available model for demo
	available_models = list(models.keys())
	if available_models:
		demo_model = available_models[0]  # Use first available model
		logger.info(f"=== Demo Predictions using {demo_model.upper()} model ===")
		demo_bundle = models[demo_model]
		yhat_demo = predict_ids(demo_bundle, df)
		_ = predict_proba(demo_bundle, df)
		yhat_demo_str = [y_spaces[demo_model][int(i)] for i in yhat_demo]
		logger.info(f"Sample predictions: {yhat_demo_str[:5]}")
	else:
		logger.warning("No models available for demo predictions.")

	# Persist models
	logger.info("=== Saving Models ===")
	import torch
	for level, bundle in models.items():
		# Move model state to CPU before saving
		model_state = bundle["model"].state_dict()
		# Convert any GPU tensors to CPU tensors
		cpu_state = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in model_state.items()}
		
		# Handle DataParallel models - save the unwrapped model state
		if isinstance(bundle["model"], torch.nn.DataParallel):
			# Remove 'module.' prefix from keys to save the unwrapped model state
			unwrapped_state = {k.replace('module.', ''): v for k, v in cpu_state.items()}
			torch.save(unwrapped_state, f"{level}_mlp_state.pt")
			logger.info(f"Saved unwrapped model state for {level} (removed DataParallel prefix)")
		else:
			torch.save(cpu_state, f"{level}_mlp_state.pt")
	keys_to_save = [
		"encoder",
		"imputer",
		"scaler",
		"var_selector",
		"feature_selector",
		"categorical_features",
		"numerical_features",
		"best_params",
		"feature_selection_method",
		"n_features_selected",
		"in_dim",
		"out_dim",
		"classes_",
	]
	preproc = {k: bundle[k] for k in keys_to_save if k in bundle}
	joblib.dump(preproc, f"{level}_preproc.joblib")

	logger.info("Models saved successfully")
	logger.info("=== CP-HOSfing Predictor Training Completed Successfully ===")


if __name__ == "__main__":
	try:
		main()
	except Exception:
		logging.getLogger("main").exception("Fatal error in main")
		raise
