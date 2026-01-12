# src/io_utils.py
from pathlib import Path
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json
import yaml

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        f.write(text)

def write_csv(path: Path, rows):
    ensure_dir(path.parent)
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def dump_env(path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for k, v in sorted(os.environ.items()):
            f.write(f"{k}={v}\n")


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging(
    level: int = logging.INFO,
    to_file: bool = False,
    log_dir: str = "logs",
    use_json: bool = False,
    quiet_libs: bool = True,
    to_console: bool = True,
) -> logging.Logger:
    """
    Set up comprehensive logging for SLURM jobs.
    
    Args:
        level: Logging level (default: INFO)
        to_file: Whether to write logs to file (default: False, logs to stdout)
        log_dir: Directory for log files (default: "logs")
        use_json: Use JSON formatting (default: False, use human-readable)
        quiet_libs: Reduce noise from third-party libraries (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Create formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        # Simplified human-readable format
        fmt = "%(asctime)sZ [%(levelname)s] - %(message)s"
        datefmt = "%Y-%m-%dT%H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        # Use UTC timestamps
        formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
    
    # Console handler (stdout) optional
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional)
    if to_file:
        ensure_dir(Path(log_dir))
        level_name = logging.getLevelName(level)
        # Map WARNING -> warn per request, keep others as-is
        level_token = "warn" if level_name == "WARNING" else str(level_name).lower()
        log_file = Path(log_dir) / f"{level_token}.log"

        # Create file immediately to ensure it exists right away
        # This ensures the file is visible immediately and timestamps are accurate
        log_file_path = str(log_file)
        # Touch the file to create it immediately (before handler opens it)
        log_file.touch(exist_ok=True)
        # Use delay=False to open file immediately (not on first write)
        # This ensures the file exists and is ready for logging from the start
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8", delay=False)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Quiet noisy third-party libraries
    if quiet_libs:
        noisy_libs = [
            "urllib3", "botocore", "matplotlib", "numba", "PIL", "PIL.Image",
            "requests", "urllib3.connectionpool", "tensorflow", "torch.distributed"
        ]
        for lib in noisy_libs:
            logging.getLogger(lib).setLevel(logging.WARNING)
    
    return logger



def log_dataset_info(logger: logging.Logger, df, name: str = "Dataset"):
    """Log dataset information for debugging."""
    logger.info(f"=== {name} Information ===")
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Data types
    dtypes = df.dtypes.value_counts()
    logger.info(f"Data types: {dict(dtypes)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = missing[missing > 0].sort_values(ascending=False)
    if not missing_info.empty:
        logger.info("Missing values:")
        for col, count in missing_info.items():
            logger.info(f"  {col}: {count:,} ({missing_pct[col]:.2f}%)")
    else:
        logger.info("No missing values found")

def log_model_info(logger: logging.Logger, model, name: str = "Model"):
    """Log model information for debugging."""
    logger.info(f"=== {name} Information ===")
    
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    if hasattr(model, 'state_dict'):
        logger.info(f"Model state dict keys: {list(model.state_dict().keys())}")

def log_training_progress(logger: logging.Logger, epoch: int, total_epochs: int, 
                         metrics: Dict[str, float], prefix: str = ""):
    """Log training progress with consistent formatting."""
    progress = f"Epoch {epoch:3d}/{total_epochs}"
    metric_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix}{progress} | {metric_str}")


def create_experiment_logger(experiment_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a dedicated logger for an experiment with file output.
    Useful for keeping separate logs for different experiments.
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Create experiment-specific log file
    ensure_dir(Path(log_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.getenv("SLURM_JOB_ID", "local")
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}_job_{job_id}.log"
    
    # Simplified formatter
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_config_and_params(resolved_path):
    """
    Load the resolved configuration and raw input parameters.

    Returns a tuple: (config, params)
    - config: the resolved configuration (parsed YAML from resolved_path)
    - params: mapping from embedded input_params in resolved config (for reproducibility)
    """
    if not resolved_path or not os.path.exists(resolved_path):
        raise FileNotFoundError("resolved_path must point to an existing YAML file")

    with open(resolved_path, "r") as rf:
        resolved_cfg = yaml.safe_load(rf) or {}

    # Preserve original behavior: the resolved file is the config
    config = resolved_cfg

    # Use embedded input_params from resolved config for reproducibility
    # This ensures the job uses the exact parameters it was submitted with,
    # not the current state of the input_params.yaml file
    embedded_params = config.get("input_params")
    if embedded_params is not None and isinstance(embedded_params, dict):
        params = embedded_params
    else:
        # Fallback: try to load from file if embedded params not available
        input_params_path = config.get("input_params_path")
        if not input_params_path or not os.path.exists(input_params_path):
            raise FileNotFoundError(
                "Both input_params (embedded) and input_params_path are missing from resolved config"
            )
        
        with open(input_params_path, "r") as f:
            loaded = yaml.safe_load(f) or {}

        if not isinstance(loaded, dict):
            raise ValueError("input_params.yaml must define a YAML mapping at the root")

        params = loaded

    return config, params

