#!/bin/bash
# CP-HOSfing Experiment Runner Entrypoint
# Usage: entrypoint.sh [data_split|predictors|confpred|all]

set -e

# Configuration
EXPERIMENTS=("data_split" "predictors" "confpred")
CONFIG_DIR="${CONFIG_DIR:-/workspace/configs}"
OUT_DIR="${OUT_DIR:-/workspace/artifacts}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
USE_GPU="${USE_GPU:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_config() {
    echo -e "${CYAN}[CONFIG]${NC} $1"
}

# Check GPU availability and configure environment
configure_gpu() {
    if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "1" ] || [ "$USE_GPU" = "yes" ]; then
        log_info "GPU mode enabled, checking availability..."
        
        # Check if CUDA is available via Python/PyTorch
        GPU_AVAILABLE=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
        
        if [ "$GPU_AVAILABLE" = "yes" ]; then
            GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
            GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "Unknown")
            
            log_success "GPU acceleration enabled"
            log_config "GPU Count: ${GPU_COUNT}"
            log_config "GPU Device: ${GPU_NAME}"
            log_config "CUDA Version: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Unknown')"
            
            export USE_GPU=true
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        else
            log_warn "GPU requested but CUDA not available, falling back to CPU"
            export USE_GPU=false
            export CUDA_VISIBLE_DEVICES=""
        fi
    else
        log_info "Running in CPU-only mode"
        export USE_GPU=false
        export CUDA_VISIBLE_DEVICES=""
    fi
}

# Generate resolved config YAML for an experiment
# The main.py scripts expect sys.argv[2] to be a resolved config path
# with embedded input_params
generate_resolved_config() {
    local exp_name="$1"
    local resolved_path="/tmp/${exp_name}_resolved.yaml"
    
    # Check for custom config file or use experiment-specific default
    # Note: redirect log output to stderr so only the path is captured by caller
    local params_file=""
    if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
        params_file="$CONFIG_FILE"
        log_info "Using custom config: $CONFIG_FILE" >&2
    elif [ -f "${CONFIG_DIR}/${exp_name}_params.yaml" ]; then
        params_file="${CONFIG_DIR}/${exp_name}_params.yaml"
        log_info "Using config: $params_file" >&2
    else
        log_error "No config file found for ${exp_name}" >&2
        log_error "Expected: ${CONFIG_DIR}/${exp_name}_params.yaml" >&2
        exit 1
    fi
    
    # Create output directory for this experiment (artifacts)
    local exp_out_dir="${OUT_DIR}/${exp_name}"
    mkdir -p "$exp_out_dir"
    
    # Create internal log directory (not persisted outside container)
    local exp_log_dir="/tmp/logs/${exp_name}"
    mkdir -p "$exp_log_dir"
    
    # Generate resolved config with embedded input_params
    # This matches the format expected by load_config_and_params()
    # Also inject GPU settings based on environment
    # Note: redirect to stderr so only the final echo is captured
    python3 >&2 << PYEOF
import yaml
import os
import sys

params_file = "${params_file}"
resolved_path = "${resolved_path}"
exp_log_dir = "${exp_log_dir}"
use_gpu = os.environ.get("USE_GPU", "false").lower() in ("true", "1", "yes")

# Load input parameters
with open(params_file, 'r') as f:
    input_params = yaml.safe_load(f) or {}

# Override GPU-related settings based on environment
if not use_gpu:
    # Force CPU mode in config
    input_params['use_amp'] = False
    input_params['num_workers'] = 0
    input_params['use_multi_gpu'] = False
    print(f"[CONFIG] GPU disabled: use_amp=False, num_workers=0, use_multi_gpu=False", file=sys.stderr)
else:
    # Ensure GPU settings are enabled (use config defaults or set them)
    if 'use_amp' not in input_params:
        input_params['use_amp'] = True
    if 'num_workers' not in input_params:
        input_params['num_workers'] = 4
    if 'use_multi_gpu' not in input_params:
        input_params['use_multi_gpu'] = True
    print(f"[CONFIG] GPU enabled: use_amp={input_params.get('use_amp')}, num_workers={input_params.get('num_workers')}, use_multi_gpu={input_params.get('use_multi_gpu')}", file=sys.stderr)

# Build resolved config structure
resolved_config = {
    'paths': {
        'out': exp_log_dir,  # Logs go to internal container path, not artifacts
    },
    'input_params': input_params,
    'input_params_path': params_file,
}

# Write resolved config
with open(resolved_path, 'w') as f:
    yaml.dump(resolved_config, f, default_flow_style=False)

print(f"Generated resolved config: {resolved_path}", file=sys.stderr)
PYEOF
    
    echo "$resolved_path"
}

# Run a single experiment
run_experiment() {
    local exp_name="$1"
    
    log_info "=========================================="
    log_info "Running experiment: ${exp_name}"
    log_info "=========================================="
    
    # Validate experiment name
    local valid=false
    for exp in "${EXPERIMENTS[@]}"; do
        if [ "$exp" == "$exp_name" ]; then
            valid=true
            break
        fi
    done
    
    if [ "$valid" = false ]; then
        log_error "Invalid experiment: ${exp_name}"
        log_error "Valid experiments: ${EXPERIMENTS[*]}"
        exit 1
    fi
    
    # Generate resolved config
    local resolved_config
    resolved_config=$(generate_resolved_config "$exp_name")
    
    # Determine log file path (matches Python's setup_logging behavior)
    # Logs go to internal container path, not artifacts
    local exp_log_dir="/tmp/logs/${exp_name}"
    local log_file="${exp_log_dir}/info.log"
    local tail_pid=""
    
    # Run the experiment
    # main.py expects: python main.py -- <resolved_config_path>
    log_info "Executing: python -m exps.${exp_name}.src.main -- ${resolved_config}"
    
    # Start experiment in background
    python -m "exps.${exp_name}.src.main" -- "$resolved_config" &
    local exp_pid=$!
    
    # Wait for log file to appear, then stream it to console
    local wait_count=0
    while [ ! -f "$log_file" ] && [ $wait_count -lt 30 ]; do
        sleep 0.5
        wait_count=$((wait_count + 1))
    done
    
    if [ -f "$log_file" ]; then
        # Stream log file to console in background
        tail -f "$log_file" &
        tail_pid=$!
    fi
    
    # Wait for experiment to complete
    wait $exp_pid
    local exit_code=$?
    
    # Stop tail process if running
    if [ -n "$tail_pid" ]; then
        kill $tail_pid 2>/dev/null || true
        wait $tail_pid 2>/dev/null || true
    fi
    
    # Clean up log file (user doesn't need to see it)
    if [ -f "$log_file" ]; then
        rm -f "$log_file"
    fi
    
    # Check experiment exit code
    if [ $exit_code -ne 0 ]; then
        log_error "Experiment ${exp_name} failed with exit code ${exit_code}"
        exit $exit_code
    fi
    
    log_success "Experiment ${exp_name} completed successfully"
}

# Run all experiments in sequence
run_all() {
    log_info "Running all experiments in sequence: ${EXPERIMENTS[*]}"
    
    for exp in "${EXPERIMENTS[@]}"; do
        run_experiment "$exp"
    done
    
    log_success "=========================================="
    log_success "All experiments completed successfully!"
    log_success "=========================================="
}

# Print usage
print_usage() {
    echo "CP-HOSfing Experiment Runner"
    echo ""
    echo "Usage: $0 [EXPERIMENT]"
    echo ""
    echo "Experiments:"
    echo "  data_split   - Preprocess and split dataset into train/caltest sets"
    echo "  predictors   - Train hierarchical predictors (family, major, leaf)"
    echo "  confpred     - Run conformal prediction experiments"
    echo "  all          - Run all experiments in sequence (default)"
    echo ""
    echo "Environment Variables:"
    echo "  USE_GPU      - Enable GPU acceleration (true/false, default: false)"
    echo "  CONFIG_FILE  - Path to custom config YAML (optional)"
    echo "  CONFIG_DIR   - Directory containing config files (default: /workspace/configs)"
    echo "  OUT_DIR      - Output directory for artifacts (default: /workspace/artifacts)"
    echo "  DATA_DIR     - Directory containing datasets (default: /workspace/data)"
    echo ""
    echo "Examples:"
    echo "  $0 all                         # Run all experiments (CPU)"
    echo "  USE_GPU=true $0 all            # Run all experiments (GPU)"
    echo "  $0 predictors                  # Run only predictors"
    echo "  CONFIG_FILE=/path/to/config.yaml $0 data_split"
    echo ""
    echo "Docker Compose:"
    echo "  docker compose run --rm cphosfing all        # CPU mode"
    echo "  docker compose run --rm cphosfing-gpu all    # GPU mode"
}

# Main entry point
main() {
    local command="${1:-all}"
    
    case "$command" in
        -h|--help|help)
            print_usage
            exit 0
            ;;
        all)
            configure_gpu
            run_all
            ;;
        data_split|predictors|confpred)
            configure_gpu
            run_experiment "$command"
            ;;
        *)
            log_error "Unknown command: $command"
            print_usage
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"
