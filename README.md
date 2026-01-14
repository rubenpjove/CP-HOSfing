# Conformal Prediction in Hierarchical Operating System Fingerprinting

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Abstract

This repository provides a reproducible experimental framework for applying **Conformal Prediction (CP)** techniques to **Hierarchical Operating System Fingerprinting**. The methodology leverages the inherent hierarchical tree structure of operating system taxonomies—organized into *family*, *major version*, and *leaf* (specific version) levels—to provide statistically valid prediction sets with guaranteed coverage properties.

The framework implements and evaluates two conformal prediction approaches:

- **Lw-CP** (Level-wise Conformal Prediction)
- **LoUP-CP** (Leaf-only with Upward Projection Conformal Prediction)

## Methodology Overview

```
  ┌─────────────────────────────────────────────────────────────┐
  │                  Hierarchical OS Taxonomy                   │
  │                                                             │
  │       Family ───────────► Major ─────────────► Leaf         │
  │   (e.g., Windows)  (e.g., Windows 10)      (e.g., 22H2)     │
  └─────────────────────────────────────────────────────────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       │                         ▼                         │
       │            ┌───────────────────────┐              │
       │            │   Network Features    │              │
       │            │   (TCP/IP signatures) │              │
       │            └───────────────────────┘              │
       │                         │                         │
       │         ┌───────────────┼───────────────┐         │
       │         ▼               ▼               ▼         │
       │    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
       │    │ Family  │    │  Major  │    │  Leaf   │      │
       │    │   MLP   │    │   MLP   │    │   MLP   │      │
       │    └────┬────┘    └────┬────┘    └────┬────┘      │
       │         │              │              │           │
       │         └──────────────┼──────────────┘           │
       │                        ▼                          │
       │           ┌────────────────────────┐              │
       │           │  Conformal Prediction  │              │
       │           │   (Lw-CP / LoUP-CP)    │              │
       │           └────────────────────────┘              │
       │                        │                          │
       │                        ▼                          │
       │           ┌────────────────────────┐              │
       │           │   Prediction Sets      │              │
       │           │   with Coverage        │              │
       │           │   Guarantees           │              │
       │           └────────────────────────┘              │
       └───────────────────────────────────────────────────┘
```

## Experiments

The experimental pipeline consists of three sequential stages:

### 1. Data Split (`exps/data_split`)

Prepares the dataset for machine learning experiments through stratified partitioning.

**Functionality:**

- Loads and preprocesses passive OS fingerprinting datasets
- Performs stratified train/calibration-test splitting respecting the hierarchical label distribution
- Ensures minimum sample requirements per class for valid conformal prediction calibration

**Key Parameters:**

- `dataset_train_frac`: Proportion allocated to training (default: 0.70)
- `caltest_min_per_class`: Minimum samples per class in calibration/test set
- `seed`: Random seed for reproducibility

### 2. Predictor Training (`exps/predictors`)

Trains hierarchical Multi-Layer Perceptron (MLP) classifiers using PyTorch with optional GPU acceleration.

**Functionality:**

- Trains separate classifiers for each granularity level (family, major, leaf)
- Implements randomized hyperparameter search with cross-validation
- Exports trained models and preprocessing pipelines

### 3. Conformal Prediction (`exps/confpred`)

Executes conformal prediction experiments with comprehensive statistical evaluation.

**Functionality:**

- Implements Lw-CP and LoUP-CP conformal prediction methods
- Performs multi-run experiments across α-levels for coverage analysis
- Aggregates results per-alpha, cross-alpha, and cross-method
- Generates publication-ready visualizations (boxplots, line plots, comparisons)

**Evaluation Metrics:**

- Coverage
- Set Size
- Empty Set Rate
- Singleton Rate
- Hierarchical Inconsistency Rate (HIR)

## Dataset

This project uses the **Passive Operating System Fingerprinting Revisited** dataset by Laštovička et al. (2023), available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7635138.svg)](https://doi.org/10.5281/zenodo.7635138)

The dataset is not included in this repository due to file size restrictions.

## Installation

### Prerequisites

- Docker and Docker Compose v2.x
- **Optional:** NVIDIA GPU with CUDA 12.8 support + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/rubenpjove/CP-HOSfing.git
cd CP-HOSfing

# Build the Docker image
docker compose build

# Download and prepare the dataset
wget https://zenodo.org/records/7635138/files/anonymized_flows.zip
unzip anonymized_flows.zip -d ./data/
mv flows_ground_truth_merged_anonymized.csv dataset.csv
```

## Usage

### Running Experiments

The Docker setup provides two services: **CPU-only** (default) and **GPU-accelerated**.

#### CPU Mode (default)

```bash
# Run the complete experimental pipeline (CPU)
docker compose run --rm cphosfing all

# Run individual experiments
docker compose run --rm cphosfing data_split    # Step 1: Data preparation
docker compose run --rm cphosfing predictors    # Step 2: Model training
docker compose run --rm cphosfing confpred      # Step 3: Conformal prediction
```

#### GPU Mode (requires NVIDIA GPU)

```bash
# Run the complete experimental pipeline (GPU)
docker compose run --rm cphosfing-gpu all

# Run individual experiments with GPU
docker compose run --rm cphosfing-gpu data_split
docker compose run --rm cphosfing-gpu predictors
docker compose run --rm cphosfing-gpu confpred
```

> **Note:** GPU mode automatically enables mixed-precision training (AMP), multi-GPU support, and optimized data loading. When GPU is unavailable or disabled, these features are automatically turned off.

### Configuration

Each experiment has its own configuration file in the `configs/` directory:

| Config File                | Experiment | Description                            |
| -------------------------- | ---------- | -------------------------------------- |
| `data_split_params.yaml` | data_split | Dataset paths, split proportions       |
| `predictors_params.yaml` | predictors | Training hyperparameters, GPU settings |
| `confpred_params.yaml`   | confpred   | CP methods, alpha values, num_runs     |

```bash
# Edit experiment-specific configurations
vim configs/data_split_params.yaml
vim configs/predictors_params.yaml
vim configs/confpred_params.yaml

# Use a custom configuration file (overrides default)
docker compose run --rm -e CONFIG_FILE=/workspace/configs/my_custom.yaml cphosfing predictors
```

**Key Configuration Options:**

| Parameter                   | Config File | Description                          | Default               |
| --------------------------- | ----------- | ------------------------------------ | --------------------- |
| `seed`                    | all         | Random seed for reproducibility      | 42                    |
| `split_proportions.train` | data_split  | Training set proportion              | 0.70                  |
| `caltest_min_per_class`   | data_split  | Min samples per class in cal/test    | 2                     |
| `cv_splits`               | predictors  | Cross-validation folds               | 5                     |
| `max_configs`             | predictors  | Hyperparameter configurations to try | 32                    |
| `models_to_train`         | predictors  | Hierarchy levels to train            | [family, major, leaf] |
| `methods`                 | confpred    | CP methods to evaluate               | [LwCP, LoUPCP]        |
| `alphas`                  | confpred    | Significance levels (1-coverage)     | [0.0, 0.01, ..., 0.5] |

### Directory Structure

```
public/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration configuration
├── entrypoint.sh           # Experiment runner CLI
├── requirements.txt        # Python dependencies
├── CITATION.cff            # Citation metadata
│
├── configs/                # Configuration files
│   ├── data_split_params.yaml
│   ├── predictors_params.yaml
│   └── confpred_params.yaml
│
├── data/                   # Dataset mount point
│   └── dataset.csv         # Your input data
│
├── artifacts/              # Output directory
│   ├── data_split/         # Split datasets and maps
│   ├── predictors/         # Trained models
│   └── confpred/           # CP results and plots
│
└── exps/                   # Experiment source code
    ├── data_split/         # Data preparation module
    ├── predictors/         # Model training module
    ├── confpred/           # Conformal prediction module
    └── utils/              # Shared utilities
```

### Accessing Results

After execution, results are available in the `./artifacts` directory:

```bash
# Model checkpoints
ls artifacts/predictors/*.pt
ls artifacts/predictors/*.joblib

# Conformal prediction results
ls artifacts/confpred/methods_aggregated/
```

## Advanced Usage

### GPU Configuration

```bash
# Run with GPU acceleration
docker compose run --rm cphosfing-gpu all

# Use specific GPUs only
docker compose run --rm -e CUDA_VISIBLE_DEVICES=0,1 cphosfing-gpu all

# Force CPU mode even with GPU service (for testing)
docker compose run --rm -e USE_GPU=false cphosfing-gpu all
```

**GPU-related configuration options** (in `predictors_params.yaml` and `confpred_params.yaml`):

| Parameter         | Description                       | Default  |
| ----------------- | --------------------------------- | -------- |
| `use_amp`       | Enable automatic mixed precision  | `true` |
| `num_workers`   | DataLoader worker processes       | `4`    |
| `use_multi_gpu` | Enable DataParallel for multi-GPU | `true` |

> When running in CPU mode, these are automatically set to `false`, `0`, and `false` respectively.

## Citation

If you use this software in your research, please cite it using the [`CITATION.cff`](CITATION.cff) file.

> **Tip:** Click **"Cite this repository"** in the GitHub sidebar for APA and BibTeX formats.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by the grant ED431C 2022/46 - Competitive Reference Groups GRC - funded by *Xunta de Galicia* (Spain). This work was also supported by CITIC, as a center accredited for excellence within the Galician University System and a member of the CIGUS Network, which receives subsidies from the Department of Education, Science, Universities, and Vocational Training of the *Xunta de Galicia*. Additionally, CITIC is co-financed by the EU through the FEDER Galicia 2021–27 operational program (Ref. ED431G 2023/01). This work was also supported by the *"Formación de Profesorado Universitario"* (FPU) grant from the Spanish Ministry of Universities to Rubén Pérez Jove (Grant FPU22/04418). This work was supported by the inMOTION programme, INDITEX-UDC Predoctoral Research Stay Grants (2025 call), under the collaboration agreement between Universidade da Coruña (UDC) and INDITEX, S.A. Funding for open access charge: Universidade da Coruña/CISUG.
