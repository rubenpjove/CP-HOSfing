"""CP-HOSfing refactored experiment package.

This package organizes the original monolithic main.py into cohesive modules:
- data: dataset IO, preprocessing, hierarchical encoding, feature selection
- models: model architectures and class-weight utilities
- train: training loops, CV/search, and orchestration
- infer: inference helpers for predictions and probabilities

Importing this package has no side effects.
"""

__all__ = [
    "data",
    "models",
    "train",
    "infer",
    "feature_encodings",
]


