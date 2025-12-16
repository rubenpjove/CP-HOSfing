import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_dataloaders(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
                     batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    ds_tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    ds_va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long())
    use_cuda = torch.cuda.is_available()
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       pin_memory=use_cuda, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       pin_memory=use_cuda, num_workers=num_workers)
    return dl_tr, dl_va


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts.astype(np.float64))
    w = np.zeros(classes.max() + 1, dtype=np.float64)
    w[classes] = weights
    return torch.tensor(w, dtype=torch.float32)


