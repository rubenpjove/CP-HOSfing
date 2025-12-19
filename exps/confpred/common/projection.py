"""
Helper module for Baseline B: projection of leaf CP sets upward to major/family levels.
"""
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple

import logging


def build_vocab_maps_for_B(
    leaf_vocab_global: np.ndarray,
    major_vocab_global: np.ndarray,
    family_vocab_global: np.ndarray,
    leaf_to_major_global: Dict[int, int],
    major_to_family_global: Dict[int, int],
    leaf_to_family_global: Dict[int, int] = None,
    logger: logging.Logger = None,
) -> Tuple[csr_matrix, csr_matrix, Dict[int, int], Dict[int, int]]:
    """Build sparse incidence matrices for upward projection.
    
    Args:
        leaf_vocab_global: Array of unique global leaf idx values from Train
        major_vocab_global: Array of unique global major idx values from Train
        family_vocab_global: Array of unique global family idx values from Train
        leaf_to_major_global: Dict mapping leaf global idx -> major global idx
        major_to_family_global: Dict mapping major global idx -> family global idx
        logger: Optional logger for warnings
        
    Returns:
        A_major: Sparse boolean CSR matrix [m_major, m_leaf], where A_major[i,j]=True 
                iff leaf[j] descends from major[i]
        A_family: Sparse boolean CSR matrix [m_family, m_leaf], same for families
        maj_g2l: Dict mapping global major idx -> local row index in A_major
        fam_g2l: Dict mapping global family idx -> local row index in A_family
    """
    m_leaf = len(leaf_vocab_global)
    m_major = len(major_vocab_global)
    m_family = len(family_vocab_global)
    
    # Build global idx -> local row index mappings
    # Use integer keys for type consistency with dictionary lookups
    leaf_g2l = {int(g): i for i, g in enumerate(leaf_vocab_global)}
    maj_g2l = {int(g): i for i, g in enumerate(major_vocab_global)}
    fam_g2l = {int(g): i for i, g in enumerate(family_vocab_global)}
    
    # Build A_major: for each leaf, find its major and set the corresponding entry
    major_rows = []
    major_cols = []
    
    for j, leaf_gid in enumerate(leaf_vocab_global):
        leaf_gid_int = int(leaf_gid)
        # Get major for this leaf (if exists) - ensure integer type
        major_gid = leaf_to_major_global.get(leaf_gid_int, None)
        
        if major_gid is not None:
            major_gid_int = int(major_gid)
            if major_gid_int in maj_g2l:
            # Leaf has a major and that major is in the vocab
                i = maj_g2l[major_gid_int]
            major_rows.append(i)
            major_cols.append(j)
        # If leaf has no major or major not in vocab, no entry added (ragged hierarchy)
    
    # Build A_family: for each leaf, find its family (direct or via major)
    family_rows = []
    family_cols = []
    
    # Use provided leaf_to_family_global if available, otherwise derive from chain
    if leaf_to_family_global is None:
        # Fall back to leaf->major->family chain where major exists
        leaf_to_family_global = {}
        for leaf_gid in leaf_vocab_global:
            leaf_gid_int = int(leaf_gid)
            major_gid = leaf_to_major_global.get(leaf_gid_int, None)
            if major_gid is not None:
                major_gid_int = int(major_gid)
                family_gid = major_to_family_global.get(major_gid_int, None)
                if family_gid is not None:
                    leaf_to_family_global[leaf_gid_int] = int(family_gid)
    
    for j, leaf_gid in enumerate(leaf_vocab_global):
        leaf_gid_int = int(leaf_gid)
        family_gid = leaf_to_family_global.get(leaf_gid_int, None)
        
        if family_gid is not None:
            family_gid_int = int(family_gid)
            if family_gid_int in fam_g2l:
                i = fam_g2l[family_gid_int]
            family_rows.append(i)
            family_cols.append(j)
        # If leaf has no family or family not in vocab, no entry added
    
    # Build sparse CSR matrices
    if major_rows:
        A_major = csr_matrix(
            (np.ones(len(major_rows), dtype=bool), (major_rows, major_cols)),
            shape=(m_major, m_leaf),
            dtype=bool
        )
    else:
        A_major = csr_matrix((m_major, m_leaf), dtype=bool)
    
    if family_rows:
        A_family = csr_matrix(
            (np.ones(len(family_rows), dtype=bool), (family_rows, family_cols)),
            shape=(m_family, m_leaf),
            dtype=bool
        )
    else:
        A_family = csr_matrix((m_family, m_leaf), dtype=bool)
    
    if logger:
        logger.info(
            f"Built incidence matrices: A_major {A_major.shape} ({A_major.nnz} nonzeros), "
            f"A_family {A_family.shape} ({A_family.nnz} nonzeros)"
        )
    
    return A_major, A_family, maj_g2l, fam_g2l


def project_leaf_masks_upward(
    keep_leaf_bool: np.ndarray,
    A_major: csr_matrix,
    A_family: csr_matrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project leaf keep masks upward to major and family levels.
    
    Args:
        keep_leaf_bool: [N, m_leaf] boolean array indicating which leaves are kept per sample
        A_major: Sparse boolean CSR [m_major, m_leaf] incidence matrix
        A_family: Sparse boolean CSR [m_family, m_leaf] incidence matrix
        
    Returns:
        keep_major_bool: [N, m_major] boolean array
        keep_family_bool: [N, m_family] boolean array
    """
    # Matrix multiplication: A @ keep_leaf.T gives [m_ancestor, N]
    # Then transpose to get [N, m_ancestor]
    # Result > 0 means at least one descendant leaf is kept
    keep_major_bool = (A_major @ keep_leaf_bool.T > 0).T.astype(bool)
    keep_family_bool = (A_family @ keep_leaf_bool.T > 0).T.astype(bool)
    
    return keep_major_bool, keep_family_bool


def evaluate_sets_from_masks(
    mask_bool: np.ndarray,
    y_true_local: np.ndarray,
) -> Dict[str, float]:
    """Evaluate CP sets from boolean masks.
    
    Args:
        mask_bool: [N, m_class] boolean array indicating which classes are in the set
        y_true_local: [N] array of true class labels (local indices)
        
    Returns:
        Dict with metrics: coverage, set_size_mean, set_size_median, empty_rate, singleton_rate
    """
    if mask_bool.shape[0] == 0:
        return {
            "coverage": float("nan"),
            "set_size_mean": float("nan"),
            "set_size_median": float("nan"),
            "empty_rate": float("nan"),
            "singleton_rate": float("nan"),
        }
    
    # Set sizes
    set_sizes = mask_bool.sum(axis=1).astype("int64")
    empty_rate = float(np.mean(set_sizes == 0))
    singleton_rate = float(np.mean(set_sizes == 1))
    set_size_mean = float(np.mean(set_sizes))
    set_size_median = float(np.median(set_sizes))
    
    # Coverage: mean(indicator{ true in set })
    rows = np.arange(mask_bool.shape[0])
    covered = mask_bool[rows, y_true_local]
    coverage = float(np.mean(covered))
    
    return {
        "coverage": coverage,
        "set_size_mean": set_size_mean,
        "set_size_median": set_size_median,
        "empty_rate": empty_rate,
        "singleton_rate": singleton_rate,
    }

