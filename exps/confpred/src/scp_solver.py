"""
Structured Conformal Prediction (S-CP) implementation for hierarchical OS fingerprinting.

Based on Zhang et al. (ICLR 2025), this method solves an optimization problem to find
the optimal set of tree nodes (mixing Family, Major, and Minor nodes) that satisfies
a probability threshold while keeping the set size small.
"""
import logging
import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict

try:
    # Prefer PuLP for better stability on Linux servers
    import pulp
    MIP_AVAILABLE = True
    USE_PULP = True
    mip = pulp
except ImportError:
    try:
        import mip
        MIP_AVAILABLE = True
        USE_PULP = False
    except ImportError:
        MIP_AVAILABLE = False
        USE_PULP = False
        mip = None


class StructuredCPSolver:
    """
    Solver for Structured Conformal Prediction using Integer Linear Programming.
    
    Solves the optimization problem:
        min |leaves(y)| s.t. sum(prob(leaf)) >= tau and |y| <= m
    
    Where y is a set of nodes (can mix Family, Major, Minor) and leaves(y) are
    the leaf descendants of nodes in y.
    """
    
    def __init__(
        self,
        leaf_probs: np.ndarray,
        leaf_vocab_global: np.ndarray,
        major_vocab_global: np.ndarray,
        family_vocab_global: np.ndarray,
        leaf_to_major: Dict[int, int],
        major_to_family: Dict[int, int],
        leaf_to_family: Dict[int, int],
        m: int = 4,
        use_fallback: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the S-CP solver.
        
        Args:
            leaf_probs: [n_leaves] array of probabilities for each leaf
            leaf_vocab_global: Array of global leaf indices
            major_vocab_global: Array of global major indices
            family_vocab_global: Array of global family indices
            leaf_to_major: Dict mapping leaf global idx -> major global idx
            major_to_family: Dict mapping major global idx -> family global idx
            leaf_to_family: Dict mapping leaf global idx -> family global idx
            m: Maximum number of nodes allowed in prediction set
            use_fallback: If True, return all families on solver failure (pragmatic). If False, return empty set (rigorous).
            logger: Optional logger instance
        """
        if not MIP_AVAILABLE:
            raise ImportError(
                "Neither 'mip' nor 'pulp' library is available. "
                "Please install one: pip install mip or pip install pulp"
            )

        # Initialize logger early so it is available for all checks below
        self.logger = logger or logging.getLogger(__name__)

        # Validate input dimensions and alignment
        leaf_probs = np.asarray(leaf_probs)
        leaf_vocab_global = np.asarray(leaf_vocab_global)
        
        if len(leaf_probs) != len(leaf_vocab_global):
            raise ValueError(
                f"Probability-vocabulary dimension mismatch: "
                f"leaf_probs has {len(leaf_probs)} elements, but leaf_vocab_global has {len(leaf_vocab_global)} elements. "
                "This indicates a critical alignment error. Probabilities must be aligned with vocabulary indices."
            )
        
        # Validate probability properties
        prob_sum = float(np.sum(leaf_probs))
        prob_min = float(np.min(leaf_probs))
        prob_max = float(np.max(leaf_probs))
        
        if prob_sum < 0.99 or prob_sum > 1.01:
            if self.logger:
                self.logger.warning(
                    f"Leaf probabilities do not sum to 1.0: sum={prob_sum:.6f}. "
                    "This may indicate normalization issues or missing classes."
                )
        
        if prob_min < 0.0 or prob_max > 1.0:
            raise ValueError(
                f"Invalid probability values: min={prob_min:.6f}, max={prob_max:.6f}. "
                "Probabilities must be in [0, 1]."
            )
        
        if self.logger:
            self.logger.debug(
                f"S-CP solver initialized: n_leaves={len(leaf_probs)}, "
                f"prob_sum={prob_sum:.6f}, prob_range=[{prob_min:.6f}, {prob_max:.6f}]"
            )
        
        self.leaf_probs = leaf_probs
        self.leaf_vocab_global = leaf_vocab_global
        self.major_vocab_global = major_vocab_global
        self.family_vocab_global = family_vocab_global
        self.leaf_to_major = leaf_to_major
        self.major_to_family = major_to_family
        self.leaf_to_family = leaf_to_family
        self.m = m
        self.use_fallback = use_fallback
        
        # Build reverse mappings and ancestor/descendant relationships
        self._build_hierarchy_maps()
    
    def _build_hierarchy_maps(self):
        """Build maps for ancestors and descendants of each node."""
        n_leaves = len(self.leaf_vocab_global)
        n_majors = len(self.major_vocab_global)
        n_families = len(self.family_vocab_global)
        
        # Build global -> local index mappings
        self.leaf_g2l = {g: i for i, g in enumerate(self.leaf_vocab_global)}
        self.major_g2l = {g: i for i, g in enumerate(self.major_vocab_global)}
        self.family_g2l = {g: i for i, g in enumerate(self.family_vocab_global)}
        
        # Build local -> global index mappings
        self.leaf_l2g = {i: g for i, g in enumerate(self.leaf_vocab_global)}
        self.major_l2g = {i: g for i, g in enumerate(self.major_vocab_global)}
        self.family_l2g = {i: g for i, g in enumerate(self.family_vocab_global)}
        
        # Build ancestor maps: for each node, store its ancestors
        # leaf ancestors: [major, family]
        # major ancestors: [family]
        # family ancestors: []
        self.leaf_ancestors: Dict[int, List[Tuple[str, int]]] = {}
        for leaf_idx, leaf_gid in enumerate(self.leaf_vocab_global):
            ancestors = []
            # Get major ancestor
            major_gid = self.leaf_to_major.get(leaf_gid)
            if major_gid is not None and major_gid in self.major_g2l:
                ancestors.append(("major", self.major_g2l[major_gid]))
            # Get family ancestor (direct or via major)
            family_gid = self.leaf_to_family.get(leaf_gid)
            if family_gid is None and major_gid is not None:
                family_gid = self.major_to_family.get(major_gid)
            if family_gid is not None and family_gid in self.family_g2l:
                ancestors.append(("family", self.family_g2l[family_gid]))
            self.leaf_ancestors[leaf_idx] = ancestors
        
        self.major_ancestors: Dict[int, List[Tuple[str, int]]] = {}
        for major_idx, major_gid in enumerate(self.major_vocab_global):
            ancestors = []
            family_gid = self.major_to_family.get(major_gid)
            if family_gid is not None and family_gid in self.family_g2l:
                ancestors.append(("family", self.family_g2l[family_gid]))
            self.major_ancestors[major_idx] = ancestors
        
        # All nodes: combine leaves, majors, families
        # Format: (node_type, local_idx) -> global_idx
        self.all_nodes: List[Tuple[str, int, int]] = []  # (type, local_idx, global_idx)
        for leaf_idx, leaf_gid in enumerate(self.leaf_vocab_global):
            self.all_nodes.append(("leaf", leaf_idx, leaf_gid))
        for major_idx, major_gid in enumerate(self.major_vocab_global):
            self.all_nodes.append(("major", major_idx, major_gid))
        for family_idx, family_gid in enumerate(self.family_vocab_global):
            self.all_nodes.append(("family", family_idx, family_gid))
        
        self.n_nodes = len(self.all_nodes)
        self.n_leaves = n_leaves
    
    def solve(self, tau: float, time_limit: float = 30.0) -> Set[int]:
        """
        Solve the ILP for a given tau with robust error handling and type casting.
        
        Args:
            tau: Probability threshold (0 <= tau <= 1)
            time_limit: Maximum time in seconds for solver
            
        Returns:
            Set of global node indices in the optimal solution
        """
        # 1. Cast tau to native python float
        tau = float(tau)
        total_mass = float(np.sum(self.leaf_probs))
        
        # Robustness: If tau is asked to be 1.0 (or higher than sum), 
        # cap it slightly below the total mass to prevent numerical infeasibility.
        # This prevents the "OptimizationStatus.ERROR" crashes.
        effective_tau = min(tau, total_mass - 1e-7)
        
        if effective_tau <= 0:
            return set()

        try:
            # Setup Model
            if USE_PULP:
                model = mip.LpProblem("scp", mip.LpMinimize)
            else:
                model = mip.Model("scp")
                model.verbose = 0  # Disable verbose output
        
            # --- Variables ---
            alpha = {}
            for node_idx in range(self.n_nodes):
                if USE_PULP:
                    alpha[node_idx] = mip.LpVariable(f"a_{node_idx}", cat=mip.LpBinary)
                else:
                    alpha[node_idx] = model.add_var(var_type=mip.BINARY)
            
            beta = {}
            for leaf_idx in range(self.n_leaves):
                if USE_PULP:
                    beta[leaf_idx] = mip.LpVariable(f"b_{leaf_idx}", cat=mip.LpBinary)
                else:
                    beta[leaf_idx] = model.add_var(var_type=mip.BINARY)
            
            # --- Objective ---
            if USE_PULP:
                model += mip.lpSum(beta[i] for i in range(self.n_leaves))
            else:
                model.objective = mip.xsum(beta[i] for i in range(self.n_leaves))
            
            # --- Constraint 1: Cardinality (m) ---
            if USE_PULP:
                model += mip.lpSum(alpha[i] for i in range(self.n_nodes)) <= self.m
            else:
                model += mip.xsum(alpha[i] for i in range(self.n_nodes)) <= self.m
            
            # --- Constraint 2: Coverage Logic ---
            for leaf_idx in range(self.n_leaves):
                # Leaf covered if self selected
                leaf_node_idx = self._get_node_idx("leaf", leaf_idx)
                model += beta[leaf_idx] >= alpha[leaf_node_idx]
                
                # Leaf covered if ancestor selected
                ancestors = self.leaf_ancestors[leaf_idx]
                for anc_type, anc_local_idx in ancestors:
                    anc_node_idx = self._get_node_idx(anc_type, anc_local_idx)
                    model += beta[leaf_idx] >= alpha[anc_node_idx]

                # Upper bound: a leaf can be marked covered only if at least one
                # covering node (itself or an ancestor) is selected. Without this,
                # the solver could satisfy the probability constraint via beta
                # alone, leaving all alpha=0 and producing empty prediction sets.
                anc_node_idxs = [self._get_node_idx(t, i) for t, i in ancestors]
                if USE_PULP:
                    model += beta[leaf_idx] <= alpha[leaf_node_idx] + mip.lpSum(alpha[i] for i in anc_node_idxs)
                else:
                    model += beta[leaf_idx] <= alpha[leaf_node_idx] + mip.xsum(alpha[i] for i in anc_node_idxs)
            
            # --- Constraint 3: Probability Mass ---
            # CRITICAL: Cast numpy probabilities to native float for the solver
            probs_native = [float(p) for p in self.leaf_probs]
            
            if USE_PULP:
                model += mip.lpSum(probs_native[i] * beta[i] for i in range(self.n_leaves)) >= effective_tau
            else:
                model += mip.xsum(probs_native[i] * beta[i] for i in range(self.n_leaves)) >= effective_tau
            
            # --- Solve ---
            if USE_PULP:
                # Use default solver, suppress output
                model.solve(mip.PULP_CBC_CMD(msg=0))
                is_optimal = (mip.LpStatus[model.status] == "Optimal")
            else:
                model.optimize(max_seconds=time_limit)
                is_optimal = (model.status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE])
            
            # Extract Solution
            if is_optimal:
                selected_nodes = set()
                for node_idx in range(self.n_nodes):
                    val = alpha[node_idx].varValue if USE_PULP else alpha[node_idx].x
                    if val is not None and val > 0.5:
                        _, _, global_idx = self.all_nodes[node_idx]
                        selected_nodes.add(global_idx)
                return selected_nodes
            else:
                # --- Fallback on solver failure ---
                if self.use_fallback:
                    # Pragmatic: return all families to ensure coverage
                    if self.logger:
                        self.logger.debug(f"Solver failed (Status={model.status}). Fallback to all families (pragmatic).")
                    return set(self.family_vocab_global)
                else:
                    # Rigorous: return empty set to expose method limitations
                    if self.logger:
                        self.logger.debug(f"Solver failed (Status={model.status}). Returning empty set (rigorous).")
                    return set()
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"ILP Exception: {str(e)}")
            # Fallback on exception
            if self.use_fallback:
                # Pragmatic: return all families to ensure coverage
                return set(self.family_vocab_global)
            else:
                # Rigorous: return empty set to expose method limitations
                return set()
    
    def _fallback_solution(self) -> Set[int]:
        """
        Fallback solution when solver fails: return the family node with highest probability mass.
        """
        # Find family with highest sum of leaf probabilities
        family_scores = {}
        for family_idx, family_gid in enumerate(self.family_vocab_global):
            score = 0.0
            # Sum probabilities of all leaves under this family
            for leaf_idx, leaf_gid in enumerate(self.leaf_vocab_global):
                family_gid_from_leaf = self.leaf_to_family.get(leaf_gid)
                if family_gid_from_leaf is None:
                    # Try via major
                    major_gid = self.leaf_to_major.get(leaf_gid)
                    if major_gid is not None:
                        family_gid_from_leaf = self.major_to_family.get(major_gid)
                if family_gid_from_leaf == family_gid:
                    score += float(self.leaf_probs[leaf_idx])
            family_scores[family_gid] = score
        
        if family_scores:
            best_family = max(family_scores.items(), key=lambda x: x[1])[0]
            return {best_family}
        else:
            # Ultimate fallback: empty set
            return set()
    
    def _get_node_idx(self, node_type: str, local_idx: int) -> int:
        """Get the index in all_nodes list for a given node type and local index."""
        if node_type == "leaf":
            return local_idx
        elif node_type == "major":
            return self.n_leaves + local_idx
        elif node_type == "family":
            return self.n_leaves + len(self.major_vocab_global) + local_idx
        else:
            raise ValueError(f"Unknown node type: {node_type}")


def calibrate_scp(
    solver_factory,
    probs_cal: np.ndarray,
    y_cal_global: np.ndarray,
    alpha: float,
    tau_grid: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, Dict]:
    """
    Calibrate tau for S-CP to achieve target coverage 1-alpha.
    
    Args:
        solver_factory: Function that creates StructuredCPSolver given leaf_probs
        probs_cal: [n_cal, n_leaves] array of leaf probabilities for calibration set
        y_cal_global: [n_cal] array of true leaf global indices
        alpha: Target miscoverage rate
        tau_grid: Optional grid of tau values to search (default: np.linspace(0, 1, 100))
        logger: Optional logger instance
        
    Returns:
        Tuple of (optimal_tau, calibration_info_dict)
    """
    """
    Calibrate tau for S-CP to achieve target coverage 1-alpha.
    
    Args:
        solver_factory: Function that creates StructuredCPSolver given leaf_probs
        probs_cal: [n_cal, n_leaves] array of leaf probabilities for calibration set
        y_cal_global: [n_cal] array of true leaf global indices
        alpha: Target miscoverage rate
        tau_grid: Optional grid of tau values to search (default: np.linspace(0, 1, 100))
        logger: Optional logger instance
        
    Returns:
        Tuple of (optimal_tau, calibration_info_dict)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate inputs
    probs_cal = np.asarray(probs_cal)
    y_cal_global = np.asarray(y_cal_global)
    
    if len(probs_cal.shape) != 2:
        raise ValueError(f"probs_cal must be 2D array, got shape {probs_cal.shape}")
    
    n_cal, n_leaves = probs_cal.shape
    
    if len(y_cal_global) != n_cal:
        raise ValueError(
            f"Dimension mismatch: probs_cal has {n_cal} samples, "
            f"but y_cal_global has {len(y_cal_global)} samples"
        )
    
    # Validate probability properties
    prob_sums = np.sum(probs_cal, axis=1)
    prob_sum_mean = float(np.mean(prob_sums))
    prob_sum_std = float(np.std(prob_sums))
    
    if prob_sum_mean < 0.99 or prob_sum_mean > 1.01:
        logger.warning(
            f"Calibration probabilities do not sum to 1.0 on average: "
            f"mean={prob_sum_mean:.6f}, std={prob_sum_std:.6f}"
        )
    
    logger.debug(
        f"Calibration input validation: "
        f"n_cal={n_cal}, n_leaves={n_leaves}, "
        f"prob_sum_mean={prob_sum_mean:.6f}, "
        f"prob_sum_std={prob_sum_std:.6f}"
    )
    
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, 100)
    
    target_coverage = 1.0 - alpha
    
    # Pre-compute maximum achievable probability mass for each calibration sample
    # This is the sum of all leaf probabilities (theoretical upper bound)
    # Note: The actual maximum might be lower due to cardinality constraint m,
    # but sum(leaf_probs) is a safe upper bound
    max_prob_masses = np.array([np.sum(probs_cal[i]) for i in range(n_cal)])
    global_max_tau = np.min(max_prob_masses)  # Minimum across all samples
    
    # Filter tau_grid to only include feasible values
    tau_grid = tau_grid[tau_grid <= global_max_tau + 1e-6]
    
    if len(tau_grid) == 0:
        logger.warning(f"No feasible tau values found. global_max_tau={global_max_tau:.4f}")
        return 0.0, {
            "tau": 0.0,
            "coverage": 0.0,
            "target_coverage": target_coverage,
            "n_cal": n_cal,
        }
    
    # Binary search for optimal tau
    left = 0
    right = len(tau_grid) - 1
    best_tau = None
    best_coverage = 0.0
    
    logger.info(f"Calibrating S-CP: target coverage={target_coverage:.4f}, n_cal={n_cal}, max_feasible_tau={global_max_tau:.4f}")
    
    # Binary search
    while left <= right:
        mid = (left + right) // 2
        tau = tau_grid[mid]
        
        # Evaluate coverage at this tau
        # Get hierarchy mappings from first solver (they're the same for all)
        temp_solver = solver_factory(probs_cal[0])
        leaf_to_major = temp_solver.leaf_to_major
        major_to_family = temp_solver.major_to_family
        leaf_to_family = temp_solver.leaf_to_family
        
        covered_count = 0
        for i in range(n_cal):
            leaf_probs = probs_cal[i]
            solver = solver_factory(leaf_probs)
            selected_nodes = solver.solve(tau)
            
            # Check if true leaf is covered (is descendant of any selected node)
            true_leaf_gid = y_cal_global[i]
            if _is_leaf_covered(true_leaf_gid, selected_nodes, leaf_to_major, major_to_family, leaf_to_family):
                covered_count += 1
        
        coverage = covered_count / n_cal
        
        logger.debug(f"tau={tau:.4f}, coverage={coverage:.4f}")
        
        if coverage >= target_coverage:
            best_tau = tau
            best_coverage = coverage
            right = mid - 1
        else:
            left = mid + 1
    
    if best_tau is None:
        # Fallback: use largest feasible tau (not tau=1.0 which might be infeasible)
        best_tau = tau_grid[-1]
        logger.warning(f"No tau found satisfying coverage requirement. Using largest feasible tau={best_tau:.4f}")
        # Re-evaluate coverage at this tau to get accurate best_coverage
        temp_solver = solver_factory(probs_cal[0])
        leaf_to_major = temp_solver.leaf_to_major
        major_to_family = temp_solver.major_to_family
        leaf_to_family = temp_solver.leaf_to_family
        covered_count = 0
        for i in range(n_cal):
            leaf_probs = probs_cal[i]
            solver = solver_factory(leaf_probs)
            selected_nodes = solver.solve(best_tau)
            true_leaf_gid = y_cal_global[i]
            if _is_leaf_covered(true_leaf_gid, selected_nodes, leaf_to_major, major_to_family, leaf_to_family):
                covered_count += 1
        best_coverage = covered_count / n_cal
    
    logger.info(f"S-CP calibration complete: tau={best_tau:.4f}, coverage={best_coverage:.4f}")
    
    return best_tau, {
        "tau": best_tau,
        "coverage": best_coverage,
        "target_coverage": target_coverage,
        "n_cal": n_cal,
    }


def _is_leaf_covered(
    leaf_gid: int,
    selected_nodes: Set[int],
    leaf_to_major: Dict[int, int],
    major_to_family: Dict[int, int],
    leaf_to_family: Dict[int, int],
) -> bool:
    """Check if a leaf is covered by any node in the selected set."""
    if leaf_gid in selected_nodes:
        return True
    
    # Check if any ancestor of leaf is selected
    # Check major ancestor
    major_gid = leaf_to_major.get(leaf_gid)
    if major_gid is not None and major_gid in selected_nodes:
        return True
    
    # Check family ancestor (direct or via major)
    family_gid = leaf_to_family.get(leaf_gid)
    if family_gid is None and major_gid is not None:
        family_gid = major_to_family.get(major_gid)
    if family_gid is not None and family_gid in selected_nodes:
        return True
    
    return False


def diagnose_probability_shift(
    probs_cal: np.ndarray,
    y_cal_global: np.ndarray,
    probs_test: np.ndarray,
    y_test_global: np.ndarray,
    leaf_vocab_global: np.ndarray,
    logger: logging.Logger
):
    """
    Analyzes if there is a distribution shift in model confidence between Calibration and Test sets.
    
    This diagnostic helps identify if S-CP under-coverage is due to the model being less confident
    on test data compared to calibration data. If test data has lower true-class probabilities,
    the tau threshold learned on calibration data may be too strict for test data.
    
    Args:
        probs_cal: [n_cal, n_leaves] array of leaf probabilities for calibration set
        y_cal_global: [n_cal] array of true leaf global indices for calibration set
        probs_test: [n_test, n_leaves] array of leaf probabilities for test set
        y_test_global: [n_test] array of true leaf global indices for test set
        leaf_vocab_global: Array of global leaf indices (vocabulary)
        logger: Logger instance for output
    """
    # Build Map: Global ID -> Local Column Index
    g2l = {int(g): i for i, g in enumerate(leaf_vocab_global)}
    
    def get_true_probs(probs, labels_global):
        """Extract probabilities of the true label for each sample."""
        true_class_probs = []
        skipped_count = 0
        for i, global_id in enumerate(labels_global):
            global_id_int = int(global_id)
            # Skip if label not in vocabulary (shouldn't happen with filtered data, but handle gracefully)
            if global_id_int not in g2l:
                skipped_count += 1
                continue
            col_idx = g2l[global_id_int]
            p_true = float(probs[i, col_idx])
            true_class_probs.append(p_true)
        
        if skipped_count > 0:
            logger.warning(
                f"diagnose_probability_shift: Skipped {skipped_count}/{len(labels_global)} samples "
                f"with labels not in vocabulary"
            )
        
        return np.array(true_class_probs)

    # Extract Probabilities of the True Label
    scores_cal = get_true_probs(probs_cal, y_cal_global)
    scores_test = get_true_probs(probs_test, y_test_global)
    
    if len(scores_cal) == 0 or len(scores_test) == 0:
        logger.warning(
            "diagnose_probability_shift: Cannot compute statistics - "
            f"cal_samples={len(scores_cal)}, test_samples={len(scores_test)}"
        )
        return

    # Compute Statistics
    mean_cal = float(np.mean(scores_cal))
    mean_test = float(np.mean(scores_test))
    median_cal = float(np.median(scores_cal))
    median_test = float(np.median(scores_test))
    std_cal = float(np.std(scores_cal))
    std_test = float(np.std(scores_test))
    
    # "Hard Samples": How often is the model < 10% confident in the truth?
    hard_cal = float(np.mean(scores_cal < 0.10))
    hard_test = float(np.mean(scores_test < 0.10))
    
    # Additional statistics: very low confidence samples (< 1%)
    very_hard_cal = float(np.mean(scores_cal < 0.01))
    very_hard_test = float(np.mean(scores_test < 0.01))
    
    # Percentiles for tail analysis
    p10_cal = float(np.percentile(scores_cal, 10))
    p10_test = float(np.percentile(scores_test, 10))
    p25_cal = float(np.percentile(scores_cal, 25))
    p25_test = float(np.percentile(scores_test, 25))

    # Log Results
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC: Calibration vs Test Probability Shift")
    logger.info("=" * 80)
    logger.info(f"Sample sizes: Cal={len(scores_cal)}, Test={len(scores_test)}")
    logger.info("")
    logger.info("True Class Probability Statistics:")
    logger.info(f"  Mean    : Cal={mean_cal:.4f} vs Test={mean_test:.4f} (Diff: {mean_test-mean_cal:+.4f})")
    logger.info(f"  Median  : Cal={median_cal:.4f} vs Test={median_test:.4f} (Diff: {median_test-median_cal:+.4f})")
    logger.info(f"  Std Dev : Cal={std_cal:.4f} vs Test={std_test:.4f}")
    logger.info("")
    logger.info("Percentiles (tail analysis):")
    logger.info(f"  10th percentile: Cal={p10_cal:.4f} vs Test={p10_test:.4f} (Diff: {p10_test-p10_cal:+.4f})")
    logger.info(f"  25th percentile: Cal={p25_cal:.4f} vs Test={p25_test:.4f} (Diff: {p25_test-p25_cal:+.4f})")
    logger.info("")
    logger.info("Hard Sample Rates:")
    logger.info(f"  P(True) < 0.10: Cal={hard_cal:.2%} vs Test={hard_test:.2%} (Diff: {hard_test-hard_cal:+.2%})")
    logger.info(f"  P(True) < 0.01: Cal={very_hard_cal:.2%} vs Test={very_hard_test:.2%} (Diff: {very_hard_test-very_hard_cal:+.2%})")
    logger.info("")
    
    # Interpretation
    if mean_test < mean_cal - 0.05:  # Significant drop (>5 percentage points)
        logger.warning(">>> DETECTED SIGNIFICANT SHIFT: Model is less confident on Test data.")
        logger.warning(">>> This explains S-CP under-coverage: The tau learned on Cal is too strict for Test.")
        logger.warning(">>> The model's average confidence dropped by {:.1%} on test data.".format(mean_cal - mean_test))
    elif mean_test < mean_cal - 0.01:  # Small drop (1-5 percentage points)
        logger.warning(">>> DETECTED MILD SHIFT: Model is slightly less confident on Test data.")
        logger.warning(">>> This may contribute to S-CP under-coverage.")
    else:
        logger.info(">>> No significant confidence drop detected between Cal and Test.")
        if hard_test > hard_cal + 0.05:
            logger.info(">>> However, Test set has more 'hard' samples (P<0.10), which may affect tail coverage.")
    
    logger.info("=" * 80)


def _get_label(gid: int, mapping: Optional[Dict[int, str]]) -> str:
    """Helper function to safely convert global ID to class label.
    
    Args:
        gid: Global ID (integer)
        mapping: Optional dictionary mapping global ID to class label string
        
    Returns:
        Class label string if mapping provided, otherwise string representation of gid
    """
    if mapping is not None:
        return mapping.get(gid, str(gid))
    return str(gid)


def project_scp_to_levels(
    scp_set: Set[int],
    leaf_vocab_global: np.ndarray,
    major_vocab_global: np.ndarray,
    family_vocab_global: np.ndarray,
    leaf_to_major: Dict[int, int],
    major_to_family: Dict[int, int],
    leaf_to_family: Dict[int, int],
    logger: Optional[logging.Logger] = None,
    idx2id_map_leaf: Optional[Dict[int, str]] = None,
    idx2id_map_major: Optional[Dict[int, str]] = None,
    idx2id_map_family: Optional[Dict[int, str]] = None,
    debug: bool = False,
) -> Dict[str, Set[int]]:
    """
    Project S-CP prediction set to Family, Major, and Minor level sets.
    
    For each node in scp_set:
    - If Family selected: Add self + all descendants (downward expansion)
    - If Major selected: Add self + all descendants + parent family (upward consistency)
    - If Leaf selected: Add self + parent nodes only (upward consistency, NO siblings/cousins)
    
    This preserves the specificity of precise predictions while maintaining hierarchical consistency.
    
    Args:
        scp_set: Set of global node indices from S-CP
        leaf_vocab_global: Array of global leaf indices
        major_vocab_global: Array of global major indices
        family_vocab_global: Array of global family indices
        leaf_to_major: Dict mapping leaf global idx -> major global idx
        major_to_family: Dict mapping major global idx -> family global idx
        leaf_to_family: Dict mapping leaf global idx -> family global idx
        logger: Optional logger instance for debugging
        idx2id_map_leaf: Optional dict mapping global leaf idx -> class label string
        idx2id_map_major: Optional dict mapping global major idx -> class label string
        idx2id_map_family: Optional dict mapping global family idx -> class label string
        debug: If True, log detailed projection results with class names before validations
        
    Returns:
        Dict with keys "family", "major", "leaf" mapping to sets of global indices
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Initialize result sets
    family_set = set()
    major_set = set()
    leaf_set = set()
    
    # Track directly selected nodes (from scp_set) for validation purposes
    # Only directly selected nodes should have full downward consistency
    directly_selected_families = set()
    directly_selected_majors = set()
    
    # Build reverse mappings for descendants
    major_to_leaves: Dict[int, Set[int]] = {}
    family_to_majors: Dict[int, Set[int]] = defaultdict(set)
    family_to_leaves: Dict[int, Set[int]] = defaultdict(set)
    
    # Initialize major_to_leaves for ALL majors in vocabulary (even if they have no leaves)
    # This ensures consistency when processing families that contain majors without leaves
    major_vocab_set = set(int(g) for g in major_vocab_global)
    for major_gid in major_vocab_global:
        major_to_leaves[int(major_gid)] = set()
    
    # Build major->leaves and family->leaves mappings from leaves
    # Ensure all IDs are converted to integers for type consistency
    for leaf_gid in leaf_vocab_global:
        leaf_gid_int = int(leaf_gid)
        major_gid = leaf_to_major.get(leaf_gid_int)
        if major_gid is not None:
            major_gid_int = int(major_gid)
        else:
            major_gid_int = None
            
        family_gid = leaf_to_family.get(leaf_gid_int)
        if family_gid is None and major_gid_int is not None:
            family_gid = major_to_family.get(major_gid_int)
        
        if major_gid_int is not None and major_gid_int in major_vocab_set:
            # Only add if major is in vocabulary
            major_to_leaves[major_gid_int].add(leaf_gid_int)
        if family_gid is not None:
            family_gid_int = int(family_gid)
            family_to_leaves[family_gid_int].add(leaf_gid_int)
    
    # Build family->majors mapping from ALL majors in vocabulary (not just those with leaves)
    # This ensures that when a family is selected, ALL its majors are added, even if some
    # majors don't have leaves in leaf_vocab_global
    for major_gid in major_vocab_global:
        major_gid_int = int(major_gid)
        family_gid = major_to_family.get(major_gid_int)
        if family_gid is not None:
            family_gid_int = int(family_gid)
            family_to_majors[family_gid_int].add(major_gid_int)
    
    # Defensive check: Verify all majors are captured
    majors_with_leaves = set(major_to_leaves.keys())
    majors_missing_leaves = major_vocab_set - majors_with_leaves
    
    if majors_missing_leaves:
        logger.debug(
            f"S-CP projection: Found {len(majors_missing_leaves)} majors in vocabulary "
            f"without leaves in leaf_vocab_global. These are still included in "
            f"family_to_majors mappings via the comprehensive build."
        )
    
    # Convert vocabularies to sets for faster O(1) membership checks and type safety
    family_vocab_set = set(int(g) for g in family_vocab_global)
    leaf_vocab_set = set(int(g) for g in leaf_vocab_global)
    
    # Log initial scp_set composition
    scp_families = [n for n in scp_set if int(n) in family_vocab_set]
    scp_majors = [n for n in scp_set if int(n) in major_vocab_set]
    scp_leaves = [n for n in scp_set if int(n) in leaf_vocab_set]
    scp_unknown = [n for n in scp_set if int(n) not in family_vocab_set and 
                    int(n) not in major_vocab_set and int(n) not in leaf_vocab_set]
    
    logger.debug(
        f"S-CP projection input: scp_set_size={len(scp_set)}, "
        f"families={len(scp_families)}, majors={len(scp_majors)}, "
        f"leaves={len(scp_leaves)}, unknown={len(scp_unknown)}"
    )
    
    if scp_unknown:
        logger.warning(
            f"S-CP projection: Found {len(scp_unknown)} nodes in scp_set not in any vocabulary. "
            f"These will be ignored. Unknown nodes: {scp_unknown[:10]}{'...' if len(scp_unknown) > 10 else ''}"
        )
    
    # Helper function: Add a family and perform full downward projection
    def add_family_with_descendants(fam_gid_int: int):
        """Add a family to family_set and immediately add all its majors and their leaves."""
        if fam_gid_int in family_set:
            return  # Already processed
        family_set.add(fam_gid_int)
        
        # Add all majors in this family
        if fam_gid_int in family_to_majors:
            majors_to_add = {int(m) for m in family_to_majors[fam_gid_int] if int(m) in major_vocab_set}
            for major_gid in majors_to_add:
                add_major_with_descendants(int(major_gid))
        
        # Also add leaves directly mapped to family (for completeness)
        if fam_gid_int in family_to_leaves:
            leaves_to_add = {int(l) for l in family_to_leaves[fam_gid_int] if int(l) in leaf_vocab_set}
            leaf_set.update(leaves_to_add)
    
    # Helper function: Add a major and perform full downward projection
    def add_major_with_descendants(maj_gid_int: int):
        """Add a major to major_set and immediately add all its leaves."""
        if maj_gid_int in major_set:
            return  # Already processed
        major_set.add(maj_gid_int)
        
        # Add all leaves under this major
        if maj_gid_int in major_to_leaves:
            leaves_to_add = {int(l) for l in major_to_leaves[maj_gid_int] if int(l) in leaf_vocab_set}
            leaf_set.update(leaves_to_add)
    
    # Process each node in scp_set
    for node_gid in scp_set:
        node_gid_int = int(node_gid)  # Ensure integer type
        # Determine node type
        if node_gid_int in family_vocab_set:
            # --- CASE A: FAMILY SELECTED (Vague) ---
            # Action: Add Self + Downward Expansion
            directly_selected_families.add(node_gid_int)
            add_family_with_descendants(node_gid_int)
            
        elif node_gid_int in major_vocab_set:
            # --- CASE B: MAJOR SELECTED (Intermediate) ---
            # Action: Add Self + Downward Expansion + Upward Consistency
            directly_selected_majors.add(node_gid_int)
            add_major_with_descendants(node_gid_int)
            
            # Upward: Add Parent Family (ONLY the parent node, not its other children)
            family_gid = major_to_family.get(node_gid_int)
            if family_gid is not None:
                family_set.add(int(family_gid))
            
        elif node_gid_int in leaf_vocab_set:
            # --- CASE C: LEAF SELECTED (Precise) ---
            # Action: Add Self + Upward Consistency (NO downward expansion)
            leaf_set.add(node_gid_int)
            
            # Upward: Add Parent Major (ONLY the parent node, not its other children)
            major_gid = leaf_to_major.get(node_gid_int)
            if major_gid is not None:
                major_gid_int = int(major_gid)
                major_set.add(major_gid_int)
                
                # Upward from Major: Add Parent Family
                family_gid = major_to_family.get(major_gid_int)
                if family_gid is not None:
                    family_set.add(int(family_gid))
            
            # Upward: Add Parent Family (Direct check if major was missing)
            family_gid = leaf_to_family.get(node_gid_int)
            if family_gid is None and major_gid is not None:
                family_gid = major_to_family.get(int(major_gid))
            if family_gid is not None:
                family_set.add(int(family_gid))
            
        else:
            # Node not found in any vocabulary - this shouldn't happen but log it
            # This could happen if there's a mismatch between solver output and vocabularies
            logger.debug(
                f"S-CP projection: Node {node_gid_int} not found in any vocabulary, skipping"
            )
    
    # Log final projection results
    logger.debug(
        f"S-CP projection output: "
        f"family_set_size={len(family_set)}, "
        f"major_set_size={len(major_set)}, "
        f"leaf_set_size={len(leaf_set)}"
    )
    
    # Debug logging: Show detailed projection results with class names
    if debug:
        def format_set_with_names(node_set: Set[int], mapping: Optional[Dict[int, str]], prefix: str = "") -> str:
            """Format a set of node IDs with their class names."""
            if not node_set:
                return f"{prefix}()"
            names = []
            for node_id in sorted(node_set):
                node_id_int = int(node_id)
                name = mapping.get(node_id_int, str(node_id_int)) if mapping else str(node_id_int)
                names.append(name)
            return f"{prefix}({', '.join(names)})"
        
        # Format S-CP predicted set
        scp_predicted = []
        for node_gid in sorted(scp_set):
            node_gid_int = int(node_gid)
            if node_gid_int in family_vocab_set:
                name = idx2id_map_family.get(node_gid_int, str(node_gid_int)) if idx2id_map_family else str(node_gid_int)
                scp_predicted.append(name)
            elif node_gid_int in major_vocab_set:
                name = idx2id_map_major.get(node_gid_int, str(node_gid_int)) if idx2id_map_major else str(node_gid_int)
                scp_predicted.append(name)
            elif node_gid_int in leaf_vocab_set:
                name = idx2id_map_leaf.get(node_gid_int, str(node_gid_int)) if idx2id_map_leaf else str(node_gid_int)
                scp_predicted.append(name)
            else:
                scp_predicted.append(str(node_gid_int))
        
        scp_predicted_str = f"({', '.join(scp_predicted)})" if scp_predicted else "()"
        family_str = format_set_with_names(family_set, idx2id_map_family, "family")
        major_str = format_set_with_names(major_set, idx2id_map_major, "major")
        leaf_str = format_set_with_names(leaf_set, idx2id_map_leaf, "leaf")
        
        logger.info(
            f"S-CP projection debug: "
            f"predicted={scp_predicted_str}, "
            f"{family_str}, "
            f"{major_str}, "
            f"{leaf_str}"
        )
    
    # ===================================================================
    # COMPREHENSIVE VALIDATION PASSES
    # ===================================================================
    
    # Validation Pass 1: Check that all nodes in sets are in vocabularies
    invalid_families = {f for f in family_set if int(f) not in family_vocab_set}
    invalid_majors = {m for m in major_set if int(m) not in major_vocab_set}
    invalid_leaves = {l for l in leaf_set if int(l) not in leaf_vocab_set}
    
    if invalid_families:
        logger.warning(
            f"S-CP projection validation: Found {len(invalid_families)} invalid family IDs "
            f"not in vocabulary: {list(invalid_families)[:5]}{'...' if len(invalid_families) > 5 else ''}"
        )
    if invalid_majors:
        logger.warning(
            f"S-CP projection validation: Found {len(invalid_majors)} invalid major IDs "
            f"not in vocabulary: {list(invalid_majors)[:5]}{'...' if len(invalid_majors) > 5 else ''}"
        )
    if invalid_leaves:
        logger.warning(
            f"S-CP projection validation: Found {len(invalid_leaves)} invalid leaf IDs "
            f"not in vocabulary: {list(invalid_leaves)[:5]}{'...' if len(invalid_leaves) > 5 else ''}"
        )
    
    # Validation Pass 2: Downward consistency - If a family is DIRECTLY SELECTED, all its majors should be in major_set
    # Note: Families added as parents (via upward consistency) don't need all majors
    downward_family_errors = 0
    for fam_gid in directly_selected_families:
        fam_gid_int = int(fam_gid)
        if fam_gid_int in family_to_majors:
            # Only check majors that are in the vocabulary
            expected_majors = {int(m) for m in family_to_majors[fam_gid_int] if int(m) in major_vocab_set}
            # Ensure major_set contains integers for comparison
            major_set_ints = {int(m) for m in major_set}
            missing_majors = expected_majors - major_set_ints
            if missing_majors:
                # Convert to labels for better readability
                fam_label = _get_label(fam_gid_int, idx2id_map_family)
                missing_labels = [_get_label(m, idx2id_map_major) for m in list(missing_majors)[:5]]
                missing_str = ", ".join(missing_labels)
                if len(missing_majors) > 5:
                    missing_str += "..."
                
                logger.warning(
                    f"S-CP projection validation: Family {fam_label} (ID: {fam_gid_int}) is directly selected, "
                    f"but {len(missing_majors)} of its majors are missing from major_set: {missing_str}"
                )
                logger.debug(
                    f"S-CP projection validation debug: Family {fam_label} (ID: {fam_gid_int}), "
                    f"expected_majors={len(expected_majors)}, "
                    f"major_set_size={len(major_set_ints)}, "
                    f"missing={len(missing_majors)}"
                )
                downward_family_errors += 1
    
    # Validation Pass 3: Downward consistency - If a major is DIRECTLY SELECTED, all its leaves should be in leaf_set
    # Note: Majors added as parents (via upward consistency) don't need all leaves
    downward_major_errors = 0
    for maj_gid in directly_selected_majors:
        maj_gid_int = int(maj_gid)
        if maj_gid_int in major_to_leaves:
            # Only check leaves that are in the vocabulary
            expected_leaves = {int(l) for l in major_to_leaves[maj_gid_int] if int(l) in leaf_vocab_set}
            # Ensure leaf_set contains integers for comparison
            leaf_set_ints = {int(l) for l in leaf_set}
            missing_leaves = expected_leaves - leaf_set_ints
            if missing_leaves:
                # Convert to labels for better readability
                maj_label = _get_label(maj_gid_int, idx2id_map_major)
                missing_labels = [_get_label(l, idx2id_map_leaf) for l in list(missing_leaves)[:5]]
                missing_str = ", ".join(missing_labels)
                if len(missing_leaves) > 5:
                    missing_str += "..."
                
                logger.warning(
                    f"S-CP projection validation: Major {maj_label} (ID: {maj_gid_int}) is directly selected, "
                    f"but {len(missing_leaves)} of its leaves are missing from leaf_set: {missing_str}"
                )
                logger.debug(
                    f"S-CP projection validation debug: Major {maj_label} (ID: {maj_gid_int}), "
                    f"expected_leaves={len(expected_leaves)}, "
                    f"leaf_set_size={len(leaf_set_ints)}, "
                    f"missing={len(missing_leaves)}"
                )
                downward_major_errors += 1
    
    # Validation Pass 4: Upward consistency - If a major is in major_set, its family should be in family_set
    upward_major_errors = 0
    for maj_gid in major_set:
        maj_gid_int = int(maj_gid)
        family_gid = major_to_family.get(maj_gid_int)
        if family_gid is not None:
            family_gid_int = int(family_gid)
            if family_gid_int not in family_set:
                maj_label = _get_label(maj_gid_int, idx2id_map_major)
                fam_label = _get_label(family_gid_int, idx2id_map_family)
                logger.warning(
                    f"S-CP projection validation: Major {maj_label} (ID: {maj_gid_int}) is in major_set, "
                    f"but its family {fam_label} (ID: {family_gid_int}) is missing from family_set"
                )
                upward_major_errors += 1
    
    # Validation Pass 5: Upward consistency - If a leaf is in leaf_set, its major should be in major_set
    upward_leaf_major_errors = 0
    for leaf_gid in leaf_set:
        leaf_gid_int = int(leaf_gid)
        major_gid = leaf_to_major.get(leaf_gid_int)
        if major_gid is not None:
            major_gid_int = int(major_gid)
            if major_gid_int not in major_set:
                leaf_label = _get_label(leaf_gid_int, idx2id_map_leaf)
                maj_label = _get_label(major_gid_int, idx2id_map_major)
                logger.warning(
                    f"S-CP projection validation: Leaf {leaf_label} (ID: {leaf_gid_int}) is in leaf_set, "
                    f"but its major {maj_label} (ID: {major_gid_int}) is missing from major_set"
                )
                upward_leaf_major_errors += 1
    
    # Validation Pass 6: Upward consistency - If a leaf is in leaf_set, its family should be in family_set
    upward_leaf_family_errors = 0
    for leaf_gid in leaf_set:
        leaf_gid_int = int(leaf_gid)
        family_gid = leaf_to_family.get(leaf_gid_int)
        if family_gid is None:
            # Try via major
            major_gid = leaf_to_major.get(leaf_gid_int)
            if major_gid is not None:
                family_gid = major_to_family.get(int(major_gid))
        if family_gid is not None:
            family_gid_int = int(family_gid)
            if family_gid_int not in family_set:
                leaf_label = _get_label(leaf_gid_int, idx2id_map_leaf)
                fam_label = _get_label(family_gid_int, idx2id_map_family)
                logger.warning(
                    f"S-CP projection validation: Leaf {leaf_label} (ID: {leaf_gid_int}) is in leaf_set, "
                    f"but its family {fam_label} (ID: {family_gid_int}) is missing from family_set"
                )
                upward_leaf_family_errors += 1
    
    # Validation Pass 7: Summary statistics
    total_errors = (
        len(invalid_families) + len(invalid_majors) + len(invalid_leaves) +
        downward_family_errors + downward_major_errors +
        upward_major_errors + upward_leaf_major_errors + upward_leaf_family_errors
    )
    
    if total_errors == 0:
        logger.debug(
            f"S-CP projection validation: PASSED - All hierarchical consistency checks passed. "
            f"Sets: family={len(family_set)}, major={len(major_set)}, leaf={len(leaf_set)}"
        )
    else:
        logger.warning(
            f"S-CP projection validation: FAILED - Found {total_errors} consistency errors: "
            f"invalid_nodes={len(invalid_families) + len(invalid_majors) + len(invalid_leaves)}, "
            f"downward_family={downward_family_errors}, downward_major={downward_major_errors}, "
            f"upward_major={upward_major_errors}, upward_leaf_major={upward_leaf_major_errors}, "
            f"upward_leaf_family={upward_leaf_family_errors}"
        )
    
    return {
        "family": family_set,
        "major": major_set,
        "leaf": leaf_set,
    }
