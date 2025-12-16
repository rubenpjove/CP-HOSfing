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
            logger: Optional logger instance
        """
        if not MIP_AVAILABLE:
            raise ImportError(
                "Neither 'mip' nor 'pulp' library is available. "
                "Please install one: pip install mip or pip install pulp"
            )
        
        self.leaf_probs = leaf_probs
        self.leaf_vocab_global = leaf_vocab_global
        self.major_vocab_global = major_vocab_global
        self.family_vocab_global = family_vocab_global
        self.leaf_to_major = leaf_to_major
        self.major_to_family = major_to_family
        self.leaf_to_family = leaf_to_family
        self.m = m
        self.logger = logger or logging.getLogger(__name__)
        
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
        
        # 2. Feasibility Check with Epsilon
        # Use a small epsilon to handle floating point errors (e.g. sum is 0.999999 vs 1.0)
        EPS = 1e-6
        max_prob_mass = float(np.sum(self.leaf_probs))
        
        # If the request is mathematically impossible, return empty
        if tau > max_prob_mass + EPS:
            if self.logger:
                self.logger.debug(f"tau={tau:.4f} exceeds max_prob_mass={max_prob_mass:.4f}, returning empty set")
            return set()
            
        # Relax tau slightly to prevent strict boundary failures
        effective_tau = max(0.0, tau - EPS)

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
                # If infeasible (rare with epsilon) or error, fallback to highest probability family
                if self.logger:
                    self.logger.debug(f"Solver non-optimal. Tau={tau:.4f}, m={self.m}")
                # Fallback: return family node with highest probability mass
                return self._fallback_solution()
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"ILP Exception: {str(e)}")
            # Fallback on exception
            return self._fallback_solution()
    
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
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, 100)
    
    n_cal = probs_cal.shape[0]
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
            # This tau works, try smaller tau
            best_tau = tau
            best_coverage = coverage
            right = mid - 1
        else:
            # Need larger tau
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


def project_scp_to_levels(
    scp_set: Set[int],
    leaf_vocab_global: np.ndarray,
    major_vocab_global: np.ndarray,
    family_vocab_global: np.ndarray,
    leaf_to_major: Dict[int, int],
    major_to_family: Dict[int, int],
    leaf_to_family: Dict[int, int],
) -> Dict[str, Set[int]]:
    """
    Project S-CP prediction set to Family, Major, and Minor level sets.
    
    For each node in scp_set:
    1. Add self to its level set
    2. Add all ancestors (upwards consistency)
    3. Add all descendants (downwards implied coverage)
    
    Args:
        scp_set: Set of global node indices from S-CP
        leaf_vocab_global: Array of global leaf indices
        major_vocab_global: Array of global major indices
        family_vocab_global: Array of global family indices
        leaf_to_major: Dict mapping leaf global idx -> major global idx
        major_to_family: Dict mapping major global idx -> family global idx
        leaf_to_family: Dict mapping leaf global idx -> family global idx
        
    Returns:
        Dict with keys "family", "major", "leaf" mapping to sets of global indices
    """
    # Initialize result sets
    family_set = set()
    major_set = set()
    leaf_set = set()
    
    # Build reverse mappings for descendants
    major_to_leaves: Dict[int, Set[int]] = defaultdict(set)
    family_to_majors: Dict[int, Set[int]] = defaultdict(set)
    family_to_leaves: Dict[int, Set[int]] = defaultdict(set)
    
    for leaf_gid in leaf_vocab_global:
        major_gid = leaf_to_major.get(leaf_gid)
        family_gid = leaf_to_family.get(leaf_gid)
        if family_gid is None and major_gid is not None:
            family_gid = major_to_family.get(major_gid)
        
        if major_gid is not None:
            major_to_leaves[major_gid].add(leaf_gid)
        if family_gid is not None:
            family_to_leaves[family_gid].add(leaf_gid)
            if major_gid is not None:
                family_to_majors[family_gid].add(major_gid)
    
    # Convert vocabularies to sets for faster O(1) membership checks and type safety
    family_vocab_set = set(int(g) for g in family_vocab_global)
    major_vocab_set = set(int(g) for g in major_vocab_global)
    leaf_vocab_set = set(int(g) for g in leaf_vocab_global)
    
    # Process each node in scp_set
    for node_gid in scp_set:
        node_gid_int = int(node_gid)  # Ensure integer type
        # Determine node type
        if node_gid_int in family_vocab_set:
            # Family node
            family_set.add(node_gid_int)
            
            # Add all descendants (majors and leaves)
            if node_gid_int in family_to_majors:
                major_set.update(family_to_majors[node_gid_int])
            if node_gid_int in family_to_leaves:
                leaf_set.update(family_to_leaves[node_gid_int])
            
            # Ancestors: none (family is root)
            
        elif node_gid_int in major_vocab_set:
            # Major node
            major_set.add(node_gid_int)
            
            # Add ancestor (family)
            family_gid = major_to_family.get(node_gid_int)
            if family_gid is not None:
                family_set.add(int(family_gid))
            
            # Add descendants (leaves)
            if node_gid_int in major_to_leaves:
                leaf_set.update(major_to_leaves[node_gid_int])
            
        elif node_gid_int in leaf_vocab_set:
            # Leaf node
            leaf_set.add(node_gid_int)
            
            # Add ancestors (major and family)
            major_gid = leaf_to_major.get(node_gid_int)
            if major_gid is not None:
                major_set.add(int(major_gid))
            
            family_gid = leaf_to_family.get(node_gid_int)
            if family_gid is None and major_gid is not None:
                family_gid = major_to_family.get(major_gid)
            if family_gid is not None:
                family_set.add(int(family_gid))
            
            # Descendants: none (leaf is terminal)
        else:
            # Node not found in any vocabulary - this shouldn't happen but log it
            # This could happen if there's a mismatch between solver output and vocabularies
            pass
    
    return {
        "family": family_set,
        "major": major_set,
        "leaf": leaf_set,
    }
