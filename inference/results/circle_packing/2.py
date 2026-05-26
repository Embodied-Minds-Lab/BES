# EVOLVE-BLOCK-START
"""Monolithic SLSQP-based circle packing for n=26 with LP polishing and robust fallbacks."""

import numpy as np

# Optional SciPy imports
try:
    from scipy.optimize import linprog as _scipy_linprog
except Exception:
    _scipy_linprog = None

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None


def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square that attempts to
    maximize the sum of radii using a joint nonlinear program with SLSQP.

    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates in [0,1]
        radii: np.array of shape (26) with radius of each circle
    """
    np.random.seed(7)
    n = 26
    # Use a slightly larger margin to preserve maneuverability near borders
    margin = 0.015
    eps = margin

    # If SLSQP is available, run monolithic optimization from several seeds.
    if _scipy_minimize is not None:
        seeds = _generate_seed_layouts(n=n, eps=eps)
        best_centers = None
        best_radii = None
        best_sum = -np.inf

        # Multi-start with small jitters to avoid symmetry traps
        for seed in seeds:
            for jitter_scale in (0.0, 0.004, 0.008, 0.012):
                centers0 = _apply_jitter(seed, jitter_scale, eps=eps)
                # Solve joint NLP
                c_opt, r_opt = _solve_joint_nlp(centers0, eps=eps)
                # LP polishing of radii given centers
                r_lp = compute_max_radii(c_opt)
                s = float(np.sum(r_lp))

                # LP-guided force refinement to further improve centers (fixed margin)
                c_ref = _refine_centers_lp_guided(
                    np.clip(c_opt, eps, 1.0 - eps),
                    steps=14,
                    margin=margin
                )
                r_ref = compute_max_radii(c_ref)
                s_ref = float(np.sum(r_ref))

                if s_ref >= s - 1e-10:
                    c_use, r_use, s_use = c_ref, r_ref, s_ref
                else:
                    c_use, r_use, s_use = c_opt, r_lp, s

                if s_use > best_sum:
                    best_sum = s_use
                    best_centers = c_use
                    best_radii = r_use

        # Final annealed LP-guided refinement with tighter force sharpness and margins
        best_centers = _refine_centers_lp_guided(
            np.clip(best_centers, eps, 1.0 - eps),
            steps=16,
            margin=margin,
            margin_end=0.008,
            sigma_pair_start=0.03,
            sigma_pair_end=0.015,
            sigma_wall_start=0.02,
            sigma_wall_end=0.012
        )
        # Tighten LP tolerances and greedy micro-polish
        best_radii = compute_max_radii(best_centers, pair_eps=2e-7)
        best_centers, best_radii = _local_refine_centers(
            best_centers, best_radii, steps=2, step_size=0.01, eps=eps, pair_eps=2e-7
        )
        return best_centers, best_radii

    # Fallback: enumerate candidate templates, solve LP radii, then LP-guided refine
    candidate_centers = _generate_seed_layouts(n=n, eps=eps)
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    for centers in candidate_centers:
        centers = np.clip(centers, eps, 1.0 - eps)
        r = compute_max_radii(centers)
        s = float(np.sum(r))

        c_ref = _refine_centers_lp_guided(
            centers, steps=14, margin=margin, margin_end=0.008,
            sigma_pair_start=0.03, sigma_pair_end=0.015,
            sigma_wall_start=0.02, sigma_wall_end=0.012
        )
        r_ref = compute_max_radii(c_ref)
        s_ref = float(np.sum(r_ref))

        if s_ref >= s - 1e-10:
            s, centers, r = s_ref, c_ref, r_ref

        if s > best_sum:
            best_sum = s
            best_centers = centers
            best_radii = r

    # Tiny local coordinate poke as a final polish with tighter LP
    best_centers, best_radii = _local_refine_centers(
        best_centers, best_radii, steps=2, step_size=0.01, eps=eps, pair_eps=2e-7
    )
    # Ensure final radii reflect tightened LP
    best_radii = compute_max_radii(best_centers, pair_eps=2e-7)
    return best_centers, best_radii


def compute_max_radii(centers, pair_eps=1e-6):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        pair_eps: small separation to avoid exact tangency degeneracy

    Returns:
        np.array of shape (n) with radius of each circle
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    # Border-limited upper bounds
    b = _border_radius_bounds(centers)
    # Try LP maximize sum r subject to:
    # - r_i >= 0
    # - r_i <= b_i
    # - r_i + r_j <= ||c_i - c_j|| - pair_eps for all i<j
    r = None
    if _scipy_linprog is not None:
        r = _solve_lp_max_sum_radii(centers, b, pair_eps=pair_eps)
    if r is None:
        # Fallback: projection-based feasible reduction from border caps
        r = _feasible_shrink_solver(centers, b)
    # Nonnegative safety clamp
    r = np.maximum(r, 0.0)
    return r


# ======================= Joint NLP Solver (SLSQP) =======================

def _solve_joint_nlp(centers0, eps=1e-3, pair_eps=1e-7, maxiter=400):
    """
    Solve the joint nonlinear program:
        maximize   sum r_i
        variables  x_i, y_i, r_i
        subject to:
            x_i - r_i >= 0, 1 - x_i - r_i >= 0,
            y_i - r_i >= 0, 1 - y_i - r_i >= 0,
            ||c_i - c_j|| - (r_i + r_j) >= 0 for all i<j
    Seeds with centers0 and initial radii from LP (or fallback).

    Returns:
        centers_opt, radii_opt
    """
    centers0 = np.asarray(centers0, dtype=float)
    n = centers0.shape[0]
    # Initialize radii by LP (or fallback)
    r0 = compute_max_radii(centers0)

    # Pack variables as z = [x1, y1, ..., xn, yn, r1, ..., rn]
    z0 = np.concatenate([centers0.reshape(-1), r0])

    # Bounds: x,y in [eps,1-eps]; r in [0, 0.5]
    bounds = [(eps, 1.0 - eps)] * (2 * n) + [(0.0, 0.5)] * n

    def obj(z):
        r = z[2 * n:]
        return -float(np.sum(r))

    def cons_border(z):
        xy = z[: 2 * n].reshape(n, 2)
        r = z[2 * n:]
        x = xy[:, 0]
        y = xy[:, 1]
        c1 = x - r
        c2 = 1.0 - x - r
        c3 = y - r
        c4 = 1.0 - y - r
        return np.concatenate([c1, c2, c3, c4])

    def cons_pairs(z):
        xy = z[: 2 * n].reshape(n, 2)
        r = z[2 * n:]
        diff = xy[:, None, :] - xy[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        d = np.sqrt(np.maximum(d2, 0.0))
        # Upper triangle pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(d[i, j] - (r[i] + r[j]) - pair_eps)
        return np.array(pairs, dtype=float)

    constraints = [
        {'type': 'ineq', 'fun': cons_border},
        {'type': 'ineq', 'fun': cons_pairs},
    ]

    res = _scipy_minimize(
        fun=obj,
        x0=z0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': False}
    )

    z = res.x if res is not None and res.x is not None else z0
    xy = z[: 2 * n].reshape(n, 2)
    # Post-project centers into box and recompute optimal radii via LP
    xy = np.clip(xy, eps, 1.0 - eps)
    r = compute_max_radii(xy)
    return xy, r


# ======================= LP radius maximization =======================

def _solve_lp_max_sum_radii(centers, b, pair_eps=1e-6):
    n = centers.shape[0]
    d = _pairwise_dists(centers)

    # Objective: maximize sum r -> minimize -sum r
    c = -np.ones(n, dtype=float)

    # Constraints: r_i + r_j <= d_ij - pair_eps, for i<j; and r_i <= b_i
    rows = []
    rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            dij = d[i, j]
            row = np.zeros(n, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            rhs.append(max(dij - pair_eps, 0.0))
            rows.append(row)
    for i in range(n):
        row = np.zeros(n, dtype=float)
        row[i] = 1.0
        rows.append(row)
        rhs.append(float(b[i]))

    A_ub = np.vstack(rows) if rows else None
    b_ub = np.asarray(rhs, dtype=float) if rhs else None
    bounds = [(0.0, None) for _ in range(n)]

    try:
        res = _scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            return np.asarray(res.x, dtype=float)
    except Exception:
        pass
    return None


def _feasible_shrink_solver(centers, b, max_iter=2000, tol=1e-10):
    # Start from border caps and shrink to resolve pairwise overlaps
    n = centers.shape[0]
    r = np.array(b, dtype=float)
    d = _pairwise_dists(centers)
    for _ in range(max_iter):
        total_violation = 0.0
        # Enforce r_i <= b_i
        r = np.minimum(r, b)
        # Pairwise projection: for any violation, reduce both radii equally
        for i in range(n):
            for j in range(i + 1, n):
                dij = d[i, j]
                if dij <= 0.0:
                    if r[i] + r[j] > 0.0:
                        total_violation += r[i] + r[j]
                        r[i] = 0.0
                        r[j] = 0.0
                else:
                    s = r[i] + r[j] - dij
                    if s > 0.0:
                        dec = 0.5 * s
                        r[i] -= dec
                        r[j] -= dec
                        total_violation += s
                        # Keep non-negative
                        if r[i] < 0.0:
                            r[j] += r[i]  # transfer the negative overshoot
                            r[i] = 0.0
                        if r[j] < 0.0:
                            r[i] += r[j]
                            r[j] = 0.0
        if total_violation <= tol:
            break
    return np.maximum(r, 0.0)


# ======================= Templates (Seeds) =======================

def _generate_seed_layouts(n=26, eps=1e-3):
    """
    Return a modest beam of diverse seed layouts (each of shape (n,2)).
    Includes hybrid rings with phase shifts, edge-belts + inner grids,
    and trimmed/skewed hexagonal lattices.
    """
    seeds = []
    # Legacy seeds
    seeds.append(_seed_hybrid_ring_A(eps=eps))
    seeds.append(_seed_hex_lattice_B(eps=eps))
    seeds.append(_seed_cross_edges_C(eps=eps))
    # Additional structurally diverse candidates
    seeds.extend(_generate_dual_ring_candidates(eps=eps))
    seeds.extend(_generate_edge_belt_candidates(eps=eps))
    seeds.extend(_generate_hex_lattice_candidates(eps=eps))
    # Ensure all seeds are clipped and correct shape
    out = []
    for s in seeds:
        arr = np.asarray(s, dtype=float).reshape(n, 2)
        arr = np.clip(arr, eps, 1.0 - eps)
        out.append(arr)
    return out


def _seed_hybrid_ring_A(eps=1e-3):
    """
    Hybrid: 1 center + 6 inner ring + 12 outer ring + 7 edge assists = 26
    """
    cx, cy = 0.5, 0.5
    R1 = 0.18
    R2 = 0.34
    em = 0.12
    pts = []
    # Center
    pts.append([cx, cy])
    # Inner ring (6)
    for k in range(6):
        ang = 2 * np.pi * k / 6.0
        pts.append([cx + R1 * np.cos(ang), cy + R1 * np.sin(ang)])
    # Outer ring (12), phase-shifted
    for k in range(12):
        ang = 2 * np.pi * k / 12.0 + np.pi / 12.0
        pts.append([cx + R2 * np.cos(ang), cy + R2 * np.sin(ang)])
    # Edge/corner assists (7)
    edge_pts = [
        [cx, em], [cx, 1.0 - em],
        [em, cy], [1.0 - em, cy],
        [em, em], [1.0 - em, 1.0 - em], [em, 1.0 - em]
    ]
    pts.extend(edge_pts)
    arr = np.array(pts, dtype=float)
    arr = np.clip(arr, eps, 1.0 - eps)
    return arr


def _seed_hex_lattice_B(eps=1e-3):
    """
    Hexagonal lattice trimmed to 26 points: rows [6,5,5,5,5].
    """
    rows = [6, 5, 5, 5, 5]  # total 26
    s = 0.18
    return _build_centered_hex_lattice(rows, s, skew=(0.012, -0.01), eps=eps)


def _seed_cross_edges_C(eps=1e-3):
    """
    Central cross + diagonal ring + edges to reach 26.
    """
    cx, cy = 0.5, 0.5
    r1 = 0.20
    r2 = 0.31
    edge_m = 0.12
    pts = []
    # Center
    pts.append([cx, cy])
    # Cross (4)
    cross_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in cross_dirs:
        pts.append([cx + r1 * dx, cy + r1 * dy])
    # Diagonal ring (8)
    for k in range(8):
        ang = np.pi / 4 * k + np.pi / 8.0
        pts.append([cx + r2 * np.cos(ang), cy + r2 * np.sin(ang)])
    # 13 edge points
    edge_pts = []
    xs = np.linspace(edge_m, 1.0 - edge_m, 5)
    for x in xs:
        edge_pts.append([x, edge_m])       # bottom
        edge_pts.append([x, 1.0 - edge_m]) # top
    edge_pts.append([edge_m, 0.5])
    edge_pts.append([1.0 - edge_m, 0.5])
    edge_pts.append([edge_m, edge_m])
    pts.extend(edge_pts[:13])
    arr = np.array(pts, dtype=float)
    arr = np.clip(arr, eps, 1.0 - eps)
    if arr.shape[0] > 26:
        arr = arr[:26]
    return arr


def _generate_dual_ring_candidates(eps=1e-3):
    """
    Dual/trimmed rings with small phase shifts and a single pocket.
    Each candidate: 1 center + 8 inner + 16 outer + 1 pocket = 26
    """
    cx = cy = 0.5
    cands = []
    extras = [(0.14, 0.14), (0.14, 0.86), (0.86, 0.14), (0.86, 0.86)]
    for R1 in (0.18, 0.22):
        for R2 in (0.34, 0.38):
            for phase in (0.0, np.pi / 16.0):
                for extra in extras[:2]:  # keep pool modest
                    centers = np.zeros((26, 2), dtype=float)
                    centers[0] = [cx, cy]
                    # Inner ring (8)
                    for i in range(8):
                        ang = 2 * np.pi * i / 8.0
                        centers[1 + i] = [cx + R1 * np.cos(ang), cy + R1 * np.sin(ang)]
                    # Outer ring (16)
                    for i in range(16):
                        ang = 2 * np.pi * i / 16.0 + phase
                        centers[9 + i] = [cx + R2 * np.cos(ang), cy + R2 * np.sin(ang)]
                    # Pocket
                    centers[25] = extra
                    cands.append(np.clip(centers, eps, 1.0 - eps))
    return cands


def _generate_edge_belt_candidates(eps=1e-3):
    """
    Edge-belt (3 per edge) + inner 3x3 grid + center + 4 corner pockets = 26
    """
    cands = []
    for e in (0.12, 0.16):
        pts = []
        # 12 edge points
        xs3 = [0.2, 0.5, 0.8]
        ys3 = [0.2, 0.5, 0.8]
        for x in xs3:
            pts.append((x, e))
            pts.append((x, 1.0 - e))
        for y in ys3:
            pts.append((e, y))
            pts.append((1.0 - e, y))
        # 9-point inner grid
        grid = [0.35, 0.5, 0.65]
        for x in grid:
            for y in grid:
                pts.append((x, y))
        # Center
        pts.append((0.5, 0.5))
        # Four corner-adjacent pockets
        d = 0.03
        pts.append((e + d, e + d))
        pts.append((e + d, 1.0 - e - d))
        pts.append((1.0 - e - d, e + d))
        pts.append((1.0 - e - d, 1.0 - e - d))
        centers = np.array(pts[:26], dtype=float)
        centers = np.clip(centers, eps, 1.0 - eps)
        cands.append(centers)
    return cands


def _generate_hex_lattice_candidates(eps=1e-3):
    """
    Additional skewed hexagonal lattices with slight parameter variations.
    """
    cands = []
    rows = [6, 5, 5, 5, 5]
    for s in (0.18, 0.19, 0.20):
        for skew in ((0.012, -0.01), (0.0, 0.0)):
            cands.append(_build_centered_hex_lattice(rows, s, skew=skew, eps=eps))
    return cands


def _build_centered_hex_lattice(rows, s, skew=None, eps=1e-3):
    """
    rows: list of ints = points per row
    s: horizontal step
    skew: optional tuple (sx, sy) to slightly skew positions
    """
    sy = np.sqrt(3.0) * 0.5 * s
    nrows = len(rows)
    # Center rows vertically around 0.5
    total_h = (nrows - 1) * sy
    y0 = 0.5 - 0.5 * total_h
    pts = []
    for r_idx, count in enumerate(rows):
        y = y0 + r_idx * sy
        # Center the row horizontally
        width = (count - 1) * s
        x_start = 0.5 - 0.5 * width
        # Alternate offset for hex staggering
        offset = 0.0 if (r_idx % 2 == 0) else 0.5 * s
        for c_idx in range(count):
            x = x_start + c_idx * s + offset
            pts.append([x, y])
    pts = np.array(pts, dtype=float)
    # Apply optional small skew to break symmetry
    if skew is not None:
        sx, sy_skew = skew
        pts[:, 0] += sx * (pts[:, 1] - 0.5)
        pts[:, 1] += sy_skew * (pts[:, 0] - 0.5)
    # Fit into [eps, 1-eps] by uniform scaling and centering if necessary
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-12)
    scale = min((1.0 - 2 * eps) / span[0], (1.0 - 2 * eps) / span[1], 1.0)
    center = 0.5 * (min_xy + max_xy)
    pts = (pts - center) * scale + np.array([0.5, 0.5])
    pts = np.clip(pts, eps, 1.0 - eps)
    return pts


def _apply_jitter(centers, scale, eps=1e-3):
    if scale <= 0:
        return np.clip(np.asarray(centers, dtype=float), eps, 1.0 - eps)
    jitter = (np.random.rand(*centers.shape) - 0.5) * 2.0 * scale
    out = np.asarray(centers, dtype=float) + jitter
    return np.clip(out, eps, 1.0 - eps)


# ======================= Utilities and Local Refinement =======================

def _border_radius_bounds(centers):
    x = centers[:, 0]
    y = centers[:, 1]
    return np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))


def _pairwise_dists(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(np.maximum(d2, 0.0))
    return d


def _local_refine_centers(centers, radii, steps=2, step_size=0.01, eps=1e-3, pair_eps=1e-6):
    """
    Small greedy coordinate refinement: tries tiny moves to increase LP objective.
    Keeps structure stable and runtime modest.
    """
    if centers is None or radii is None:
        return centers, radii
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = float(np.sum(best_radii))
    n = centers.shape[0]
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)

    for _ in range(steps):
        improved = False
        for i in range(n):
            base = best_centers[i].copy()
            for d in dirs:
                cand = best_centers.copy()
                cand[i] = np.clip(base + step_size * d, eps, 1.0 - eps)
                r = compute_max_radii(cand, pair_eps=pair_eps)
                s = float(np.sum(r))
                if s > best_sum + 1e-9:
                    best_sum = s
                    best_centers = cand
                    best_radii = r
                    improved = True
                    base = cand[i].copy()
        if not improved:
            step_size *= 0.5
    return best_centers, best_radii


def _refine_centers_lp_guided(centers, steps=12, margin=0.015, margin_end=None,
                              sigma_pair_start=0.025, sigma_pair_end=None,
                              sigma_wall_start=0.015, sigma_wall_end=None,
                              backtracks=5, improve_tol=1e-5):
    """
    Alternate between LP (to get radii) and a small nudge on centers computed
    from constraint 'tensions' of near-active constraints. Uses backtracking
    and decay to ensure monotonic non-decrease in LP objective.

    Annealed scheduling:
    - margin decays from margin to margin_end (defaults to margin if None)
    - sigma_pair decays from sigma_pair_start to sigma_pair_end (defaults keep)
    - sigma_wall decays from sigma_wall_start to sigma_wall_end (defaults keep)
    """
    centers = np.clip(np.asarray(centers, dtype=float), margin, 1.0 - margin)
    n = centers.shape[0]
    alpha = 0.05
    decay = 0.88
    min_alpha = 0.004

    if margin_end is None:
        margin_end = margin
    if sigma_pair_end is None:
        sigma_pair_end = sigma_pair_start
    if sigma_wall_end is None:
        sigma_wall_end = sigma_wall_start

    best_centers = centers.copy()
    best_radii = compute_max_radii(best_centers)
    best_score = float(np.sum(best_radii))

    no_improve_streak = 0
    for it in range(steps):
        # Anneal parameters
        frac = it / max(steps - 1, 1)
        cur_margin = (1.0 - frac) * margin + frac * margin_end
        sigma_pair = (1.0 - frac) * sigma_pair_start + frac * sigma_pair_end
        sigma_wall = (1.0 - frac) * sigma_wall_start + frac * sigma_wall_end

        centers = np.clip(centers, cur_margin, 1.0 - cur_margin)

        radii = compute_max_radii(centers)
        score = float(np.sum(radii))

        F = _compute_forces_from_constraints(centers, radii, sigma_pair=sigma_pair, sigma_wall=sigma_wall)

        # Propose an update with backtracking line search
        accepted = False
        trial_alpha = alpha
        for _ in range(backtracks):
            new_centers = centers + trial_alpha * F
            new_centers = np.clip(new_centers, cur_margin, 1.0 - cur_margin)
            new_radii = compute_max_radii(new_centers)
            new_score = float(np.sum(new_radii))
            if new_score >= score - 1e-10:
                centers = new_centers
                radii = new_radii
                score = new_score
                accepted = True
                break
            trial_alpha *= 0.5

        if accepted and score > best_score + improve_tol:
            best_score = score
            best_centers = centers.copy()
            best_radii = radii.copy()
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        alpha = max(min_alpha, decay * alpha)
        if no_improve_streak >= 6:
            break

    return best_centers


def _compute_forces_from_constraints(centers, radii, sigma_pair=0.025, sigma_wall=0.015):
    """
    Compute forces that push circle centers to increase LP objective:
    - For near-active pair constraints r_i + r_j <= d_ij, push i and j apart.
    - For near-active wall bounds r_i <= b_i, push away from the closest wall.

    Uses vectorized accumulation for stability and performance.
    """
    n = centers.shape[0]
    F = np.zeros_like(centers)

    # Pairwise distances and unit vectors
    diff = centers[:, None, :] - centers[None, :, :]
    D = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    np.fill_diagonal(D, np.inf)
    U = np.zeros_like(diff)
    mask = np.isfinite(D) & (D > 1e-12)
    U[mask] = diff[mask] / D[mask][..., None]

    # Pair forces: higher weight for smaller slack
    slack = D - (radii[:, None] + radii[None, :])
    slack_pos = np.maximum(slack, 0.0)
    W = np.exp(-slack_pos / max(sigma_pair, 1e-9))
    np.fill_diagonal(W, 0.0)
    # Skew-symmetrize weights to accumulate +/- on pairs (i<j)
    W_upper = np.triu(W, 1)
    S = W_upper - W_upper.T
    # Accumulate forces
    F += np.einsum('ij,ijx->ix', S, U, optimize=True)

    # Wall forces: push away from the nearest wall when r_i ~ b_i
    x = centers[:, 0]
    y = centers[:, 1]
    b_left = x
    b_bottom = y
    b_right = 1.0 - x
    b_top = 1.0 - y
    b_all = np.stack([b_left, b_bottom, b_right, b_top], axis=1)
    nearest_idx = np.argmin(b_all, axis=1)
    nearest_b = b_all[np.arange(n), nearest_idx]

    wall_normals = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    wall_slack = nearest_b - radii
    wall_w = np.exp(-np.maximum(0.0, wall_slack) / max(sigma_wall, 1e-9))
    F += wall_w[:, None] * wall_normals[nearest_idx]

    # Normalize forces per point to avoid extreme updates
    norms = np.linalg.norm(F, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        F = F / (1e-9 + max_norm)

    return F


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii