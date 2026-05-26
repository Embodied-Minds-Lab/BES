# EVOLVE-BLOCK-START
import numpy as np

# --- Module-level constants and caches (deterministic, reused across calls) ---
_N = 13

# Precompute all C(13,3) index triples once at import-time, contiguous int64
_I_list = []
for _i in range(_N - 2):
    for _j in range(_i + 1, _N - 1):
        for _k in range(_j + 1, _N):
            _I_list.append((_i, _j, _k))
_TRIPLES = np.ascontiguousarray(np.array(_I_list, dtype=np.int64))
I = np.ascontiguousarray(_TRIPLES[:, 0])
J = np.ascontiguousarray(_TRIPLES[:, 1])
K = np.ascontiguousarray(_TRIPLES[:, 2])

# Precompute two 8-direction unit bases: axis/diagonals and rotated by +11.25°
_base_dirs = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ],
    dtype=np.float64,
)
_base_dirs /= np.linalg.norm(_base_dirs, axis=1, keepdims=True)
# Rotation by +11.25° (pi/16)
_theta = np.pi / 16.0
_c, _s = np.cos(_theta), np.sin(_theta)
_R = np.array([[const for const in row] for row in ((_c, -_s), (_s, _c))], dtype=np.float64)
_rot_dirs = _base_dirs @ _R.T
_rot_dirs /= np.linalg.norm(_rot_dirs, axis=1, keepdims=True)
DIRS_A = np.ascontiguousarray(_base_dirs, dtype=np.float64)
DIRS_B = np.ascontiguousarray(_rot_dirs, dtype=np.float64)


# --- Geometry helpers: convex hull (Andrew's monotone chain) + area (shoelace) ---
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.shape[0] <= 1:
        return pts.copy()
    # Lexicographic sort by x then y (deterministic)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]
    # Deduplicate consecutive near-equal points
    uniq = [P[0]]
    for p in P[1:]:
        if not np.allclose(p, uniq[-1], atol=1e-14, rtol=0.0):
            uniq.append(p)
    P = np.ascontiguousarray(np.array(uniq, dtype=np.float64))
    if P.shape[0] <= 1:
        return P.copy()
    # Lower chain
    lower = []
    for p in P:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(tuple(p))
    # Upper chain
    upper = []
    for p in P[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def _polygon_area(poly: np.ndarray) -> float:
    if poly is None or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --- Vectorized min-triangle evaluation, reusing global I,J,K (expects float64 C-contiguous P) ---
def _min_triangle_area(P: np.ndarray, I_=I, J_=J, K_=K) -> float:
    Pi = P[I_]
    Pj = P[J_]
    Pk = P[K_]
    v1x = Pj[:, 0] - Pi[:, 0]
    v1y = Pj[:, 1] - Pi[:, 1]
    v2x = Pk[:, 0] - Pi[:, 0]
    v2y = Pk[:, 1] - Pi[:, 1]
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    return float(areas.min(initial=np.inf))


def _min_area_with_argmin(P: np.ndarray, I_=I, J_=J, K_=K):
    Pi = P[I_]
    Pj = P[J_]
    Pk = P[K_]
    v1x = Pj[:, 0] - Pi[:, 0]
    v1y = Pj[:, 1] - Pi[:, 1]
    v2x = Pk[:, 0] - Pi[:, 0]
    v2y = Pk[:, 1] - Pi[:, 1]
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    idx = int(np.argmin(areas))
    return float(areas[idx]), (int(I_[idx]), int(J_[idx]), int(K_[idx]))


# --- Normalized objective: (min triangle area) / (convex hull area) with degeneracy check ---
def _objective_values(P: np.ndarray):
    # Returns (normalized_objective, hull_area, min_triangle_area) or (-inf, hull_area, min) on invalid
    if not np.all(np.isfinite(P)):
        return float("-inf"), 0.0, 0.0
    hull = _convex_hull(P)
    hull_area = _polygon_area(hull)
    if not np.isfinite(hull_area) or hull_area <= 1e-14:
        return float("-inf"), hull_area, 0.0
    mta = _min_triangle_area(P)
    if not np.isfinite(mta) or mta <= 0.0:
        return float("-inf"), hull_area, mta
    return mta / hull_area, hull_area, mta


def _better(norm_val: float, hull_area: float, best_norm: float, best_hull: float, tol_norm: float, eps_hull: float) -> bool:
    if norm_val > best_norm + tol_norm:
        return True
    if abs(norm_val - best_norm) <= tol_norm and hull_area > best_hull + eps_hull:
        return True
    return False


# --- Dual deterministic seeds with slight de-symmetrization, clipped to square ---
def _seed_two_ring_5_8() -> np.ndarray:
    # Outer ring: 8 points; Inner ring: 5 points; small deterministic radial modulations
    r_out, r_in = 0.45, 0.27
    phi_out, phi_in = -0.311, 0.137
    k_out = np.arange(8, dtype=np.float64)
    k_in = np.arange(5, dtype=np.float64)
    ang_out = phi_out + (2.0 * np.pi / 8.0) * k_out
    ang_in = phi_in + (2.0 * np.pi / 5.0) * k_in
    r_out_k = r_out + 0.015 * r_out * np.cos(ang_out * np.sqrt(2.0))
    r_in_k = r_in - 0.012 * r_in * np.sin(ang_in * np.sqrt(3.0))
    outer = np.stack([r_out_k * np.cos(ang_out), r_out_k * np.sin(ang_out)], axis=1)
    inner = np.stack([r_in_k * np.cos(ang_in), r_in_k * np.sin(ang_in)], axis=1)
    P = np.vstack([outer, inner])  # 8 + 5 = 13
    return np.ascontiguousarray(np.clip(P, -0.5, 0.5), dtype=np.float64)


def _seed_6_6_1() -> np.ndarray:
    # 6 outer + 6 inner (30° offset) + 1 interior off axes; small deterministic modulation
    r_out, r_in = 0.46, 0.30
    phi_out = -0.08
    phi_in = phi_out + np.pi / 6.0
    k = np.arange(6, dtype=np.float64)
    a_out = phi_out + (2.0 * np.pi / 6.0) * k
    a_in = phi_in + (2.0 * np.pi / 6.0) * k
    r_ok = r_out * (1.0 + 0.020 * np.sin(2.0 * a_out + 0.31))
    r_ik = r_in * (1.0 + 0.018 * np.cos(3.0 * a_in - 0.19))
    outer = np.stack([r_ok * np.cos(a_out), r_ok * np.sin(a_out)], axis=1)
    inner = np.stack([r_ik * np.cos(a_in), r_ik * np.sin(a_in)], axis=1)
    center = np.array([[0.043, -0.037]], dtype=np.float64)
    P = np.vstack([outer, inner, center])  # 6 + 6 + 1 = 13
    return np.ascontiguousarray(np.clip(P, -0.5, 0.5), dtype=np.float64)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic constructor for 13 points inside the fixed square [-0.5, 0.5]^2.
    Backbone:
      - Import-time cached C(13,3) triples (vectorized, contiguous int64).
      - Two deterministic seeds (5/8 two-ring and 6/6/1), clipped to the square.
      - Greedy 8-direction pattern search with alternating rotated bases per sweep.
      - Short deterministic micro line search per direction with in-place overwrite/restore.
      - Hull-aware tie-break when normalized objective ties within tolerance.
      - Objective: smallest triangle area normalized by convex hull area (Andrew + shoelace).
      - No randomness; no post-rescaling (remain inside the fixed square).
    """
    n = _N
    assert n == 13

    # Select better deterministic seed
    seedA = _seed_two_ring_5_8()
    seedB = _seed_6_6_1()
    a_norm, a_hull, _ = _objective_values(seedA)
    b_norm, b_hull, _ = _objective_values(seedB)
    # Prefer higher normalized objective; if equal within tol, prefer larger hull area
    tol_seed = 1e-12
    eps_hull = 1e-14
    if _better(b_norm, b_hull, a_norm, a_hull, tol_seed, eps_hull):
        P = np.ascontiguousarray(seedB, dtype=np.float64)
    else:
        P = np.ascontiguousarray(seedA, dtype=np.float64)

    # Optimization setup
    step_schedule = [0.12, 0.06, 0.03, 0.015, 0.0075, 0.0035]
    tol_improve = 1e-12
    eps_hull_tb = 1e-14  # tie-break epsilon
    max_extend_steps = 12  # bounded greedy forward extension after commit

    # Prebind hot references
    min_area_with_argmin = _min_area_with_argmin
    dirs_a = DIRS_A
    dirs_b = DIRS_B
    clip = np.clip
    allclose = np.allclose

    # Current best values
    best_norm, best_hull, _ = _objective_values(P)
    if not np.isfinite(best_norm):
        # Deterministic safety: mild shrink towards center
        P *= 0.98
        P = np.ascontiguousarray(np.clip(P, -0.5, 0.5), dtype=np.float64)
        best_norm, best_hull, _ = _objective_values(P)

    # Local search with alternating direction bases and micro line search
    for step in step_schedule:
        use_rotated = False  # toggle each full sweep
        while True:
            dirs = dirs_b if use_rotated else dirs_a
            use_rotated = not use_rotated

            # Worst triangle prioritization; build unique-stable order
            _, worst = min_area_with_argmin(P)
            order = list(dict.fromkeys(list(worst) + list(range(n))))
            improved_any = False

            for i in order:
                base = P[i].copy()
                local_best_norm = best_norm
                local_best_hull = best_hull
                local_best_pt = None
                local_best_dir = None

                for d in dirs:
                    # Gate 1: try 1.0 * step
                    cand1 = clip(base + step * d, -0.5, 0.5)
                    if allclose(cand1, base, atol=1e-15, rtol=0.0):
                        continue
                    P[i] = cand1
                    v1, h1, _ = _objective_values(P)

                    if _better(v1, h1, local_best_norm, local_best_hull, tol_improve, eps_hull_tb):
                        # Gate 2: try 2.0 * step only if 1.0 improved
                        cand2 = clip(base + (2.0 * step) * d, -0.5, 0.5)
                        if not allclose(cand2, cand1, atol=1e-15, rtol=0.0):
                            P[i] = cand2
                            v2, h2, _ = _objective_values(P)
                        else:
                            v2, h2 = -np.inf, h1

                        # Choose best among 1x and 2x
                        if _better(v2, h2, v1, h1, tol_improve, eps_hull_tb):
                            cand_best = cand2
                            v_best, h_best = v2, h2
                        else:
                            cand_best = cand1
                            v_best, h_best = v1, h1

                        if _better(v_best, h_best, local_best_norm, local_best_hull, tol_improve, eps_hull_tb):
                            local_best_norm = v_best
                            local_best_hull = h_best
                            local_best_pt = cand_best
                            local_best_dir = d

                    else:
                        # Gate 3: try 0.5 * step if 1.0 failed
                        cand05 = clip(base + (0.5 * step) * d, -0.5, 0.5)
                        if not allclose(cand05, base, atol=1e-15, rtol=0.0):
                            P[i] = cand05
                            v05, h05, _ = _objective_values(P)
                            if _better(v05, h05, local_best_norm, local_best_hull, tol_improve, eps_hull_tb):
                                local_best_norm = v05
                                local_best_hull = h05
                                local_best_pt = cand05
                                local_best_dir = d

                    # Restore before next direction probe
                    P[i] = base

                # Commit the best local move for this point (hull-aware tie-break)
                if local_best_pt is not None and _better(local_best_norm, local_best_hull, best_norm, best_hull, tol_improve, eps_hull_tb):
                    P[i] = local_best_pt
                    best_norm, best_hull = local_best_norm, local_best_hull
                    improved_any = True

                    # Greedy forward extension along the same direction at the same step
                    if local_best_dir is not None:
                        curr = P[i].copy()
                        for _ in range(max_extend_steps):
                            nxt = clip(curr + step * local_best_dir, -0.5, 0.5)
                            if allclose(nxt, curr, atol=1e-15, rtol=0.0):
                                break
                            P[i] = nxt
                            v_next, h_next, _ = _objective_values(P)
                            if _better(v_next, h_next, best_norm, best_hull, tol_improve, eps_hull_tb):
                                curr = nxt
                                best_norm, best_hull = v_next, h_next
                            else:
                                # Revert and stop extension
                                P[i] = curr
                                break

            if not improved_any:
                # Full sweep stall at this step size
                break

    # Final clip, type, and shape safety (remain inside fixed square; no post-rescale)
    P = np.ascontiguousarray(np.clip(P, -0.5, 0.5), dtype=np.float64)
    if P.shape != (13, 2) or not np.all(np.isfinite(P)):
        # Deterministic safe fallback to better seed
        P = seedB if _better(b_norm, b_hull, a_norm, a_hull, tol_seed, eps_hull) else seedA
        P = np.ascontiguousarray(np.clip(P, -0.5, 0.5), dtype=np.float64)

    return P


# EVOLVE-BLOCK-END