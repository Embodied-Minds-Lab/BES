# EVOLVE-BLOCK-START
import numpy as np


# ----------------------- Global constants and cached structures -----------------------
PI = np.pi
TWO_PI = 2.0 * np.pi
TWO_PI_OVER_3 = 2.0 * np.pi / 3.0

# Penalty thresholds (tie-breakers only)
PAIR_TAU = 0.035    # pairwise distance min threshold
PHASE_TAU = 0.065   # phase separation modulo 2π/3
SOFT_GAP = 0.085    # soft preference for ring gaps (slightly stronger)
PEN_W = 0.01        # penalty weight

# Radii mapping bounds
R_MIN = 0.18
R_MAX = 0.98
DR_MIN_GAP = 0.06   # slightly larger hard gap for stability

# Precompute all triangle triplets for n=13 (C(13,3) = 286), int32 for speed
def _precompute_tris_13():
    tris = []
    for i in range(13 - 2):
        for j in range(i + 1, 13 - 1):
            for k in range(j + 1, 13):
                tris.append((i, j, k))
    return np.asarray(tris, dtype=np.int32)


TRIS_13 = _precompute_tris_13()
# Cached triangle metadata: ring ids per vertex (-1 for center), and mask for any-center
TRI_RING_IDS = ((TRIS_13 - 1) // 3).astype(np.int8)  # center (0) -> -1
TRI_HAS_CENTER = np.any(TRIS_13 == 0, axis=1)


# ----------------------- Geometry utilities -----------------------
def _convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Andrew’s monotone chain convex hull, excluding collinear edge points (cross <= 0).
    Returns hull vertices in CCW order as coordinates.
    """
    P = np.asarray(points, dtype=np.float64)
    n = P.shape[0]
    if n <= 1:
        return P.copy()
    idx = np.lexsort((P[:, 1], P[:, 0]))
    S = P[idx]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in S:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in S[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """
    Cross-weighted centroid from shoelace terms. Works for CCW or CW simple polygons.
    """
    if poly.shape[0] < 3:
        return np.array([0.0, 0.0], dtype=np.float64)
    x = poly[:, 0]
    y = poly[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-18 or not np.isfinite(A):
        return np.array([0.0, 0.0], dtype=np.float64)
    cx = np.sum((x + x2) * cross) / (6.0 * A)
    cy = np.sum((y + y2) * cross) / (6.0 * A)
    return np.array([cx, cy], dtype=np.float64)


def _normalize_hull_area(P: np.ndarray):
    """
    Two-pass normalization routine.
      1) Scales P by 1/sqrt(hull_area),
      2) Recenters P by the hull centroid,
      3) Recomputes hull and rescales to enforce final convex_hull_area exactly 1.0.
    Returns (ok: bool, Q: np.ndarray).
    """
    Q = np.asarray(P, dtype=np.float64).copy()
    H = _convex_hull(Q)
    A = _polygon_area(H)
    if not np.isfinite(A) or A <= 0.0:
        return False, Q
    s1 = 1.0 / np.sqrt(A)
    Q *= s1

    H = _convex_hull(Q)
    if H.shape[0] < 3:
        return False, Q
    c = _polygon_centroid(H)
    Q -= c

    H = _convex_hull(Q)
    A2 = _polygon_area(H)
    if not np.isfinite(A2) or A2 <= 0.0:
        return False, Q
    s2 = 1.0 / np.sqrt(A2)
    Q *= s2
    return True, Q


# ----------------------- Triangle areas (specialized for n=13) -----------------------
def _all_triangle_areas_13(P: np.ndarray) -> np.ndarray:
    """
    Vectorized triangle areas for n=13, using precomputed TRIS_13.
    Returns areas for all 286 triples.
    """
    I = TRIS_13[:, 0]
    J = TRIS_13[:, 1]
    K = TRIS_13[:, 2]
    A = P[I]
    B = P[J]
    C = P[K]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    return 0.5 * np.abs(cross)


def _min_triangle_area(P: np.ndarray) -> float:
    return float(np.min(_all_triangle_areas_13(P)))


# ----------------------- 3-fold ring parameterization (8D) -----------------------
def _softplus_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    big = x > 40.0
    small = x < -40.0
    mid = ~(big | small)
    out[big] = x[big]
    out[small] = np.exp(x[small])
    out[mid] = np.log1p(np.exp(x[mid]))
    return out


def _map_v_to_radii(v: np.ndarray,
                    r_min: float = R_MIN, r_max: float = R_MAX,
                    gap: float = DR_MIN_GAP) -> np.ndarray:
    """
    Map unconstrained v in R^4 to strictly increasing radii in [r_min, r_max],
    then project to enforce a hard minimum inter-ring gap.
    """
    v = np.asarray(v, dtype=np.float64)
    inc = _softplus_vec(v) + 1e-4
    cum = np.cumsum(inc)
    if not np.isfinite(cum[-1]) or cum[-1] <= 0.0:
        cum = np.arange(1, 5, dtype=np.float64)
    u = cum / (cum[-1] + 1e-12)
    r = r_min + (r_max - r_min) * u

    # Enforce hard gaps deterministically
    r0_hi = r_max - 3.0 * gap
    r[0] = float(np.clip(r[0], r_min, r0_hi))
    for i in range(1, 4):
        lo = r[i - 1] + gap
        hi = r_max - (3 - i) * gap
        r[i] = float(np.clip(r[i], lo, hi))
        if r[i] <= r[i - 1] + 1e-8:
            r[i] = min(r_max, r[i - 1] + max(gap, 1e-8))
    return r


def _wrap_phi_mod(phi: np.ndarray) -> np.ndarray:
    """
    Wrap per-ring base phases into [0, 2π/3).
    """
    phi = np.asarray(phi, dtype=np.float64)
    return phi % TWO_PI_OVER_3


def _build_points_from_r_phi(r: np.ndarray, phi_mod: np.ndarray) -> np.ndarray:
    """
    Build 13 points: 1 center + 4 rings × 3 points at 120° spacing.
    """
    pts = np.empty((13, 2), dtype=np.float64)
    pts[0] = (0.0, 0.0)
    k = 1
    for s in range(4):
        base = phi_mod[s]
        ang = base + TWO_PI_OVER_3 * np.arange(3, dtype=np.float64)
        cs = np.cos(ang)
        sn = np.sin(ang)
        rs = r[s]
        pts[k:k + 3, 0] = rs * cs
        pts[k:k + 3, 1] = rs * sn
        k += 3
    return pts


# ----------------------- Degeneracy penalties (used only as tie-breakers) -----------------------
def _pairwise_min_dist(P: np.ndarray) -> float:
    D = P[:, None, :] - P[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', D, D)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(np.min(d2)))


def _penalty_structured(P: np.ndarray, r: np.ndarray, phi_mod: np.ndarray) -> float:
    pen = 0.0
    # Pairwise distance penalty
    dmin = _pairwise_min_dist(P)
    if dmin < PAIR_TAU:
        pen += PEN_W * (PAIR_TAU - dmin)
    # Soft ring gaps
    gaps = np.diff(r)
    for g in gaps:
        if g < SOFT_GAP:
            pen += PEN_W * (SOFT_GAP - g)
    # Phase crowding modulo 2π/3 (circular)
    ph = np.sort(phi_mod, kind="mergesort")
    diffs = np.diff(np.concatenate([ph, ph[:1] + TWO_PI_OVER_3]))
    for d in diffs:
        if d < PHASE_TAU:
            pen += PEN_W * (PHASE_TAU - float(d))
    return float(pen)


# ----------------------- Evaluation and comparison -----------------------
class EvalResult:
    __slots__ = ("ok", "P", "m", "pen", "r", "phi_mod")
    def __init__(self, ok: bool, P: np.ndarray, m: float, pen: float, r: np.ndarray, phi_mod: np.ndarray):
        self.ok = ok
        self.P = P
        self.m = float(m)
        self.pen = float(pen)
        self.r = r
        self.phi_mod = phi_mod


def _eval_from_params(v: np.ndarray, phi: np.ndarray) -> EvalResult:
    r = _map_v_to_radii(v)
    ph = _wrap_phi_mod(phi)
    P = _build_points_from_r_phi(r, ph)
    ok, Pn = _normalize_hull_area(P)
    if not ok:
        return EvalResult(False, Pn, -np.inf, np.inf, r, ph)
    m = _min_triangle_area(Pn)
    pen = _penalty_structured(Pn, r, ph)
    return EvalResult(True, Pn, m, pen, r, ph)


def _is_better(new: EvalResult, cur: EvalResult) -> bool:
    """
    Primary comparison on true min-area (strict), break ties with smaller penalty.
    """
    if new.m > cur.m + 1e-12:
        return True
    if abs(new.m - cur.m) <= 1e-12 and new.pen < cur.pen - 1e-15:
        return True
    return False


# ----------------------- K-guided weighted co-participation -----------------------
def _ring_guidance_from_smallest(Pn: np.ndarray, K: int = 96, tau: float = 0.018):
    """
    Weighted K-guidance with co-participation:
      - From K smallest non-center triangles, accumulate weights w = exp(-area/τ)
      - Produce per-ring scores C[s] and a 4x4 co-participation matrix W[a,b]
      - Return C, W, ring_order by descending C (stable), and top_pair = argmax W (lexicographic tie-break)
    """
    areas = _all_triangle_areas_13(Pn)
    Keff = int(K if K > 0 else 0)
    Keff = min(Keff, areas.size)
    if Keff <= 0:
        return np.zeros(4, dtype=np.float64), np.zeros((4, 4), dtype=np.float64), list(np.arange(4)), (0, 1)

    idxK = np.argpartition(areas, Keff - 1)[:Keff]
    mask = ~TRI_HAS_CENTER[idxK]
    if not np.any(mask):
        return np.zeros(4, dtype=np.float64), np.zeros((4, 4), dtype=np.float64), list(np.arange(4)), (0, 1)
    sel = idxK[mask]
    A = areas[sel]
    w = np.exp(-A / float(tau))
    rings = TRI_RING_IDS[sel]  # shape (m,3), entries in {0,1,2,3}

    C = np.zeros(4, dtype=np.float64)
    W = np.zeros((4, 4), dtype=np.float64)

    for t in range(rings.shape[0]):
        a, b, c = int(rings[t, 0]), int(rings[t, 1]), int(rings[t, 2])
        wt = float(w[t])
        C[a] += wt; C[b] += wt; C[c] += wt
        W[a, b] += wt; W[b, a] += wt
        W[a, c] += wt; W[c, a] += wt
        W[b, c] += wt; W[c, b] += wt

    ring_order = list(np.argsort(-C, kind="mergesort"))

    # Argmax W over upper triangle with deterministic lex tie-break
    s0, s1 = 0, 1
    bestw = -1.0
    for i in range(4):
        for j in range(i + 1, 4):
            wij = float(W[i, j])
            if (wij > bestw + 1e-18) or (abs(wij - bestw) <= 1e-18 and (i < s0 or (i == s0 and j < s1))):
                bestw = wij
                s0, s1 = i, j

    return C, W, ring_order, (s0, s1)


# ----------------------- Deterministic pre-refinement -----------------------
def _short_prefine(v: np.ndarray, phi: np.ndarray,
                   step_v: float = 0.14, step_phi: float = 0.27):
    """
    One sweep first-improvement over 8 coordinates.
    Accept only strict increases in true min-area (ignore penalties).
    Returns refined (v, phi).
    """
    v = np.asarray(v, dtype=np.float64).copy()
    phi = np.asarray(phi, dtype=np.float64).copy()
    cur = _eval_from_params(v, phi)
    if not cur.ok:
        v = np.clip(v, -4.0, 4.0)
        phi = _wrap_phi_mod(phi)
        cur = _eval_from_params(v, phi)

    # Visit coordinates in ring order 0..3
    for s in range(4):
        # v coordinate
        for d in (step_v, -step_v):
            v_try = v.copy()
            v_try[s] += d
            nxt = _eval_from_params(v_try, phi)
            if nxt.ok and (nxt.m > cur.m + 1e-12):
                v = v_try
                cur = nxt
                break
        # phi coordinate
        for d in (step_phi, -step_phi):
            phi_try = phi.copy()
            phi_try[s] = (phi_try[s] + d) % TWO_PI_OVER_3
            nxt = _eval_from_params(v, phi_try)
            if nxt.ok and (nxt.m > cur.m + 1e-12):
                phi = phi_try
                cur = nxt
                break
    return v, phi


# ----------------------- Symmetric dv-for-dr with adaptive epsilon and caching -----------------------
def _approx_dv_for_dr_sym(v: np.ndarray, ring: int, dr_target: float) -> float:
    """
    Symmetric finite-difference approximation of dv required to induce dr at ring.
    - Applies constraint-aware scaling of dr_target based on slack to hard bounds.
    - Uses adaptive epsilon per ring; doubles once if derivative is too small.
    - Clamps dv to ±0.5 for robustness.
    """
    r = _map_v_to_radii(v)
    lo_bound = R_MIN + ring * DR_MIN_GAP
    hi_bound = R_MAX - (3 - ring) * DR_MIN_GAP
    lo_slack = max(0.0, float(r[ring] - lo_bound))
    hi_slack = max(0.0, float(hi_bound - r[ring]))
    slack = hi_slack if dr_target > 0.0 else lo_slack
    # Slack-aware scaling; avoid overshoot near bounds
    scale = float(np.clip((slack / 0.12) if slack > 0.0 else 0.0, 0.3, 1.0))
    dr_eff = dr_target * scale

    # Adaptive symmetric epsilon per ring
    eps = float(np.clip(0.05 * (1.0 + 0.5 * ring), 0.03, 0.12))
    v_p = v.copy(); v_m = v.copy()
    v_p[ring] += eps
    v_m[ring] -= eps
    r_p = _map_v_to_radii(v_p)[ring]
    r_m = _map_v_to_radii(v_m)[ring]
    deriv = (r_p - r_m) / (2.0 * eps)

    if not np.isfinite(deriv) or abs(deriv) < 1e-8:
        # Try a larger epsilon once
        eps2 = min(0.2, 2.0 * eps)
        v_p = v.copy(); v_m = v.copy()
        v_p[ring] += eps2
        v_m[ring] -= eps2
        r_p = _map_v_to_radii(v_p)[ring]
        r_m = _map_v_to_radii(v_m)[ring]
        deriv = (r_p - r_m) / (2.0 * eps2)
        if not np.isfinite(deriv) or abs(deriv) < 1e-8:
            return 0.0

    dv = dr_eff / deriv
    return float(np.clip(dv, -0.5, 0.5))


# ----------------------- CPS with adaptive shrinking and guided block polls -----------------------
def _cps_optimize(v: np.ndarray, phi: np.ndarray,
                  step_v: float = 0.18, step_phi: float = 0.28,
                  shrink: float = 0.62, tol: float = 8e-4,
                  Kpoll: int = 96, max_sweeps: int = 5000):
    """
    First-improvement CPS over (v, phi) with adaptive shrinking.
    Guidance:
      - Weighted K-guidance (+co-participation) for ring order and best pair.
      - Smallest-triangle micro-probe splaying phases with coordinated radial tweaks.
    Per-ring adaptive step schedules:
      - Maintain dv_s, dph_s per ring (init from step_v/step_phi).
      - On acceptance for ring s, grow steps (dv_s*=1.22 up to 0.45; dph_s*=1.12 up to 0.45).
      - After 3 consecutive failures for s in a sweep, shrink (*=0.7 down to tol).
    Plateau escape:
      - After 6 non-improving sweeps, deterministic phase nudge on all rings.
    """
    v = np.asarray(v, dtype=np.float64).copy()
    phi = np.asarray(phi, dtype=np.float64).copy()

    best = _eval_from_params(v, phi)
    if not best.ok:
        v = np.clip(v, -4.0, 4.0)
        phi = _wrap_phi_mod(phi)
        best = _eval_from_params(v, phi)

    dv_base = float(step_v)
    dph_base = float(step_phi)
    dv_s = np.full(4, dv_base, dtype=np.float64)
    dph_s = np.full(4, dph_base, dtype=np.float64)
    fail_s = np.zeros(4, dtype=np.int64)

    flat = 0

    for _ in range(max_sweeps):
        improved_any = False
        ring_improved = np.zeros(4, dtype=bool)

        # dv cache per sweep; cleared whenever v/phi accept
        dv_cache = {}

        def get_dv_cached(ring_idx: int, dr_value: float) -> float:
            key = (int(ring_idx), float(np.round(dr_value, 12)))
            if key in dv_cache:
                return dv_cache[key]
            dv_val = _approx_dv_for_dr_sym(v, ring_idx, dr_value)
            dv_cache[key] = dv_val
            return dv_val

        # Smallest-triangle micro-probe (excluding center)
        try:
            areas_full = _all_triangle_areas_13(best.P)
            order = np.argsort(areas_full, kind="mergesort")
            tri_idx = None
            for t in order:
                if not TRI_HAS_CENTER[t]:
                    tri_idx = int(t)
                    break
            if tri_idx is not None:
                rings_in_tri = TRI_RING_IDS[tri_idx]
                rings_seq = [int(r) for r in rings_in_tri if r >= 0]
                uniq = []
                for rr in rings_seq:
                    if rr not in uniq:
                        uniq.append(rr)
                if len(uniq) >= 2:
                    phm = _wrap_phi_mod(phi)
                    L = TWO_PI_OVER_3

                    def circ_dist(a, b):
                        d = abs((phm[a] - phm[b]) % L)
                        return d if d <= L - d else L - d

                    if len(uniq) == 2:
                        a, b = uniq[0], uniq[1]
                        c = None
                    else:
                        pairs = [(uniq[i], uniq[j]) for i in range(3) for j in range(i + 1, 3)]
                        a, b = min(pairs, key=lambda pr: circ_dist(pr[0], pr[1]))
                        c = next(r for r in uniq if r not in (a, b))

                    dphi_pair = 0.10
                    dr_pair = 0.05
                    dr_third = 0.03
                    cand_specs = [
                        (+dphi_pair, -dphi_pair, +dr_pair, +dr_pair, -dr_third),
                        (+dphi_pair, -dphi_pair, -dr_pair, -dr_pair, +dr_third),
                        (-dphi_pair, +dphi_pair, +dr_pair, +dr_pair, -dr_third),
                        (-dphi_pair, +dphi_pair, -dr_pair, -dr_pair, +dr_third),
                    ]

                    accepted = False
                    for dp_a, dp_b, dr_a, dr_b, dr_c in cand_specs:
                        v_try = v.copy()
                        phi_try = phi.copy()

                        dv_a = _approx_dv_for_dr_sym(v_try, a, dr_a)
                        dv_b = _approx_dv_for_dr_sym(v_try, b, dr_b)
                        v_try[a] += dv_a
                        v_try[b] += dv_b
                        if c is not None:
                            dv_c = _approx_dv_for_dr_sym(v_try, c, dr_c)
                            v_try[c] += dv_c

                        phi_try[a] = (phi_try[a] + dp_a) % TWO_PI_OVER_3
                        phi_try[b] = (phi_try[b] + dp_b) % TWO_PI_OVER_3

                        cand = _eval_from_params(v_try, phi_try)
                        if cand.ok and _is_better(cand, best):
                            v, phi = v_try, phi_try
                            best = cand
                            improved_any = True
                            for rr in [a, b] + ([c] if c is not None else []):
                                ring_improved[rr] = True
                                dv_s[rr] = min(0.45, dv_s[rr] * 1.22)
                                dph_s[rr] = min(0.45, dph_s[rr] * 1.12)
                            dv_cache = {}
                            break
                    if improved_any:
                        # Restart sweep after micro-probe acceptance
                        pass
        except Exception:
            pass

        # Weighted ring guidance (order + best pair)
        try:
            C, W, ring_order, top_pair = _ring_guidance_from_smallest(best.P, K=Kpoll, tau=0.018)
        except Exception:
            C = np.zeros(4, dtype=np.float64)
            W = np.zeros((4, 4), dtype=np.float64)
            ring_order = list(np.arange(4))
            top_pair = (0, 1)

        cmax = float(np.max(C))

        # Guided block polls if we have any signal
        if cmax > 0.0:
            # Per-ring adaptive step magnitudes for the block polls
            sr_s = 0.06 + 0.05 * (C / cmax)
            sp_s = 0.18 + 0.08 * (C / cmax)

            # Single-ring joint 3x3 for top-2 rings (first-improvement)
            top2 = ring_order[:2]
            accepted_outer = False
            for s in top2:
                sr = float(sr_s[s])
                sp = float(sp_s[s])
                dr_choices = (-sr, 0.0, sr)
                ph_choices = (-sp, 0.0, sp)
                # Precompute dv for dr choices (cache-aware)
                dv_map = [get_dv_cached(s, dr) for dr in dr_choices]
                for di in range(3):
                    for pj in range(3):
                        if di == 1 and pj == 1:
                            continue  # skip (0,0)
                        v_try = v.copy()
                        phi_try = phi.copy()
                        v_try[s] += dv_map[di]
                        phi_try[s] = (phi_try[s] + ph_choices[pj]) % TWO_PI_OVER_3
                        cand = _eval_from_params(v_try, phi_try)
                        if cand.ok and _is_better(cand, best):
                            v, phi = v_try, phi_try
                            best = cand
                            improved_any = True
                            ring_improved[s] = True
                            dv_s[s] = min(0.45, dv_s[s] * 1.22)
                            dph_s[s] = min(0.45, dph_s[s] * 1.12)
                            dv_cache = {}
                            accepted_outer = True
                            break
                    if accepted_outer:
                        break
                if accepted_outer:
                    break

            # Paired-ring coordination if still no improvement
            if not improved_any:
                s0, s1 = top_pair
                denom = max(1.0, float(C[s0] + C[s1]))
                alpha = float(np.clip(0.6 + 0.4 * (W[s0, s1] / denom), 0.5, 1.0))
                done = False
                for sign in (-1.0, 1.0):
                    dr0 = alpha * float(sr_s[s0]) * sign
                    dr1 = alpha * float(sr_s[s1]) * sign
                    dv0 = get_dv_cached(s0, dr0)
                    dv1 = get_dv_cached(s1, dr1)
                    for psign in (-1.0, 1.0):
                        dp0 = alpha * float(sp_s[s0]) * psign
                        dp1 = alpha * float(sp_s[s1]) * psign
                        # Pattern A: same sign in phase
                        v_try = v.copy()
                        phi_try = phi.copy()
                        v_try[s0] += dv0
                        v_try[s1] += dv1
                        phi_try[s0] = (phi_try[s0] + dp0) % TWO_PI_OVER_3
                        phi_try[s1] = (phi_try[s1] + dp1) % TWO_PI_OVER_3
                        cand = _eval_from_params(v_try, phi_try)
                        if cand.ok and _is_better(cand, best):
                            v, phi = v_try, phi_try
                            best = cand
                            improved_any = True
                            ring_improved[s0] = True
                            ring_improved[s1] = True
                            dv_s[s0] = min(0.45, dv_s[s0] * 1.22)
                            dv_s[s1] = min(0.45, dv_s[s1] * 1.22)
                            dph_s[s0] = min(0.45, dph_s[s0] * 1.12)
                            dph_s[s1] = min(0.45, dph_s[s1] * 1.12)
                            dv_cache = {}
                            done = True
                            break
                        # Pattern B: opposite sign in phase
                        v_try = v.copy()
                        phi_try = phi.copy()
                        v_try[s0] += dv0
                        v_try[s1] += dv1
                        phi_try[s0] = (phi_try[s0] + dp0) % TWO_PI_OVER_3
                        phi_try[s1] = (phi_try[s1] - dp1) % TWO_PI_OVER_3
                        cand = _eval_from_params(v_try, phi_try)
                        if cand.ok and _is_better(cand, best):
                            v, phi = v_try, phi_try
                            best = cand
                            improved_any = True
                            ring_improved[s0] = True
                            ring_improved[s1] = True
                            dv_s[s0] = min(0.45, dv_s[s0] * 1.22)
                            dv_s[s1] = min(0.45, dv_s[s1] * 1.22)
                            dph_s[s0] = min(0.45, dph_s[s0] * 1.12)
                            dph_s[s1] = min(0.45, dph_s[s1] * 1.12)
                            dv_cache = {}
                            done = True
                            break
                    if done:
                        break

        # Scalar CPS sweep (first-improvement), visit rings by guided order
        for s in ring_order:
            # v step +/-
            if not ring_improved[s]:
                for d in (dv_s[s], -dv_s[s]):
                    v_try = v.copy()
                    v_try[s] += d
                    cand = _eval_from_params(v_try, phi)
                    if cand.ok and _is_better(cand, best):
                        v = v_try
                        best = cand
                        improved_any = True
                        ring_improved[s] = True
                        dv_s[s] = min(0.45, dv_s[s] * 1.22)
                        dv_cache = {}
                        break
            # phi step +/-
            if not ring_improved[s]:
                for d in (dph_s[s], -dph_s[s]):
                    phi_try = phi.copy()
                    phi_try[s] = (phi_try[s] + d) % TWO_PI_OVER_3
                    cand = _eval_from_params(v, phi_try)
                    if cand.ok and _is_better(cand, best):
                        phi = phi_try
                        best = cand
                        improved_any = True
                        ring_improved[s] = True
                        dph_s[s] = min(0.45, dph_s[s] * 1.12)
                        dv_cache = {}
                        break

        # Update per-ring failure counters and apply targeted shrink
        for s in range(4):
            if ring_improved[s]:
                fail_s[s] = 0
            else:
                fail_s[s] += 1
                if fail_s[s] >= 3:
                    dv_s[s] = max(tol, dv_s[s] * 0.7)
                    dph_s[s] = max(tol, dph_s[s] * 0.7)
                    fail_s[s] = 0

        if improved_any:
            flat = 0
        else:
            flat += 1
            # Plateau handling: deterministic per-ring φ nudges wrapped into [0, 2π/3)
            if flat == 6:
                phi_try = phi.copy()
                for j in range(4):
                    phi_try[j] = (phi_try[j] + 0.07 * (1.0 + (j - 3) * 0.13)) % TWO_PI_OVER_3
                cand = _eval_from_params(v, phi_try)
                if cand.ok and _is_better(cand, best):
                    phi = phi_try
                    best = cand
                    flat = 0

            # Global shrink
            dv_s *= shrink
            dph_s *= shrink

            if float(np.max(dv_s)) < tol and float(np.max(dph_s)) < tol:
                break

    return v, phi


# ----------------------- Seeds (deterministic) -----------------------
def _seed_lists():
    v_seeds = [
        [0.5, 0.5, 0.5, 0.5],
        [0.3, 0.6, 0.9, 1.2],
        [1.2, 0.9, 0.6, 0.3],
        [0.2, 0.4, 0.8, 1.2],
        [0.8, 0.4, 0.6, 0.9],
        [0.2, 0.9, 1.4, 2.0],
        [2.0, 1.4, 0.9, 0.2],
        [0.1, 0.7, 1.1, 1.6],
        [0.0, 0.9, 0.0, 0.9],      # extra diverse seeds
        [1.3, 0.4, 1.0, 0.2],
    ]
    phi_seeds = [
        [0.00, 0.37, 0.74, 1.08],
        [0.12, 0.59, 1.03, 1.49],
        [0.21, 0.68, 1.16, 1.62],
        [0.28, 0.94, 1.57, 2.18],
        [0.45, 0.91, 1.37, 1.83],
        [0.07, 0.51, 0.98, 1.41],
        [0.19, 0.66, 1.11, 1.56],
    ]
    return [np.array(v, dtype=np.float64) for v in v_seeds], [np.array(p, dtype=np.float64) for p in phi_seeds]


# ----------------------- Main API -----------------------
def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic constructor for 13 points maximizing the smallest triangle area
    under convex hull area normalization to 1.0.
    - 3-fold ring parameterization (8D)
    - Robust two-pass convex hull normalization
    - Deterministic multi-seed pre-refinement and pure-objective ranking
    - CPS with adaptive shrinking, weighted K=96 guidance, symmetric dv-for-dr, and micro-probes
    Returns:
        np.ndarray of shape (13,2)
    """
    v_list, p_list = _seed_lists()

    # Rank seeds by exact normalized min-area only (penalties excluded)
    best_seed = None
    best_m = -np.inf
    seed_idx = 0
    best_seed_idx = None
    for v0 in v_list:
        for p0 in p_list:
            v_pre, p_pre = _short_prefine(v0, p0, step_v=0.14, step_phi=0.27)
            ev = _eval_from_params(v_pre, p_pre)
            m_val = ev.m if ev.ok else -np.inf
            # Pure m ranking; tie-break by deterministic earlier index
            if (m_val > best_m + 1e-12) or (abs(m_val - best_m) <= 1e-12 and (best_seed_idx is None or seed_idx < best_seed_idx)):
                best_m = m_val
                best_seed = (v_pre.copy(), p_pre.copy())
                best_seed_idx = seed_idx
            seed_idx += 1

    if best_seed is None:
        # deterministic safe fallback
        v_start = np.array([0.3, 0.6, 0.9, 1.2], dtype=np.float64)
        p_start = np.array([0.00, 0.37, 0.74, 1.08], dtype=np.float64)
    else:
        v_start, p_start = best_seed

    # Local CPS with K-guided block polls and symmetric dv mapping
    v_opt, p_opt = _cps_optimize(v_start, p_start,
                                 step_v=0.18, step_phi=0.28,
                                 shrink=0.62, tol=8e-4, Kpoll=96, max_sweeps=5000)

    # Build final normalized points
    ev_fin = _eval_from_params(v_opt, p_opt)
    if ev_fin.ok:
        P = ev_fin.P
    else:
        # robust fallback: regular 12-gon + center
        m = 12
        ang = TWO_PI * (np.arange(m) / m)
        ring = np.column_stack([np.cos(ang), np.sin(ang)])
        P = np.vstack([np.zeros((1, 2), dtype=np.float64), ring]).astype(np.float64)
        ok, P = _normalize_hull_area(P)
        if not ok:
            # Last resort: return centered points
            P -= np.mean(P, axis=0, keepdims=True)

    # Final deterministic angle sort
    th = np.arctan2(P[:, 1], P[:, 0])
    order = np.argsort(th, kind="mergesort")
    P = P[order]

    # Tiny deterministic jitter if any coincidences, then renormalize
    D = P[:, None, :] - P[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', D, D)
    np.fill_diagonal(d2, np.inf)
    min_d2 = float(np.min(d2))
    if min_d2 < 1e-18 or not np.all(np.isfinite(d2)):
        idx = np.arange(P.shape[0], dtype=np.float64)
        eps = 1e-9
        jitter = eps * np.column_stack([np.cos(1.618 * idx), np.sin(2.414 * idx)])
        ok, P = _normalize_hull_area(P + jitter)
        if not ok:
            # If normalization fails, just apply jitter without scaling
            P = P + jitter
        th = np.arctan2(P[:, 1], P[:, 0])
        order = np.argsort(th, kind="mergesort")
        P = P[order]

    # Enforce exact unit area once more
    ok, P = _normalize_hull_area(P)
    if not ok:
        # Should not happen; but ensure return shape and finite numbers
        P = np.nan_to_num(P)
    assert P.shape == (13, 2)
    assert np.all(np.isfinite(P))
    return P


# EVOLVE-BLOCK-END