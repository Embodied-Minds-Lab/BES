# EVOLVE-BLOCK-START
import numpy as np


def _equilateral_unit_area_vertices():
    """
    Return vertices (A,B,C), centroid G, side length s, height h, inradius r_in,
    for an equilateral triangle of area 1 aligned with base on x-axis.
    """
    # Area = (sqrt(3)/4) * s^2 = 1 => s = 2 / 3^(1/4)
    s = 2.0 / (3.0 ** 0.25)
    h = (np.sqrt(3.0) / 2.0) * s
    A = np.array([0.0, 0.0])
    B = np.array([s, 0.0])
    C = np.array([0.5 * s, h])
    # Centroid G (which is also incenter/circumcenter for equilateral)
    G = np.array([0.5 * s, h / 3.0])
    r_in = h / 3.0  # inradius
    return A, B, C, G, s, h, r_in


def _rot2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _normalize_angles(phases):
    return (phases + np.pi) % (2.0 * np.pi) - np.pi


def _prepare_triangle_sides(tri):
    A, B, C = tri
    sides = []
    for V0, V1 in ((A, B), (B, C), (C, A)):
        D = V1 - V0
        sides.append((V0, D))
    return sides


def _ray_to_triangle_boundary_t(G, u, tri_sides, eps_det=1e-12, eps_seg=1e-12):
    """
    Compute smallest positive t such that G + t*u intersects triangle boundary segments.
    Uses Cramer's rule via cross products for speed and robustness.
    Accept intersections with t > eps_det and segment parameter s in [-eps_seg, 1+eps_seg].
    """
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    t_best = np.inf
    for V0, D in tri_sides:
        denom = cross(u, D)
        if abs(denom) < eps_det:
            continue
        rhs = V0 - G
        # t = cross(rhs, D) / cross(u, D)
        # s = -cross(u, rhs) / cross(u, D)
        t = cross(rhs, D) / denom
        s = -cross(u, rhs) / denom
        if t > eps_det and s >= -eps_seg and s <= 1.0 + eps_seg:
            if t < t_best:
                t_best = t
    if not np.isfinite(t_best):
        return 0.0
    return t_best


def _build_points_from_rings(radii, phases, G, tri=None, boundary_tol=1e-9):
    """
    Given 4 radii and 4 phases, build 12 ring points via 120-degree rotations plus the centroid.
    Mandatory exact ray-to-triangle clamping along each ring direction.
    Returns an array of shape (13,2).
    """
    assert len(radii) == 4 and len(phases) == 4
    R120 = _rot2(2.0 * np.pi / 3.0)
    tri_sides = _prepare_triangle_sides(tri) if tri is not None else None

    pts = []
    for r, phi in zip(radii, phases):
        # Three coherent directions by 120° rotations
        u0 = np.array([np.cos(phi), np.sin(phi)])
        u1 = R120 @ u0
        u2 = R120 @ u1

        for u in (u0, u1, u2):
            if tri_sides is not None:
                tmax = _ray_to_triangle_boundary_t(G, u, tri_sides)
                r_eff = max(0.0, min(r, tmax - boundary_tol))
            else:
                r_eff = r
            pts.append(G + r_eff * u)

    pts.append(G.copy())  # centroid as 13th point
    return np.vstack(pts)


def _monotonic_chain_convex_hull(points):
    """
    Monotonic chain convex hull. Returns hull vertices in CCW order.
    """
    pts = np.asarray(points)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _all_triangle_areas(points, triplets=None):
    """
    Compute areas of all C(n,3) triangles; returns array of areas and (optionally) the triplets used.
    """
    P = np.asarray(points)
    n = P.shape[0]
    if triplets is None:
        idx = []
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    idx.append((i, j, k))
        triplets = np.array(idx, dtype=int)
    A = P[triplets[:, 0]]
    B = P[triplets[:, 1]]
    C = P[triplets[:, 2]]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas, triplets


def _objective(points, triplets=None, k_secondary=10, eps=1e-12):
    """
    Return:
      - primary: min triangle area normalized by convex hull area
      - secondary: mean of k smallest normalized triangle areas (tie-breaker)
      - hull_area
    """
    hull = _monotonic_chain_convex_hull(points)
    hull_area = max(_polygon_area(hull), eps)
    areas, _ = _all_triangle_areas(points, triplets)
    areas_norm = areas / hull_area
    areas_sorted = np.sort(areas_norm)
    primary = areas_sorted[0] if len(areas_sorted) > 0 else 0.0
    k = min(k_secondary, len(areas_sorted))
    secondary = float(np.mean(areas_sorted[:k])) if k > 0 else 0.0
    return primary, secondary, hull_area


def _phase_spread_mod120(phases):
    """
    Tertiary tie-breaker: minimal separation among base phases modulo 120 degrees.
    Larger is better.
    """
    L = 2.0 * np.pi / 3.0
    phi = ((np.asarray(phases) % L) + L) % L  # map to [0, L)
    phi = np.sort(phi)
    gaps = np.diff(phi, append=phi[0] + L)
    return float(np.min(gaps))


def _lex_better(a, b, tol=1e-12):
    """
    Tolerant lexicographic comparison for (primary, secondary, tertiary_spread).
    Primary wins; if within tol, compare secondary; if both within tol, compare tertiary.
    """
    a1, a2, a3 = a
    b1, b2, b3 = b
    if a1 > b1 + tol:
        return True
    if b1 > a1 + tol:
        return False
    if a2 > b2 + tol:
        return True
    if b2 > a2 + tol:
        return False
    return a3 > b3 + tol


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic D3-symmetric multi-start enriched pattern-search constructor for 13 points inside a unit-area
    equilateral triangle (a convex region). It maximizes the minimum triangle area among all
    C(13,3) triangles, normalized by the convex hull area of the configuration.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    # Geometry of unit-area equilateral triangle
    A, B, C, G, s, h, r_in = _equilateral_unit_area_vertices()
    tri = (A, B, C)

    # Deterministic multi-start initializations
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    ang_to_A = np.arctan2(A[1] - G[1], A[0] - G[0])

    starts = []

    # 1) Compact prior-like
    starts.append((
        np.array([0.18, 0.29, 0.37, 0.438], dtype=float) * r_in,
        _normalize_angles(np.array([0.0, np.pi / 11.0, np.pi / 6.5, np.pi / 4.2], dtype=float))
    ))

    # 2) Golden-angle phased
    starts.append((
        np.array([0.17, 0.28, 0.36, 0.445], dtype=float) * r_in,
        _normalize_angles(np.array([0.0, 0.5 * golden_angle, 1.0 * golden_angle, 1.5 * golden_angle], dtype=float))
    ))

    # 3) Staggered/side-normal flavored
    starts.append((
        np.array([0.15, 0.26, 0.345, 0.44], dtype=float) * r_in,
        _normalize_angles(np.array([np.pi / 20.0, np.pi / 7.0, np.pi / 3.7, np.pi / 2.8], dtype=float))
    ))

    # 4) Geometry-aware boundary-biased (relies on clamping)
    starts.append((
        np.array([0.42, 0.86, 1.28, 1.70], dtype=float) * r_in,
        _normalize_angles(np.array([
            -np.pi / 2.0 + np.deg2rad(4.0),     # near base normal
            ang_to_A + np.deg2rad(9.0),         # near vertex ray, slight offset
            ang_to_A + np.deg2rad(33.0),        # off-vertex direction
            np.deg2rad(19.0)                    # another off-side direction
        ], dtype=float))
    ))

    # 5) Optional mirrored-phase variant to diversify deterministically
    r_mir, phi_mir = starts[0]
    starts.append((r_mir.copy(), _normalize_angles(-phi_mir.copy())))

    # Bounds and steps
    r_min = 0.06 * r_in
    r_max = 2.0 * r_in - 1e-6
    base_dr = 0.11 * r_in
    base_dphi = np.deg2rad(10.0)
    step_reduce = 0.6

    # Precompute triplets for n=13 once using the first start
    dummy_pts = _build_points_from_rings(starts[0][0], starts[0][1], G, tri=tri)
    _, triplets = _all_triangle_areas(dummy_pts)

    # Helper to evaluate config with tertiary phase-spread
    def eval_config(radii, phases):
        P = _build_points_from_rings(radii, phases, G, tri=tri)
        prim, sec, _ = _objective(P, triplets=triplets)
        spread = _phase_spread_mod120(phases)
        return (prim, sec, spread)

    # Global best across starts
    global_best_val = (-1.0, -1.0, -1.0)
    global_best_radii = None
    global_best_phases = None

    for radii0, phases0 in starts:
        # Initialize per start
        radii = np.clip(radii0.astype(float), r_min, r_max)
        phases = _normalize_angles(phases0.astype(float))
        dr = base_dr
        dphi = base_dphi

        best_radii = radii.copy()
        best_phases = phases.copy()
        best_val = eval_config(best_radii, best_phases)

        # Enriched deterministic pattern search
        for _iter in range(26):
            improved = False

            # Sweep rings with greedy acceptance
            for k in range(4):
                candidates = []

                # Radial ±dr
                for sign in (+1.0, -1.0):
                    radii_try = best_radii.copy()
                    radii_try[k] = float(np.clip(radii_try[k] + sign * dr, r_min, r_max))
                    val = eval_config(radii_try, best_phases)
                    candidates.append((val, radii_try, best_phases, True, False, sign, 0.0))

                # Angular ±dphi
                for sign in (+1.0, -1.0):
                    phases_try = best_phases.copy()
                    phases_try[k] = phases_try[k] + sign * dphi
                    phases_try = _normalize_angles(phases_try)
                    val = eval_config(best_radii, phases_try)
                    candidates.append((val, best_radii, phases_try, False, True, 0.0, sign))

                # Coupled moves (±dr, ±dphi)
                for s_r in (+1.0, -1.0):
                    for s_p in (+1.0, -1.0):
                        radii_try = best_radii.copy()
                        phases_try = best_phases.copy()
                        radii_try[k] = float(np.clip(radii_try[k] + s_r * dr, r_min, r_max))
                        phases_try[k] = phases_try[k] + s_p * dphi
                        phases_try = _normalize_angles(phases_try)
                        val = eval_config(radii_try, phases_try)
                        candidates.append((val, radii_try, phases_try, True, True, s_r, s_p))

                # Pick best improving candidate
                chosen = None
                for cand in candidates:
                    if _lex_better(cand[0], best_val):
                        if chosen is None or _lex_better(cand[0], chosen[0]):
                            chosen = cand

                if chosen is not None:
                    best_val, best_radii, best_phases, chg_r, chg_p, s_r, s_p = chosen
                    improved = True

                    # Short deterministic line-search: one more step along same signed direction(s)
                    radii_ls = best_radii.copy()
                    phases_ls = best_phases.copy()
                    if chg_r:
                        radii_ls[k] = float(np.clip(radii_ls[k] + s_r * dr, r_min, r_max))
                    if chg_p:
                        phases_ls[k] = phases_ls[k] + s_p * dphi
                        phases_ls = _normalize_angles(phases_ls)
                    val_ls = eval_config(radii_ls, phases_ls)
                    if _lex_better(val_ls, best_val):
                        best_val = val_ls
                        best_radii = radii_ls
                        best_phases = phases_ls

            # Coherent global moves (once per outer iteration)
            global_candidates = []

            # Global phase shift ±0.5*dphi
            for sign in (+1.0, -1.0):
                phases_try = _normalize_angles(best_phases + sign * 0.5 * dphi)
                val = eval_config(best_radii, phases_try)
                global_candidates.append((val, best_radii, phases_try))

            # Uniform radius scaling {1.08, 0.92}
            for sf in (1.08, 0.92):
                radii_try = np.clip(best_radii * sf, r_min, r_max)
                val = eval_config(radii_try, best_phases)
                global_candidates.append((val, radii_try, best_phases))

            chosen = None
            for val, rad_c, pha_c in global_candidates:
                if _lex_better(val, best_val):
                    if chosen is None or _lex_better(val, chosen[0]):
                        chosen = (val, rad_c, pha_c)
            if chosen is not None:
                best_val, best_radii, best_phases = chosen
                improved = True

            if not improved:
                dr *= step_reduce
                dphi *= step_reduce
                if dr < 1e-5 and dphi < 1e-5:
                    break

        # Update global best
        if _lex_better(best_val, global_best_val):
            global_best_val = best_val
            global_best_radii = best_radii
            global_best_phases = best_phases

    # Final point set from the best start
    final_points = _build_points_from_rings(global_best_radii, global_best_phases, G, tri=tri)
    return final_points


# EVOLVE-BLOCK-END