# EVOLVE-BLOCK-START
import numpy as np
from typing import Tuple, List, Optional

try:
    import scipy.optimize as spo
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def _pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    iu, ju = np.triu_indices(n, k=1)
    return iu.astype(int), ju.astype(int)


def _unpack(z: np.ndarray, n: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    w = z[0]
    xs = z[1 : 1 + n]
    ys = z[1 + n : 1 + 2 * n]
    rs = z[1 + 2 * n : 1 + 3 * n]
    return w, xs, ys, rs


def _objective(z: np.ndarray, n: int) -> float:
    # Maximize sum(r_i) -> minimize -sum(r_i)
    return -np.sum(z[1 + 2 * n : 1 + 3 * n])


def _objective_grad(z: np.ndarray, n: int) -> np.ndarray:
    g = np.zeros_like(z)
    g[1 + 2 * n : 1 + 3 * n] = -1.0
    return g


def _constraints_vec(z: np.ndarray, n: int, iu: np.ndarray, ju: np.ndarray) -> np.ndarray:
    # Returns g(z) >= 0 vector: boundary (4n) then pairwise (nC2), squared geometry
    w, xs, ys, rs = _unpack(z, n)
    h = 2.0 - w
    # Boundary constraints
    c1 = xs - rs
    c2 = (w - xs) - rs
    c3 = ys - rs
    c4 = (h - ys) - rs
    # Pairwise non-overlap (squared)
    dx = xs[iu] - xs[ju]
    dy = ys[iu] - ys[ju]
    dist2 = dx * dx + dy * dy
    sr = rs[iu] + rs[ju]
    cp = dist2 - sr * sr
    return np.concatenate([c1, c2, c3, c4, cp])


def _constraints_jac(z: np.ndarray, n: int, iu: np.ndarray, ju: np.ndarray, sparse: bool = True):
    """
    Analytic Jacobian for constraints vector, including exact partials wrt width w.
    Layout:
      vars: [w, x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
      rows: [c1(0..n-1), c2(0..n-1), c3(0..n-1), c4(0..n-1), cp(0..m-1)]
    """
    w, xs, ys, rs = _unpack(z, n)
    m = 4 * n + iu.size
    p = 1 + 3 * n

    use_sparse = False
    if SCIPY_AVAILABLE and sparse:
        try:
            from scipy.sparse import coo_matrix  # type: ignore
            use_sparse = True
        except Exception:
            use_sparse = False

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    def add(rw: int, cl: int, val: float):
        rows.append(rw)
        cols.append(cl)
        data.append(val)

    # Boundary parts
    for i in range(n):
        xi = 1 + i
        yi = 1 + n + i
        ri = 1 + 2 * n + i
        # c1: x_i - r_i
        add(i, xi, 1.0)
        add(i, ri, -1.0)
        # c2: (w - x_i) - r_i
        add(n + i, 0, 1.0)     # d/dw = +1
        add(n + i, xi, -1.0)   # d/dx_i = -1
        add(n + i, ri, -1.0)   # d/dr_i = -1
        # c3: y_i - r_i
        add(2 * n + i, yi, 1.0)
        add(2 * n + i, ri, -1.0)
        # c4: (h - y_i) - r_i with h=2-w -> d/dw = -1
        add(3 * n + i, 0, -1.0)   # d/dw = -1
        add(3 * n + i, yi, -1.0)  # d/dy_i = -1
        add(3 * n + i, ri, -1.0)  # d/dr_i = -1

    # Pairwise squared parts
    dx = xs[iu] - xs[ju]
    dy = ys[iu] - ys[ju]
    srij = rs[iu] + rs[ju]
    for k in range(iu.size):
        rw = 4 * n + k
        i = int(iu[k])
        j = int(ju[k])
        xi = 1 + i
        xj = 1 + j
        yi = 1 + n + i
        yj = 1 + n + j
        ri = 1 + 2 * n + i
        rj = 1 + 2 * n + j
        ddx = 2.0 * dx[k]
        ddy = 2.0 * dy[k]
        s2 = -2.0 * srij[k]
        # center derivatives
        add(rw, xi, ddx)
        add(rw, xj, -ddx)
        add(rw, yi, ddy)
        add(rw, yj, -ddy)
        # radii derivatives
        add(rw, ri, s2)
        add(rw, rj, s2)

    if use_sparse:
        from scipy.sparse import coo_matrix  # type: ignore
        return coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(m, p)).tocsr()
    else:
        J = np.zeros((m, p), dtype=float)
        if rows:
            J[np.array(rows, dtype=int), np.array(cols, dtype=int)] = np.array(data, dtype=float)
        return J


def _bounds(n: int) -> List[Tuple[float, float]]:
    # Static box bounds; geometry constraints ensure validity
    # w in [0.05, 1.95], x in [0, 2], y in [0, 2], r in [1e-8, 1]
    b = [(0.05, 1.95)]
    b += [(0.0, 2.0)] * n
    b += [(0.0, 2.0)] * n
    b += [(1e-8, 1.0)] * n
    return b


def _grid_centers_7x3(w: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    # Deterministic 7x3 grid (exactly 21 points) with margins
    cols, rows = 7, 3
    mx, my = 0.06 * w, 0.06 * h
    xs = np.linspace(mx, w - mx, cols)
    ys = np.linspace(my, h - my, rows)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def _hex21_centers(w: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    # Hex-like pattern with row counts [5,4,5,4,3] summing to 21
    # Slightly tighter vertical spacing and margins to better utilize h while remaining feasible.
    row_counts = [5, 4, 5, 4, 3]
    R = len(row_counts)
    cmax = max(row_counts)
    mx, my = 0.04 * w, 0.04 * h
    usable_w = max(w - 2 * mx, 1e-8)
    dx = usable_w / cmax
    dy = min(0.97 * h / max(R - 1, 1), (np.sqrt(3.0) / 2.0) * dx)
    # vertically center rows with spacing dy
    y0 = (h - (R - 1) * dy) * 0.5
    xs_all, ys_all = [], []
    for r, c in enumerate(row_counts):
        L = c * dx
        # center this row's block horizontally within [mx, w-mx]
        left_pad = (usable_w - L) * 0.5
        base = mx + left_pad
        shift = 0.5 * dx if (r % 2 == 1) else 0.0
        for k in range(c):
            xs_all.append(base + shift + (k + 0.5) * dx)
            ys_all.append(y0 + r * dy)
    return np.array(xs_all), np.array(ys_all)


def _diamond21_centers(w: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diamond-like stagger with row counts summing to 21: [3,4,5,4,3,2]
    This offers a different vertical packing profile that can be advantageous for
    slightly taller or squatter rectangles.
    """
    row_counts = [3, 4, 5, 4, 3, 2]
    R = len(row_counts)
    cmax = max(row_counts)  # 5
    mx, my = 0.05 * w, 0.05 * h
    usable_w = max(w - 2 * mx, 1e-8)
    dx = usable_w / cmax
    # Keep a conservative vertical spacing consistent with near-hex relationship
    dy = min(0.95 * h / max(R - 1, 1), (np.sqrt(3.0) / 2.0) * dx)
    y0 = (h - (R - 1) * dy) * 0.5
    xs_all, ys_all = [], []
    for r, c in enumerate(row_counts):
        L = c * dx
        left_pad = (usable_w - L) * 0.5
        base = mx + left_pad
        shift = 0.5 * dx if (r % 2 == 1) else 0.0
        for k in range(c):
            xs_all.append(base + shift + (k + 0.5) * dx)
            ys_all.append(y0 + r * dy)
    return np.array(xs_all), np.array(ys_all)


def _greedy_radii(w: float, h: float, xs: np.ndarray, ys: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    n = xs.size
    rs = np.minimum.reduce([xs, w - xs, ys, h - ys]).copy()
    rs = np.maximum(rs, 1e-6)
    iu, ju = _pair_indices(n)
    eps = 1e-10
    for _ in range(max_iter):
        dx = xs[iu] - xs[ju]
        dy = ys[iu] - ys[ju]
        d = np.hypot(dx, dy)
        s = rs[iu] + rs[ju]
        over = s - d + eps
        viol = over > 0
        if not np.any(viol):
            break
        idx_i = iu[viol]
        idx_j = ju[viol]
        delta = over[viol] * 0.5
        # Reduce both radii equally
        np.subtract.at(rs, idx_i, delta)
        np.subtract.at(rs, idx_j, delta)
        rs = np.maximum(rs, 1e-6)
        # Keep inside bounds
        rs = np.minimum(rs, np.minimum.reduce([xs, w - xs, ys, h - ys]))
    return np.maximum(rs, 1e-6)


def _micro_inflate(w: float, h: float, xs: np.ndarray, ys: np.ndarray, rs: np.ndarray, iters: int = 10, step: float = 0.4) -> np.ndarray:
    """
    Targeted micro-inflation: increase each r_i by a conservative fraction of the
    minimum of its boundary slack and pairwise slack while preserving feasibility.
    """
    n = xs.size
    caps = np.minimum.reduce([xs, w - xs, ys, h - ys])
    r = np.minimum(rs.copy(), caps)
    for _ in range(iters):
        # Pairwise slack matrix: slack_ij = dist(i,j) - (r_i + r_j)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        D = np.hypot(dx, dy)
        Slack = D - (r[:, None] + r[None, :])
        # Ignore self by setting diagonal to +inf
        np.fill_diagonal(Slack, np.inf)
        pair_slack = np.min(Slack, axis=1)
        pair_slack = np.maximum(pair_slack, 0.0)
        # Boundary slack
        b_slack = np.maximum(caps - r, 0.0)
        delta = np.minimum(b_slack, pair_slack)
        inc = step * delta
        if np.max(inc) < 1e-12:
            break
        r = np.minimum(r + inc, caps)
    return r


def _lp_radii_given_centers(w: float, h: float, xs: np.ndarray, ys: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Solve LP to maximize sum r_i subject to:
      r_i + r_j <= dist(i,j) - eps  for all i<j
      0 < r_i <= caps_i := min(x_i, w-x_i, y_i, h-y_i)
    Implemented as minimization of -sum r_i via scipy.optimize.linprog (HiGHS).
    """
    n = xs.size
    caps = np.minimum.reduce([xs, w - xs, ys, h - ys])
    caps = np.maximum(caps, 0.0)
    iu, ju = _pair_indices(n)
    m = iu.size
    # Vectorized build of A_ub
    A_ub = np.zeros((m, n), dtype=float)
    rows = np.arange(m, dtype=int)
    A_ub[rows, iu] = 1.0
    A_ub[rows, ju] += 1.0
    dx = xs[iu] - xs[ju]
    dy = ys[iu] - ys[ju]
    d = np.hypot(dx, dy)
    b_ub = d - eps
    # Objective: minimize -sum r_i
    c = -np.ones(n, dtype=float)
    bounds = [(1e-8, float(caps[i])) for i in range(n)]
    if SCIPY_AVAILABLE:
        try:
            res = spo.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success and res.x is not None:
                r = np.array(res.x, dtype=float)
                r = np.clip(r, 1e-8, caps)
                return r
        except Exception:
            pass
    # Fallback: greedy radii if LP not available or fails
    r = _greedy_radii(w, h, xs, ys, max_iter=400)
    return np.minimum(np.maximum(r, 1e-8), caps)


def _repair(w: float, h: float, xs: np.ndarray, ys: np.ndarray, rs: np.ndarray, passes: int = 4) -> np.ndarray:
    n = xs.size
    iu, ju = _pair_indices(n)
    r = rs.copy()
    for _ in range(passes):
        # Enforce boundary
        r = np.minimum(r, np.minimum.reduce([xs, w - xs, ys, h - ys]))
        r = np.maximum(r, 1e-8)
        # Resolve overlaps
        for _ in range(200):
            dx = xs[iu] - xs[ju]
            dy = ys[iu] - ys[ju]
            d = np.hypot(dx, dy)
            s = r[iu] + r[ju]
            over = s - d + 1e-12
            viol = over > 0
            if not np.any(viol):
                break
            idx_i = iu[viol]
            idx_j = ju[viol]
            delta = over[viol] * 0.5
            np.subtract.at(r, idx_i, delta)
            np.subtract.at(r, idx_j, delta)
            r = np.maximum(r, 1e-8)
    # Transplanted trick: targeted micro-inflation to reclaim slack safely
    r = _micro_inflate(w, h, xs, ys, r, iters=10, step=0.4)
    return r


def _z_from_whxr(w: float, xs: np.ndarray, ys: np.ndarray, rs: np.ndarray) -> np.ndarray:
    n = xs.size
    z = np.zeros(1 + 3 * n)
    z[0] = w
    z[1 : 1 + n] = xs
    z[1 + n : 1 + 2 * n] = ys
    z[1 + 2 * n : 1 + 3 * n] = rs
    return z


def _feasible_score(z: np.ndarray, n: int) -> float:
    # Return sum radii if feasible within small tolerance, else negative
    w, xs, ys, rs = _unpack(z, n)
    h = 2 - w
    tol = 1e-7
    if w <= 0 or h <= 0:
        return -1.0
    if np.any(rs <= 0):
        return -1.0
    if np.any(xs - rs < -tol) or np.any((w - xs) - rs < -tol) or np.any(ys - rs < -tol) or np.any((h - ys) - rs < -tol):
        return -1.0
    iu, ju = _pair_indices(n)
    dx = xs[iu] - xs[ju]
    dy = ys[iu] - ys[ju]
    if np.any((dx * dx + dy * dy) - (rs[iu] + rs[ju]) ** 2 < -1e-7):
        return -1.0
    return float(np.sum(rs))


def _feas_viol(z: np.ndarray, n: int) -> float:
    """
    Return the infinity-norm of constraint violation for g(z) >= 0:
      max(0, -min(g_i(z))) over all constraints, consistent with squared geometry.
    """
    iu, ju = _pair_indices(n)
    vals = _constraints_vec(z, n, iu, ju)
    mn = float(np.min(vals)) if vals.size else 0.0
    return max(0.0, -mn)


def _optimize_once(z0: np.ndarray, n: int, iu: np.ndarray, ju: np.ndarray, maxiter: int = 300) -> Optional[np.ndarray]:
    # Try trust-constr vectorized constraint first with analytic Jacobian
    if SCIPY_AVAILABLE:
        try:
            bounds = spo.Bounds(*np.array(_bounds(n)).T)
            nlc = spo.NonlinearConstraint(
                lambda z: _constraints_vec(z, n, iu, ju),
                0.0,
                np.inf,
                jac=lambda z: _constraints_jac(z, n, iu, ju, sparse=True),
            )
            res = spo.minimize(
                fun=lambda z: _objective(z, n),
                x0=z0,
                method="trust-constr",
                jac=lambda z: _objective_grad(z, n),
                constraints=[nlc],
                bounds=bounds,
                options=dict(maxiter=maxiter, verbose=0, xtol=1e-12, gtol=1e-12, barrier_tol=1e-14),
            )
            if res.success:
                return res.x
        except Exception:
            pass
        # Fallback to SLSQP with scalar constraints and analytic Jacobians
        try:
            cons = []
            # Boundary constraints as scalars with Jacobians
            def mk_edge_fun(idx, kind):
                # kind: 0:x - r; 1:w-x - r; 2:y - r; 3:h-y - r
                def fun(z):
                    w, xs, ys, rs = _unpack(z, n)
                    h = 2 - w
                    if kind == 0:
                        return xs[idx] - rs[idx]
                    elif kind == 1:
                        return (w - xs[idx]) - rs[idx]
                    elif kind == 2:
                        return ys[idx] - rs[idx]
                    else:
                        return (h - ys[idx]) - rs[idx]
                return fun

            def mk_edge_jac(idx, kind):
                def jac(z):
                    g = np.zeros(1 + 3 * n, dtype=float)
                    xi = 1 + idx
                    yi = 1 + n + idx
                    ri = 1 + 2 * n + idx
                    if kind == 0:
                        g[xi] = 1.0
                        g[ri] = -1.0
                    elif kind == 1:
                        g[0] = 1.0
                        g[xi] = -1.0
                        g[ri] = -1.0
                    elif kind == 2:
                        g[yi] = 1.0
                        g[ri] = -1.0
                    else:
                        g[0] = -1.0
                        g[yi] = -1.0
                        g[ri] = -1.0
                    return g
                return jac

            for i in range(n):
                cons.append(dict(type="ineq", fun=mk_edge_fun(i, 0), jac=mk_edge_jac(i, 0)))
                cons.append(dict(type="ineq", fun=mk_edge_fun(i, 1), jac=mk_edge_jac(i, 1)))
                cons.append(dict(type="ineq", fun=mk_edge_fun(i, 2), jac=mk_edge_jac(i, 2)))
                cons.append(dict(type="ineq", fun=mk_edge_fun(i, 3), jac=mk_edge_jac(i, 3)))

            # Pairwise constraints with Jacobians
            def mk_pair_fun(i, j):
                def fun(z):
                    _, xs, ys, rs = _unpack(z, n)
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    return (dx * dx + dy * dy) - (rs[i] + rs[j]) ** 2
                return fun

            def mk_pair_jac(i, j):
                def jac(z):
                    _, xs, ys, rs = _unpack(z, n)
                    g = np.zeros(1 + 3 * n, dtype=float)
                    xi = 1 + i
                    xj = 1 + j
                    yi = 1 + n + i
                    yj = 1 + n + j
                    ri = 1 + 2 * n + i
                    rj = 1 + 2 * n + j
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    g[xi] = 2.0 * dx
                    g[xj] = -2.0 * dx
                    g[yi] = 2.0 * dy
                    g[yj] = -2.0 * dy
                    s2 = -2.0 * (rs[i] + rs[j])
                    g[ri] = s2
                    g[rj] = s2
                    return g
                return jac

            for a, b in zip(iu, ju):
                ia, ib = int(a), int(b)
                cons.append(dict(type="ineq", fun=mk_pair_fun(ia, ib), jac=mk_pair_jac(ia, ib)))

            res2 = spo.minimize(
                fun=lambda z: _objective(z, n),
                x0=z0,
                method="SLSQP",
                jac=lambda z: _objective_grad(z, n),
                bounds=_bounds(n),
                constraints=cons,
                options=dict(maxiter=maxiter, ftol=1e-12, eps=1e-8, disp=False),
            )
            if res2.success:
                return res2.x
        except Exception:
            return None
    return None


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 to maximize sum of radii.

    Returns:
        circles: np.array of shape (21,3), each row as (x, y, r).
    """
    rng = np.random.default_rng(1337)
    n = 21
    iu, ju = _pair_indices(n)

    # Deterministic width sweep aligned with top entries (robust endpoints included)
    width_list = [0.75, 0.82, 0.90, 0.96, 1.00, 1.04, 1.10, 1.16, 1.22, 1.26, 1.30, 1.34]
    patterns = ["hex", "diamond", "grid"]

    best_sum = -1.0
    best_z: Optional[np.ndarray] = None

    for w0 in width_list:
        # Clip w0 defensively to keep h well-conditioned
        w0 = float(np.clip(w0, 0.05, 1.95))
        h0 = 2.0 - w0
        for pat in patterns:
            if pat == "hex":
                xs0, ys0 = _hex21_centers(w0, h0)
            elif pat == "diamond":
                xs0, ys0 = _diamond21_centers(w0, h0)
            else:
                xs0, ys0 = _grid_centers_7x3(w0, h0)
            # Small deterministic jitter to break symmetries
            xs0 = np.clip(xs0 + (rng.uniform(-0.01, 0.01, size=n) * w0), 0.02 * w0, 0.98 * w0)
            ys0 = np.clip(ys0 + (rng.uniform(-0.01, 0.01, size=n) * h0), 0.02 * h0, 0.98 * h0)

            # Baseline: LP radii (HiGHS) then strict repair + micro-inflate
            r0 = _lp_radii_given_centers(w0, h0, xs0, ys0, eps=1e-11)
            z0 = _z_from_whxr(w0, xs0, ys0, r0)
            wb, xb, yb, rb = _unpack(z0, n)
            rb = _repair(wb, 2.0 - wb, xb, yb, rb, passes=4)
            z_base = _z_from_whxr(wb, xb, yb, rb)
            base_sum = float(np.sum(rb))
            base_viol = _feas_viol(z_base, n)

            # NLP refinement from the same seed
            z_opt = _optimize_once(z0, n, iu, ju, maxiter=400)
            if z_opt is None:
                z_opt = z0.copy()
            w1, x1, y1, r1 = _unpack(z_opt, n)
            h1 = 2.0 - w1
            # Post-solver center clipping before repair
            x1 = np.clip(x1, 0.0, w1)
            y1 = np.clip(y1, 0.0, h1)
            r1 = _repair(w1, h1, x1, y1, r1, passes=4)
            z_nlp = _z_from_whxr(w1, x1, y1, r1)
            nlp_sum = float(np.sum(r1))
            nlp_viol = _feas_viol(z_nlp, n)

            # Keep-best-of-two with feasibility guard
            if (base_viol <= 1e-9) and (nlp_viol <= 1e-9):
                z_cand = z_nlp if (nlp_sum > base_sum + 1e-12) else z_base
                cand_viol = nlp_viol if (nlp_sum > base_sum + 1e-12) else base_viol
            else:
                if nlp_viol + 1e-15 < base_viol:
                    z_cand, cand_viol = z_nlp, nlp_viol
                elif base_viol + 1e-15 < nlp_viol:
                    z_cand, cand_viol = z_base, base_viol
                else:
                    # tie-break by sum if violations are similar
                    z_cand = z_nlp if (nlp_sum > base_sum + 1e-12) else z_base
                    cand_viol = _feas_viol(z_cand, n)

            # Post-NLP repolish: LP radii -> micro-inflate -> short trust-constr (accept with strict guard)
            wk, xk, yk, rk = _unpack(z_cand, n)
            hk = 2.0 - wk
            xk = np.clip(xk, 0.0, wk)
            yk = np.clip(yk, 0.0, hk)
            r_lp = _lp_radii_given_centers(wk, hk, xk, yk, eps=1e-11)
            r_pol = _micro_inflate(wk, hk, xk, yk, r_lp, iters=3, step=0.25)
            r_pol = _micro_inflate(wk, hk, xk, yk, r_pol, iters=3, step=0.12)
            r_pol = _micro_inflate(wk, hk, xk, yk, r_pol, iters=2, step=0.08)
            z_lp = _z_from_whxr(wk, xk, yk, r_pol)
            z_try = _optimize_once(z_lp, n, iu, ju, maxiter=120)
            if z_try is None:
                z_try = z_lp
            wt, xt, yt, rt = _unpack(z_try, n)
            ht = 2.0 - wt
            # Clip centers and final repair
            xt = np.clip(xt, 0.0, wt)
            yt = np.clip(yt, 0.0, ht)
            rt = _repair(wt, ht, xt, yt, rt, passes=3)
            z_polished = _z_from_whxr(wt, xt, yt, rt)
            pol_sum = float(np.sum(rt))
            pol_viol = _feas_viol(z_polished, n)

            # Strict regression guard for polish
            z_final = z_cand
            if (pol_sum > np.sum(_unpack(z_cand, n)[3]) + 1e-12) and (pol_viol <= cand_viol + 1e-13):
                z_final = z_polished

            ssum = _feasible_score(z_final, n)
            if ssum > best_sum:
                best_sum = ssum
                best_z = z_final

    # Final safeguard: if all failed, produce a minimal feasible with tiny radii at grid
    if best_z is None or best_sum <= 0:
        w0 = 1.0
        h0 = 1.0
        xs0, ys0 = _grid_centers_7x3(w0, h0)
        rs0 = np.full(n, 1e-3)
        best_z = _z_from_whxr(w0, xs0, ys0, rs0)

    # Return circles (x, y, r)
    w, xs, ys, rs = _unpack(best_z, n)
    circles = np.stack([xs, ys, rs], axis=1)
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")