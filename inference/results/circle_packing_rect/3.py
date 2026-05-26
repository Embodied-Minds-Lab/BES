# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Deterministic hybrid LP–SLSQP packing of 21 circles inside a rectangle of perimeter 4 (w + h = 2),
    maximizing the sum of radii with strict feasibility. Uses:
      - Deterministic hex-like multistart initialization with horizontal and vertical phases
      - Single stacked SLSQP inequality with vectorized dense Jacobian in block layout
      - LP polish with wall caps as variable bounds and only pairwise A_ub
      - Deterministic tie-break by min global slack plus final guarded feasibility pass
    Returns:
        circles: np.ndarray of shape (21,3), rows are (x, y, r)
    """
    # Problem constants
    n = 21
    assert n == 21
    eps = 1e-9
    eps_opt = 1e-6  # tightening for constraints and LP caps
    # Deterministic aspect sweep
    height_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    # Required hex-like staggered layout: 5 rows, counts sum to 21
    row_counts = [4, 5, 4, 4, 4]
    assert sum(row_counts) == n

    # Optional scipy imports; graceful fallback if not available
    try:
        from scipy.optimize import minimize, linprog
        SCIPY_AVAILABLE = True
    except Exception:
        minimize = None
        linprog = None
        SCIPY_AVAILABLE = False

    # Pairwise index arrays (upper-triangle, i<j), reused across all calls
    I_pairs, J_pairs = np.triu_indices(n, k=1)

    # -------- Utility functions --------
    def micro_shrink_r(r: np.ndarray) -> np.ndarray:
        return np.maximum(eps, r * (1.0 - 1e-12))

    def min_slack(c: np.ndarray, w: float, h: float) -> float:
        # Boundary slack: min(x, w-x, y, h-y) - r
        sb = np.minimum.reduce([c[:, 0], w - c[:, 0], c[:, 1], h - c[:, 1]]) - c[:, 2]
        sb_min = float(np.min(sb)) if sb.size else np.inf
        # Pairwise slack: dij - (ri + rj)
        dx = c[I_pairs, 0] - c[J_pairs, 0]
        dy = c[I_pairs, 1] - c[J_pairs, 1]
        dij = np.hypot(dx, dy)
        sij = c[I_pairs, 2] + c[J_pairs, 2]
        sp_min = float(np.min(dij - sij)) if dij.size else np.inf
        return min(sb_min, sp_min)

    def repair_uniform_strict(circles: np.ndarray, w: float, h: float, tighten: float, iters: int = 2) -> np.ndarray:
        """
        Deterministic global uniform-shrink repair to enforce strict feasibility with a given tightening.
        Enforces for all i:
          x_i - r_i - tighten >= 0
          w - x_i - r_i - tighten >= 0
          y_i - r_i - tighten >= 0
          h - y_i - r_i - tighten >= 0
        and for all i<j:
          ||(xi,yi) - (xj,yj)|| >= ri + rj + tighten
        """
        c = circles.copy()
        for _ in range(max(1, iters)):
            # Clamp centers inside the box
            c[:, 0] = np.clip(c[:, 0], eps, w - eps)
            c[:, 1] = np.clip(c[:, 1], eps, h - eps)
            # Boundary-based scaling factor
            wall_clear = np.minimum.reduce([c[:, 0], w - c[:, 0], c[:, 1], h - c[:, 1]])
            sl_b = wall_clear - tighten
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha_b = np.min(sl_b / np.maximum(c[:, 2], eps))
            if not np.isfinite(alpha_b):
                alpha_b = 1.0
            alpha_b = float(np.clip(alpha_b, 0.0, 1.0))

            # Pairwise-based scaling factor
            if I_pairs.size > 0:
                dx = c[I_pairs, 0] - c[J_pairs, 0]
                dy = c[I_pairs, 1] - c[J_pairs, 1]
                dij = np.hypot(dx, dy)
                sij = c[I_pairs, 2] + c[J_pairs, 2]
                with np.errstate(divide='ignore', invalid='ignore'):
                    cand = np.maximum(0.0, dij - tighten) / np.maximum(sij, eps)
                cand = cand[np.isfinite(cand)]
                alpha_o = float(np.min(cand)) if cand.size else 1.0
            else:
                alpha_o = 1.0

            alpha = float(np.clip(min(1.0, alpha_b, alpha_o), 0.0, 1.0))
            if alpha < 1.0 - 1e-16:
                c[:, 2] = np.maximum(eps, alpha * c[:, 2])

        # Tiny safety shrink for strictness
        c[:, 2] = micro_shrink_r(c[:, 2])
        return c

    # Hex-like staggered initialization with horizontal and vertical phases
    def hex_init(w: float, h: float, alpha: float = 0.5, vbeta: float = 0.0, sign_pattern=None) -> np.ndarray:
        margin = 1e-6
        xs_5 = np.linspace(margin, w - margin, 5)
        # 4-count row nodes centered between 5-grid columns shifted by alpha in (0,1)
        xs_4 = xs_5[:-1] + alpha * (xs_5[1:] - xs_5[:-1])
        ys = np.linspace(margin, h - margin, 5)
        dy = ys[1] - ys[0] if len(ys) > 1 else (h - 2 * margin)

        # Conservative initial radius per spec
        dx = xs_5[1] - xs_5[0] if len(xs_5) > 1 else (w - 2 * margin)
        r_same = dx / 2.0 if dx > 0 else eps
        r_stag = 0.5 * np.sqrt((0.5 * dx) ** 2 + dy ** 2) if dx > 0 and dy > 0 else eps
        r_bound = 0.25 * min(w, h)
        r0 = 0.4 * min(r_same, r_stag, r_bound)
        r0 = max(r0, 1e-5 * min(w, h))

        # Deterministic vertical phase for 4-count rows
        if sign_pattern is None:
            sign_pattern = np.array([-1, +1, -1, +1, -1], dtype=float)
        vshift = vbeta * dy
        circles = []
        for row_idx, cnt in enumerate(row_counts):
            ycoord = ys[row_idx]
            if cnt == 4 and vbeta > 0.0:
                ycoord = float(np.clip(ycoord + sign_pattern[row_idx] * vshift, margin, h - margin))
            xs = xs_5 if cnt == 5 else xs_4
            for k in range(cnt):
                circles.append([xs[k], ycoord, r0])
        return np.array(circles, dtype=float)

    # Fixed-center LP: maximize sum(r) with wall caps as bounds; only pairwise in A_ub
    def lp_optimize_radii_fixed_centers(c0: np.ndarray, w: float, h: float, tighten: float) -> np.ndarray | None:
        if not SCIPY_AVAILABLE or linprog is None:
            return None
        x = c0[:, 0].copy()
        y = c0[:, 1].copy()
        nloc = x.size
        # Upper bounds by walls (per-circle caps)
        wall_cap = np.minimum.reduce([x, w - x, y, h - y]) - tighten
        wall_cap = np.maximum(wall_cap, 0.0)
        rmax_cap = 0.5 * min(w, h)
        ub = np.clip(wall_cap, 0.0, rmax_cap)
        bounds = [(eps, float(max(eps, ub[i]))) for i in range(nloc)]

        # Objective: maximize sum r -> minimize -sum r
        c_vec = -np.ones(nloc, dtype=float)

        # Pairwise constraints: ri + rj <= dij - tighten
        I, J = I_pairs, J_pairs
        m_pairs = I.size
        if m_pairs > 0:
            A_ub = np.zeros((m_pairs, nloc), dtype=float)
            rows = np.arange(m_pairs, dtype=int)
            A_ub[rows, I] = 1.0
            A_ub[rows, J] = 1.0
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            dij = np.hypot(dx, dy)
            b_ub = np.maximum(dij - tighten, 0.0)
        else:
            A_ub = None
            b_ub = None

        try:
            res = linprog(c=c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success or res.x is None:
                return None
            r = np.maximum(res.x, eps)
            c = np.stack([x, y, r], axis=1)
            # Post LP: enforce strict feasibility and micro-shrink
            c = repair_uniform_strict(c, w, h, tighten=tighten, iters=1)
            c[:, 2] = micro_shrink_r(c[:, 2])
            return c
        except Exception:
            return None

    # Build SLSQP components with analytic Jacobians (block order: x(0:n)|y(n:2n)|r(2n:3n))
    def make_slsqp_components(w: float, h: float, tighten: float):
        rmax = 0.5 * min(w, h)
        m = I_pairs.size

        def fun(z):
            r = z[2 * n:3 * n]
            return -np.sum(r)

        def jac(z):
            g = np.zeros_like(z)
            g[2 * n:3 * n] = -1.0
            return g

        # Single stacked inequality: [c1|c2|c3|c4|c_pairs] >= 0 with eps_opt tightening
        def ineq_fun(z):
            x = z[0:n]
            y = z[n:2 * n]
            r = z[2 * n:3 * n]
            c = np.empty(4 * n + m, dtype=float)
            # Containment
            c[0:n] = (x - r) - tighten
            c[n:2 * n] = (w - x - r) - tighten
            c[2 * n:3 * n] = (y - r) - tighten
            c[3 * n:4 * n] = (h - y - r) - tighten
            # Pairwise squared distances
            I, J = I_pairs, J_pairs
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            s = r[I] + r[J]
            c[4 * n:] = dx * dx + dy * dy - s * s - tighten
            return c

        def ineq_jac(z):
            x = z[0:n]
            y = z[n:2 * n]
            r = z[2 * n:3 * n]
            J = np.zeros((4 * n + m, 3 * n), dtype=float)
            idx = np.arange(n, dtype=int)
            # Containment rows (vectorized scatter)
            # left: d/dx=+1, d/dr=-1
            J[idx, idx] = 1.0
            J[idx, 2 * n + idx] = -1.0
            # right: d/dx=-1, d/dr=-1
            rows_r = n + idx
            J[rows_r, idx] = -1.0
            J[rows_r, 2 * n + idx] = -1.0
            # bottom: d/dy=+1, d/dr=-1
            rows_b = 2 * n + idx
            J[rows_b, n + idx] = 1.0
            J[rows_b, 2 * n + idx] = -1.0
            # top: d/dy=-1, d/dr=-1
            rows_t = 3 * n + idx
            J[rows_t, n + idx] = -1.0
            J[rows_t, 2 * n + idx] = -1.0

            # Pairwise rows (vectorized)
            if m > 0:
                rows = 4 * n + np.arange(m, dtype=int)
                I, Jp = I_pairs, J_pairs
                dx = x[I] - x[Jp]
                dy = y[I] - y[Jp]
                s = r[I] + r[Jp]
                # d/dx_i =  2*dx, d/dx_j = -2*dx
                J[rows, I] += 2.0 * dx
                J[rows, Jp] += -2.0 * dx
                # d/dy_i =  2*dy, d/dy_j = -2*dy
                J[rows, n + I] += 2.0 * dy
                J[rows, n + Jp] += -2.0 * dy
                # d/dr_i = -2*(ri+rj), d/dr_j = -2*(ri+rj)
                J[rows, 2 * n + I] += -2.0 * s
                J[rows, 2 * n + Jp] += -2.0 * s
            return J

        bounds = []
        for _ in range(n):
            bounds.append((0.0, w))
        for _ in range(n):
            bounds.append((0.0, h))
        for _ in range(n):
            bounds.append((eps, rmax))

        constraints = [{'type': 'ineq', 'fun': ineq_fun, 'jac': ineq_jac}]
        return fun, jac, bounds, constraints

    def slsqp_stage(c0: np.ndarray, w: float, h: float, tighten: float, maxiter: int, ftol: float) -> np.ndarray | None:
        if not SCIPY_AVAILABLE or minimize is None:
            return None
        # Pre-solver gating: repair + micro-shrink
        c0 = repair_uniform_strict(c0, w, h, tighten=tighten, iters=1)
        c0[:, 2] = micro_shrink_r(c0[:, 2])

        fun, jac, bnds, constraints = make_slsqp_components(w, h, tighten)
        z0 = np.empty(3 * n, dtype=float)
        z0[0:n] = c0[:, 0]
        z0[n:2 * n] = c0[:, 1]
        z0[2 * n:3 * n] = c0[:, 2]
        try:
            res = minimize(fun, z0, method='SLSQP', jac=jac, bounds=bnds,
                           constraints=constraints,
                           options=dict(maxiter=int(maxiter), ftol=float(ftol), disp=False))
            if res is None or res.x is None:
                return None
            z = res.x
            c = np.stack([z[0:n], z[n:2 * n], z[2 * n:3 * n]], axis=1)
            # Post-solver gating: repair + micro-shrink
            c = repair_uniform_strict(c, w, h, tighten=tighten, iters=1)
            c[:, 2] = micro_shrink_r(c[:, 2])
            return c
        except Exception:
            return None

    # -------- Phased multistart + 2-pass LP–SLSQP pipeline per aspect --------
    def run_pipeline_for_aspect(h: float) -> np.ndarray:
        w = 2.0 - h
        # 0) Deterministic phased hex-seed pool: alphas x betas x signs with quick repair+LP
        pre_tight = 2e-4
        alphas = [0.43, 0.50, 0.57]
        betas = [0.0, 0.20, 0.40]
        sign_pattern = np.array([-1, +1, -1, +1, -1], dtype=float)
        seeds = []
        for a in alphas:
            for b in betas:
                # +/- vertical phase realized via sign_pattern only; b=0 means no vertical shift
                seed = hex_init(w, h, alpha=a, vbeta=b, sign_pattern=sign_pattern)
                seed = repair_uniform_strict(seed, w, h, tighten=pre_tight, iters=2)
                lp_seed = lp_optimize_radii_fixed_centers(seed, w, h, tighten=pre_tight)
                if lp_seed is not None:
                    seed = lp_seed
                seed[:, 2] = micro_shrink_r(seed[:, 2])
                s_val = float(np.sum(seed[:, 2]))
                sl_val = min_slack(seed, w, h)
                seeds.append((s_val, sl_val, seed))
        # Keep top-2 seeds by sum, tiebreak by slack
        seeds.sort(key=lambda t: (t[0], t[1]))
        top_seeds = [seeds[-1][2]]
        if len(seeds) > 1:
            top_seeds.append(seeds[-2][2])

        best_cand = None
        best_sum = -np.inf
        best_sl = -np.inf

        # For each top seed, run deeper pipeline
        for cand in top_seeds:
            # Pass 1: SLSQP -> repair -> LP
            sls1 = slsqp_stage(cand, w, h, tighten=eps_opt, maxiter=900, ftol=1e-12)
            if sls1 is not None:
                cand = sls1
            cand = repair_uniform_strict(cand, w, h, tighten=eps_opt, iters=1)
            lp1 = lp_optimize_radii_fixed_centers(cand, w, h, tighten=eps_opt)
            if lp1 is not None:
                cand = lp1
            cand[:, 2] = micro_shrink_r(cand[:, 2])

            # Pass 2: SLSQP -> repair -> LP
            sls2 = slsqp_stage(cand, w, h, tighten=eps_opt, maxiter=1200, ftol=1e-12)
            if sls2 is not None:
                cand = sls2
            cand = repair_uniform_strict(cand, w, h, tighten=eps_opt, iters=1)
            lp2 = lp_optimize_radii_fixed_centers(cand, w, h, tighten=eps_opt)
            if lp2 is not None:
                cand = lp2
            cand[:, 2] = micro_shrink_r(cand[:, 2])

            # Evaluate
            s_val = float(np.sum(cand[:, 2]))
            sl_val = min_slack(cand, w, h)
            if (s_val > best_sum) or (np.isclose(s_val, best_sum, atol=1e-12) and sl_val > best_sl):
                best_sum = s_val
                best_sl = sl_val
                best_cand = cand

        # Final safety repair with tiny tighten and micro-shrink
        if best_cand is None:
            best_cand = top_seeds[0]
        best_cand = repair_uniform_strict(best_cand, w, h, tighten=1e-9, iters=2)
        best_cand[:, 2] = micro_shrink_r(best_cand[:, 2])
        return best_cand

    # -------- Deterministic aspect sweep with tie-breaker --------
    best_circles = None
    best_sum_r = -np.inf
    best_slack = -np.inf

    for h in height_candidates:
        w = 2.0 - h
        if w <= 0 or h <= 0:
            continue
        candidate = run_pipeline_for_aspect(h)
        # Final ensure strict feasibility and micro-shrink
        candidate = repair_uniform_strict(candidate, w, h, tighten=1e-9, iters=2)
        candidate[:, 2] = micro_shrink_r(candidate[:, 2])

        s = float(np.sum(candidate[:, 2]))
        sl = min_slack(candidate, w, h)
        if (s > best_sum_r) or (np.isclose(s, best_sum_r, atol=1e-12) and sl > best_slack):
            best_sum_r = s
            best_slack = sl
            best_circles = candidate.copy()

    # Fallback (no SciPy or unexpected failure): simple feasible hex seed in a square
    if best_circles is None or best_circles.shape != (n, 3):
        w = h = 1.0
        base = hex_init(w, h, alpha=0.50)
        base = repair_uniform_strict(base, w, h, tighten=1e-6, iters=3)
        base[:, 2] = micro_shrink_r(base[:, 2])
        best_circles = base

    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")