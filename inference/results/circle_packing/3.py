# EVOLVE-BLOCK-START
"""
SLP trust-region archipelago for n=26 circle packing in unit square.

Core idea:
- Replace force annealing with Sequential Linear Programming (SLP) in center space.
- Linearize pairwise distances around current centers; solve LP for center displacements and radii within a box trust region.
- Activate only near-contact constraints, validate every accepted move via exact radii LP to guarantee feasibility.
- Deterministic multi-seed archipelago with Halton diversification and periodic migration.
- Coordinate-poll fallback when SLP LP is unavailable.

Public API:
- construct_packing() -> (centers, radii) with shapes (26,2) and (26,)
"""

import numpy as np

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------- Utilities -----------------------------
def _clip_unit(centers, eps):
    return np.clip(centers, eps, 1.0 - eps)


def _pairwise_dist(a, b):
    return float(np.linalg.norm(a - b))


def _safe_unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


# ----------------------------- Halton Sampler -----------------------------
def _van_der_corput_single(index, base):
    x = index
    f = 1.0
    r = 0.0
    while x > 0:
        f /= base
        r += f * (x % base)
        x //= base
    return r


class HaltonSampler:
    def __init__(self, bases=(2, 5), start_index=1):
        self.bases = bases
        self.idx = start_index

    def next(self, dim=2):
        vals = []
        for d in range(dim):
            base = self.bases[d % len(self.bases)]
            vals.append(_van_der_corput_single(self.idx, base))
        self.idx += 1
        return np.array(vals, dtype=float)

    def batch(self, m, dim=2):
        out = np.zeros((m, dim), dtype=float)
        for i in range(m):
            out[i] = self.next(dim=dim)
        return out


# ----------------------------- LP Oracle for Radii -----------------------------
class LPOracle:
    def __init__(self, eps_default=1e-3):
        self.eps_default = eps_default

    def solve(self, centers, eps=None):
        if eps is None:
            eps = self.eps_default
        if _HAS_SCIPY:
            try:
                return self._solve_lp_scipy(centers, eps)
            except Exception:
                return self._solve_fallback(centers, eps)
        else:
            return self._solve_fallback(centers, eps)

    @staticmethod
    def _solve_lp_scipy(centers, eps):
        n = centers.shape[0]
        x = centers[:, 0]
        y = centers[:, 1]
        b = np.minimum.reduce([x, y, 1.0 - x, 1.0 - y]) - eps
        b = np.clip(b, 0.0, None)

        c = -np.ones(n)
        A_rows = []
        b_ub = []

        # r_i <= b_i
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_ub.append(b[i])

        # r_i + r_j <= d_ij - eps
        for i in range(n):
            ci = centers[i]
            for j in range(i + 1, n):
                d = float(np.linalg.norm(ci - centers[j]))
                rhs = max(0.0, d - eps)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_ub.append(rhs)

        A_ub = np.vstack(A_rows) if A_rows else None
        b_ub = np.array(b_ub) if b_ub else None
        bounds = [(0.0, None) for _ in range(n)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            return LPOracle._solve_fallback(centers, eps)
        r = np.clip(res.x, 0.0, b)
        return r

    @staticmethod
    def _solve_fallback(centers, eps, max_iter=700, tol=1e-11):
        """
        Projection-based feasible radii with conservative greedy ascent.
        Always returns feasible radii for given centers and eps.
        """
        n = centers.shape[0]
        x = centers[:, 0]
        y = centers[:, 1]
        b = np.minimum.reduce([x, y, 1.0 - x, 1.0 - y]) - eps
        b = np.clip(b, 0.0, None)
        r = b.copy()

        # Projection loop to remove violations
        for _ in range(max_iter):
            violations = 0
            # Borders
            r = np.minimum(r, b)
            # Pairs
            for i in range(n):
                ci = centers[i]
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(ci - centers[j]))
                    lim = max(0.0, d - eps)
                    s = r[i] + r[j] - lim
                    if s > tol:
                        violations += 1
                        S = r[i] + r[j] + 1e-16
                        r[i] -= s * (r[i] / S)
                        r[j] -= s * (r[j] / S)
                        r[i] = max(0.0, min(r[i], b[i]))
                        r[j] = max(0.0, min(r[j], b[j]))
            if violations == 0:
                break

        # Conservative greedy ascent with 90% slack increments
        for _ in range(5 * n):
            inc = np.zeros(n)
            for i in range(n):
                lim = b[i] - r[i]
                if lim <= 0:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    d = float(np.linalg.norm(centers[i] - centers[j]))
                    lim = min(lim, max(0.0, d - eps) - r[i] - r[j])
                    if lim <= 0:
                        break
                inc[i] = max(0.0, lim)
            k = int(np.argmax(inc))
            if inc[k] <= tol:
                break
            r[k] = min(b[k], r[k] + 0.9 * inc[k])

        return np.clip(r, 0.0, b)


# ----------------------------- Seeds -----------------------------
def _seed_hex_like(n, margin=0.06):
    counts = [5, 5, 5, 5, 6]
    while sum(counts) < n: counts.append(5)
    while sum(counts) > n:
        idx = int(np.argmax(counts))
        counts[idx] = max(1, counts[idx] - 1)
    rows = len(counts)
    ys = np.linspace(margin, 1.0 - margin, rows)
    pts = []
    for r, cnt in enumerate(counts):
        xs = np.linspace(margin, 1.0 - margin, cnt)
        if r % 2 == 1:
            xs += 0.5 * (1.0 - 2 * margin) / max(1, cnt)
        xs = np.clip(xs, margin, 1.0 - margin)
        for x in xs:
            pts.append([x, ys[r]])
    pts = np.array(pts, dtype=float)[:n]
    if pts.shape[0] < n:
        rng = np.random.default_rng(2026)
        add = rng.uniform(low=margin, high=1.0 - margin, size=(n - pts.shape[0], 2))
        pts = np.vstack([pts, add])
    return np.clip(pts, margin, 1.0 - margin)


def _seed_rotated_hex(n, margin=0.06, angle_deg=30.0):
    pts = _seed_hex_like(n, margin=margin).copy()
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    c = np.array([0.5, 0.5])
    P = (pts - c) @ R.T + c
    return np.clip(P, margin, 1.0 - margin)


def _seed_edge_biased(n, margin=0.06):
    pts = []
    for cx in [margin, 1.0 - margin]:
        for cy in [margin, 1.0 - margin]:
            pts.append([cx, cy])
    for t in np.linspace(margin, 1.0 - margin, 4)[1:3]:
        pts += [[t, margin], [t, 1.0 - margin], [margin, t], [1.0 - margin, t]]
    need = n - len(pts)
    rows = int(np.ceil(np.sqrt(need)))
    cols = int(np.ceil(need / max(1, rows)))
    xs = np.linspace(margin + 0.08, 1.0 - margin - 0.08, max(1, cols))
    ys = np.linspace(margin + 0.08, 1.0 - margin - 0.08, max(1, rows))
    for yi in ys:
        for xi in xs:
            if len(pts) >= n: break
            pts.append([xi, yi])
        if len(pts) >= n: break
    return np.array(pts, dtype=float)


def _seed_two_belts(n, margin=0.06, width=0.055):
    pts = []
    m = max(6, int(np.ceil(n / 6)))
    ts = np.linspace(margin, 1.0 - margin, m)
    for off in [-width, 0.0, width]:
        for t in ts:
            x = t
            y = np.clip(t + off, margin, 1.0 - margin)
            pts.append([x, y])
    for off in [-width, 0.0, width]:
        for t in ts:
            x = t
            y = np.clip(1.0 - t + off, margin, 1.0 - margin)
            pts.append([x, y])
    for cx in [margin, 1.0 - margin]:
        for cy in [margin, 1.0 - margin]:
            pts.append([cx, cy])
    pts = np.array(pts, dtype=float)
    if pts.shape[0] > n:
        idx = np.linspace(0, pts.shape[0] - 1, n).astype(int)
        pts = pts[idx]
    return np.clip(pts, margin, 1.0 - margin)


def _seed_ring_center(n, r1=0.23, r2=0.46, margin=0.06):
    pts = []
    pts.append([0.5, 0.5])
    for i in range(6):
        ang = 2 * np.pi * i / 6.0
        pts.append([0.5 + r1 * np.cos(ang), 0.5 + r1 * np.sin(ang)])
    for i in range(12):
        ang = 2 * np.pi * i / 12.0
        pts.append([0.5 + r2 * np.cos(ang), 0.5 + r2 * np.sin(ang)])
    while len(pts) < n:
        for t in np.linspace(margin, 1.0 - margin, 6):
            if len(pts) < n: pts.append([t, margin])
            if len(pts) < n: pts.append([t, 1.0 - margin])
            if len(pts) < n: pts.append([margin, t])
            if len(pts) < n: pts.append([1.0 - margin, t])
            if len(pts) >= n: break
    pts = np.array(pts, dtype=float)[:n]
    return np.clip(pts, margin, 1.0 - margin)


def _seed_corner_star(n, margin=0.06, r1=0.24):
    pts = []
    for cx in [margin, 1.0 - margin]:
        for cy in [margin, 1.0 - margin]:
            pts.append([cx, cy])
    for t in [0.25, 0.5, 0.75]:
        pts += [[t, margin], [t, 1.0 - margin], [margin, t], [1.0 - margin, t]]
    if len(pts) < n:
        pts.append([0.5, 0.5])
    for i in range(6):
        if len(pts) >= n: break
        ang = 2 * np.pi * i / 6.0
        pts.append([0.5 + r1 * np.cos(ang), 0.5 + r1 * np.sin(ang)])
    diag = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
    d = r1 * 1.35
    for sx, sy in diag:
        if len(pts) >= n: break
        pts.append([0.5 + sx * d / np.sqrt(2.0), 0.5 + sy * d / np.sqrt(2.0)])
    pts = np.array(pts, dtype=float)[:n]
    return np.clip(pts, margin, 1.0 - margin)


# ----------------------------- Schedules -----------------------------
class Schedules:
    def __init__(self, tau=70.0, tr_init=0.06, tr_max=0.08, tr_min=0.002,
                 lp_eps_start=0.012, lp_eps_end=0.0015):
        self.tau = tau
        self.tr_init = tr_init
        self.tr_max = tr_max
        self.tr_min = tr_min
        self.lp_eps_start = lp_eps_start
        self.lp_eps_end = lp_eps_end

    def lp_eps(self, t):
        return self.lp_eps_end + (self.lp_eps_start - self.lp_eps_end) * np.exp(-t / (0.7 * self.tau))

    def slack_threshold(self, t):
        # Active-pair slack cutoff schedule
        # Start permissive and tighten gradually
        start = 0.065
        end = 0.015
        return end + (start - end) * np.exp(-t / max(1.0, self.tau * 0.9))


# ----------------------------- SLP Builder -----------------------------
class SLPBuilder:
    """
    Build and solve the SLP (Sequential Linear Program) step:
    Variables per circle: (dx_i, dy_i, r_i). Objective: maximize sum r_i.
    Constraints:
      - Trust region and interior box for dx_i, dy_i via bounds.
      - Wall constraints: r_i <= x_i + dx_i - eps; r_i <= 1 - x_i - dx_i - eps; similarly for y.
      - Pairwise linearized: r_i + r_j <= d_ij - eps + u_ij^T (delta_i - delta_j).
    If SciPy unavailable, return None to trigger fallback.
    """
    def __init__(self):
        pass

    def solve(self, centers, radii, eps, tr, active_pairs, margin_eps):
        if not _HAS_SCIPY:
            return None
        n = centers.shape[0]
        m = 3 * n
        # Variable ordering: for i in 0..n-1 -> [dx_i, dy_i, r_i]
        def vidx_dx(i): return 3 * i + 0
        def vidx_dy(i): return 3 * i + 1
        def vidx_r(i):  return 3 * i + 2

        c = np.zeros(m, dtype=float)
        for i in range(n):
            c[vidx_r(i)] = -1.0  # maximize sum r -> minimize -sum r

        A = []
        b = []

        x = centers[:, 0]
        y = centers[:, 1]

        # Wall constraints (exact geometry)
        for i in range(n):
            # r_i <= x_i + dx_i - eps -> r_i - dx_i <= x_i - eps
            row = np.zeros(m)
            row[vidx_r(i)] = 1.0
            row[vidx_dx(i)] = -1.0
            A.append(row); b.append(x[i] - eps)

            # r_i <= 1 - x_i - dx_i - eps -> r_i + dx_i <= 1 - x_i - eps
            row = np.zeros(m)
            row[vidx_r(i)] = 1.0
            row[vidx_dx(i)] = 1.0
            A.append(row); b.append(1.0 - x[i] - eps)

            # r_i <= y_i + dy_i - eps -> r_i - dy_i <= y_i - eps
            row = np.zeros(m)
            row[vidx_r(i)] = 1.0
            row[vidx_dy(i)] = -1.0
            A.append(row); b.append(y[i] - eps)

            # r_i <= 1 - y_i - dy_i - eps -> r_i + dy_i <= 1 - y_i - eps
            row = np.zeros(m)
            row[vidx_r(i)] = 1.0
            row[vidx_dy(i)] = 1.0
            A.append(row); b.append(1.0 - y[i] - eps)

        # Pairwise linearized constraints for active pairs
        for (i, j, u, d_ij) in active_pairs:
            row = np.zeros(m)
            # r_i + r_j - u·delta_i + u·delta_j <= d - eps
            row[vidx_r(i)] += 1.0
            row[vidx_r(j)] += 1.0
            row[vidx_dx(i)] += -u[0]
            row[vidx_dy(i)] += -u[1]
            row[vidx_dx(j)] += +u[0]
            row[vidx_dy(j)] += +u[1]
            A.append(row); b.append(max(0.0, d_ij - eps))

        A_ub = np.vstack(A) if A else None
        b_ub = np.array(b, dtype=float) if b else None

        # Anisotropic per-circle trust regions guided by active contacts and near walls
        bx_acc = np.zeros(n, dtype=float)  # accumulate |u_x|
        by_acc = np.zeros(n, dtype=float)  # accumulate |u_y|
        cntx = np.zeros(n, dtype=float)
        cnty = np.zeros(n, dtype=float)

        for (i, j, u, _) in active_pairs:
            ax = abs(u[0]); ay = abs(u[1])
            bx_acc[i] += ax; bx_acc[j] += ax
            by_acc[i] += ay; by_acc[j] += ay
            cntx[i] += 1.0; cntx[j] += 1.0
            cnty[i] += 1.0; cnty[j] += 1.0

        # Walls contribute axis blocking when slack small (<= 2*eps)
        for i in range(n):
            ri = radii[i]
            sL = x[i] - eps - ri
            sR = 1.0 - x[i] - eps - ri
            sB = y[i] - eps - ri
            sT = 1.0 - y[i] - eps - ri
            if (sL <= 2.0 * eps) or (sR <= 2.0 * eps):
                bx_acc[i] += 1.0
                cntx[i] += 1.0
            if (sB <= 2.0 * eps) or (sT <= 2.0 * eps):
                by_acc[i] += 1.0
                cnty[i] += 1.0

        bx_avg = np.where(cntx > 0.0, bx_acc / np.maximum(1.0, cntx), 0.5)
        by_avg = np.where(cnty > 0.0, by_acc / np.maximum(1.0, cnty), 0.5)

        tr_x = tr * (1.0 + 0.8 * (1.0 - bx_avg))
        tr_y = tr * (1.0 + 0.8 * (1.0 - by_avg))
        tr_x = np.clip(tr_x, 0.5 * tr, 1.8 * tr)
        tr_y = np.clip(tr_y, 0.5 * tr, 1.8 * tr)

        # Bounds for variables: anisotropic trust region + interior box
        bounds = []
        for i in range(n):
            lo_dx = max(-float(tr_x[i]), (margin_eps - x[i]))
            hi_dx = min(+float(tr_x[i]), (1.0 - margin_eps - x[i]))
            lo_dy = max(-float(tr_y[i]), (margin_eps - y[i]))
            hi_dy = min(+float(tr_y[i]), (1.0 - margin_eps - y[i]))
            bounds.append((lo_dx, hi_dx))    # dx_i
            bounds.append((lo_dy, hi_dy))    # dy_i
            bounds.append((0.0, None))       # r_i >= 0

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            return None

        sol = res.x
        deltas = np.zeros_like(centers)
        r_lin = np.zeros(n, dtype=float)
        for i in range(n):
            deltas[i, 0] = sol[vidx_dx(i)]
            deltas[i, 1] = sol[vidx_dy(i)]
            r_lin[i] = sol[vidx_r(i)]
        return deltas, r_lin


# ----------------------------- Active Set Builder -----------------------------
def _build_active_pairs(centers, radii, eps, slack_cut, k_per=8):
    """
    Select a focused active set of near-contact pairs:
    - all pairs with slack <= slack_cut
    - ensure each circle gets its k_per tightest pairs included
    Returns list of tuples (i, j, u_ij, d_ij)
    """
    n = centers.shape[0]
    slacks = []
    per_circle = [[] for _ in range(n)]
    for i in range(n):
        ci = centers[i]
        ri = radii[i]
        for j in range(i + 1, n):
            cj = centers[j]
            d = float(np.linalg.norm(ci - cj))
            u = _safe_unit(ci - cj)
            s = d - eps - ri - radii[j]
            slacks.append((s, i, j, u, d))
            per_circle[i].append((s, j, u, d))
            per_circle[j].append((s, i, -u, d))  # reverse consistent u

    # Global near-contact selection
    active = set()
    for (s, i, j, u, d) in slacks:
        if s <= slack_cut:
            active.add((min(i, j), max(i, j)))

    # Per-circle top-k
    for i in range(n):
        per_circle[i].sort(key=lambda t: t[0])
        for t in per_circle[i][:k_per]:
            j = t[1]
            active.add((min(i, j), max(i, j)))

    # Assemble final list with consistent u (computed fresh)
    out = []
    for (i, j) in active:
        v = centers[i] - centers[j]
        d = float(np.linalg.norm(v))
        u = _safe_unit(v)
        out.append((i, j, u, d))
    return out


# ----------------------------- Coordinate Poll Fallback -----------------------------
def _coordinate_poll(centers, radii, eps, lp_oracle, step, k_focus=10, margin_eps=0.008):
    """
    Deterministic coordinate poll search:
    - Focus on k_focus circles with smallest pairwise slack.
    - Try 8 compass directions of magnitude step for each focused circle (one at a time).
    - Evaluate each candidate via LP oracle, keep the best non-decreasing candidate.
    """
    n = centers.shape[0]
    # Compute per-circle min slack to others
    mins = np.zeros(n, dtype=float)
    for i in range(n):
        smin = np.inf
        for j in range(n):
            if i == j: continue
            d = float(np.linalg.norm(centers[i] - centers[j]))
            s = d - eps - radii[i] - radii[j]
            smin = min(smin, s)
        # also consider walls
        x, y = centers[i, 0], centers[i, 1]
        wl = x - eps - radii[i]
        wr = (1.0 - x) - eps - radii[i]
        wb = y - eps - radii[i]
        wt = (1.0 - y) - eps - radii[i]
        smin = min(smin, wl, wr, wb, wt)
        mins[i] = smin

    focus_idx = np.argsort(mins)[:max(3, min(k_focus, n))]
    dirs = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ], dtype=float)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]

    best_centers = centers
    best_r = radii
    best_s = float(np.sum(radii))

    for i in focus_idx:
        for dvec in dirs:
            cand = centers.copy()
            cand[i] = centers[i] + step * dvec
            cand = _clip_unit(cand, margin_eps)
            r_try = lp_oracle.solve(cand, eps=eps)
            s_try = float(np.sum(r_try))
            if s_try > best_s + 1e-12:
                best_centers, best_r, best_s = cand, r_try, s_try

    return best_centers, best_r, best_s


# ----------------------------- Edge-Snap Operator -----------------------------
class EdgeSnap:
    """
    Relocate k smallest circles to boundary slots chosen by allowable-radius proxy.
    Permissive acceptance is handled by the caller with LP validation and cooldown.
    """
    def __init__(self, n_slots=64, snap_k=4, margin=0.055):
        self.margin = margin
        self.snap_k = snap_k
        self.slots = self._build_slots(n_slots, margin)

    @staticmethod
    def _unique_rows(a, tol=1e-6):
        if len(a) == 0:
            return a
        b = np.round(a / tol).astype(int)
        _, idx = np.unique(b, axis=0, return_index=True)
        return a[np.sort(idx)]

    def _build_slots(self, n_slots, margin):
        slots = []
        # Corners
        for ax in [margin, 1.0 - margin]:
            for ay in [margin, 1.0 - margin]:
                slots.append([ax, ay])
        # Evenly along edges
        per_edge = max(6, n_slots // 4)
        ts = np.linspace(margin, 1.0 - margin, per_edge)
        for t in ts:
            slots.append([t, margin])
            slots.append([t, 1.0 - margin])
            slots.append([margin, t])
            slots.append([1.0 - margin, t])
        return self._unique_rows(np.array(slots, dtype=float))

    def __call__(self, centers, radii):
        n = centers.shape[0]
        k = min(self.snap_k, n)
        idx_small = np.argsort(radii)[:k]
        used = set()
        new_centers = centers.copy()

        for i in idx_small:
            best_slot = None
            best_score = -1e18
            for s_idx, s in enumerate(self.slots):
                if s_idx in used:
                    continue
                # Approx allowable radius bound at slot s
                b = min(s[0], s[1], 1.0 - s[0], 1.0 - s[1])
                min_pair = b
                for j in range(n):
                    if j == i: continue
                    d = float(np.linalg.norm(s - new_centers[j]))
                    min_pair = min(min_pair, max(0.0, d) - radii[j])
                    if min_pair <= 0:
                        break
                score = min_pair
                if score > best_score:
                    best_score = score
                    best_slot = (s_idx, s)
            if best_slot is not None:
                used.add(best_slot[0])
                new_centers[i] = best_slot[1]
        return new_centers


# ----------------------------- SLP Island -----------------------------
class SLPIsland:
    def __init__(self, centers, lp_oracle, sched, rng, edge_snap, margin_eps=0.008, accept_tol=1e-11):
        self.centers = centers.astype(float).copy()
        self.lp = lp_oracle
        self.sched = sched
        self.rng = rng
        self.edge_snap = edge_snap
        self.margin_eps = margin_eps
        self.accept_tol = accept_tol

        self.radii = None
        self.score = -np.inf

        self.tr = self.sched.tr_init  # trust region
        self.slp = SLPBuilder()

        # Permissive snap cooldown tracker
        self.last_nonimprove_snap = -10**9

        # Deterministic low-discrepancy jitter generator for antithetic twins
        self.halton = HaltonSampler(bases=(2, 5), start_index=1)

    def evaluate(self, eps):
        self.radii = self.lp.solve(self.centers, eps=eps)
        self.score = float(np.sum(self.radii))
        return self.score

    def step(self, t, T_total):
        eps_t = self.sched.lp_eps(t)
        slack_cut = self.sched.slack_threshold(t)

        # Refresh radii at current eps
        self.evaluate(eps_t)

        # Build active pairs with dynamic per-circle coverage (8 -> 12 after 60% of total time)
        k_per = 8 if (t < 0.6 * T_total) else 12
        active_pairs = _build_active_pairs(self.centers, self.radii, eps_t, slack_cut, k_per=k_per)

        accepted = False
        tr_try = self.tr
        max_backtracks = 6

        best_centers = self.centers
        best_radii = self.radii
        best_score = self.score

        n = self.centers.shape[0]

        for _ in range(max_backtracks):
            # Attempt SLP step
            out = self.slp.solve(self.centers, self.radii, eps_t, tr_try, active_pairs, self.margin_eps)
            if out is not None:
                deltas, _ = out
                cand0 = _clip_unit(self.centers + deltas, self.margin_eps)

                # Validate SLP cand and antithetic twins with deterministic Halton jitter
                amp = 0.35 * tr_try
                H = self.halton.batch(n, dim=2)
                delta_j = (H * 2.0 - 1.0) * amp

                cands = [cand0,
                         _clip_unit(cand0 + delta_j, self.margin_eps),
                         _clip_unit(cand0 - delta_j, self.margin_eps)]
                local_best_c = None
                local_best_r = None
                local_best_s = -np.inf
                for C in cands:
                    r_try = self.lp.solve(C, eps=eps_t)
                    s_try = float(np.sum(r_try))
                    if s_try > local_best_s:
                        local_best_s = s_try
                        local_best_c = C
                        local_best_r = r_try

                if local_best_s >= self.score - self.accept_tol:
                    best_centers, best_radii, best_score = local_best_c, local_best_r, local_best_s
                    accepted = True
                    break

            # No SLP success; try coordinate poll fallback at same tr
            cand2, r2, s2 = _coordinate_poll(self.centers, self.radii, eps_t, self.lp, step=tr_try, k_focus=10, margin_eps=self.margin_eps)
            if s2 >= self.score - self.accept_tol:
                best_centers, best_radii, best_score = cand2, r2, s2
                accepted = True
                break

            # Backtrack trust region
            tr_try = max(self.sched.tr_min, tr_try * 0.5)

        # Edge-snap cadence: permissive LP-validated acceptance with cooldown for non-degrading
        if ((t + 1) % 25) == 0:
            snapped = self.edge_snap(best_centers, best_radii)
            snapped = _clip_unit(snapped, self.margin_eps)
            r3 = self.lp.solve(snapped, eps=eps_t)
            s3 = float(np.sum(r3))
            if s3 > best_score + 1e-12:
                best_centers, best_radii, best_score = snapped, r3, s3
            elif s3 >= best_score - 1e-12 and (t - self.last_nonimprove_snap) >= 50:
                best_centers, best_radii, best_score = snapped, r3, s3
                self.last_nonimprove_snap = t

        # Update trust region adaptively (considering snap acceptance too)
        accepted_final = (best_score >= self.score - self.accept_tol)
        if accepted_final:
            gain = best_score - self.score
            if gain > 3e-4:
                self.tr = min(self.sched.tr_max, max(self.tr, tr_try) * 1.12)
            elif gain > 1e-6:
                self.tr = min(self.sched.tr_max, (0.8 * self.tr + 0.2 * tr_try))
            else:
                self.tr = max(self.sched.tr_min, 0.9 * self.tr)
            self.centers, self.radii, self.score = best_centers, best_radii, best_score
        else:
            # No acceptance; shrink trust region slightly
            self.tr = max(self.sched.tr_min, 0.6 * self.tr)

    def polish(self, steps=24):
        """
        Conservative final SLP and coordinate-poll passes with occasional strict edge-snap.
        """
        tr0 = min(self.tr, 0.015)
        self.tr = max(self.sched.tr_min, tr0)
        for k in range(steps):
            eps_t = max(0.0014, self.sched.lp_eps(9999))
            self.evaluate(eps_t)
            active_pairs = _build_active_pairs(self.centers, self.radii, eps_t, slack_cut=0.0105, k_per=12)
            out = self.slp.solve(self.centers, self.radii, eps_t, self.tr, active_pairs, self.margin_eps) if _HAS_SCIPY else None
            improved = False
            if out is not None:
                deltas, _ = out
                cand = _clip_unit(self.centers + deltas, self.margin_eps)
                r_try = self.lp.solve(cand, eps=eps_t)
                s_try = float(np.sum(r_try))
                if s_try > self.score + 1e-12:
                    self.centers, self.radii, self.score = cand, r_try, s_try
                    improved = True
            if not improved:
                cand2, r2, s2 = _coordinate_poll(self.centers, self.radii, eps_t, self.lp, step=self.tr, k_focus=12, margin_eps=self.margin_eps)
                if s2 > self.score + 1e-12:
                    self.centers, self.radii, self.score = cand2, r2, s2
                    improved = True

            # Occasional strict edge-snap attempt
            if ((k + 1) % 8) == 0:
                snapped = self.edge_snap(self.centers, self.radii)
                snapped = _clip_unit(snapped, self.margin_eps)
                r_snap = self.lp.solve(snapped, eps=eps_t)
                s_snap = float(np.sum(r_snap))
                if s_snap > self.score + 1e-12:
                    self.centers, self.radii, self.score = snapped, r_snap, s_snap
                    improved = True

            self.tr = max(self.sched.tr_min, self.tr * (1.05 if improved else 0.8))


# ----------------------------- Archipelago Orchestrator -----------------------------
class Archipelago:
    def __init__(self, n=26, n_islands=14, iters=210, margin_eps=0.008, rng_seed=12345):
        self.n = n
        self.n_islands = n_islands
        self.iters = iters
        self.margin_eps = margin_eps
        self.rng = np.random.default_rng(rng_seed)
        self.halton = HaltonSampler(bases=(2, 5), start_index=1)

        self.lp = LPOracle()
        self.sched = Schedules()
        self.edge_snap = EdgeSnap(n_slots=64, snap_k=4, margin=0.055)

        # Antithetic migration sign
        self._mig_sign = 1

    def _seed_islands(self):
        seeds = []
        # Structured diversity
        seeds.append(_seed_hex_like(self.n, margin=0.08))
        seeds.append(_seed_rotated_hex(self.n, margin=0.075, angle_deg=30.0))
        seeds.append(_seed_edge_biased(self.n, margin=0.06))
        seeds.append(_seed_two_belts(self.n, margin=0.06, width=0.055))
        seeds.append(_seed_ring_center(self.n, r1=0.23, r2=0.46, margin=0.06))
        seeds.append(_seed_corner_star(self.n, margin=0.06, r1=0.24))

        # Halton-jittered variants from a strong base
        base = _seed_hex_like(self.n, margin=0.08)
        need = max(0, self.n_islands - len(seeds))
        if need > 0:
            H = self.halton.batch(need, dim=2)
            for i in range(need):
                j = (H[i] * 2.0 - 1.0) * 0.014
                s = _clip_unit(base + j, self.margin_eps)
                seeds.append(s)
        return seeds[:self.n_islands]

    def _instantiate_islands(self, seeds):
        islands = []
        for s in seeds:
            isl = SLPIsland(s, self.lp, self.sched, self.rng, self.edge_snap, margin_eps=self.margin_eps, accept_tol=1e-11)
            isl.evaluate(self.sched.lp_eps(0))
            islands.append(isl)
        return islands

    def evolve(self):
        seeds = self._seed_islands()
        islands = self._instantiate_islands(seeds)

        migrate_K = 28
        prune_points = {45, 95}

        for t in range(self.iters):
            for isl in islands:
                isl.step(t, self.iters)

            # Periodic migration: replace worst with lightly jittered clone of best (antithetic alternation)
            if (t + 1) % migrate_K == 0:
                scores = np.array([isl.score for isl in islands])
                best_idx = int(np.argmax(scores))
                worst_idx = int(np.argmin(scores))
                if best_idx != worst_idx:
                    elite = islands[best_idx]
                    j = (self.halton.next(dim=2) * 2.0 - 1.0) * (0.0055 * self._mig_sign)
                    self._mig_sign *= -1
                    clone_centers = _clip_unit(elite.centers + j, self.margin_eps)
                    new_isl = SLPIsland(clone_centers, self.lp, self.sched, self.rng, self.edge_snap, margin_eps=self.margin_eps, accept_tol=1e-11)
                    new_isl.evaluate(self.sched.lp_eps(t))
                    islands[worst_idx] = new_isl

            # Early pruning: replace bottom quartile with best clones (antithetic)
            if (t + 1) in prune_points:
                scores = np.array([isl.score for isl in islands])
                order = np.argsort(scores)
                losers = list(order[:max(1, len(islands) // 4)])
                best_two = list(order[-2:])
                for k, li in enumerate(losers):
                    parent = islands[best_two[k % 2]]
                    j = (self.halton.next(dim=2) * 2.0 - 1.0) * 0.01
                    if k % 2 == 1:
                        j = -j
                    clone_centers = _clip_unit(parent.centers + j, self.margin_eps)
                    new_isl = SLPIsland(clone_centers, self.lp, self.sched, self.rng, self.edge_snap, margin_eps=self.margin_eps, accept_tol=1e-11)
                    new_isl.evaluate(self.sched.lp_eps(t))
                    islands[li] = new_isl

        # Final evaluate and polish top-2
        final_eps = self.sched.lp_eps(self.iters + 15)
        for isl in islands:
            isl.evaluate(final_eps)

        order = np.argsort([isl.score for isl in islands])[::-1][:2]
        for idx in order:
            islands[idx].polish(steps=26)

        # Select best
        best_state = None
        best_score = -np.inf
        for isl in islands:
            isl.evaluate(final_eps)
            if isl.score > best_score:
                best_score = isl.score
                best_state = isl

        return best_state.centers.copy(), best_state.radii.copy()


# ----------------------------- Public API -----------------------------
def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        centers: np.ndarray of shape (26, 2)
        radii:   np.ndarray of shape (26,)
    """
    arch = Archipelago(n=26, n_islands=14, iters=210, margin_eps=0.008, rng_seed=12345)
    centers, radii = arch.evolve()
    return centers, radii
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii