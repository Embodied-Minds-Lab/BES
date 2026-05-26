# EVOLVE-BLOCK-START
import numpy as np

# Optional SciPy import guarded for robustness
try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Internally searches over rectangle widths w in [0.85, 1.15] (h = 2 - w), two deterministic hex patterns,
    and performs local inflation and optional SLSQP refinement. Returns only the (x,y,r) array.
    """
    rng = np.random.default_rng(12345)

    n = 21
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def sum_r(circles):
        return float(np.sum(circles[:, 2]))

    def make_hex_seed(pattern, w, h):
        # Conservative initial radius from geometry
        R = len(pattern)
        max_cols = max(pattern)
        r0 = 0.49 * min(w / (2.0 * max_cols), h / (2.0 + (R - 1) * np.sqrt(3.0)))
        # Build centers
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.full(n, r0)
        idx = 0
        y_center = h / 2.0
        y_step = np.sqrt(3.0) * r0
        # center rows vertically
        row_offsets = [(ri - (R - 1) / 2.0) for ri in range(R)]
        for ri, cols in enumerate(pattern):
            y = y_center + row_offsets[ri] * y_step
            # Alternating horizontal offset for hex packing
            offset_flag = 1 if (ri % 2 == 1) else 0
            # Center horizontally
            for k in range(cols):
                x = w / 2.0 + (2 * k - (cols - 1)) * r0 + offset_flag * r0
                xs[idx] = x
                ys[idx] = y
                idx += 1
        # Deterministic tiny jitter for symmetry breaking
        jitter_amp = 1e-4 * r0
        xs += jitter_amp * rng.standard_normal(size=xs.shape)
        ys += jitter_amp * rng.standard_normal(size=ys.shape)
        circles = np.column_stack([xs, ys, rs])
        return circles

    def clip_boundaries(circles, w, h, r_min=1e-8):
        # Shrink radii to fit inside [0,w]x[0,h] without moving centers
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        r_clip = np.minimum.reduce([r, x, y, w - x, h - y])
        r_clip = np.maximum(r_clip, r_min)
        out = np.column_stack([x, y, r_clip])
        return out

    def overlaps(circles, tol=1e-9):
        # Return True if any pair overlaps or any negative radius
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        if np.any(r <= 0):
            return True
        for (i, j) in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dij = np.hypot(dx, dy)
            if dij + tol < (r[i] + r[j]):
                return True
        return False

    def feasible(circles, w, h, tol=1e-6):
        # Check boundaries and non-overlap
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        if np.any(r <= 0):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol) or np.any((x + r) - w > tol) or np.any((y + r) - h > tol):
            return False
        for (i, j) in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dij = np.hypot(dx, dy)
            if dij + tol < (r[i] + r[j]):
                return False
        return True

    def shrink_overlaps(circles, w, h, passes=4, tol=1e-9, r_min=1e-8):
        # Shrink-only projection: resolve boundary then overlaps
        c = circles.copy()
        c = clip_boundaries(c, w, h, r_min=r_min)
        x = c[:, 0]
        y = c[:, 1]
        r = c[:, 2]
        for _ in range(passes):
            changed = False
            # Resolve pairwise overlaps by equal split shrink
            for (i, j) in pairs:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dij = np.hypot(dx, dy)
                s = r[i] + r[j]
                ov = s - dij
                if ov > tol:
                    delta = ov / 2.0
                    # apply shrink with floor
                    new_ri = max(r[i] - delta, r_min)
                    new_rj = max(r[j] - delta, r_min)
                    if new_ri < r[i] - 1e-16 or new_rj < r[j] - 1e-16:
                        r[i] = new_ri
                        r[j] = new_rj
                        changed = True
            # Re-clip to boundaries after overlap shrink
            r = np.minimum.reduce([r, x, y, w - x, h - y])
            r = np.maximum(r, r_min)
            if not changed:
                break
        return np.column_stack([x, y, r])

    def inflate_radii(circles, w, h, iters=8, r_min=1e-8):
        # Greedy Gauss-Seidel inflation: increase each radius up to nearest constraint (boundary or other circle)
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        for _ in range(iters):
            for i in range(n):
                # boundary limit
                rmax = min(x[i], y[i], w - x[i], h - y[i])
                # pairwise limits
                for (a, b) in pairs:
                    if a == i or b == i:
                        j = b if a == i else a if b == i else None
                        if j is None:
                            continue
                        dij = np.hypot(x[i] - x[j], y[i] - y[j])
                        rmax = min(rmax, max(0.0, dij - r[j]))
                r[i] = max(r_min, min(rmax, r[i]))
        return np.column_stack([x, y, r])

    def slsqp_refine(circles, w_init, allow_w=False):
        if not HAVE_SCIPY:
            return None
        x0 = circles[:, 0].copy()
        y0 = circles[:, 1].copy()
        r0 = circles[:, 2].copy()

        def pack(x, y, r, w):
            if allow_w:
                return np.concatenate([x, y, r, np.array([w])])
            else:
                return np.concatenate([x, y, r])

        def unpack(z):
            x = z[0:n]
            y = z[n:2 * n]
            r = z[2 * n:3 * n]
            if allow_w:
                w = z[3 * n]
            else:
                w = w_init
            h = 2.0 - w
            return x, y, r, w, h

        z0 = pack(x0, y0, r0, w_init)

        def obj(z):
            _, _, r, _, _ = unpack(z)
            return -np.sum(r)

        def obj_grad(z):
            # gradient wrt x,y is 0; wrt r is -1; wrt w is 0
            g = np.zeros_like(z)
            g[2 * n:3 * n] = -1.0
            return g

        def ineq_boundary(z):
            x, y, r, w, h = unpack(z)
            # x - r >= 0, y - r >= 0, w - (x + r) >= 0, h - (y + r) >= 0
            return np.concatenate([x - r, y - r, w - (x + r), h - (y + r)])

        def ineq_boundary_jac(z):
            # Analytical Jacobian of boundary constraints
            p = z.size
            J = np.zeros((4 * n, p))
            # left: x - r
            for i in range(n):
                J[i, i] = 1.0                 # d/dx_i
                J[i, 2 * n + i] = -1.0        # d/dr_i
            # bottom: y - r
            for i in range(n):
                row = n + i
                J[row, n + i] = 1.0           # d/dy_i
                J[row, 2 * n + i] = -1.0      # d/dr_i
            # right: w - (x + r)
            for i in range(n):
                row = 2 * n + i
                J[row, i] = -1.0              # d/dx_i
                J[row, 2 * n + i] = -1.0      # d/dr_i
                if allow_w:
                    J[row, 3 * n] = 1.0       # d/dw
            # top: h - (y + r), h = 2 - w
            for i in range(n):
                row = 3 * n + i
                J[row, n + i] = -1.0          # d/dy_i
                J[row, 2 * n + i] = -1.0      # d/dr_i
                if allow_w:
                    J[row, 3 * n] = -1.0      # d/dw (since dh/dw = -1)
            return J

        def ineq_pairs(z):
            x, y, r, _, _ = unpack(z)
            vals = np.empty(len(pairs))
            for k, (i, j) in enumerate(pairs):
                dij = np.hypot(x[i] - x[j], y[i] - y[j])
                vals[k] = dij - (r[i] + r[j])
            return vals

        def ineq_pairs_jac(z):
            x, y, r, _, _ = unpack(z)
            p = z.size
            J = np.zeros((len(pairs), p))
            for k, (i, j) in enumerate(pairs):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dij = np.hypot(dx, dy)
                if dij >= 1e-12:
                    J[k, i] = dx / dij            # d/dx_i
                    J[k, j] = -dx / dij           # d/dx_j
                    J[k, n + i] = dy / dij        # d/dy_i
                    J[k, n + j] = -dy / dij       # d/dy_j
                # d/dr_i and d/dr_j
                J[k, 2 * n + i] = -1.0
                J[k, 2 * n + j] = -1.0
            return J

        cons = [
            {'type': 'ineq', 'fun': ineq_boundary, 'jac': ineq_boundary_jac},
            {'type': 'ineq', 'fun': ineq_pairs, 'jac': ineq_pairs_jac},
        ]

        bounds = []
        # x bounds
        bounds.extend([(0.0, 2.0)] * n)
        # y bounds
        bounds.extend([(0.0, 2.0)] * n)
        # r bounds
        bounds.extend([(1e-8, 1.0)] * n)
        # optional w bound
        if allow_w:
            bounds.append((0.2, 1.8))

        try:
            res = minimize(
                obj, z0, method='SLSQP', jac=obj_grad, bounds=bounds, constraints=cons,
                options={'ftol': 1e-10, 'maxiter': 1200, 'disp': False}
            )
            z = res.x
            x, y, r, w, h = unpack(z)
            cand = np.column_stack([x, y, r])
            # finalize feasibility
            cand = shrink_overlaps(cand, w, h, passes=5)
            cand = inflate_radii(cand, w, h, iters=6)
            if feasible(cand, w, h):
                return cand, w
            else:
                return None
        except Exception:
            return None

    # Multi-start hex seeding with width sweep
    patterns = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [5, 5, 4, 4, 3],
        [4, 5, 4, 4, 4],
        [5, 4, 4, 4, 4],
    ]
    width_grid = np.linspace(0.75, 1.25, 11)

    best_sum = -1.0
    best_circles = None

    for pattern in patterns:
        for w in width_grid:
            h = 2.0 - w
            if h <= 0:
                continue
            # Create seed
            seed = make_hex_seed(pattern, w, h)
            # Ensure within boundaries conservatively
            seed = clip_boundaries(seed, w, h)
            # Pre-inflate radii deterministically (robust fallback)
            seed = inflate_radii(seed, w, h, iters=14)
            seed = shrink_overlaps(seed, w, h, passes=4)
            # Try SLSQP Stage 1 (fixed w)
            refined1 = slsqp_refine(seed, w_init=w, allow_w=False)
            candidates = []
            if refined1 is not None:
                cand1, w1 = refined1
                candidates.append((cand1, w1))
                # Stage 2: unlock w starting from best of stage 1
                refined2 = slsqp_refine(cand1, w_init=w1, allow_w=True)
                if refined2 is not None:
                    cand2, w2 = refined2
                    candidates.append((cand2, w2))
            # If no SciPy or failure, keep seed itself
            if not candidates:
                candidates.append((seed, w))
            # Evaluate candidates
            for cand, wc in candidates:
                hc = 2.0 - wc
                # Final polish: inflate slightly and ensure feasibility
                cand = inflate_radii(cand, wc, hc, iters=6)
                cand = shrink_overlaps(cand, wc, hc, passes=5)
                if feasible(cand, wc, hc):
                    sr = sum_r(cand)
                    if sr > best_sum:
                        best_sum = sr
                        best_circles = cand.copy()

    # Fallback: if somehow none feasible, return a tiny centered layout
    if best_circles is None:
        w = 1.0
        h = 1.0
        best_circles = make_hex_seed([5, 4, 5, 4, 3], w, h)
        best_circles = clip_boundaries(best_circles, w, h)
        best_circles[:, 2] = np.minimum(best_circles[:, 2], 0.01)

    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")