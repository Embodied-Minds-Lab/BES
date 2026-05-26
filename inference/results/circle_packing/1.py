# EVOLVE-BLOCK-START
"""Tri-stage with adaptive cutting-plane LP, SA with drift lock-ins and micro SLSQP, hybrid seeds, and projector polish."""

import numpy as np

try:
    from scipy.optimize import linprog, minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    linprog = None
    minimize = None


def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    n = 26
    eps = 1e-6
    rng = np.random.default_rng(424242)

    # Restarts and budgets
    n_restarts = 10 if _HAS_SCIPY else 4

    # SA params
    sa_iters = 820
    sa_step0, sa_stepf = 0.072, 0.017
    sa_T0, sa_Tf = 1.8e-3, 8.0e-6
    adapt_window = 30
    target_accept = 0.31
    reheat_wait = 150

    # Projector/neighbor params
    k_nn = 10
    proj_cycles_early = (1, 1)  # cycles, passes
    proj_cycles_late = (3, 2)
    safety_every = 3

    # LP reuse params (baseline; LP has internal annealing + adaptive rebuild)
    tau_lp = 1e-3
    rebuild_M = 16  # periodic full rebuild cadence (adapted in LP when heavy augment)

    # SLSQP params
    slsqp_warm_iter = 150
    slsqp_polish_iter = 90
    slsqp_ftol = 1e-9

    # Contact polish
    polish_passes = 3
    contact_steps = 50
    contact_step_size = 0.006
    contact_tau0, contact_tauf = 0.004, 0.0015

    best_sum = -np.inf
    best_C = None
    best_r = None

    # Deterministic parameter schedules for seeding
    s_grid = [0.175, 0.182, 0.188, 0.194, 0.198]
    margin_grid = [0.040, 0.045, 0.050, 0.035]
    ring_grid = [0.205, 0.225, 0.245]
    belt_grid = [0.050, 0.060, 0.070]
    s_interior_grid = [0.165, 0.180, 0.195]
    spokes_R1_grid = [0.20, 0.215, 0.230]
    spokes_R2_grid = [0.28, 0.295, 0.310]
    jitter_grid = [0.006, 0.010, 0.014, 0.018, 0.012]

    for k in range(n_restarts):
        # Per-restart LP active-set state
        state = {"active_pairs": set(), "lp_calls": 0, "last_full": -10, "loose_counts": {}}

        # Build 4 seed candidates from distinct families + jitters
        seeds = []
        s = s_grid[k % len(s_grid)]
        margin = margin_grid[(2 * k) % len(margin_grid)]
        C_hex = _seed_hex_rows_26(s=s, margin=margin, rng=rng)
        seeds.append(C_hex)

        edge_margin = margin_grid[(k + 1) % len(margin_grid)]
        interior_ring = ring_grid[(3 * k) % len(ring_grid)]
        C_ring = _seed_edge_ring_26(edge_margin=edge_margin, interior_ring=interior_ring, rng=rng)
        seeds.append(C_ring)

        belt_margin = belt_grid[k % len(belt_grid)]
        s_in = s_interior_grid[(2 * k + 1) % len(s_interior_grid)]
        C_belt = _seed_corner_weighted_26(margin=belt_margin, s_in=s_in, rng=rng)
        seeds.append(C_belt)

        R1 = spokes_R1_grid[k % len(spokes_R1_grid)]
        R2 = spokes_R2_grid[(k + 1) % len(spokes_R2_grid)]
        C_spokes = _seed_spokes_26(edge_margin=edge_margin, R1=R1, R2=R2, rng=rng)
        seeds.append(C_spokes)

        # Apply deterministic small jitters (one anisotropic)
        cand_list = []
        for idx, C0 in enumerate(seeds):
            jitter = jitter_grid[(k + idx) % len(jitter_grid)]
            if idx == 1:
                # edge-ring: anisotropic jitter to favor tangential slides
                J = np.stack([rng.normal(0.0, jitter, size=n),
                              rng.normal(0.0, 0.5 * jitter, size=n)], axis=1)
            elif idx == 3:
                # spokes: radial-biased jitter
                V = C0 - 0.5
                NV = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
                dirv = V / NV
                mag = rng.normal(0.0, jitter, size=(n, 1))
                J = dirv * mag + rng.normal(0.0, 0.4 * jitter, size=C0.shape)
            else:
                J = rng.normal(0.0, jitter, size=C0.shape)
            C0j = np.clip(C0 + J, eps, 1.0 - eps)
            cand_list.append(C0j)

        # Probe candidates via short SA using fast projector; score with one LP
        probe_iters = 160
        C0 = _select_seed_with_probe(cand_list, rng=rng, eps=eps, iters=probe_iters,
                                     step0=0.052, stepf=0.018, T0=1.3e-3, Tf=1.0e-5,
                                     k_nn=k_nn, safety_every=safety_every,
                                     state=state, tau_lp=tau_lp, rebuild_M=rebuild_M)

        # Initial radii via exact LP (or projector fallback)
        r0 = _compute_max_radii_lp_active(C0, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M, force_full=True)

        # Mandatory SLSQP warm-start with analytic Jacobians, then exact LP resync
        if _HAS_SCIPY:
            C1, r1 = _slsqp_joint(C0, r0, eps=eps, maxiter=slsqp_warm_iter, ftol=slsqp_ftol,
                                  state=state, tau_lp=tau_lp, rebuild_M=rebuild_M)
        else:
            # SciPy-free: force-based relaxer + projector
            C1 = _force_relax_centers(C0, r0, steps=240, step_size=0.010, eps=eps, k_nn=k_nn)
            r1 = _projected_max_radii_knn(C1, None, _build_knn_indices(C1, k=k_nn), eps=eps,
                                          cycles=4, passes=2, grow_rate=0.55, safety_every=2)

        # SA with two-tier objective, enriched moves, adaptive acceptance, lock-ins
        C_sa, r_sa, s_sa = _sa_two_tier_lockins(
            C1, r1, state=state, iters=sa_iters,
            step0=sa_step0, stepf=sa_stepf, T0=sa_T0, Tf=sa_Tf,
            adapt_window=adapt_window, target_accept=target_accept, reheat_wait=reheat_wait,
            proj_cycles_early=proj_cycles_early, proj_cycles_late=proj_cycles_late,
            k_nn=k_nn, safety_every=safety_every, eps=eps,
            tau_lp=tau_lp, rebuild_M=rebuild_M
        )

        # Short SLSQP polish alternating with LP resync
        if _HAS_SCIPY:
            C_pol, r_pol = _slsqp_joint(C_sa, r_sa, eps=eps, maxiter=slsqp_polish_iter, ftol=slsqp_ftol,
                                        state=state, tau_lp=tau_lp, rebuild_M=rebuild_M)
            # One extra LP/SLSQP alternation
            r_pol = _compute_max_radii_lp_active(C_pol, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M)
            C_pol, r_pol = _slsqp_joint(C_pol, r_pol, eps=eps, maxiter=int(0.6 * slsqp_polish_iter), ftol=slsqp_ftol,
                                        state=state, tau_lp=tau_lp, rebuild_M=rebuild_M)
        else:
            C_pol = C_sa.copy()
            r_pol = _projected_max_radii_knn(C_pol, r_sa, _build_knn_indices(C_pol, k=k_nn), eps=eps,
                                             cycles=4, passes=2, grow_rate=0.55, safety_every=2)

        # Contact-guided polish interleaved with exact LP and short SLSQP
        C_alt = C_pol.copy()
        tau_sched = np.linspace(contact_tau0, contact_tauf, max(1, polish_passes))
        for idx in range(polish_passes):
            C_alt = _contact_guided_nudge(C_alt, steps=contact_steps, step_size=contact_step_size,
                                          tau=float(tau_sched[idx]), eps=eps, k_nn=k_nn)
            r_alt = _compute_max_radii_lp_active(C_alt, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M)
            if _HAS_SCIPY:
                C_alt, r_alt = _slsqp_joint(C_alt, r_alt, eps=eps, maxiter=int(0.6 * slsqp_polish_iter), ftol=slsqp_ftol,
                                            state=state, tau_lp=tau_lp, rebuild_M=rebuild_M)

        # Final exact radii for this restart
        r_final = _compute_max_radii_lp_active(C_alt, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M, force_full=True)
        s_final = float(np.sum(r_final))

        # Select best among stage outputs
        candidates = [
            (float(np.sum(r1)), C1, r1),
            (s_sa, C_sa, r_sa),
            (float(np.sum(r_pol)), C_pol, r_pol),
            (s_final, C_alt, r_final),
        ]
        s_k, C_k, r_k = max(candidates, key=lambda t: t[0])
        if s_k > best_sum:
            best_sum = s_k
            best_C = C_k.copy()
            best_r = r_k.copy()

    # Final strict feasibility and clamps
    best_C = np.clip(best_C, eps, 1.0 - eps)
    if _HAS_SCIPY:
        final_state = {"active_pairs": set(), "lp_calls": 0, "last_full": -10, "loose_counts": {}}
        best_r = _compute_max_radii_lp_active(best_C, final_state, eps=eps, tau_lp=1e-3, rebuild_M=12, force_full=True)
    else:
        best_r = _projected_max_radii_knn(best_C, None, _build_knn_indices(best_C, k=10), eps=eps,
                                          cycles=6, passes=3, grow_rate=0.60, safety_every=2)
        best_r = _project_radii_feasible(best_C, best_r, eps=eps, passes=3)
    best_r = np.maximum(best_r, 0.0)
    return best_C, best_r


# ----------------------------
# Geometry helpers
# ----------------------------
def _pairwise_dists(C):
    diff = C[:, None, :] - C[None, :, :]
    return np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=2)))


def _boundary_limits(C, eps=0.0):
    x = C[:, 0]; y = C[:, 1]
    return np.minimum.reduce([x - eps, y - eps, 1.0 - x - eps, 1.0 - y - eps])


# ----------------------------
# k-NN helpers and projector
# ----------------------------
def _build_knn_indices(C, k=10):
    n = C.shape[0]
    D = _pairwise_dists(C)
    np.fill_diagonal(D, np.inf)
    # partial argsort for k nearest
    idx = np.argpartition(D, kth=min(k, n - 1) - 1, axis=1)[:, :min(k, n - 1)]
    return idx


def _projected_max_radii_knn(C, r_init, knn_idx, eps=1e-6, cycles=2, passes=1, grow_rate=0.5, safety_every=3):
    """
    Fast feasibility projector using k-NN pairs with periodic global shrink-only safety sweeps.
    Deterministic; warm-start from r_init if provided.
    """
    n = C.shape[0]
    b = np.maximum(_boundary_limits(C, eps=eps), 0.0)
    r = np.clip((r_init if r_init is not None else 0.9 * b), 0.0, b)
    D = _pairwise_dists(C)

    for cyc in range(max(1, cycles)):
        # Shrink passes over k-NN pairs
        for _ in range(max(1, passes)):
            for i in range(n):
                nbrs = knn_idx[i]
                ri = r[i]
                for j in nbrs:
                    if j <= i:
                        continue
                    dij = D[i, j]
                    s = ri + r[j]
                    if s > dij - eps:
                        excess = s - (dij - eps)
                        denom = ri + r[j] + 1e-16
                        di = excess * (ri / denom)
                        dj = excess - di
                        ri = max(0.0, ri - di)
                        r[j] = max(0.0, r[j] - dj)
                r[i] = ri
            r = np.minimum(r, b)

        # Regrow with approximate local gaps
        for i in range(n):
            if r[i] <= 0.0:
                continue
            mb = b[i] - r[i]
            if mb <= 0.0:
                continue
            gaps = D[i, knn_idx[i]] - (r[i] + r[knn_idx[i]]) - eps
            g = np.min(gaps) if gaps.size > 0 else np.inf
            grow = max(0.0, min(mb, g if np.isfinite(g) else mb))
            if grow > 0.0:
                r[i] += grow_rate * grow
        r = np.clip(r, 0.0, b)

        # Global safety sweep
        if ((cyc + 1) % max(1, safety_every)) == 0:
            for i in range(n):
                for j in range(i + 1, n):
                    dij = D[i, j]
                    s = r[i] + r[j]
                    if s > dij - eps:
                        excess = s - (dij - eps)
                        if r[i] >= r[j]:
                            r[i] = max(0.0, r[i] - excess)
                        else:
                            r[j] = max(0.0, r[j] - excess)
            r = np.minimum(r, b)

    return r


def _project_radii_feasible(C, r, eps=1e-6, passes=3):
    """
    Shrink-only projection to satisfy constraints while staying close to input r.
    """
    n = C.shape[0]
    b = np.maximum(_boundary_limits(C, eps=eps), 0.0)
    r = np.clip(np.asarray(r, dtype=float), 0.0, b)
    D = _pairwise_dists(C)
    for _ in range(passes):
        for i in range(n):
            for j in range(i + 1, n):
                dij = D[i, j]
                s = r[i] + r[j]
                if s > dij - eps:
                    excess = s - (dij - eps)
                    if r[i] >= r[j]:
                        r[i] = max(0.0, r[i] - excess)
                    else:
                        r[j] = max(0.0, r[j] - excess)
        r = np.minimum(r, b)
    return r


# ----------------------------
# Exact LP with cutting-plane active-set reuse, augmentation, pruning, and annealed tightness
# ----------------------------
def _compute_max_radii_lp_active(C, state, eps=1e-6, tau_lp=1e-3, rebuild_M=16, force_full=False):
    """
    Exact LP for radii with fixed centers using an active-set cutting-plane scheme:
      maximize sum r
      s.t. r_i >= 0
           r_i <= boundary_i
           r_i + r_j <= d_ij - eps

    Strategy:
      - Periodic full vectorized LP (boundary + all pairs).
      - Otherwise, solve with boundary + active_pairs, then augment with the M most
        violated pairs for up to 'rounds_max' rounds.
      - Maintain and prune active_pairs via slack tracking; anneal near-active tau.
      - If augmentation is heavy or residual violations persist, force early full rebuild.
      - SA can set state["anneal_alpha"] in [0,1] to steer the annealing of tau.
    """
    n = C.shape[0]
    b = np.maximum(_boundary_limits(C, eps=eps), 0.0)

    if not _HAS_SCIPY:
        # Fast projector fallback with strict shrink-only at the end
        knn = _build_knn_indices(C, k=10)
        r = _projected_max_radii_knn(C, None, knn, eps=eps, cycles=4, passes=2, grow_rate=0.55, safety_every=2)
        r = _project_radii_feasible(C, r, eps=eps, passes=3)
        return np.clip(r, 0.0, b)

    # Initialize state
    active_pairs = set()
    for p in state.get("active_pairs", set()):
        i, j = int(p[0]), int(p[1])
        if i > j:
            i, j = j, i
        active_pairs.add((i, j))
    lp_calls = int(state.get("lp_calls", 0))
    last_full = int(state.get("last_full", -10))
    loose_counts = state.get("loose_counts", {})
    if loose_counts is None:
        loose_counts = {}

    # Anneal near-active threshold from 2e-3 to 6e-4 driven by optional SA alpha; fallback to lp_calls
    tau_start, tau_end = 2e-3, 6e-4
    alpha = state.get("anneal_alpha", None)
    if alpha is None:
        alpha = min(1.0, lp_calls / 60.0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    tau_eff = (1.0 - alpha) * tau_start + alpha * tau_end

    # Cutting-plane controls
    M = 120
    rounds_max = 4

    D = _pairwise_dists(C)
    iu, ju = np.triu_indices(n, 1)
    m_pairs = iu.size
    # Map pair -> triu index
    pair_to_idx = {}
    for idx in range(m_pairs):
        i, j = int(iu[idx]), int(ju[idx])
        pair_to_idx[(i, j)] = idx

    def solve_lp_with_pairs(pairs):
        I = np.eye(n, dtype=float)
        A_rows = [I[i].copy() for i in range(n)]
        b_rows = [b[i] for i in range(n)]
        for (i, j) in sorted(pairs):
            row = np.zeros(n, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(max(0.0, D[i, j] - eps))
        A_ub = np.array(A_rows, dtype=float)
        b_ub = np.array(b_rows, dtype=float)
        c = -np.ones(n, dtype=float)
        bounds = [(0.0, None)] * n
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            return None
        return np.asarray(res.x, dtype=float)

    do_full = force_full or (lp_calls - last_full >= rebuild_M) or (len(active_pairs) == 0)
    heavy_augment = False

    if do_full:
        # Full vectorized LP build
        c = -np.ones(n, dtype=float)
        I = np.eye(n, dtype=float)
        A_pairs = np.zeros((m_pairs, n), dtype=float)
        A_pairs[np.arange(m_pairs), iu] = 1.0
        A_pairs[np.arange(m_pairs), ju] = 1.0
        A_ub = np.vstack([I, A_pairs])
        d_ij = np.maximum(D[iu, ju] - eps, 0.0)
        b_ub = np.concatenate([b, d_ij])
        bounds = [(0.0, None)] * n
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            # rare fallback: try reduced solve on current active set if any
            r = solve_lp_with_pairs(active_pairs) if len(active_pairs) > 0 else None
            if r is None:
                knn = _build_knn_indices(C, k=10)
                r = _projected_max_radii_knn(C, None, knn, eps=eps, cycles=4, passes=2, grow_rate=0.55, safety_every=2)
        else:
            r = np.asarray(res.x, dtype=float)

        # Update active set with near-active pairs (annealed threshold)
        slack_all = (D[iu, ju] - eps) - (r[iu] + r[ju])
        near = np.where(slack_all <= tau_eff)[0]
        new_active = set((int(iu[idx]), int(ju[idx])) for idx in near)
        state["active_pairs"] = new_active
        # Reset loose counts to those in new active set
        loose_counts = {p: loose_counts.get(p, 0) for p in new_active}
        state["loose_counts"] = loose_counts
        state["last_full"] = lp_calls
    else:
        # Reduced LP with current active set and cutting-plane augmentation
        r = solve_lp_with_pairs(active_pairs)
        if r is None:
            # fallback to full
            return _compute_max_radii_lp_active(C, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M, force_full=True)

        slack_all = (D[iu, ju] - eps) - (r[iu] + r[ju])
        viol_idx = np.where(slack_all < -1e-12)[0]
        rounds = 0
        added_total = 0

        while viol_idx.size > 0 and rounds < rounds_max:
            # Add up to M most violated new pairs
            order = np.argsort(slack_all[viol_idx])  # ascending: most negative first
            to_consider = viol_idx[order][:M]
            added = 0
            for idx in to_consider:
                i, j = int(iu[idx]), int(ju[idx])
                if i > j:
                    i, j = j, i
                if (i, j) not in active_pairs:
                    active_pairs.add((i, j))
                    added += 1
            added_total += added
            if added == 0:
                break
            r = solve_lp_with_pairs(active_pairs)
            if r is None:
                return _compute_max_radii_lp_active(C, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M, force_full=True)
            slack_all = (D[iu, ju] - eps) - (r[iu] + r[ju])
            viol_idx = np.where(slack_all < -1e-12)[0]
            rounds += 1

        heavy_augment = (added_total > (M // 2)) or (viol_idx.size > 0)

        # Refresh near-active set with annealed threshold
        near = np.where(slack_all <= tau_eff)[0]
        for idx in near:
            i, j = int(iu[idx]), int(ju[idx])
            if i > j:
                i, j = j, i
            active_pairs.add((i, j))

        # Prune loose pairs: if slack > 2*tau_eff for 3 consecutive calls
        to_remove = []
        for (i, j) in list(active_pairs):
            pidx = pair_to_idx.get((i, j), None)
            if pidx is None:
                continue
            s = float(slack_all[pidx])
            if s > 2.0 * tau_eff:
                loose_counts[(i, j)] = loose_counts.get((i, j), 0) + 1
                if loose_counts[(i, j)] >= 3:
                    to_remove.append((i, j))
            else:
                loose_counts[(i, j)] = 0
        for p in to_remove:
            if p in active_pairs:
                active_pairs.remove(p)
            if p in loose_counts:
                del loose_counts[p]

        # Keep only counts for active pairs
        loose_counts = {p: loose_counts.get(p, 0) for p in active_pairs}
        state["active_pairs"] = active_pairs
        state["loose_counts"] = loose_counts

        # If augmentation was heavy or residual violations remain, force a sooner full rebuild
        if heavy_augment:
            state["last_full"] = lp_calls - rebuild_M

    # Book-keeping
    state["lp_calls"] = lp_calls + 1

    # Post-solve strict guard
    r = np.minimum(r, b)
    r = _project_radii_feasible(C, r, eps=eps, passes=2)
    r = np.clip(r, 0.0, b)
    return r


# ----------------------------
# SA with two-tier scoring and opportunistic LP lock-ins
# ----------------------------
def _sa_two_tier_lockins(C_init, r_init, state, iters=700, step0=0.06, stepf=0.012, T0=1e-3, Tf=1e-5,
                         adapt_window=30, target_accept=0.30, reheat_wait=120,
                         proj_cycles_early=(1, 1), proj_cycles_late=(2, 2),
                         k_nn=10, safety_every=3, eps=1e-6, tau_lp=1e-3, rebuild_M=16):
    rng = np.random.default_rng(101)
    n = C_init.shape[0]
    C = np.clip(C_init.copy(), eps, 1.0 - eps)

    # Accurate baseline
    state["anneal_alpha"] = 0.0
    r_acc = _compute_max_radii_lp_active(C, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M)
    s_acc = float(np.sum(r_acc))

    # Fast baseline warm-start
    knn = _build_knn_indices(C, k=k_nn)
    cyc, pas = proj_cycles_early
    r_fast = _projected_max_radii_knn(C, r_acc, knn, eps=eps, cycles=cyc, passes=pas, grow_rate=0.5, safety_every=safety_every)
    cur_fast = float(np.sum(r_fast))

    bestC = C.copy()
    bestR = r_acc.copy()
    bestS = s_acc
    last_improve_iter = 0

    accept_flags = np.zeros(adapt_window, dtype=np.int8)
    step_scale = 1.0

    # Surrogate drift threshold for immediate LP lock-in
    drift_thresh0, drift_threshf = 3.0e-3, 9.0e-4

    # Lock-in helper
    def lock_in_and_nudge(t, C_ref, r_hint, step_alpha=1.0):
        r_now = _compute_max_radii_lp_active(C_ref, state, eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M)
        return C_ref, r_now

    for t in range(iters):
        frac = t / max(1, iters - 1)
        state["anneal_alpha"] = float(frac)
        step = (step0 * (1.0 - frac) + stepf * frac) * step_scale
        step = float(np.clip(step, min(step0, stepf) * 0.6, max(step0, stepf) * 1.8))
        T = T0 * (Tf / max(T0, 1e-16)) ** frac
        drift_thresh = (1 - frac) * drift_thresh0 + frac * drift_threshf
        # Anneal projector effort
        cyc = int(round((1 - frac) * proj_cycles_early[0] + frac * proj_cycles_late[0]));  cyc = max(1, cyc)
        pas = int(round((1 - frac) * proj_cycles_early[1] + frac * proj_cycles_late[1]));  pas = max(1, pas)
        # Adaptive LP cadence (cools over time)
        k_lp_cur = int(np.clip(round(40 - 26 * frac), 14, 40))

        # Propose a move using annealed move mix
        prop = C.copy()
        targeted = False

        # Schedule move probabilities and normalize to 1
        p_jit = (1 - frac) * 0.70 + frac * 0.60
        p_edge = (1 - frac) * 0.10 + frac * 0.15
        p_corner = (1 - frac) * 0.07 + frac * 0.12
        p_pair = (1 - frac) * 0.03 + frac * 0.13
        weights = np.array([p_jit, p_edge, p_corner, p_pair], dtype=float)
        weights = weights / np.sum(weights)
        thr = np.cumsum(weights)
        u = rng.uniform()

        if u < thr[0]:
            # Single-point jitter
            i = rng.integers(0, n)
            delta = rng.normal(0.0, step, size=2)
            prop[i] = np.clip(prop[i] + delta, eps, 1.0 - eps)
        elif u < thr[1]:
            # Edge-slide along nearest-edge tangent (targeted)
            i = rng.integers(0, n)
            bx = min(prop[i, 0], 1.0 - prop[i, 0])
            by = min(prop[i, 1], 1.0 - prop[i, 1])
            if bx < by:
                delta = np.array([0.0, rng.normal(0.0, step)])
            else:
                delta = np.array([rng.normal(0.0, step), 0.0])
            prop[i] = np.clip(prop[i] + delta, eps, 1.0 - eps)
            targeted = True
        elif u < thr[2]:
            # Corner-pull toward nearest corner (targeted)
            i = rng.integers(0, n)
            corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
            dvecs = corners - prop[i]
            dists = np.linalg.norm(dvecs, axis=1) + 1e-12
            j = int(np.argmin(dists))
            dirv = dvecs[j] / dists[j]
            prop[i] = np.clip(prop[i] + step * dirv, eps, 1.0 - eps)
            targeted = True
        else:
            # Paired orthogonal translate of nearest neighbor pair (targeted)
            i = rng.integers(0, n)
            Dloc = _pairwise_dists(prop)
            Dloc[i, i] = np.inf
            j = int(np.argmin(Dloc[i]))
            v = prop[j] - prop[i]
            dv = np.linalg.norm(v)
            if dv < 1e-12:
                ortho = rng.normal(0.0, 1.0, size=2); ortho /= (np.linalg.norm(ortho) + 1e-12)
            else:
                ortho = np.array([-v[1], v[0]]); ortho /= (np.linalg.norm(ortho) + 1e-12)
            dvec = 0.5 * step * ortho
            prop[i] = np.clip(prop[i] + dvec, eps, 1.0 - eps)
            prop[j] = np.clip(prop[j] - dvec, eps, 1.0 - eps)
            targeted = True

        # Refresh k-NN structure mildly
        if (t % 20) == 0 or targeted:
            knn = _build_knn_indices(prop, k=k_nn)

        # Fast evaluation with warm-started projector
        prev_fast = cur_fast
        r_fast_prop = _projected_max_radii_knn(prop, r_fast, knn, eps=eps, cycles=cyc, passes=pas, grow_rate=0.5, safety_every=safety_every)
        s_fast_prop = float(np.sum(r_fast_prop))
        delta_fast = s_fast_prop - prev_fast

        # Accept/reject
        accept = False
        if s_fast_prop >= cur_fast - 1e-15:
            accept = True
        elif T > 0 and (rng.uniform() < np.exp((s_fast_prop - cur_fast) / max(T, 1e-16))):
            accept = True
        accept_flags[t % adapt_window] = 1 if accept else 0

        if accept:
            C = prop
            r_fast = r_fast_prop
            cur_fast = s_fast_prop

            # Opportunistic LP lock-in on targeted accepts
            if targeted:
                C, r_now = lock_in_and_nudge(t, C, r_fast, step_alpha=1.0)
                s_now = float(np.sum(r_now))
                cur_fast = s_now
                # Refresh projector warm-start
                knn = _build_knn_indices(C, k=k_nn)
                r_fast = _projected_max_radii_knn(C, r_now, knn, eps=eps, cycles=cyc, passes=pas, grow_rate=0.5, safety_every=safety_every)
                if s_now > bestS + 1e-12:
                    bestS = s_now; bestC = C.copy(); bestR = r_now.copy(); last_improve_iter = t

            # Immediate LP lock-in if surrogate drifted a lot on this accepted move
            if abs(delta_fast) > drift_thresh:
                C, r_now = lock_in_and_nudge(t, C, r_fast, step_alpha=1.0)
                s_now = float(np.sum(r_now))
                cur_fast = s_now
                knn = _build_knn_indices(C, k=k_nn)
                r_fast = _projected_max_radii_knn(C, r_now, knn, eps=eps, cycles=cyc, passes=pas, grow_rate=0.5, safety_every=safety_every)
                if s_now > bestS + 1e-12:
                    bestS = s_now; bestC = C.copy(); bestR = r_now.copy(); last_improve_iter = t

        # Periodic exact synchronization with adaptive cadence
        if (t % max(1, k_lp_cur)) == 0:
            C, r_now = lock_in_and_nudge(t, C, r_fast, step_alpha=0.8)
            s_now = float(np.sum(r_now))
            cur_fast = s_now
            if s_now > bestS + 1e-12:
                bestS = s_now; bestC = C.copy(); bestR = r_now.copy(); last_improve_iter = t
            # Optional late-stage micro-burst to tighten local contacts
            if _HAS_SCIPY and frac > 0.60:
                C, r_now = _slsqp_micro_burst(C, r_now, state=state, eps=eps, K=10, maxiter=12, ftol=1e-10, tau_lp=tau_lp, rebuild_M=rebuild_M)
                s_now2 = float(np.sum(r_now))
                if s_now2 > s_now + 1e-12:
                    s_now = s_now2
                    if s_now > bestS + 1e-12:
                        bestS = s_now; bestC = C.copy(); bestR = r_now.copy(); last_improve_iter = t
            # refresh projector state
            knn = _build_knn_indices(C, k=k_nn)
            r_fast = _projected_max_radii_knn(C, r_now, knn, eps=eps, cycles=cyc, passes=pas, grow_rate=0.5, safety_every=safety_every)
            cur_fast = s_now

        # Adaptive step scaling
        if (t + 1) % adapt_window == 0:
            acc_rate = float(np.sum(accept_flags)) / float(adapt_window)
            if acc_rate < (target_accept - 0.03):
                step_scale *= 0.90
            elif acc_rate > (target_accept + 0.05):
                step_scale *= 1.08
            step_scale = float(np.clip(step_scale, 0.6, 1.8))

        # Reheating on stagnation
        if (t - last_improve_iter) >= reheat_wait:
            step_scale = float(min(1.8, step_scale * 1.4))
            # Immediate LP recalibration
            C, r_now = lock_in_and_nudge(t, C, r_fast, step_alpha=1.0)
            s_now = float(np.sum(r_now))
            cur_fast = s_now
            if s_now > bestS + 1e-12:
                bestS = s_now; bestC = C.copy(); bestR = r_now.copy(); last_improve_iter = t
            # refresh projector state
            knn = _build_knn_indices(C, k=k_nn)
            r_fast = _projected_max_radii_knn(C, r_now, knn, eps=eps, cycles=cyc, passes=pas, grow_rate=0.5, safety_every=safety_every)

    return bestC, bestR, bestS


# ----------------------------
# Contact-guided nudge
# ----------------------------
def _contact_guided_nudge(C_init, steps=60, step_size=0.006, tau=0.003, eps=1e-6, k_nn=10):
    """
    Radii-aware polishing: push near-contacting pairs apart; push boundary-active inward.
    Uses fast projector radii estimate for responsiveness; final stages will LP-resync.
    """
    C = C_init.copy()
    n = C.shape[0]
    knn = _build_knn_indices(C, k=k_nn)
    for it in range(steps):
        # approximate radii
        r = _projected_max_radii_knn(C, None, knn, eps=eps, cycles=1, passes=1, grow_rate=0.5, safety_every=2)
        D = _pairwise_dists(C)
        slack = D - (r[:, None] + r[None, :])
        grads = np.zeros_like(C)

        # Near-active pairs (threshold tau)
        for i in range(n):
            # use k-NN subset for speed
            for j in knn[i]:
                if j <= i:
                    continue
                s_ij = slack[i, j]
                if np.isfinite(s_ij) and s_ij < tau:
                    u = C[i] - C[j]
                    d = np.linalg.norm(u) + 1e-12
                    u = u / d
                    w = (tau - s_ij) / max(tau, 1e-12)
                    grads[i] += w * u
                    grads[j] -= w * u

        # Boundary-active inward push
        b_cands = np.stack([C[:, 0], C[:, 1], 1.0 - C[:, 0], 1.0 - C[:, 1]], axis=1)
        bmin_idx = np.argmin(b_cands, axis=1)
        bmin = b_cands[np.arange(n), bmin_idx]
        for i in range(n):
            if r[i] >= bmin[i] - tau:
                if bmin_idx[i] == 0:   # left
                    grads[i, 0] += 1.0
                elif bmin_idx[i] == 1: # bottom
                    grads[i, 1] += 1.0
                elif bmin_idx[i] == 2: # right
                    grads[i, 0] -= 1.0
                else:                  # top
                    grads[i, 1] -= 1.0

        # Mild central regularizer for stability
        grads += 0.03 * (0.5 - C)

        # Normalize and step
        norms = np.linalg.norm(grads, axis=1)
        gnorm = np.max(norms)
        if gnorm > 0:
            grads = grads / gnorm
        C = np.clip(C + step_size * grads, eps, 1.0 - eps)
        # Update knn every few steps
        if (it % 10) == 0:
            knn = _build_knn_indices(C, k=k_nn)
    return C


# ----------------------------
# Joint SLSQP with full analytic Jacobians
# ----------------------------
def _pack_vars(C, r):
    return np.concatenate([C.reshape(-1), r.reshape(-1)], axis=0)


def _unpack_vars(v, n):
    C = v[:2 * n].reshape(n, 2)
    r = v[2 * n:]
    return C, r


def _ineq_constraints_full(v, n, eps=0.0):
    C, r = _unpack_vars(v, n)
    D = _pairwise_dists(C)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(D[i, j] - r[i] - r[j] - eps)
    x, y = C[:, 0], C[:, 1]
    rows.extend(list(x - r - eps))
    rows.extend(list(y - r - eps))
    rows.extend(list(1.0 - x - r - eps))
    rows.extend(list(1.0 - y - r - eps))
    rows.extend(list(r - eps))
    return np.array(rows, dtype=float)


def _ineq_jacobian_full(v, n, eps=0.0):
    C, _ = _unpack_vars(v, n)
    x, y = C[:, 0], C[:, 1]
    m_pairs = n * (n - 1) // 2
    m_total = m_pairs + 5 * n
    J = np.zeros((m_total, 3 * n), dtype=float)

    # pairwise
    row = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dij = np.hypot(dx, dy)
            if dij < 1e-12:
                ux = uy = vx = vy = 0.0
            else:
                ux, uy, vx, vy = dx / dij, dy / dij, -dx / dij, -dy / dij
            J[row, 2 * i + 0] = ux
            J[row, 2 * i + 1] = uy
            J[row, 2 * j + 0] = vx
            J[row, 2 * j + 1] = vy
            J[row, 2 * n + i] = -1.0
            J[row, 2 * n + j] = -1.0
            row += 1

    # x - r
    for i in range(n):
        J[row, 2 * i + 0] = 1.0
        J[row, 2 * n + i] = -1.0
        row += 1
    # y - r
    for i in range(n):
        J[row, 2 * i + 1] = 1.0
        J[row, 2 * n + i] = -1.0
        row += 1
    # 1 - x - r
    for i in range(n):
        J[row, 2 * i + 0] = -1.0
        J[row, 2 * n + i] = -1.0
        row += 1
    # 1 - y - r
    for i in range(n):
        J[row, 2 * i + 1] = -1.0
        J[row, 2 * n + i] = -1.0
        row += 1
    # r >= 0
    for i in range(n):
        J[row, 2 * n + i] = 1.0
        row += 1
    return J


def _objective_joint(v, n):
    r = v[2 * n:]
    return -np.sum(r)


def _objective_joint_jac(v, n):
    g = np.zeros_like(v)
    g[2 * n:] = -1.0
    return g


def _bounds_joint(n, eps=1e-6):
    bnds = []
    for _ in range(n):
        bnds.append((eps, 1.0 - eps))  # x
        bnds.append((eps, 1.0 - eps))  # y
    for _ in range(n):
        bnds.append((0.0, 0.5))        # r
    return bnds


def _slsqp_joint(C0, r0, eps=1e-6, maxiter=120, ftol=1e-9, state=None, tau_lp=1e-3, rebuild_M=16):
    """
    Jointly optimize centers and radii to locally increase sum of radii.
    After SLSQP, recompute radii via exact LP to resynchronize exact feasibility.
    """
    if not _HAS_SCIPY:
        knn = _build_knn_indices(C0, k=10)
        r1 = _projected_max_radii_knn(C0, r0, knn, eps=eps, cycles=4, passes=2, grow_rate=0.55, safety_every=2)
        return C0.copy(), r1
    n = C0.shape[0]
    v0 = _pack_vars(C0, r0)
    cons = ({
        "type": "ineq",
        "fun": lambda v: _ineq_constraints_full(v, n, eps=eps),
        "jac": lambda v: _ineq_jacobian_full(v, n, eps=eps),
    },)
    res = minimize(
        fun=lambda v: _objective_joint(v, n),
        x0=v0,
        jac=lambda v: _objective_joint_jac(v, n),
        constraints=cons,
        bounds=_bounds_joint(n, eps=eps),
        method="SLSQP",
        options={"maxiter": int(maxiter), "ftol": float(min(ftol, 1e-9)), "disp": False},
    )
    C, _ = _unpack_vars(res.x, n)
    # Always resync radii via LP / projector
    r = _compute_max_radii_lp_active(C, state if state is not None else {"active_pairs": set(), "lp_calls": 0, "last_full": -10, "loose_counts": {}},
                                     eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M)
    return C, r


def _slsqp_micro_burst(C0, r0, state=None, eps=1e-6, K=10, maxiter=12, ftol=1e-10, tau_lp=1e-3, rebuild_M=16):
    """
    Late-stage micro SLSQP on a focused subset of K circles with highest active degree
    and smallest boundary slack; freezes other centers. Immediately LP-resyncs radii.
    """
    if not _HAS_SCIPY:
        return C0.copy(), r0.copy()
    n = C0.shape[0]
    # Active degrees from current LP active set
    deg = np.zeros(n, dtype=int)
    for (i, j) in sorted(state.get("active_pairs", set())):
        deg[i] += 1; deg[j] += 1
    # Boundary slack
    b = np.maximum(_boundary_limits(C0, eps=eps), 0.0)
    slack_b = np.clip(b - r0, 0.0, None)
    # Select K by degree then boundary slack
    k1 = min(max(0, K // 2), n)
    top_deg = list(np.argsort(-deg)[:k1])
    remaining = [i for i in range(n) if i not in top_deg]
    need = max(0, K - len(top_deg))
    if len(remaining) > 0 and need > 0:
        by_bslack = list(np.argsort(slack_b[remaining])[:need])
        sel = set(top_deg + [remaining[i] for i in by_bslack])
    else:
        sel = set(top_deg)
    if len(sel) < K:
        # fill by largest radii
        for i in np.argsort(-r0):
            sel.add(int(i))
            if len(sel) >= K:
                break
    sel = sorted(list(sel))
    # Build bounds freezing non-selected centers
    bnds = _bounds_joint(n, eps=eps)
    for i in range(n):
        if i not in sel:
            bnds[2 * i + 0] = (float(C0[i, 0]), float(C0[i, 0]))
            bnds[2 * i + 1] = (float(C0[i, 1]), float(C0[i, 1]))
    v0 = _pack_vars(C0, r0)
    cons = ({
        "type": "ineq",
        "fun": lambda v: _ineq_constraints_full(v, n, eps=eps),
        "jac": lambda v: _ineq_jacobian_full(v, n, eps=eps),
    },)
    res = minimize(
        fun=lambda v: _objective_joint(v, n),
        x0=v0,
        jac=lambda v: _objective_joint_jac(v, n),
        constraints=cons,
        bounds=bnds,
        method="SLSQP",
        options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": False},
    )
    C_new, _ = _unpack_vars(res.x, n)
    # Exact LP resync
    r_new = _compute_max_radii_lp_active(C_new, state if state is not None else {"active_pairs": set(), "lp_calls": 0, "last_full": -10, "loose_counts": {}},
                                         eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M)
    return np.clip(C_new, eps, 1.0 - eps), r_new


# ----------------------------
# Fallback center relaxer (no SciPy)
# ----------------------------
def _force_relax_centers(C, r, steps=240, step_size=0.010, eps=1e-6, k_nn=10):
    """
    Lightweight fallback if SciPy is unavailable:
    push overlapping pairs apart and gently toward interior, then recompute r.
    """
    n = C.shape[0]
    X = C.copy()
    knn = _build_knn_indices(X, k=k_nn)
    for t in range(steps):
        grads = np.zeros_like(X)
        D = _pairwise_dists(X)
        for i in range(n):
            for j in range(i + 1, n):
                dij = D[i, j]
                overlap = (r[i] + r[j] + eps) - dij
                if overlap > 0:
                    dx = X[i] - X[j]
                    d = np.linalg.norm(dx) + 1e-12
                    push = (overlap / d) * dx
                    grads[i] += push
                    grads[j] -= push
        grads += 0.05 * (0.5 - X)
        X = np.clip(X + step_size * grads, eps, 1 - eps)
        if (t % 10) == 0:
            knn = _build_knn_indices(X, k=k_nn)
        r = _projected_max_radii_knn(X, r, knn, eps=eps, cycles=1, passes=1, grow_rate=0.5, safety_every=2)
    return X


# ----------------------------
# Seeding strategies and probing
# ----------------------------
def _seed_hex_rows_26(s=0.18, margin=0.04, rng=None):
    """
    7-row hex-like layout totaling 26 centers: [3,4,4,5,4,3,3]
    Rows within [margin, 1-margin]; alternate rows offset by s/2.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    Ls = [3, 4, 4, 5, 4, 3, 3]
    sv = s * np.sqrt(3.0) / 2.0
    num_rows = len(Ls)
    row_idx = np.arange(num_rows) - (num_rows - 1) / 2.0
    ys = 0.5 + row_idx * sv
    # clamp rows within margins
    top_excess = max(0.0, np.max(ys) - (1.0 - margin))
    bot_excess = max(0.0, margin - np.min(ys))
    ys = ys - top_excess + bot_excess

    centers = []
    for k, L in enumerate(Ls):
        xs = 0.5 + (np.arange(L) - (L - 1) / 2.0) * s
        if (k % 2) == 1:
            xs = xs + 0.5 * s
        xs = np.clip(xs, margin, 1.0 - margin)
        yk = float(np.clip(ys[k], margin, 1.0 - margin))
        for x in xs:
            centers.append([float(x), yk])

    C = np.array(centers, dtype=float)
    rng.shuffle(C)
    return C


def _seed_edge_ring_26(edge_margin=0.08, interior_ring=0.22, rng=None):
    """
    4 corners + 16 edge points (4 per side) + 6 interior (center + 5-ring) = 26
    """
    if rng is None:
        rng = np.random.default_rng(1234)
    m = edge_margin
    centers = []

    # Corners slightly inset
    corners = [(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m)]
    centers.extend(corners)

    # Helper for interior linspace (exclude exact corners)
    def linspace_interior(a, b, k):
        return np.linspace(a, b, k + 2)[1:-1]

    k_side = 4
    xs = linspace_interior(m, 1 - m, k_side)
    ys = linspace_interior(m, 1 - m, k_side)

    # Bottom and Top
    for x in xs:
        centers.append((x, m))
        centers.append((x, 1 - m))
    # Left and Right
    for y in ys:
        centers.append((m, y))
        centers.append((1 - m, y))

    # Interior: 1 center + ring of 5
    centers.append((0.5, 0.5))
    R = interior_ring
    for t in range(5):
        ang = 2.0 * np.pi * t / 5.0
        centers.append((0.5 + R * np.cos(ang), 0.5 + R * np.sin(ang)))

    C = np.array(centers, dtype=float)
    C = np.clip(C, 1e-3, 1.0 - 1e-3)
    return C


def _seed_corner_weighted_26(margin=0.06, s_in=0.18, rng=None):
    """
    Corner-weighted belt: 4 corners + 12 edge points (3 per side staggered) + small hex-core interior.
    Total = 26.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    C = []

    # Corners
    m = margin
    C.extend([(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m)])

    # Edge belts: 3 per side, staggered
    t = np.linspace(m, 1 - m, 5)[1:-1]  # 3 interior points along an edge
    offsets = [0.0, 0.5 * (t[1] - t[0]), 0.0]
    for idx, x in enumerate(t):
        C.append((x + (offsets[idx] if idx < len(offsets) else 0.0), m))
        C.append((x, 1 - m))
    for idx, y in enumerate(t):
        C.append((m, y + (offsets[idx] if idx < len(offsets) else 0.0)))
        C.append((1 - m, y))

    # Interior hex-core rows [3,4,3]
    rows = [3, 4, 3]
    sv = s_in * np.sqrt(3.0) / 2.0
    y0s = 0.5 + (np.arange(len(rows)) - (len(rows) - 1) / 2.0) * sv
    for ridx, L in enumerate(rows):
        xs = 0.5 + (np.arange(L) - (L - 1) / 2.0) * s_in
        if (ridx % 2) == 1:
            xs = xs + 0.5 * s_in
        for x in xs:
            C.append((float(x), float(y0s[ridx])))

    C = np.array(C, dtype=float)
    rng.shuffle(C)
    C = np.clip(C, 1e-3, 1.0 - 1e-3)
    assert C.shape[0] == 26
    return C


def _seed_spokes_26(edge_margin=0.06, R1=0.22, R2=0.30, rng=None):
    """
    Hybrid seed: 4 corners + 12 edge points (3 per side evenly spaced) +
    two concentric pentagon rings (5 inner at R1, 5 outer at R2) = 26.
    Interior rings are rotated relative to each other.
    """
    if rng is None:
        rng = np.random.default_rng(9)
    m = edge_margin
    centers = []

    # Corners slightly inset
    corners = [(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m)]
    centers.extend(corners)

    # 12 edge points: 3 per side, avoid exact corners
    t = np.linspace(m, 1 - m, 5)[1:-1]
    for x in t:
        centers.append((x, m))
    for x in t:
        centers.append((x, 1 - m))
    for y in t:
        centers.append((m, y))
    for y in t:
        centers.append((1 - m, y))

    # Interior: two concentric pentagons, rotated
    ang0 = 0.0
    for k in range(5):
        ang = ang0 + 2 * np.pi * k / 5.0
        centers.append((0.5 + R1 * np.cos(ang), 0.5 + R1 * np.sin(ang)))
    ang1 = np.pi / 5.0
    for k in range(5):
        ang = ang1 + 2 * np.pi * k / 5.0
        centers.append((0.5 + R2 * np.cos(ang), 0.5 + R2 * np.sin(ang)))

    C = np.array(centers, dtype=float)
    C = np.clip(C, 1e-3, 1.0 - 1e-3)
    assert C.shape[0] == 26
    return C


def _select_seed_with_probe(cand_list, rng=None, eps=1e-6, iters=150, step0=0.05, stepf=0.018, T0=1.2e-3, Tf=1e-5,
                            k_nn=10, safety_every=3, state=None, tau_lp=1e-3, rebuild_M=16):
    """
    Short SA probe per candidate using fast projector only; rank with one exact LP.
    """
    if rng is None:
        rng = np.random.default_rng(303)
    bestC = None
    bestS = -np.inf
    for Ci in cand_list:
        C = np.clip(Ci.copy(), eps, 1.0 - eps)
        knn = _build_knn_indices(C, k=k_nn)
        r_fast = _projected_max_radii_knn(C, None, knn, eps=eps, cycles=2, passes=1, grow_rate=0.5, safety_every=safety_every)
        cur_fast = float(np.sum(r_fast))

        for t in range(iters):
            frac = t / max(1, iters - 1)
            step = step0 * (1.0 - frac) + stepf * frac
            T = T0 * (Tf / max(T0, 1e-16)) ** frac
            prop = C.copy()
            u = rng.uniform()
            if u < 0.80:
                i = rng.integers(0, C.shape[0])
                delta = rng.normal(0.0, step, size=2)
                prop[i] = np.clip(prop[i] + delta, eps, 1.0 - eps)
            elif u < 0.93:
                i = rng.integers(0, C.shape[0])
                bx = min(prop[i, 0], 1.0 - prop[i, 0])
                by = min(prop[i, 1], 1.0 - prop[i, 1])
                if bx < by:
                    delta = np.array([0.0, rng.normal(0.0, step)])
                else:
                    delta = np.array([rng.normal(0.0, step), 0.0])
                prop[i] = np.clip(prop[i] + delta, eps, 1.0 - eps)
            else:
                i = rng.integers(0, C.shape[0])
                Dloc = _pairwise_dists(prop)
                Dloc[i, i] = np.inf
                j = int(np.argmin(Dloc[i]))
                v = prop[j] - prop[i]
                dv = np.linalg.norm(v)
                if dv < 1e-12:
                    ortho = rng.normal(0.0, 1.0, size=2); ortho /= (np.linalg.norm(ortho) + 1e-12)
                else:
                    ortho = np.array([-v[1], v[0]]); ortho /= (np.linalg.norm(ortho) + 1e-12)
                dvec = 0.5 * step * ortho
                prop[i] = np.clip(prop[i] + dvec, eps, 1.0 - eps)
                prop[j] = np.clip(prop[j] - dvec, eps, 1.0 - eps)

            if (t % 20) == 0:
                knn = _build_knn_indices(prop, k=k_nn)
            r_prop = _projected_max_radii_knn(prop, r_fast, knn, eps=eps, cycles=1, passes=1, grow_rate=0.5, safety_every=safety_every)
            s_prop = float(np.sum(r_prop))
            accept = (s_prop >= cur_fast - 1e-15) or (rng.uniform() < np.exp((s_prop - cur_fast) / max(T, 1e-16)))
            if accept:
                C = prop
                r_fast = r_prop
                cur_fast = s_prop

        ri = _compute_max_radii_lp_active(C, state if state is not None else {"active_pairs": set(), "lp_calls": 0, "last_full": -10, "loose_counts": {}},
                                          eps=eps, tau_lp=tau_lp, rebuild_M=rebuild_M, force_full=True)
        si = float(np.sum(ri))
        if si > bestS:
            bestS = si
            bestC = C.copy()
    return bestC


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii