"""
tighten_bounds_linprog.py (scipy.optimize.linprog / HiGHS backend)

Simplified OBBT (Optimization-Based Bound Tightening) for linear relaxations.

Given:
    A_eq x = b_eq
    A_ub x <= b_ub
    bounds: (lo, hi) per variable

This tightens bounds by solving two LPs per target variable:
    min  +x_j   -> lower bound
    min  -x_j   -> upper bound (via x_j at optimum)

Parallel progress:
- If n_workers > 1 and tqdm is available, a per-variable progress bar is shown.
- Each worker sends one "tick" after finishing each variable via a Manager().Queue().

Important differences vs SCIP version:
- Bounds are only updated when HiGHS returns an OPTIMAL solution for that LP.
  (No dual-bound tightening on early termination.)
- MILP integrality is not supported (linprog is continuous LP only).

Windows note:
- If you run with n_workers > 1 on Windows, ensure the *caller* is protected by:
      if __name__ == "__main__":
  (Your previous ProcessPoolExecutor usage likely already required this.)

"""

from __future__ import annotations

import copy
import os
import pickle
import queue
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor

import multiprocessing as mp
import numpy as np
import scipy.sparse
from scipy.optimize import linprog, OptimizeWarning

try:
    import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =============================================================================
# Helpers
# =============================================================================

def _as_csr(A):
    if A is None:
        return None
    if scipy.sparse.issparse(A):
        return A.tocsr().astype(float)
    return scipy.sparse.csr_matrix(np.asarray(A, dtype=float))


def _as_1d_float(x):
    if x is None:
        return None
    return np.asarray(x, dtype=float).reshape(-1)


def _highs_options(time_limit, presolve):
    # SciPy expects bool for presolve
    opts = {"presolve": bool(presolve)}
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)
    return opts


def _tighten_one_linprog(j, A_eq, b_eq, A_ub, b_ub, lp_bounds, highs_opts, eps=1e-9):
    """
    Returns (lo, hi) where lo/hi are conservative tightened bounds when the
    corresponding LP solves to optimality; otherwise None for that side.
    """
    n = len(lp_bounds)

    def _solve(c):
        return linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=lp_bounds,
            method="highs-ipm",
            options=highs_opts,
        )

    # -- minimize +x_j (lower bound) --
    c = np.zeros(n, dtype=float)
    c[j] = 1.0
    res_min = _solve(c)

    lo = None
    if getattr(res_min, "status", None) == 0 and res_min.x is not None:
        lo = float(res_min.x[j] - eps * (abs(res_min.x[j]) + 1.0))

    # -- minimize -x_j (upper bound) --
    c[j] = -1.0
    res_max = _solve(c)

    hi = None
    if getattr(res_max, "status", None) == 0 and res_max.x is not None:
        hi = float(res_max.x[j] + eps * (abs(res_max.x[j]) + 1.0))

    return lo, hi


def _dump_problem(folder, A_eq, b_eq, A_in, b_in, non_collapsed_variables, non_collapsed_bounds):
    os.makedirs(folder, exist_ok=True)

    if A_eq is not None:
        scipy.sparse.save_npz(os.path.join(folder, "A_eq.npz"), A_eq)
    if A_in is not None:
        scipy.sparse.save_npz(os.path.join(folder, "A_in.npz"), A_in)

    with open(os.path.join(folder, "b_eq.pkl"), "wb") as f:
        pickle.dump(b_eq, f)
    with open(os.path.join(folder, "b_in.pkl"), "wb") as f:
        pickle.dump(b_in, f)

    with open(os.path.join(folder, "meta.pkl"), "wb") as f:
        pickle.dump(
            {
                "non_collapsed_variables": non_collapsed_variables,
                "non_collapsed_bounds": non_collapsed_bounds,
            },
            f,
        )


def _load_problem(folder):
    def _load_npz(name):
        p = os.path.join(folder, name)
        return scipy.sparse.load_npz(p) if os.path.exists(p) else None

    A_eq = _load_npz("A_eq.npz")
    A_in = _load_npz("A_in.npz")

    with open(os.path.join(folder, "b_eq.pkl"), "rb") as f:
        b_eq = pickle.load(f)
    with open(os.path.join(folder, "b_in.pkl"), "rb") as f:
        b_in = pickle.load(f)

    with open(os.path.join(folder, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    return A_eq, b_eq, A_in, b_in, meta


def _worker(chunk, folder, base_bounds_dict, time_limit, presolve, progress_queue=None):
    """
    Worker tightens exactly the variables listed in `chunk`.
    (This is effectively that worker's share of variables_to_tighten.)
    """
    A_eq, b_eq, A_in, b_in, meta = _load_problem(folder)

    non_collapsed_variables = meta["non_collapsed_variables"]
    non_collapsed_bounds = meta["non_collapsed_bounds"]

    v_index = {v: i for i, v in enumerate(non_collapsed_variables)}
    lp_bounds = list(non_collapsed_bounds)
    highs_opts = _highs_options(time_limit=time_limit, presolve=presolve)

    out = {}
    for vname in chunk:
        if vname not in v_index:
            continue

        j = v_index[vname]
        orig_lo, orig_hi = base_bounds_dict[vname]

        lo, hi = _tighten_one_linprog(j, A_eq, b_eq, A_in, b_in, lp_bounds, highs_opts)

        new_lo, new_hi = orig_lo, orig_hi
        if lo is not None:
            new_lo = lo if orig_lo is None else max(orig_lo, lo)
        if hi is not None:
            new_hi = hi if orig_hi is None else min(orig_hi, hi)

        # Numerical inversion guard
        if (new_lo is not None) and (new_hi is not None) and (new_lo > new_hi):
            new_lo, new_hi = orig_lo, orig_hi

        out[vname] = (new_lo, new_hi)

        if progress_queue is not None:
            progress_queue.put(1)

    return out


# =============================================================================
# Public API
# =============================================================================

def tighten_bounds(
    A_eq, b_eq, A_in, b_in,
    variables,                   # kept for drop-in compatibility
    non_collapsed_variables,
    bounds,                      # dict: varname -> (lo,hi)
    non_collapsed_bounds,        # list aligned with non_collapsed_variables
    verbose="changes only",
    variables_to_tighten=None,
    time_limit=1.0,
    presolve=True,
    integrality=None,
    n_workers=1,
):
    """
    OBBT bound tightening using scipy.optimize.linprog (HiGHS).

    Returns:
        tightened_bounds_dict (same structure as `bounds`)
    """
    assert verbose in ["all", "changes only", "off"]

    # linprog can't do MILP integrality; keep behavior explicit.
    if integrality is not None:
        if isinstance(integrality, dict):
            if any(bool(v) for v in integrality.values()):
                raise ValueError("linprog backend does not support integrality (MILP).")
        else:
            arr = np.asarray(integrality)
            if arr.size and np.any(arr.astype(bool)):
                raise ValueError("linprog backend does not support integrality (MILP).")

    # Convert to CSR / float
    A_eq = _as_csr(A_eq)
    A_in = _as_csr(A_in)
    b_eq = _as_1d_float(b_eq)
    b_in = _as_1d_float(b_in)

    if variables_to_tighten is None:
        variables_to_tighten = list(non_collapsed_variables)
    else:
        variables_to_tighten = list(variables_to_tighten)

    v_index = {v: i for i, v in enumerate(non_collapsed_variables)}
    lp_bounds = list(non_collapsed_bounds)

    # Tighten only variables that are actually present in this relaxation
    targets = [v for v in variables_to_tighten if v in v_index]

    tightened = copy.deepcopy(bounds)

    # -------------------------------------------------------------------------
    # Parallel path
    # -------------------------------------------------------------------------
    if n_workers is not None and int(n_workers) > 1 and len(targets) > 0:
        bounds_dict = {v: tightened[v] for v in targets}

        with tempfile.TemporaryDirectory() as td:
            _dump_problem(td, A_eq, b_eq, A_in, b_in, non_collapsed_variables, non_collapsed_bounds)

            chunks = [targets[i::int(n_workers)] for i in range(int(n_workers))]
            chunks = [c for c in chunks if len(c) > 0]

            # Progress bar: one tick per variable, sent from workers via a queue
            with mp.Manager() as mgr:
                progress_queue = mgr.Queue() if (tqdm is not None and len(targets) > 0) else None
                pbar = None
                if progress_queue is not None:
                    pbar = tqdm.tqdm(total=len(targets), desc="tighten (linprog, parallel)")

                with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
                    futs = [
                        ex.submit(
                            _worker,
                            c, td, bounds_dict,
                            float(time_limit),
                            bool(presolve),
                            progress_queue,
                        )
                        for c in chunks
                    ]

                    # Consume progress updates from workers (1 tick per variable)
                    if progress_queue is not None and pbar is not None:
                        processed = 0
                        try:
                            while True:
                                try:
                                    inc = progress_queue.get(timeout=0.1)
                                    processed += int(inc)
                                    pbar.update(int(inc))
                                except queue.Empty:
                                    if all(f.done() for f in futs):
                                        break
                        finally:
                            # Drain any remaining updates
                            while True:
                                try:
                                    inc = progress_queue.get_nowait()
                                    processed += int(inc)
                                    pbar.update(int(inc))
                                except queue.Empty:
                                    break
                            # If any updates were missed, fast-forward
                            if processed < len(targets):
                                pbar.update(len(targets) - processed)
                            pbar.close()

                    # Apply tightened bounds
                    for fut in futs:
                        part = fut.result()
                        for vname, (new_lo, new_hi) in part.items():
                            orig_lo, orig_hi = tightened[vname]
                            tightened[vname] = (new_lo, new_hi)

                            if verbose == "all":
                                print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")
                            elif verbose == "changes only":
                                if (orig_lo != new_lo) or (orig_hi != new_hi):
                                    print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")

        return tightened

    # -------------------------------------------------------------------------
    # Serial path
    # -------------------------------------------------------------------------
    highs_opts = _highs_options(time_limit=time_limit, presolve=presolve)

    it = targets
    if tqdm is not None:
        it = tqdm.tqdm(it, desc="tighten (linprog)")

    for vname in it:
        j = v_index[vname]
        orig_lo, orig_hi = tightened[vname]

        lo, hi = _tighten_one_linprog(j, A_eq, b_eq, A_in, b_in, lp_bounds, highs_opts)

        new_lo, new_hi = orig_lo, orig_hi
        if lo is not None:
            new_lo = lo if orig_lo is None else max(orig_lo, lo)
        if hi is not None:
            new_hi = hi if orig_hi is None else min(orig_hi, hi)

        if (new_lo is not None) and (new_hi is not None) and (new_lo > new_hi):
            new_lo, new_hi = orig_lo, orig_hi

        tightened[vname] = (new_lo, new_hi)

        if verbose == "all":
            print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")
        elif verbose == "changes only":
            if (orig_lo != new_lo) or (orig_hi != new_hi):
                print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")

    return tightened
