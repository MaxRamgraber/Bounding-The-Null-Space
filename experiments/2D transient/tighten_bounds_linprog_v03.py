"""
This code tightens the bounds of a linear program. 



tighten_bounds_linprog.py (scipy.optimize.linprog / HiGHS backend)

Simplified OBBT (Optimization-Based Bound Tightening) for linear relaxations.

Given:
    A_eq x = b_eq
    A_ub x <= b_ub
    bounds: (lo, hi) per variable

This tightens bounds by solving two LPs per target variable:
    min  +x -> lower bound
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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import scipy.sparse
from scipy.optimize import linprog
import tqdm



# =============================================================================
# Helpers
# =============================================================================

def convert_to_sparse_matrix(A):
    """
    This function takes a dense matrix and converts it into a sparse matrix.
    """
    if A is None:
        return None
    if scipy.sparse.issparse(A):
        return A.tocsr().astype(float)
    return scipy.sparse.csr_matrix(np.asarray(A, dtype=float))


def convert_to_1D_array(x):
    """
    This function takes a vector-like object and converts it into a 1D array.
    """
    if x is None:
        return None
    return np.asarray(x, dtype=float).reshape(-1)


def build_HiGHS_options_dictionary(time_limit, presolve):
    opts = {"presolve": bool(presolve)}
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)
    return opts


def tighten_variable(j, A_eq, b_eq, A_ub, b_ub, lp_bounds, highs_opts, eps=1e-9):
    """
    This function solves extremization operations (minimization and 
    maximization) for a variable in a linear program. If either operation does 
    not solve to optimality, it returns None for that side.
    """
    
    # Get the number of bounds
    n = len(lp_bounds)

    # =========================================================================
    # Minimization / Lower bound: minimize +x_j 
    # =========================================================================
    
    # Set the optimization objective
    c = np.zeros(n, dtype=float)
    c[j] = 1.0
    
    # Call the LP solver
    res_min = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=lp_bounds,
        method="highs-ipm",
        options=highs_opts,
    )
    
    # Extract the marginal coordinate of this optimum
    lo = None # Initiate as None
    if getattr(res_min, "status", None) == 0 and res_min.x is not None:
        lo = res_min.x[j]
        #lo = float(res_min.x[j] - eps * (abs(res_min.x[j]) + 1.0))

    # =========================================================================
    # Maximization / Upper bound: minimize -x_j 
    # =========================================================================

    # Set the optimization objective
    c[j] = -1.0
    
    # Call the LP solver
    res_max = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=lp_bounds,
        method="highs-ipm",
        options=highs_opts,
    )

    # Extract the marginal coordinate of this optimum
    hi = None # Initiate as None
    if getattr(res_max, "status", None) == 0 and res_max.x is not None:
        hi = res_max.x[j]
        # hi = float(res_max.x[j] + eps * (abs(res_max.x[j]) + 1.0))

    # Return the bounds
    return lo, hi


def save_problem_definition(folder, A_eq, b_eq, A_in, b_in, non_collapsed_variables, non_collapsed_bounds):
    
    """
    This function stores a linear program to the disk so that parallel workers 
    can load it from memory, without having to pass large arrays through the 
    multiprocessing.
    """
    
    # Create a folder, if it doesn't exist yet
    os.makedirs(folder, exist_ok=True)

    # Store the sparse constraint matrices
    if A_eq is not None:
        scipy.sparse.save_npz(os.path.join(folder, "A_eq.npz"), A_eq)
    if A_in is not None:
        scipy.sparse.save_npz(os.path.join(folder, "A_in.npz"), A_in)

    # Store the RHS vectors
    with open(os.path.join(folder, "b_eq.pkl"), "wb") as f:
        pickle.dump(b_eq, f)
    with open(os.path.join(folder, "b_in.pkl"), "wb") as f:
        pickle.dump(b_in, f)

    # Store the meta information
    with open(os.path.join(folder, "meta.pkl"), "wb") as f:
        pickle.dump(
            {
                "non_collapsed_variables": non_collapsed_variables,
                "non_collapsed_bounds": non_collapsed_bounds,
            },
            f,
        )


def load_problem_definition(folder):
    
    """
    This function retrieves the information stored in save_problem_definition.
    """
    
    # Create a function to load the sparse matrixces
    def load_stored_sparse_matrices(name):
        p = os.path.join(folder, name)
        return scipy.sparse.load_npz(p) if os.path.exists(p) else None

    # Load the sparse matrices from memory
    A_eq = load_stored_sparse_matrices("A_eq.npz")
    A_in = load_stored_sparse_matrices("A_in.npz")

    # Load the RHS vectors from memory
    with open(os.path.join(folder, "b_eq.pkl"), "rb") as f:
        b_eq = pickle.load(f)
    with open(os.path.join(folder, "b_in.pkl"), "rb") as f:
        b_in = pickle.load(f)

    # Load the meta information
    with open(os.path.join(folder, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    # Return everything
    return A_eq, b_eq, A_in, b_in, meta


def worker(chunk, folder, base_bounds_dict, time_limit, presolve, progress_queue=None):
    
    """
    For multiprocessing, this function defines the tasks of one worker. "Chunk"
    contains a list of variables this worker has to tighten, which is a share 
    of the full variables_to_tighten.
    """
    
    # Start by loading the problem from memory
    A_eq, b_eq, A_in, b_in, meta = load_problem_definition(folder)

    # Extract the meta information
    non_collapsed_variables = meta["non_collapsed_variables"]
    non_collapsed_bounds = meta["non_collapsed_bounds"]

    # Create an index dictionary that returns the index position in 
    # non_collapsed_variables for each variable in non_collapsed_variables.
    v_index = {v: i for i, v in enumerate(non_collapsed_variables)}
    
    # Get a list of the bounds of the non-collapsed variables
    lp_bounds = list(non_collapsed_bounds)
    
    # Create the dictionary with HiGHSoptions
    highs_opts = build_HiGHS_options_dictionary(
        time_limit  = time_limit, 
        presolve    = presolve)

    # Create an output dictionary
    out = {}
    
    # Create a counter for failed extremization operations
    n_failed = 0
    
    # Go through every variable in the chunk
    for vname in chunk:
        
        # If this variable isn't in the non-collapsed variables (i.e., it is
        # collapsed), skip it.
        if vname not in v_index:
            continue
    
        # Find the position index of this variable
        j = v_index[vname]
        
        # Extract the initial bounds
        orig_lo, orig_hi = base_bounds_dict[vname]
    
        # Tighten that variable
        lo, hi = tighten_variable(j, A_eq, b_eq, A_in, b_in, lp_bounds, highs_opts)
    
        # If either tightened bound is None, the LP solver did not solve to
        # optimality. Increment the failure counter.
        failed = (lo is None) or (hi is None)
        if lo is None:
            n_failed += 1
        if hi is None:
            n_failed += 1
    
        # Initiate the tightened bounds as the original bounds, then tighten
        # based on the LP output
        new_lo, new_hi = orig_lo, orig_hi
        if lo is not None:
            new_lo = lo if orig_lo is None else max(orig_lo, lo)
        if hi is not None:
            new_hi = hi if orig_hi is None else min(orig_hi, hi)
    
        # Numerical inversion guard: the lower bound should never become larger
        # than the upper bound. If it does, revert to the original bounds.
        if (new_lo is not None) and (new_hi is not None) and (new_lo > new_hi):
            new_lo, new_hi = orig_lo, orig_hi
    
        # Store the results of this tightening operation
        out[vname] = (new_lo, new_hi, failed)
    
        # Save the progress for the progress bar
        if progress_queue is not None:
            progress_queue.put(1)
    
    # Return the results of a job well done. Hopefully.
    return out, n_failed

#%%

# =============================================================================
# Main function
# =============================================================================

def tighten_bounds(
    A_eq, b_eq, A_in, b_in,
    variables,
    non_collapsed_variables,
    bounds,
    non_collapsed_bounds,
    verbose="changes only",
    variables_to_tighten=None,
    time_limit=1.0,
    presolve=True,
    n_workers=1,
):
    """
    This function implements Optimization-based bound tightening (OBBT) for a 
    defined set of variables using HiGHS. It returns a tightened_bounds_dict in 
    the same structure as the bounds dictionary.
    """
    
    # Check that the verbose flag is in the options
    assert verbose in ["all", "changes only", "off"]

    # Convert the constraint matrices and RHS vectors in the standard format
    A_eq = convert_to_sparse_matrix(A_eq)
    A_in = convert_to_sparse_matrix(A_in)
    b_eq = convert_to_1D_array(b_eq)
    b_in = convert_to_1D_array(b_in)

    # Prepare the list of variables to tighten
    if variables_to_tighten is None:
        variables_to_tighten = list(non_collapsed_variables)
    else:
        variables_to_tighten = list(variables_to_tighten)

    # Create a dictionary of position indices for each variable in 
    # non_collapsed_variables
    v_index = {v: i for i, v in enumerate(non_collapsed_variables)}
    
    # Create a list of bounds for non_collapsed_variables non_
    lp_bounds = list(non_collapsed_bounds)

    # Tighten only variables that haven't collapsed
    targets = [v for v in variables_to_tighten if v in v_index]

    # Create a copy of the original bounds as a basis to store the results
    tightened = copy.deepcopy(bounds)

    # Now we have two paths: a parallelized path, and a serial path. The serial
    # path only becomes active when n_workers = 1

    # =========================================================================
    # Parallel path
    # =========================================================================
    
    # If we have specified a number of workers greater than 1, and there are 
    # actually enough jobs to split
    if n_workers is not None and int(n_workers) > 1 and len(targets) > 0:
        
        # Create a dictionary of bounds
        bounds_dict = {v: tightened[v] for v in targets}

        # Create a temporary directory to store the problem definitions
        with tempfile.TemporaryDirectory() as td:
            
            # Save the problem definition
            save_problem_definition(td, A_eq, b_eq, A_in, b_in, non_collapsed_variables, non_collapsed_bounds)

            # Split the bound tightening operations into chunks
            chunks = [targets[i::int(n_workers)] for i in range(int(n_workers))]
            chunks = [c for c in chunks if len(c) > 0] # Only keep non-empty chunks

            # We use a multiprocessing manager to create a shared queue to show
            # the progress of the OBBT
            with mp.Manager() as mgr:
                
                # Progress bar: one tick per variable, sent from workers via a queue
                progress_queue = mgr.Queue() if (tqdm is not None and len(targets) > 0) else None
                pbar = None
                if progress_queue is not None:
                    pbar = tqdm.tqdm(total=len(targets), desc="OBBT (parallel)")

                # Create a Process Pool Executor
                with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
                    
                    # Submit the jobs for every worker
                    worker_jobs = [
                        ex.submit(
                            worker,
                            c, td, bounds_dict,
                            float(time_limit),
                            bool(presolve),
                            progress_queue,
                        )
                        for c in chunks
                    ]

                    # Consume progress updates from workers
                    if progress_queue is not None and pbar is not None:
                        processed = 0
                        try:
                            while True:
                                try:
                                    inc = progress_queue.get(timeout=0.1)
                                    processed += int(inc)
                                    pbar.update(int(inc))
                                except queue.Empty:
                                    if all(f.done() for f in worker_jobs):
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

                    # Resert the counter for failed updates
                    total_failed = 0
                    
                    # Apply tightened bounds
                    for job in worker_jobs:
                        
                        # Extract the results and increment the failure counter
                        part, n_failed = job.result()
                        total_failed += n_failed
                    
                        # Extract the tightened bounds, failure status, and variable name
                        for vname, (new_lo, new_hi, failed) in part.items():
                            
                            # Extract the initial bounds, then write in the tightened bounds
                            orig_lo, orig_hi = tightened[vname]
                            tightened[vname] = (new_lo, new_hi)
                    
                            # Print information
                            if verbose == "all":
                                if failed:
                                    print(f"{vname}: [{orig_lo}, {orig_hi}] -> failed")
                                else:
                                    print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")
                            elif verbose == "changes only":
                                if failed:
                                    print(f"{vname}: [{orig_lo}, {orig_hi}] -> failed")
                                elif (orig_lo != new_lo) or (orig_hi != new_hi):
                                    print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")

        # Print how many optimizations failed
        print(f"linprog diagnostics: {total_failed} failed optimizations out of {2*len(targets)}")

        # Return the tightened bounds
        return tightened

    # =========================================================================
    # Serial path
    # =========================================================================
    
    # Create a dictionray of HiGHS options
    highs_opts = build_HiGHS_options_dictionary(time_limit=time_limit, presolve=presolve)

    # Create a progress bar
    it = targets
    if tqdm is not None:
        it = tqdm.tqdm(it, desc="OBBT (serial)")
        
        # Initiate a counter of failed optimizations
        total_failed = 0

    # Go through every target variable
    for vname in it:
        
        # Extract the position index of that variable
        j = v_index[vname]
        
        # Extract the original bounds of that variable
        orig_lo, orig_hi = tightened[vname]
    
        # Tighten its bounds
        lo, hi = tighten_variable(j, A_eq, b_eq, A_in, b_in, lp_bounds, highs_opts)
    
        # Increment the failure counter if either extremization did not 
        # optimize successfully
        failed = (lo is None) or (hi is None)
        if lo is None:
            total_failed += 1
        if hi is None:
            total_failed += 1
    
        # Store the new bounds
        new_lo, new_hi = orig_lo, orig_hi
        if lo is not None:
            new_lo = lo if orig_lo is None else max(orig_lo, lo)
        if hi is not None:
            new_hi = hi if orig_hi is None else min(orig_hi, hi)
            
        # Numerical inversion guard: the lower bound should never become larger
        # than the upper bound. If it does, revert to the original bounds.
        if (new_lo is not None) and (new_hi is not None) and (new_lo > new_hi):
            new_lo, new_hi = orig_lo, orig_hi
    
        # Save the new bounds
        tightened[vname] = (new_lo, new_hi)
    
        # Print information
        if verbose == "all":
            if failed:
                print(f"{vname}: [{orig_lo}, {orig_hi}] -> failed")
            else:
                print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")
        elif verbose == "changes only":
            if failed:
                print(f"{vname}: [{orig_lo}, {orig_hi}] -> failed")
            elif (orig_lo != new_lo) or (orig_hi != new_hi):
                print(f"{vname}: [{orig_lo}, {orig_hi}] -> [{new_lo}, {new_hi}]")

    # Return the tightened bounds
    return tightened
