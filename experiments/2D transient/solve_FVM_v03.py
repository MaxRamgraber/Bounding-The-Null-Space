import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# =============================================================================
# Deterministic FVM solve (steady-state or one implicit transient step)
# =============================================================================

def solve_FVM(
    grid,
    T,                  # Dictionary with transmissivities; key: ((t,j),(t,i))
    Sy,                 # Dictionary with specific yields; key: ((t,i))
    R,                  # Dictionary with recharge; key: ((t,i))
    fixed_heads,        # Dictionary with fixed hydraulic heads; key: ((t,i))
    prev_heads=None,    # Dictionary with prior hydraulic heads; key: ((t,i))
    dt=None,            # Time step length for transient steps
    return_flows=False, # Flag to return flow estimates
    gauge_node=None,    # Optional node used to fix the head datum in steady pure-Neumann systems
    gauge_value=0.0,    # Head value imposed at gauge_node in steady pure-Neumann systems
    mass_balance_tol=1e-10  # Tolerance for zero net source/sink in steady pure-Neumann systems
):
    
    """
    Solves hydraulic heads on `grid` using the same discrete forms as in build_constraints.py.

    Steady-state mode:
        For each cell i, across all neighbours j
        sum_j (T*w/dx)*(h_i - h_j) = R_i * A_i

    Transient (single step implicit/backward Euler):
        For each cell i at time t, across all neighbours j
        sum_j (T*w/dx)*(h_i - h_j) + (Sy_i*A_i/dt)*(h_i - h_prev_i) = R_i*A_i

    Notes
    -----
    - `grid` is expected to be a spatial graph (one timestep). Nodes can be ints or tuples.
    - Edge attributes used (if present): "dx", "w"
    - Node attribute used (if present): "area"
    - Dirichlet nodes are removed from unknown set; their contributions go to RHS.
    - If there are no fixed heads in steady state, the system is singular up to an additive
      constant. In that case, a numerical gauge condition h[gauge_node] = gauge_value is added.
    """

    # -----------------------------
    # Unknown set
    # -----------------------------
    
    # Get a dictionary of all fixed heads
    fixed_heads = dict(fixed_heads or {})
    
    # List all the nodes in the grid (may be transient)
    nodes = list(grid.nodes)
    
    # Detect whether this is a transient solve
    transient = (prev_heads is not None)
    
    # Detect the steady-state pure-Neumann case (no fixed heads)
    pure_neumann_steady = (len(fixed_heads) == 0 and not transient)
    
    # In the steady pure-Neumann case, require zero net source/sink for consistency
    if pure_neumann_steady:
        
        # Default to zero source balance
        total_source = 0.0
        
        # Add up all the source and sink terms
        for u in nodes:
            total_source += float(R[u]) * float(grid.nodes[u]["area"])
            
        # Make sure the balance is within tolerance
        if abs(total_source) > mass_balance_tol:
            raise Exception("No fixed_heads provided and net source/sink is nonzero. Steady-state pure-Neumann system is inconsistent. Residual: {}".format(total_source))
    
    # Find all cells with mass-balance equalities
    if pure_neumann_steady:
        unk = list(nodes)
    else: # If we aren't in steady-state with no fixed heads, subtract the fixed-head cells
        unk = [u for u in nodes if u not in fixed_heads]

    # If there are no cells with mass-balance equalities, return trivial solution
    if len(unk) == 0:
        
        # All heads must be fixed
        h = dict(fixed_heads)
        
        # Return the flows, if desired
        if return_flows:
            q = edge_fluxes_from_heads(grid, h, T)
            return h, q
        
        # Else return just the heads
        return h

    # In the steady pure-Neumann case, choose a gauge node if none was provided
    # The gauge_value is necessary to remove the additive null space
    if pure_neumann_steady and gauge_node is None:
        gauge_node = nodes[0]
        print("Warning: no gauge_node specified in pure Neumann steady-state. Selecting head in first node.")

    # Safety check: gauge node must be part of the unknown set
    if pure_neumann_steady and gauge_node not in unk:
        raise ValueError("gauge_node must be a node in the grid.")

    # Create a dictionary of node indices
    idx = {u: k for k, u in enumerate(unk)}
    
    # Get the number of cells with mass balance equalities
    n = len(unk)

    # Initialize a sparse matrix and the RHS vector
    A = scipy.sparse.lil_matrix((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    # =========================================================================
    # Assemble the matrix
    # =========================================================================
    
    # For every active clel
    for i in unk:
        
        # Get the node index of that cell
        ri = idx[i]
        
        # Read out the surface area of that cell
        Ai = float(grid.nodes[i]["area"])
        
        # Initialize the RHS of the mass balance with the recharge in this cell
        rhs = float(R[i]) * Ai
        
        # Initialize the diagonal term of the A matrix; this collects dependencies on the head in this cell
        diag = 0.0

        # Implicit formulation for the storage term
        if transient:
            
            # Read out the specific yield
            Sy_i = float(Sy[i])
            
            # Compute the "constant" part of the storage term
            alpha = Sy_i * Ai / float(dt)
            
            # Add that one to the diagonal
            diag += alpha
            try:
                
                # Add the dependence on the previous head to the RHS
                rhs += alpha * float(prev_heads[i])
                
            # Raise an error if no previous head was defined
            except KeyError:
                raise KeyError(f"prev_heads missing node {i}")

        # Account for the spatial neighbors
        for j in grid.neighbors(i):
            
            # Get canonical edge orientation
            if (j,i) in list(grid.edges):
                u,v = j,i
            else:
                u,v = i,j
            
            # Read out the required parameters
            Tij = float(T[(u, v)]) # Transmissivity along this edge
            dxij = float(grid.edges[(u, v)]["dx"]) # Grid spacing
            wij  = float(grid.edges[(u, v)]["w"]) # Width of flow-active area
            
            # Compute the "constant" part of the flow term
            kij  = Tij * wij / dxij  # matches q = (h_i-h_j)/dx * T * w

            # Add the dependency on the head in cell i to the diagonal
            diag += kij

            # Add the dependency on the neighbour
            if j in fixed_heads:
                
                # If the head is fixed, only add a dependency to the RHS
                rhs += kij * float(fixed_heads[j])
                
            else:
                
                # Else, add a dependency on the head of the neighbour to the A matrix
                cj = idx[j]
                A[ri, cj] += -kij

        # Write the results into the matrix and vector
        A[ri, ri] += diag
        b[ri] = rhs

    # In the steady pure-Neumann case, replace one mass-balance equation by a gauge condition
    if pure_neumann_steady:
        
        # Find the observation_node
        rg = idx[gauge_node]
        
        # Set the diagonal to one, everything else to zero
        A[rg, :] = 0.0
        A[rg, rg] = 1.0
        
        # Set the RHS of that cell to the observation value to enforce a match
        b[rg] = float(gauge_value)

    # Solve the sparse linear system
    A = A.tocsr()
    x = scipy.sparse.linalg.spsolve(A, b)

    # Create a dictionary of all fixed heads
    h = dict(fixed_heads)
    
    # Add keys for every active cells
    for u in unk:
        h[u] = float(x[idx[u]])

    # If desired, also compute the edge fluxes
    if return_flows:
        q = edge_fluxes_from_heads(grid, h, T)
        return h, q

    # Else return only the heads
    return h

# =============================================================================
# Flux postprocessing (for flow directions)
# =============================================================================

def edge_fluxes_from_heads(grid, heads, T):
    """
    Returns fluxes oriented by the edge attributes:
      key = (start, end)
      q > 0 means flow start -> end

    Only computes spatial fluxes (edges that have dx and w).
    """

    q = {}
    for a, b, d in grid.edges(data=True):

        # Skip temporal edges (they have dt but no dx/w)
        if "dx" not in d or "w" not in d:
            continue

        j = d["start"]
        i = d["end"]
        
        # Get canonical edges
        if (j,i) in list(grid.edges):
            u,v = j,i
        else:
            u,v = i,j
        

        dx = float(d["dx"]) 
        w  = float(d["w"])
        
        Tuv = float(T[(u, v)]) 

        # start->end orientation
        q[(u, v)] = (float(heads[u]) - float(heads[v])) / dx * Tuv * w

    return q


