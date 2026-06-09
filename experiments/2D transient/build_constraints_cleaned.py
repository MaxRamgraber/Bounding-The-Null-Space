def build_constraints(variables, bounds, timesteps, grid,
    flow_directions = None, return_sympy = True, 
    T_correlation_edges = [], fixed_head_cells = [],
    fundamental_cycles = None, add_no_curl_constraints = False,
    T_flow_direction_edges = [], T_correlation_frac = 0.20):
    
    """
    This function builds the linear equality and inequality constraints for the
    groundwater linear-program relaxation.
    
    It converts symbolic model relationships into matrices of the form:
        A_eq x = b_eq
        A_in x <= b_in
        
    The function also removes collapsed variables from the LP columns, substitutes
    their fixed values into the constraints, and optionally returns the SymPy
    expressions used to build the matrix rows.
    """
    
    # Import the dependencies used by the constraint builder. They are kept here
    # so the function remains self-contained when copied into another script.
    import numpy as np
    import sympy
    import copy
    import networkx as nx
    import scipy.sparse
    import ast

    # =========================================================================
    # Helper functions
    # =========================================================================

    def xreplace_tol(expr, repl, atol=1e-10, rtol=1e-9):
        """
        This function performs symbolic substitution with a numerical tolerance.

        It behaves like xreplace(), except that it prevents SymPy from collapsing
        relational expressions too early. If a substitution removes all remaining
        free symbols, it evaluates the relation using an absolute and relative
        tolerance and returns S.true or S.false.
        """

        # If there is nothing to substitute, return the original expression.
        if not repl:
            return expr

        # Only special-case SymPy relations such as Eq, LessThan, and 
        # GreaterThan are allowed. Else just substitute directly.
        if isinstance(expr, sympy.core.relational.Relational):

            # Substitute into both sides separately so the relational operator is
            # preserved until we decide whether the expression is fully numeric.
            lhs = expr.lhs.xreplace(repl)
            rhs = expr.rhs.xreplace(repl)

            # If everything is numeric after substitution, check the relation
            # with a tolerance instead of using exact symbolic equality.
            if len((lhs - rhs).free_symbols) == 0:

                # Convert both sides to floats for a numerical comparison.
                lhsf = float(lhs)
                rhsf = float(rhs)

                # Scale the tolerance so larger numbers get a larger allowance.
                scale = max(1.0, abs(lhsf), abs(rhsf))
                tol = atol + rtol * scale

                # Evaluate the relation type using the tolerant comparison.
                if expr.rel_op == "==":
                    return sympy.S.true if abs(lhsf - rhsf) <= tol else sympy.S.false
                elif expr.rel_op == "<=":
                    return sympy.S.true if lhsf <= rhsf + tol else sympy.S.false
                elif expr.rel_op == ">=":
                    return sympy.S.true if lhsf + tol >= rhsf else sympy.S.false

            # If symbols remain, rebuild the relation with evaluate=False so
            # SymPy does not prematurely simplify it to True or False.
            if expr.rel_op == "==":
                return sympy.Eq(lhs, rhs, evaluate=False)
            elif expr.rel_op == "<=":
                return sympy.LessThan(lhs, rhs, evaluate=False)
            elif expr.rel_op == ">=":
                return sympy.GreaterThan(lhs, rhs, evaluate=False)
            else:
                return expr.func(lhs, rhs, evaluate=False)

        # For non-relational expressions, ordinary xreplace is sufficient.
        return expr.xreplace(repl)

    def isolate_variables_to_LHS(expression):
        
        """
        This function takes an expression and isolates all variables to the LHS.
        After this transform:
          - inequalities are in the canonical form   (linear form) <= const
          - equalities   are in the canonical form   (linear form) == const
        The RHS will be a numeric constant (no free symbols).
        """
        
        # Create a copy of the expression
        expression = copy.copy(expression)
        
        if not isinstance(expression, sympy.core.relational.Relational):
            # BooleanTrue  ➜  always satisfied  → caller may skip
            # BooleanFalse ➜  infeasible        → caller should raise
            return expression
        
        # Normalize the relational operator
        op = expression.rel_op  # one of: '<', '<=', '==', '>', '>='
        
        # 1) Move everything to the LHS (as a difference of sides)
        if op == "<=":
            # a <= b becomes a - b <= 0
            expr = sympy.LessThan(expression.lhs - expression.rhs, 0, evaluate=False)
        elif op == "==":
            # a == b becomes a - b == 0
            expr = sympy.Eq(expression.lhs - expression.rhs, 0, evaluate=False)
        elif op == ">=":
            # a >= b becomes b - a <= 0
            expr = sympy.LessThan(expression.rhs - expression.lhs, 0, evaluate=False)
        elif op == "<":
            # a < b becomes a - b < 0
            # (We treat a strict inequality as a non-strict inequality; should be safe for LP)
            expr = sympy.LessThan(expression.lhs - expression.rhs, 0, evaluate=False)
        elif op == ">":
            # a > b becomes b - a < 0
            # (We treat a strict inequality as a non-strict inequality; should be safe for LP)
            expr = sympy.LessThan(expression.rhs - expression.lhs, 0, evaluate=False)
        else:
            raise Exception(f"Unknown relation operator {op}.")
        
        # 2) Split constant vs. variable parts on the LHS and move constant to RHS
        symbols = expr.free_symbols
        const_part, other_part = expr.lhs.as_independent(*symbols, as_Add=True, evaluate=False)
        
        # Re-assemble into either an inequality or an equality
        if expr.rel_op == "<=":
            expr = sympy.LessThan(other_part, -const_part, evaluate=False)
        elif expr.rel_op == "==":
            expr = sympy.Eq(other_part, -const_part, evaluate=False)
        else:
            # Should not happen; keep original
            expr = expression

        # Return the reformulated expression        
        return expr

    def parse_relational_LHS(expression):
        
        """
        This function parses the left-hand side of a canonical SymPy relation.
        
        It breaks the LHS into additive terms and records, for each term:
            - the constant coefficient
            - whether the term is linear or bilinear
            - which variables appear in the term
            
        The output is used to convert symbolic constraints into matrix rows.
        """
        
        # Create a copy of the expression
        expression = copy.copy(expression)
        
        # What variables are in this expression?
        symbols = expression.free_symbols
        
        # Extract the terms on the LHS
        LHS_terms = sympy.Add.make_args(expression.lhs)
        
        # Create a list of constants and term types
        terms = []
        
        # Go through each term
        for term in LHS_terms:
            
            # Split the term into a constant factor and the body
            constant, body = term.as_independent(*symbols, as_Add = False)
            
            # Store the constant coefficient first.
            terms.append({
                "constant": constant})

            # Analyze the symbolic body of the term.
            powdict = body.as_powers_dict() # A dictionary of powers in this term's body
            exps    = list(powdict.values()) # List of exponents in this term's body

            # Classify the term as either linear or bilinear. Anything more
            # nonlinear than bilinear is not supported by this linear relaxation.
            if len(exps) == 1 and exps[0] == 1: # This term is linear
                terms[-1]["type"] = "linear"
            elif len(exps) == 2 and max(exps) == 1: # This term is bilinear
                terms[-1]["type"] = "bilinear"
            else:
                raise Exception("Error parsing constraint. Term {} is neither linear nor bilinear.".format(body))
                
            # Store the symbols involved in this term.
            terms[-1]["vars"] = term.free_symbols

        # Return the parsed term metadata.
        return terms

    def spatial_orientation_from_grid(grid):
        
        """
        Build a lookup that maps each undirected spatial edge in the base graph to
        the orientation that was actually used when the qx symbol was created.
        
        This function is used in the generation of the fundamental cycles in 
        space.
        """
        
        # Create a dictionary indexed by the unordered spatial edge.
        oriented_edge_by_key = {}

        # Go through all graph edges and retain only spatial edges with qx symbols.
        for e in grid.edges:
            ed = grid.edges[e]
            if "qx" not in ed:
                continue

            # Prefer the stored start/end orientation, and otherwise fall back to
            # the edge tuple used by networkx.
            start = ed.get("start", e[0])
            end   = ed.get("end", e[1])

            # Skip temporal edges, since qx is only meaningful for spatial flow.
            if start[0] != end[0]:
                continue

            # Strip the time index so the orientation is stored in spatial cells.
            u = start[1]
            v = end[1]

            # Store the orientation using an unordered key for later lookup.
            oriented_edge_by_key[frozenset((u, v))] = (u, v)

        # Return the orientation dictionary.
        return oriented_edge_by_key


    def signed_fundamental_cycles_from_grid(grid):
        
        """
        Infer a fundamental cycle basis directly from the spatial part of `grid`.
        Each cycle is returned as a list of (u, v, sign), where (u, v) is the edge
        in the same orientation as its qx variable, and sign is +1/-1 depending on
        whether the cycle traverses the edge in that stored direction.
        """
        
        # Start from an undirected spatial graph.
        spatial_graph = nx.Graph()

        # Retrieve the qx orientation for every spatial edge.
        oriented_edge_by_key = spatial_orientation_from_grid(grid)

        # Add every spatial edge to the undirected graph used for cycle detection.
        for key in oriented_edge_by_key:
            uu, vv = tuple(key)
            spatial_graph.add_edge(uu, vv)

        # Create a list of signed fundamental cycles.
        signed_cycles = []

        # Use networkx to find a cycle basis for the spatial graph.
        for cycle_nodes in nx.cycle_basis(spatial_graph):

            # Ignore degenerate cycles.
            if len(cycle_nodes) < 3:
                continue

            # Convert the node cycle into signed directed edges.
            cycle = []
            for idx, u in enumerate(cycle_nodes):

                # Get the next node in the cycle, wrapping around at the end.
                v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
                key = frozenset((u, v))

                # Make sure this cycle edge has an orientation in the grid.
                if key not in oriented_edge_by_key:
                    raise KeyError("Cycle edge {} not found in spatial orientation lookup.".format((u, v)))

                # Compare the cycle traversal direction to the stored qx direction.
                eu, ev = oriented_edge_by_key[key]
                sign = +1 if (u, v) == (eu, ev) else -1

                # Store the edge in qx orientation plus its signed contribution.
                cycle.append((eu, ev, sign))

            # Store this signed cycle.
            signed_cycles.append(cycle)

        # Return all signed cycles.
        return signed_cycles


    def normalize_fundamental_cycles(fundamental_cycles, grid):
        """
        Accept either
          - cycles as lists of spatial nodes, or
          - cycles as lists of (u, v, sign) tuples,
        and return the signed-edge representation expected by the no-curl block.
        """
        # If no cycles were provided, infer a cycle basis from the grid.
        if fundamental_cycles is None:
            return signed_fundamental_cycles_from_grid(grid)

        # Otherwise, prepare to normalize the user-provided cycles.
        oriented_edge_by_key = spatial_orientation_from_grid(grid)
        normalized_cycles = []

        # Go through each cycle specification.
        for cycle_idx, cycle in enumerate(fundamental_cycles):

            # Skip empty cycle definitions.
            if cycle is None or len(cycle) == 0:
                continue

            # Inspect the first entry to determine the input format.
            first = cycle[0]

            # The cycle is already in signed-edge form.
            if isinstance(first, (tuple, list)) and len(first) == 3 and isinstance(first[2], (int, float, np.integer, np.floating)):
                # Validate and copy the signed-edge entries.
                signed_cycle = []
                for entry in cycle:
                    u, v, sign = entry

                    # Normalize any nonzero numeric sign to -1 or +1.
                    sign = int(np.sign(sign))
                    if sign not in (-1, 1):
                        raise ValueError("Fundamental cycle {} has invalid sign {} on edge {}.".format(cycle_idx, entry[2], (u, v)))
                    signed_cycle.append((u, v, sign))

                # Store this normalized cycle and move to the next one.
                normalized_cycles.append(signed_cycle)
                continue

            # Otherwise, interpret the cycle as an ordered list of spatial nodes.
            cycle_nodes = list(cycle)
            signed_cycle = []

            # Convert each adjacent node pair into a signed qx-oriented edge.
            for idx, u in enumerate(cycle_nodes):
                v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
                key = frozenset((u, v))

                # Make sure the edge exists in the spatial orientation map.
                if key not in oriented_edge_by_key:
                    raise KeyError("Cycle edge {} not found in spatial orientation lookup.".format((u, v)))

                # Use the stored qx orientation and record the traversal sign.
                eu, ev = oriented_edge_by_key[key]
                sign = +1 if (u, v) == (eu, ev) else -1
                signed_cycle.append((eu, ev, sign))

            # Store the normalized cycle.
            normalized_cycles.append(signed_cycle)

        # Return all normalized cycles.
        return normalized_cycles
    
    # =========================================================================
    # Initial bookkeeping
    # =========================================================================

    # Find all nodes and edges in the graph
    nodes = list(grid.nodes)
    edges = list(grid.edges)

    # First, check which variables have been collapsed to a single fixed value.
    collapsed = {}
    non_collapsed_variables = [] # Non-collapsed variables
    for var in variables:
        
        # A variable is collapsed if both bounds exist and lower bound equals upper bound.
        if (not (bounds[var][0] is None or bounds[var][1] is None)) and np.diff(bounds[var]) == 0: # This variable is collapsed
            collapsed[var] = True
        else: # This variable is not collapsed

            # Non-collapsed variables become columns in the final LP matrices.
            collapsed[var] = False
            non_collapsed_variables.append(var)

    # Also mark variables that appear in bounds but not in the variables list.
    for k, (L, U) in bounds.items():
        if k not in collapsed:
            collapsed[k] = (L is not None and U is not None and U == L)

    # Normalize the fundamental cycles into signed-edge form.
    # Each entry becomes a list of (u, v, sign), where (u, v) follows the qx
    # orientation stored on the grid edge, and sign is the traversal sign in the
    # cycle equation.
    fundamental_cycles = normalize_fundamental_cycles(fundamental_cycles, grid)
    
    # Get the number of non-collapsed variables.
    num_variables = len(non_collapsed_variables)

    # Create a fast lookup from variable name to LP column index.
    non_collapsed_index = {v: k for k, v in enumerate(non_collapsed_variables)}

    # Initiate arrays for the constraints.
    A_eq = [] # Equality constraints
    A_in = [] # Inequality constraints A x <= b
    b_eq = [] # RHS of the equalities
    b_in = [] # RHS of the inequalities
    descr_eq = [] # Descriptions of the equalities
    descr_in = [] # Descriptions of the inequalities
    sympy_eq = [] # A list for the equality SymPy expressions
    sympy_in = [] # A list for the inequality SymPy expressions
    
    # =========================================================================
    # Correlate transmissivities (star pattern for T_correlation_edges)
    # =========================================================================

    # If we have specified correlation edges
    if T_correlation_edges is not None and len(T_correlation_edges) > 0:

        # Convert the fractional tolerance into lower and upper multiplicative factors.
        rho = float(T_correlation_frac)
        alpha_hi = 1.0 + rho

        # You cannot correlate more than 1.0 =P.
        if rho >= 1.0:
            raise ValueError("T_correlation_frac must be < 1.0")

        # Create a fast index lookup for non-collapsed variables.
        nc_idx = {v: k for k, v in enumerate(non_collapsed_variables)}

        # This helper strips the time index from a time-layered node.
        def node_key(u):
            
            # If u has the form (t, i), return only the spatial cell index i.
            if isinstance(u, tuple) and len(u) == 2 and isinstance(u[0], (int, np.integer)):
                return u[1]

            # Otherwise, the node is already a spatial node key. May happen in
            # steady-state systems.
            return u

        # Map "spatial edge" -> T variable name by scanning the graph once.
        # Key is (min(node_key), max(node_key)) to be order-independent.
        T_by_pair = {}
        
        # Go through every edge in the grid
        for e in grid.edges:

            # Extract edge attributes and skip edges without transmissivity.
            # (i.e., temporal edges)
            ed = grid.edges[e]
            if "T" not in ed:
                continue

            # Prefer explicit start/end if present; else fall back to edge endpoints
            if "start" in ed and "end" in ed:
                u = ed["start"]
                v = ed["end"]
            else:
                u, v = e[0], e[1]

            # Convert endpoint nodes to spatial keys.
            uk = node_key(u)
            vk = node_key(v)

            # Create an order-independent key for this spatial edge.
            key = tuple(sorted((uk, vk), key=str))

            # One representative T variable is enough, since T is often shared
            # across all time layers for the same spatial edge.
            if key not in T_by_pair:
                T_by_pair[key] = str(ed["T"])

        # Build incident sets per node, but only using user-provided correlation edges
        # Accept edges either as (i,j) or as ((t,i),(t,j)) – we strip time via node_key.
        incident_T = {}
        
        # Go through all correlated edges
        for e in T_correlation_edges:
            
            # If this edge has a weird format, skip it
            if not isinstance(e, (tuple, list)) or len(e) < 2:
                continue
            
            # Extract the start and end nodes, then extract the cell index
            u, v = e[0], e[1]
            uk = node_key(u)
            vk = node_key(v)
            key = tuple(sorted((uk, vk), key=str))
            
            # If this edge does not have a transmissivity variable, skip it.
            # Shouldn't happen in practice.
            if key not in T_by_pair:
                continue

            # Retrieve the transmissivity variable
            Tvar = T_by_pair[key]
            
            # This function is funky. If incident_T[uk] does not yet exist, 
            # create an empty set with that key, then add Tvar.
            incident_T.setdefault(uk, set()).add(Tvar)
            incident_T.setdefault(vk, set()).add(Tvar)

        # Helper: add coef * var to a single-row LHS, respecting collapsed vars
        def add_linear_term_to_constraint_row(A_row, b_entry, var, coef):

            # If the variable has collapsed, move its contribution to the RHS.
            if collapsed.get(var, False):
                return b_entry - float(coef) * float(bounds[var][0])

            # Otherwise, add the coefficient to the appropriate LP column.
            A_row[0, nc_idx[var]] += float(coef)
            
            # Return the RHS entry. Because this is a scalar variable, we can't
            # just edit it globally. Python variable weirdness.
            return b_entry

        # Add all-pairs transmissivity comparison constraints around each node.
        for nodekey, Tset in incident_T.items():

            # A node needs at least two incident transmissivities to compare them.
            if len(Tset) < 2:
                continue

            # Sort the transmissivities to make the row order deterministic.
            Ts = sorted(Tset)

            # Go through every unordered pair of transmissivity variables that
            # touch this node. If a node has transmissivities [T1, T2, T3], this
            # loop creates the pairs (T1,T2), (T1,T3), and (T2,T3), without
            # repeating pairs in the opposite order.
            for a in range(len(Ts)):
                for b in range(a + 1, len(Ts)):

                    # Extract the two transmissivity variables in this pair.
                    # These are variable names/keys, not numerical values.
                    Ta = Ts[a]
                    Tb = Ts[b]

                    # =========================================================
                    # First ratio constraint:
                    #
                    #     Ta <= alpha_hi * Tb
                    #
                    # This prevents Ta from being too large relative to Tb.
                    # Written in standard LP inequality form A x <= b:
                    #
                    #     Ta - alpha_hi * Tb <= 0
                    #
                    # The row A_row stores the coefficients of the active
                    # variables on the LHS, and b_entry stores the RHS.
                    # =========================================================

                    # Create an empty sparse row for one inequality constraint.
                    A_row = scipy.sparse.lil_matrix((1, num_variables))

                    # Initialize the right-hand side.
                    b_entry = 0.0

                    # Add +1.0 * Ta to the LHS.
                    b_entry = add_linear_term_to_constraint_row(A_row, b_entry, Ta, +1.0)

                    # Add -alpha_hi * Tb to the LHS.
                    b_entry = add_linear_term_to_constraint_row(A_row, b_entry, Tb, -alpha_hi)

                    # Store this inequality row
                    A_in.append(A_row)
                    b_in.append(b_entry)

                    # Store a description of this constraint.
                    descr_in.append(f"T corr for cell {nodekey}: {Ta} <= {alpha_hi}*{Tb}")

                    # =========================================================
                    # Second ratio constraint:
                    #
                    #     Tb <= alpha_hi * Ta
                    #
                    # This prevents Tb from being too large relative to Ta.
                    # Together with the first constraint, this gives:
                    #
                    #     max(Ta, Tb) / min(Ta, Tb) <= alpha_hi
                    #
                    # assuming Ta and Tb are positive.
                    #
                    # Written in standard LP inequality form:
                    #
                    #     Tb - alpha_hi * Ta <= 0
                    # =========================================================

                    # Create a new empty sparse row for the opposite-direction
                    # ratio constraint.
                    A_row = scipy.sparse.lil_matrix((1, num_variables))

                    # Initialize the RHS for this second inequality.
                    b_entry = 0.0

                    # Add +1.0 * Tb to the LHS
                    b_entry = add_linear_term_to_constraint_row(A_row, b_entry, Tb, +1.0)

                    # Add -alpha_hi * Ta to the LHS
                    b_entry = add_linear_term_to_constraint_row(A_row, b_entry, Ta, -alpha_hi)

                    # Store the second inequality row.
                    A_in.append(A_row)
                    b_in.append(b_entry)

                    # Store a description of this constraint.
                    descr_in.append(f"T corr for cell {nodekey}: {Tb} <= {alpha_hi}*{Ta}")

    # =========================================================================
    # Prescribe head differences
    # =========================================================================
    

    # Go through all edges
    for edge in edges:
        
        # Extract cell indices and timestep
        tj, j = grid.edges[edge]["start"]
        ti, i = grid.edges[edge]["end"]
        
        # Check if both nodes are at the same timestep
        if tj == ti :
            
            # Create the sympy constraint dhx_ji = (h_j - h_i) / dx
            constraint = sympy.Eq(
                grid.edges[edge]["dhx"], 
                (grid.nodes[(tj,j)]["h"] - grid.nodes[(ti,i)]["h"])/grid.edges[edge]["dx"], 
                evaluate = False)
            
            # Substitute in any degenerate bounds
            var = grid.edges[edge]["dhx"]
            if collapsed[str(var)]: constraint = xreplace_tol(constraint, {var: bounds[str(var)][0]})
            var = grid.nodes[(tj,j)]["h"]
            if collapsed[str(var)]: constraint = xreplace_tol(constraint, {var: bounds[str(var)][0]})
            var = grid.nodes[(ti,i)]["h"]
            if collapsed[str(var)]: constraint = constraint = xreplace_tol(constraint, {var: bounds[str(var)][0]})
            
            # Bring the constraint into canonical form
            constraint = isolate_variables_to_LHS(constraint)
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):
                # If the constraint becomes a boolean, it either becomes a 
                # tautology (True) or a contradiction (False)
                if constraint: # Tautologies aren't a problem.
                    continue
                else: # Contradictions imply the system is unfeasible
                    print(f"Error detected in dhx definition for edge {edge}.")
                    print(f"    dht: {bounds[str(grid.edges[edge]['dhx'])]}")
                    print(f"    h {edge[0]}: {bounds[str(grid.nodes[(tj,j)]['h'])]}")
                    print(f"    h {edge[1]}: {bounds[str(grid.nodes[(ti,i)]['h'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False")
            
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # This term is a linear equality by definition
            A_row = scipy.sparse.lil_matrix((1,num_variables))
            for term in terms:
                if list(term["vars"])[0] == grid.edges[edge]["dhx"]:
                    A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["dhx"]))] += term["constant"]
                elif list(term["vars"])[0] == grid.nodes[(tj,j)]["h"]:
                    A_row[0,non_collapsed_variables.index(str(grid.nodes[(tj,j)]["h"]))] += term["constant"]
                elif list(term["vars"])[0] == grid.nodes[(ti,i)]["h"]:
                    A_row[0,non_collapsed_variables.index(str(grid.nodes[(ti,i)]["h"]))] += term["constant"]
                else: # This should never happen.
                    raise Exception("Term {} in head difference constraint parsing for identifier {} not understood. Terms are {}.".format((tj,j,i),term["vars"],term))
            
            # Add the results to the inequality constraints
            A_eq.append(copy.copy(A_row))
            b_eq.append(float(constraint.rhs))
            descr_eq.append("spatial head difference constraint {}".format((tj,j,i)))
            sympy_eq.append(copy.copy(constraint))
            
            
        else: # No, the edges are at different timesteps
        
            # Create the sympy constraint dht_i = h_i_{t} - h_i_{t-1}
            constraint = sympy.Eq(
                grid.edges[edge]["dht"], 
                (grid.nodes[(ti,i)]["h"] - grid.nodes[(tj,j)]["h"])/grid.edges[edge]["dt"], 
                evaluate = False)
            
            # Substitute in any degenerate bounds
            var = grid.edges[edge]["dht"]
            if collapsed[str(var)]: constraint = xreplace_tol(constraint, {var: bounds[str(var)][0]})
            var = grid.nodes[(tj,j)]["h"]
            if collapsed[str(var)]: constraint = xreplace_tol(constraint, {var: bounds[str(var)][0]})
            var = grid.nodes[(ti,i)]["h"]
            if collapsed[str(var)]: constraint = xreplace_tol(constraint, {var: bounds[str(var)][0]})
            
            # Bring the constraint into canonical form
            constraint = isolate_variables_to_LHS(constraint)
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):
                # If the constraint becomes a boolean, it either becomes a 
                # tautology (True) or a contradiction (False)
                if constraint: # Tautologies aren't a problem.
                    continue
                else: # Contradictions imply the system is unfeasible
                    print(f"Error detected in dht definition for edge {edge}.")
                    print(f"    dht: {bounds[str(grid.edges[edge]['dht'])]}")
                    print(f"    h {edge[0]}: {bounds[str(grid.nodes[(tj,j)]['h'])]}")
                    print(f"    h {edge[1]}: {bounds[str(grid.nodes[(ti,i)]['h'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False")
            
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # This term is a linear equality by definition
            A_row = scipy.sparse.lil_matrix((1,num_variables))
            for term in terms:
                if list(term["vars"])[0] == grid.edges[edge]["dht"]:
                    A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["dht"]))] += term["constant"]
                elif list(term["vars"])[0] == grid.nodes[(tj,j)]["h"]:
                    A_row[0,non_collapsed_variables.index(str(grid.nodes[(tj,j)]["h"]))] += term["constant"]
                elif list(term["vars"])[0] == grid.nodes[(ti,i)]["h"]:
                    A_row[0,non_collapsed_variables.index(str(grid.nodes[(ti,i)]["h"]))] += term["constant"]
                else: # This should never happen.
                    raise Exception("Term {} in head difference constraint parsing for identifier {} not understood. Terms are {}.".format((tj,ti,j),term["vars"],term))
            
            # Add the results to the inequality constraints
            A_eq.append(copy.copy(A_row))
            b_eq.append(float(constraint.rhs))
            descr_eq.append("temporal head difference constraint {}".format((tj,ti,i)))
            sympy_eq.append(copy.copy(constraint))

    
    # =========================================================================
    # Define Convex Hull with McCormick for spatial fluxes
    # =========================================================================
    
    # Go through all edges
    for edge in list(grid.edges):
        
        # Unpack the time indices and cell indices
        tj, j = grid.edges[edge]["start"]
        ti, i = grid.edges[edge]["end"]
        
        # Only consider spatial edges within the same timestep
        if tj == ti:
            
            # Set the time index
            t = tj
            
            # -----------------------------------------------------------------
            # Extract edge-attached symbols
            # -----------------------------------------------------------------
            T   = grid.edges[edge]["T"]
            dhx = grid.edges[edge]["dhx"]
            qx  = grid.edges[edge]["qx"]
            w   = grid.edges[edge]["w"]
            
            # Create the bilinear flux definition qx = dhx * T * w.
            constraint = sympy.Eq(qx, dhx*T*w, evaluate = False)
            
            # Substitute in any degenerate bounds
            var = str(qx)
            if collapsed[var]: constraint = constraint.xreplace({qx: bounds[var][0]})
            var = str(dhx)
            if collapsed[var]: constraint = constraint.xreplace({dhx: bounds[var][0]})
            var = str(T)
            if collapsed[var]: constraint = constraint.xreplace({T: bounds[var][0]})
            
            # Save the bilinear constraint
            sympy_eq.append(copy.copy(constraint))
            
            # Bring the constraint into canonical form
            constraint = isolate_variables_to_LHS(constraint)
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):
                # If the constraint becomes a boolean, it either becomes a 
                # tautology (True) or a contradiction (False)
                if constraint: # Tautologies aren't a problem.
                    continue
                else: # Contradictions imply the system is unfeasible
                    print(f"Error detected in McCormick definition for edge {edge}.")
                    print(f"    qx: {bounds[str(grid.edges[edge]['qx'])]}")
                    print(f"    T: {bounds[str(grid.edges[edge]['T'])]}")
                    print(f"    dhx: {bounds[str(grid.edges[edge]['dhx'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False")
                    
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # Check whether all bilinear variables were collapsed. If so, this
            # equation has reduced to an ordinary linear equality.

            # -----------------------------------------------------------------
            # This term has become linear
            # -----------------------------------------------------------------
            
            if all([True if term["type"] == "linear" else False for term in terms]):
                
                A_row = scipy.sparse.lil_matrix((1,num_variables))
                for term in terms:
                    if list(term["vars"])[0] == qx:
                        A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["qx"]))] += term["constant"]
                    elif list(term["vars"])[0] == dhx:
                        A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["dhx"]))] += term["constant"]
                    elif list(term["vars"])[0] == T:
                        A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["T"]))] += term["constant"]
                    else: # This should never happen.
                        raise Exception("Term {} in McCormick constraint parsing for identifier {} not understood. Terms are {}.".format((t,j,i),term["vars"],term))
                        
                # Add the results to the inequality constraints
                A_eq.append(copy.copy(A_row))
                b_eq.append(float(constraint.rhs))
                descr_eq.append("qx {} linear constraint".format((t,j,i)))
            
            # -----------------------------------------------------------------
            # This term remains bilinear
            # -----------------------------------------------------------------
            
            else: 
                
                # -------------------------------------------------------------
                # This term is still bilinear, so we relax it with McCormick planes.
                # -------------------------------------------------------------

                # Implement a standard four-plane McCormick relaxation
                for (dhx_bound, T_bound, sense) in [(0,0,"<="),(1,1,"<="),(1,0,">="),(0,1,">=")]:
                    
                    # Create placeholders for the linear inequality constraints
                    A_row = scipy.sparse.lil_matrix((1, num_variables))
                    b_entry = 0.0
                    
                    # Read the bounds of dhx and T
                    coef_dhx = bounds[str(grid.edges[edge]["dhx"])][dhx_bound]
                    coef_T   = bounds[str(grid.edges[edge]["T"])][T_bound]
                    
                    # Add flux dependence
                    if collapsed[str(grid.edges[edge]["qx"])]: 
                        b_entry -= bounds[str(grid.edges[edge]["qx"])][0]
                    else: 
                        A_row[0, non_collapsed_variables.index(str(grid.edges[edge]["qx"]))] += -1.0
                        
                    # Add transmissivity dependence
                    if collapsed[str(grid.edges[edge]["T"])]: 
                        b_entry += w * coef_dhx * bounds[str(grid.edges[edge]["T"])][0]
                    else: 
                        A_row[0, non_collapsed_variables.index(str(grid.edges[edge]["T"]))] += w * coef_dhx
                        
                    # Add head gradient dependence
                    if collapsed[str(grid.edges[edge]["dhx"])]: 
                        b_entry += w * coef_T * bounds[str(grid.edges[edge]["dhx"])][0]
                    else: 
                        A_row[0, non_collapsed_variables.index(str(grid.edges[edge]["dhx"]))] += w * coef_T
                        
                    # Add the McCormick corner constant to the RHS.
                    b_entry += w * coef_dhx * coef_T

                    # Store either the <= plane directly or multiply by -1 to
                    # convert a >= plane into the matrix convention A_in x <= b_in.
                    if sense == "<=": 
                        A_in.append(A_row) 
                        b_in.append(b_entry)
                    else: 
                        A_in.append(-A_row)
                        b_in.append(-b_entry)
                    descr_in.append(f"qx {(t,j,i)} McCormick relaxation")
                    
        else:            
        
            # Set the time index
            t = tj
            
            # -----------------------------------------------------------------
            # Extract edge-attached symbols
            # -----------------------------------------------------------------
            Sy  = grid.nodes[(t,i)]["Sy"]
            dht = grid.edges[edge]["dht"]
            qt  = grid.edges[edge]["qt"]
            A   = grid.nodes[(t,i)]["area"]
            
            # Create the bilinear temporal flux definition qt = dht * Sy * A.
            constraint = sympy.Eq(qt, dht*Sy*A, evaluate = False)
            
            # Substitute in any degenerate bounds
            var = str(qt)
            if collapsed[var]: constraint = constraint.xreplace({qt: bounds[var][0]})
            var = str(dht)
            if collapsed[var]: constraint = constraint.xreplace({dht: bounds[var][0]})
            var = str(Sy)
            if collapsed[var]: constraint = constraint.xreplace({Sy: bounds[var][0]})
            
            # Save the bilinear constraint
            sympy_eq.append(copy.copy(constraint))
            
            # Bring the constraint into canonical form
            constraint = isolate_variables_to_LHS(constraint)
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):
                # If the constraint becomes a boolean, it either becomes a 
                # tautology (True) or a contradiction (False)
                if constraint: # Tautologies aren't a problem.
                    continue
                else: # Contradictions imply the system is unfeasible
                    print(f"Error detected in McCormick definition for edge {edge}.")
                    print(f"    qt: {bounds[str(grid.edges[edge]['qt'])]}")
                    print(f"    Sy: {bounds[str(grid.nodes[(t,i)]['Sy'])]}")
                    print(f"    dhx: {bounds[str(grid.edges[edge]['qt'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False")
            
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # Check whether all bilinear variables were collapsed. If so, this
            # equation has reduced to an ordinary linear equality.

            # -----------------------------------------------------------------
            # This term has become linear
            # -----------------------------------------------------------------
            
            if all([True if term["type"] == "linear" else False for term in terms]):
                
                A_row = scipy.sparse.lil_matrix((1,num_variables))
                for term in terms:
                    if list(term["vars"])[0] == qt:
                        A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["qt"]))] += term["constant"]
                    elif list(term["vars"])[0] == dht:
                        A_row[0,non_collapsed_variables.index(str(grid.edges[edge]["dht"]))] += term["constant"]
                    elif list(term["vars"])[0] == Sy:
                        A_row[0,non_collapsed_variables.index(str(grid.nodes[(tj,j)]["Sy"]))] += term["constant"]
                    else: # This should never happen.
                        raise Exception("Term {} in McCormick constraint parsing for identifier {} not understood. Terms are {}.".format((t,j,i),term["vars"],term))
                        
                # Add the results to the inequality constraints
                A_eq.append(copy.copy(A_row))
                b_eq.append(float(constraint.rhs))
                descr_eq.append("qt {} linear constraint".format((tj,ti,i)))
            
            # -----------------------------------------------------------------
            # This term remains bilinear
            # -----------------------------------------------------------------
            
            else: 
                
                # -------------------------------------------------------------
                # This term is still bilinear, so we relax it with McCormick planes.
                # -------------------------------------------------------------

                # Implement a standard four-plane McCormick relaxation
                for (dht_bound, Sy_bound, sense) in [(0,0,"<="),(1,1,"<="),(1,0,">="),(0,1,">=")]:
                    
                    # Create placeholders for the linear inequality constraints
                    A_row = scipy.sparse.lil_matrix((1, num_variables))
                    b_entry = 0.0
                    
                    # Read the bounds of dhx and T
                    coef_dht = bounds[str(grid.edges[edge]["dht"])][dht_bound]
                    coef_Sy  = bounds[str(grid.nodes[(ti,i)]["Sy"])][Sy_bound]
                    
                    # Add flux dependence
                    if collapsed[str(grid.edges[edge]["qt"])]: b_entry -= bounds[str(grid.edges[edge]["qt"])][0]
                    else: A_row[0, non_collapsed_variables.index(str(grid.edges[edge]["qt"]))] += -1.0
                        
                    # Add Sy dependence
                    if collapsed[str(grid.nodes[(tj,j)]["Sy"])]: b_entry += A * coef_dht * bounds[str(grid.nodes[(tj,j)]["Sy"])][0]
                    else: A_row[0, non_collapsed_variables.index(str(grid.nodes[(tj,j)]["Sy"]))] += A * coef_dht
                        
                    # Add head gradient dependence
                    if collapsed[str(grid.edges[edge]["dht"])]: b_entry += A * coef_Sy * bounds[str(grid.edges[edge]["dht"])][0]
                    else: A_row[0, non_collapsed_variables.index(str(grid.edges[edge]["dht"]))] += A * coef_Sy
                        
                    # Add the McCormick corner constant to the RHS.
                    b_entry += A * coef_dht * coef_Sy

                    # Store either the <= plane directly or multiply by -1 to
                    # convert a >= plane into the matrix convention A_in x <= b_in.
                    if sense == "<=": 
                        A_in.append(A_row) 
                        b_in.append(b_entry)
                    else: 
                        A_in.append(-A_row)
                        b_in.append(-b_entry)
                    descr_in.append(f"qt {(t,j,i)} McCormick relaxation")
                    
    # =========================================================================
    # Enforce no-curl on all fundamental spatial cycles
    # =========================================================================
    
    # Do we have to enforce irrotationality constraints?
    if add_no_curl_constraints and fundamental_cycles is not None and len(fundamental_cycles) > 0:

        # Go through every fundamental spatial cycle.
        for cycle_idx, cycle in enumerate(fundamental_cycles):

            # Skip empty cycles.
            if cycle is None or len(cycle) == 0:
                continue

            # Apply each no-curl cycle constraint at every timestep.
            for t in range(timesteps):

                # Create placeholders for the qx circulation constraint.
                A_row = scipy.sparse.lil_matrix((1, num_variables))
                b_entry = 0.0
                constraint_lhs = sympy.Integer(0)
                valid_cycle = True

                # First enforce zero signed circulation of qx around the cycle.
                for u, v, sign in cycle:

                    # Convert the signed spatial edge to this timestep.
                    sign = float(sign)
                    edge_id = ((t, u), (t, v))

                    # Some spatial edges may be absent in a given time layer,
                    # e.g. if the whole edge was removed because both cells are fixed-head.
                    if not grid.has_edge(*edge_id):
                        valid_cycle = False
                        break

                    # Extract the spatial flux and add its signed contribution.
                    qx = grid.edges[edge_id]["qx"]
                    constraint_lhs += int(sign) * qx

                    # If qx is collapsed, move it to the RHS. Otherwise, add it
                    # to the row using its non-collapsed column index.
                    var = str(qx)
                    if collapsed[var]:
                        b_entry -= sign * float(bounds[var][0])
                    else:
                        A_row[0, non_collapsed_index[var]] += sign

                # Skip the cycle at this timestep if one of its edges was missing.
                if not valid_cycle:
                    continue

                # Store the symbolic no-curl constraint.
                constraint = sympy.Eq(constraint_lhs, 0, evaluate = False)
                sympy_eq.append(copy.copy(constraint))

                # If all variables in the row collapsed, check whether the
                # resulting constant equation is feasible.
                if A_row.nnz == 0:
                    if abs(b_entry) <= 1e-10:
                        continue
                    raise ValueError(
                        "Infeasible no-curl constraint on cycle {} at timestep {}: all involved qx variables are collapsed, but the signed circulation is {:.6e}.".format(
                            cycle_idx, t, -b_entry)
                    )

                A_eq.append(copy.copy(A_row))
                b_eq.append(float(b_entry))
                descr_eq.append("no-curl fundamental cycle {}".format((t, cycle_idx)))
                
                
                
                
                
                A_row = scipy.sparse.lil_matrix((1, num_variables))
                b_entry = 0.0
                constraint_lhs = sympy.Integer(0)
                valid_cycle = True

                # Then enforce zero signed circulation of dhx around the cycle.
                for u, v, sign in cycle:

                    # Convert the signed spatial edge to this timestep.
                    sign = float(sign)
                    edge_id = ((t, u), (t, v))

                    # Some spatial edges may be absent in a given time layer,
                    # e.g. if the whole edge was removed because both cells are fixed-head.
                    if not grid.has_edge(*edge_id):
                        valid_cycle = False
                        break

                    # Extract the head-gradient variable and add its signed contribution.
                    qx = grid.edges[edge_id]["dhx"]
                    constraint_lhs += int(sign) * qx

                    # If dhx is collapsed, move it to the RHS. Otherwise, add it
                    # to the row using its non-collapsed column index.
                    var = str(qx)
                    if collapsed[var]:
                        b_entry -= sign * float(bounds[var][0])
                    else:
                        A_row[0, non_collapsed_index[var]] += sign

                # Skip the cycle at this timestep if one of its edges was missing.
                if not valid_cycle:
                    continue

                # Store the symbolic no-curl constraint.
                constraint = sympy.Eq(constraint_lhs, 0, evaluate = False)
                sympy_eq.append(copy.copy(constraint))

                # If all variables in the row collapsed, check whether the
                # resulting constant equation is feasible.
                if A_row.nnz == 0:
                    if abs(b_entry) <= 1e-10:
                        continue
                    raise ValueError(
                        "Infeasible no-curl constraint on cycle {} at timestep {}: all involved qx variables are collapsed, but the signed circulation is {:.6e}.".format(
                            cycle_idx, t, -b_entry)
                    )

                # Add the dhx no-curl constraint to the equality system.
                A_eq.append(copy.copy(A_row))
                b_eq.append(float(b_entry))
                descr_eq.append("no-curl fundamental cycle dhx {}".format((t, cycle_idx)))
                
                
                

    # =========================================================================
    # Define mass balances
    # =========================================================================
    
    for t in range(timesteps):

        # Loop over non-fixed-head cells only.
        for i in list(set(range(len(nodes))) - set(fixed_head_cells)):
            
            # Skip if this node does not exist in the graph (safety)
            if (t, i) not in grid.nodes:
                continue
            
            # Create an empty equality constraint
            constraint = sympy.Eq(0, 0, evaluate = False)
            
            # -----------------------------------------------------------------
            # Spatial flow term (net outflow convention)
            # -----------------------------------------------------------------
            
            # Define qx variables
            qx = {}
            
            # Use graph connectivity to find neighbours within the same timestep
            for nbr in grid.neighbors((t, i)):
                
                # Unpack the neighboring node.
                tj, j = nbr

                # Ignore temporal links while assembling the spatial flow term.
                if tj != t:
                    continue  # ignore any temporal links (if present)
                
                # Retrieve the unique edge identifier for this timestep
                # Determine the canonical edge order (sorted to match grid.edges keys)
                if j < i:
                    identifier = ((t, j), (t, i))
                else:
                    identifier = ((t, i), (t, j))
                    
                # # Canonical spatial edge and sign:
                # # qx_(jj,ii) is positive from jj → ii
                # jj, ii = (j, i) if j < i else (i, j)
                # sign = +1.0 if i == jj else -1.0
                
                # Canonical spatial edge sign from variable orientation:
                # qx_(t, a, b) is positive from a → b
                try:
                    _tt, a, b = ast.literal_eval(str(grid.edges[identifier]["qx"]).replace("qx_", "", 1))
                except Exception as e:
                    raise Exception("Could not parse qx orientation from {}".format(grid.edges[identifier]["qx"])) from e
                
                # Use the qx orientation to determine whether the flux is
                # outflow or inflow for the current cell.
                if i == a:
                    sign = +1.0  # outflow from node i
                elif i == b:
                    sign = -1.0  # inflow to node i
                else:
                    raise Exception("qx {} is not incident to node {}".format(grid.edges[identifier]["qx"], (t, i)))

                
                # Pull the UNIQUE flux symbol from the grid edge
                qx_sym = grid.edges[identifier]["qx"]
                qx[j]  = qx_sym  # keep your dict structure
                
                # # Net outflow with area scaling (from recharge)
                # factor = 1 / grid.edges[identifier]["area"]
                
                # Add the flow term to the SymPy equality
                constraint = sympy.Eq(constraint.lhs + qx[j] * sign,
                                      constraint.rhs, evaluate = False)
                
                # Check if the flux has collapsed
                var = str(qx[j])
                if collapsed[var]:
                    constraint = constraint.xreplace({qx[j]: bounds[var][0]})
                    
            # -----------------------------------------------------------------
            # Temporal flow term
            # -----------------------------------------------------------------
            
            if t > 0:
            
                # Retrieve the unique edge identifier for this timestep
                identifier = ((t-1, i), (t, i))
                
                # Pull the UNIQUE flux symbol from the grid edge
                qt = grid.edges[identifier]["qt"]
                
                # Add the flow term to the SymPy equality
                constraint = sympy.Eq(constraint.lhs + qt,
                                      constraint.rhs, evaluate = False)
                
                # Check if the flux has collapsed
                var = str(qt)
                if collapsed[var]:
                    constraint = constraint.xreplace({qt: bounds[var][0]})
            
            # -----------------------------------------------------------------
            # Recharge term
            # -----------------------------------------------------------------
            
            # Add the recharge attached to this node
            R = grid.nodes[(t, i)]["R"]
            constraint = sympy.Eq(constraint.lhs, constraint.rhs + R * grid.nodes[(t, i)]["area"], evaluate = False)
            
            # Check if the recharge has collapsed
            var = str(R)
            if collapsed[var]:
                constraint = constraint.xreplace({R: bounds[var][0]})
            
            # -----------------------------------------------------------------
            # Analyze the expression and assemble the equality constraint
            # -----------------------------------------------------------------
            
            # Bring the constraint into canonical form
            constraint = isolate_variables_to_LHS(constraint)
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):      # catches S.true / S.false ONLY
                if constraint:                           # S.true  ➜  tautology
                    continue
                else:                                    # S.false ➜  contradiction
                    print(f"Error detected in Mass balance definition for cell {i}.")
                    print(f"    R: {bounds[str(grid.nodes[(t, i)]['R'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False")
            
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # This term is an equality by definition
            A_row = scipy.sparse.lil_matrix((1, num_variables))
            for term in terms:
                v = list(term["vars"])[0]
                A_row[0, non_collapsed_variables.index(str(v))] += term["constant"]
            
            # Add the results to the equality constraints
            A_eq.append(copy.copy(A_row))
            b_eq.append(float(constraint.rhs))
            descr_eq.append("mass balance {}".format((t, i)))
            sympy_eq.append(copy.copy(constraint))

    def to_csr(mat_list):
        """
        This function converts a list of sparse rows or a sparse matrix to CSR.
        """

        # Preserve missing constraint blocks as None.
        if mat_list is None:
            return None

        # If we accumulated a list of rows, stack them into one sparse matrix.
        if isinstance(mat_list, list):                 # exactly one row
            mat_list = scipy.sparse.vstack(mat_list, format="csr")
        elif not scipy.sparse.isspmatrix_csr(mat_list):

            # Convert any other sparse matrix format to CSR.
            mat_list = mat_list.tocsr()

        # Return the CSR matrix.
        return mat_list
    
    
    # Convert the equality constraints into one CSR matrix, if any exist.
    if A_eq != []:
        A_eq = to_csr(A_eq)
    else:
        A_eq = None
    # Convert the inequality constraints into one CSR matrix, if any exist.
    if A_in != []:
        A_in = to_csr(A_in)
    else:
        A_in = None
        
    # Get the bounds for the non-collapsed variables in LP column order.
    non_collapsed_bounds = [bounds[var] for var in non_collapsed_variables]
    
    # Return either the matrix data plus SymPy constraints, or just the matrix data.
    if return_sympy:
        return A_eq, b_eq, descr_eq, sympy_eq, A_in, b_in, descr_in, sympy_in, non_collapsed_variables, non_collapsed_bounds
    else:
        return A_eq, b_eq, descr_eq, A_in, b_in, descr_in, non_collapsed_variables, non_collapsed_bounds
