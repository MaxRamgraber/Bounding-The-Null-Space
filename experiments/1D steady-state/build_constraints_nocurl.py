def build_constraints(variables, bounds, timesteps, grid,
    flow_directions = None, return_sympy = True, 
    T_correlation_edges = [], fixed_head_cells = [],
    fundamental_cycles = None, add_no_curl_constraints = False,
    T_flow_direction_edges = [], T_correlation_frac = 0.20):
    
    import numpy as np
    import sympy
    import copy
    import networkx as nx
    import scipy.sparse
    import ast

    def xreplace_tol(expr, repl, atol=1e-10, rtol=1e-9):
        """
        Like xreplace(), but prevents premature Eq/LessThan evaluation.
        If substitution removes all free symbols, compare numerically with tolerance
        and return S.true/S.false.
        """
        if not repl:
            return expr

        # Only special-case SymPy relations (Eq, <=, >=, ...)
        if isinstance(expr, sympy.core.relational.Relational):

            lhs = expr.lhs.xreplace(repl)
            rhs = expr.rhs.xreplace(repl)

            # "last DOF": everything numeric now -> accept within tolerance
            if len((lhs - rhs).free_symbols) == 0:
                lhsf = float(lhs)
                rhsf = float(rhs)
                scale = max(1.0, abs(lhsf), abs(rhsf))
                tol = atol + rtol * scale

                if expr.rel_op == "==":
                    return sympy.S.true if abs(lhsf - rhsf) <= tol else sympy.S.false
                elif expr.rel_op == "<=":
                    return sympy.S.true if lhsf <= rhsf + tol else sympy.S.false
                elif expr.rel_op == ">=":
                    return sympy.S.true if lhsf + tol >= rhsf else sympy.S.false
                else:
                    # unexpected relation type
                    return sympy.S.false

            # Still has symbols: rebuild relation with evaluate=False (no boolean collapse)
            if expr.rel_op == "==":
                return sympy.Eq(lhs, rhs, evaluate=False)
            elif expr.rel_op == "<=":
                return sympy.LessThan(lhs, rhs, evaluate=False)
            elif expr.rel_op == ">=":
                return sympy.GreaterThan(lhs, rhs, evaluate=False)
            else:
                return expr.func(lhs, rhs, evaluate=False)

        # Non-relational expressions: normal xreplace is fine
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
            expr = sympy.LessThan(expression.lhs - expression.rhs, 0, evaluate=False)
        elif op == "==":
            expr = sympy.Eq(expression.lhs - expression.rhs, 0, evaluate=False)
        elif op == ">=":
            # a >= b  ⇔  b - a <= 0
            expr = sympy.LessThan(expression.rhs - expression.lhs, 0, evaluate=False)
        elif op == "<":
            # treat strict as non-strict (safe for LP)
            expr = sympy.LessThan(expression.lhs - expression.rhs, 0, evaluate=False)
        elif op == ">":
            expr = sympy.LessThan(expression.rhs - expression.lhs, 0, evaluate=False)
        else:
            # Unknown relation: return as-is; caller will likely raise
            return expression
        
        # 2) Split constant vs. variable parts on the LHS and move constant to RHS
        symbols = expr.free_symbols
        const_part, other_part = expr.lhs.as_independent(*symbols, as_Add=True, evaluate=False)
        
        if expr.rel_op == "<=":
            expr = sympy.LessThan(other_part, -const_part, evaluate=False)
        elif expr.rel_op == "==":
            expr = sympy.Eq(other_part, -const_part, evaluate=False)
        else:
            # Should not happen; keep original
            expr = expression
        
        return expr


    
    def parse_relational_LHS(expression):
        
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
            
            # Split the term in a constant factor and the body
            constant, body = term.as_independent(*symbols, as_Add = False)
            
            # Append the results
            terms.append({
                "constant": constant})
            
            # Analyze the body
            powdict = body.as_powers_dict() # A dictionary of powers in this term's body
            exps    = list(powdict.values()) # List of exponents in this term's body
            if len(exps) == 1 and exps[0] == 1: # This term is linear
                # term_type.append("linear")
                terms[-1]["type"] = "linear"
            elif len(exps) == 2 and max(exps) == 1: # This term is bilinear
                # term_type.append("bilinear")
                terms[-1]["type"] = "bilinear"
            else:
                raise Exception("Error parsing constraint. Term {} is neither linear nor bilinear.".format(body))
                
            terms[-1]["vars"] = term.free_symbols
        
        return terms

    def _spatial_orientation_from_grid(grid):
        """
        Build a lookup that maps each undirected spatial edge in the base graph to
        the orientation that was actually used when the qx symbol was created.
        """
        oriented_edge_by_key = {}
        for e in grid.edges:
            ed = grid.edges[e]
            if "qx" not in ed:
                continue

            start = ed.get("start", e[0])
            end   = ed.get("end", e[1])
            if start[0] != end[0]:
                continue

            u = start[1]
            v = end[1]
            oriented_edge_by_key[frozenset((u, v))] = (u, v)

        return oriented_edge_by_key


    def _signed_fundamental_cycles_from_grid(grid):
        """
        Infer a fundamental cycle basis directly from the spatial part of `grid`.
        Each cycle is returned as a list of (u, v, sign), where (u, v) is the edge
        in the same orientation as its qx variable, and sign is +1/-1 depending on
        whether the cycle traverses the edge in that stored direction.
        """
        spatial_graph = nx.Graph()
        oriented_edge_by_key = _spatial_orientation_from_grid(grid)

        for key in oriented_edge_by_key:
            uu, vv = tuple(key)
            spatial_graph.add_edge(uu, vv)

        signed_cycles = []
        for cycle_nodes in nx.cycle_basis(spatial_graph):
            if len(cycle_nodes) < 3:
                continue

            cycle = []
            for idx, u in enumerate(cycle_nodes):
                v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
                key = frozenset((u, v))
                if key not in oriented_edge_by_key:
                    raise KeyError("Cycle edge {} not found in spatial orientation lookup.".format((u, v)))

                eu, ev = oriented_edge_by_key[key]
                sign = +1 if (u, v) == (eu, ev) else -1
                cycle.append((eu, ev, sign))

            signed_cycles.append(cycle)

        return signed_cycles


    def _normalize_fundamental_cycles(fundamental_cycles, grid):
        """
        Accept either
          - cycles as lists of spatial nodes, or
          - cycles as lists of (u, v, sign) tuples,
        and return the signed-edge representation expected by the no-curl block.
        """
        if fundamental_cycles is None:
            return _signed_fundamental_cycles_from_grid(grid)

        oriented_edge_by_key = _spatial_orientation_from_grid(grid)
        normalized_cycles = []

        for cycle_idx, cycle in enumerate(fundamental_cycles):
            if cycle is None or len(cycle) == 0:
                continue

            first = cycle[0]

            # Already in signed-edge form
            if isinstance(first, (tuple, list)) and len(first) == 3 and isinstance(first[2], (int, float, np.integer, np.floating)):
                signed_cycle = []
                for entry in cycle:
                    u, v, sign = entry
                    sign = int(np.sign(sign))
                    if sign not in (-1, 1):
                        raise ValueError("Fundamental cycle {} has invalid sign {} on edge {}.".format(cycle_idx, entry[2], (u, v)))
                    signed_cycle.append((u, v, sign))
                normalized_cycles.append(signed_cycle)
                continue

            # Otherwise interpret the cycle as an ordered list of spatial nodes
            cycle_nodes = list(cycle)
            signed_cycle = []
            for idx, u in enumerate(cycle_nodes):
                v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
                key = frozenset((u, v))
                if key not in oriented_edge_by_key:
                    raise KeyError("Cycle edge {} not found in spatial orientation lookup.".format((u, v)))

                eu, ev = oriented_edge_by_key[key]
                sign = +1 if (u, v) == (eu, ev) else -1
                signed_cycle.append((eu, ev, sign))

            normalized_cycles.append(signed_cycle)

        return normalized_cycles
    
    
    # Find all nodes and edges
    nodes = list(grid.nodes)
    edges = list(grid.edges)
    
    # First, check how many variables have been collapsed
    collapsed = {}
    non_collapsed_variables = [] # Non-collapsed variables
    for var in variables:
        if (not (bounds[var][0] is None or bounds[var][1] is None)) and np.diff(bounds[var]) == 0: # This variable is collapsed
            collapsed[var] = True
        else: # This variable is not collapsed
            collapsed[var] = False
            non_collapsed_variables.append(var)
            
    # After building `collapsed` from `variables`:
    for k, (L, U) in bounds.items():
        if k not in collapsed:
            collapsed[k] = (L is not None and U is not None and U == L)

    
    # Normalize the fundamental cycles into signed-edge form.
    # Each entry becomes a list of (u, v, sign), where (u, v) follows the qx
    # orientation stored on the grid edge, and sign is the traversal sign in the
    # cycle equation.
    fundamental_cycles = _normalize_fundamental_cycles(fundamental_cycles, grid)
    
    
    # Get the number of non-collapsed variables
    num_variables = len(non_collapsed_variables)
    non_collapsed_index = {v: k for k, v in enumerate(non_collapsed_variables)}
    
    # Initiate arrays for the constraints
    A_eq = [] # Equality constraints
    A_in = [] # Inequality constraints A x <= b
    b_eq = []
    b_in = []
    descr_eq = []
    descr_in = []
    sympy_eq = [] # A list for the sympy expressions
    sympy_in = [] # A list for the sympy expressions
    
    # =========================================================================
    # Correlate transmissivities (star pattern, restricted to T_correlation_edges)
    # =========================================================================

    if T_correlation_edges is not None and len(T_correlation_edges) > 0:

        rho = float(T_correlation_frac)
        alpha_lo = 1.0 - rho
        alpha_hi = 1.0 + rho
        if alpha_lo <= 0.0:
            raise ValueError("T_correlation_frac must be < 1.0")

        # Fast index lookup for non-collapsed vars
        nc_idx = {v: k for k, v in enumerate(non_collapsed_variables)}

        # Treat (t,i) nodes as "spatial i" when present (works for non-time grids too)
        def _node_key(u):
            if isinstance(u, tuple) and len(u) == 2 and isinstance(u[0], (int, np.integer)):
                return u[1]
            return u

        # Map "spatial edge" -> T variable name by scanning the graph once.
        # Key is (min(node_key), max(node_key)) to be order-independent.
        T_by_pair = {}
        for e in grid.edges:
            ed = grid.edges[e]
            if "T" not in ed:
                continue

            # Prefer explicit start/end if present; else fall back to edge endpoints
            if "start" in ed and "end" in ed:
                u = ed["start"]
                v = ed["end"]
            else:
                u, v = e[0], e[1]

            uk = _node_key(u)
            vk = _node_key(v)
            key = tuple(sorted((uk, vk), key=str))
            # representative is enough (often T is already shared across time layers)
            if key not in T_by_pair:
                T_by_pair[key] = str(ed["T"])

        # Build incident sets per node, but ONLY using user-provided correlation edges
        # Accept edges either as (i,j) or as ((t,i),(t,j)) – we strip time via _node_key.
        incident_T = {}  # node_key -> set(T_var_str)
        for e in T_correlation_edges:
            if not isinstance(e, (tuple, list)) or len(e) < 2:
                continue
            u, v = e[0], e[1]
            uk = _node_key(u)
            vk = _node_key(v)
            key = tuple(sorted((uk, vk), key=str))
            if key not in T_by_pair:
                continue

            Tvar = T_by_pair[key]
            incident_T.setdefault(uk, set()).add(Tvar)
            incident_T.setdefault(vk, set()).add(Tvar)

        # Helper: add coef * var to a single-row LHS, respecting collapsed vars
        def _add_lin(A_row, b_entry, var, coef):
            if collapsed.get(var, False):
                return b_entry - float(coef) * float(bounds[var][0])
            A_row[0, nc_idx[var]] += float(coef)
            return b_entry

        # All-pairs constraints per node
        for nkey, Tset in incident_T.items():

            if len(Tset) < 2:
                continue

            Ts = sorted(Tset)

            for a in range(len(Ts)):
                for b in range(a + 1, len(Ts)):

                    Ta = Ts[a]
                    Tb = Ts[b]

                    # Ta <= alpha_hi * Tb  ->  Ta - alpha_hi*Tb <= 0
                    A_row = scipy.sparse.lil_matrix((1, num_variables))
                    b_entry = 0.0
                    b_entry = _add_lin(A_row, b_entry, Ta, +1.0)
                    b_entry = _add_lin(A_row, b_entry, Tb, -alpha_hi)
                    A_in.append(A_row)
                    b_in.append(b_entry)
                    descr_in.append(f"T corr @ {nkey}: {Ta} <= {alpha_hi}*{Tb}")

                    # Tb <= alpha_hi * Ta  ->  Tb - alpha_hi*Ta <= 0
                    A_row = scipy.sparse.lil_matrix((1, num_variables))
                    b_entry = 0.0
                    b_entry = _add_lin(A_row, b_entry, Tb, +1.0)
                    b_entry = _add_lin(A_row, b_entry, Ta, -alpha_hi)
                    A_in.append(A_row)
                    b_in.append(b_entry)
                    descr_in.append(f"T corr @ {nkey}: {Tb} <= {alpha_hi}*{Ta}")


        # # Alternative: Star constraints per node
        # for nkey, Tset in incident_T.items():
        #     if len(Tset) < 2:
        #         continue

        #     Ts = sorted(Tset)           # deterministic
        #     anchor = Ts[0]              # star anchor
        #     for other in Ts[1:]:
        #         if other == anchor:
        #             continue

        #         # other <= alpha_hi * anchor  ->  other - alpha_hi*anchor <= 0
        #         A_row = scipy.sparse.lil_matrix((1, num_variables))
        #         b_entry = 0.0
        #         b_entry = _add_lin(A_row, b_entry, other, +1.0)
        #         b_entry = _add_lin(A_row, b_entry, anchor, -alpha_hi)
        #         A_in.append(A_row)
        #         b_in.append(b_entry)
        #         descr_in.append(f"T corr @ {nkey}: {other} <= {alpha_hi}*{anchor}")

        #         # other >= alpha_lo * anchor  ->  -other + alpha_lo*anchor <= 0
        #         A_row = scipy.sparse.lil_matrix((1, num_variables))
        #         b_entry = 0.0
        #         b_entry = _add_lin(A_row, b_entry, other, -1.0)
        #         b_entry = _add_lin(A_row, b_entry, anchor, +alpha_lo)
        #         A_in.append(A_row)
        #         b_in.append(b_entry)
        #         descr_in.append(f"T corr @ {nkey}: {other} >= {alpha_lo}*{anchor}")

    
    
    
    
    # =========================================================================
    # Prescribe head differences
    # =========================================================================
    
    for edge in edges:
        
        # Extract cell indices and timestep
        tj, j = grid.edges[edge]["start"]
        ti, i = grid.edges[edge]["end"]
        
        # Check if both nodes are at the same timestep
        if tj == ti :
            
            # Create the sympy constraint dhx_ji = h_j - h_i
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
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):      # catches S.true / S.false ONLY
                if constraint:                           # S.true  ➜  tautology
                    continue
                else:                                    # S.false ➜  contradiction
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
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):      # catches S.true / S.false ONLY
                if constraint:                           # S.true  ➜  tautology
                    continue
                else:                                    # S.false ➜  contradiction
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
            
            # Check if the constraint is still undefined
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
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):      # catches S.true / S.false ONLY
                if constraint: 
                    continue # S.true  ➜  tautology
                else: 
                    print(f"Error detected in McCormick definition for edge {edge}.")
                    print(f"    qx: {bounds[str(grid.edges[edge]['qx'])]}")
                    print(f"    T: {bounds[str(grid.edges[edge]['T'])]}")
                    print(f"    dhx: {bounds[str(grid.edges[edge]['dhx'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False") # S.false ➜  contradiction
            
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # Check if this term is linear
            
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
                # This edge is not ambiguous
                # -------------------------------------------------------------
                
                # Implement a standard four-edge McCormick relaxation
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
                        
                    # RHS corner constant
                    b_entry += w * coef_dhx * coef_T
                    if sense == "<=": 
                        A_in.append(A_row) 
                        b_in.append(b_entry)
                    else: 
                        A_in.append(-A_row)
                        b_in.append(-b_entry)
                    descr_in.append(f"qx {(t,j,i)} McCormick 4-plane")
                    
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
            
            # Check if the constraint is still undefined
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
            if isinstance(constraint, sympy.logic.boolalg.BooleanAtom):      # catches S.true / S.false ONLY
                if constraint: 
                    continue # S.true  ➜  tautology
                else: 
                    print(f"Error detected in McCormick definition for edge {edge}.")
                    print(f"    qt: {bounds[str(grid.edges[edge]['qt'])]}")
                    print(f"    Sy: {bounds[str(grid.nodes[(t,i)]['Sy'])]}")
                    print(f"    dhx: {bounds[str(grid.edges[edge]['qt'])]}")
                    raise ValueError("Infeasible: a constraint collapsed to False") # S.false ➜  contradiction
            
            # Analyze the LHS
            terms = parse_relational_LHS(constraint)
            
            # Check if this term is linear
            
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
                # This edge is not ambiguous
                # -------------------------------------------------------------
                
                # Implement a standard four-edge McCormick relaxation
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
                        
                    # RHS corner constant
                    b_entry += A * coef_dht * coef_Sy
                    if sense == "<=": 
                        A_in.append(A_row) 
                        b_in.append(b_entry)
                    else: 
                        A_in.append(-A_row)
                        b_in.append(-b_entry)
                    descr_in.append(f"qt {(t,j,i)} McCormick 4-plane")
                    
                    
                    
                    
                    
    # =========================================================================
    # Enforce no-curl on all fundamental spatial cycles
    # =========================================================================

    if add_no_curl_constraints and fundamental_cycles is not None and len(fundamental_cycles) > 0:

        for cycle_idx, cycle in enumerate(fundamental_cycles):
            if cycle is None or len(cycle) == 0:
                continue

            for t in range(timesteps):

                A_row = scipy.sparse.lil_matrix((1, num_variables))
                b_entry = 0.0
                constraint_lhs = sympy.Integer(0)
                valid_cycle = True

                for u, v, sign in cycle:
                    sign = float(sign)
                    edge_id = ((t, u), (t, v))

                    # Some spatial edges may be absent in a given time layer,
                    # e.g. if the whole edge was removed because both cells are fixed-head.
                    if not grid.has_edge(*edge_id):
                        valid_cycle = False
                        break

                    qx = grid.edges[edge_id]["qx"]
                    constraint_lhs += int(sign) * qx

                    var = str(qx)
                    if collapsed[var]:
                        b_entry -= sign * float(bounds[var][0])
                    else:
                        A_row[0, non_collapsed_index[var]] += sign

                if not valid_cycle:
                    continue

                constraint = sympy.Eq(constraint_lhs, 0, evaluate = False)
                sympy_eq.append(copy.copy(constraint))

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

                for u, v, sign in cycle:
                    sign = float(sign)
                    edge_id = ((t, u), (t, v))

                    # Some spatial edges may be absent in a given time layer,
                    # e.g. if the whole edge was removed because both cells are fixed-head.
                    if not grid.has_edge(*edge_id):
                        valid_cycle = False
                        break

                    qx = grid.edges[edge_id]["dhx"]
                    constraint_lhs += int(sign) * qx

                    var = str(qx)
                    if collapsed[var]:
                        b_entry -= sign * float(bounds[var][0])
                    else:
                        A_row[0, non_collapsed_index[var]] += sign

                if not valid_cycle:
                    continue

                constraint = sympy.Eq(constraint_lhs, 0, evaluate = False)
                sympy_eq.append(copy.copy(constraint))

                if A_row.nnz == 0:
                    if abs(b_entry) <= 1e-10:
                        continue
                    raise ValueError(
                        "Infeasible no-curl constraint on cycle {} at timestep {}: all involved qx variables are collapsed, but the signed circulation is {:.6e}.".format(
                            cycle_idx, t, -b_entry)
                    )

                A_eq.append(copy.copy(A_row))
                b_eq.append(float(b_entry))
                descr_eq.append("no-curl fundamental cycle dhx {}".format((t, cycle_idx)))
                
                
                

    # =========================================================================
    # Define mass balances
    # =========================================================================
    
    for t in range(timesteps):
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
                
                tj, j = nbr
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
        if mat_list is None:
            return None
        if isinstance(mat_list, list):                 # exactly one row
            mat_list = scipy.sparse.vstack(mat_list, format="csr")
        elif not scipy.sparse.isspmatrix_csr(mat_list):
            mat_list = mat_list.tocsr()
        return mat_list
    
    
    if A_eq != []:
        A_eq = to_csr(A_eq)
    else:
        A_eq = None
    if A_in != []:
        A_in = to_csr(A_in)
    else:
        A_in = None
        
    # Get the bounds for the collapsed variables
    non_collapsed_bounds = [bounds[var] for var in non_collapsed_variables]
    
    if return_sympy:
        return A_eq, b_eq, descr_eq, sympy_eq, A_in, b_in, descr_in, sympy_in, non_collapsed_variables, non_collapsed_bounds
    else:
        return A_eq, b_eq, descr_eq, A_in, b_in, descr_in, non_collapsed_variables, non_collapsed_bounds