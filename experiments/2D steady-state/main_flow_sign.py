import numpy as np
import copy
import matplotlib.pyplot as plt
import sympy
import networkx as nx
import pickle
import os
from build_constraints_nocurl import build_constraints
from tighten_bounds_linprog import tighten_bounds
from grid_tools import build_square_grid, plot_labelled_grid
import time
from plot_bounds_from_grid import plot_bounds

np.random.seed(1)

#%%

import multiprocessing as mp

def find_oriented_fundamental_cycles(base_grid):
    """
    Return a fundamental cycle basis for `base_grid`.

    Each cycle is returned as a list of (u, v, sign), where (u, v) is the edge
    in the same orientation that `base_grid.edges` used when the spatial qx
    symbols are created later, and sign is +1/-1 depending on whether the cycle
    traverses that edge in the stored direction.
    """
    oriented_edge_by_key = {}
    for u, v in list(base_grid.edges):
        oriented_edge_by_key[frozenset((u, v))] = (u, v)

    fundamental_cycles = []
    for cycle_nodes in nx.cycle_basis(base_grid):
        if len(cycle_nodes) < 3:
            continue

        signed_cycle = []
        for idx, u in enumerate(cycle_nodes):
            v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
            key = frozenset((u, v))
            if key not in oriented_edge_by_key:
                raise KeyError("Cycle edge {} not found in base_grid.".format((u, v)))

            eu, ev = oriented_edge_by_key[key]
            sign = +1 if (u, v) == (eu, ev) else -1
            signed_cycle.append((eu, ev, sign))

        fundamental_cycles.append(signed_cycle)

    return fundamental_cycles


if __name__ == "__main__":
    mp.freeze_support()

    class Model:
        
        def __init__(self, grid, setup_dictionary, timesteps, fixed_head_cellset = [], 
                     T_correlation_edges = [], initial_bounds = None, T_correlation_frac = 0.2,
                     fundamental_cycles = None, add_no_curl_constraints = False):
            
            self.grid               = grid
            self.setup_dictionary   = setup_dictionary
            self.timesteps          = timesteps
            self.fixed_head_cellset = fixed_head_cellset
            self.T_correlation_edges= T_correlation_edges
            self.T_correlation_frac = T_correlation_frac
            self.fundamental_cycles = fundamental_cycles
            self.add_no_curl_constraints = add_no_curl_constraints
            
            self.start              = time.time()
    
            # =============================================================================
            # Continue by initiating the bounds
            # =============================================================================
            
            # Create space for the variables and bounds
            self.bounds = {}
            
            for identifier in list(self.grid.nodes):
                
                # Get time and node index
                t, node = identifier
                
                # -------------------------------------------------------------------------
                # Extract the variables
                # -------------------------------------------------------------------------
                
                # Hydraulic head
                key = str(self.grid.nodes[identifier]["h"])
                if initial_bounds is not None and key in initial_bounds:
                    self.bounds[key] = initial_bounds[key]
                else:
                    if identifier in list(self.setup_dictionary["h"].keys()):
                        self.bounds[key] = self.setup_dictionary["h"][identifier]
                    else:
                        self.bounds[key] = self.setup_dictionary["h"]["default"]
                            
                # Recharge
                key = str(self.grid.nodes[identifier]["R"])
                if initial_bounds is not None and key in initial_bounds:
                    self.bounds[key] = initial_bounds[key]
                else:
                    if identifier in list(self.setup_dictionary["R"].keys()):
                        self.bounds[key] = self.setup_dictionary["R"][identifier]
                    else:
                        self.bounds[key] = self.setup_dictionary["R"]["default"]    
                            
                # Specific yield
                if self.timesteps != 1: 
                    key = str(self.grid.nodes[identifier]["Sy"])
                    if initial_bounds is not None and key in initial_bounds:
                        self.bounds[key] = initial_bounds[key]
                    else:
                        if identifier in list(self.setup_dictionary["Sy"].keys()):
                            self.bounds[key] = self.setup_dictionary["Sy"][identifier]
                        else:
                            self.bounds[key] = self.setup_dictionary["Sy"]["default"]  
            
            for edge in list(self.grid.edges):
                
                # Get time and node index
                tj, j = self.grid.edges[edge]["start"]
                ti, i = self.grid.edges[edge]["end"]
    
                
                # Edge is at the same time
                if tj == ti:
                    
                    # Set the time index
                    t = tj
            
                    # -------------------------------------------------------------------------
                    # Extract the variables
                    # -------------------------------------------------------------------------
                
                    # Transmissivity
                    key = str(self.grid.edges[edge]["T"])
                    if initial_bounds is not None and key in initial_bounds:
                        self.bounds[key] = initial_bounds[key]
                    else:
                        if edge in list(self.setup_dictionary["T"].keys()):
                            self.bounds[key] = self.setup_dictionary["T"][edge]
                        else:
                            self.bounds[key] = self.setup_dictionary["T"]["default"]  
            
                    # Head difference
                    key = str(self.grid.edges[edge]["dhx"])
                    hj_key = str(self.grid.nodes[(t,j)]["h"])
                    hi_key = str(self.grid.nodes[(t,i)]["h"])
                    if initial_bounds is not None and key in initial_bounds:
                        self.bounds[key] = initial_bounds[key]
                    else:
                        self.bounds[key] = (
                            (self.bounds[hj_key][0] - self.bounds[hi_key][1])/self.grid.edges[edge]["dx"],
                            (self.bounds[hj_key][1] - self.bounds[hi_key][0])/self.grid.edges[edge]["dx"])
                    
                    # Fluxes
                    outer_product = np.outer(
                        self.bounds[str(self.grid.edges[edge]["dhx"])],  # (dL,dU)
                        self.bounds[str(self.grid.edges[edge]["T"])]     # (TL,TU)
                    ) * self.grid.edges[edge]["w"]
                    key = str(self.grid.edges[edge]["qx"])
                    if initial_bounds is not None and key in initial_bounds:
                        self.bounds[key] = initial_bounds[key]
                    else:
                        self.bounds[key] = (
                            np.min(outer_product),
                            np.max(outer_product))
                    
                # Edge s across time
                else:
                    
                    # -------------------------------------------------------------------------
                    # Extract the variables
                    # -------------------------------------------------------------------------
                
                    # Head difference (forward in time: h_t - h_{t-1})
                    key = str(self.grid.edges[edge]["dht"])
                    htj_key = str(self.grid.nodes[(tj,j)]["h"])
                    hti_key = str(self.grid.nodes[(ti,i)]["h"])
                    if initial_bounds is not None and key in initial_bounds:
                        self.bounds[key] = initial_bounds[key]
                    else:
                        self.bounds[key] = (
                            (self.bounds[hti_key][0] - self.bounds[htj_key][1]) / self.grid.edges[edge]["dt"],
                            (self.bounds[hti_key][1] - self.bounds[htj_key][0]) / self.grid.edges[edge]["dt"]
                        )
                    
                    # Fluxes
                    outer_product = np.outer(
                        self.bounds[str(self.grid.edges[edge]["dht"])],  # (dL,dU)
                        self.bounds[str(self.grid.nodes[(tj,j)]["Sy"])]     # (SyL,SyU)
                    ) * self.grid.nodes[(ti, i)]["area"]
                    key = str(self.grid.edges[edge]["qt"])
                    if initial_bounds is not None and key in initial_bounds:
                        self.bounds[key] = initial_bounds[key]
                    else:
                        self.bounds[key] = (
                            np.min(outer_product),
                            np.max(outer_product))
            
            symset = []
            for n, d in self.grid.nodes(data=True):
                for key in ("h","R","Sy"):
                    if key in d: symset.append(d[key])
            for j, i, d in self.grid.edges(data=True):
                for key in ("T","dhx","qx","dht","qt"):
                    if key in d: symset.append(d[key])
            
            # stable, de-duplicated names
            self.variables = list(dict.fromkeys(map(str, symset)))  # preserves order, removes dups
            
            # Sort the variables
            self.variables.sort()
            
            #%%
            
            # Prescribe flow directions
            for edge in list(self.grid.edges):
                
                # Extract the identifiers
                tj, j = self.grid.edges[edge]["start"]
                ti, i = self.grid.edges[edge]["end"]
    
                
                # This is an edge at the same timestep
                if tj == ti:
                
                    if self.grid.edges[edge]["flow sign"] == 1:
                        
                        self.bounds[str(self.grid.edges[edge]["dhx"])] = (
                            np.maximum(0, self.bounds[str(self.grid.edges[edge]["dhx"])][0]), 
                            self.bounds[str(self.grid.edges[edge]["dhx"])][1])
                        self.bounds[str(self.grid.edges[edge]["qx"])] = (
                            np.maximum(0, self.bounds[str(self.grid.edges[edge]["qx"])][0]),  
                            self.bounds[str(self.grid.edges[edge]["qx"])][1])
                        
                    elif self.grid.edges[edge]["flow sign"] == 0:
                        
                        self.bounds[str(self.grid.edges[edge]["dhx"])] = (0., 0.)
                        self.bounds[str(self.grid.edges[edge]["qx"])] = (0., 0.)
                        
                    elif self.grid.edges[edge]["flow sign"] == -1:
                        
                        self.bounds[str(self.grid.edges[edge]["dhx"])] = (
                            self.bounds[str(self.grid.edges[edge]["dhx"])][0], 
                            np.minimum(0, self.bounds[str(self.grid.edges[edge]["dhx"])][1]))
                        self.bounds[str(self.grid.edges[edge]["qx"])] = (
                            self.bounds[str(self.grid.edges[edge]["qx"])][0],
                            np.minimum(0, self.bounds[str(self.grid.edges[edge]["qx"])][1]))
                     
                # This is an edge across time
                else:
                    
                    if self.grid.edges[edge]["flow sign"] == 1:
                        
                        self.bounds[str(self.grid.edges[edge]["dht"])] = (
                            np.maximum(0,  self.bounds[str(self.grid.edges[edge]["dht"])][0]), 
                            self.bounds[str(self.grid.edges[edge]["dht"])][1])
                        self.bounds[str(self.grid.edges[edge]["qt"])] = (
                            np.maximum(0,  self.bounds[str(self.grid.edges[edge]["qt"])][0]), 
                            self.bounds[str(self.grid.edges[edge]["qt"])][1])
                        
                    elif self.grid.edges[edge]["flow sign"] == 0:
                        
                        self.bounds[str(self.grid.edges[edge]["dht"])] = (0., 0.)
                        self.bounds[str(self.grid.edges[edge]["qt"])] = (0., 0.)
                        
                    elif self.grid.edges[edge]["flow sign"] == -1:
                        
                        self.bounds[str(self.grid.edges[edge]["dht"])] = (
                            self.bounds[str(self.grid.edges[edge]["dht"])][0], 
                            np.minimum(0, self.bounds[str(self.grid.edges[edge]["dht"])][1]))
                        self.bounds[str(self.grid.edges[edge]["qt"])] = (
                            self.bounds[str(self.grid.edges[edge]["qt"])][0], 
                            np.minimum(0, self.bounds[str(self.grid.edges[edge]["qt"])][1]))
            
            bounds_initial = copy.deepcopy(self.bounds)
            
            #%%
            
            # =============================================================================
            # Check the feasibility of the bounds
            # =============================================================================
            
            feasible = True
            for key in list(self.bounds.keys()):
                bound = self.bounds[key]
                if bound[0] > bound[1]:
                    feasible = False
                    print(f"Bound {key} is invalid: {bound}")
            if not feasible:
                raise ValueError("At least some initial bounds are not feasible.")
            
            #%% 
            
        def tighten(self, 
                max_iterations_LP = 100,
                time_limit = 1.0, 
                presolve = True, 
                variables_to_tighten = None,
                variables_to_tighten_prefixes = ["h","R","T","dhx","qx","Sy","dht","qt"],
                tighten_fluxes_from_outer_product = True
                ):
            
            def contract_product(a, b, c):
    
                # Forward: c = a * b
                p = np.array([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
                c = (np.maximum(c[0], np.min(p)), np.minimum(c[1], np.max(p)))
                
                # Reverse: a = c / b
                if not (b[0] <= 0. and b[1] >= 0.):
                    r = np.array([1./b[0], 1./b[1]])
                    q = np.array([c[0]*np.min(r), c[0]*np.max(r), c[1]*np.min(r), c[1]*np.max(r)])
                    a = (np.maximum(a[0], np.min(q)), np.minimum(a[1], np.max(q)))
                
                # Reverse: b = c / a
                if not (a[0] <= 0. and a[1] >= 0.):
                    r = np.array([1./a[0], 1./a[1]])
                    q = np.array([c[0]*np.min(r), c[0]*np.max(r), c[1]*np.min(r), c[1]*np.max(r)])
                    b = (np.maximum(b[0], np.min(q)), np.minimum(b[1], np.max(q)))
                
                # One more forward pass
                p = np.array([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
                c = (np.maximum(c[0], np.min(p)), np.minimum(c[1], np.max(p)))
                
                return a, b, c
            
            # =============================================================================
            # Initial LP bound tightening
            # =============================================================================
            
            if variables_to_tighten is None:
                variables_to_tighten = [v for v in self.variables if v.split("_")[0] in variables_to_tighten_prefixes]
    
            print("==================================")
            print("LP bound tightening")
            print("==================================")
            
            # Create a folder for the checkpoints, if it does not exist yet
            checkpoint_dir = "checkpoints_flow_sign"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Store the states
            state = {
                "iteration": None,
                "bounds": self.bounds,
                "variables": self.variables,
                "timesteps": self.timesteps,
                "start time" : self.start,
                "grid" : self.grid,
                "current time" : time.time(),
                "fixed_head_cells" : self.fixed_head_cellset,
                "T_correlation_edges" : self.T_correlation_edges,
                "T_correlation_frac" : self.T_correlation_frac,
                "fundamental_cycles" : self.fundamental_cycles,
                "add_no_curl_constraints" : self.add_no_curl_constraints
            }
            
            pickle.dump(
                state,
                open(os.path.join(
                    checkpoint_dir, 
                    "iteration_initial.p"),"wb")
                )
            
            
            
            # Initialize for warm-up: no binaries, no lambdas
    
            overall_volume_fraction = []
            
            for iteration in range(max_iterations_LP):
                
                print("=== Iteration "+str(iteration).zfill(3)+" ===")
                
                # Check if we have already calculated this iteration before
                if "iteration_"+str(iteration).zfill(3)+".p" in os.listdir(checkpoint_dir):
                    
                    print("Loading past iteration")
                    
                    state = pickle.load(open(os.path.join(
                        checkpoint_dir, 
                        "iteration_"+str(iteration).zfill(3)+".p"),"rb"))
                    
                    self.bounds = copy.deepcopy(state["bounds"])
                    self.variables = copy.deepcopy(state["variables"])
                    self.timesteps = copy.deepcopy(state["timesteps"])
                    self.grid = copy.deepcopy(state["grid"])
                    self.fixed_head_cellset = copy.deepcopy(state["fixed_head_cells"])
                    self.T_correlation_edges = copy.deepcopy(state["T_correlation_edges"])
                    self.T_correlation_frac = copy.deepcopy(state["T_correlation_frac"])
                    self.fundamental_cycles = copy.deepcopy(state.get("fundamental_cycles", None))
                    self.add_no_curl_constraints = copy.deepcopy(state.get("add_no_curl_constraints", False))
                    
                    continue
                
                
                
                # Remove duplicates
                self.variables = list(dict.fromkeys(self.variables))
                
                self.A_eq, self.b_eq, self.descr_eq, self.sympy_eq, self.A_in, \
                self.b_in, self.descr_in, self.sympy_in, self.non_collapsed_variables, \
                self.non_collapsed_bounds = build_constraints(
                    variables = self.variables,
                    bounds = self.bounds,
                    timesteps = self.timesteps,
                    grid = self.grid,
                    fixed_head_cells = self.fixed_head_cellset,
                    T_correlation_edges = self.T_correlation_edges,
                    T_correlation_frac = self.T_correlation_frac,
                    fundamental_cycles = self.fundamental_cycles,
                    add_no_curl_constraints = self.add_no_curl_constraints
                )
            
                # All-continuous solve (LP). Keep your MILP options if you like.
                tightened_bounds = tighten_bounds(
                    self.A_eq, self.b_eq, self.A_in, self.b_in,
                    self.variables,
                    self.non_collapsed_variables,
                    self.bounds, self.non_collapsed_bounds,
                    verbose = "all",
                    variables_to_tighten = variables_to_tighten, 
                    time_limit = time_limit,
                    presolve = presolve,
                    n_workers=8, 
                )
                
                # -------------------------------------------------------------
                # Compute reduction
                # -------------------------------------------------------------
                
                product_reduction = []
                max_reduction = 0
                max_reduction_var = "None"
                
                for idx,var2 in enumerate(self.non_collapsed_variables):
                    product_reduction.append(np.diff(tightened_bounds[var2])/np.diff(self.bounds[var2]))
                    reduction = tightened_bounds[var2][0] - self.bounds[var2][0]
                    reduction += self.bounds[var2][1] - tightened_bounds[var2][1]
                    if reduction > max_reduction:
                        max_reduction = reduction
                        max_reduction_var = var2
                        
                    self.bounds[var2] = tightened_bounds[var2]
                    
                # Check bounds for feasibility
                bounds_feasible = [True if self.bounds[var2][0] <= self.bounds[var2][1] else False for var2 in self.variables]
                if not all(bounds_feasible):
                    for idx,var2 in enumerate(self.variables):
                        if not bounds_feasible[idx]:
                            print(f"Bound infeasible for variable {var2}: {self.bounds[var2]}")
                    raise Exception
                    
                # Check bounds for feasibility
                bound_width = [self.bounds[var2][1] - self.bounds[var2][0] for var2 in self.variables]
                bound_width = [np.nan if x == 0. else x for x in bound_width]
                quasi_collapsed = [True if (self.bounds[var2][1] - self.bounds[var2][0] <= 1E-10 and self.bounds[var2][1] - self.bounds[var2][0] != 0) else False for var2 in self.variables]
                    
                percent_reduction = (1 - np.prod(product_reduction))*100
                
                overall_volume_fraction.append(np.prod(product_reduction))
                    
                print("Iteration {} - Hyperbox volume reduced by {}%.".format(iteration,percent_reduction))
                    
                if percent_reduction <= 0.1:
                    idx = np.where(bound_width == np.nanmin(bound_width))[0][0]
                    print("Terminating tightening. Last volume reduction: {:.2f} | Total volume reduction: {:.2f}".format(
                        percent_reduction,
                        (1-np.prod(overall_volume_fraction))*100) )
                    plt.figure()
                    plt.plot(
                        overall_volume_fraction)
                    break
                elif iteration == max_iterations_LP-1:
                    idx = np.where(bound_width == np.nanmin(bound_width))[0][0]
                    print("Iteration maximum reached.")
                    print("Terminating tightening. Last volume reduction: {:.2f} | Total volume reduction: {:.2f}".format(
                        percent_reduction,
                        (1-np.prod(overall_volume_fraction))*100) )
            
                # -------------------------------------------------------------
                # Contract bilinear products with interval arithmetic
                # -------------------------------------------------------------
                
                # Go through all edges
                for edge in self.grid.edges:
                    
                    (tj,j),(ti,i) = edge
                    
                    # Edge in space
                    if tj == ti:
                        
                        # Extract the relevant bounds
                        dhx = self.bounds[str(self.grid.edges[edge]["dhx"])]
                        qx = self.bounds[str(self.grid.edges[edge]["qx"])]
                        T = self.bounds[str(self.grid.edges[edge]["T"])]
                        w = self.grid.edges[edge]["w"]
                        
                        dhx, Tw, qx = contract_product(dhx, (w*T[0], w*T[1]), qx)
                        self.bounds[str(self.grid.edges[edge]["dhx"])] = dhx
                        self.bounds[str(self.grid.edges[edge]["T"])]   = (Tw[0]/w, Tw[1]/w)
                        self.bounds[str(self.grid.edges[edge]["qx"])]  = qx
                        
                    # Edge in time
                    elif tj != ti:
                        
                        # Extract the relevant bounds
                        dht = self.bounds[str(grid.edges[edge]["dht"])]
                        qt = self.bounds[str(grid.edges[edge]["qt"])]
                        Sy = self.bounds[str(grid.nodes[(ti,i)]["Sy"])]
                        A = self.grid.nodes[edge[0]]["area"]
                        
                        dht, Asy, qt = contract_product(dht, (A*Sy[0], A*Sy[1]), qt)
                        self.bounds[str(self.grid.edges[edge]["dht"])] = dht
                        self.bounds[str(self.grid.nodes[(ti,i)]["Sy"])]  = (Asy[0]/A, Asy[1]/A)
                        self.bounds[str(self.grid.edges[edge]["qt"])]  = qt
            
                # =============================================================
                # Store checkpoint at end of iteration
                # =============================================================
                
                # Create a folder for the checkpoints, if it does not exist yet
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Store the states
                state = {
                    "iteration": iteration,
                    "bounds": self.bounds,
                    "variables": self.variables,
                    "timesteps": self.timesteps,
                    "start time" : self.start,
                    "grid" : self.grid,
                    "current time": time.time(),
                    "fixed_head_cells" : self.fixed_head_cellset,
                    "T_correlation_edges" : self.T_correlation_edges,
                    "T_correlation_frac" : self.T_correlation_frac,
                    "fundamental_cycles" : self.fundamental_cycles,
                    "add_no_curl_constraints" : self.add_no_curl_constraints
                }
                
                pickle.dump(
                    state,
                    open(os.path.join(
                        checkpoint_dir, 
                        "iteration_"+str(iteration).zfill(3)+".p"),"wb")
                    )
                
                plot_bounds(
                    self.bounds, 
                    self.grid, 
                    self.timesteps,
                    save_figures = True,
                    figure_name = os.path.join(
                        checkpoint_dir, 
                        "iteration="+str(iteration).zfill(3)+".png"))
                
                plt.show()
    
    
    
    #%%
    
    
    # =============================================================================
    # Start by building the grid
    # =============================================================================
    
    # Define spacing in time and space
    dx = 10 # in m
    dt = 1 # in s
    
    # Define dimensions in time
    timesteps = 1
    
    # Define spatial grid
    polygon = [(0, 0), (dx*5, 0), (dx*5, dx*5), (0, dx*5)]
    seed = (dx/2, 50-dx/2)
    
    # Create the base grid
    base_grid = build_square_grid(polygon, seed, dx)
    
    # Flip the coordinates
    for node in list(base_grid.nodes):
    
        safe = base_grid.nodes[node]["xpos"]
        base_grid.nodes[node]["xpos"] = base_grid.nodes[node]["ypos"]
        base_grid.nodes[node]["ypos"] = 40 - safe
        base_grid.nodes[node]["vertices"] = [(y, 40 - x) for x,y in base_grid.nodes[node]["vertices"]]
        
    
    # Prescribe flow directions
    for edge in list(base_grid.edges):
        base_grid.edges[edge]["flow sign"] = 1

    # Detect a fundamental cycle basis in the base grid, signed using the
    # edge orientation that will later define each qx_(t,j,i) variable.
    fundamental_cycles = find_oriented_fundamental_cycles(base_grid)
    print("Detected {} fundamental cycles in base_grid.".format(len(fundamental_cycles)))
    
    
    # Draw the labelled grid
    plot_labelled_grid(base_grid)

    # Define the fixed head cells
    cellset = []
    fixed_head_cells = []

    #%%
    
    # =============================================================================
    # Set up the spatio-temporal grid and the Sympy variables
    # =============================================================================
        
    # Define spatio-temporal grid
        
    # Create a directed graph that represents the numerical grid
    grid = nx.Graph()
    nodes = list(base_grid.nodes)
    for t in range(timesteps):
        
        # Add nodes
        for node in nodes:
            
            # Create and initiate the nodes
            grid.add_node((t,node))
            
            # Assign geometry attributes
            grid.nodes[(t,node)]["xpos"] = base_grid.nodes[node]["xpos"]
            grid.nodes[(t,node)]["ypos"] = base_grid.nodes[node]["ypos"]
            grid.nodes[(t,node)]["area"] = base_grid.nodes[node]["area"]
            grid.nodes[(t,node)]["vertices"] = base_grid.nodes[node]["vertices"]
            
            # Assign the flow-relevant variables
            grid.nodes[(t,node)]["h"] = sympy.Symbol(f'h_{(t,node)}')
            grid.nodes[(t,node)]["R"] = sympy.Symbol(f'R_{(t,node)}')
            if timesteps != 1: # We have a transient system
                grid.nodes[(t,node)]["Sy"] = sympy.Symbol(f'Sy_{node}')
    
        # Add spatial edges
        for edge in list(base_grid.edges):
            
            # Extract the start and end nodes
            j,i = edge
            
            # Only add the edge if it is not between two fixed-head cells
            if not ((t,j) in fixed_head_cells and (t,i) in fixed_head_cells):
            
                # Add the edge
                grid.add_edge((t,j), (t,i))
                grid.edges[(t,j), (t,i)]["start"] = (t,j)
                grid.edges[(t,j), (t,i)]["end"] = (t,i)
                
                # Add geometry attributes
                grid.edges[(t,j), (t,i)]["w"] = dx
                grid.edges[(t,j), (t,i)]["dx"] = dx
                
                # Assign the flow-relevant variables
                # grid.edges[(t,j), (t,i)]["T"] = sympy.Symbol(f'T_{edge}')
                grid.edges[(t,j), (t,i)]["T"] = sympy.Symbol('T')
                grid.edges[(t,j), (t,i)]["qx"] = sympy.Symbol(f'qx_{(t,j,i)}')
                grid.edges[(t,j), (t,i)]["dhx"] = sympy.Symbol(f'dhx_{(t,j,i)}')
                
                grid.edges[(t,j), (t,i)]["flow sign"] = base_grid.edges[j,i]["flow sign"]
    
            
        # Add temporal edges
        if t > 0: # We have a transient system
        
            # Go through all grid cells
            for i in nodes:
                
                # Only add the edge if it is not between two fixed-head cells
                if not ((t-1,i) in fixed_head_cells and (t,i) in fixed_head_cells):
                
                    # Add an edge to the previous timestep
                    grid.add_edge((t-1,i), (t,i))
                    grid.edges[(t-1,i), (t,i)]["start"] = (t-1,i)
                    grid.edges[(t-1,i), (t,i)]["end"] = (t,i)
                    
                    # Add geometry attributes
                    grid.edges[(t-1,i), (t,i)]["dt"] = dt
                    
                    # Add the flux and temporal head gradient
                    grid.edges[(t-1,i), (t,i)]["qt"] = sympy.Symbol(f'qt_{(t-1,t,i)}')
                    grid.edges[(t-1,i), (t,i)]["dht"] = sympy.Symbol(f'dht_{(t-1,t,i)}')
                    
                    grid.edges[(t-1,i), (t,i)]["flow sign"] = None
                    
        
    #%%
    
    # =============================================================================
    # Continue by initiating the bounds
    # =============================================================================
        
    bounds = {}
    
    # Setup dictionary for bounds
    setup_dictionary = {
        "h"     : {
            "default":      (3.0, 12.0)},
        "T"     : {
            "default":      (1E-3, 1E-1)},
        "R"     : {
            "default":      (0., 0.)},
        "Sy"     : {
            "default":      (0.15, 0.25)}
        }
    
    # Set observations
    setup_dictionary["h"][(0,12)] = (8, 8)

    # Define source and sink cell
    setup_dictionary["R"][(0,0)] = (1E-5, 1E-4)
    setup_dictionary["R"][(0,24)] = (-1E-3,-1E-5)
        
    # No no-flow edges in this model
    no_flow_edges = []    
    
    #%%
    
    # =============================================================================
    # Initiate the bulk model
    # =============================================================================
    
    m = Model(
        grid,
        setup_dictionary,
        timesteps,
        fixed_head_cellset = cellset,
        fundamental_cycles = fundamental_cycles,
        add_no_curl_constraints = False
    )
    
    m.tighten(
        max_iterations_LP=1000,
        time_limit=60.0,
        presolve=True,
        variables_to_tighten_prefixes = ["h","R","T","qx","dhx","Sy","dht","qt"]
    )
    