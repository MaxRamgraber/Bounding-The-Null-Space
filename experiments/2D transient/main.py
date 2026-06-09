import numpy as np
import copy
import matplotlib.pyplot as plt
import sympy
from plot_bounds_from_grid import plot_bounds, plot_grid_dirs
import networkx as nx
import pickle
import os
from build_constraints_cleaned import build_constraints
from tighten_bounds_linprog_v03 import tighten_bounds
from grid_tools import build_hex_grid, plot_labelled_grid, plot_node_values
from solve_FVM_v03 import solve_FVM
import time

print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))

try:
    aff = os.sched_getaffinity(0)
    print("Visible CPUs =", len(aff), sorted(aff))
except AttributeError:
    pass


np.random.seed(1)

#%%

import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()   # safe to keep; needed for some Windows setups

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
            checkpoint_dir = "checkpoints"
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
                    n_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), 
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
    dt = 3600*24*1 # in s
    
    # Define dimensions in time
    timesteps = 5
    
    # Define spatial grid
    polygon = [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)]
    seed = (0, 0)
    hex_side_length = dx/np.sqrt(3)
    
    # Create the base grid
    base_grid = build_hex_grid(polygon, seed, hex_side_length)
    
    # Draw the labelled grid
    plot_labelled_grid(base_grid)
    
    # Define the fixed head cells
    # cellset = [9]
    cellset = [0,1,2,3,4,5,6,7,8,9]
    fixed_head_cells = []
    for t in range(timesteps):
        fixed_head_cells += [(t,node) for node in cellset]
        
    # Delete some cells
    deletion_cells = [22,23,32,33,34,42,43,53,54,63]
    for node in deletion_cells:
        base_grid.remove_node(node)
        
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
                grid.edges[(t,j), (t,i)]["w"] = hex_side_length
                grid.edges[(t,j), (t,i)]["dx"] = dx
                
                # Assign the flow-relevant variables
                grid.edges[(t,j), (t,i)]["T"] = sympy.Symbol(f'T_{edge}')
                grid.edges[(t,j), (t,i)]["qx"] = sympy.Symbol(f'qx_{(t,j,i)}')
                grid.edges[(t,j), (t,i)]["dhx"] = sympy.Symbol(f'dhx_{(t,j,i)}')
                
                grid.edges[(t,j), (t,i)]["flow sign"] = None
    
            
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
            "default":      (0.0, 12.0)},
        "T"     : {
            "default":      (1E-4, 1E-2)},
        "R"     : {
            "default":      (0., 0.)},
        "Sy"     : {
            "default":      (0.15, 0.25)}
        }
    
    
    # Create more fixed head cells
    for node in fixed_head_cells:
        
        # Set all the fixed head cells to the same boundary
        setup_dictionary["h"][node] = (7.0, 7.0)
        
    # Add the extraction well
    wellnode = 77
    for t in np.arange(1,timesteps):
        
        # Set the pumping rate negative past the first timestep
        # setup_dictionary["R"][(t,wellnode)] = (-1E-7,-1E-7)
        setup_dictionary["R"][(t,wellnode)] =  (-1E-3,-1E-5)
        
    # Add recharge area
    recharge_cells = [100,101,90,91,92,80,81]
    for t in np.arange(0,np.minimum(timesteps,3)): # Turn off source after three days
        for cell in recharge_cells:
            # Set the recharge rate positive past the first timestep
            setup_dictionary["R"][(t,cell)] = (1E-6, 1E-5)
            
    # Define observation wells
    observation_wells = [31,77,91]
        
    # Set shared variables
    for t in range(timesteps):
        
        # The extraction well shares a pumping rate across time
        if t != 0: # Only add this past the first timestep
            grid.nodes[(t,wellnode)]["R"] = sympy.Symbol('R_well')
        
    # The fixed head cells share a fixed head across space and time
    for node in fixed_head_cells:
        grid.nodes[node]["h"] = sympy.Symbol('h_boundary')
        
    no_flow_edges = []    
    
    #%%
    
    # =============================================================================
    # Initiate the bulk model
    # =============================================================================
    
    m = Model(grid, setup_dictionary, timesteps, fixed_head_cellset = cellset)
    
    #%%
    
    # =============================================================================
    # Solve Finite Volume Model
    # =============================================================================
    
    hs = {}
    qs = {}
    flow_signs = {}
    
    for t in range(timesteps):
        
        print(f"Solve FVM model for timestep {t}")
        
        # Create the subgraph
        subgraph_nodes = []
        for node in base_grid.nodes:
            subgraph_nodes.append((t,node))
        subgraph = grid.subgraph(subgraph_nodes).copy()
        
        # Set the mid values
        T_mid = {}
        Sy_mid = {}
        R_mid = {}
        fixed_heads_mid = {}
        for node in subgraph.nodes:
            t,i = node
            # R_mid[node] = np.mean(m.bounds[str(subgraph.nodes[node]["R"])])
            l,u = m.bounds[str(subgraph.nodes[node]["R"])]
            if l != 0 and u != 0:
                R_mid[node] = np.sign(l)*10**np.nanmean([np.log10(np.abs(l)),np.log10(np.abs(u))])
            else:
                R_mid[node] = 0
            if t != 0:
                Sy_mid[node] = np.mean(m.bounds[str(subgraph.nodes[node]["Sy"])])
            if node in fixed_head_cells:
                fixed_heads_mid[node] = np.mean(m.bounds[str(subgraph.nodes[node]["h"])])
        for edge in subgraph.edges:
            (tj,j),(ti,i) = edge
            if tj == t and ti == t:
                l,u = m.bounds[str(subgraph.edges[edge]["T"])]
                T_mid[edge] = 10**np.mean([np.sign(l)*np.log10(np.abs(l)),np.sign(u)*np.log10(np.abs(u))])
                # T_mid[edge] = np.mean(m.bounds[str(subgraph.edges[edge]["T"])])
        
        # Save the previous heads, if we are not in the first timestep
        if t > 0:
            prev_heads = {(t, i): hs[t - 1][(t - 1, i)] for s,i in subgraph.nodes}
        else:
            prev_heads = None
        
        # Solve the FVM
        hs[t], qs[t] = solve_FVM(
            subgraph,
            T=T_mid,
            Sy=Sy_mid,
            R=R_mid,
            fixed_heads=fixed_heads_mid,
            prev_heads=prev_heads,
            dt=dt,
            return_flows=True,
        )
        
        # Plot the result
        plot_node_values(subgraph, hs[t], cmap="turbo", title=f"Heads (t={t})", vmin = 0, vmax = 10)
        
        # Assign flow directions
        for edge in subgraph.edges:
            
            # Extract the nodes
            (tj, j), (ti, i) = edge
            
            # Get the canonical start and end
            start = subgraph.edges[edge]["start"]
            end = subgraph.edges[edge]["end"]
            canonical_edge = (start,end)
            
            # This edge is in space
            if tj == t and ti == t:
                
                # Get the head difference
                dh = hs[t][start] - hs[t][end]
                
                # Assign the sign
                flow_signs[canonical_edge] = int(np.sign(dh))
                
                
        if t > 0:
            for node in subgraph.nodes:
                s,i = node
                if s == t and node not in fixed_head_cells:
                    
                    # Get the edge across time
                    canonical_edge = ((t-1,i),(t,i))
                    
                    if canonical_edge in grid.edges:
                    
                        flow_signs[canonical_edge] = np.sign(hs[t][(t,i)] - hs[t-1][(t-1,i)])
        
    #%%
    
    
    # =============================================================================
    # Initiate the bulk model a second time with flow signs
    # =============================================================================
    
    # Assign the flow directions to the grid graph
    for edge in list(flow_signs.keys()):
        grid.edges[edge]["flow sign"] = flow_signs[edge]
        
    # Plot the grid directions
    plot_grid_dirs(grid,timesteps)
    plt.show()
    
    # Assign groundwater observations
    observations = []
    for t in range(timesteps):
        for i in observation_wells:
            observations.append((t,i))
            
            
            
    # observations = [(t,wellnode) for t in range(timesteps)]
    observation_dictionary = {
        "obs": [],
        "hs" : hs,
        "qs" : qs,
        "T_mid" : T_mid,
        "Sy_mid": Sy_mid,
        "R_mid": R_mid}
    
    # Assign observation in initial bounds
    for obs in observations:
        t,node = obs
        setup_dictionary["h"][obs] = (hs[t][obs],hs[t][obs])
        observation_dictionary["obs"].append(hs[t][obs])
    
    pickle.dump(
        observation_dictionary,
        open("observation_dictionary.p","wb"))
    
    # Re-create the model
    m = Model(grid, setup_dictionary, timesteps, fixed_head_cellset = cellset,
              T_correlation_edges = list(subgraph.edges),T_correlation_frac = 0.05)
    
    m.tighten(
        max_iterations_LP=25,
        time_limit=60.0,
        presolve=True,
        variables_to_tighten_prefixes = ["h","R","T","qx","dhx","Sy","qt","dht"]
    )
    