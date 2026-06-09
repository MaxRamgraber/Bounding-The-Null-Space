import pickle
import numpy as np
import sympy
import networkx as nx
from grid_tools import build_square_grid
from solve_FVM_v03 import solve_FVM
from mcmc_sampler_pymc import mcmc_sampler_pymc

# =============================================================================
# Sample a bounded partition
# =============================================================================

def sample_within_bounds(total, lower, upper):
    
    """
    This code draws a number of samples within specified lower and upper bounds
    such that the sum of these samples adds up to a total.
    
    This code is used to sample a head drop over a number of cells, which is 
    subsequently used to derive the other parameters.    
    """
    
    # lower = np.asarray(lower, dtype=float)
    # upper = np.asarray(upper, dtype=float)

    # Make sure that it is actually possible to satisfy both constraints, within machine tolerance
    if total < np.sum(lower) - 1e-12:
        return None
    if total > np.sum(upper) + 1e-12:
        return None

    # Sample sequentially while preserving feasibility of the remainder
    out = np.zeros(len(lower))
    remainder = float(total)

    # Go through all edges except the last
    for i in range(len(lower) - 1):
        
        # Adjust the bounds to honor the remainder
        lo = max(lower[i], remainder - np.sum(upper[i + 1:]))
        hi = min(upper[i], remainder - np.sum(lower[i + 1:]))

        # If infeasible, reject
        if lo > hi + 1e-12:
            return None

        # Draw a random sample
        out[i] = np.random.uniform(lo, hi)
        
        # Adjust the remainder accordingly
        remainder -= out[i]

    # Set the last drop to the remainder
    out[-1] = remainder

    # Final consistency check
    if out[-1] < lower[-1] - 1e-10:
        return None
    if out[-1] > upper[-1] + 1e-10:
        return None

    # Return the result
    return out

# =============================================================================
# Draw one exact-fit sample
# =============================================================================

def sample_perfect_fit(grid, bounds, gauge_nodes, gauge_values, source_node, sink_node):
    
    """
    This function draws one exact fit sample from the 1D model, replicating the
    setup in our OBBT code.
    """

    # -------------------------------------------------------------------------
    # First, order nodes and edges from left to right
    # -------------------------------------------------------------------------
    
    # Extract all nodes from the grid, then order them from left to right
    ordered_nodes = [u for u in grid.nodes]
    ordered_nodes = sorted(
        ordered_nodes,
        key=lambda u: (grid.nodes[u]["xpos"], grid.nodes[u]["ypos"], u[1])
    )

    # Create all ordered edges from the ordered list of nodes. Because my graph
    # representation of the grid is unordered, make sure that the edges are 
    # listed in the correct orientation.
    ordered_edges = []
    for a, b in zip(ordered_nodes[:-1], ordered_nodes[1:]):
        if (a, b) in grid.edges:
            ordered_edges.append((a, b))
        elif (b, a) in grid.edges:
            ordered_edges.append((b, a))
        else:
            raise Exception(f"No spatial edge found between consecutive nodes {a} and {b}.")

    # Create a dictionary of where each node is in the ordered list
    node_pos = {node: idx for idx, node in enumerate(ordered_nodes)}

    # -------------------------------------------------------------------------
    # Read bounds
    # -------------------------------------------------------------------------
    
    # Read out the state and parameter bounds of each node and edge
    h_bounds = {node: bounds[str(grid.nodes[node]["h"])] for node in ordered_nodes}
    T_bounds = {edge: bounds[str(grid.edges[edge]["T"])] for edge in ordered_edges}
    R_bounds = {node: bounds[str(grid.nodes[node]["R"])] for node in ordered_nodes}

    # -------------------------------------------------------------------------
    # Read source/sink bounds and observation positions
    # -------------------------------------------------------------------------
    
    # Read out the source and sink of the area; both are equal
    area_source = float(grid.nodes[source_node]["area"])
    area_sink = float(grid.nodes[sink_node]["area"])

    # Total flow q must satisfy both the source and sink bounds
    q_lo = max(R_bounds[source_node][0] * area_source, -R_bounds[sink_node][1] * area_sink)
    q_hi = min(R_bounds[source_node][1] * area_source, -R_bounds[sink_node][0] * area_sink)

    # To find a perfect solution, we use the first observations as a gauge 
    # value for the FVM solver. Extract the two observation wells and their
    # head values.
    obs_left, obs_right = gauge_nodes
    h_left, h_right = gauge_values

    # Where in the list of ordered nodes are the two observation nodes?
    i_left = node_pos[obs_left]
    i_right = node_pos[obs_right]

    # Split the edges up into a left, central, and right part, divided by the
    # observation nodes.
    left_edges = ordered_edges[:i_left]
    mid_edges = ordered_edges[i_left:i_right]
    right_edges = ordered_edges[i_right:]

    # Also extract the left and right nodes
    left_nodes = ordered_nodes[:i_left]
    right_nodes = ordered_nodes[i_right + 1:]

    # Compute the prescribed head drop in the middle, the only part that has a 
    # definitive, non-random head drop
    middle_total_drop = h_left - h_right

    # -------------------------------------------------------------------------
    # Compute admissible total drops outside the observation interval
    # -------------------------------------------------------------------------
    
    # To the left, the maximum head drop is defined by the upper bound and the
    # first observation value
    left_total_cap = 0.0
    if len(left_nodes) > 0:
        left_total_cap = min([h_bounds[node][1] - h_left for node in left_nodes])

    # To the right, the maximum head drop is defined by the lower bound and the
    # second observation value
    right_total_cap = 0.0
    if len(right_nodes) > 0:
        right_total_cap = min([h_right - h_bounds[node][0] for node in right_nodes])

    # Create two little helper functions compute the maximum and minimum head
    # drops over an edge set
    def sum_min_drops(edge_list):
        """
        The minimum possible head drop depends on the upper transmissivity bounds.
        """
        total = 0.0
        for edge in edge_list:
            dx = float(grid.edges[edge]["dx"])
            w = float(grid.edges[edge]["w"])
            T_hi = T_bounds[edge][1]
            total += dx / (T_hi * w)
        return total

    def sum_max_drops(edge_list):
        """
        The maximum possible head drop depends on the lower transmissivity bounds.
        """
        total = 0.0
        for edge in edge_list:
            dx = float(grid.edges[edge]["dx"])
            w = float(grid.edges[edge]["w"])
            T_lo = T_bounds[edge][0]
            total += dx / (T_lo * w)
        return total

    # Tighten the flow bounds for the middle section based on the permissible 
    # head drop and the physical cosntraints
    q_lo = max(q_lo, middle_total_drop / sum_max_drops(mid_edges))
    q_hi = min(q_hi, middle_total_drop / sum_min_drops(mid_edges))

    # Tighten the flow bounds for the left and right side, too. Lower bounds
    # are never an issue.
    if len(left_edges) > 0:
        q_hi = min(q_hi, left_total_cap / sum_min_drops(left_edges))
    if len(right_edges) > 0:
        q_hi = min(q_hi, right_total_cap / sum_min_drops(right_edges))

    # Check for feasibility
    if q_lo > q_hi:
        return None

    # -------------------------------------------------------------------------
    # Sample a total flow
    # -------------------------------------------------------------------------
    
    # Now sample a flow rate from left to right within these bounds
    q = np.random.uniform(q_lo, q_hi)

    # -------------------------------------------------------------------------
    # Sample edgewise drops between the observation nodes
    # -------------------------------------------------------------------------
    
    # Based with this knowledge, compute bounds for the head drops between the two observations
    mid_lower = []
    mid_upper = []
    
    # Go through all center edges
    for edge in mid_edges:
        
        # Extract geometry of this edge
        dx = float(grid.edges[edge]["dx"])
        w = float(grid.edges[edge]["w"])
        
        # Computer lower and upper head drop bounds based on Darcy's Law and the prescribed flow 
        mid_lower.append(q * dx / (T_bounds[edge][1] * w))
        mid_upper.append(q * dx / (T_bounds[edge][0] * w))

    # Sample the head drops in the center section
    mid_drops = sample_within_bounds(middle_total_drop, mid_lower, mid_upper)
    if mid_drops is None:
        return None

    # -------------------------------------------------------------------------
    # Sample edgewise drops to the left of the first observation
    # -------------------------------------------------------------------------
    
    # Now repeat the same for the left and right
    if len(left_edges) > 0:
        
        left_lower = []
        left_upper = []
        
        # Go through all left edges
        for edge in left_edges:
            
            # Extract geometry of this edge
            dx = float(grid.edges[edge]["dx"])
            w = float(grid.edges[edge]["w"])
            
            # Computer lower and upper head drop bounds based on Darcy's Law and the prescribed flow 
            left_lower.append(q * dx / (T_bounds[edge][1] * w))
            left_upper.append(q * dx / (T_bounds[edge][0] * w))

        # Compute the lower and upper bounds for the total head drop
        left_total_lo = np.sum(left_lower) # Lower bound isn't an issue
        left_total_hi = min(np.sum(left_upper), left_total_cap) # Upper bound may be capped

        # Check for feasibility
        if left_total_lo > left_total_hi:
            return None

        # Sample the total head drop
        left_total = np.random.uniform(left_total_lo, left_total_hi)
        
        # Sample the individual head drops
        left_drops = sample_within_bounds(left_total, left_lower, left_upper)
        if left_drops is None:
            return None
    else:
        left_drops = np.zeros(0)

    # -------------------------------------------------------------------------
    # Sample edgewise drops to the right of the second observation
    # -------------------------------------------------------------------------
    
    if len(right_edges) > 0:
        
        right_lower = []
        right_upper = []
        
        # Go through all right edges
        for edge in right_edges:
            
            # Extract geometry of this edge
            dx = float(grid.edges[edge]["dx"])
            w = float(grid.edges[edge]["w"])
            
            # Computer lower and upper head drop bounds based on Darcy's Law and the prescribed flow 
            right_lower.append(q * dx / (T_bounds[edge][1] * w))
            right_upper.append(q * dx / (T_bounds[edge][0] * w))

        # Compute the lower and upper bounds for the total head drop
        right_total_lo = np.sum(right_lower) # Lower bound isn't an issue
        right_total_hi = min(np.sum(right_upper), right_total_cap) # Upper bound may be capped

        # Check for feasibility
        if right_total_lo > right_total_hi:
            return None

        # Sample the total head drop
        right_total = np.random.uniform(right_total_lo, right_total_hi)
        
        # Sample the individual head drops
        right_drops = sample_within_bounds(right_total, right_lower, right_upper)
        if right_drops is None:
            return None
    else:
        right_drops = np.zeros(0)

    # -------------------------------------------------------------------------
    # Convert sampled drops into transmissivities
    # -------------------------------------------------------------------------
    
    # Combine all the head drops we have sampled
    all_drops = np.concatenate([left_drops, mid_drops, right_drops])

    # Now based on the flow and the head drops, calculate the transmissivities
    T = {}
    
    # Go through each edge
    for idx, edge in enumerate(ordered_edges):
        
        # Extract geometry of this edge
        dx = float(grid.edges[edge]["dx"])
        w = float(grid.edges[edge]["w"])
        
        # Use Darcy's Law, the flow rate, and the head drop for this edge to compute the transmissivity
        T[edge] = q * dx / (w * all_drops[idx])

    # -------------------------------------------------------------------------
    # Build recharge and storage dictionaries
    # -------------------------------------------------------------------------
    
    # Compute the recharge based on the flow rate
    R = {node: 0.0 for node in ordered_nodes}
    R[source_node] = q / area_source
    R[sink_node] = -q / area_sink

    # This is just a dummy value, our computation is steady-state
    Sy = {node: 0.0 for node in ordered_nodes}

    # -------------------------------------------------------------------------
    # Solve the forward model, using the left observation as the head datum
    # -------------------------------------------------------------------------
    
    # Run the FVM model with the prescribed T and R, with the observations as a
    # reference level; strictly speaking, this FVM simulation isn't necessary,
    # since we can also compute the heads from the head drops, but it serves to 
    # verify that the samples we created are physically valid.
    heads = solve_FVM(
        grid        = grid,
        T           = T,
        Sy          = Sy,
        R           = R,
        fixed_heads = {},
        prev_heads  = None,
        dt          = None,
        return_flows=False,
        gauge_node  = obs_left,
        gauge_value = h_left,
    )

    # -------------------------------------------------------------------------
    # Verify the sample
    # -------------------------------------------------------------------------
    
    # Check that we match the observations
    if abs(heads[obs_left] - h_left) > 1e-8:
        return None
    if abs(heads[obs_right] - h_right) > 1e-8:
        return None

    # Check that the heads we obtain lie inside the bounds
    for node in ordered_nodes:
        if heads[node] < h_bounds[node][0] - 1e-8 or heads[node] > h_bounds[node][1] + 1e-8:
            return None

    # Check that the transmissivities we obtain lie inside the bounds
    for edge in ordered_edges:
        if T[edge] < T_bounds[edge][0] - 1e-12 or T[edge] > T_bounds[edge][1] + 1e-12:
            return None

    # Check that the recharge we obtain lies inside the bounds
    for node in ordered_nodes:
        if R[node] < R_bounds[node][0] - 1e-12 or R[node] > R_bounds[node][1] + 1e-12:
            return None

    # Return the results
    return {
        "heads": heads,
        "T": T,
        "R": R,
        "q": q,
    }


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    
    random_seeds = [0,1,2,3,4,5,6,7,8,9]
    
    for random_seed in random_seeds:
        
        np.random.seed(random_seed)

        # -------------------------------------------------------------------------
        # Settings
        # -------------------------------------------------------------------------
        
        # We want 1000 samples; let the solver try 100 times per sample
        N = 100000
        max_attempts = 10000000
    
        # -------------------------------------------------------------------------
        # Build the same steady 1D model as in main.py
        # -------------------------------------------------------------------------
        
        # Define grid spacig´ng
        dx = 10  # in m
        timesteps = 1
    
        # Set up the grid geometry
        polygon = [(0, -dx / 2), (dx * 10, -dx / 2), (dx * 10, dx / 2), (0, dx / 2)]
        seed_point = (dx / 2, dx / 2)
    
        # Build a square grid
        base_grid = build_square_grid(polygon, seed_point, dx)
    
        # Initialize the grid graph
        grid = nx.Graph()
        nodes = list(base_grid.nodes)
    
        # Well, we only have one timestep =P
        for t in range(timesteps):
    
            # Add the nodes
            for node in nodes:
                grid.add_node((t, node))
                grid.nodes[(t, node)]["xpos"]       = base_grid.nodes[node]["xpos"]
                grid.nodes[(t, node)]["ypos"]       = base_grid.nodes[node]["ypos"]
                grid.nodes[(t, node)]["area"]       = base_grid.nodes[node]["area"]
                grid.nodes[(t, node)]["vertices"]   = base_grid.nodes[node]["vertices"]
                grid.nodes[(t, node)]["h"]          = sympy.Symbol(f'h_{(t, node)}')
                grid.nodes[(t, node)]["R"]          = sympy.Symbol(f'R_{(t, node)}')
    
            # Add the spatial edges
            for edge in list(base_grid.edges):
                j, i = edge
                grid.add_edge((t, j), (t, i))
                grid.edges[(t, j), (t, i)]["start"]     = (t, j)
                grid.edges[(t, j), (t, i)]["end"]       = (t, i)
                grid.edges[(t, j), (t, i)]["w"]         = dx
                grid.edges[(t, j), (t, i)]["dx"]        = dx
                grid.edges[(t, j), (t, i)]["T"]         = sympy.Symbol(f'T_{edge}')
                grid.edges[(t, j), (t, i)]["qx"]        = sympy.Symbol(f'qx_{(t, j, i)}')
                grid.edges[(t, j), (t, i)]["dhx"]       = sympy.Symbol(f'dhx_{(t, j, i)}')
                grid.edges[(t, j), (t, i)]["flow sign"] = None
    
        # -------------------------------------------------------------------------
        # Define the same bounds as in main.py
        # -------------------------------------------------------------------------
        
        setup_dictionary = {
            "h": {
                "default": (3.0, 12.0),
                (0, 3): (10.0, 10.0),
                (0, 6): (7.0, 7.0),
            },
            "T": {
                "default": (1e-4, 1e-1),
            },
            "R": {
                "default": (0.0, 0.0),
                (0, 0): (1e-6, 1e-4),
                (0, 9): (-1e-3, -1e-5),
            },
        }
    
        bounds = {}
    
        for identifier in list(grid.nodes):
    
            # Head bounds
            key = str(grid.nodes[identifier]["h"])
            if identifier in setup_dictionary["h"]:
                bounds[key] = setup_dictionary["h"][identifier]
            else:
                bounds[key] = setup_dictionary["h"]["default"]
    
            # Recharge bounds
            key = str(grid.nodes[identifier]["R"])
            if identifier in setup_dictionary["R"]:
                bounds[key] = setup_dictionary["R"][identifier]
            else:
                bounds[key] = setup_dictionary["R"]["default"]
    
        for edge in list(grid.edges):
    
            # Transmissivity bounds
            key = str(grid.edges[edge]["T"])
            if edge in setup_dictionary["T"]:
                bounds[key] = setup_dictionary["T"][edge]
            else:
                bounds[key] = setup_dictionary["T"]["default"]
    
        # -------------------------------------------------------------------------
        # Define observations and source/sink cells
        # -------------------------------------------------------------------------
        
        gauge_nodes = [(0, 3), (0, 6)]
        gauge_values = [10.0, 7.0]
        source_node = (0, 0)
        sink_node = (0, 9)
    
        # -------------------------------------------------------------------------
        # Start the sampling
        # -------------------------------------------------------------------------
        
        samples = []
        attempts = 0
    
        # Fill the list
        while len(samples) < N and attempts < max_attempts:
            
            # Increment the attempt counter
            attempts += 1
    
            # Draw a sample that matches the observations
            sample = sample_perfect_fit(
                grid=grid,
                bounds=bounds,
                gauge_nodes=gauge_nodes,
                gauge_values=gauge_values,
                source_node=source_node,
                sink_node=sink_node,
            )
            
            # If infeasible, skip to next iteration
            if sample is None:
                continue
    
            # Else, append the result
            samples.append(sample)
            
            # Provide incremental updates
            if len(samples) % 100 == 0:
                print(f"Accepted {len(samples)} samples after {attempts} attempts.")
    
        # If we didn't get enough samples, raise an error
        if len(samples) < N:
            raise Exception(
                f"Only found {len(samples)} exact-fit samples after {attempts} attempts."
            )
    
        # -------------------------------------------------------------------------
        # Save results for plotting
        # -------------------------------------------------------------------------
        
        # Create a dictionary with the results...
        out = {
            "samples": samples,
            "attempts": attempts,
            "acceptance_rate": len(samples) / attempts,
            "gauge_nodes": gauge_nodes,
            "gauge_values": gauge_values,
            "source_node": source_node,
            "sink_node": sink_node,
            "bounds": bounds,
            "grid": grid,
        }
    
        # ...and save it. Done!
        pickle.dump(out, open("Monte_Carlo_samples_RS="+str(random_seed)+".p", "wb"))
        
        obs_sigmas = [0.001, 0.01, 0.1]
        
        for obs_sigma in obs_sigmas:
        
            proposal_space_T = "log"      # "linear", "log", or "symlog"
            proposal_space_q = "symlog"      # "linear", "log", or "symlog"
            proposal_space_h_ref = "linear"  # "linear", "log", or "symlog"
        
            symlog_T_linthresh = 1e-6
            symlog_q_linthresh = 1e-6
            symlog_h_ref_linthresh = 1.0
            
            out = mcmc_sampler_pymc(
                grid=grid,
                bounds=bounds,
                gauge_nodes=[(0, 3), (0, 6)],
                gauge_values=[10.0, 7.0],
                source_node=(0, 0),
                sink_node=(0, 9),
                obs_sigma=obs_sigma,   # scalar or length-2 array
                draws=25000,
                tune=10000,
                chains=4,
                cores=4,
                random_seed=random_seed,
                transform_T=proposal_space_T,
                transform_q=proposal_space_q,
                transform_h_ref=proposal_space_h_ref,
                symlog_T_linthresh=symlog_T_linthresh,
                symlog_q_linthresh=symlog_q_linthresh,
                symlog_h_ref_linthresh=symlog_h_ref_linthresh,
            )
            
            
            print(f"Acceptance rate was {out['acceptance_rate']}.")
            pickle.dump(out["samples"], open("MCMC_samples_sigma="+str(obs_sigma)+"_RS="+str(random_seed)+".p", "wb"))
            
        
    
    
    