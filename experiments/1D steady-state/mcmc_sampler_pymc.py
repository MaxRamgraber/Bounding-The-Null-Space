import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from solve_FVM_v03 import solve_FVM

pytensor.config.cxx = ""

try:
    from mc_sampler_simplified import sample_perfect_fit
except Exception:
    sample_perfect_fit = None

# =============================================================================
# Helpers
# =============================================================================

def order_1d_grid(grid):
    """
    Order nodes and edges from left to right for a 1D chain-like grid.
    """

    ordered_nodes = [u for u in grid.nodes]
    ordered_nodes = sorted(
        ordered_nodes,
        key=lambda u: (grid.nodes[u]["xpos"], grid.nodes[u]["ypos"], u[1] if isinstance(u, tuple) and len(u) > 1 else u),
    )

    ordered_edges = []
    for a, b in zip(ordered_nodes[:-1], ordered_nodes[1:]):
        if (a, b) in grid.edges:
            ordered_edges.append((a, b))
        elif (b, a) in grid.edges:
            ordered_edges.append((b, a))
        else:
            raise Exception(f"No spatial edge found between consecutive nodes {a} and {b}.")

    return ordered_nodes, ordered_edges


def extract_bounds(grid, bounds, ordered_nodes, ordered_edges):

    h_bounds = {node: bounds[str(grid.nodes[node]["h"])] for node in ordered_nodes}
    T_bounds = {edge: bounds[str(grid.edges[edge]["T"])] for edge in ordered_edges}
    R_bounds = {node: bounds[str(grid.nodes[node]["R"])] for node in ordered_nodes}

    return h_bounds, T_bounds, R_bounds


def build_balanced_recharge(grid, ordered_nodes, source_node, sink_node, q):

    area_source = float(grid.nodes[source_node]["area"])
    area_sink = float(grid.nodes[sink_node]["area"])

    R = {node: 0.0 for node in ordered_nodes}
    R[source_node] = float(q) / area_source
    R[sink_node] = -float(q) / area_sink

    return R


def compute_q_bounds(grid, ordered_nodes, bounds, source_node, sink_node):
    """
    Compute feasible total-flow bounds implied by the source/sink recharge bounds.
    """

    _, _, R_bounds = extract_bounds(grid, bounds, ordered_nodes, [])

    area_source = float(grid.nodes[source_node]["area"])
    area_sink = float(grid.nodes[sink_node]["area"])

    q_lo = max(R_bounds[source_node][0] * area_source, -R_bounds[sink_node][1] * area_sink)
    q_hi = min(R_bounds[source_node][1] * area_source, -R_bounds[sink_node][0] * area_sink)

    if q_lo > q_hi:
        raise ValueError("Source/sink recharge bounds imply an empty feasible interval for q.")

    return float(q_lo), float(q_hi)


def compute_h_ref_bounds(h_bounds, observation_nodes=None, use_observation_bounds=False):
    """
    Use a broad absolute-head interval for the datum shift.

    By default observation-node bounds are excluded because in the original exact-fit
    setup they encode the observations as hard constraints. In an MCMC likelihood-
    based formulation those should typically be treated as data instead.
    """

    observation_nodes = set(observation_nodes or [])

    lowers = []
    uppers = []
    for node, (lo, hi) in h_bounds.items():
        if (node in observation_nodes) and (not use_observation_bounds):
            continue
        lowers.append(float(lo))
        uppers.append(float(hi))

    if len(lowers) == 0:
        raise ValueError("No head bounds available to define a prior interval for h_ref.")

    return float(min(lowers)), float(max(uppers))


def forward_absolute_heads(
    grid,
    ordered_nodes,
    ordered_edges,
    T_values,
    q,
    h_ref,
    source_node,
    sink_node,
    gauge_node=None,
):
    """
    Solve the steady-state Neumann problem with an arbitrary zero gauge, then shift
    all heads by a datum offset h_ref.
    """

    if gauge_node is None:
        gauge_node = ordered_nodes[0]

    T = {edge: float(T_values[k]) for k, edge in enumerate(ordered_edges)}
    R = build_balanced_recharge(grid, ordered_nodes, source_node, sink_node, q)
    Sy = {node: 0.0 for node in ordered_nodes}

    rel_heads = solve_FVM(
        grid=grid,
        T=T,
        Sy=Sy,
        R=R,
        fixed_heads={},
        prev_heads=None,
        dt=None,
        return_flows=False,
        gauge_node=gauge_node,
        gauge_value=0.0,
    )

    abs_heads = {node: float(rel_heads[node]) + float(h_ref) for node in ordered_nodes}

    return abs_heads, T, R


def gaussian_loglike(predicted, observed, sigma):

    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    return float(
        -0.5 * np.sum(((observed - predicted) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))
    )


def forward_transform_np(x, mode, linthresh):
    x = np.asarray(x, dtype=float)
    if mode == "linear":
        return x
    if mode == "log":
        if np.any(x <= 0.0):
            raise ValueError("log transform requires strictly positive bounds.")
        return np.log(x)
    if mode == "symlog":
        return np.sign(x) * np.log1p(np.abs(x) / float(linthresh))
    raise ValueError(f"Unknown transform mode: {mode}")


def inverse_transform_pt(z, mode, linthresh):
    if mode == "linear":
        return z
    if mode == "log":
        return pt.exp(z)
    if mode == "symlog":
        return pt.sgn(z) * float(linthresh) * (pt.exp(pt.abs(z)) - 1.0)
    raise ValueError(f"Unknown transform mode: {mode}")


def logabsdet_inverse_transform_pt(z, mode, linthresh):
    if mode == "linear":
        return pt.zeros_like(z)
    if mode == "log":
        return z
    if mode == "symlog":
        return np.log(float(linthresh)) + pt.abs(z)
    raise ValueError(f"Unknown transform mode: {mode}")




# =============================================================================
# Main sampler
# =============================================================================

def mcmc_sampler_pymc(
    grid,
    bounds,
    gauge_nodes,
    gauge_values,
    source_node,
    sink_node,
    obs_sigma,
    draws=1000,
    tune=1000,
    chains=4,
    cores=1,
    random_seed=42,
    progressbar=True,
    return_idata=True,
    exclude_observation_bounds=True,
    transform_T="linear",
    transform_q="linear",
    transform_h_ref="linear",
    symlog_T_linthresh=1e-6,
    symlog_q_linthresh=1e-6,
    symlog_h_ref_linthresh=1.0,
):
    """
    MCMC reference sampler using PyMC + DEMetropolisZ.
    """

    ordered_nodes, ordered_edges = order_1d_grid(grid)
    h_bounds, T_bounds, R_bounds = extract_bounds(grid, bounds, ordered_nodes, ordered_edges)

    obs_nodes = list(gauge_nodes)
    obs_values = np.asarray(gauge_values, dtype=float)
    obs_sigma = np.asarray(obs_sigma, dtype=float)
    if obs_sigma.ndim == 0:
        obs_sigma = np.full(obs_values.shape, float(obs_sigma))

    if obs_sigma.shape != obs_values.shape:
        raise ValueError("obs_sigma must be scalar or have the same length as gauge_values.")

    T_lower = np.asarray([T_bounds[edge][0] for edge in ordered_edges], dtype=float)
    T_upper = np.asarray([T_bounds[edge][1] for edge in ordered_edges], dtype=float)

    q_lo, q_hi = compute_q_bounds(grid, ordered_nodes, bounds, source_node, sink_node)
    h_ref_lo, h_ref_hi = compute_h_ref_bounds(
        h_bounds,
        observation_nodes=obs_nodes,
        use_observation_bounds=not exclude_observation_bounds,
    )

    T_lower_z = forward_transform_np(T_lower, transform_T, symlog_T_linthresh)
    T_upper_z = forward_transform_np(T_upper, transform_T, symlog_T_linthresh)
    q_lo_z = float(forward_transform_np(q_lo, transform_q, symlog_q_linthresh))
    q_hi_z = float(forward_transform_np(q_hi, transform_q, symlog_q_linthresh))
    h_ref_lo_z = float(forward_transform_np(h_ref_lo, transform_h_ref, symlog_h_ref_linthresh))
    h_ref_hi_z = float(forward_transform_np(h_ref_hi, transform_h_ref, symlog_h_ref_linthresh))

    # Remove observation nodes from the hard bound checks unless explicitly requested.
    bounded_nodes = []
    for node in ordered_nodes:
        if exclude_observation_bounds and node in obs_nodes:
            continue
        bounded_nodes.append(node)

    @as_op(itypes=[pt.dvector, pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
    def loglike_op(T_values, q, h_ref):
        try:
            heads, _, R = forward_absolute_heads(
                grid=grid,
                ordered_nodes=ordered_nodes,
                ordered_edges=ordered_edges,
                T_values=T_values,
                q=q,
                h_ref=h_ref,
                source_node=source_node,
                sink_node=sink_node,
                gauge_node=ordered_nodes[0],
            )
        except Exception:
            return np.array(-np.inf, dtype=np.float64)

        # Hard support checks for derived recharge.
        for node in ordered_nodes:
            lo, hi = R_bounds[node]
            if R[node] < lo - 1e-12 or R[node] > hi + 1e-12:
                return np.array(-np.inf, dtype=np.float64)

        # Hard support checks for simulated heads.
        for node in bounded_nodes:
            lo, hi = h_bounds[node]
            if heads[node] < lo - 1e-10 or heads[node] > hi + 1e-10:
                return np.array(-np.inf, dtype=np.float64)

        predicted = np.asarray([heads[node] for node in obs_nodes], dtype=float)
        ll = gaussian_loglike(predicted, obs_values, obs_sigma)

        return np.array(ll, dtype=np.float64)

    init_sample = None
    if sample_perfect_fit is not None:
        try:
            init_sample = sample_perfect_fit(
                grid=grid,
                bounds=bounds,
                gauge_nodes=gauge_nodes,
                gauge_values=gauge_values,
                source_node=source_node,
                sink_node=sink_node,
            )
        except Exception:
            init_sample = None

    if init_sample is not None:
        T_init = np.asarray([init_sample["T"][edge] for edge in ordered_edges], dtype=float)
        q_init = float(init_sample["q"])
    else:
        T_init = np.sqrt(T_lower * T_upper)
        q_init = 0.5 * (q_lo + q_hi)

    try:
        rel_heads_init, _, _ = forward_absolute_heads(
            grid=grid,
            ordered_nodes=ordered_nodes,
            ordered_edges=ordered_edges,
            T_values=T_init,
            q=q_init,
            h_ref=0.0,
            source_node=source_node,
            sink_node=sink_node,
            gauge_node=ordered_nodes[0],
        )
        h_ref_init = float(np.mean([obs_values[i] - rel_heads_init[node] for i, node in enumerate(obs_nodes)]))
    except Exception:
        h_ref_init = 0.5 * (h_ref_lo + h_ref_hi)

    h_ref_init = float(np.clip(h_ref_init, h_ref_lo, h_ref_hi))

    initvals = {
        "T_vec_z": np.asarray(forward_transform_np(T_init, transform_T, symlog_T_linthresh), dtype=float),
        "q_z": np.array(forward_transform_np(q_init, transform_q, symlog_q_linthresh), dtype=float),
        "h_ref_z": np.array(forward_transform_np(h_ref_init, transform_h_ref, symlog_h_ref_linthresh), dtype=float),
    }

    with pm.Model() as model:
        
        T_vec_z = pm.Uniform(
            "T_vec_z",
            lower=T_lower_z,
            upper=T_upper_z,
            shape=len(ordered_edges),
            default_transform=None,
        )
        q_z = pm.Uniform(
            "q_z",
            lower=q_lo_z,
            upper=q_hi_z,
            default_transform=None,
        )
        h_ref_z = pm.Uniform(
            "h_ref_z",
            lower=h_ref_lo_z,
            upper=h_ref_hi_z,
            default_transform=None,
        )
        
        T_vec = pm.Deterministic("T_vec", inverse_transform_pt(T_vec_z, transform_T, symlog_T_linthresh))
        q = pm.Deterministic("q", inverse_transform_pt(q_z, transform_q, symlog_q_linthresh))
        h_ref = pm.Deterministic("h_ref", inverse_transform_pt(h_ref_z, transform_h_ref, symlog_h_ref_linthresh))

        pm.Potential("jac_T", pt.sum(logabsdet_inverse_transform_pt(T_vec_z, transform_T, symlog_T_linthresh)))
        pm.Potential("jac_q", logabsdet_inverse_transform_pt(q_z, transform_q, symlog_q_linthresh))
        pm.Potential("jac_h_ref", logabsdet_inverse_transform_pt(h_ref_z, transform_h_ref, symlog_h_ref_linthresh))
        pm.Potential("obs_loglike", loglike_op(T_vec, q, h_ref))

        step = pm.DEMetropolisZ(
            vars=[T_vec_z, q_z, h_ref_z],
            initial_point=initvals,
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            step=step,
            initvals=initvals,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    samples = []

    T_all = np.asarray(idata.posterior["T_vec"]).reshape(-1, len(ordered_edges))
    q_all = np.asarray(idata.posterior["q"]).reshape(-1)
    h_ref_all = np.asarray(idata.posterior["h_ref"]).reshape(-1)

    for k in range(T_all.shape[0]):
        heads, T, R = forward_absolute_heads(
            grid=grid,
            ordered_nodes=ordered_nodes,
            ordered_edges=ordered_edges,
            T_values=T_all[k],
            q=q_all[k],
            h_ref=h_ref_all[k],
            source_node=source_node,
            sink_node=sink_node,
            gauge_node=ordered_nodes[0],
        )

        samples.append(
            {
                "heads": heads,
                "T": T,
                "R": R,
                "q": float(q_all[k]),
                "h_ref": float(h_ref_all[k]),
            }
        )

    out = {
        "samples": samples,
        "gauge_nodes": list(gauge_nodes),
        "gauge_values": list(np.asarray(gauge_values, dtype=float)),
        "source_node": source_node,
        "sink_node": sink_node,
        "bounds": bounds,
        "grid": grid,
        "ordered_nodes": ordered_nodes,
        "ordered_edges": ordered_edges,
        "obs_sigma": obs_sigma,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "sampler": "PyMC DEMetropolisZ",
        "q_bounds": (q_lo, q_hi),
        "h_ref_bounds": (h_ref_lo, h_ref_hi),
        "exclude_observation_bounds": bool(exclude_observation_bounds),
        "transform_T": transform_T,
        "transform_q": transform_q,
        "transform_h_ref": transform_h_ref,
        "symlog_T_linthresh": symlog_T_linthresh,
        "symlog_q_linthresh": symlog_q_linthresh,
        "symlog_h_ref_linthresh": symlog_h_ref_linthresh,
    }

    if "accepted" in idata.sample_stats:
        out["acceptance_rate"] = float(np.asarray(idata.sample_stats["accepted"]).mean())

    if return_idata:
        out["idata"] = idata

    return out
