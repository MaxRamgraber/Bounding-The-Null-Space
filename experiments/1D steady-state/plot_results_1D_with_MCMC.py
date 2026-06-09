import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import pickle
from build_constraints_nocurl import build_constraints



#%%

def ensure_coverage_results_nested(
    grid,
    bounds,
    random_seeds=None,
    obs_sigmas=None,
    sizes=None,
    output_path="coverage_results_nested.p",
    R_linthresh=1e-5,
):
    
    """
    This function computes the coverage of the OBBT interval by the exact-fit
    sampler and the MCMC chains.
    """
    
    if random_seeds is None:
        random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if obs_sigmas is None:
        obs_sigmas = [0.001, 0.01, 0.1]

    if os.path.exists(output_path):
        return pickle.load(open(output_path, "rb"))

    # -------------------------------------------------------------------------
    # Order the 1D grid from left to right
    # -------------------------------------------------------------------------

    ordered_nodes = list(grid.nodes)
    ordered_nodes = sorted(
        ordered_nodes,
        key=lambda u: (grid.nodes[u]["xpos"], grid.nodes[u]["ypos"], u[1])
    )

    ordered_edges = []
    for a, b in zip(ordered_nodes[:-1], ordered_nodes[1:]):
        if (a, b) in grid.edges:
            ordered_edges.append((a, b))
        elif (b, a) in grid.edges:
            ordered_edges.append((b, a))
        else:
            raise Exception(f"No edge found between consecutive nodes {a} and {b}.")

    # -------------------------------------------------------------------------
    # Read final OBBT bounds in plotting order
    # -------------------------------------------------------------------------

    h_bounds = [tuple(map(float, bounds[str(grid.nodes[node]["h"])])) for node in ordered_nodes]
    R_bounds = [tuple(map(float, bounds[str(grid.nodes[node]["R"])])) for node in ordered_nodes]
    T_bounds = [tuple(map(float, bounds[str(grid.edges[edge]["T"])])) for edge in ordered_edges]
    qx_bounds = [tuple(map(float, bounds[str(grid.edges[edge]["qx"])])) for edge in ordered_edges]

    # Ignore zero-width observation cells in h
    h_keep = [j for j, (lo, hi) in enumerate(h_bounds) if abs(hi - lo) > 1e-12]

    # Ignore zero-recharge cells in R
    R_keep = [j for j, (lo, hi) in enumerate(R_bounds) if not (abs(lo) <= 1e-12 and abs(hi) <= 1e-12)]

    # -------------------------------------------------------------------------
    # Determine valid sample sizes
    # -------------------------------------------------------------------------

    all_lengths = []

    for rs in random_seeds:
        mc_path = "Monte_Carlo_samples_RS=" + str(rs) + ".p"
        mc_obj = pickle.load(open(mc_path, "rb"))
        if isinstance(mc_obj, dict) and "samples" in mc_obj:
            mc_samples = mc_obj["samples"]
        elif isinstance(mc_obj, list):
            mc_samples = mc_obj
        else:
            raise Exception("Monte Carlo pickle must be either a dict with 'samples' or a plain list.")
        all_lengths.append(len(mc_samples))

    for obs_sigma in obs_sigmas:
        for rs in random_seeds:
            mcmc_path = "MCMC_samples_sigma=" + str(obs_sigma) + "_RS=" + str(rs) + ".p"
            mcmc_obj = pickle.load(open(mcmc_path, "rb"))
            if isinstance(mcmc_obj, dict) and "samples" in mcmc_obj:
                mcmc_samples = mcmc_obj["samples"]
            elif isinstance(mcmc_obj, list):
                mcmc_samples = mcmc_obj
            else:
                raise Exception("MCMC pickle must be either a dict with 'samples' or a plain list.")
            all_lengths.append(len(mcmc_samples))

    common_max = min(all_lengths)
    if common_max < 1:
        raise Exception("All sample files must contain at least one sample.")

    if sizes is None:
        sizes = []
        p = 10
        while p <= common_max:
            sizes.append(p)
            p *= 10
        if len(sizes) == 0 or sizes[-1] != common_max:
            sizes.append(common_max)
    else:
        sizes = sorted([int(n) for n in sizes if 1 <= int(n) <= common_max])
        if len(sizes) == 0:
            raise Exception("No valid ensemble sizes remain.")

    # -------------------------------------------------------------------------
    # Prepare output
    # -------------------------------------------------------------------------

    results = {}
    results["MC"] = {}
    for n in sizes:
        results["MC"][n] = {"h": [], "qx": [], "R": [], "T": []}

    for obs_sigma in obs_sigmas:
        method = "MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=" + str(obs_sigma)
        results[method] = {}
        for n in sizes:
            results[method][n] = {"h": [], "qx": [], "R": [], "T": []}

    # -------------------------------------------------------------------------
    # Helper for sample -> arrays
    # -------------------------------------------------------------------------

    def sample_list_to_arrays(samples):
        all_h = []
        all_qx = []
        all_R = []
        all_T = []

        for sample in samples:
            h_row = [sample["heads"][node] for node in ordered_nodes]
            R_row = [sample["R"][node] for node in ordered_nodes]

            T_row = []
            qx_row = []
            for edge in ordered_edges:
                if edge in sample["T"]:
                    T_val = sample["T"][edge]
                elif (edge[1], edge[0]) in sample["T"]:
                    T_val = sample["T"][(edge[1], edge[0])]
                else:
                    raise Exception(f"Edge {edge} not found in sample transmissivities.")

                a, b = edge
                dx = float(grid.edges[edge]["dx"])
                w = float(grid.edges[edge]["w"])
                qx_val = (sample["heads"][a] - sample["heads"][b]) / dx * T_val * w

                T_row.append(T_val)
                qx_row.append(qx_val)

            all_h.append(h_row)
            all_qx.append(qx_row)
            all_R.append(R_row)
            all_T.append(T_row)

        return (
            np.asarray(all_h, dtype=float),
            np.asarray(all_qx, dtype=float),
            np.asarray(all_R, dtype=float),
            np.asarray(all_T, dtype=float),
        )

    # -------------------------------------------------------------------------
    # Monte Carlo
    # -------------------------------------------------------------------------

    for rs in random_seeds:
        mc_path = "Monte_Carlo_samples_RS=" + str(rs) + ".p"
        mc_obj = pickle.load(open(mc_path, "rb"))
        if isinstance(mc_obj, dict) and "samples" in mc_obj:
            mc_samples = mc_obj["samples"]
        elif isinstance(mc_obj, list):
            mc_samples = mc_obj
        else:
            raise Exception("Monte Carlo pickle must be either a dict with 'samples' or a plain list.")

        all_h, all_qx, all_R, all_T = sample_list_to_arrays(mc_samples)
        
        sample_order = np.arange(all_h.shape[0]).reshape(4, -1).T.ravel()
        all_h, all_qx, all_R, all_T = [x[sample_order] for x in (all_h, all_qx, all_R, all_T)]

        for n in sizes:
            h_now = all_h[:n, :]
            qx_now = all_qx[:n, :]
            R_now = all_R[:n, :]
            T_now = all_T[:n, :]

            h_fracs = []
            for j in h_keep:
                lo, hi = h_bounds[j]
                s_lo = float(np.min(h_now[:, j]))
                s_hi = float(np.max(h_now[:, j]))
                width = hi - lo
                overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                h_fracs.append(overlap / width)

            qx_fracs = []
            for j in range(qx_now.shape[1]):
                lo, hi = np.log(qx_bounds[j][0]), np.log(qx_bounds[j][1])
                s_lo = np.log(float(np.min(qx_now[:, j])))
                s_hi = np.log(float(np.max(qx_now[:, j])))
                width = hi - lo
                overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                qx_fracs.append(overlap / width)

            R_fracs = []
            for j in R_keep:
                lo = np.sign(R_bounds[j][0]) * np.log1p(np.abs(R_bounds[j][0]) / R_linthresh)
                hi = np.sign(R_bounds[j][1]) * np.log1p(np.abs(R_bounds[j][1]) / R_linthresh)
                s_lo_raw = float(np.min(R_now[:, j]))
                s_hi_raw = float(np.max(R_now[:, j]))
                s_lo = np.sign(s_lo_raw) * np.log1p(np.abs(s_lo_raw) / R_linthresh)
                s_hi = np.sign(s_hi_raw) * np.log1p(np.abs(s_hi_raw) / R_linthresh)
                width = hi - lo
                overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                R_fracs.append(overlap / width)

            T_fracs = []
            for j in range(T_now.shape[1]):
                lo, hi = np.log(T_bounds[j][0]), np.log(T_bounds[j][1])
                s_lo = np.log(float(np.min(T_now[:, j])))
                s_hi = np.log(float(np.max(T_now[:, j])))
                width = hi - lo
                overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                T_fracs.append(overlap / width)

            results["MC"][n]["h"].append(float(np.mean(h_fracs)))
            results["MC"][n]["qx"].append(float(np.mean(qx_fracs)))
            results["MC"][n]["R"].append(float(np.mean(R_fracs)))
            results["MC"][n]["T"].append(float(np.mean(T_fracs)))

    # -------------------------------------------------------------------------
    # MCMC
    # -------------------------------------------------------------------------

    for obs_sigma in obs_sigmas:
        method = "MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=" + str(obs_sigma)

        for rs in random_seeds:
            mcmc_path = "MCMC_samples_sigma=" + str(obs_sigma) + "_RS=" + str(rs) + ".p"
            mcmc_obj = pickle.load(open(mcmc_path, "rb"))
            if isinstance(mcmc_obj, dict) and "samples" in mcmc_obj:
                mcmc_samples = mcmc_obj["samples"]
            elif isinstance(mcmc_obj, list):
                mcmc_samples = mcmc_obj
            else:
                raise Exception("MCMC pickle must be either a dict with 'samples' or a plain list.")

            all_h, all_qx, all_R, all_T = sample_list_to_arrays(mcmc_samples)

            for n in sizes:
                h_now = all_h[:n, :]
                qx_now = all_qx[:n, :]
                R_now = all_R[:n, :]
                T_now = all_T[:n, :]

                h_fracs = []
                for j in h_keep:
                    lo, hi = h_bounds[j]
                    s_lo = float(np.min(h_now[:, j]))
                    s_hi = float(np.max(h_now[:, j]))
                    width = hi - lo
                    overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                    h_fracs.append(overlap / width)

                qx_fracs = []
                for j in range(qx_now.shape[1]):
                    lo, hi = np.log(qx_bounds[j][0]), np.log(qx_bounds[j][1])
                    s_lo = np.log(float(np.min(qx_now[:, j])))
                    s_hi = np.log(float(np.max(qx_now[:, j])))
                    width = hi - lo
                    overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                    qx_fracs.append(overlap / width)

                R_fracs = []
                for j in R_keep:
                    lo = np.sign(R_bounds[j][0]) * np.log1p(np.abs(R_bounds[j][0]) / R_linthresh)
                    hi = np.sign(R_bounds[j][1]) * np.log1p(np.abs(R_bounds[j][1]) / R_linthresh)
                    s_lo_raw = float(np.min(R_now[:, j]))
                    s_hi_raw = float(np.max(R_now[:, j]))
                    s_lo = np.sign(s_lo_raw) * np.log1p(np.abs(s_lo_raw) / R_linthresh)
                    s_hi = np.sign(s_hi_raw) * np.log1p(np.abs(s_hi_raw) / R_linthresh)
                    width = hi - lo
                    overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                    R_fracs.append(overlap / width)

                T_fracs = []
                for j in range(T_now.shape[1]):
                    lo, hi = np.log(T_bounds[j][0]), np.log(T_bounds[j][1])
                    s_lo = np.log(float(np.min(T_now[:, j])))
                    s_hi = np.log(float(np.max(T_now[:, j])))
                    width = hi - lo
                    overlap = max(0.0, min(hi, s_hi) - max(lo, s_lo))
                    T_fracs.append(overlap / width)

                results[method][n]["h"].append(float(np.mean(h_fracs)))
                results[method][n]["qx"].append(float(np.mean(qx_fracs)))
                results[method][n]["R"].append(float(np.mean(R_fracs)))
                results[method][n]["T"].append(float(np.mean(T_fracs)))

    # -------------------------------------------------------------------------
    # Average over random seeds
    # -------------------------------------------------------------------------

    for method in results:
        for n in results[method]:
            for var in results[method][n]:
                results[method][n][var] = float(np.mean(results[method][n][var]))

    pickle.dump(results, open(output_path, "wb"))
    return results

#%%






work_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(work_dir, "checkpoints")
coverage_pickle = os.path.join(work_dir, "coverage_results_nested.p")
# coverage_builder = os.path.join(work_dir, "compare_obbt_coverage_1d_simple_transforms_v05_nested_pickle.py")
obs_sigmas = [0.001, 0.01, 0.1]

# =============================================================================
# Average MCMC acceptance rates by sigma_obs
# =============================================================================

random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def get_mcmc_acceptance_rate(mcmc_obj):
    # Directly stored acceptance rate
    for key in ["acceptance_rate", "avg_acceptance_rate", "mean_acceptance_rate", "accept_rate"]:
        if isinstance(mcmc_obj, dict) and key in mcmc_obj:
            val = mcmc_obj[key]
            if np.isscalar(val):
                return float(val)
            val = np.asarray(val, dtype=float).ravel()
            if val.size > 0:
                return float(np.mean(val))

    # Stored accept/reject flags
    for key in ["accepted", "accept_flags", "is_accepted", "accept"]:
        if isinstance(mcmc_obj, dict) and key in mcmc_obj:
            val = np.asarray(mcmc_obj[key], dtype=float).ravel()
            if val.size > 0:
                return float(np.mean(val))

    # Stored counts
    pairs = [
        ("n_accept", "n_proposals"),
        ("n_accepted", "n_proposals"),
        ("num_accepted", "num_proposals"),
        ("accepted_count", "proposal_count"),
    ]
    for k_num, k_den in pairs:
        if isinstance(mcmc_obj, dict) and k_num in mcmc_obj and k_den in mcmc_obj:
            den = float(mcmc_obj[k_den])
            if den > 0:
                return float(mcmc_obj[k_num]) / den

    # Fallback: estimate from repeated consecutive samples
    # (unchanged state -> rejected proposal)
    if isinstance(mcmc_obj, dict) and "samples" in mcmc_obj:
        samples = mcmc_obj["samples"]
    elif isinstance(mcmc_obj, list):
        samples = mcmc_obj
    else:
        samples = None

    if samples is not None and len(samples) > 1:
        def sample_signature(sample):
            heads = tuple((k, float(v)) for k, v in sorted(sample["heads"].items()))
            R = tuple((k, float(v)) for k, v in sorted(sample["R"].items()))
            T = tuple((k, float(v)) for k, v in sorted(sample["T"].items()))
            return (heads, R, T)

        sig_prev = sample_signature(samples[0])
        accepted = []

        for sample in samples[1:]:
            sig_now = sample_signature(sample)
            accepted.append(sig_now != sig_prev)
            sig_prev = sig_now

        return float(np.mean(accepted))

    return np.nan


acceptance_summary = {}

for obs_sigma in obs_sigmas:
    rates = []

    for rs in random_seeds:
        mcmc_path = os.path.join(work_dir, "MCMC_samples_sigma=" + str(obs_sigma) + "_RS=" + str(rs) + ".p")
        mcmc_obj = pickle.load(open(mcmc_path, "rb"))
        rate = get_mcmc_acceptance_rate(mcmc_obj)

        if not np.isnan(rate):
            rates.append(rate)

    acceptance_summary[obs_sigma] = float(np.mean(rates)) if len(rates) > 0 else np.nan

print("\nAverage MCMC acceptance rates")
for obs_sigma in obs_sigmas:
    rate = acceptance_summary[obs_sigma]
    if np.isnan(rate):
        print("sigma_obs = {} : unavailable".format(obs_sigma))
    else:
        print("sigma_obs = {} : {:.2%}".format(obs_sigma, rate))
        
#%%

files = os.listdir(checkpoint_dir)
iteration_files = []
for file in files:
    if file.startswith("iteration_") and file.endswith(".p") and file != "iteration_initial.p":
        try:
            int(file.replace("iteration_", "").replace(".p", ""))
            iteration_files.append(file)
        except ValueError:
            pass
iteration_files = sorted(iteration_files)

iteration_first = pickle.load(
    open(os.path.join(
        checkpoint_dir,
        "iteration_initial.p"), "rb")
)
iteration_last = pickle.load(
    open(os.path.join(
        checkpoint_dir,
        iteration_files[-1]), "rb")
)

# Optional Monte Carlo overlay
# mc_path = os.path.join(work_dir, "Monte_Carlo_samples.p")
mc_path = os.path.join(work_dir, "Monte_Carlo_samples_RS=" + str(0) + ".p")
mc_data = pickle.load(open(mc_path, "rb")) if os.path.exists(mc_path) else None
if mc_data is not None:
    mc_data["samples"] = mc_data["samples"][:1000]


# raise Exception

# Create nested coverage pickle on demand
coverage_results = ensure_coverage_results_nested(
    grid=iteration_last["grid"],
    bounds=iteration_last["bounds"],
    random_seeds=[0,1,2,3,4,5,6,7,8,9],
    obs_sigmas=[0.001, 0.01, 0.1],
    sizes=[10, 100, 1000, 10000, 100000],
    output_path="coverage_results_nested.p",
)

# if not os.path.exists(coverage_pickle):
#     if not os.path.exists(coverage_builder):
#         raise FileNotFoundError(
#             "coverage_results_nested.p not found and the helper script "
#             "compare_obbt_coverage_1d_simple_transforms_v05_nested_pickle.py is missing."
#         )
#     print("coverage_results_nested.p not found. Computing it now...")
#     runpy.run_path(coverage_builder, run_name="__main__")
#     if not os.path.exists(coverage_pickle):
#         raise FileNotFoundError("Failed to create coverage_results_nested.p")

C_INIT   = 'xkcd:silver'
C_TIGHT  = 'xkcd:cerulean'
LW_INIT  = 3.5          # grey line width
LW_TIGHT = 3.0          # blue line width
CAP_INIT  = 0.35        # half-width of end-caps (in x-axis data units, fraction of cell)
CAP_TIGHT = 0.25
MC_COLOR = 'xkcd:light orange'
MC_ALPHA = 0.25 # 0.05
MC_LW    = 0.2

def draw_intervals(ax, xpos, bounds_init, bounds_tight,
                   cap_i=CAP_INIT, cap_t=CAP_TIGHT):
    """
    Draw initial (grey) and tightened (blue) interval lines for each cell.
    """
    dx = xpos[1] - xpos[0] if len(xpos) > 1 else 10
    ci = cap_i * dx
    ct = cap_t * dx

    for xi, bi, bt in zip(xpos, bounds_init, bounds_tight):
        lo_i, hi_i = bi
        lo_t, hi_t = bt

        ax.plot([xi, xi], [lo_i, hi_i],
                color=C_INIT, lw=LW_INIT, solid_capstyle='round', zorder=2)
        ax.plot([xi - ci, xi + ci], [lo_i, lo_i],
                color=C_INIT, lw=LW_INIT, zorder=2)
        ax.plot([xi - ci, xi + ci], [hi_i, hi_i],
                color=C_INIT, lw=LW_INIT, zorder=2)

        ax.plot([xi, xi], [lo_t, hi_t],
                color=C_TIGHT, lw=LW_TIGHT, solid_capstyle='round', zorder=3)
        ax.plot([xi - ct, xi + ct], [lo_t, lo_t],
                color=C_TIGHT, lw=LW_TIGHT, zorder=3)
        ax.plot([xi - ct, xi + ct], [hi_t, hi_t],
                color=C_TIGHT, lw=LW_TIGHT, zorder=3)


def draw_mc_lines(ax, xpos, values, zorder=1):
    for row in values:
        ax.plot(xpos, row, color=MC_COLOR, alpha=MC_ALPHA, lw=MC_LW, zorder=zorder)


plt.figure(figsize=(12, 9))

gs = GridSpec(nrows=3, ncols=2)

xpos = np.arange(5, 105, 10)

mc_h = []
mc_qx = []
mc_R = []
mc_T = []
if mc_data is not None:
    for sample in mc_data["samples"]:
        mc_h.append([sample["heads"][(0, cell)] for cell in range(10)])
        mc_R.append([sample["R"][(0, cell)] for cell in range(10)])
        mc_T.append([sample["T"][((0, cell), (0, cell + 1))] for cell in range(9)])
        mc_qx.append([
            (sample["heads"][(0, cell)] - sample["heads"][(0, cell + 1)])
            / float(iteration_first["grid"].edges[((0, cell), (0, cell + 1))]["dx"])
            * sample["T"][((0, cell), (0, cell + 1))]
            * float(iteration_first["grid"].edges[((0, cell), (0, cell + 1))]["w"])
            for cell in range(9)
        ])

# =============================================================================
# A : Hydraulic heads h
# =============================================================================

plt.subplot(gs[0, 0])

h_init  = [iteration_first["bounds"][f"h_(0, {cell})"] for cell in range(10)]
h_final = [iteration_last["bounds"][f"h_(0, {cell})"] for cell in range(10)]

if mc_data is not None:
    draw_mc_lines(plt.gca(), xpos, mc_h)
draw_intervals(plt.gca(), xpos, h_init, h_final)

plt.axhline(0, color='k', lw=0.6)
plt.ylabel("hydraulic head in [m]")
plt.gca().set_xticks(xpos)
plt.gca().set_xticklabels([])
plt.grid(axis='y', lw=0.4, color='#ccc', zorder=0)
plt.title(r"$\mathbf{A}:$ hydraulic heads $h_i$", loc="left")

plt.scatter(
    35,
    10,
    color = "xkcd:orangish red",
    marker = "x",
    lw = 2,
    s = 200)
plt.text(
    35,
    9.5,
    "observation",
    color="xkcd:orangish red",
    rotation = 90,
    ha = "center",
    va = "top")

plt.scatter(
    65,
    7,
    color = "xkcd:orangish red",
    marker = "x",
    lw = 2,
    s = 200)
plt.text(
    65,
    6.5,
    "observation",
    color="xkcd:orangish red",
    rotation = 90,
    ha = "center",
    va = "top")

xlims = plt.gca().get_xlim()
ylims = plt.gca().get_ylim()
plt.gca().set_ylim([0,ylims[1]])

# =============================================================================
# B : Darcy flow qx
# =============================================================================

plt.subplot(gs[0, 1])

qx_xpos  = (xpos[:-1] + xpos[1:]) / 2
qx_init  = [iteration_first["bounds"][f"qx_(0, {cell}, {cell+1})"] for cell in range(9)]
qx_final = [iteration_last["bounds"][f"qx_(0, {cell}, {cell+1})"] for cell in range(9)]

if mc_data is not None:
    draw_mc_lines(plt.gca(), qx_xpos, mc_qx)
draw_intervals(plt.gca(), qx_xpos, qx_init, qx_final)

plt.axhline(0, color='k', lw=1.0, ls='--', zorder=1)
plt.ylabel("Darcy flow in [m³/s]")
plt.gca().set_xticks(xpos)
plt.gca().set_xticklabels([])
plt.grid(axis='y', lw=0.4, color='#ccc', zorder=0)
plt.gca().set_xlim(xlims)
plt.title(r"$\mathbf{B}:$ Darcy flow $q_{j \rightarrow i}$", loc="left")
plt.yscale('symlog', linthresh=1E-4)
plt.ylim(-2E0,2E0)

# =============================================================================
# C : Recharge R
# =============================================================================

plt.subplot(gs[1, 0])

R_init  = [iteration_first["bounds"][f"R_(0, {cell})"] for cell in range(10)]
R_final = [iteration_last["bounds"][f"R_(0, {cell})"] for cell in range(10)]

if mc_data is not None:
    draw_mc_lines(plt.gca(), xpos, mc_R)
draw_intervals(plt.gca(), xpos, R_init, R_final)

plt.axhline(0, color='k', lw=1.0, ls='--', zorder=1)
plt.ylabel("recharge in [m/s]")
plt.yscale('symlog', linthresh=1e-5)
plt.gca().set_xticks(xpos)
plt.gca().set_xticklabels([cell + 1 for cell in np.arange(10)])
plt.xlabel("grid cell")
plt.grid(axis='y', lw=0.4, color='#ccc', zorder=0)
plt.title(r"$\mathbf{C}:$ recharge $R_i$", loc="left")
plt.ylim(-2E-3,2E-4)


plt.yscale('symlog', linthresh=2E-6)

# =============================================================================
# D : Transmissivity T
# =============================================================================

plt.subplot(gs[1, 1])

T_xpos  = (xpos[:-1] + xpos[1:]) / 2
T_init  = [iteration_first["bounds"][f"T_({cell}, {cell+1})"] for cell in range(9)]
T_final = [iteration_last["bounds"][f"T_({cell}, {cell+1})"] for cell in range(9)]

if mc_data is not None:
    draw_mc_lines(plt.gca(), T_xpos, mc_T)
draw_intervals(plt.gca(), T_xpos, T_init, T_final)

plt.ylabel("transmissivity in [m²/s]")
plt.yscale('log')
plt.gca().set_xticks(xpos)
plt.gca().set_xticklabels([cell + 1 for cell in np.arange(10)])
plt.xlabel("grid cell")
plt.grid(axis='y', lw=0.4, color='#ccc', zorder=0)
plt.gca().set_xlim(xlims)
plt.title(r"$\mathbf{D}:$ transmissivity $T_{j,i}$", loc="left")


#%%

# =============================================================================
# E: Overall sampler coverage (averaged across random seeds)
# =============================================================================

plt.subplot(gs[2, :])

coverage_results = pickle.load(open(coverage_pickle, "rb"))

h_count = sum(1 for lo, hi in h_final if abs(float(hi) - float(lo)) > 1e-12)
qx_count = len(qx_final)
R_count = sum(1 for lo, hi in R_final if not (abs(float(lo)) <= 1e-12 and abs(float(hi)) <= 1e-12))
T_count = len(T_final)

total_count = h_count + qx_count + R_count + T_count

method_order = ["MC"] + ["MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=" + str(obs_sigma) for obs_sigma in obs_sigmas]
method_colors = {
    "MC": "xkcd:light orange",
    "MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=0.001": "xkcd:orange",
    "MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=0.01": "xkcd:orangish red",
    "MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=0.1": "crimson",
}


for method in method_order:
    if method not in coverage_results:
        continue

    size_keys = sorted(coverage_results[method].keys(), key=lambda x: int(x))
    xs = []
    ys = []

    for n in size_keys:
        vals = coverage_results[method][n]
        overall = (
            h_count * float(vals["h"]) +
            qx_count * float(vals["qx"]) +
            R_count * float(vals["R"]) +
            T_count * float(vals["T"])
        ) / float(total_count)
        xs.append(int(n))
        ys.append(overall)

    plt.plot(
        xs,
        ys,
        marker="o",
        lw=2,
        label=method,
        color=method_colors.get(method, None),
    )

plt.xscale("log")
plt.ylim(0.0, 1.05)
xlims = plt.gca().get_xlim()
plt.gca().set_xlim(xlims)
plt.plot(
    xlims,
    [1,1],
    color = "xkcd:cerulean")


plt.xlabel("sample size")
plt.ylabel("avg. coverage fraction of OBBT")
plt.grid(True, which="both", alpha=0.3)
plt.title(r"$\mathbf{E}:$ sample coverage of tightened bounds", loc="left")

#%%

# =============================================================================
# Shared legend
# =============================================================================

patch_i = mpatches.Patch(color=C_INIT,  label='initial bounds')
patch_t = mpatches.Patch(color=C_TIGHT, label='tightened bounds')
patch_exact = Line2D([0], [0], color=MC_COLOR, lw=2, label='exact sampler')
patch_MCMC1 = Line2D([0], [0], color="xkcd:orange", lw=2, label="MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=0.001")
patch_MCMC2 = Line2D([0], [0], color="xkcd:orangish red", lw=2, label="MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=0.01")
patch_MCMC3 = Line2D([0], [0], color="xkcd:crimson", lw=2, label="MCMC " + r"$\sigma_{\mathrm{obs}}$" + "=0.1")

plt.gcf().legend(
    handles=[patch_i,patch_MCMC1, patch_t,patch_MCMC2, patch_exact,patch_MCMC3],
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, -0.075),
    frameon=False,
    fontsize=10
)

plt.tight_layout()
plt.savefig("experiment_1D.pdf", dpi=300, bbox_inches="tight")
plt.savefig("experiment_1D.png", dpi=300, bbox_inches="tight")
plt.show()

time_span = iteration_last["current time"] - iteration_last["start time"]
print("One LP operation takes approximately {} seconds".format(
    time_span / max(len(iteration_files), 1) / len(iteration_last["variables"]) / 2))
print("One full OBBT pass takes approximately {} seconds".format(time_span))