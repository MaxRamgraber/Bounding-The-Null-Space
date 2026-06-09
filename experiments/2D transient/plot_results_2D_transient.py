import math
import os
import pickle
import re
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, SymLogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =============================================================================
# Colormaps
# =============================================================================

colors = ["#ffd6cb", "xkcd:orangish red", "xkcd:burgundy"]
cmap_red = matplotlib.colors.LinearSegmentedColormap.from_list("cmap_red", colors, N=256)

colors = ["xkcd:light sky blue", "xkcd:cerulean", "xkcd:midnight"]
cmap_blue = matplotlib.colors.LinearSegmentedColormap.from_list("cmap_blue", colors, N=256)

# Symmetric divergent map requested by user: cerulean -> white -> orangish red
cmap_divergent = LinearSegmentedColormap.from_list(
    "cmap_divergent",
    [
        (0.0, "xkcd:cerulean"),
        (0.5, "w"),
        (1.0, "xkcd:orangish red"),
    ],
    N=256,
)


# =============================================================================
# Load iterations
# =============================================================================

checkpoint_dir = "checkpoints"


def _iteration_key(name):
    match = re.search(r"(\d+)(?=\.p$)", name)
    return int(match.group(1)) if match else -1


iteration_files = sorted(
    [
        file for file in os.listdir(checkpoint_dir)
        if file.startswith("iteration")
        and file.endswith(".p")
        and file != "iteration_initial.p"
    ],
    key=_iteration_key,
)

if len(iteration_files) == 0:
    raise FileNotFoundError("No iteration*.p files found in the checkpoints directory.")

iteration_first = pickle.load(
    open(os.path.join(checkpoint_dir, "iteration_initial.p"), "rb")
)
iteration_last = pickle.load(
    open(os.path.join(checkpoint_dir, iteration_files[-1]), "rb")
)


# =============================================================================
# Figure setup
# =============================================================================

T = 5
source_cells = [100, 101, 90, 91, 92, 80, 81]
well_cell = 77

eps = 1e-30

fig = plt.figure(figsize=(12, 14))

gs = GridSpec(
    nrows=4,
    ncols=5,
    height_ratios=[1.25, 2.0, 2.0, 1.0],
    hspace=0.5,
)

subgs_top_row = GridSpecFromSubplotSpec(
    nrows=1,
    ncols=5,
    subplot_spec=gs[0, :],
    hspace=0.05,
    wspace=0.25,
    width_ratios=[1] * 4 + [0.1],
)


# =============================================================================
# Helpers
# =============================================================================


def _xy(t, i):
    x = iteration_last["grid"].nodes[(t, i)].get("xpos", None)
    y = iteration_last["grid"].nodes[(t, i)].get("ypos", None)
    if x is None or y is None:
        vv = np.asarray(iteration_last["grid"].nodes[(t, i)]["vertices"], dtype=float)
        x = float(np.mean(vv[:, 0]))
        y = float(np.mean(vv[:, 1]))
    return float(x), float(y)


def _add_base_grid(ax, polys):
    if len(polys) == 0:
        return
    pc = PolyCollection(
        polys,
        facecolors="w",
        edgecolors="k",
        linewidths=0.35,
        zorder=0,
    )
    ax.add_collection(pc)
    vv = np.vstack(polys)
    ax.set_xlim(vv[:, 0].min(), vv[:, 0].max())
    ax.set_ylim(vv[:, 1].min(), vv[:, 1].max())


def _plot_node_field(ax, iteration, t, varname, value_func, cmap, norm, missing_fc="w"):
    for nd in nodes:
        node = (t, nd)
        if node not in iteration["grid"].nodes:
            continue

        verts = np.asarray(iteration["grid"].nodes[node]["vertices"], dtype=float)

        if varname in iteration["grid"].nodes[node]:
            bnd = iteration["bounds"][str(iteration["grid"].nodes[node][varname])]
            val = value_func(bnd)
            fc = cmap(norm(val))
        else:
            fc = missing_fc

        ax.fill(verts[:, 0], verts[:, 1], fc=fc, ec="k")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")


# =============================================================================
# A : Setup
# =============================================================================

plt.subplot(subgs_top_row[0, 0])

nodes = []

observation_wells = [31,77,91]

for node in iteration_first["grid"].nodes:
    if node[0] == 0:
        nodes.append(node[1])

        
        if node[1] in iteration_first["fixed_head_cells"]:
            fc = "#666"
        elif node[1] == well_cell:
            fc = "xkcd:orangish red"
        elif node[1] in source_cells:
            fc = "xkcd:cerulean"
        else:
            fc = "None"
            
        verts = np.asarray(iteration_first["grid"].nodes[node]["vertices"])
            
        ec = "k"
        zorder = 10
        if node[1] in observation_wells:
            ec = "k"
            zorder = 15
            
            plt.scatter(
                np.mean(verts[:, 0]),
                np.mean(verts[:, 1]),
                color = "k",
                s = 15,
                marker = "o",
                zorder = 20)
            
            # for i in range(3):
            #     plt.plot(verts[i::3, 0], verts[i::3, 1], color=ec, zorder = zorder)

        

        plt.fill(verts[:, 0], verts[:, 1], fc=fc, ec=ec, zorder = zorder)

plt.title(r"$\mathbf{A}:$ Model overview", loc="left")
plt.gca().yaxis.set_ticks_position("right")
plt.ylabel("distance [m]", fontsize=12)
plt.xlabel("distance [m]", fontsize=12)
plt.axis("equal")

axA = plt.gca()



# Legend panel
plt.subplot(subgs_top_row[0, 1])

verts = verts - np.mean(verts, axis=0)
verts /= 10

plt.fill(verts[:, 0], verts[:, 1] + 4, fc="xkcd:cerulean", ec="k")
plt.text(1, 4, "source cell", ha="left", va="center")

plt.fill(verts[:, 0], verts[:, 1] + 2, fc="xkcd:orangish red", ec="k")
plt.text(1, 2, "extraction well", ha="left", va="center")

plt.fill(verts[:, 0], verts[:, 1] + 0, fc="#666", ec="k")
plt.text(1, 0, "fixed-head cell", ha="left", va="center")

plt.fill(verts[:, 0], verts[:, 1] - 2, fc="w", ec="k")
plt.scatter(0, -2, color="k", marker = "o", s = 25)
plt.text(1, -2, "observed cell", ha="left", va="center")

plt.gca().set_aspect("equal", adjustable="box")
plt.gca().set_xlim(-1.5, 7)
plt.axis("off")


# =============================================================================
# B : Transmissivity — midpoint + difference
# =============================================================================

t_plot = 0

T_log = [1e10, -1e10]
Td_lin = [1e10, -1e10]

polys = []
for nd in nodes:
    node = (t_plot, nd)
    if node not in iteration_last["grid"].nodes:
        continue
    polys.append(np.asarray(iteration_last["grid"].nodes[node]["vertices"], dtype=float))

X = []
Y = []
Tmid_log = []
Tdiff_lin = []
segments = []

for e in iteration_last["grid"].edges:
    (t1, j), (t2, i) = e

    if t1 != t2:
        continue
    if j not in nodes or i not in nodes:
        continue

    T_key = str(iteration_last["grid"].edges[e]["T"])
    bnd = iteration_last["bounds"][T_key]

    tmin = float(np.maximum(bnd[0], eps))
    tmax = float(np.maximum(bnd[1], eps))

    ltmin = float(np.log10(tmin))
    ltmax = float(np.log10(tmax))
    ltmid = 0.5 * (ltmin + ltmax)

    if ltmid <= T_log[0]:
        T_log[0] = ltmid
    if ltmid >= T_log[1]:
        T_log[1] = ltmid

    d = float(np.maximum(tmax - tmin, 0.0))

    if d <= Td_lin[0]:
        Td_lin[0] = d
    if d >= Td_lin[1]:
        Td_lin[1] = d

    if t1 == t_plot:
        xj, yj = _xy(t_plot, j)
        xi, yi = _xy(t_plot, i)

        X.append(0.5 * (xj + xi))
        Y.append(0.5 * (yj + yi))
        Tmid_log.append(ltmid)
        Tdiff_lin.append(d)
        segments.append([(xj, yj), (xi, yi)])

X = np.asarray(X, dtype=float)
Y = np.asarray(Y, dtype=float)
Tmid_log = np.asarray(Tmid_log, dtype=float)
Tdiff_lin = np.asarray(Tdiff_lin, dtype=float)

cmap = plt.get_cmap("turbo")

if T_log[1] <= T_log[0]:
    T_log[1] = T_log[0] + 1e-6
if Td_lin[1] <= Td_lin[0]:
    Td_lin[1] = Td_lin[0] + 1e-12

Td_log = np.log10(np.maximum(Td_lin, eps))

norm_T = Normalize(vmin=T_log[0], vmax=T_log[1])
norm_Tdlog = Normalize(vmin=Td_log[0], vmax=Td_log[1])

# Midpoint (log10)
plt.subplot(subgs_top_row[0, 2])

_add_base_grid(plt.gca(), polys)

plt.ylabel("midpoint $T$", fontsize=12)

lc = LineCollection(segments, cmap=cmap, norm=norm_T, linewidths=3)
lc.set_array(np.asarray(Tmid_log, dtype=float))
plt.gca().add_collection(lc)

plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.axis("equal")

plt.title(r"$\mathbf{B}:$ Transmissivity", loc="left")

cbaxes = inset_axes(
    plt.gca(),
    width="100%",
    height="7%",
    bbox_to_anchor=(0.05, -1.2, 1, 1),
    bbox_transform=plt.gca().transAxes,
)

cbar = plt.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm_T, cmap=cmap),
    cax=cbaxes,
    pad=0.05,
    fraction=1,
    orientation="horizontal",
)
cbar.set_label(
    r"$\log_{10} T$ [$\log_{10}$ m²/s]",
    rotation=0,
    labelpad=-50,
    fontsize=12,
)
cbar.ax.tick_params(labelsize=12)

# Difference (log10 of linear width)
plt.subplot(subgs_top_row[0, 3])

_add_base_grid(plt.gca(), polys)

plt.ylabel(r"interval width $\Delta T$", fontsize=12)


lc = LineCollection(segments, cmap=cmap, norm=norm_Tdlog, linewidths=3)
lc.set_array(np.asarray(np.log10(np.maximum(Tdiff_lin, eps)), dtype=float))
plt.gca().add_collection(lc)

plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.axis("equal")

cbaxes = inset_axes(
    plt.gca(),
    width="100%",
    height="7%",
    bbox_to_anchor=(0.05, -1.2, 1, 1),
    bbox_transform=plt.gca().transAxes,
)

cbar = plt.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm_Tdlog, cmap=cmap),
    cax=cbaxes,
    pad=0.05,
    fraction=1,
    orientation="horizontal",
)
cbar.set_label(
    r"$\log_{10} \Delta T$ [$\log_{10}$ m²/s]",
    rotation=0,
    labelpad=-50,
    fontsize=12,
)
cbar.ax.tick_params(labelsize=12)


# =============================================================================
# C : Recharge — midpoint + difference
# =============================================================================

R_mid = [1e10, -1e10]
R_diff = [1e10, -1e10]

for node in iteration_last["grid"].nodes:
    if "R" not in iteration_last["grid"].nodes[node]:
        continue

    bnd = iteration_last["bounds"][str(iteration_last["grid"].nodes[node]["R"])]
    rmid = 0.5 * (bnd[0] + bnd[1])
    rdiff = bnd[1] - bnd[0]

    if rmid <= R_mid[0]:
        R_mid[0] = rmid
    if rmid >= R_mid[1]:
        R_mid[1] = rmid
    if rdiff > 0.0 and rdiff <= R_diff[0]:
        R_diff[0] = rdiff
    if rdiff >= R_diff[1]:
        R_diff[1] = rdiff

R_abs = max(abs(R_mid[0]), abs(R_mid[1]), 1e-12)
R_linthresh = max(R_abs * 1e-3, eps)
if R_diff[1] <= 0.0:
    R_diff[0] = 1e-12
    R_diff[1] = 1e-11
elif R_diff[0] == 1e10:
    R_diff[0] = max(R_diff[1] * 1e-6, eps)
elif R_diff[1] <= R_diff[0]:
    R_diff[1] = R_diff[0] * 10.0

norm_R_mid = SymLogNorm(linthresh=R_linthresh, vmin=-R_abs, vmax=R_abs, base=10)
norm_R_diff = LogNorm(vmin=R_diff[0], vmax=R_diff[1])

cmap_R_mid = cmap_divergent
cmap_R_diff = plt.get_cmap("turbo").copy()
cmap_R_diff.set_bad("w")

subgs_R = GridSpecFromSubplotSpec(
    nrows=2,
    ncols=T + 1,
    subplot_spec=gs[1, :],
    hspace=0.05,
    wspace=0.25,
    width_ratios=[1] * T + [0.1],
)

for t in range(T):
    # Row 0 : midpoint
    plt.subplot(subgs_R[0, t])

    _plot_node_field(
        plt.gca(),
        iteration_last,
        t,
        "R",
        lambda bnd: 0.5 * (bnd[0] + bnd[1]),
        cmap_R_mid,
        norm_R_mid,
        missing_fc="w",
    )

    if t == 0:
        plt.ylabel(r"midpoint $R$", fontsize=12)
        plt.title(r"$\mathbf{C}:$ Recharge", loc="left")

    # Row 1 : interval width
    plt.subplot(subgs_R[1, t])

    _plot_node_field(
        plt.gca(),
        iteration_last,
        t,
        "R",
        lambda bnd: np.ma.masked_less_equal(bnd[1] - bnd[0], 0.0),
        cmap_R_diff,
        norm_R_diff,
        missing_fc="w",
    )

    if t == 0:
        plt.ylabel(r"interval width $\Delta R$", fontsize=12)

    plt.xlabel(f"timestep {t + 1}", fontsize=12)

# Colorbar : midpoint
plt.subplot(subgs_R[:1, -1])

sm = matplotlib.cm.ScalarMappable(norm=norm_R_mid, cmap=cmap_R_mid)
sm.set_array([])
cbar = plt.colorbar(sm, cax=plt.gca(), orientation="vertical")
cbar.set_label("$R$ [m/s]", fontsize=12, labelpad = -75)
cbar.ax.tick_params(labelsize=12)

# Colorbar : difference
plt.subplot(subgs_R[1:, -1])

sm = matplotlib.cm.ScalarMappable(norm=norm_R_diff, cmap=cmap_R_diff)
sm.set_array([])
cbar = plt.colorbar(sm, cax=plt.gca(), orientation="vertical")
cbar.set_label(r"$\Delta R$ [m/s]", fontsize=12, labelpad = -65)
cbar.ax.tick_params(labelsize=12)


# =============================================================================
# D : Hydraulic head — midpoint + difference
# =============================================================================

h_mid = [1e10, -1e10]
h_diff = [1e10, -1e10]

cmap = plt.get_cmap("turbo")

for node in iteration_last["grid"].nodes:
    h_val = iteration_last["bounds"][str(iteration_last["grid"].nodes[node]["h"])]
    hmid = 0.5 * (h_val[0] + h_val[1])
    hdiff = h_val[1] - h_val[0]
    if hmid <= h_mid[0]:
        h_mid[0] = hmid
    if hmid >= h_mid[1]:
        h_mid[1] = hmid
    if hdiff <= h_diff[0]:
        h_diff[0] = hdiff
    if hdiff >= h_diff[1]:
        h_diff[1] = hdiff

subgs_head = GridSpecFromSubplotSpec(
    nrows=2,
    ncols=T + 1,
    subplot_spec=gs[2, :],
    hspace=0.05,
    wspace=0.25,
    width_ratios=[1] * T + [0.1],
)

for t in range(T):
    # Row 0 : midpoint
    plt.subplot(subgs_head[0, t])

    mid_vert_vals = []
    diff_vert_vals = []

    for nd in nodes:
        node = (t, nd)
        verts = np.asarray(iteration_last["grid"].nodes[node]["vertices"])
        bnd = iteration_last["bounds"][str(iteration_last["grid"].nodes[node]["h"])]

        hmid = 0.5 * (bnd[0] + bnd[1])
        hdiff = bnd[1] - bnd[0]

        mid_vert_vals.append(hmid)
        diff_vert_vals.append(hdiff)

        plt.fill(
            verts[:, 0],
            verts[:, 1],
            fc=cmap((hmid - h_mid[0]) / (h_mid[1] - h_mid[0])),
            ec="k",
        )

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("equal")

    if t == 0:
        plt.ylabel(r"midpoint $h$", fontsize=12)
        plt.title(r"$\mathbf{D}:$ Hydraulic head", loc="left")

    # Row 1 : difference
    plt.subplot(subgs_head[1, t])

    for idx, nd in enumerate(nodes):
        node = (t, nd)
        verts = np.asarray(iteration_last["grid"].nodes[node]["vertices"])

        plt.fill(
            verts[:, 0],
            verts[:, 1],
            fc=cmap((diff_vert_vals[idx] - h_diff[0]) / (h_diff[1] - h_diff[0])),
            ec="k",
        )

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("equal")

    if t == 0:
        plt.ylabel(r"interval width $\Delta h$", fontsize=12)

    plt.xlabel(f"timestep {t + 1}", fontsize=12)

plt.gca().tick_params(axis="both", which="major", labelsize=12)

# Colorbar : midpoint
plt.subplot(subgs_head[:1, -1])

norm = Normalize(vmin=h_mid[0], vmax=h_mid[1])
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="turbo")
sm.set_array([])
cbar = plt.colorbar(sm, cax=plt.gca(), orientation="vertical")
cbar.set_label("$h$ [m]", fontsize=12, labelpad = -45)
stepsize = 4
cbar.set_ticks(
    list(
        range(
            stepsize * math.ceil(h_mid[0] / stepsize),
            stepsize * math.floor(h_mid[1] / stepsize) + 1,
            stepsize,
        )
    )
)
cbar.ax.tick_params(labelsize=12)

# Colorbar : difference
plt.subplot(subgs_head[1:, -1])

norm = Normalize(vmin=h_diff[0], vmax=h_diff[1])
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="turbo")
sm.set_array([])
cbar = plt.colorbar(sm, cax=plt.gca(), orientation="vertical")
cbar.set_label(r"$\Delta h$ [m]", fontsize=12, labelpad = -45)
stepsize = 4
cbar.set_ticks(
    list(
        range(
            stepsize * math.ceil(h_diff[0] / stepsize),
            stepsize * math.floor(h_diff[1] / stepsize) + 1,
            stepsize,
        )
    )
)
cbar.ax.tick_params(labelsize=12)


# =============================================================================
# E : tightening
# =============================================================================

subgs_iter = GridSpecFromSubplotSpec(
    nrows=1,
    ncols=6,
    subplot_spec=gs[3, :],
    hspace=0.05,
    wspace=0.25,
    width_ratios=[1] * 5 + [0.1],
)

variable_width_initial = {}
num_vars = 0

for key in list(iteration_first["bounds"].keys()):
    variable_width_initial[key] = iteration_first["bounds"][key][1] - iteration_first["bounds"][key][0]

    if variable_width_initial[key] != 0:
        num_vars += 1

variables = ["h", "R", "T", "Sy", "dhx", "qx", "dht", "qt"]
avg_log_fraction_per_variable = {key: [] for key in variables}
iteration_plot = [0]

log_fraction_per_variable = {key: [] for key in variables}
for key in list(iteration_first["bounds"].keys()):
    var_identifier = key.split("_")[0]

    if variable_width_initial[key] != 0 and var_identifier in variables:
        log_fraction_per_variable[var_identifier].append(1.0)

for var in variables:
    if len(log_fraction_per_variable[var]) > 0:
        avg_log_fraction_per_variable[var].append(
            np.log10(np.maximum(np.mean(log_fraction_per_variable[var]), eps))
        )
    else:
        avg_log_fraction_per_variable[var].append(np.nan)

for idx, iter_file in enumerate(iteration_files[:-1]):
    iteration = pickle.load(open(os.path.join(checkpoint_dir, iter_file), "rb"))

    log_fraction_per_variable = {key: [] for key in variables}

    variable_width = {}
    for key in list(iteration["bounds"].keys()):
        variable_width[key] = iteration["bounds"][key][1] - iteration["bounds"][key][0]

    for key in list(iteration["bounds"].keys()):
        var_identifier = key.split("_")[0]

        if (
            key in variable_width_initial
            and variable_width_initial[key] != 0
            and var_identifier in variables
        ):
            log_fraction_per_variable[var_identifier].append(
                variable_width[key] / variable_width_initial[key]
            )

    for var in variables:
        if len(log_fraction_per_variable[var]) > 0:
            avg_log_fraction_per_variable[var].append(
                np.log10(np.maximum(np.mean(log_fraction_per_variable[var]), eps))
            )
        else:
            avg_log_fraction_per_variable[var].append(np.nan)

    iteration_plot.append(idx + 1)

plt.subplot(subgs_iter[0, :-1])
plot_colors = {
    "h": "xkcd:orangish red",
    "T": "xkcd:cerulean",
    "R": "xkcd:grass green",
    "Sy": "xkcd:royal blue",
    "dhx": "xkcd:crimson",
    "qx": "xkcd:goldenrod",
    "dht": "xkcd:orange",
    "qt": "xkcd:puce",
}
markers = {
    "h": "+",
    "T": "o",
    "R": "x",
    "Sy": "d",
    "dhx": "1",
    "qx": "2",
    "dht": "3",
    "qt": "4",
}
for var in variables:
    plt.plot(
        iteration_plot,
        avg_log_fraction_per_variable[var],
        label=var,
        color=plot_colors[var],
        marker= markers[var]
    )

ax = plt.gca()
ax.set_xticks([0, 1, 2, 3, 4, 5, 10, 15, 20, 25])
ax.grid(axis="y", which="major", color="#999", linewidth=0.8)
ax.grid(axis="y", which="minor", color="#999", linewidth=0.5)
plt.legend(ncols=4, frameon=True, loc="lower left")

ax.yaxis.set_ticks_position("right")
plt.ylabel(r"avg. $\log_{10}$ fraction"+"\n"+"of original width", fontsize=12)
plt.xlabel("iteration", fontsize=12)
plt.title(r"$\mathbf{E}:$ Average variable tightening", loc="left")


# =============================================================================
# Finish
# =============================================================================

fig.subplots_adjust(
    left=0.05,
    right=0.925,
    bottom=0.075,
    top=0.945,
)

simulation_time = iteration_last["current time"] - iteration_last["start time"]
num_obbt_iterations = max(len(iteration_files[:-1]), 1)

print("The simulations took " + str(simulation_time / 60) + " minutes.")
print("The simulations took " + str(simulation_time / 60 / 60) + " hours.")
print("Each tightening operation took " + str(simulation_time / (num_vars * num_obbt_iterations * 2)) + " seconds.")

plt.savefig("experiment_2D_transient.pdf", dpi=300)
plt.savefig("experiment_2D_transient.png", dpi=300)
plt.show()
