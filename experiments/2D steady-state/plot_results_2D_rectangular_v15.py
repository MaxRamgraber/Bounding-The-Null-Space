import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec


# =============================================================================
# Load states
# =============================================================================

def last_iter(directory):
    files = sorted(
        f for f in os.listdir(directory)
        if f.startswith("iteration") and f.endswith(".p") and f != "iteration_initial.p"
    )
    if not files:
        raise FileNotFoundError(f"No iteration files found in '{directory}'.")
    return os.path.join(directory, files[-1])


iter_prior = pickle.load(open(os.path.join("checkpoints_basic", "iteration_initial.p"), "rb"))
iter_no_flow = pickle.load(open(last_iter("checkpoints_basic"), "rb"))
iter_flow = pickle.load(open(last_iter("checkpoints_flow_sign"), "rb"))
iter_no_curl = pickle.load(open(last_iter("checkpoints_nocurl"), "rb"))
iter_ref = {
    "bounds": pickle.load(open("pure_LP_bounds.p", "rb")),
    "grid": iter_no_curl["grid"],
    "timesteps": iter_no_curl["timesteps"],
}

SCENARIOS = [
    (r"$\mathbf{A}$: initial bounds"+ "\n", iter_prior),
    (r"$\mathbf{B}$: reference" + "\n" + r"(pure LP)", iter_ref),
    (r"$\mathbf{C}$: tightened bounds" + "\n" + r"(basic)", iter_no_flow),
    (r"$\mathbf{D}$: tightened bounds" + "\n" + r"(irrotational flow)", iter_no_curl),
    (r"$\mathbf{E}$: tightened bounds" + "\n" + r"(flow signs)", iter_flow),
]


t = 0
nodes = sorted({nd for (tt, nd) in iter_no_curl["grid"].nodes if tt == t})


# =============================================================================
# Styling
# =============================================================================

H_CMAP = plt.get_cmap("turbo")
C_INIT = "xkcd:silver"
C_TIGHT = "xkcd:cerulean"
LW_INIT = 3.0
LW_TIGHT = 2.6
CAP = 0.11
R_LINTHRESH = 1e-5


# =============================================================================
# Helpers
# =============================================================================

def bnd(state, sym):
    lo, hi = state["bounds"][sym]
    return float(lo), float(hi)


def cell_centre(state, nd):
    verts = np.asarray(state["grid"].nodes[(t, nd)]["vertices"], dtype=float)
    return float(verts[:, 0].mean()), float(verts[:, 1].mean())


def h_mid_width(state, nd):
    sym = str(state["grid"].nodes[(t, nd)]["h"])
    lo, hi = bnd(state, sym)
    return 0.5 * (lo + hi), hi - lo


def has_flow_signs(state):
    for e in state["grid"].edges:
        if e[0][0] != t or e[1][0] != t:
            continue
        sign = state["grid"].edges[e].get("flow sign")
        if sign not in (None, 0):
            return True
    return False


def draw_flow(ax, state):
    qx, qy, qu, qv = [], [], [], []

    for e in state["grid"].edges:
        (t1, j), (t2, i) = e
        if t1 != t2 or t1 != t or j not in nodes or i not in nodes:
            continue

        sign = state["grid"].edges[e].get("flow sign")
        if sign in (None, 0):
            continue

        xj, yj = cell_centre(state, j)
        xi, yi = cell_centre(state, i)

        if sign > 0:
            sx, sy, dx, dy = xj, yj, xi, yi
        else:
            sx, sy, dx, dy = xi, yi, xj, yj

        qx.append(sx + 0.30 * (dx - sx))
        qy.append(sy + 0.30 * (dy - sy))
        qu.append(0.40 * (dx - sx))
        qv.append(0.40 * (dy - sy))

    if qx:
        ax.quiver(
            qx, qy, qu, qv,
            angles="xy", scale_units="xy", scale=1,
            color="k", width=0.004,
            headwidth=4, headlength=4, headaxislength=3.5,
            zorder=4,
        )


def draw_h_map(ax, state, norm, mode, map_xlim, map_ylim):
    polys = []
    vals = []

    for nd in nodes:
        verts = np.asarray(state["grid"].nodes[(t, nd)]["vertices"], dtype=float)
        h_mid, h_wid = h_mid_width(state, nd)
        polys.append(verts)
        vals.append(h_mid if mode == "mid" else h_wid)

    pc = PolyCollection(
        polys,
        facecolors=[H_CMAP(norm(v)) for v in vals],
        edgecolors="k",
        linewidths=0.35,
        zorder=1,
    )
    ax.add_collection(pc)
    ax.set_xlim(map_xlim)
    ax.set_ylim(map_ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_box_aspect(1)


def draw_interval(ax, x, bounds_now, color, lw):
    lo, hi = bounds_now
    ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.plot([x - CAP, x + CAP], [lo, lo], color=color, lw=lw, zorder=3)
    ax.plot([x - CAP, x + CAP], [hi, hi], color=color, lw=lw, zorder=3)


def draw_interval_panel(ax, items, state, scale="linear", zero_line=False, show_tight=True):
    x = np.arange(len(items), dtype=float) + 1.0

    for xi, item in zip(x, items):
        draw_interval(ax, xi - 0.16, bnd(iter_prior, item["sym"]), C_INIT, LW_INIT)
        if show_tight:
            draw_interval(ax, xi + 0.16, bnd(state, item["sym"]), C_TIGHT, LW_TIGHT)

    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in items], fontsize=9)
    ax.grid(axis="y", lw=0.4, color="#ccc", zorder=0)

    if scale == "log":
        ax.set_yscale("log")
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=R_LINTHRESH)

    if zero_line:
        ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=1)

    ax.tick_params(axis="y", labelsize=8)
    ax.set_box_aspect(1)


# =============================================================================
# Global ranges
# =============================================================================

all_verts = np.vstack([
    np.asarray(iter_prior["grid"].nodes[(t, nd)]["vertices"], dtype=float)
    for nd in nodes
])
map_xlim = (float(all_verts[:, 0].min()), float(all_verts[:, 0].max()))
map_ylim = (float(all_verts[:, 1].min()), float(all_verts[:, 1].max()))

h_mid_vals = []
h_wid_vals = []
for _, state in SCENARIOS:
    for nd in nodes:
        hm, hw = h_mid_width(state, nd)
        h_mid_vals.append(hm)
        h_wid_vals.append(hw)

norm_h_mid = mcolors.Normalize(vmin=min(h_mid_vals), vmax=max(h_mid_vals))
norm_h_wid = mcolors.Normalize(vmin=0, vmax=max(h_wid_vals))


# =============================================================================
# T items
# =============================================================================

T_items = []
seen_T = {}

for e in sorted(iter_prior["grid"].edges, key=lambda ee: (min(ee[0][1], ee[1][1]), max(ee[0][1], ee[1][1]))):
    (t1, j), (t2, i) = e
    if t1 != t2 or t1 != t or j not in nodes or i not in nodes:
        continue

    sym = str(iter_prior["grid"].edges[e]["T"])
    if sym not in seen_T:
        seen_T[sym] = {"sym": sym, "edges": []}
        T_items.append(seen_T[sym])
    seen_T[sym]["edges"].append((j, i))

if len(T_items) == 1:
    T_items[0]["label"] = "all edges"
else:
    for item in T_items:
        j, i = item["edges"][0]
        item["label"] = str(j + 1) + "↔" + str(i + 1)

T_vals = []
for _, state in SCENARIOS:
    for item in T_items:
        T_vals.extend(bnd(state, item["sym"]))

T_lo = max(min(T_vals), 1e-12)
T_hi = max(T_vals)
T_ylim = (T_lo / (10 ** 0.08), T_hi * (10 ** 0.08))


# =============================================================================
# R items
# =============================================================================

seen_R = {}
for nd in nodes:
    sym = str(iter_prior["grid"].nodes[(t, nd)]["R"])
    if sym not in seen_R:
        seen_R[sym] = {"sym": sym, "nodes": []}
    seen_R[sym]["nodes"].append(nd)

R_items = []
for item in seen_R.values():
    mids = []
    active = False

    for _, state in SCENARIOS:
        if item["sym"] not in state["bounds"]:
            continue
        lo, hi = bnd(state, item["sym"])
        mids.append(0.5 * (lo + hi))
        if abs(lo) > 1e-14 or abs(hi) > 1e-14:
            active = True

    if active:
        item["sign"] = 1 if np.mean(mids) >= 0 else -1
        R_items.append(item)

R_items = sorted(R_items, key=lambda item: (0 if item["sign"] > 0 else 1, min(item["nodes"])))

npos = sum(item["sign"] > 0 for item in R_items)
nneg = sum(item["sign"] < 0 for item in R_items)
ipos = 0
ineg = 0
for item in R_items:
    if item["sign"] > 0:
        ipos += 1
        item["label"] = "source" if npos == 1 else "source " + str(ipos)
    else:
        ineg += 1
        item["label"] = "sink" if nneg == 1 else "sink " + str(ineg)

R_vals = []
for _, state in SCENARIOS:
    for item in R_items:
        R_vals.extend(bnd(state, item["sym"]))

R_abs = max([abs(v) for v in R_vals] + [R_LINTHRESH])
R_ylim = (-1.25 * R_abs, 1.25 * R_abs)


# =============================================================================
# Figure layout
# =============================================================================

fig = plt.figure(figsize=(17, 13))
outer = GridSpec(
    nrows=2,
    ncols=1,
    figure=fig,
    height_ratios=[0.95, 4.2],
    hspace=0.34,
)

gs_top = outer[0].subgridspec(1, 5, wspace=0.35)
gs_bot = outer[1].subgridspec(4, 5, wspace=0.22, hspace=0.16)


# =============================================================================
# Top row
# =============================================================================

# Geometry sketch
ax = fig.add_subplot(gs_top[0, 0])
for nd in nodes:
    key = (t, nd)
    cx, cy = cell_centre(iter_prior, nd)
    verts = np.asarray(iter_prior["grid"].nodes[key]["vertices"], dtype=float)

    if key == (t, 12):
        fc = "#666"
    elif key == (t, 0):
        fc = "xkcd:cerulean"
    elif key == (t, 24):
        fc = "xkcd:orangish red"
    else:
        fc = "None"

    ax.fill(verts[:, 0], verts[:, 1], fc=fc, ec="k")
    ax.text(cx, cy, str(nd + 1), ha="center", va="center", fontsize=7)

ax.set_aspect("equal")
ax.set_box_aspect(1)
ax.axis("off")
ax.set_title("Legend", fontsize=12, pad=6)

# Well legend
ax = fig.add_subplot(gs_top[0, 1])
sq = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * 0.6
for k, (fc, label) in enumerate([
    ("xkcd:cerulean", "injection well"),
    ("xkcd:orangish red", "extraction well"),
    ("#666", "observation well"),
]):
    y = 2 - 2 * k
    ax.fill(sq[:, 0], sq[:, 1] + y, fc=fc, ec="k")
    ax.text(0.9, y, label, ha="left", va="center", fontsize=12)

ax.set_xlim(-1.5, 6.8)
ax.set_ylim(-3.0, 3.5)
ax.axis("off")

# Interval legend
ax = fig.add_subplot(gs_top[0, 2])
draw_interval(ax, 0.95, (0.20, 0.80), C_INIT, LW_INIT)
draw_interval(ax, 1.35, (0.25, 0.75), C_TIGHT, LW_TIGHT)
ax.text(1.70, 0.6, "initial bounds", color="#999", va="center", fontsize=12)
ax.text(1.70, 0.45, "tightened bounds", color=C_TIGHT, va="center", fontsize=12)
ax.set_xlim(0.45, 4.8)
ax.set_ylim(0.0, 1.0)
ax.axis("off")

# h midpoint colorbar
ax = fig.add_subplot(gs_top[0, 3])
ax.axis("off")
cax = ax.inset_axes([0.06, 0.40, 0.88, 0.20])
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm_h_mid, cmap=H_CMAP),
    cax=cax,
    orientation="horizontal",
)
cb.set_label(r"$h$ midpoint [m]", fontsize=12, labelpad=2)
cb.ax.tick_params(labelsize=10)

# h width colorbar
ax = fig.add_subplot(gs_top[0, 4])
ax.axis("off")
cax = ax.inset_axes([0.06, 0.40, 0.88, 0.20])
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm_h_wid, cmap=H_CMAP),
    cax=cax,
    orientation="horizontal",
)
cb.set_label(r"$h$ width [m]", fontsize=12, labelpad=2)
cb.ax.tick_params(labelsize=10)


# =============================================================================
# Scenario grid
# =============================================================================

for col, (title, state) in enumerate(SCENARIOS):

    ax = fig.add_subplot(gs_bot[0, col])
    draw_h_map(ax, state, norm_h_mid, "mid", map_xlim, map_ylim)
    ax.set_title(title, fontsize=12, loc = "left", pad=4)
    if col == 0:
        ax.set_ylabel(r"$h$ midpoint", fontsize=12)
    if has_flow_signs(state):
        draw_flow(ax, state)

    ax = fig.add_subplot(gs_bot[1, col])
    draw_h_map(ax, state, norm_h_wid, "wid", map_xlim, map_ylim)
    if col == 0:
        ax.set_ylabel(r"$h$ width", fontsize=12)
    if has_flow_signs(state):
        draw_flow(ax, state)

    ax = fig.add_subplot(gs_bot[2, col])
    draw_interval_panel(ax, T_items, state, scale="log", show_tight=(col != 0))
    ax.set_ylim(T_ylim)
    if col == 0:
        ax.set_ylabel(r"$T$ [m$^2$/s]", fontsize=12)

    ax = fig.add_subplot(gs_bot[3, col])
    draw_interval_panel(ax, R_items, state, scale="symlog", zero_line=True, show_tight=(col != 0))
    ax.set_ylim(R_ylim)
    if col == 0:
        ax.set_ylabel(r"$R$ [m/s]", fontsize=12)


plt.savefig("experiment_2D_steady_state.pdf", dpi=300, bbox_inches="tight")
plt.savefig("experiment_2D_steady_state.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved experiment_2D_by_scenario_v12.pdf / .png")
