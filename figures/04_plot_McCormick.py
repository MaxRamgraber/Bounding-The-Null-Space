import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import matplotlib
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import ConnectionPatch

matplotlib.rcParams["text.usetex"] = False

root_directory = os.path.dirname(os.path.realpath(__file__))

np.random.seed(0)

#%%

# Create an arrow
arrow_x     = np.asarray([1,1,2,0,-2,-1,-1])
arrow_y     = np.asarray([-2,0,0,2.5,0,0,-2])

arrow       = np.column_stack((
    arrow_x,
    arrow_y ))

import math

def rotate_around_origin(points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    
    newpoints   = np.zeros(points.shape)

    newpoints[:,0] = math.cos(angle) * points[:,0] - math.sin(angle) * points[:,1]
    newpoints[:,1] = math.sin(angle) * points[:,0] + math.cos(angle) * points[:,1]
    return newpoints

def add_arrow(line, position=None, direction='right', size=25, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    lastvec = [np.diff(xdata[-2:]),np.diff(ydata[-2:])]

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind] + lastvec[0]*0.05, ydata[end_ind] + lastvec[1]*0.05),
        arrowprops=dict(arrowstyle="->", color=color, linewidth = 2),
        size=size
    )

arrow   = rotate_around_origin(arrow,-np.pi*0.75)*0.5

wgts = np.linspace(0,1,61)#[:-1]

def polygon_from_inequalities(A, b, tol=1e-9):
    
    def intersect_lines(a1, b1, a2, b2, tol=1e-9):
        """
        Solve:
          a1[0]*x + a1[1]*y = b1
          a2[0]*x + a2[1]*y = b2
        Returns (x, y) or None if lines are parallel.
        """
        A_mat = np.vstack([a1, a2])   # shape (2,2)
        if abs(np.linalg.det(A_mat)) < tol:
            return None  # parallel or nearly parallel
        x = np.linalg.solve(A_mat, np.array([b1, b2]))
        return x
    
    m = A.shape[0]
    points = []

    # 1) all pairwise intersections
    for i in range(m):
        for j in range(i+1, m):
            p = intersect_lines(A[i], b[i], A[j], b[j], tol=tol)
            if p is None:
                continue
            # 2) check feasibility: A p <= b + tol
            if np.all(A @ p <= b + tol):
                points.append(p)

    if not points:
        return None  # empty feasible set or unbounded (no finite polygon)

    pts = np.unique(np.array(points), axis=0)  # remove duplicates

    # 3) sort points by angle around centroid to form polygon
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    order = np.argsort(angles)
    return pts[order]




def build_Ab(a_bounds, b_bounds, c_bounds):
    a_min, a_max = a_bounds
    b_min, b_max = b_bounds
    c_min, c_max = c_bounds

    A = np.array([
        [ 1,  0],   # a <= a_max
        [-1,  0],   # -a <= -a_min
        [ 0,  1],   # b <= b_max
        [ 0, -1],   # -b <= -b_min
        [ 1,  1],   # a + b <= c_max
        [-1, -1],   # -(a + b) <= -c_min
    ], dtype=float)

    b = np.array([
        a_max,
        -a_min,
        b_max,
        -b_min,
        c_max,
        -c_min,
    ], dtype=float)

    return A, b


bounds_T = [1E-2,1E-1]
bounds_dhx = [-5,5]
bounds_qx = [-0.5,0.5]

cmap = plt.get_cmap("turbo")


wgts = [1]

plt.close("all")

fig = plt.figure(figsize=(12,7))

gs = GridSpec(
    nrows = 2,
    ncols = 2,
    wspace = 0.,
    width_ratios = [1.25,1],
    height_ratios = [1,0.2],
    hspace = 0.)


ax_legend = plt.subplot(gs[1,:])

ax_legend.set_xlim(0,1)
ax_legend.set_ylim(0,1)

ax_legend.fill(
    [0.6,0.7,0.7,0.6],
    [0,0,0.25,0.25],
    facecolor='none',
    edgecolor='xkcd:orangish red',
    linewidth=1,
    hatch=r"\\\\",
    alpha = 0.4,
    zorder = -2)

ax_legend.text(
    0.65,
    0.5,
    "non-physical"+"\n"+"regions",
    ha = "center",
    va = "center")


ax_legend.fill(
    [0.2,0.3,0.3,0.2],
    [0,0,0.25,0.25],
    alpha=0.3,
    color="xkcd:grey",
    zorder = -2)

ax_legend.text(
    0.25,
    0.5,
    "McCormick"+"\n"+"envelope",
    ha = "center",
    va = "center")

ax_legend.axis("off")


ax2d = plt.subplot(gs[0,1])

plt.ylim(-5.,5.)
plt.xlim(-0.5,0.5)

plt.xlabel(r"flow $q_{j \rightarrow i}$ [m³/s]",loc="right", labelpad = -35)
plt.ylabel(r"head gradient dhx$_{j,i}$ [-]",loc="top", labelpad = 0)

plt.gca().spines["left"].set_position("zero")
plt.gca().spines["bottom"].set_position("zero")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

ticks = [-5,-4,-3,-2,-1,1,2,3,4,5]

plt.gca().set_xticks(np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]))
plt.gca().set_yticks([-5,-4,-3,-2,-1,1,2,3,4,5])




xlims = plt.gca().get_xlim()
ylims = plt.gca().get_ylim()
plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)



for idx,T in enumerate(np.linspace(np.log10(bounds_T[0]),np.log10(bounds_T[1]),201)):
    
    plt.plot(
        bounds_qx,
        np.asarray(bounds_qx)/10**T,
        color = cmap(idx/200),
        zorder = -100)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib




danger_zone_1 = np.asarray([
    [0,0],
    [-0.5,-5],
    [0.05,5]])

plt.fill(
    danger_zone_1[:,0],
    danger_zone_1[:,1],
    fc = "xkcd:grey",
    ec = "None",
    zorder = -1,
    alpha = 0.5)

danger_zone_2 = np.asarray([
    [0,0],
    [0.5,5],
    [-0.05,-5]])

plt.fill(
    danger_zone_2[:,0],
    danger_zone_2[:,1],
    fc = "xkcd:grey",
    ec = "None",
    zorder = -1,
    alpha = 0.5)


plt.fill(
    [-0.5,0,0,-0.5],
    [0,0,5,5],
    facecolor='none',
    edgecolor='xkcd:orangish red',
    linewidth=1,
    hatch=r"\\\\",
    alpha = 0.4,
    zorder = -2)

plt.fill(
    [0.5,0,0,0.5],
    [0,0,-5,-5],
    facecolor='none',
    edgecolor='xkcd:orangish red',
    linewidth=1,
    hatch=r"\\\\",
    alpha = 0.4,
    zorder = -2)


plt.fill(
    [-0.5,0.5,0.5,-0.5],
    [-5,-5,5,5],
    facecolor='none',
    edgecolor='#999',
    linewidth=3,
    ls = "--")






plt.title(r"$\mathbf{B}$: 2D projection along $T_{j,i}$ axis",loc="left")

cbaxes = inset_axes(plt.gca(), width="30%", height="3%", bbox_to_anchor=(-0.1,-1.15,1,1), bbox_transform=plt.gca().transAxes) 

norm = matplotlib.colors.Normalize(vmin=np.log10(bounds_T[0]), vmax=np.log10(bounds_T[1]))
cbar = plt.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
    cax = cbaxes, 
    pad=.05, 
    fraction=1,
    orientation="horizontal")
cbar.set_label(r'transmissivity $T_{j,i}$'+"\n"+r'[$\log_{10}$ m²/s]', rotation=0, labelpad = -60)





# ---------------------------------------------------------------------
# Bounds (you can adjust these to match your problem)
# ---------------------------------------------------------------------
bounds_T   = [1e-2, 1e-1]   # T_min, T_max
bounds_dhx = [-5.0, 5.0]    # dhx_min, dhx_max

T_min, T_max   = bounds_T
dhx_min, dhx_max = bounds_dhx

# Grids in T and dhx
nT, nH = 120, 120
T = np.linspace(T_min, T_max, nT)
H = np.linspace(dhx_min, dhx_max, nH)  # H = dhx

Tg, Hg = np.meshgrid(T, H)

# ---------------------------------------------------------------------
# Bilinear manifold: Qx = T * dhx
# ---------------------------------------------------------------------
Q_true = Tg * Hg

# ---------------------------------------------------------------------
# McCormick relaxation for Qx = T * dhx on
#   T ∈ [T_min, T_max], dhx ∈ [dhx_min, dhx_max]
#
# Here:
#   x = T, y = dhx, w = Qx
# McCormick planes:
#   w >= xL*y + yL*x - xL*yL
#   w >= xU*y + yU*x - xU*yU
#   w <= xL*y + yU*x - xL*yU
#   w <= xU*y + yL*x - xU*yL
# ---------------------------------------------------------------------
xL, xU = T_min, T_max
yL, yU = dhx_min, dhx_max

Q1 = xL * Hg + yL * Tg - xL * yL
Q2 = xU * Hg + yU * Tg - xU * yU
Q3 = xL * Hg + yU * Tg - xL * yU
Q4 = xU * Hg + yL * Tg - xU * yL

# Minimal hull surfaces (only envelope, no redundant planes)
Q_low  = np.maximum(Q1, Q2)   # lower envelope
Q_high = np.minimum(Q3, Q4)   # upper envelope

# ---------------------------------------------------------------------
# Plot: X = T, Y = Qx, Z = dhx
# ---------------------------------------------------------------------

# wgts = np.linspace(0,1,61)[:-1]

#%%


# plt.subplot(gs[1])

ax3d = plt.gcf().add_subplot(gs[0, 0], projection="3d")

# McCormick hull – transparent cerulean
ax3d.plot_surface(
    Tg, Q_low, Hg,
    rstride=1, cstride=1,
    alpha=0.3,
    color="xkcd:grey",
    edgecolor="none"
)
ax3d.plot_surface(
    Tg, Q_high, Hg,
    rstride=1, cstride=1,
    alpha=0.3,
    color="xkcd:grey",
    edgecolor="none"
)





norm = Normalize(vmin=Tg[:, 0].min(), vmax=Tg[:, 0].max())

ax3d.plot_surface(
    Tg, Q_true, Hg,
    alpha=0.15,
    color="lightgray",
    edgecolor="none"
)

for idx in range(Tg.shape[1]):
    T = Tg[0, idx]
    ax3d.plot(
        Tg[:, idx],
        Q_true[:, idx],
        Hg[:, idx],
        color=cmap(idx/(Tg.shape[0]-1)),
        linewidth=1.5
    )


ax3d.plot(
    [0.1,0.1,0.1,0.1,0.1],
    [-0.5,0.5,0.5,-0.5,-0.5],
    [-5,-5,5,5,-5],
    color = "#999",
    ls = "--"
    )





# ax.plot_surface(
#     Tg, Q_true, Hg,
#     rstride=1, cstride=1,
#     alpha=0.4,
#     color="xkcd:orangish red",
#     edgecolor="none"
# )

# ---------------------------------------------------------------------
# Axes labels & view
# ---------------------------------------------------------------------


ax3d.set_xlabel(r"transmissivity $T_{j,i}$"+"\n"+"[m$^2$/s]",labelpad=10)
ax3d.set_ylabel(r"flow $q_{j \rightarrow i}$ [m³/s]")
ax3d.set_zlabel(r"head gradient dhx$_{j,i}$ [-]")

ax3d.set_xlim(T_min, T_max)
ax3d.set_ylim(-0.5, 0.5)
ax3d.set_zlim(dhx_min, dhx_max)


ax3d.set_xticks([0.01,0.05,0.1])
ax3d.set_yticks(np.array([-0.4,-0.2,-0,0.2,0.4]))
ax3d.set_zticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])

# ax.set_title("Minimal McCormick Hull (cerulean) enclosing Qx = T · dhx (orangish red)")
ax3d.view_init(elev=12.5, azim=15)
# ax.view_init(elev=0, azim=0)

ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.set_facecolor("none")

ax3d.set_title(r"$\mathbf{A}$: Null manifold and McCormick envelope for Darcy flow",loc="left")

from mpl_toolkits.mplot3d import proj3d
x2, y2, _ = proj3d.proj_transform(0.1,-0.5,-5, ax3d.get_proj())
con = ConnectionPatch(
    xyA=(x2, y2), 
    coordsA=ax3d.transData,   # projected point from 3D axes
    xyB=(-0.5,-5),  
    coordsB=ax2d.transData,   # point in 2D axes coords
    arrowstyle="-",
    shrinkA=0,
    shrinkB=0,
    mutation_scale=15,
    lw=1.5,
    color="#999",
    ls = "--",
    zorder = -1000
)
fig.add_artist(con)


x2, y2, _ = proj3d.proj_transform(0.1,-0.5,5, ax3d.get_proj())
con2 = ConnectionPatch(
    xyA=(x2, y2), 
    coordsA=ax3d.transData,   # projected point from 3D axes
    xyB=(-0.5,5),  
    coordsB=ax2d.transData,   # point in 2D axes coords
    arrowstyle="-",
    shrinkA=0,
    shrinkB=0,
    mutation_scale=15,
    lw=1.5,
    color="#999",
    ls = "--",
    zorder = -1000
)
fig.add_artist(con2)


x2, y2, _ = proj3d.proj_transform(0.1,0.5,5, ax3d.get_proj())
con3 = ConnectionPatch(
    xyA=(x2, y2), 
    coordsA=ax3d.transData,   # projected point from 3D axes
    xyB=(0.5,5),  
    coordsB=ax2d.transData,   # point in 2D axes coords
    arrowstyle="-",
    shrinkA=0,
    shrinkB=0,
    mutation_scale=15,
    lw=1.5,
    color="#999",
    ls = "--",
    zorder = -1000
)
fig.add_artist(con3)


x2, y2, _ = proj3d.proj_transform(0.1,0.5,-5, ax3d.get_proj())
con4 = ConnectionPatch(
    xyA=(x2, y2), 
    coordsA=ax3d.transData,   # projected point from 3D axes
    xyB=(0.5,-5),  
    coordsB=ax2d.transData,   # point in 2D axes coords
    arrowstyle="-",
    shrinkA=0,
    shrinkB=0,
    mutation_scale=15,
    lw=1.5,
    color="#999",
    ls = "--",
    zorder = -1000
)
fig.add_artist(con4)



fig.subplots_adjust(
    left=0.05,
    right=0.925,
    bottom = 0.075,
    top = 0.945,
)


# plt.savefig("McCormick.pdf")
plt.savefig("McCormick.pdf")



plt.show()