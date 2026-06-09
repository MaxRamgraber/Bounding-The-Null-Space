import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}\usepackage{bm}"


plt.figure(figsize=(12,3.5))

gs = GridSpec(
    nrows = 1,
    ncols = 3,
    wspace = 0.05)

plt.subplot(gs[0,:2])

x1 = np.linspace(-2,2,2)
x2 = - x1 + 1

plt.plot(x1,x2,color="xkcd:dark grey",zorder = 0)


plt.gca().set_aspect("equal", adjustable="box")
plt.xlim([-1,7])
# plt.axis("equal")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(r"$\mathbf{A}$: Linear system $x_1 + x_2 = 1$", loc = "left")


x0, y0 = 0, 1
dx, dy = 1/np.sqrt(2), -1/np.sqrt(2)

plt.annotate(
    "",
    xy=(x0 + dx, y0 + dy),
    xytext=(x0, y0),
    arrowprops=dict(
        arrowstyle="->",
        color="#FF5000",
        linewidth=2
    )
)

plt.text(-0.15, 0.85, "null space $v_2$", ha="left", va="top", color = "#FF5000",rotation = -45)


x0, y0 = 0, 1
dx, dy = 1/np.sqrt(2), 1/np.sqrt(2)

plt.annotate(
    "",
    xy=(x0 + dx, y0 + dy),
    xytext=(x0, y0),
    arrowprops=dict(
        arrowstyle="->",
        color="#1988B8",
        linewidth=2
    )
)

plt.text(-0.15, 1.15, "active space $v_1$", ha="left", va="bottom", color = "#1988B8",rotation = 45)


plt.gca().plot([-1,2],[0,0], color="xkcd:grey", linewidth=1, zorder = -1)
plt.gca().axvline(0, color="xkcd:grey", linewidth=1, zorder = -1)

from matplotlib.ticker import MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().set_xticks([-1,0,1,2])



# plt.subplot(gs[0,1])


# plt.axis("off")

plt.text(0.4, 0.975, "Original linear system:", ha="left", va="top", transform=plt.gca().transAxes)
latex_matrix = (
    r"$y = \bm{A}\bm{x} = "
    r"\left[\begin{matrix} 1 && 1 \end{matrix}\right]\left[\begin{matrix} x_1 \\ x_2 \end{matrix}\right] = 1$"
)
plt.text(0.4, 0.9, latex_matrix, ha="left", va="top", transform=plt.gca().transAxes)




U,S,Vt = np.linalg.svd(np.array([[1, 1]]))

plt.text(0.4, 0.65, r"SVD of matrix $\bm{A} = \left[\begin{matrix} 1 && 1 \end{matrix}\right]$:", ha="left", va="top", transform=plt.gca().transAxes)
latex_matrix = (
    r"$\bm{A} = \bm{U}\bm{S}\bm{V}^\intercal = \left[\begin{matrix} 1 \end{matrix}\right]\left[\begin{matrix} \sqrt{2} && 0 \end{matrix}\right]\left[\begin{matrix} +\frac{1}{\sqrt{2}} && +\frac{1}{\sqrt{2}} \\ +\frac{1}{\sqrt{2}} && -\frac{1}{\sqrt{2}} \end{matrix}\right]^\intercal$"
)
plt.text(0.4, 0.575, latex_matrix, ha="left", va="top", transform=plt.gca().transAxes)

plt.fill(
    0.865 + np.asarray([-1,1,1,-1])*0.095,
    0.535 + np.asarray([-1,-1,1,1])*0.04,
    fc = "#1988B8",
    ec = "#1988B8", 
    alpha = 0.25,
    zorder = -1,
    transform=plt.gca().transAxes)

plt.fill(
    0.865 + np.asarray([-1,1,1,-1])*0.095,
    0.455 + np.asarray([-1,-1,1,1])*0.04,
    fc = "#FF5000",
    ec = "#FF5000", 
    alpha = 0.25,
    zorder = -1,
    transform=plt.gca().transAxes)




plt.text(0.4, 0.3, r"Active space:", ha="left", va="top", transform=plt.gca().transAxes)
plt.text(0.4, 0.15, "$v_{1}$", ha="left", va="top", transform=plt.gca().transAxes, color="#1988B8")
latex_matrix = (
    r"$=\left[\begin{matrix} +\frac{1}{\sqrt{2}} \\ +\frac{1}{\sqrt{2}} \end{matrix}\right]$"
)
plt.text(0.435, 0.2, latex_matrix, ha="left", va="top", transform=plt.gca().transAxes)




plt.text(0.7, 0.3, r"Null space:", ha="left", va="top", transform=plt.gca().transAxes)
plt.text(0.7, 0.15, "$v_{2}$", ha="left", va="top", transform=plt.gca().transAxes, color="#FF5000")
latex_matrix = (
    r"$=\left[\begin{matrix} +\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{matrix}\right]$"
)
plt.text(0.735, 0.2, latex_matrix, ha="left", va="top", transform=plt.gca().transAxes)


#%%

plt.subplot(gs[0,2])

# x1*x2 = 1
x1 = np.linspace(-3,0,101)[:-1]
x2 = 1/x1
plt.plot(x1,x2,color="xkcd:dark grey",zorder = 0)

x1 = np.linspace(0,3,101)[1:]
x2 = 1/x1
plt.plot(x1,x2,color="xkcd:dark grey",zorder = 0)



plt.gca().set_aspect("equal", adjustable="box")
plt.xlim([-3,3])
plt.ylim([-3,3])
# plt.axis("equal")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(r"$\mathbf{B}$: Nonlinear system $x_1 \cdot x_2 = 1$", loc = "left")



def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

# Pick your three x1 locations (avoid 0)
x1_vals = [-1,0.5, 2]

# Visual tuning
L = 1                 # arrow length in data units
t_off = 0.10             # text offset along the arrow direction
n_off = 0.10             # text offset perpendicular to the arrow direction

for x0 in x1_vals:
    y0 = 1.0 / x0

    # Tangent slope for y = 1/x: dy/dx = -1/x^2
    m = -1.0 / (x0**2)

    # Tangent (null space) direction and orthogonal (active space) direction
    t_hat = unit([1.0, m])            # tangent unit vector
    n_hat = np.array([-t_hat[1], t_hat[0]])  # rotate by +90° (orthogonal)

    # Arrow vectors
    t_vec = L * t_hat
    n_vec = L * n_hat

    # Angles for label rotation
    t_ang = np.degrees(np.arctan2(t_hat[1], t_hat[0]))
    n_ang = np.degrees(np.arctan2(n_hat[1], n_hat[0]))

    # --- null space (tangent) arrow ---
    plt.annotate(
        "",
        xy=(x0 + t_vec[0], y0 + t_vec[1]),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color="#FF5000", linewidth=2),
    )

    plt.text(
        x0 + t_vec[0]/2 - n_vec[0]/4, 
        y0 + t_vec[1]/2 - n_vec[1]/4, 
        "$v_2$", 
        ha = "center",
        va = "center",
        color = "#FF5000",
        rotation = np.degrees(np.atan2(t_vec[1]/2,(t_vec[0]/2))))

    # --- active space (orthogonal) arrow ---
    plt.annotate(
        "",
        xy=(x0 + n_vec[0], y0 + n_vec[1]),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color="#1988B8", linewidth=2),
    )


    plt.text(
        x0 + n_vec[0]/2 - t_vec[0]/4, 
        y0 + n_vec[1]/2 - t_vec[1]/4, 
        "$v_1$", 
        ha = "center",
        va = "center",
        color = "#1988B8",
        rotation = np.degrees(np.atan2(n_vec[1]/2,(n_vec[0]/2))))




# (optional) plot the curve for context
xx = np.linspace(min(x1_vals)*0.6, max(x1_vals)*1.4, 400)
plt.plot(xx, 1/xx, color="k", linewidth=1)

# plt.axis("equal")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
# plt.show()



# x0, y0 = 0, 1
# dx, dy = 1/np.sqrt(2), -1/np.sqrt(2)

# plt.annotate(
#     "",
#     xy=(x0 + dx, y0 + dy),
#     xytext=(x0, y0),
#     arrowprops=dict(
#         arrowstyle="->",
#         color="#FF5000",
#         linewidth=2
#     )
# )

# plt.text(-0.15, 0.85, "null space $v_2$", ha="left", va="top", color = "#FF5000",rotation = -45)


# x0, y0 = 0, 1
# dx, dy = 1/np.sqrt(2), 1/np.sqrt(2)

# plt.annotate(
#     "",
#     xy=(x0 + dx, y0 + dy),
#     xytext=(x0, y0),
#     arrowprops=dict(
#         arrowstyle="->",
#         color="#1988B8",
#         linewidth=2
#     )
# )

# plt.text(-0.15, 1.15, "active space $v_1$", ha="left", va="bottom", color = "#1988B8",rotation = 45)


# plt.gca().axhline(0, color="xkcd:grey", linewidth=1, zorder = -1)
# plt.gca().axvline(0, color="xkcd:grey", linewidth=1, zorder = -1)



plt.savefig("null_space.pdf",bbox_inches="tight")

plt.show()