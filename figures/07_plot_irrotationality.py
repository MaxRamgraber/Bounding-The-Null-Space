import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_3D_shape(ax, x, y, z, bottom = 0., facecolor="C0", edgecolor="k", alpha=1.0, zorder = None, lw = 1):
    """
    Extrude a closed 2D polygon from z=0 to z and draw all faces.
    """
    x = list(x)
    y = list(y)

    if x[0] != x[-1] or y[0] != y[-1]:
        x.append(x[0])
        y.append(y[0])

    faces = []

    # bottom
    faces.append([(xi, yi, bottom) for xi, yi in zip(x[:-1], y[:-1])])

    # top
    faces.append([(xi, yi, z) for xi, yi in zip(x[:-1], y[:-1])])

    # sides
    for i in range(len(x) - 1):
        faces.append([
            (x[i],   y[i],   bottom),
            (x[i+1], y[i+1], bottom),
            (x[i+1], y[i+1], z),
            (x[i],   y[i],   z),
        ])

    poly = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        lw = lw
    )
    ax.add_collection3d(poly)
    if zorder is None:
        poly.set_sort_zpos(-np.mean(y))
        
    
    return poly

arrow_x = np.array([-1,0,0,1,0,0,-1])
arrow_y = np.array([0.5,0.5,1,0,-1,-0.5,-0.5])*0.66

plt.figure(figsize=(12,4))

supgs = GridSpec(
    nrows   = 1,
    ncols   = 2,
    hspace = 0.)


subgs1 = GridSpecFromSubplotSpec(
    nrows   = 1,
    ncols   = 2,
    subplot_spec = supgs[0,0])

plt.subplot(subgs1[0,0], projection="3d")

box_x = np.array([0, 1, 0, -1, 0])
box_y = np.array([1, 0, -1, 0, 1])

for idx, (pos,label,fontcolor,cellcolor,z) in enumerate([
        ((0,1),"N","xkcd:cerulean","xkcd:cerulean",1),
        ((1,0),"E","k","#999",0.75),
        ((0,-1),"S","xkcd:orangish red","xkcd:orangish red",0.5),
        ((-1,0),"W","k","#999",0.75)]):
    
    create_3D_shape(
        ax  = plt.gca(), 
        x   = box_x + pos[0], 
        y   = box_y + pos[1], 
        z   = z, 
        facecolor   = "w", 
        edgecolor   = cellcolor, 
        alpha       = 0.75)
    
    plt.gca().text(
        pos[0],
        pos[1],
        z,
        label,
        ha = "center",
        va = "center",
        color = fontcolor,
        zorder = 100)

arrow_scale = 0.35

angle = -np.pi*0.25
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale + 0.5,
    arrow_y*np.cos(angle)*arrow_scale + arrow_x*np.sin(angle)*arrow_scale + 0.5,
    z           = 0.875,
    bottom      = 0.825,
    facecolor   = "xkcd:cerulean", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)
    
angle = -np.pi*0.75
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale + 0.5,
    arrow_y*np.cos(angle)*arrow_scale + arrow_x*np.sin(angle)*arrow_scale - 0.5,
    z           = 0.625,
    bottom      = 0.575,
    facecolor   = "xkcd:cerulean", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)

angle = -np.pi*0.25
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale - 0.5,
    arrow_y*np.cos(angle)*arrow_scale + arrow_x*np.sin(angle)*arrow_scale - 0.5,
    z           = 0.625,
    bottom      = 0.575,
    facecolor   = "xkcd:cerulean", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)

angle = -np.pi*0.75
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale - 0.5,
    arrow_y*np.cos(angle)*arrow_scale + arrow_x*np.sin(angle)*arrow_scale + 0.5,
    z           = 0.875,
    bottom      = 0.825,
    facecolor   = "xkcd:cerulean", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)

plt.title(r"$\mathbf{A1}:$ irrotational flow", loc = "left")

plt.gca().view_init(elev=45, azim=-90, roll=0)

plt.gca().set_xlim([-1.5,1.5])
plt.gca().set_ylim([-1.5,1.5])
plt.gca().set_zlim([0,1.5])
plt.gca().set_axis_off()



plt.subplot(subgs1[0,1], projection="3d")

box_x = np.array([0, 1, 0, -1, 0])
box_y = np.array([1, 0, -1, 0, 1])

for idx, (pos,label,fontcolor,cellcolor,z) in enumerate([
        ((0,1),"N","xkcd:cerulean","xkcd:cerulean",1),
        ((1,0),"E","k","#999",0.75),
        ((0,-1),"S","xkcd:orangish red","xkcd:orangish red",0.5),
        ((-1,0),"W","k","#999",0.75)]):
    
    create_3D_shape(
        ax  = plt.gca(), 
        x   = box_x + pos[0], 
        y   = box_y + pos[1], 
        z   = z, 
        facecolor   = "w", 
        edgecolor   = cellcolor, 
        alpha       = 0.75)
    
    plt.gca().text(
        pos[0],
        pos[1],
        z,
        label,
        ha = "center",
        va = "center",
        color = fontcolor,
        zorder = 100)
    
    
angle = -np.pi*0.25
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale/0.66 + 0.5,
    arrow_y*np.cos(angle)*arrow_scale/0.66 + arrow_x*np.sin(angle)*arrow_scale + 0.5,
    z           = 0.875,
    bottom      = 0.825,
    facecolor   = "xkcd:cerulean", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)
    
angle = -np.pi*0.75
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale/0.66 + 0.5,
    arrow_y*np.cos(angle)*arrow_scale/0.66 + arrow_x*np.sin(angle)*arrow_scale - 0.5,
    z           = 0.625,
    bottom      = 0.575,
    facecolor   = "xkcd:cerulean", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)

angle = np.pi*0.75
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale*0.66 - 0.5,
    arrow_y*np.cos(angle)*arrow_scale*0.66 + arrow_x*np.sin(angle)*arrow_scale - 0.5,
    z           = 0.625,
    bottom      = 0.575,
    facecolor   = "xkcd:orangish red", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)

angle = np.pi*0.25
create_3D_shape(
    plt.gca(), 
    arrow_x*np.cos(angle)*arrow_scale - arrow_y*np.sin(angle)*arrow_scale*0.66 - 0.5,
    arrow_y*np.cos(angle)*arrow_scale*0.66 + arrow_x*np.sin(angle)*arrow_scale + 0.5,
    z           = 0.875,
    bottom      = 0.825,
    facecolor   = "xkcd:orangish red", 
    edgecolor   = "#666", 
    alpha       = 0.75,
    zorder      = 1000,
    lw          = 0.5)


plt.title(r"$\mathbf{A2}:$ rotational flow", loc = "left")
    
    

plt.gca().view_init(elev=45, azim=-90, roll=0)

plt.gca().set_xlim([-1.5,1.5])
plt.gca().set_ylim([-1.5,1.5])
plt.gca().set_zlim([0,1.5])
plt.gca().set_axis_off()


#%%

import numpy as np

def curved_arrow_xy(
    center=(0.0, 0.0),
    radius=0.75,
    theta0=np.deg2rad(45),
    theta1=np.deg2rad(-45),
    width=0.18,
    start_length=0.3,
    end_length=0.3,
    head_length=0.18,
    head_width=None,
    n_arc=60,
    n_straight=8,
):
    """
    Closed 2D polygon for a thick curved arrow with:
      straight start -> circular arc -> straight end -> arrow head
    """
    if head_width is None:
        head_width = 1.8 * width

    dtheta = theta1 - theta0
    sgn = np.sign(dtheta)
    if sgn == 0:
        raise ValueError("theta0 and theta1 must differ")

    c = np.array(center, dtype=float)

    def p_arc(theta):
        return c + radius * np.array([np.cos(theta), np.sin(theta)])

    def t_arc(theta):
        # unit tangent in direction theta0 -> theta1
        t = sgn * np.array([-np.sin(theta), np.cos(theta)])
        return t / np.linalg.norm(t)

    p0_arc = p_arc(theta0)
    p1_arc = p_arc(theta1)

    t0 = t_arc(theta0)
    t1 = t_arc(theta1)

    # centerline endpoints of straight shaft
    p_start = p0_arc - start_length * t0
    p_end   = p1_arc + end_length * t1   # head base center

    # build centerline samples
    pts_start = np.linspace(p_start, p0_arc, n_straight, endpoint=False)

    th = np.linspace(theta0, theta1, n_arc)
    pts_arc = np.column_stack([
        c[0] + radius * np.cos(th),
        c[1] + radius * np.sin(th)
    ])

    pts_end = np.linspace(p1_arc, p_end, n_straight)

    pts = np.vstack([
        pts_start,
        pts_arc,
        pts_end[1:]   # skip duplicate p1_arc
    ])

    # tangents from centered differences
    d = np.zeros_like(pts)
    d[0] = pts[1] - pts[0]
    d[-1] = pts[-1] - pts[-2]
    d[1:-1] = pts[2:] - pts[:-2]

    dnorm = np.linalg.norm(d, axis=1, keepdims=True)
    tang = d / dnorm
    norm = np.column_stack([-tang[:, 1], tang[:, 0]])

    left = pts + 0.5 * width * norm
    right = pts - 0.5 * width * norm

    # arrow head
    t_end = tang[-1]
    n_end = norm[-1]
    tip = p_end + head_length * t_end

    head_left = p_end + 0.5 * head_width * n_end
    head_right = p_end - 0.5 * head_width * n_end

    x = np.concatenate([
        left[:, 0],
        [head_left[0], tip[0], head_right[0]],
        right[::-1, 0],
        [left[0, 0]]
    ])
    y = np.concatenate([
        left[:, 1],
        [head_left[1], tip[1], head_right[1]],
        right[::-1, 1],
        [left[0, 1]]
    ])

    return x, y

subgs2 = GridSpecFromSubplotSpec(
    nrows   = 1,
    ncols   = 2,
    subplot_spec = supgs[0,1])


plt.subplot(subgs2[0,0], projection="3d")

box_x = np.array([0, 1, 0, -1, 0])
box_y = np.array([1, 0, -1, 0, 1])

for idx, (pos,label,fontcolor,cellcolor,z) in enumerate([
        ((0,1),"N","xkcd:cerulean","xkcd:cerulean",1),
        ((1,0),"E","k","#999",0.75),
        ((0,-1),"S","xkcd:orangish red","xkcd:orangish red",0.5),
        ((-1,0),"W","k","#999",0.75)]):
    
    create_3D_shape(
        ax  = plt.gca(), 
        x   = box_x + pos[0], 
        y   = box_y + pos[1], 
        z   = z, 
        facecolor   = "w", 
        edgecolor   = cellcolor, 
        alpha       = 0.75)
    
    plt.gca().text(
        pos[0],
        pos[1],
        z,
        label,
        ha = "center",
        va = "center",
        color = fontcolor,
        zorder = 100)

# east path: N -> E -> S
x_e, y_e = curved_arrow_xy(
    center=(0.0, 0.0),
    radius=1,
    theta0=np.deg2rad(45),
    theta1=np.deg2rad(-45),
    width=0.3,
    head_length=0.45,
    head_width=0.45*4/3,
)

# west path: N -> W -> S
x_w, y_w = curved_arrow_xy(
    center=(0.0, 0.0),
    radius=1,
    theta0=np.deg2rad(135),
    theta1=np.deg2rad(225),
    width=0.3,
    head_length=0.45,
    head_width=0.45*4/3,
)

# B1: equal split
create_3D_shape(
    plt.gca(),
    x_e, y_e,
    z=0.75,
    bottom=0.7,
    facecolor="xkcd:cerulean",
    edgecolor="#666",
    alpha=0.75,
    zorder=1000,
    lw=0.5,
)

create_3D_shape(
    plt.gca(),
    x_w, y_w,
    z=0.75,
    bottom=0.7,
    facecolor="xkcd:cerulean",
    edgecolor="#666",
    alpha=0.75,
    zorder=1000,
    lw=0.5,
)


plt.title(r"$\mathbf{B1}:$ homogeneous $T$", loc = "left")

plt.gca().view_init(elev=45, azim=-90, roll=0)

plt.gca().set_xlim([-1.5,1.5])
plt.gca().set_ylim([-1.5,1.5])
plt.gca().set_zlim([0,1.5])
plt.gca().set_axis_off()



plt.subplot(subgs2[0,1], projection="3d")

box_x = np.array([0, 1, 0, -1, 0])
box_y = np.array([1, 0, -1, 0, 1])

for idx, (pos,label,fontcolor,cellcolor,z) in enumerate([
        ((0,1),"N","xkcd:cerulean","xkcd:cerulean",1),
        ((1,0),"E","k","#999",0.825),
        ((0,-1),"S","xkcd:orangish red","xkcd:orangish red",0.5),
        ((-1,0),"W","k","#999",0.625)]):
    
    create_3D_shape(
        ax  = plt.gca(), 
        x   = box_x + pos[0], 
        y   = box_y + pos[1], 
        z   = z, 
        facecolor   = "w", 
        edgecolor   = cellcolor, 
        alpha       = 0.75)
    
    plt.gca().text(
        pos[0],
        pos[1],
        z,
        label,
        ha = "center",
        va = "center",
        color = fontcolor,
        zorder = 100)

arrow_scale = 0.35


# east path: N -> E -> S
x_e, y_e = curved_arrow_xy(
    center=(0.0, 0.0),
    radius=1,
    theta0=np.deg2rad(45),
    theta1=np.deg2rad(-45),
    width=0.5,
    head_length=0.7,
    head_width=0.7*4/3,
    start_length=0.3,
    end_length=0.05,
)

# west path: N -> W -> S
x_w, y_w = curved_arrow_xy(
    center=(0.0, 0.0),
    radius=1,
    theta0=np.deg2rad(135),
    theta1=np.deg2rad(225),
    width=0.15,
    head_length=0.3,
    head_width=0.4,
)

# B1: equal split
create_3D_shape(
    plt.gca(),
    x_e, y_e,
    z=0.75,
    bottom=0.7,
    facecolor="xkcd:cerulean",
    edgecolor="#666",
    alpha=0.75,
    zorder=1000,
    lw=0.5,
)

create_3D_shape(
    plt.gca(),
    x_w, y_w,
    z=0.75,
    bottom=0.7,
    facecolor="xkcd:cerulean",
    edgecolor="#666",
    alpha=0.75,
    zorder=1000,
    lw=0.5,
)


plt.title(r"$\mathbf{B2}:$ heterogeneous $T$", loc = "left")

plt.gca().view_init(elev=45, azim=-90, roll=0)

plt.gca().set_xlim([-1.5,1.5])
plt.gca().set_ylim([-1.5,1.5])
plt.gca().set_zlim([0,1.5])
plt.gca().set_axis_off()



plt.savefig("irrotational_flow.png",dpi=300,bbox_inches="tight")
plt.savefig("irrotational_flow.pdf",dpi=300,bbox_inches="tight")