def point_in_polygon(x, y, polygon):
    """
    Ray-casting point-in-polygon test.
    polygon: list of (x, y) vertices, closed or open.
    """
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Edge straddles horizontal ray at y
        if (y1 > y) != (y2 > y):
            x_int = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            if x_int > x:  # intersection to the right of (x, y)
                inside = not inside
    return inside


def build_hex_grid(polygon, seed, dx):
    """
    Build a pointy-top hexagonal grid inside a polygon.

    Parameters
    ----------
    polygon : list[(float, float)]
        Vertices of the polygon in world coords, e.g.
        [(0,0), (5,0), (5,5), (0,5)].
    seed : (float, float)
        World coord of the hex that should sit at axial (q=0, r=0),
        e.g. (2.5, 2.5).
    dx : float
        Hex "size" (side length = distance from center to any vertex).

    Returns
    -------
    grid : nx.Graph
        Nodes: integer ids 0..N-1 with attributes:
            - 'q', 'r'   : axial hex coordinates (pointy-top)
            - 'xpos','ypos' : world coordinates of cell center
        Edges: between neighboring hexes, with attributes:
            - 'dx' : centre-to-centre distance between the two cells
            - 'w'  : face width (shared edge length)
    """
    
    import math
    import numpy as np
    import networkx as nx
    
    size = float(dx)  # hex side length
    corner_angles = [math.radians(30.0 + 60.0 * k) for k in range(6)]  # pointy-top
    sx, sy = seed
    
    # Translate polygon so seed is at the origin (local coordinates)
    poly_local = [(x - sx, y - sy) for (x, y) in polygon]
    minx = min(x for x, y in poly_local)
    maxx = max(x for x, y in poly_local)
    miny = min(y for x, y in poly_local)
    maxy = max(y for x, y in poly_local)

    # Pointy-top axial -> local (x, y) mapping:
    #   x = size * sqrt(3) * (q + r/2)
    #   y = size * 3/2 * r
    #
    # Invert roughly to get a bounding box in (q, r) to iterate over.
    r_min = math.floor((2.0 / 3.0) * (miny / size)) - 2
    r_max = math.ceil((2.0 / 3.0) * (maxy / size)) + 2

    grid = nx.Graph()
    axial_to_node = {}
    node_id = 0

    for r in range(r_min, r_max + 1):
        # For this r, x = size * sqrt(3) * (q + r/2)
        # ⇒ q = x / (size*sqrt(3)) - r/2
        q_min = math.floor(minx / (size * math.sqrt(3.0)) - r / 2.0) - 2
        q_max = math.ceil(maxx / (size * math.sqrt(3.0)) - r / 2.0) + 2

        for q in range(q_min, q_max + 1):
            x_local = size * math.sqrt(3.0) * (q + r / 2.0)
            y_local = size * 1.5 * r

            if not point_in_polygon(x_local, y_local, poly_local):
                continue

            x_world = sx + x_local
            y_world = sy + y_local
            
            vertices = [(x_world + size * math.cos(a), y_world + size * math.sin(a)) for a in corner_angles]
            
            area = (3.0 * np.sqrt(3.0) / 2.0) * size * size

            grid.add_node(
                node_id,
                q=q,
                r=r,
                xpos=x_world,
                ypos=y_world,
                vertices=vertices,
                area=area,
            )
            axial_to_node[(q, r)] = node_id
            node_id += 1

    # Axial neighbor directions for pointy-top hexes
    neighbor_dirs = [(1, 0), (1, -1), (0, -1),
                     (-1, 0), (-1, 1), (0, 1)]

    # Geometric edge parameters for a regular hex grid
    center_dist = size * math.sqrt(3.0)  # centre-to-centre distance
    width = size                          # face width (side length)

    # Connect neighbors and attach dx, w to each edge
    for (q, r), i in axial_to_node.items():
        for dq, dr in neighbor_dirs:
            nbr = (q + dq, r + dr)
            j = axial_to_node.get(nbr)
            if j is None or j <= i:
                continue
            grid.add_edge(i, j, dx=center_dist, w=width)

    return grid

def plot_labelled_grid(grid, ax=None, show=True, facecolor="none", edgecolor="k",
                  linewidth=1.0, fontsize=8):
    """
    Plot a hex grid (nx.Graph) where each node has:
      - corners : list[(x,y)] length 6 (hex vertices in world coords)
      - xpos,ypos : center (for label placement)
    Labels each cell with its node id.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    if ax is None:
        fig, ax = plt.subplots()

    patches = []
    xs = []
    ys = []

    for n, data in grid.nodes(data=True):
        vertices = data.get("vertices", None)
        if vertices is None:
            raise ValueError(f"Node {n} missing 'corners' attribute")

        patches.append(MplPolygon(vertices, closed=True))
        for x, y in vertices:
            xs.append(x)
            ys.append(y)

        cx = data.get("xpos", sum(x for x, _ in vertices) / 6.0)
        cy = data.get("ypos", sum(y for _, y in vertices) / 6.0)
        ax.text(cx, cy, str(n), ha="center", va="center", fontsize=fontsize)

    pc = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection(pc)

    if xs and ys:
        pad = 0.05 * max((max(xs) - min(xs)), (max(ys) - min(ys)), 1.0)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal", adjustable="box")

    if show:
        plt.show()

    return ax

#%%

def plot_node_values(
    grid,
    values,                 # dict[node] -> float
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    show=True,
    colorbar=True,
    edgecolor="k",
    linewidth=0.6,
    title=None,
    pad_frac=0.05,
    layer_t=None,           # optional: if nodes are (t,i), plot only this t
    annotate=False,
    annotate_fmt="{:.2f}",
    annotate_fontsize=8,
):
    """
    Plot a scalar field on a hex grid using the per-node polygon geometry stored in `grid`.

    Parameters
    ----------
    grid : nx.Graph
        Nodes must carry 'vertices' = list[(x,y)] (polygon), and optionally 'xpos','ypos'.
        Nodes may be ints (spatial) or tuples like (t, i).
    values : dict
        Mapping {node_name: scalar}. Missing nodes are shown as "bad" (masked).
    cmap : str or Colormap
        Matplotlib colormap name/object.
    vmin, vmax : float or None
        Color limits. If None, inferred from finite values.
    layer_t : int or None
        If provided and nodes are (t,i), only plot nodes with that t.
        (If your grid truly has only one timestep, you can ignore this.)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # -----------------------------
    # Select nodes to plot
    # -----------------------------
    nodes = list(grid.nodes)
    if layer_t is not None:
        nodes = [u for u in nodes if isinstance(u, tuple) and len(u) == 2 and u[0] == layer_t]

    polys = []
    vals = []
    centers = []

    for u in nodes:
        verts = grid.nodes[u].get("vertices", None)
        if verts is None:
            continue

        poly = np.asarray(verts, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2:
            continue

        polys.append(poly)

        # value (masked if missing / non-finite)
        val = values.get(u, np.nan)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        vals.append(val)

        cx = grid.nodes[u].get("xpos", float(np.mean(poly[:, 0])))
        cy = grid.nodes[u].get("ypos", float(np.mean(poly[:, 1])))
        centers.append((float(cx), float(cy)))

    if len(polys) == 0:
        raise ValueError("No plottable polygons found (missing 'vertices'?).")

    vals = np.asarray(vals, dtype=float)
    vals_m = np.ma.masked_invalid(vals)

    # -----------------------------
    # Figure / axes
    # -----------------------------
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Make a copy of colormap so we can set a 'bad' color without side effects
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(alpha=0.0)  # missing values transparent (change if you prefer)

    pc = PolyCollection(
        polys,
        array=vals_m,
        cmap=cm,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    ax.add_collection(pc)

    # color limits
    finite = np.asarray(vals[np.isfinite(vals)], dtype=float)
    if vmin is None:
        vmin = float(np.min(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if finite.size else 1.0
    pc.set_clim(vmin, vmax)

    # bounds / aspect
    vv = np.vstack(polys)
    dx = float(vv[:, 0].max() - vv[:, 0].min())
    dy = float(vv[:, 1].max() - vv[:, 1].min())
    pad = pad_frac * max(dx, dy, 1.0)
    ax.set_xlim(vv[:, 0].min() - pad, vv[:, 0].max() + pad)
    ax.set_ylim(vv[:, 1].min() - pad, vv[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    if colorbar:
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    if annotate:
        for (cx, cy), v in zip(centers, vals):
            if np.isfinite(v):
                ax.text(cx, cy, annotate_fmt.format(v), ha="center", va="center",
                        fontsize=annotate_fontsize)

    if show:
        plt.show()

    return ax, pc

