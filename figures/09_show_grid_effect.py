import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import networkx as nx


# ===================== Helpers =====================

def rotate_points(points, angle_deg, origin):
    """
    Rotate an array of 2D points by angle_deg around origin.

    points: (N, 2)
    origin: (2,) array-like
    """
    if angle_deg == 0:
        return points

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])

    origin = np.asarray(origin, dtype=float)
    pts = np.asarray(points, dtype=float)

    return (pts - origin) @ R.T + origin


# ===================== Point generators =====================

def rectangular_grid_points(xmin, xmax, ymin, ymax, spacing):
    """
    Regular rectangular lattice of points.
    Voronoi cells of this lattice are square-ish inside the domain.
    """
    xs = np.arange(xmin, xmax + spacing * 0.5, spacing)
    ys = np.arange(ymin, ymax + spacing * 0.5, spacing)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts


def triangular_grid_points(xmin, xmax, ymin, ymax, spacing):
    """
    Generate points on an equilateral triangular lattice.

    - 'spacing' is the side length between neighboring points.
    - The Voronoi diagram of these points gives hexagonal cells.
    - The Delaunay triangulation of these points gives equilateral triangles.

    Returns: (N, 2) array of (x, y) points inside the rectangle.
    """
    a1 = np.array([spacing, 0.0])
    a2 = np.array([spacing / 2.0, spacing * np.sqrt(3) / 2.0])

    width = xmax - xmin
    height = ymax - ymin

    ni = int(np.ceil(width / spacing)) + 3
    nj = int(np.ceil(height / (spacing * np.sqrt(3) / 2.0))) + 3

    points = []
    origin = np.array([xmin, ymin])

    for i in range(-ni, ni + 1):
        for j in range(-nj, nj + 1):
            p = origin + i * a1 + j * a2
            x, y = p
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                points.append(p)

    return np.asarray(points)


# ===================== Graph builders =====================

def voronoi_cells_to_graph(points, vor=None):
    """
    Voronoi-based cells → NetworkX graph.

    Nodes = cells with attributes:
        - center: seed point (x, y)
        - vertices: (N, 2) polygon vertices of the cell (or None if infinite)

    Edges = adjacency between cells (shared Voronoi ridge).
    """
    if vor is None:
        vor = scipy.spatial.Voronoi(points)

    G = nx.Graph()

    # nodes
    for i, p in enumerate(points):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]

        if region is None or len(region) == 0 or -1 in region:
            vertices = None
        else:
            vertices = vor.vertices[region]

        G.add_node(
            i,
            center=np.asarray(p, dtype=float),
            region_index=region_index,
            vertices=vertices,
        )

    # edges
    for (i, j), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        G.add_edge(int(i), int(j), ridge_vertices=(v1, v2))

    return G, vor


def delaunay_cells_to_graph(points, tri=None):
    """
    Delaunay-based *triangular* cells → NetworkX graph.

    Nodes = triangles, with attributes:
        - center: centroid of triangle (x, y)
        - vertices: (3, 2) triangle vertices

    Edges = adjacency between triangles (sharing an edge).
    """
    if tri is None:
        tri = scipy.spatial.Delaunay(points)

    G = nx.Graph()

    # nodes = triangles
    for k, simplex in enumerate(tri.simplices):
        verts = points[simplex]  # (3, 2)
        center = verts.mean(axis=0)
        G.add_node(
            k,
            center=center,
            vertices=verts,
            simplex_indices=simplex,
        )

    # edges via Delaunay neighbor info
    for k, neighs in enumerate(tri.neighbors):
        for n in neighs:
            if n >= 0 and n > k:  # avoid duplicates, ignore -1
                G.add_edge(int(k), int(n))

    return G, tri


# ===================== Main class =====================

class RegularGridGraph:
    """
    Regular grid → cells → NetworkX graph.

    grid_type:
        - 'rectangular' : square-ish Voronoi cells (rectangular point lattice)
        - 'hexagonal'   : hexagonal Voronoi cells (triangular point lattice)
        - 'triangular'  : triangular cells via Delaunay (triangular point lattice)

    rotation_deg:
        rotation angle in degrees, applied to the point lattice
        around rotation_origin (default: center of the bounding box).
    """

    def __init__(
        self,
        grid_type,
        xmin, xmax,
        ymin, ymax,
        spacing,
        rotation_deg=0.0,
        rotation_origin=None,
    ):
        self.grid_type = grid_type.lower()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.spacing = spacing
        
        self.offset = np.zeros(2)

        self.rotation_deg = float(rotation_deg)
        # if None, we’ll use center of the domain
        self.rotation_origin = rotation_origin

        self.points = None
        self.graph = None
        self.backend = None   # 'voronoi' or 'delaunay'
        self._raw_structure = None  # Voronoi or Delaunay object

    # ---------- point generation ----------
    
    def center_grid_at_point(self, target):
        
        distance = np.inf
        
        self.target_node = -1
        
        for node in list(self.graph.nodes):
            
            distance_to_target = np.linalg.norm(rect_grid.graph.nodes[node]["center"] - target)
            
            if distance_to_target < distance:
                
                self.target_node = node
                self.offset = -rect_grid.graph.nodes[node]["center"]
                distance = distance_to_target
    

    def _default_rotation_origin(self):
        return np.array(
            [(self.xmin + self.xmax) / 2.0,
             (self.ymin + self.ymax) / 2.0],
            dtype=float
        )

    def generate_points(self):
        gt = self.grid_type
        if gt in ("rect", "square", "rectangular"):
            pts = rectangular_grid_points(
                self.xmin, self.xmax, self.ymin, self.ymax, self.spacing
            )
        elif gt in ("hex", "hexagonal"):
            # hexagonal cells → Voronoi of triangular lattice
            pts = triangular_grid_points(
                self.xmin, self.xmax, self.ymin, self.ymax, self.spacing
            )
        elif gt in ("tri", "triangle", "triangular"):
            # triangular cells → Delaunay of triangular lattice
            pts = triangular_grid_points(
                self.xmin, self.xmax, self.ymin, self.ymax, self.spacing
            )
        else:
            raise ValueError(f"Unknown grid_type: {self.grid_type}")

        # Apply rotation if requested
        if self.rotation_deg != 0.0:
            origin = (
                self.rotation_origin
                if self.rotation_origin is not None
                else self._default_rotation_origin()
            )
            pts = rotate_points(pts, self.rotation_deg, origin)

        self.points = pts
        return pts

    # ---------- graph building ----------

    def build_graph(self):
        if self.points is None:
            self.generate_points()

        gt = self.grid_type
        if gt in ("rect", "square", "rectangular", "hex", "hexagonal"):
            G, vor = voronoi_cells_to_graph(self.points)
            self.backend = "voronoi"
            self._raw_structure = vor
        elif gt in ("tri", "triangle", "triangular"):
            G, tri = delaunay_cells_to_graph(self.points)
            self.backend = "delaunay"
            self._raw_structure = tri
        else:
            raise ValueError(f"Unknown grid_type: {self.grid_type}")

        self.graph = G
        return G

    # ---------- rotation after creation ----------

    def rotate(self, new_rotation_deg, rotation_origin=None):
        """
        Change rotation angle, regenerate points and graph.

        new_rotation_deg: absolute angle (not incremental)
        """
        self.rotation_deg = float(new_rotation_deg)
        if rotation_origin is not None:
            self.rotation_origin = rotation_origin

        # Regenerate everything
        self.points = None
        self.graph = None
        self._raw_structure = None
        return self.build_graph()

    # ---------- plotting ----------

    def plot(self, ax=None, plot_centers=False, cell_alpha=0.3):
        """
        Plot the cell polygons using the graph's 'vertices' attributes.
        Works for both Voronoi (polygons) and Delaunay (triangles).
        """
        if self.graph is None:
            self.build_graph()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        # draw cells
        for _, data in self.graph.nodes(data=True):
            poly = data["vertices"]
            if poly is None:
                # For Voronoi, some cells are infinite; skip them.
                continue
            poly = np.asarray(poly)
            poly_closed = np.vstack([poly, poly[0]])  + self.offset
            ax.fill(
                poly_closed[:, 0], 
                poly_closed[:, 1],
                alpha=cell_alpha, 
                facecolor="None",
                edgecolor="k", 
                linewidth=0.5)

        # draw centers
        if plot_centers:
            centers = np.vstack([d["center"] for _, d in self.graph.nodes(data=True)]) + self.offset
            ax.plot(centers[:, 0], centers[:, 1], "ro", markersize=2)

        ax.set_aspect("equal")
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        # ax.set_title(
        #     f"{self.grid_type.capitalize()} grid "
        #     f"({self.backend}, rotation={self.rotation_deg}°)"
        # )
        return fig, ax



# Define the flow direction
flow_direction = np.array([1,-1])

plt.figure(figsize=(12,5))    

from matplotlib.gridspec import GridSpec
gs = GridSpec(
    nrows   = 1,
    ncols   = 3)

for idx,(grid_type,rotation) in enumerate([("triangular",-15),("rectangular",0),("hexagonal",15)]):

    plt.subplot(gs[0,idx])
    
    if idx == 0:
        plt.title(r"$\boldsymbol{A}$: triangular", loc = "left")
    elif idx == 1:
        plt.title(r"$\boldsymbol{B}$: rectangular", loc = "left")
    else:
        plt.title(r"$\boldsymbol{C}$: hexagonal", loc = "left")

    # Rectangular grid rotated by 30°
    rect_grid = RegularGridGraph(
        grid_type=grid_type,
        xmin=-2.0, xmax=2.0,
        ymin=-2.0, ymax=2.0,
        spacing=0.2,
        rotation_deg=rotation,
    )
    
    rect_grid.build_graph()
    
    rect_grid.center_grid_at_point(np.zeros(2))
    
    # Assign flow directions
    for edge in list(rect_grid.graph.edges):
        
        j,i = edge
        
        edge_vector = rect_grid.graph.nodes[j]["center"] - rect_grid.graph.nodes[i]["center"]
        
        flow_sign = np.sign(np.inner(flow_direction,edge_vector))
        
        rect_grid.graph.edges[edge]["flow sign"] = flow_sign
    
    
    fig1, ax1 = rect_grid.plot(ax = plt.gca())
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Plot the center node
    poly = rect_grid.graph.nodes[rect_grid.target_node]["vertices"]
    if poly is not None:
        # For Voronoi, some cells are infinite; skip them.
        poly = np.asarray(poly)
        poly_closed = np.vstack([poly, poly[0]])  + rect_grid.offset
        plt.fill(
            poly_closed[:, 0], 
            poly_closed[:, 1],
            facecolor="xkcd:dark grey",
            edgecolor="k", 
            linewidth=0.5)
        
    # Plot all upstream and downstream nodes
    nodelist = list(rect_grid.graph.nodes)
    processed = [rect_grid.target_node]
    candidates = []
    for node in list(rect_grid.graph.neighbors(rect_grid.target_node)):
        
        edge = tuple(sorted([node,rect_grid.target_node]))
        flow_sign = rect_grid.graph.edges[edge]["flow sign"]
        if flow_sign != 0:
            if edge[0] == node:
                candidates.append([node,flow_sign])
            else:
                candidates.append([node,-flow_sign])
    # candidates = [[rect_grid.graph.edges[tuple(np.sort([node,rect_grid.target_node]))]["flow sign"], node] for node in list(rect_grid.graph.neighbors(rect_grid.target_node))]

    # While list of candidates is not empty
    while len(candidates) != 0:
        
        # Read the first candidate
        node, flow_sign = candidates[0]
        
        print(node)
        
        # Remove the candidate from the list
        candidates = candidates[1:]
        
        # Plot the candidate node
        poly = rect_grid.graph.nodes[node]["vertices"]
        if poly is not None:
            # For Voronoi, some cells are infinite; skip them.
            poly = np.asarray(poly)
            poly_closed = np.vstack([poly, poly[0]])  + rect_grid.offset
            
            if flow_sign == 1:
                facecolor = "xkcd:orangish red"
            else:
                facecolor = "xkcd:cerulean"
            
            plt.fill(
                poly_closed[:, 0], 
                poly_closed[:, 1],
                facecolor=facecolor,
                edgecolor="k", 
                linewidth=0.5)
        
        # Find the neighbors of the candidate
        ngbs = list(rect_grid.graph.neighbors(node))
        
        # Go through all neighbors
        for ngb in ngbs:
            
            # Get the edge
            edge = tuple(sorted([node,ngb]))
            
            # If the edge shares the flow sign, add it
            edge_sign = rect_grid.graph.edges[edge]["flow sign"]
            
            if edge[0] != ngb:
                edge_sign = -edge_sign
                
            if edge_sign == flow_sign and ngb not in processed:
                candidates.append([ngb,edge_sign])
                processed.append(ngb)
    
    # Draw the flow signs
    for edge in rect_grid.graph.edges:
        j, i = edge
    
        ci = rect_grid.graph.nodes[i]["center"]
        cj = rect_grid.graph.nodes[j]["center"]
    
        # vector between centroids
        vec = cj - ci
    
        # orientation given by flow_sign
        s = rect_grid.graph.edges[edge]["flow sign"]
        if s == 0:
            continue  # skip edges orthogonal to flow
    
        v = vec * s          # oriented along flow
        mid = 0.5 * (ci + cj)
    
        # arrow from 25% to 75% along v, centered on mid
        tail = mid - 0.25 * v
        dx, dy = 0.5 * v
    
        plt.arrow(
            tail[0] + rect_grid.offset[0],
            tail[1] + rect_grid.offset[1],
            dx,
            dy,
            length_includes_head=True,
            linewidth=0.8,
            head_width=0.025,
            head_length=0.025,
            alpha=0.25,
            fc="k",
            ec="k",
            zorder=4,
        )

    
    # for edge in list(rect_grid.graph.edges):
        
    #     j,i = edge
        
    #     edge_vector = rect_grid.graph.nodes[j]["center"] - rect_grid.graph.nodes[i]["center"]
        
    #     flow_sign = np.sign(np.inner(flow_direction,edge_vector))
        
    #     rect_grid.graph.edges[edge]["flow sign"] = flow_sign
    
    # raise Exception

plt.savefig("grid_constraints.pdf",dpi=300,bbox_inches="tight")
plt.savefig("grid_constraints.png",dpi=300,bbox_inches="tight")