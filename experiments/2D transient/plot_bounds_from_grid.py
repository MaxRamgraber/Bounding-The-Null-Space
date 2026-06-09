def plot_bounds(bounds, grid, timesteps, save_figures = False, figure_name = "plot.png"):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    # =============================================================================
    # Helpers
    # =============================================================================

    def _get_vertices_for_node(node, i):
        # Prefer vertices on time-node; fall back to spatial node i (like Sy)
        verts = grid.nodes[node].get("vertices", None)
        if verts is None and (i in grid.nodes):
            try:
                verts = grid.nodes[i].get("vertices", None)
            except Exception:
                verts = None
        return verts

    def build_node_arrays_for_t(t):
        # Collect nodes at time t with their polygons, positions and values
        pts, polys, keys_h, keys_R, keys_Sy = [], [], [], [], []
        idxs = []

        for node in grid.nodes:
            if not (isinstance(node, tuple) and len(node) == 2):
                continue
            tn, i = node
            if tn != t:
                continue

            verts = _get_vertices_for_node(node, i)
            if verts is None:
                continue
            poly = np.asarray(verts, dtype=float)
            if poly.ndim != 2 or poly.shape[1] != 2:
                continue

            x = grid.nodes[node].get("xpos", None)
            y = grid.nodes[node].get("ypos", None)
            if x is None or y is None:
                x = float(np.mean(poly[:, 0]))
                y = float(np.mean(poly[:, 1]))

            polys.append(poly)
            pts.append([x, y])
            idxs.append(i)

            keys_h.append(str(grid.nodes[node]["h"]))
            keys_R.append(str(grid.nodes[node]["R"]))

            if timesteps != 1:
                # Sy is not time-indexed; stored on spatial node
                sy_key = str(grid.nodes.get(i, {}).get("Sy", None)) if (i in grid.nodes) else None
                if sy_key is None:
                    sy_key = f"Sy_{i}"
                keys_Sy.append(sy_key)

        pts = np.asarray(pts, dtype=float)

        h_min = np.array([bounds[k][0] for k in keys_h])
        h_max = np.array([bounds[k][1] for k in keys_h])
        R_min = np.array([bounds[k][0] for k in keys_R])
        R_max = np.array([bounds[k][1] for k in keys_R])

        if timesteps != 1:
            Sy_min = np.array([bounds[k][0] for k in keys_Sy])
            Sy_max = np.array([bounds[k][1] for k in keys_Sy])
        else:
            Sy_min = Sy_max = None

        return pts, polys, h_min, h_max, R_min, R_max, Sy_min, Sy_max

    def set_limits_from_polys(ax, polys, pad=0.5):
        if not polys:
            return
        vv = np.vstack(polys)
        ax.set_xlim(vv[:, 0].min() - pad, vv[:, 0].max() + pad)
        ax.set_ylim(vv[:, 1].min() - pad, vv[:, 1].max() + pad)

    def add_cells(ax, polys, vals, vmin, vmax, cmap="turbo"):
        if len(polys) == 0:
            return None
        pc = PolyCollection(polys, array=vals, cmap=cmap, edgecolors='k', linewidths=0.3)
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        set_limits_from_polys(ax, polys, pad=0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        return pc

    def add_cell_outlines(ax, polys, edgecolor="#cccccc", linewidth=0.3, zorder=0):
        if len(polys) == 0:
            return None
        pc = PolyCollection(polys, facecolors="none", edgecolors=edgecolor, linewidths=linewidth, zorder=zorder)
        ax.add_collection(pc)
        set_limits_from_polys(ax, polys, pad=0.5)
        return pc

    def edge_midpoints_for_t(t):
        xs, ys, Tmin, Tmax = [], [], [], []
        for e in grid.edges:
            (t1, j), (t2, i) = e
            if t1 != t or t2 != t:
                continue
            xj, yj = grid.nodes[(t, j)]["xpos"], grid.nodes[(t, j)]["ypos"]
            xi, yi = grid.nodes[(t, i)]["xpos"], grid.nodes[(t, i)]["ypos"]
            xs.append(0.5*(xj+xi)); ys.append(0.5*(yj+yi))
            Tkey = str(grid.edges[e]["T"])
            Tmin.append(bounds[Tkey][0]); Tmax.append(bounds[Tkey][1])
        if len(xs) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return np.array(xs), np.array(ys), np.array(Tmin), np.array(Tmax)

    # =============================================================================
    # Global mins/maxes for consistent colorbars
    # =============================================================================

    all_h_min, all_h_max, all_R_min, all_R_max = [], [], [], []
    for t in range(timesteps):
        pts, polys, hmin, hmax, Rmin, Rmax, _, _ = build_node_arrays_for_t(t)
        if len(polys) == 0:
            continue
        all_h_min.append(hmin); all_h_max.append(hmax)
        all_R_min.append(Rmin); all_R_max.append(Rmax)
    if len(all_h_min) == 0:
        return

    Hmin_global = np.nanmin(np.concatenate(all_h_min))
    Hmax_global = np.nanmax(np.concatenate(all_h_max))
    Rmin_global = np.nanmin(np.concatenate(all_R_min))
    Rmax_global = np.nanmax(np.concatenate(all_R_max))

    all_T_min, all_T_max = [], []
    for t in range(timesteps):
        _, _, Tmin, Tmax = edge_midpoints_for_t(t)
        if Tmin.size:
            all_T_min.append(Tmin); all_T_max.append(Tmax)
    if len(all_T_min):
        T_min_global = np.nanmin(np.concatenate(all_T_min))
        T_max_global = np.nanmax(np.concatenate(all_T_max))
    else:
        T_min_global = 1.0
        T_max_global = 1.0

    # =============================================================================
    # Plot per timestep
    # =============================================================================

    for t in range(timesteps):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(nrows=3, ncols=3, figure=fig)

        pts, polys, hmin, hmax, Rmin, Rmax, _, _ = build_node_arrays_for_t(t)
        hdiff = hmax - hmin
        Rdiff = Rmax - Rmin

        # --- h min / max / diff (cells from vertices) ---
        ax = fig.add_subplot(gs[0, 0])
        pc = add_cells(ax, polys, hmin, Hmin_global, Hmax_global, cmap="turbo")
        if pc is not None:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label(r'hydraulic head' + "\n" + '[m]')
        ax.set_title(f"h t={t} min")

        ax = fig.add_subplot(gs[1, 0])
        pc = add_cells(ax, polys, hmax, Hmin_global, Hmax_global, cmap="turbo")
        if pc is not None:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label(r'hydraulic head' + "\n" + '[m]')
        ax.set_title(f"h t={t} max")

        ax = fig.add_subplot(gs[2, 0])
        dmin = np.nanmin(hdiff); dmax = np.nanmax(hdiff)
        pc = add_cells(ax, polys, hdiff, dmin, dmax, cmap="turbo")
        if pc is not None:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label(r'hydraulic head' + "\n" + '[m]')
        ax.set_title(f"h t={t} diff")

        # --- R min / max / diff (cells from vertices) ---
        ax = fig.add_subplot(gs[0, 1])
        pc = add_cells(ax, polys, Rmin, Rmin_global, Rmax_global, cmap="turbo")
        if pc is not None:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label(r'recharge' + "\n" + '[m/s]')
        ax.set_title(f"R t={t} min")

        ax = fig.add_subplot(gs[1, 1])
        pc = add_cells(ax, polys, Rmax, Rmin_global, Rmax_global, cmap="turbo")
        if pc is not None:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label(r'recharge' + "\n" + '[m/s]')
        ax.set_title(f"R t={t} max")

        ax = fig.add_subplot(gs[2, 1])
        dmin = np.nanmin(Rdiff); dmax = np.nanmax(Rdiff)
        pc = add_cells(ax, polys, Rdiff, dmin, dmax, cmap="turbo")
        if pc is not None:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label(r'recharge' + "\n" + '[m/s]')
        ax.set_title(f"R t={t} diff")

        # --- T min / max / diff (edge midpoints scatter; use cell outlines for context) ---
        axmin = fig.add_subplot(gs[0, 2])
        axmax = fig.add_subplot(gs[1, 2])
        axdif = fig.add_subplot(gs[2, 2])

        X, Y, Tmin, Tmax = edge_midpoints_for_t(t)

        for axT in (axmin, axmax, axdif):
            add_cell_outlines(axT, polys, edgecolor="#dddddd", linewidth=0.25, zorder=0)

            # draw light graph context
            for e in grid.edges:
                (t1, j), (t2, i) = e
                if t1 != t or t2 != t:
                    continue
                xj, yj = grid.nodes[(t, j)]["xpos"], grid.nodes[(t, j)]["ypos"]
                xi, yi = grid.nodes[(t, i)]["xpos"], grid.nodes[(t, i)]["ypos"]
                axT.plot([xj, xi], [yj, yi], lw=0.3, color='#cccccc', zorder=1)

            axT.set_aspect('equal')
            axT.invert_yaxis()
            axT.set_xticks([]); axT.set_yticks([])

        if X.size:
            sc1 = axmin.scatter(
                X, Y, c=Tmin,
                norm=LogNorm(vmin=max(T_min_global, 1e-16), vmax=max(T_max_global, T_min_global*1.0001)),
                cmap="turbo", s=20, zorder=2
            )
            axmin.set_title(f"T t={t} min")
            cbar = fig.colorbar(sc1, ax=axmin)
            cbar.set_label(r'transmissivity' + "\n" + '[m²/s]')

            sc2 = axmax.scatter(
                X, Y, c=Tmax,
                norm=LogNorm(vmin=max(T_min_global, 1e-16), vmax=max(T_max_global, T_min_global*1.0001)),
                cmap="turbo", s=20, zorder=2
            )
            axmax.set_title(f"T t={t} max")
            cbar = fig.colorbar(sc2, ax=axmax)
            cbar.set_label(r'transmissivity' + "\n" + '[m²/s]')

            Tdiff = Tmax - Tmin
            sc3 = axdif.scatter(X, Y, c=Tdiff, cmap="turbo", s=20, zorder=2)
            axdif.set_title(f"T t={t} diff")
            cbar = fig.colorbar(sc3, ax=axdif)
            cbar.set_label(r'transmissivity' + "\n" + '[m²/s]')
        else:
            axmin.set_title(f"T t={t} min")
            axmax.set_title(f"T t={t} max")
            axdif.set_title(f"T t={t} diff")
            
            
        if save_figures:
            plt.savefig(
                figure_name.split(".")[0]+"_t="+str(t).zfill(3)+".png",
                dpi = 300)

        # plt.show()  # keep your original behavior (commented out)

    return

def plot_grid_dirs(grid, timesteps,
                   show_flow_dirs=True,
                   show_storage_signs=True,
                   flow_arrow_frac=0.25,
                   flow_arrow_lw=0.9,
                   flow_arrow_color="k",
                   flow_arrow_alpha=0.9,
                   storage_fontsize=12,
                   storage_color="k",
                   storage_bbox_alpha=0.35,
                   storage_zero=False,
                   outline_color="#dddddd",
                   outline_lw=0.35,
                   figsize=(6.5, 6.5),
                   title_prefix="dirs"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # =============================================================================
    # Helpers
    # =============================================================================

    def _get_vertices(node):
        return grid.nodes[node].get("vertices", None)

    def _build_geom_for_t(t):
        polys = []
        poly_by_i = {}
        xy_by_i = {}
        for node in grid.nodes:
            if not (isinstance(node, tuple) and len(node) == 2):
                continue
            tn, i = node
            if tn != t:
                continue
            verts = _get_vertices(node)
            if verts is None:
                continue
            poly = np.asarray(verts, dtype=float)
            if poly.ndim != 2 or poly.shape[1] != 2:
                continue

            x = grid.nodes[node].get("xpos", None)
            y = grid.nodes[node].get("ypos", None)
            if x is None or y is None:
                x = float(np.mean(poly[:, 0]))
                y = float(np.mean(poly[:, 1]))

            polys.append(poly)
            poly_by_i[i] = poly
            xy_by_i[i] = (float(x), float(y))

        return polys, poly_by_i, xy_by_i

    def _set_limits_from_polys(ax, polys, pad=0.5):
        if not polys:
            return
        vv = np.vstack(polys)
        ax.set_xlim(vv[:, 0].min() - pad, vv[:, 0].max() + pad)
        ax.set_ylim(vv[:, 1].min() - pad, vv[:, 1].max() + pad)

    def _edge_flow_sign(u, v):
        try:
            return grid.edges[(u, v)].get("flow sign", None)
        except Exception:
            try:
                return grid.edges[(v, u)].get("flow sign", None)
            except Exception:
                return None

    def _shared_face_midpoint(poly_a, poly_b, decimals=6):
        if poly_a is None or poly_b is None:
            return None
        A = {(round(float(x), decimals), round(float(y), decimals)) for x, y in np.asarray(poly_a)}
        B = {(round(float(x), decimals), round(float(y), decimals)) for x, y in np.asarray(poly_b)}
        common = list(A.intersection(B))
        if len(common) < 2:
            return None
        pts = np.array(common, dtype=float)

        best = (0, 1)
        best_d2 = -1.0
        for p in range(len(pts)):
            for q in range(p + 1, len(pts)):
                d2 = float(np.sum((pts[p] - pts[q]) ** 2))
                if d2 > best_d2:
                    best_d2 = d2
                    best = (p, q)

        return 0.5 * (pts[best[0]] + pts[best[1]])

    def _overlay_flow_arrows(ax, t, poly_by_i, xy_by_i):
        if not show_flow_dirs:
            return

        for e in grid.edges:
            (t1, j), (t2, i) = e
            if t1 != t or t2 != t:
                continue

            sign = _edge_flow_sign((t1, j), (t2, i))
            if sign not in (-1, 0, 1):
                continue

            if (j not in xy_by_i) or (i not in xy_by_i):
                continue

            cj = np.array(xy_by_i[j], dtype=float)
            ci = np.array(xy_by_i[i], dtype=float)
            v = ci - cj
            dist = float(np.linalg.norm(v))
            if dist <= 0.0:
                continue
            n = v / dist

            mid = _shared_face_midpoint(poly_by_i.get(j, None), poly_by_i.get(i, None))
            if mid is None:
                mid = 0.5 * (cj + ci)

            L = float(flow_arrow_frac) * dist
            p0 = mid - 0.5 * L * n
            p1 = mid + 0.5 * L * n

            if sign == 0:
                ax.plot([mid[0]], [mid[1]],
                        marker='o', markersize=2.0,
                        color=flow_arrow_color, alpha=flow_arrow_alpha, zorder=6)
            else:
                if sign < 0:
                    p0, p1 = p1, p0

                ax.annotate(
                    "",
                    xy=(p1[0], p1[1]),
                    xytext=(p0[0], p0[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=float(flow_arrow_lw),
                        color=flow_arrow_color,
                        alpha=float(flow_arrow_alpha),
                        shrinkA=0.0,
                        shrinkB=0.0,
                        mutation_scale=8.0,
                    ),
                    zorder=6,
                )

    def _overlay_storage_signs(ax, t, xy_by_i):
        if not show_storage_signs:
            return
        if t <= 0:
            return

        for i, (x, y) in xy_by_i.items():
            sign = _edge_flow_sign((t - 1, i), (t, i))
            if sign not in (-1, 0, 1):
                continue
            if sign == 0 and (not storage_zero):
                continue

            if sign > 0:
                s = "+"
            elif sign < 0:
                s = "\u2212"  # proper minus
            else:
                s = "0"

            ax.text(
                x, y, s,
                ha="center", va="center",
                fontsize=float(storage_fontsize),
                color=storage_color,
                fontweight="bold",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.05",
                    facecolor="white",
                    edgecolor="none",
                    alpha=float(storage_bbox_alpha),
                ),
            )

    # =============================================================================
    # Plot per timestep
    # =============================================================================

    for t in range(timesteps):
        polys, poly_by_i, xy_by_i = _build_geom_for_t(t)
        if len(polys) == 0:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        pc = PolyCollection(polys, facecolors="none",
                            edgecolors=outline_color,
                            linewidths=float(outline_lw),
                            zorder=0)
        ax.add_collection(pc)

        _set_limits_from_polys(ax, polys, pad=0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_prefix} t={t}")

        _overlay_flow_arrows(ax, t, poly_by_i, xy_by_i)
        _overlay_storage_signs(ax, t, xy_by_i)

    return
