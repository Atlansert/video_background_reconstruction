"""Self-contained interactive reconstruction viewer."""

from pathlib import Path

import numpy as np


def write_html(
    points,
    colors,
    path,
    mesh_path=None,
    extrinsics=None,
    title="Background reconstruction",
    max_points=120000,
    max_triangles=120000,
):
    import open3d as o3d
    import plotly.graph_objects as go

    points = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3) if colors is not None else None
    if len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[selected]
        if colors is not None:
            colors = colors[selected]

    traces = []
    if len(points):
        marker = {"size": 1.5, "opacity": 0.75}
        if colors is not None and len(colors) == len(points):
            rgb = colors.astype(float)
            if rgb.max(initial=0) <= 1:
                rgb *= 255
            marker["color"] = [
                f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb.clip(0, 255)
            ]
        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
                name="Background points",
            )
        )

    if mesh_path and Path(mesh_path).exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if len(triangles) > max_triangles:
            selected = np.linspace(0, len(triangles) - 1, max_triangles, dtype=int)
            triangles = triangles[selected]
        vertex_colors = np.asarray(mesh.vertex_colors)
        kwargs = {}
        if len(vertex_colors) == len(vertices):
            kwargs["vertexcolor"] = (vertex_colors.clip(0, 1) * 255).astype(np.uint8)
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.82,
                name="Completed mesh",
                flatshading=False,
                **kwargs,
            )
        )

    if extrinsics is not None and len(extrinsics):
        centers = []
        for extrinsic in np.asarray(extrinsics):
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            centers.append(-rotation.T @ translation)
        centers = np.asarray(centers)
        traces.append(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="lines+markers",
                marker={"size": 3, "color": "#d62728"},
                line={"width": 4, "color": "#d62728"},
                name="Camera path",
            )
        )

    figure = go.Figure(traces)
    figure.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "aspectmode": "data",
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        legend={"orientation": "h"},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
