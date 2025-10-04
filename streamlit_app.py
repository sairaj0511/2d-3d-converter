import os
import uuid
import tempfile
import importlib.util
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(HERE, '2d_to_3d.py')

# Dynamically load 2d_to_3d.py (filename starts with a digit)
spec = importlib.util.spec_from_file_location("depth_module", MOD_PATH)
depth_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_module)

DepthEstimator = depth_module.DepthEstimator
create_3d_point_cloud = depth_module.create_3d_point_cloud
refine_depth_edge_aware = depth_module.refine_depth_edge_aware

# Cache estimator in session state
if "_estimators" not in st.session_state:
    st.session_state._estimators = {}

def get_estimator(model: str) -> DepthEstimator:
    if model not in st.session_state._estimators:
        st.session_state._estimators[model] = DepthEstimator(model_type=model)
    return st.session_state._estimators[model]

def resize_keep_longest(img: Image.Image, max_res: int) -> Image.Image:
    """Resize so the longest side <= max_res, keeping aspect ratio."""
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_res:
        return img
    scale = max_res / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

st.set_page_config(page_title="2D → 3D Depth & Point Cloud", layout="centered")
st.title("2D → 3D Depth + Point Cloud")
st.write("Upload an image to estimate depth and download a PLY point cloud.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["MiDaS_small", "DPT_Hybrid", "DPT_Large"], index=0)
    refine = st.selectbox("Edge Refinement", ["none", "bilateral", "guided"], index=0)
    scale = st.number_input("Depth scale factor (for PLY Z)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    max_res = st.slider("Max input resolution (long side, px)", min_value=480, max_value=2048, value=960, step=64)
    preview_points = st.slider("Preview points (downsample)", min_value=5000, max_value=100000, value=20000, step=5000)
    marker_size = st.slider("Marker size", min_value=1.0, max_value=4.0, value=1.6, step=0.2)

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","tif","tiff"]) 

if uploaded is not None:
    # Show preview
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    run = st.button("Run")
    if run:
        with st.spinner("Running depth estimation..."):
            # Save to temp file and call process_image with path
            img_proc = resize_keep_longest(image, max_res=max_res)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_proc.save(tmp.name)
                tmp_path = tmp.name
            try:
                estimator = get_estimator(model)
                img_np, depth_map = estimator.process_image(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            # Edge-aware refinement
            if refine != "none":
                depth_map = refine_depth_edge_aware(
                    img_np,
                    depth_map,
                    method=refine,
                    bilateral_d=9,
                    bilateral_sigma_color=0.1,
                    bilateral_sigma_space=5.0,
                    guided_radius=8,
                    guided_eps=1e-3,
                )

            # Show depth preview
            depth_vis = (255 * (depth_map - depth_map.min()) / (depth_map.ptp() + 1e-6)).astype(np.uint8)
            depth_img = Image.fromarray(depth_vis)
            st.image(depth_img, caption="Depth Map", use_column_width=True)

            # Build PLY
            points, colors = create_3d_point_cloud(img_np, depth_map, scale=scale)
            # Filter out non-finite points to ensure plotting works
            mask = np.isfinite(points).all(axis=1)
            if mask.sum() == 0:
                st.error("No valid 3D points to display. Try a different image or settings.")
                st.stop()
            points = points[mask]
            colors = colors[mask]

            # Interactive 3D preview (downsampled for performance)
            # Build a reduced set for both preview and static export
            n = points.shape[0]
            k = min(preview_points, n)
            if k < n:
                idx = np.random.choice(n, size=k, replace=False)
                pts_small = points[idx]
                cols_small = (colors[idx] * 255).clip(0,255).astype(np.uint8)
            else:
                pts_small = points
                cols_small = (colors * 255).clip(0,255).astype(np.uint8)
            color_str_small = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in cols_small]
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=pts_small[:,0], y=pts_small[:,1], z=pts_small[:,2],
                    mode='markers',
                    marker=dict(size=marker_size, color=color_str_small, opacity=0.9)
                )
            ])
            fig.update_layout(
                scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                margin=dict(l=0, r=0, b=0, t=0),
                height=600,
            )
            with st.spinner("Rendering 3D preview..."):
                st.plotly_chart(fig, use_container_width=True)
            out_dir = os.path.join(HERE, "web_outputs")
            os.makedirs(out_dir, exist_ok=True)
            stem = str(uuid.uuid4())
            ply_path = os.path.join(out_dir, f"{stem}.ply")

            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(ply_path, pcd)
            except Exception:
                # ASCII PLY fallback
                with open(ply_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {points.shape[0]}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    cols = (colors * 255).clip(0,255).astype(np.uint8)
                    for (x,y,z), (r,g,b) in zip(points, cols):
                        f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

            # Save static 3D image (PNG) using kaleido
            png_path = os.path.join(out_dir, f"{stem}.png")
            try:
                fig.write_image(png_path, format="png", scale=2)
            except Exception as e:
                # If kaleido missing or error, skip silently
                png_path = None

            st.success("Done!")
            with open(ply_path, "rb") as f:
                st.download_button("Download PLY", f, file_name="point_cloud.ply", mime="application/octet-stream")
            if png_path and os.path.exists(png_path):
                st.image(png_path, caption="3D Point Cloud (static)", use_column_width=True)
                with open(png_path, "rb") as f:
                    st.download_button("Download 3D PNG", f, file_name="point_cloud.png", mime="image/png")
else:
    st.info("Upload an image to begin.")
