import os
import glob
import shutil
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings

class DepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load MiDaS model
        self.midas = self._load_midas_with_retry(model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # Set up MiDaS transforms for the chosen model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            # DPT models take PIL Image RGB
            self.transform = midas_transforms.dpt_transform
            self._uses_small = False
        else:
            # MiDaS_small uses small_transform (expects NumPy BGR)
            self.transform = midas_transforms.small_transform
            self._uses_small = True

    def _load_midas_with_retry(self, model_type: str):
        try:
            return torch.hub.load("intel-isl/MiDaS", model_type)
        except Exception as e:
            msg = str(e)
            if "PytorchStreamReader failed reading zip archive" in msg or "failed finding central directory" in msg:
                print("Detected corrupted MiDaS checkpoint in torch hub cache. Deleting and retrying...")
                self._delete_midas_checkpoints()
                # Retry once
                return torch.hub.load("intel-isl/MiDaS", model_type)
            raise

    def _delete_midas_checkpoints(self):
        # Typical cache dirs on Windows
        candidates = []
        # TORCH_HOME override
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            candidates.append(Path(torch_home) / "hub" / "checkpoints")
        # Default locations
        candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            candidates.append(Path(local_app) / "torch" / "hub" / "checkpoints")

        patterns = [
            "dpt_*",  # DPT_Large / DPT_Hybrid
            "midas_*",  # MiDaS_small
            "*.pt",
        ]
        removed_any = False
        for base in candidates:
            try:
                for pat in patterns:
                    for f in base.glob(pat):
                        # Only remove likely MiDaS weights
                        if any(tok in f.name.lower() for tok in ["midas", "dpt", "hybrid", "large", "small"]):
                            try:
                                print(f"Removing cached checkpoint: {f}")
                                f.unlink(missing_ok=True)
                                removed_any = True
                            except Exception:
                                pass
            except Exception:
                pass
        if not removed_any:
            print("No cached MiDaS checkpoints found to delete. Will retry download anyway.")
    
    def process_image(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        if getattr(self, "_uses_small", False):
            # Transform expects NumPy BGR
            img_np = np.array(img)  # RGB uint8
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            input_batch = self.transform(img_bgr).to(self.device)
        else:
            input_batch = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get depth prediction
        with torch.no_grad():
            t0 = time.time()
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            infer_ms = (time.time() - t0) * 1000
            print(f"Inference time: {infer_ms:.1f} ms on {self.device}")
        
        # Convert to numpy and normalize
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return np.array(img), depth_map

def create_3d_point_cloud(image, depth_map, scale=1.0):
    # Convert depth map to 3D point cloud
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D coordinates
    points = np.dstack((x, y, depth_map * scale * 1000))  # Scale for better visualization
    
    # Flatten the points and corresponding colors
    points = points.reshape(-1, 3)
    colors = image.reshape(-1, 3) / 255.0
    
    return points, colors

def refine_depth_edge_aware(image_rgb, depth_map,
                            method='none',
                            bilateral_d=9, bilateral_sigma_color=0.1, bilateral_sigma_space=5.0,
                            guided_radius=8, guided_eps=1e-3):
    """Optionally refine depth using edge-aware filtering.
    image_rgb: HxWx3 uint8 RGB image
    depth_map: HxW float32 in [0,1]
    method: 'none' | 'bilateral' | 'guided'
    """
    if method == 'none':
        return depth_map

    img = image_rgb.astype(np.uint8)
    d = depth_map.astype(np.float32)

    if method == 'bilateral':
        # Scale sigmaColor to 0-255 space then map back
        # cv2 bilateral works on single-channel 8U or float; we use float but sigmaColor in same units as values
        sigma_color = max(1e-6, float(bilateral_sigma_color))  # assumes depth normalized [0,1]
        sigma_space = max(0.1, float(bilateral_sigma_space))
        # Use diameter in pixels (odd). If d<=0, OpenCV computes from sigma_space; we set reasonable default
        filt = cv2.bilateralFilter(d, d=int(bilateral_d), sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filt

    if method == 'guided':
        # Try to use OpenCV ximgproc guided filter
        try:
            import cv2.ximgproc as xip
            guide = img
            # ximgproc.guidedFilter expects guide 8U/32F 1-3ch, src 8U/32F 1ch
            gf = xip.guidedFilter(guide=guide, src=d, radius=int(guided_radius), eps=float(guided_eps))
            return gf.astype(np.float32)
        except Exception as e:
            warnings.warn(f"Guided filter unavailable ({e}); falling back to bilateral filter.")
            sigma_color = max(1e-6, float(bilateral_sigma_color))
            sigma_space = max(0.1, float(bilateral_sigma_space))
            return cv2.bilateralFilter(d, d=int(bilateral_d), sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return depth_map

def visualize_3d(points, colors, output_path='point_cloud.ply'):
    try:
        import open3d as o3d
    except Exception as e:
        print("Open3D is not available. Skipping 3D visualization. Install with: pip install open3d")
        return
    
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

def compute_metrics(pred_depth, gt_depth):
    """
    Compute basic depth estimation metrics given predicted depth in [0,1]
    and ground-truth depth (any positive scale). The gt will be resized to match pred.
    Returns a dict of metrics.
    """
    # Ensure same size
    ph, pw = pred_depth.shape
    gh, gw = gt_depth.shape
    if (ph, pw) != (gh, gw):
        gt_depth = cv2.resize(gt_depth, (pw, ph), interpolation=cv2.INTER_NEAREST)

    # Mask invalid gt
    mask = np.isfinite(gt_depth) & (gt_depth > 0)
    if mask.sum() == 0:
        return {"valid_pixels": 0}

    pd = pred_depth[mask].astype(np.float32)
    gd = gt_depth[mask].astype(np.float32)

    # Align scales by median ratio (simple robust alignment)
    scale = np.median(gd) / max(np.median(pd), 1e-6)
    pd_s = pd * scale

    # Errors
    abs_rel = np.mean(np.abs(pd_s - gd) / np.maximum(gd, 1e-6))
    rmse = np.sqrt(np.mean((pd_s - gd) ** 2))
    mae = np.mean(np.abs(pd_s - gd))

    # Delta thresholds
    ratio = np.maximum(pd_s / np.maximum(gd, 1e-6), gd / np.maximum(pd_s, 1e-6))
    d1 = np.mean(ratio < 1.25)
    d2 = np.mean(ratio < 1.25 ** 2)
    d3 = np.mean(ratio < 1.25 ** 3)

    return {
        "valid_pixels": int(mask.sum()),
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "mae": float(mae),
        "delta1": float(d1),
        "delta2": float(d2),
        "delta3": float(d3),
        "scale": float(scale),
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert 2D image to 3D point cloud')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output.ply', help='Output PLY file path')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for depth')
    parser.add_argument('--gt', type=str, default=None, help='Optional ground-truth depth image (single-channel). Used to compute metrics.')
    parser.add_argument('--model', type=str, default='MiDaS_small', choices=['DPT_Large','DPT_Hybrid','MiDaS_small'], help='MiDaS model to use')
    parser.add_argument('--refine', type=str, default='none', choices=['none','bilateral','guided'], help='Edge-aware refine method')
    parser.add_argument('--refine_bilateral_d', type=int, default=9, help='Bilateral diameter (pixels)')
    parser.add_argument('--refine_bilateral_sigma_color', type=float, default=0.1, help='Bilateral sigmaColor (depth units, ~0-1)')
    parser.add_argument('--refine_bilateral_sigma_space', type=float, default=5.0, help='Bilateral sigmaSpace (pixels)')
    parser.add_argument('--refine_guided_radius', type=int, default=8, help='Guided filter radius (pixels)')
    parser.add_argument('--refine_guided_eps', type=float, default=1e-3, help='Guided filter eps')
    
    args = parser.parse_args()
    
    # Initialize depth estimator
    print("Initializing depth estimator...")
    depth_estimator = DepthEstimator(model_type=args.model)
    
    # Process image
    print(f"Processing image: {args.input}")
    image, depth_map = depth_estimator.process_image(args.input)
    # Optional edge-aware refinement
    if args.refine != 'none':
        print(f"Applying {args.refine} refinement...")
        depth_map = refine_depth_edge_aware(
            image,
            depth_map,
            method=args.refine,
            bilateral_d=args.refine_bilateral_d,
            bilateral_sigma_color=args.refine_bilateral_sigma_color,
            bilateral_sigma_space=args.refine_bilateral_sigma_space,
            guided_radius=args.refine_guided_radius,
            guided_eps=args.refine_guided_eps,
        )
    print(f"Depth map stats -> min: {depth_map.min():.3f}, max: {depth_map.max():.3f}, mean: {depth_map.mean():.3f}")
    
    # Create 3D point cloud
    print("Generating 3D point cloud...")
    points, colors = create_3d_point_cloud(image, depth_map, scale=args.scale)
    
    # Visualize and save results
    print("Saving results...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Depth Map')
    plt.imshow(depth_map, cmap='viridis')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('3D Projection')
    plt.imshow(depth_map, cmap='inferno')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_visualization.png')
    print("Saved depth visualization to depth_visualization.png")

    # If ground-truth provided, compute metrics
    if args.gt is not None:
        print(f"Loading ground-truth depth: {args.gt}")
        gt_img = Image.open(args.gt)
        gt_np = np.array(gt_img).astype(np.float32)
        if gt_np.ndim == 3:
            # If RGB depth image, convert to grayscale
            gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)
        metrics = compute_metrics(depth_map, gt_np)
        if metrics.get("valid_pixels", 0) > 0:
            print("Evaluation metrics (after median scaling):")
            print(f"  valid_pixels: {metrics['valid_pixels']}")
            print(f"  AbsRel: {metrics['abs_rel']:.4f}")
            print(f"  RMSE:   {metrics['rmse']:.4f}")
            print(f"  MAE:    {metrics['mae']:.4f}")
            print(f"  delta1: {metrics['delta1']:.4f}")
            print(f"  delta2: {metrics['delta2']:.4f}")
            print(f"  delta3: {metrics['delta3']:.4f}")
        else:
            print("No valid pixels found in ground-truth for evaluation.")
    
    # Create and save 3D point cloud
    visualize_3d(points, colors, args.output)
    print("Done!")

if __name__ == "__main__":
    main()
