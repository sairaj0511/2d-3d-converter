import os
import urllib.request

def download_sample_image():
    """Download a sample image for testing."""
    urls = [
        "https://raw.githubusercontent.com/intel-isl/MiDaS/master/input/bridge.jpg",
        "https://raw.githubusercontent.com/intel-isl/MiDaS/master/input/parthenon.jpg",
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/example.jpg",
    ]
    output_path = "example.jpg"

    if os.path.exists(output_path):
        print(f"Sample image already exists at {output_path}")
        return output_path

    last_err = None
    for url in urls:
        try:
            print(f"Downloading sample image from: {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"Sample image saved as {output_path}")
            return output_path
        except Exception as e:
            last_err = e
            print(f"Failed to download from this URL: {e}")

    raise RuntimeError(f"Failed to download sample image from all sources: {last_err}")

if __name__ == "__main__":
    # Download sample image
    image_path = download_sample_image()
    
    # Run the 2D to 3D conversion
    print("\nRunning 2D to 3D conversion...")
    os.system(f"python 2d_to_3d.py --input {image_path} --output output_3d.ply")
    
    print("\nProcessing complete! Check the following files:")
    print("- depth_visualization.png: 2D visualization of the depth map")
    print("- output_3d.ply: 3D point cloud file (open with a 3D viewer)")
