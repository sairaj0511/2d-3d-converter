# 2D to 3D Image Conversion

This project converts 2D images into 3D point clouds using deep learning. It leverages the MiDaS (Mixed Dataset) model for monocular depth estimation and generates a 3D point cloud that can be visualized or used for further processing.

## Features

- Depth map estimation from a single 2D image
- 3D point cloud generation
- Interactive 3D visualization
- Support for various image formats
- GPU acceleration (if available)

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Open3D (for 3D visualization)
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 2d-to-3d-conversion
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Basic usage:
   ```bash
   python 2d_to_3d.py --input path/to/your/image.jpg --output output.ply
   ```

2. Adjust the depth scale (default: 1.0):
   ```bash
   python 2d_to_3d.py --input image.jpg --output output.ply --scale 0.5
   ```

## Output

- `depth_visualization.png`: Shows the original image, depth map, and 3D projection
- `output.ply`: 3D point cloud file that can be viewed in any 3D viewer

## Example

```bash
python 2d_to_3d.py --input example.jpg --output example_3d.ply
```

## Notes

- For best results, use high-quality images with clear foreground and background separation.
- The processing time depends on your hardware and image size.
- For better performance, a CUDA-enabled GPU is recommended.

## License

This project is open source and available under the MIT License.
## image running
Here are the simplest one‑liners. Pick the one that fits your case.

If your image is in the project folder (e.g., human.jpg):
powershell
.\.venv\Scripts\python.exe 2d_to_3d.py --input human.jpg --output human_3d.ply --model MiDaS_small --refine guided

If your image is anywhere else (absolute path, quotes handle spaces):
powershell
.\.venv\Scripts\python.exe 2d_to_3d.py --input "C:\full\path\to\your\photo.jpg" --output human_3d.ply --model MiDaS_small --refine guided

With Windows file picker in one line (no path typing):
powershell
Add-Type -AssemblyName System.Windows.Forms; $dlg=New-Object System.Windows.Forms.OpenFileDialog; $dlg.Filter="Images|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"; if($dlg.ShowDialog() -eq "OK"){ .\.venv\Scripts\python.exe 2d_to_3d.py --input "$($dlg.FileName)" --output human_3d.ply --model MiDaS_small --refine guided }
Using the helper batch (activates venv automatically):
powershell
.\run_image.bat "C:\full\path\to\your\photo.jpg"

depth_visualization.png
human_3d.ply (view in MeshLab/Blender)
Feedback submitted

TO RUN THE APP (Streamlit)

Proposed Commands (Windows PowerShell)

Create venv (Python 3.11 recommended):

```powershell
py -3.11 -m venv .venv
```

Activate venv:

```powershell
\.\.venv\Scripts\Activate.ps1
```

Upgrade installer tooling (recommended):

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
