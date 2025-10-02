import os
import sys
import subprocess
from tkinter import Tk, filedialog, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, '.venv', 'Scripts', 'python.exe')
SCRIPT = os.path.join(HERE, '2d_to_3d.py')

def main():
    root = Tk()
    # Make dialog appear on top
    try:
        root.lift()
        root.attributes('-topmost', True)
    except Exception:
        pass
    root.withdraw()
    path = filedialog.askopenfilename(
        title='Select an image',
        filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
    )
    # Remove topmost so message boxes do not stay forced on top
    try:
        root.attributes('-topmost', False)
    except Exception:
        pass

    if not path:
        # Fallback to console prompt (useful when GUI is blocked by environment)
        try:
            path = input('Enter full path to an image (or press Enter to cancel): ').strip('"')
        except Exception:
            path = ''
        if not path:
            print('No file selected.')
            try:
                messagebox.showinfo('2D to 3D', 'No file selected.')
            except Exception:
                pass
            return

    if not os.path.exists(PY):
        # Fallback to system python
        PY_EXEC = sys.executable
    else:
        PY_EXEC = PY

    name, _ = os.path.splitext(os.path.basename(path))
    out = os.path.join(HERE, f'{name}_3d.ply')

    cmd = [PY_EXEC, SCRIPT, '--input', path, '--output', out, '--model', 'MiDaS_small', '--refine', 'guided']
    print('Running:', ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
    try:
        rc = subprocess.call(cmd, cwd=HERE)
    except FileNotFoundError:
        rc = 2
    if rc != 0:
        try:
            messagebox.showerror('2D to 3D', f'Conversion failed with code {rc}.\nCheck that the image path exists and is accessible.')
        except Exception:
            pass
    else:
        msg = f'Done!\n\nOutputs:\n - depth_visualization.png\n - {out}'
        try:
            messagebox.showinfo('2D to 3D', msg)
        except Exception:
            print(msg)

if __name__ == '__main__':
    main()
