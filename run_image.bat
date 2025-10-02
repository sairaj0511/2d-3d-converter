@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Usage: Drag-and-drop an image onto this .bat, or run: run_image.bat path\to\image.jpg
if "%~1"=="" (
  echo [Usage]
  echo   Drag and drop an image onto this file, or run:
  echo   run_image.bat ^<path\to\image.jpg^>
  pause
  exit /b 1
)

REM Check file exists
if not exist "%~1" (
  echo File not found: %~1
  pause
  exit /b 1
)

REM Activate venv if available
if exist ".\.venv\Scripts\activate.bat" (
  call ".\.venv\Scripts\activate.bat"
) else (
  echo WARNING: .venv not found. Attempting to use system Python.
)

REM Run conversion with MiDaS_small and name output based on input
set INPUT=%~1
set NAME=%~n1
python 2d_to_3d.py --input "%INPUT%" --output "%NAME%_3d.ply" --model MiDaS_small
if errorlevel 1 (
  echo Conversion failed.
  pause
  exit /b 1
)

echo.
echo Done.
echo Outputs:
if exist depth_visualization.png echo  - depth_visualization.png
if exist "%NAME%_3d.ply" echo  - %NAME%_3d.ply
pause
