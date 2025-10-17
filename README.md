# mocokit

MRI reconstruction from Siemens `.dat` files with motion correction.
Includes: noise prewhitening, iPAT/GRAPPA, non-Cartesian gridding (KbNUFFT), PF/POCS, coil combination, NIfTI export.

## Installation

### Option 1: conda (recommended)
```bash
conda env create -f environment.yml
conda activate moco
pip install -e .
```

### Option 2: pip
```bash
pip install -e .

# For GPU
pip install -r requirements-gpu.txt
pip install -e ".[gpu]"
```

## Usage

Basic usage:
```bash
mocokit -i /path/to/folder/dat \
    -tcl -td /path/to/tcl_dir \
    -reverse -smooth \
    -orig -center \
    -device cuda:0 \
    --cuda-visible-devices 0 \
    --headless \
    --numpy-precision 6 \
    -v
```

### Parameters

- `-i`: Input directory containing `.dat` files
- `-tcl`: Enable TCL processing
- `-td`: TCL directory path
- `-reverse`: Reverse motion correction
- `-smooth`: Apply smoothing
- `-orig`: Use original coordinates
- `-center`: Center reconstruction
- `-device`: Specify compute device (e.g., `cuda:0`, `cpu`)
- `--cuda-visible-devices`: Set visible CUDA devices
- `--headless`: Run without GUI
- `--numpy-precision`: Set numerical precision
- `-v`: Verbose output

## Requirements

- Python 3.10+
- CUDA-compatible GPU (optional)
- Required packages listed in `environment.yml`

