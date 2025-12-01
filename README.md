# pyMocoKit

MRI reconstruction pipeline with retrospective motion correction and option to reverse a prospectively applied correction.
Supports raw data (Siemens `.dat` files) from MPRAGE and T2-SPACE sequences (an extension to diffusion EPI support is planned).

Includes: noise prewhitening, OS removal, iPAT/GRAPPA, non-Cartesian gridding (KbNUFFT), PF/POCS, coil combination, NIfTI export.

## Installation

### Option 1: conda
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
- `-orig`: Use original kspace (if -reverse option is used and reacquisition data exist, they'll be used!)
- `-orig_noreacq`: Use original kspace (without reacquisition data; -reverse and -orig will be set to True!)
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

## Citation
If you use this repository, please cite:

> Zariry Z, Lamberton F, Frost R, Gaass T, Troalen T, Rayson H, Slipsager JM, Richard N, van der Kouwe A, Bonaiuto J, Hiba B. *An in-vivo approach to quantify intra-MRI head motion tracking accuracy: comparison of markerless optical tracking versus fat-navigators.*
> **medRxiv** [Preprint]. 2025 Jul 17:2025.04.23. 
> DOI: [10.1101/2025.04.23.25326185] (https://www.medrxiv.org/content/10.1101/2025.04.23.25326185v2)

