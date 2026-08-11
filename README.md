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
- `-nthreads` : Set number of threads to use in GRAPPA reconstruction (default: 1)
- `--headless`: Run without GUI
- `--numpy-precision`: Set numerical precision (default: 6)
- `-v`: Verbose output

## Requirements

- Python 3.10+
- CUDA-compatible GPU (optional)
- Required packages listed in `environment.yml`

## Citation
If you use this repository, please cite:

> Z. Zariry, F. Lamberton, R. Frost, et al., “ Intra-MRI Head Motion Tracking and Correction: A Quantitative In Vivo Evaluation Framework,”
> NMR in Biomedicine 39, no. 9 (2026): e70368, https://doi.org/10.1002/nbm.70368.

