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

## Developer notes: nD GRAPPA utilities

The package exposes a small MD-GRAPPA engine in `mocokit.nD_grappa` with two
primary entrypoints useful for programmatic reconstruction and tests:

- `train_kernels_streaming(calib, kernel_size, patterns, lamda=0.01, use_gpu=True, batch_rows=4096, dtype='complex64')`
    - Trains pattern-specific GRAPPA kernels from ACS data in a streaming fashion
        (low memory). `calib` is a fully-sampled block with shape `(...spatial..., nc)`.
    - `patterns` can be the keys returned by `build_patterns`.

- `apply_mdgrappa(kspace, calib, kernel_size, lamda=0.01, nnz_min=None, coil_axis=-1, use_gpu=True, batch_rows=4096, dtype='complex64', apply_on_gpu=None)`
    - Applies trained kernels to fill missing k-space samples.
    - `apply_on_gpu`: controls where the filling is done:
        - `None` (default) — automatic decision based on an estimate of the padded
            k-space memory footprint and current free GPU memory. This is the safest
            option to avoid OOM on GPU.
        - `False` — force CPU application (default-safe path, preserves VRAM).
        - `True` — force GPU application (faster only if VRAM is sufficient).

Recommendations and notes
- By default the library preserves GPU VRAM and performs the application on CPU
    unless the padded k-space comfortably fits in free GPU memory. This behaviour
    avoids unexpected OOM when processing large datasets.
- If you have a GPU with a lot of free memory and prefer speed, pass
    `apply_on_gpu=True` to `apply_mdgrappa`. Monitor GPU memory usage.
- For reproducible unit tests, the repository includes a CPU-only test that
    runs even without CuPy/GPU. A GPU test is available but will be skipped when
    CuPy or a CUDA device is missing.

Running the GPU test (if you have CuPy + CUDA available):

```bash
# from repo root
python3 -m pytest -q tests/test_nd_grappa_gpu.py
```
