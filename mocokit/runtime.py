import os
import numpy as np
import matplotlib

def configure_runtime(
    cuda_visible_devices    : str | None = None,
    headless                : bool | None = None,
    numpy_precision         : int = 6,
    numpy_suppress          : bool = True,
) -> None:
    
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    if headless is None:
        headless = not os.environ.get("DISPLAY")
        
    if headless:
        matplotlib.use("Agg") 

    np.set_printoptions(precision=numpy_precision, suppress=numpy_suppress)