from dataclasses import dataclass
import numpy as np


PF_MAPPING      = {1: 0.5, ## 4/8
                    2: 0.625, ## 5/8
                    4: 0.75, ## 6/8
                    8: 0.875 ## 7/8
                    }

ASYM_ECHO       = {0: 1.0, 
                    1: 0.0}

LPS_TO_RAS = np.diag([-1., -1., 1., 1.])


@dataclass
class DatBasics:
    eHrps           : np.ndarray
    # eFOVrps       = np.ndarray
    rHrps           : np.ndarray
    rFOVrps         : np.ndarray
    nC              : int
    nAve            : int
    hostVox_mm      : np.ndarray
    kcent           : np.ndarray
    pf              : np.ndarray
    Arps            : np.ndarray
    affinematrix    : np.ndarray
    data_set        : np.ndarray
    dims_dict       : np.ndarray
    sPosition_mm    : np.ndarray
    sPosition_vox   : np.ndarray
    dimspermute     : np.ndarray
    FOV_rot         : np.ndarray
    ticks           : np.ndarray
    ticks_formated  : np.ndarray
    max_calib_shape : np.ndarray
    ksp_exist       : dict
    ipatref         : list | None = None
    moco_sys        : dict | None = None
    separate_nomoco : bool | None = False
    nomoco_ksp      : np.ndarray | None = None

    