## ipat.py
import numpy as np
import logging
from time import time
import os
import gc

from .mtypes import DatBasics
from .utils import saveImage




def find_acs(kspace : np.ndarray, ctrs : np.ndarray = None, coil_axis : int = -1):
    '''Find the largest centered hyper-rectangle possible.

    Parameters
    ----------
    kspace : array_like
        Measured undersampled complex k-space data. N-1 dimensions
        hold spatial frequency axes (kx, ky, kz, etc.).  1 dimension
        holds coil images (`coil_axis`).  The missing entries should
        have exactly 0.
    coil_axis : int, optional
        Dimension holding coil images.

    Returns
    -------
    calib : array_like
        Fully sampled calibration data extracted from the largest
        possible hypercube with origin at the center of k-space.

    Notes
    -----
    This algorithm is not especially elegant, but works just fine
    with the assumption that the ACS region will be significantly
    smaller than the entirety of the data.  It grows a hyper-
    rectangle from the center and checks to see if there are any
    new holes in the region each time it expands.
    '''
    ## pygrappa function with minor modification

    kspace  = np.moveaxis(kspace, coil_axis, -1)
    mask    = np.abs(kspace[..., 0]) > 0

    # Start by finding the largest hypercube
    if ctrs is None:
        ctrs = [d // 2 for d in mask.shape]  # assume ACS is at center
    
    slices  = [[c, c+1] for c in ctrs]  # start with 1 voxel region
    t0      = time()
    while (all(l > 0 and r < mask.shape[ii] for
               ii, (l, r) in enumerate(slices)) and  # bounds check
           np.all(mask[tuple([slice(l-1, r+1) for
                              l, r in slices])])):  # hole check
        
        # expand isotropically until we can't no more
        slices = [[l0-1, r0+1] for l0, r0 in slices]
    logging.info('Took %g sec to find hyper-cube', (time() - t0))

    # Stretch left/right in each dimension
    t0 = time()
    for dim in range(mask.ndim):
        # left: only check left condition on the current dimension
        while (slices[dim][0] > 0 and
               np.all(mask[tuple([slice(l-(dim == k), r) for
                                  k, (l, r) in enumerate(slices)])])):
            slices[dim][0] -= 1
        # right: only check right condition on the current dimension
        while (slices[dim][1] < mask.shape[dim] and
               np.all(mask[tuple([slice(l, r+(dim == k)) for
                                  k, (l, r) in enumerate(slices)])])):
            slices[dim][1] += 1
    logging.info('Took %g sec to find hyper-rect', (time() - t0))

    return np.moveaxis(
        kspace[tuple([slice(l0, r0) for l0, r0 in slices] +
                     [slice(None)])].copy(),  # extra dim for coils
        -1, coil_axis)


def prepare_acs(dat_basics : DatBasics, data_var : np.ndarray, iAve : int = 0) -> np.ndarray:
    """
    Function to prepare the ACS data for Grappa calibration
    dat_basics  : DatBasics object containing all the necessary information
    data_var    : kspace data variable to work on
    """
    Acs     = None

    if dat_basics.ipatref:
        
        logging.info("Using embedded ACS for Grappa calibration ! ")
        ## Only using the first average for ACS
        ## TODO: maybe in future implemente a test on motion score to choose the best average for ACS

        mask    = np.where(dat_basics.dims_dict['cAve'][dat_basics.ipatref]==iAve)
        acs_lin = dat_basics.dims_dict['cLin'][mask]
        acs_par = dat_basics.dims_dict['cPar'][mask]

        sz_lin  = max(acs_lin) - min(acs_lin) + 1
        sz_par  = max(acs_par) - min(acs_par) + 1

        Acs     = np.zeros((*data_var.shape[:2], sz_lin, sz_par), dtype=data_var.dtype)

        Acs[..., acs_lin - min(acs_lin), acs_par - min(acs_par)] = data_var[..., acs_lin, acs_par, iAve]

        logging.info("ACS shape before cropping: {}".format(Acs.shape))

    else:

        Acs     = find_acs(data_var[..., iAve], dat_basics.kcent, coil_axis=1)
        logging.info("ACS shape before cropping: {}".format(Acs.shape))

    if Acs is None:
        raise ValueError("No embedded ACS found in the data ! Grappa cannot be applied ! \nThe use of external ACS not implemented yet !")

    Acs         = np.moveaxis(Acs, 1, -1)

    if dat_basics.max_calib_shape:
        calib_ctr       = [int(k/2) for k in Acs.shape]
        mcalib_shape    = [min(m, c) for m, c in zip(dat_basics.max_calib_shape, Acs.shape)]
        slices          = [slice(c - int(np.ceil(m/2)), c + int(np.floor(m/2))) for c, m in zip(calib_ctr, mcalib_shape)]
        Acs             = Acs[tuple(slices + [slice(None)])]
    
    return np.moveaxis(Acs, -1, 1)


def parallel_imaging(dat_basics : DatBasics, 
                     working_path : str, 
                     is_nomoco : bool = False,
                     use_multithreading: bool = True,
                     max_workers: int = 12) -> None:
    """
    Function to handle parallel imaging using mdgrappa from sigpy package
    dat_basics  : DatBasics object containing all the necessary information
    working_path: path to the folder containing the .dat file
    is_nomoco   : whether to use the nomoco kspace or not
    """

    logging.info("Parallel imaging found in at least one encoding direction! Grappa will be used to fill 0 kspace elements!")

    ## TODO: multithreading grappa implementation is not stable yet --> forcing single-threading
    use_multithreading = False
    
    if use_multithreading:
        logging.info("Using multithreaded Grappa implementation ! ")
        from .modified_mdgrappa import mdgrappa as grappa_func

    else:
        from pygrappa import mdgrappa as grappa_func
    


    ## TODO: Optimize the usage of memory in this function
    
    if not is_nomoco:
        acs_kern_name   = "Ipat_kernel_ws.npy"
        dname           = "filled_ksp.npy"
        data_var        = dat_basics.data_set

    else:
        acs_kern_name   = "Ipat_kernel_ws_nomoco.npy"
        dname           = "filled_ksp_nomoco.npy"

        assert dat_basics.nomoco_ksp is not None, "No nomoco kspace found !"
        data_var        = dat_basics.nomoco_ksp

    ## Prepare ACS data
    Acs     = prepare_acs(dat_basics, data_var, iAve=0)

    ## ensure data used for ipat is not including PF 0 lines (that will be handled after moco)
    min_lin = min(dat_basics.dims_dict['cLin']) - dat_basics.Arps[1] + 1
    max_lin = max(dat_basics.dims_dict['cLin']) + dat_basics.Arps[1]
    min_par = min(dat_basics.dims_dict['cPar']) - dat_basics.Arps[2] + 1
    max_par = max(dat_basics.dims_dict['cPar']) + dat_basics.Arps[2]

    logging.info("Working on base dataset")
    Kern_Ws             = None

    if os.path.exists(os.path.join(working_path, acs_kern_name)):
        Kern_Ws         = np.load(os.path.join(working_path, acs_kern_name), allow_pickle='TRUE').item()

    func_args = {
        'kspace'        : data_var[..., min_lin:max_lin, min_par:max_par, 0],
        'calib'         : Acs,
        'kernel_size'   : (3, 3, 3), #(3,)*(data_var.ndim-1),
        'coil_axis'     : 1, 
        'ret_weights'   : True,
        'weights'       : Kern_Ws}
    
    if use_multithreading:
        func_args['max_workers'] = max_workers

    data_var[..., min_lin:max_lin, min_par:max_par, 0], Kern_Ws = grappa_func(**func_args)
    
    ## This task is time consuming even using cython, so save the kernel weights for future use
    if not os.path.exists(os.path.join(working_path, acs_kern_name)):
        logging.info("Grappa kernel weights will be saved as {}".format(acs_kern_name))
        np.save(os.path.join(working_path, acs_kern_name), Kern_Ws)


    if dat_basics.nAve > 1:
        ## ensure data used for ipat is not including PF 0 lines (that will be handled after moco)

        Avemask = np.where(dat_basics.dims_dict['cAve'] == 1)

        min_lin = min(dat_basics.dims_dict['cLin'][Avemask]) - dat_basics.Arps[1] + 1
        max_lin = max(dat_basics.dims_dict['cLin'][Avemask]) + dat_basics.Arps[1]
        min_par = min(dat_basics.dims_dict['cPar'][Avemask]) - dat_basics.Arps[2] + 1
        max_par = max(dat_basics.dims_dict['cPar'][Avemask]) + dat_basics.Arps[2]

        rKern_Ws         = None
        if os.path.exists(os.path.join(working_path, "r"+acs_kern_name)):
            rKern_Ws     = np.load(os.path.join(working_path, "r"+acs_kern_name), allow_pickle='TRUE').item()

        func_args = {
        'kspace'        : data_var[..., min_lin:max_lin, min_par:max_par, 1],
        'calib'         : Acs,
        'kernel_size'   : (3, 3, 3), #(3,)*(data_var.ndim-1),
        'coil_axis'     : 1,
        'ret_weights'   : True,
        'weights'       : rKern_Ws}

        if use_multithreading:
            func_args['max_workers'] = max_workers

        test = grappa_func(**func_args)
        data_var[..., min_lin:max_lin, min_par:max_par, 1], rKern_Ws = test

        if not os.path.exists(os.path.join(working_path, "r"+acs_kern_name)):
            logging.info("Grappa kernel weights will be saved as {}".format("r"+acs_kern_name))
            np.save(os.path.join(working_path, "r"+acs_kern_name), rKern_Ws)

    data_var = data_var.astype(np.complex64)

    logging.info("Filled kspace will be saved as {}".format(dname))
    np.save(os.path.join(working_path, dname), data_var)

    del Acs, Kern_Ws
    gc.collect()

    ## Save the filled kspace image
    coil_idx    = min(dat_basics.nC - 1, 5)  # just to have a look at one coil
    saveImage(np.abs(data_var[:, coil_idx, ..., 0]).transpose(dat_basics.dimspermute), 
              dat_basics.affinematrix, 
              os.path.join(working_path, f"""filled_ksp.nii"""))
    
    ## Update dat_basics
    ## Return trigger when done with mention if nomoco or not
    if not is_nomoco:
        dat_basics.data_set     = data_var
        return "moco"
    else:
        dat_basics.nomoco_ksp   = data_var
        return "nomoco"