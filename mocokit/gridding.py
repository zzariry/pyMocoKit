import numpy as np
import logging
from time import time
from tqdm import tqdm
import gc
import math

import torch
import torchkbnufft
from torchkbnufft._nufft.fft import ifft_and_scale

from .mtypes import DatBasics
from .coils import coil_combine
from .pf_handling import POCS




def correct_kspace_and_save(
        dat_basics    : DatBasics,
        var           : str,
        working_path  : str,
        is_nomoco     : bool = False,
        no_reacq      : bool = False,
        use_pocs      : bool = True,
        device        : str = 'cuda:0'
        ) -> None:
    
    """ Correct Translation and rotation on Fourier Space
    dat_basics  : DatBasics object containing all the necessary information
    var         : 'data_set' or 'nomoco_ksp' to select the kspace to correct
    working_path: path to the folder containing the .dat file
    is_nomoco   : whether to use the nomoco kspace or not
    device      : device to use for torch computations
    """
    if is_nomoco and not no_reacq:
        data = dat_basics.nomoco_ksp.copy()

    elif is_nomoco and no_reacq:
        data = dat_basics.data_set.copy()

    else:
        data = dat_basics.data_set.copy()

    if not (data.ndim == 4 or data.ndim == 5):
        raise ValueError("Only 3D or 4D data are handled for now (+ 1D for coils) !")

    rHrps           = dat_basics.rHrps
    hostVox_mm      = dat_basics.hostVox_mm
    mvt_matrices    = dat_basics.moco_sys[var]

    ndim            = len(rHrps) ## not considering coil or ave dimension
    # D_rps           = np.asarray(data[:, 0, ...].shape, dtype=np.uint16)

    kspaceCenter    = rHrps // 2 ## assume even dimensions with 0 filling if PF

    kx, ky, kz      = np.indices(rHrps, np.single)-kspaceCenter[:, None, None, None]

    kx = 2*kx/rHrps[0]/hostVox_mm[0]
    ky = 2*ky/rHrps[1]/hostVox_mm[1]
    kz = 2*kz/rHrps[2]/hostVox_mm[2]

    Lin_, Par_ = map(lambda x: x.ravel(order='C'),
                     np.meshgrid(*[np.arange(rHrps[d], dtype=np.int16) for d in range(1, ndim)], indexing="ij"))
    
    ## All good, lets go 
    t0 = time()

    ## Create nkx, nky, nkz to not modify original kx, ky, kz
    ## Take in account nAve dimension

    nkx = kx.copy()
    nky = ky.copy()
    nkz = kz.copy()

    # if dat_basics.nAve > 1:
    nkx = np.repeat(nkx[..., np.newaxis], dat_basics.nAve, axis=-1)
    nky = np.repeat(nky[..., np.newaxis], dat_basics.nAve, axis=-1)
    nkz = np.repeat(nkz[..., np.newaxis], dat_basics.nAve, axis=-1)

    logging.info("Computing Phase Translation and applying to kspace ...")

    for iAve in range(dat_basics.nAve):

        PhaseTranslation    = np.empty_like(data[:, 0, ..., iAve],)

        for step_y, step_z in zip(Lin_, Par_):
            ## Base data
            newVec = np.array(mvt_matrices[step_y, step_z, iAve, :3, :3] @\
                [kz[:, step_y, step_z], ky[:, step_y, step_z], kx[:, step_y, step_z]]).T

            nkz[:, step_y, step_z, iAve] = newVec[:, 0]
            nky[:, step_y, step_z, iAve] = newVec[:, 1]
            nkx[:, step_y, step_z, iAve] = newVec[:, 2]

            PhaseTranslation[..., step_y, step_z] = np.exp(-1j*np.pi*(
                mvt_matrices[step_y, step_z, iAve, 0, 3]*kz[:, step_y, step_z] +
                mvt_matrices[step_y, step_z, iAve, 1, 3]*ky[:, step_y, step_z] +
                mvt_matrices[step_y, step_z, iAve, 2, 3]*kx[:, step_y, step_z]))

        ## Apply PhaseTranslation on data
        ## NOTE: X axis must be reversed
        data[..., iAve]    = np.einsum('icjk, ijk -> icjk', data[::-1, ..., iAve], PhaseTranslation)

    logging.info("Phase Translation applied, duration = %d secondes", (time()-t0))

    del PhaseTranslation, kx, ky, kz

    logging.info("Preparing interpolation modules  ...")

    ### Initiate interpolation operators
    oversamp    = 2
    kbwidth     = 6
    adjinterp   = torchkbnufft.KbInterpAdjoint(rHrps, rHrps*oversamp, device=device, )
    
    Lin = dat_basics.dims_dict['cLin']
    Par = dat_basics.dims_dict['cPar']

    ## We assume here that maximum cAve len is 2
    if dat_basics.nAve > 2:
        raise NotImplementedError("Only nAve=2 is handled for now !")
    
    om      = []
    dcf     = []
    mask    = []

    for iAve in range(dat_basics.nAve):
        maskAve             = np.where(dat_basics.dims_dict['cAve'] == iAve)

        ### Create grid coords for base data
        om_nonuniform       = np.hstack([f.reshape((-1, 1), order="C") for f in [nkx[..., iAve]*hostVox_mm[0],
                                                                                 nky[..., iAve]*hostVox_mm[1],
                                                                                 nkz[..., iAve]*hostVox_mm[2]]])*np.pi
        # omega_nonuniform    = torch.from_numpy(om_nonuniform.T).to(device)

        om.append(torch.from_numpy(om_nonuniform.T).to(device))

        del om_nonuniform

        ## Density compensation
        logging.info("Computing density compensation fonction - Average {}...".format(iAve))

        tmp_dcf         = torchkbnufft.calc_density_compensation_function(om[-1], 
                                                            rHrps,
                                                            num_iterations=10,
                                                            grid_size=rHrps*oversamp,
                                                            numpoints=6, 
                                                            n_shift=dat_basics.kcent).to('cpu').numpy()

        dcf.append(tmp_dcf)

        min_Lin  = np.min(Lin[maskAve])*oversamp if Lin[maskAve] is not None else 0
        min_Par  = np.min(Par[maskAve])*oversamp if Par[maskAve] is not None else 0
        max_Lin  = np.max(Lin[maskAve])*oversamp if Lin[maskAve] is not None else (rHrps[1]-1)*oversamp
        max_Par  = np.max(Par[maskAve])*oversamp if Par[maskAve] is not None else (rHrps[2]-1)*oversamp

        tmp_mask     = np.zeros(rHrps*oversamp, dtype=bool)
        tmp_mask[..., min_Lin:max_Lin+1, min_Par:max_Par+1] = True

        # tmp_mask     = torch.from_numpy(tmp_mask).to(device)
        mask.append(torch.from_numpy(tmp_mask).to(device))
    
    del nkx, nky, nkz

    scal_coef   = torchkbnufft.nufft_utils.compute_scaling_coefs(rHrps, rHrps*oversamp, numpoints=[6]*3, alpha=[2.34*6]*3, order=[0]*3).to(device)

    t0          = time()

    im_data     = np.zeros_like(data[..., 0], dtype=np.complex64)

    logging.info("Start regriding on each coil kspace ")
    for iC in tqdm(range(dat_basics.nC), desc="Working on coils", ascii=False, total=dat_basics.nC):

        tmp = adjinterp(torch.from_numpy(data[:, iC, ..., 0].flatten()[np.newaxis, np.newaxis, ...]*dcf[0]).to(device), om[0])

        ## Combine the two averages if exist
        if dat_basics.nAve > 1:
            tmp_1       = adjinterp(torch.from_numpy(data[:, iC, ..., 1].flatten()[np.newaxis, np.newaxis, ...]*dcf[1]).to(device), om[1])

            ## Handle phase difference between the two averages before combining them
            phase_diff  = torch.angle(torch.vdot(tmp.flatten(), tmp_1.flatten()))
            tmp_1       = tmp_1*torch.exp(-1j*phase_diff)

            tmp_1       = torch.fft.ifftshift(tmp_1, )
            tmp         = torch.fft.ifftshift(tmp, )

            ## mask applied after kspace shift
            tmp[..., mask[-1]] += tmp_1[..., mask[-1]]
            tmp[..., mask[-1]] /= 2
            tmp[..., ~mask[0]] = 0
        
        else:
            tmp = torch.fft.ifftshift(tmp, )
            tmp[..., ~mask[0]] = 0

        ## divide by the difference
        if dat_basics.pf[1] != 1 and use_pocs:
            tmp = POCS(tmp, dim_pf=3, dim_enc=[2, 3, 4], number_of_iterations=7,) #

        if dat_basics.pf[2] != 1 and use_pocs:
            tmp = POCS(tmp, dim_pf=4, dim_enc=[2, 3, 4], number_of_iterations=7,) # 

        # tmp *= gaussian_window
        tmp = torch.fft.ifftshift(tmp, )

        tmp = torch.squeeze(
            ifft_and_scale(
                tmp, 
                scal_coef, 
                torch.from_numpy(rHrps.astype(np.int32)),
                torch.from_numpy((rHrps*2).astype(np.int32)),
                norm='ortho'))
        
        # tmp = apodize_torch(tmp, ndim=3, oversamp=oversamp, width=kbwidth)

        im_data[:, iC, ...] = tmp.to('cpu').numpy()

        del tmp
        if dat_basics.nAve == 2:
            del tmp_1, phase_diff

        ## Clear unused data allocation in GPU
        gc.collect()
        torch.cuda.empty_cache()
        
    del scal_coef, om, adjinterp, dcf, mask 

    logging.info('Motion correction done, total duration = %d', (time()-t0))

    ## NOTE: Do not update dat_basics, keep original kspace for future reference
    # if is_nomoco:
    #     dat_basics.nomoco_ksp   = im_data
    # else:
    #     dat_basics.data_set     = im_data

    ## Save final image
    coil_combine(dat_basics, im_data, var, working_path, no_reacq=no_reacq)



## Apodization function
def apodize_torch(input_tensor : torch.Tensor, ndim: int, oversamp: float, width: float) -> torch.Tensor:
    """
    Apodization (Kaiser-Bessel window) in PyTorch.
    
    Args:
        input_tensor: Input img data [..., *dims] (torch.Tensor)
        ndim: Number of dimensions to apodize (int)
        oversamp: Oversampling factor (float)
        width: Window width (float)
    
    Returns:
        Apodized img data (same shape as input)
    """

    device  = input_tensor.device    
    beta    = math.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5

    for dim in range(-ndim, 0):
        i       = input_tensor.shape[dim]
        os_i    = math.ceil(oversamp * i)
        
        # Create apodization window and change type to match input_tensor
        idx     = torch.arange(i, device=device)
        
        center  = i // 2
        
        term    = (math.pi * width * (idx - center) / os_i)
        apod    = torch.sqrt(torch.clamp(beta**2 - term**2, min=0.0))  # Ensure non-negative values

        # Normalize the apodization window
        denom   = torch.sinh(apod)
        denom   = torch.where(denom == 0, torch.ones_like(denom)*1e-7, denom)
        apod    = apod / denom
        
        # Reshape for broadcasting
        view_shape  = [1] * len(input_tensor.shape)
        view_shape[dim] = i
        apod        = apod.view(view_shape)
        
        # Apply window
        input_tensor = input_tensor * apod
    
    return input_tensor
