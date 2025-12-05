import os
import re
import math
import logging
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

import twixtools as twx

try:
    import cupy as cp
    from cupyx.scipy.fft import (
        fftshift as cfftshift, ifftshift as cifftshift,
        fftn as cfftn, ifftn as cifftn
    )
    _GPU_OK = True
except Exception:
    cp = None
    _GPU_OK = False

from .mtypes import DatBasics, PF_MAPPING, LPS_TO_RAS




_cfg_pat = re.compile(
    r'\{\s*"\s*([\w\d]+)\s*"\s*\}\s*\{\s*([-\d.Ee]+)\s*\}\s*\{\s*([-\d.Ee]+)\s*\}\s*\{\s*([-\d.Ee]+)\s*\}'
)

def _parse_cfg(config_raw: str,) -> tuple[np.ndarray,np.ndarray,int]:
    """ Parse Config_raw string to get dFFTScale and dRawCorr arrays \n
    config_raw  : string from img.hdr['Config_raw']

    return      : dFFTScale, dRawCorr, nC
    """
    
    ## config_raw = img.hdr['Config_raw']
    seen, dfft, draw = set(), [], []

    for m in _cfg_pat.findall(config_raw):
        ## cid = CoilID - s = dFFTScale 
        # re_ = dRawDataCorrectionFactorRe - im_ = dRawDataCorrectionFactorIm
        cid, s, re_, im_ = m

        if cid in seen: 
            continue

        seen.add(cid)
        dfft.append(float(s))
        draw.append(complex(float(re_), float(im_)))

    dfft = np.asarray(dfft, np.float32)
    draw = np.asarray(draw, np.complex64)

    return dfft, draw


def _pf_factor(yaps: dict) -> np.ndarray:

    ro = yaps['sKSpace'].get('ucReadoutPartialFourier', 0)
    ph = yaps['sKSpace'].get('ucPhasePartialFourier', 0)
    sl = yaps['sKSpace'].get('ucSlicePartialFourier', 0)

    return np.array([PF_MAPPING.get(ro, 1.0), PF_MAPPING.get(ph, 1.0), PF_MAPPING.get(sl, 1.0)], np.float32)


def is_image_scan(mdh):
    disqualifier = [
        'ACQEND', 'RTFEEDBACK', 'HPFEEDBACK', 'SYNCDATA', 'REFPHASESTABSCAN',
        'PHASESTABSCAN', 'PHASCOR', 'NOISEADJSCAN', 'noname60']
    
    for name in disqualifier:
        if mdh.is_flag_set(name):
            return False
        
    # check for patref scan
    if mdh.is_flag_set('PATREFSCAN')\
            and not mdh.is_flag_set('PATREFANDIMASCAN'):
        return False
    return True


def calculate_prewhitening(noise : np.ndarray, scale_factor: float = 1.0, return_psi: bool = False):
    '''
    Calculates the noise prewhitening matrix

    :param noise: Input noise data (array), shape (n_coils, n_samples)
    :param scale_factor: Correction factor for noise covariance scaling,
                         e.g., (T_acq_dwell / T_noise_dwell) * ReceiverBandwidthRatio
    :param return_psi: If True, also returns the estimated noise covariance matrix

    :returns: 
        dmtx: Prewhitening matrix, shape (n_coils, n_coils)
        psi: (optional) Noise covariance matrix
    '''

    n_coils, n_samples = noise.shape[0], noise.size // noise.shape[0]
    noise_int = noise.reshape(n_coils, n_samples)

    # Compute noise covariance matrix (Psi)
    psi = (1 / (n_samples - 1)) * (noise_int @ noise_int.conj().T)

    # Cholesky decomposition + inversion (whitening matrix)
    L = np.linalg.cholesky(psi)
    dmtx = np.linalg.inv(L)

    # Apply scaling
    dmtx *= np.sqrt(2) * np.sqrt(scale_factor)

    if return_psi:
        return dmtx, psi
    else:
        return dmtx
    

def apply_prewhitening(data: np.ndarray, dmtx: np.ndarray) -> np.ndarray:
    """Apply the noise prewhitening matrix to multichannel data.

    Parameters
    ----------
    data : ndarray
        Input data with shape (n_coils, ...), e.g., (n_coils, x, y, z).
    dmtx : ndarray
        Prewhitening matrix of shape (n_coils, n_coils), typically the inverse Cholesky of the noise covariance.

    Returns
    -------
    w_data : ndarray
        Prewhitened data, same shape as input.
    """
    orig_shape  = data.shape
    w           = np.einsum('ij,jk->ik', dmtx, data.reshape(orig_shape[0], -1), optimize='greedy')

    return np.asarray(w).reshape(orig_shape)


def remove_ro_oversampling_gpu(data : np.ndarray, chunk_size : int = 5000) -> np.ndarray:

    if not _GPU_OK:
        # Fallback CPU. Lent mais fonctionnel.
        save_shape = list(data.shape)
        data = data.reshape([data.shape[0], -1])
        nx = data.shape[0] // 2
        out = []
        for i in range(0, data.shape[1], chunk_size):
            chunk = data[:, i:i+chunk_size]
            chunk = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(chunk, axes=(0,)), axis=0), axes=(0,))
            chunk = chunk[(0 + (nx // 2)):(nx + (nx // 2)), ...]
            chunk = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(chunk, axes=(0,)), axis=0), axes=(0,))
            out.append(chunk)
        data = np.concatenate(out, axis=1)
        save_shape[0] = nx
        return data.reshape(save_shape)
    

    save_shape = list(data.shape)
    # Reshape data for processing
    data = data.reshape([data.shape[0], -1])

    # Compute the number of chunks
    num_chunks = (data.shape[1] + chunk_size - 1) // chunk_size 
    new_nx = data.shape[0] >> 1

    result = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, data.shape[1])
        
        # Convert chunk to CuPy array
        data_chunk = cp.array(data[:, start:end])
        
        # Perform FFT operations with shift and cropping
        data_chunk = cifftshift(cifftn(cfftshift(data_chunk), axes=(0,)))
        data_chunk = data_chunk[(0 + (new_nx >> 1)):(new_nx + (new_nx >> 1)), ...]
        data_chunk = cifftshift(cfftn(cfftshift(data_chunk), axes=(0,)))
        
        # Convert CuPy array back to NumPy array and store the result
        result.append(cp.asnumpy(data_chunk))
        
        # Clear the memory for the current chunk
        del data_chunk
        cp.get_default_memory_pool().free_all_blocks()
    
    # Combine all chunks into the final result
    data = np.concatenate(result, axis=1)

    # Update shape
    save_shape[0] = new_nx

    return data.reshape(save_shape)


def prepare_noise_adjustment(adjdata, NoiseScaleFactor):

    
    adjdata_mdb     = np.array(adjdata['mdb'])
    # noise_yaps      = adjdata['hdr']['MeasYaps']

    noise           = np.array([tmp.data for tmp in adjdata_mdb if tmp.is_flag_set('NOISEADJSCAN')]).transpose((1, 2, 0))
    # rawdatacorr     = [tmp.data for tmp in adjdata_mdb if tmp.is_flag_set('RAWDATACORRECTION')]

    return calculate_prewhitening(noise, NoiseScaleFactor)


def handle_geometry(geometry, ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:

    hostVox_mm  = np.asarray(geometry.voxelsize) ## in mm

    res         = np.asarray(geometry.resolution.copy())
    res[0]      //=2

    ## Geometry
    sPosition_mm  = np.asarray(geometry.offset) # [dSag, dCor, dTra] in mm

    sPosition_vox = sPosition_mm.copy()
    sPosition_vox[2] *= -1  # Tra to Tra in RAS
    sPosition_vox = np.divide(sPosition_vox, hostVox_mm)

    ##########################################
    sNormal     = geometry.normal # [dSag, dCor, dTra]
    sNormal     = sNormal / np.linalg.norm(sNormal)

    dInplaneRot = geometry.inplane_rot * 180/math.pi # orig in radian

    match np.argmax(np.abs(sNormal)):
        case 0:  # Sag
            FOV_rot             = R.from_euler('xyz', [dInplaneRot, 0, 0], degrees=True).as_matrix().T
            # must be xzy: rot around x(RL) slice, then freq is z(SI), y(AP) as phase
            if abs(dInplaneRot) < 45:
                dimspermute    = [2, 1, 0]
            else:
                dimspermute    = [2, 0, 1]
            logging.info("Slice orientation - Sagittal")
    
        case 1:  # Coronal
            FOV_rot             = R.from_euler('xyz', [math.acos(sNormal[1]), 0, 0], degrees=False).as_matrix().T
            if abs(dInplaneRot) < 45:
                dimspermute    = [1, 2, 0]
            else:
                dimspermute    = [0, 2, 1]
            logging.info("Slice orientation - Coronal")

        case 2:  # Trans
            FOV_rot             = R.from_euler('xyz', [math.acos(sNormal[2]), 0, 0], degrees=False).as_matrix().T
            if abs(dInplaneRot) < 45:
                dimspermute    = [0, 1, 2]
            else:
                dimspermute    = [1, 0, 2]
            logging.info("Slice orientation - Transversal")

    # dimspermute = [<R1, R2, R3>]

    rps2xyz   = geometry.rps_to_xyz().copy()  # from RPS to XYZ (RAS)
    ## flip x 
    rps2xyz[0, :] *= -1

    ##TODO: check and generalize
    maffine         = np.diag([*hostVox_mm, 1])
    maffine[:3, :3] = rps2xyz @ maffine[:3, :3]
    maffine[:3, 3]  = sPosition_mm

    Pos_mat = np.eye(4)
    Pos_mat[:3, 3] = ((-res/2)[dimspermute]) + [1, 0, .5]

    # LPS_to_RAS = np.diag([-1, -1, 1, 1]) # ro and pe are flipped

    maffine = LPS_TO_RAS @ maffine @ Pos_mat

    return hostVox_mm, sPosition_mm, sPosition_vox, dimspermute, FOV_rot, maffine


def is_image_scan_embacs(mdh):
    from twixtools.mdh_def import is_flag_set

    disqualifier = [
        'ACQEND', 'RTFEEDBACK', 'HPFEEDBACK', 'SYNCDATA', 'REFPHASESTABSCAN',
        'PHASESTABSCAN', 'PHASCOR', 'NOISEADJSCAN', 'noname60']
    for name in disqualifier:
        if is_flag_set(mdh, name):
            return False
        
    # check for patref scan
    # if is_flag_set(mdh, 'PATREFSCAN')\
    #         and not is_flag_set(mdh, 'PATREFANDIMASCAN'):
    #     return False
    return True


def load_dat_basics(working_dir: str, reverse_moco: bool = False, noMOCO: bool = False) -> DatBasics:

    pathDAT = glob(os.path.join(working_dir, "*.dat"))

    if not pathDAT:
        raise FileNotFoundError("no .dat file found in working directory")

    pathDAT = pathDAT[0]

    ## no syncdata for now
    adjdata, rawd   = twx.read_twix(pathDAT, parse_pmu=False,) #  keep_syncdata=False

    mdb     = np.array(rawd['mdb'])
    yaps    = rawd['hdr']['MeasYaps']
    config  = rawd['hdr']['Config']

    valid   = [is_image_scan_embacs(tmp.mdh) for tmp in mdb]

    logging.info(f"Found {len(mdb)} total scans, {sum(valid)} valid image scans.")
    
    # patref_data     = mdb[patref]
    
    ## mask img with valid lines only
    mdb     = mdb[valid]

    is_surfcoilcorrdata = [tmp.is_flag_set('RAWDATACORRECTION') for tmp in mdb]

    # eNx     = int(config['NCol'] if 'NCol' in config else config['NColMeas']) ## not accurate if RO asymm
    eNx     = int(mdb[0].mdh.SamplesInScan)

    eNy     = int(config['NLinMeas'])
    eNz     = int(config['NParMeas'])
    eHrps   = np.array([eNx, eNy, eNz], np.uint32)

    nC      = int(mdb[0].mdh.UsedChannels)
    
    kcent   = np.array([mdb[0].mdh.CenterCol//2, 
                    mdb[0].mdh.CenterLin, 
                    mdb[0].mdh.CenterPar], np.int32)

    rNx     = int(yaps['sKSpace']['lBaseResolution'])
    rNy     = int(yaps['sKSpace']['lPhaseEncodingLines'])
    rNz     = int(yaps['sKSpace']['lPartitions'])
    rHrps   = np.array([rNx, rNy, rNz], int)

    rFOVrps = np.array([
        yaps['sSliceArray']['asSlice'][0]['dReadoutFOV'],
        yaps['sSliceArray']['asSlice'][0]['dPhaseFOV'],
        yaps['sSliceArray']['asSlice'][0]['dThickness']], float)
    
    hostVox_mm, sPosition_mm, sPosition_vox, dimspermute, FOV_rot, maffine = handle_geometry(rawd['geometry'][-1])
    
    logging.info(f"Image geometry: FOV {'x'.join(map(str, rFOVrps))} mm^3, "
                 f"Reconstruction matrix size {'x'.join(map(str, rHrps))}, "
                 f"Voxel size {'x'.join(map(lambda x: f'{x:.2f}', hostVox_mm))} mm^3, "
                 f"{nC} channels.")

    ## PF factors and assymetric echo
    pf          = _pf_factor(yaps)
    assym_echo  = yaps['sKSpace'].get('ucAsymmetricEchoAllowed', 0)

    logging.info(f"Partial Fourier factors (RO, PE, 3D): {pf[0]}, {pf[1]}, {pf[2]}")
    if assym_echo:
        logging.info("Asymmetric echo acquisition detected on readout dimension.")

    # channels and scaling
    dfft, draw  = _parse_cfg(rawd['hdr'].get('Config_raw',''))
    
    logging.info("Correcting for coil receiver gain variations.")

    ## filling dims
    all_dims    = ['cLin', 'cPar', 'cAve']
    dims_dict   = {dim: [] for dim in all_dims}
    for tmp in mdb:
        for dim in all_dims:
            dims_dict[dim].append(getattr(tmp, dim))

    dims_dict   = {dim: np.array(v, np.uint32) for dim, v in dims_dict.items()}

    # turbofactor = int(yaps.get('sFastImaging', {}).get('lTurboFactor', 1) or 1)
    ## find if there is reacq data
    reacq_data      = np.zeros(len(mdb), np.uint16)
    last_scan_index = next((i for i, tmp in enumerate(mdb) if tmp.is_flag_set('LASTSCANINSLICE') and dims_dict['cAve'][i] == 0), len(mdb)-1)
    
    if last_scan_index < len(mdb)-1:
        # reacq_data  = np.zeros(len(mdb), np.uint16)
        reacq_data[last_scan_index+1:] = 1

    if any(reacq_data):
        dims_dict['reacq'] = reacq_data

    cAve            = dims_dict.get('cAve', None)
    nAve            = int(cAve.max()) + 1 if cAve is not None and len(cAve) > 0 else 1
    
    reacq_max       = int(reacq_data.max()) + 1 if any(reacq_data) else 1

    logging.info(f"Number of Averages: {nAve},"
                    f" Number of Reacquisitions: {(np.count_nonzero(reacq_data)+1)} Readout lines.")

    ticks           = np.array([tmp.mdh.TimeStamp*2.5 for tmp in mdb], np.int64)
    ## format timestamps
    ticks_formated  = np.zeros((rHrps[1], rHrps[2], nAve, reacq_max), np.uint64)

    ticks_formated[dims_dict['cLin'], dims_dict['cPar'], dims_dict['cAve'], dims_dict['reacq']] = ticks

    # RO indices with PF
    if (pf[0] != 1.0 or assym_echo) and not math.isclose(kcent[0], rHrps[0]//2, abs_tol=1): ## PF before kspace center
        Col = np.arange(rHrps[0] - eHrps[0]//2, rHrps[0])

    else: ## handle all other cases ?
        Col = np.arange(0, eHrps[0]//2)
    
    ## Handle cLin and cPar PF order 
    if pf[1] != 1.0 and not math.isclose(kcent[1], rHrps[1]//2, abs_tol=1): ## PF before kspace center
        dims_dict['cLin'] += kcent[1] 

    if pf[2] != 1.0 and not math.isclose(kcent[2], rHrps[2]//2, abs_tol=1): ## PF before kspace center
        dims_dict['cPar'] += kcent[2]

    # acceleration factors
    acc_y       = int(yaps.get('sPat', {}).get('lAccelFactPE', 1) or 1)
    acc_z       = int(yaps.get('sPat', {}).get('lAccelFact3D', 1) or 1)
    Arps        = np.array([1, acc_y, acc_z], int)

    logging.info(f"Acceleration factors (PE, 3D): {Arps[1]}, {Arps[2]}")

    nb_reflinesPE = int(yaps.get('sPat', {}).get('lRefLinesPE', 32) or 0)
    nb_reflines3D = int(yaps.get('sPat', {}).get('lRefLines3D', 32) or 0)

    max_calib_shape = [36, nb_reflinesPE, nb_reflines3D]

    ipatref     = np.zeros(len(mdb), np.bool_)
    if max(Arps) > 1:
        ipatref = [tmp.is_flag_set('PATREFSCAN') or tmp.is_flag_set('PATREFANDIMASCAN') for tmp in mdb]
        ipatref = ipatref[:last_scan_index+1]
    

    ## Load or prepare kspace data
    nomoco_ksp  = None
    ksp_exist   = {"base": False, "nomoco": False}

    if os.path.exists(os.path.join(working_dir, 'filled_ksp.npy')):
        logging.info("Reloading previously saved k-space for base data ! ")
        # data_set    = np.load(os.path.join(working_dir, 'filled_ksp.npy'))
        ksp_exist["base"]   = True

    if os.path.exists(os.path.join(working_dir, 'filled_ksp_nomoco.npy')):
        logging.info("Reloading previously saved k-space for noMoco data ! ")
        # nomoco_ksp      = np.load(os.path.join(working_dir, 'filled_ksp_nomoco.npy'))
        ksp_exist["nomoco"] = True


    if not ksp_exist["base"] or not ksp_exist["nomoco"]:
        ## prepare data
        ## TODO: Implement POCS for RO asymmetric 

        ksp         = np.array([tmp.data for tmp in mdb]).transpose((2, 1, 0))
        logging.info("Performing oversampling removal on readout dimension.")
        ksp         = remove_ro_oversampling_gpu(ksp, chunk_size=8000).transpose((1,0,2))

        ksp         *= dfft[:, None, None]
        ksp[..., is_surfcoilcorrdata]   *= draw[:, None, None]
            
        ## prepare noise weights
        logging.info("Preparing noise prewhitening matrix and applying it to k-space data.")
        noise_ws    = prepare_noise_adjustment(adjdata, config['NoiseScaleFactor'])
        ksp         = apply_prewhitening(ksp, noise_ws).transpose((1,0,2))

        data_set    = np.ascontiguousarray(np.zeros((rHrps[0], nC, rHrps[1], rHrps[2], nAve, reacq_max), np.complex64))

        ## NOTE: if reacq exist, sometimes the same ro are reacquired more than once
        ## in that case, the last one is kept

        data_set[Col[:, None, None], np.arange(nC)[:, None], 
                dims_dict['cLin'], 
                dims_dict['cPar'], 
                dims_dict['cAve'], 
                dims_dict['reacq']] = ksp 

    ################
    ## Use reacq data if exist to replace original data for the case of noMoco with reacq
    if not ksp_exist["nomoco"] and noMOCO and reverse_moco and 'reacq' in dims_dict.keys():
            
        logging.info("Reacq data found - preparing noMoco reacq kspace ! ")

        mask = np.where(dims_dict['reacq'] == 1)
        rLin = dims_dict['cLin'][mask]
        rPar = dims_dict['cPar'][mask]
        rAve = dims_dict['cAve'][mask]

        nomoco_ksp = data_set[..., 0].copy()
        nomoco_ksp[..., rLin, rPar, rAve] = data_set[..., rLin, rPar, rAve, 1]

    elif ksp_exist["nomoco"]:
        nomoco_ksp = np.load(os.path.join(working_dir, 'filled_ksp_nomoco.npy'))


    if not ksp_exist["base"]:
        ## Then squeeze reacq dimension and adapt lin/par/ave dict
        data_set = data_set[..., 0]
    else:
        data_set = np.load(os.path.join(working_dir, 'filled_ksp.npy'))
    
    ## Remove reacq entries from dims_dict
    mask                = dims_dict['reacq'] == 0
    dims_dict['cLin']   = dims_dict['cLin'][mask]
    dims_dict['cPar']   = dims_dict['cPar'][mask]
    dims_dict['cAve']   = dims_dict['cAve'][mask]
    
    del dims_dict['reacq'], mask
    
    logging.info("Data preparation complete !")

    return DatBasics(eHrps          = eHrps,
                     rHrps          = rHrps,
                     rFOVrps        = rFOVrps,
                     hostVox_mm     = hostVox_mm,
                     nC             = nC,
                     nAve           = nAve,
                     kcent          = kcent,
                     pf             = pf,
                     Arps           = Arps,
                     affinematrix   = maffine,
                     data_set       = data_set,
                     dims_dict      = dims_dict,
                     sPosition_mm   = sPosition_mm,
                     sPosition_vox  = sPosition_vox,
                     dimspermute    = dimspermute,
                     FOV_rot        = FOV_rot,
                     ticks          = ticks,
                     ticks_formated = ticks_formated,
                     ksp_exist      = ksp_exist,
                     max_calib_shape= max_calib_shape,
                     ipatref        = ipatref,
                     nomoco_ksp     = nomoco_ksp
                     )