import os
import numpy as np
import logging
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata, interp1d
from scipy import signal

from .utils import plot_curves
from .mtypes import DatBasics


def file_reader(filename : str, path : str) -> list:
    with open(os.path.join(path, filename)) as file:
        m_reader = file.read().splitlines()
        file.close()
    return m_reader

def time_str2int(mstr : str) -> np.ndarray:
    assert (mstr != '')
    return np.asarray([mstr[0:2], mstr[3:5], mstr[6:8], mstr[9:12]], dtype=np.uint32)

def to_msec(time_obj : list, toint_: bool = True) -> np.ndarray:
  if toint_:
      return [((tmp.hour*6e1 + tmp.minute)*6e1 + tmp.second)*1e3 \
        + tmp.microsecond for tmp in time_obj] #*1e-3 for ms
  else:
      return np.int_([((time_obj.hour*6e1 + time_obj.minute)*6e1 \
        + time_obj.second)*1e3 + time_obj.microsecond]) #*1e-3 for ms


def prepare_tcl_data(working_path   : str,
                     tcl_dir        : str, 
                     dat_basics     : DatBasics,
                     reverse_moco   : bool=False,
                     mv2center      : bool=False,
                     smooth_Tcl     : bool=False):
    """
    working_path    : path to save plots
    tcl_dir         : path to tcl files
    dat_basics      : DatBasics object
    reverse_moco    : if True, reverse the motion correction (for testing purpose)
    mv2center       : if True, move the estimation to the center of the kspace (for testing purpose)
    smooth_Tcl      : if True, smooth the motion curves using butterworth filter (for testing purpose)
    """

    for filename in os.listdir(tcl_dir):

        # Reading Transf matrix
        if filename.endswith('_POA.tsp'):
            m_reader    = file_reader(filename, tcl_dir)[11:]
            tmp         = np.asarray([raw.split() for raw in m_reader], dtype=np.float32)
            A_vector    = tmp[..., 1:-1].reshape((-1, 4, 4)).transpose((1,2,0)).astype(np.float32)
            idx_Avector = tmp[..., 0].astype(np.uint32)

        # Reading ALN Matrix
        elif filename.endswith('_ALN.tsa'):
            m_reader    = file_reader(filename, tcl_dir)[16:]
            Align_mat   = np.asarray([raw.split() 
                                      for raw in (m_reader[:4] if len(m_reader)>4 else m_reader)], 
                                      dtype=np.float32)
        # Read time file
        elif filename.endswith('_TIM.tst'):

            m_reader                = file_reader(filename, tcl_dir)[11:]
            idx_Tcltime, tcl_time   = zip(
                *map(lambda x: [x[0].astype(np.uint32), datetime(1, 1, 1, *time_str2int(x[2])).time()], 
                        np.asarray([raw.split() for raw in m_reader], )))

            tcl_time    = np.asarray(to_msec(tcl_time), np.int64) - 20 # in ms, -20ms taking in account transfer latency
            idx_Tcltime = np.asarray(idx_Tcltime, np.uint32)

    if A_vector is None or Align_mat is None or tcl_time is None:
        raise ValueError("Some Tcl files are missing ! ")

    logging.info("Tcl motion estimation loaded - #matrices: {}, #timestamps: {}".format(A_vector.shape[2], len(tcl_time)))

    if not (dat_basics.ticks[0] > tcl_time[0] and dat_basics.ticks[-1] < tcl_time[-1]):
        raise ValueError("Tcl time file does not cover the whole acquisition time ! ")
        
    ## Check all timestamps match
    _, _, cross_idx = np.intersect1d(idx_Avector, idx_Tcltime, assume_unique=True, return_indices=True)
    tcl_time        = tcl_time[cross_idx]

    ## Bring estimation to MRI space
    offsetMat   = np.diag([-1, 1, -1, 1])
    rAlign_mat  = offsetMat @ Align_mat
    A_inv       = np.linalg.inv(rAlign_mat)

    ## have to multiply as follows: Result = rAlign_mat @ A_vector @ np.linalg.inv(rAlign_mat)
    A_vector    = np.einsum('ij,jkn,kl->iln', rAlign_mat, A_vector, A_inv)

    ## taking in account patient position
    new_tmat    = np.zeros_like(A_vector)

    Rot_           = A_vector[:3, :3, :]
    t_vox       = A_vector[:3, 3, :] / dat_basics.hostVox_mm[:, None]

    delta_vox   = dat_basics.sPosition_vox[:, None] - np.einsum('ijn,j->in', Rot_, dat_basics.sPosition_vox) # np.tensordot
    
    t_mm        = (t_vox + delta_vox) * dat_basics.hostVox_mm[:, None]
    t_mm_rot    = dat_basics.FOV_rot @ t_mm

    new_tmat[:3, :3, :] = Rot_
    new_tmat[:3, 3, :]  = t_mm_rot
    new_tmat[3, 3, :]   = 1

    del A_vector, Rot_, t_vox, delta_vox, t_mm, t_mm_rot

    ## Find MRI/TCL timestamps closest to each other
    mri_tcl_match   = (interp1d(tcl_time, np.arange(len(tcl_time)),  kind='nearest', 
                                        fill_value='extrapolate', assume_sorted=True)(dat_basics.ticks)).astype(np.uint32)
    
    ct_idx          = next(i for i in range(len(dat_basics.ticks)) if dat_basics.dims_dict['cLin'][i] == dat_basics.kcent[1] and \
                                                        dat_basics.dims_dict['cPar'][i] == dat_basics.kcent[2] and \
                                                        dat_basics.dims_dict['cAve'][i] == 0)

    center_ro       = (dat_basics.ticks[ct_idx] - dat_basics.ticks[0])*1e-3 # in s
    center_ro       = np.arange(center_ro, center_ro+2)

    ## Plot and save this step
    list_x          = tcl_time[mri_tcl_match[0]:mri_tcl_match[-1]]*1e-3 # in s
    list_x          = list_x - list_x[0]

    logging.info("Tcl motion timestamps aligned with MRI timestamps.")
    plot_curves(new_tmat[..., mri_tcl_match[0]:mri_tcl_match[-1]], list_x, center_ro, working_path, 'Tcl_full', smooth_Tcl)
    
    if reverse_moco:
        try:
            with open(os.path.join(working_path,
                *[k for k in os.listdir(tcl_dir) + os.listdir(working_path) if k.endswith("_motion.log")]), 'r') as f:
                mri_used_idx    = f.readlines()

            mri_used_idx        = np.unique(np.asarray([k.split(' ')[0] for k in mri_used_idx], dtype=np.uint32))
            logging.info("Reverse MOCO used, found %d lines in log file !", len(mri_used_idx))

            assert np.isin(mri_used_idx, idx_Avector, assume_unique=True).all(), \
                "Some mri_used_idx are not in idx_Avector !"
            
            ## Find closest indices
            _, _, cross_idx   = np.intersect1d(mri_used_idx, idx_Avector, assume_unique=False, return_indices=True)

            tcl_time    = tcl_time[cross_idx]
            new_tmat    = new_tmat[..., cross_idx]

            mri_tcl_match = (interp1d(tcl_time, np.arange(len(tcl_time)),  kind='nearest', 
                                        fill_value='extrapolate', assume_sorted=True)(dat_basics.ticks)).astype(np.uint32)

            new_tmat    = np.linalg.inv(new_tmat.transpose((2,0,1))).transpose((1,2,0))

        except:
            raise ValueError("Reverse MOCO selected but no log file found !")
    
    new_tmat    = new_tmat[..., mri_tcl_match]
    tcl_time    = tcl_time[mri_tcl_match]

    ## Plot and save this step
    list_x      = tcl_time * 1e-3 # in s
    list_x      = list_x - list_x[0]
    plot_curves(new_tmat, list_x, center_ro, working_path, 'Tcl_reduced')

    ## Now we grid the motion matrices to have one matrix per kspace line in case iPAT 
    final_matrix    = np.zeros((*dat_basics.rHrps[1:], dat_basics.nAve, 4, 4), np.float32)
    rotrans         = np.zeros((6, new_tmat.shape[2]), np.float32)

    if max(dat_basics.Arps) != 1:
        ix, iy          = np.meshgrid(np.arange(dat_basics.rHrps[1]), np.arange(dat_basics.rHrps[2]), indexing='ij')

        rotrans[:3, :]  = new_tmat[:3, 3, :]
        rotrans[3:, :]  = R.from_matrix(new_tmat[:3, :3, :].transpose((2,0,1))).as_euler('xyz', degrees=False).T

        for iA in np.unique(dat_basics.dims_dict['cAve']):

            mask            = np.where(dat_basics.dims_dict['cAve'] == iA)[0]

            src_p           = np.stack((dat_basics.dims_dict['cLin'][mask], 
                                        dat_basics.dims_dict['cPar'][mask])).T

            filled_matrix   = np.transpose([griddata(src_p, rotrans[iP, mask], (ix, iy), method='cubic', fill_value=0,) \
                        for iP in range(6)], (1,2,0))

            ## Handle border values
            if np.max(src_p[:, 0]) < np.max(ix):
                for i in range(np.max(src_p[:, 0])+1, np.max(ix)+1):
                    filled_matrix[i, ...] = filled_matrix[np.max(src_p[:, 0]), ...]

            if np.min(src_p[:, 0]) > np.min(ix):
                for i in range(np.min(ix), np.min(src_p[:, 0])+1):
                    filled_matrix[i, ...] = filled_matrix[np.min(src_p[:, 0]), ...]

            if np.max(src_p[:, 1]) < np.max(iy):
                for i in range(np.max(src_p[:, 1])+1, np.max(iy)+1):
                    filled_matrix[:, i, :] = filled_matrix[:, np.max(src_p[:, 1]), :]

            if np.min(src_p[:, 1]) > np.min(iy):
                for i in range(np.min(iy), np.min(src_p[:, 1])+1):
                    filled_matrix[i, ...] = filled_matrix[np.min(src_p[:, 1]), ...]
            
            if smooth_Tcl:
                logging.info("Smoothing Tcl motion curves using Butterworth filter.")
                # Preparing Butterworth filter parameters
                fs      = 1/(30*1e-3)   # Sampling frequency 30ms --> Hz
                fc      = 0.1           # Cut-off frequency - Hz
                w       = fc / (fs/2)   # Normalize the frequency
                forder  = 4
                b, a    = signal.butter(forder, w, 'low')

                ## applying Low pass butterworth filter
                filled_matrix = np.array(list(map(lambda x: signal.filtfilt(b, a, x), 
                                    [*filled_matrix.reshape(-1, 6, order='C')]))).reshape(filled_matrix.shape, order='C')

            final_matrix[..., iA, :3, 3]    = filled_matrix[..., :3]
            final_matrix[..., iA, :3, :3]   = R.from_euler('xyz', filled_matrix[..., 3:].reshape(-1, 3), degrees=False).as_matrix()\
                .reshape(*dat_basics.rHrps[1:], 3, 3, order='C')
            final_matrix[..., iA, 3, 3]     = 1

        logging.info("iPAT detected - Motion gridding done using cubic interpolation ! "
                        "final_matrix shape: {}".format(final_matrix.shape))
        
    else:
        logging.info("No iPAT detected - projecting motion estimation to all acquired lines.")
        
        # projeter les N estimations 1D sur les lignes acquises
        rotrans[:3, :]  = new_tmat[:3, 3, :]
        rotrans[3:, :]  = R.from_matrix(new_tmat[:3, :3, :].transpose(2,0,1)).as_euler('xyz', degrees=False).T
        
        if smooth_Tcl:
            # Preparing Butterworth filter parameters
            fs      = 1/(30*1e-3)   # Sampling frequency 30ms --> Hz
            fc      = 0.1           # Cut-off frequency - Hz
            w       = fc / (fs/2)   # Normalize the frequency
            forder  = 4
            b, a    = signal.butter(forder, w, 'low')

            ## applying Low pass butterworth filter
            rotrans = np.array(list(map(lambda x: signal.filtfilt(b, a, x), 
                                    [*rotrans.reshape(-1, 6, order='C')]))).reshape(rotrans.shape, order='C')

        lin   = dat_basics.dims_dict['cLin']
        par   = dat_basics.dims_dict['cPar']
        ave   = dat_basics.dims_dict['cAve']

        final_matrix[lin, par, ave, :3, 3]  = rotrans[:3, :].T
        final_matrix[lin, par, ave, :3, :3] = R.from_euler('xyz', rotrans[3:, :].T, degrees=False).as_matrix()\
            .reshape(*dat_basics.rHrps[1:], dat_basics.nAve, 3, 3, order='C')
        final_matrix[..., 3, 3]             = 1
        logging.info("Motion projection done ! final_matrix shape: {}".format(final_matrix.shape))

    if mv2center:
        tmp_mat         = np.linalg.inv(final_matrix[dat_basics.kcent[1], dat_basics.kcent[2], 0, ...])
        logging.info("option mv2center selected - moving all matrices to kspace center !")
        final_matrix    = np.einsum('...ij,jk->...ik', final_matrix, tmp_mat)
        # final_matrix    = np.einsum('ij,...jk->...ik', tmp_mat, final_matrix)

    else:
        tmp_mat         = np.linalg.inv(final_matrix[0, 0, 0, ...])
        logging.info("option mv2center not selected - moving all matrices to first line/acq !")
        final_matrix    = np.einsum('...ij,jk->...ik', final_matrix, tmp_mat)
        # final_matrix    = np.einsum('ij,...jk->...ik', tmp_mat, final_matrix)


    ## Save matrices to apply later
    dat_basics.moco_sys['Tcl'] = final_matrix

    ##########




def process_nav(vox_fitMat, 
                rHrps,
                Arps,
                alignDim,
                m_header, 
                var, 
                rotMatDisplacement_voxels, 
                hostVoxDim_mm, 
                thisRot,
                C_lines, 
                ref_line,
                iMPR_kspaceCentre,
                outputDir):
    
    raise NotImplementedError("Nav motion correction not implemented yet")

    this_fitMat_mm  = np.zeros(vox_fitMat.shape)

    def reader_mat(Nav_file) :
        mreader=None
        try:
            for filename in os.listdir(os.getcwd()):
                # reading Transf matrix
                if Nav_file in filename and filename.endswith('.mat'):
                    with h5py.File(filename, 'r') as file:
                        mreader = np.array(file['MPos_cent']['mats'])
                        #print(mreader.shape)

            return mreader #fitpars #fitpars_cent
        except:
            logging.error("No matlab file found")

    for i in range(vox_fitMat.shape[2]):
        vox_fitMat[:3, 3, i]    = np.divide(vox_fitMat[:3, 3, i], hostVoxDim_mm)
        thisRotMat              = vox_fitMat[:3, :3, i]
        thisRotMatDisplacement  = rotMatDisplacement_voxels - (thisRotMat @ rotMatDisplacement_voxels)
        this_fitMat_mm[:3, :3, i]   = thisRotMat
        this_fitMat_mm[:3, 3, i]    = thisRotMatDisplacement + vox_fitMat[:3, 3, i]
        this_fitMat_mm[:3, 3, i]    = np.multiply(this_fitMat_mm[:3, 3, i], hostVoxDim_mm) #put transl into mm

    theseDisplacements  = np.squeeze(this_fitMat_mm[:3, 3, :])

    newDisplacements            = thisRot @ theseDisplacements
    this_fitMat_mm[:3, 3, :]    = newDisplacements
    # this_fitMat_mm[3,3, :]      = 1
    logging.info("Nav motion array shape: {}".format(this_fitMat_mm.shape))

    rotTrans            = np.zeros((6, this_fitMat_mm.shape[2]))
    rotTrans[:3, :]     = (R.from_matrix(np.moveaxis(this_fitMat_mm, -1, 0)[:, :3, :3])).as_euler('xyz', degrees=False).T
    rotTrans[3:, :]     = np.squeeze(this_fitMat_mm[:3, 3, :])
    
    first_col           = rotTrans[:, 0][:, None]
    avg_adjacent_cols   = (rotTrans[:, :-1] + rotTrans[:, 1:])/2
    newRotTrans         = np.concatenate([first_col, avg_adjacent_cols], axis=1)

    this_fitMat_mm               = np.zeros(this_fitMat_mm.shape)
    this_fitMat_mm[3, 3, :]      = 1
    this_fitMat_mm[:3, 3, :]     = newRotTrans[3:, :]
    this_fitMat_mm[:3, :3, :]    = np.moveaxis(np.array(R.from_euler('xyz', newRotTrans[:3, ...].T, degrees=False).as_matrix()), 0, -1)

    if moco_sys[var]['plot_curve']:
        vlist_x = np.arange(this_fitMat_mm.shape[2],)*m_header.sequenceParameters.TR[0]*1e-3

        plot_curves(this_fitMat_mm, vlist_x, C_lines, outputDir, var)
        
    # Grid for missing lines due to iPAT
    if max(Arps) != 1:

        rotTrans        = np.zeros((6, this_fitMat_mm.shape[2]))
        rotTrans[:3, :] = (R.from_matrix(np.moveaxis(this_fitMat_mm, -1, 0)[:, :3, :3])).as_euler('xyz', degrees=False).T
        rotTrans[3:, :] = np.squeeze(this_fitMat_mm[:3, 3, :])
        dst_p           = np.arange(rHrps[alignDim])

        filled_matrix   = np.transpose([griddata(iSamp, rotTrans[iP, :] , dst_p, method='cubic', fill_value=0,) \
            for iP in range(6)], (1,0))

        if np.max(iSamp) < np.max(dst_p):
            for i in range(np.max(iSamp)+1, np.max(dst_p)+1):
                filled_matrix[i, ...] = filled_matrix[np.max(iSamp), ...]

        if np.min(iSamp) > np.min(dst_p):
            for i in range(np.min(dst_p), np.min(iSamp)+1):
                filled_matrix[i, ...] = filled_matrix[np.min(iSamp), ...]

        # startIndex      = iSamp[0]
        # if startIndex > 0:
        #     filled_matrix[:startIndex-1, :] = np.repeat(filled_matrix[startIndex, :], startIndex)\
        #         .reshape((startIndex, filled_matrix.shape[1]), order='F')

        # endIndex        = iSamp[-1]
        # if endIndex < rHrps[alignDim]-1:
        #     filled_matrix[endIndex+1:, :]   = np.repeat(filled_matrix[endIndex, :], rHrps[alignDim]-endIndex-1)\
        #         .reshape((rHrps[alignDim]-endIndex-1, filled_matrix.shape[1]), order='F')

        fitMats_mm_toApply_NavMOCO              = np.zeros((rHrps[1],4,4))
        fitMats_mm_toApply_NavMOCO[:, :3, 3]    = filled_matrix[:, 3:]
        fitMats_mm_toApply_NavMOCO[:, :3, :3]   = np.array([R.from_euler('xyz', filled_matrix[i, :3], degrees=False).as_matrix() for i in range(filled_matrix.shape[0])])
        fitMats_mm_toApply_NavMOCO[:, 3, 3]     = 1

    else:
        fitMats_mm_toApply_NavMOCO              = this_fitMat_mm
        fitMats_mm_toApply_NavMOCO[:, 3, 3]     = 1
        
    if ref_line == 'center':
        ## TODO: Correct index 
        fitMats_mm_toApply_NavMOCO  = np.moveaxis(recentre_affmats(
            np.moveaxis(fitMats_mm_toApply_NavMOCO, 0, -1), Lin[iMPR_kspaceCentre]), -1, 0)
    elif ref_line == 'start':
        fitMats_mm_toApply_NavMOCO  = np.moveaxis(recentre_affmats(
            np.moveaxis(fitMats_mm_toApply_NavMOCO, 0, -1), 0), -1, 0)
        
    del this_fitMat_mm
    fitMats_mm_toApply_NavMOCO          = np.repeat(fitMats_mm_toApply_NavMOCO, rHrps[2], axis=0).reshape((rHrps[1], rHrps[2], 4, 4), order='C')
    
    logging.info("{} - Rot/Trans array prepared - shape = {}".format(moco_sys[var]['name'], fitMats_mm_toApply_NavMOCO.shape))
    moco_sys[var].update({'Mat2Apply': fitMats_mm_toApply_NavMOCO})
