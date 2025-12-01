import numpy as np
import sigpy

from .mtypes import DatBasics
from .utils import scalingArray, saveImage
import os


## RMSE coil combine
def coil_combine(dat_basics : DatBasics, data : np.ndarray, var : str, working_path : str, no_reacq: bool) -> None:

    Coils_im    = np.abs(np.sqrt(np.sum(np.abs(data * np.conj(data)), axis=1, keepdims=False)))

    img         = scalingArray(Coils_im, np.min(Coils_im), np.max(Coils_im))
    img         = sigpy.resize(img, dat_basics.rHrps)[..., ::-1, ::-1]

    saveImage(img.transpose(dat_basics.dimspermute), dat_basics.affinematrix, 
              os.path.join(working_path, f"Reco_{var}" + ("_noreacq" if no_reacq and var == 'noMoco' else "") + ".nii"))



## TODO: Working on adaptive coil combine
def adaptive_coil_combine(dat_basics : DatBasics, data : np.ndarray, var : str, working_path : str) -> None:

    raise NotImplementedError("Adaptive coil combine not implemented yet !")
    ##############################################################################
    ### Adj sens map preparation
    
    m_adjsens       = ismrmrd.Dataset('Im_noise.h5', file_info('Im_noise.h5')[0])
    m_adjsens_head  = ismrmrd.xsd.CreateFromDocument(m_adjsens.read_xml_header())
    adjs_enc        = m_adjsens_head.encoding[0]
    adjs_nC         = 2 ## Body coil channels = 2

    adjNx     = int(adjs_enc.encodedSpace.matrixSize.x)
    adjNy     = int(adjs_enc.encodedSpace.matrixSize.y) #int(enc.encodingLimits.kspace_encoding_step_1.maximum)+1
    adjNz     = int(adjs_enc.encodedSpace.matrixSize.z) #int(enc.encodingLimits.kspace_encoding_step_2.maximum)+1
    adjHrps   = np.array([adjNx//2, adjNy, adjNz], dtype=np.int32)

    adjFOVx   = adjs_enc.reconSpace.fieldOfView_mm.x
    adjFOVy   = adjs_enc.reconSpace.fieldOfView_mm.y
    adjFOVz   = adjs_enc.reconSpace.fieldOfView_mm.z
    adjFOVrps = np.array([adjFOVx, adjFOVy, adjFOVz])

    center_line_y = int(adjs_enc.encodingLimits.kspace_encoding_step_1.center)
    center_line_z = int(adjs_enc.encodingLimits.kspace_encoding_step_2.center)

    adjsens_header  = m_adjsens._dataset['data'][:]['head'][:]
    tmp_data        = m_adjsens._dataset['data'][:]['data']
    m_adjsens.close()
    
    adjsflags   = np.asarray([adjsens_header[i][1] for i in range(len(adjsens_header))], dtype=np.uint32)
    adjs_idx    = [not is_flag_set(adjsflags[i], ismrmrd.ACQ_IS_NOISE_MEASUREMENT) for i in range(len(adjsflags))]
    
    ########################################################################################
    #Noise prewhitening
    adjnoise_idx    = [is_flag_set(adjsflags[i], ismrmrd.ACQ_IS_NOISE_MEASUREMENT) for i in range(len(adjsflags))]
    adj_noise       = np.asarray([tmp_data[i].view(np.complex64).reshape(nC, -1) 
                                    for i in range(0, len(tmp_data)) if adjnoise_idx[i]],).transpose((1, 2, 0))
    
    noise_dwell_time = adjsens_header[next(i for i in range(0, len(adjnoise_idx)) if adjnoise_idx[i])][-9]
    
    scale_factor    = data_dwell_time / noise_dwell_time * receiver_noise_bandwidth
    noise_ws, psi   = calculate_prewhitening(adj_noise, scale_factor=scale_factor, return_psi=True)

    ########################################################################################

    tmp, adjs_idx, adjsens_header = [list(k) 
                                    for k in zip(*[[tmp_data[i], adjs_idx[i], adjsens_header[i]] 
                                    for i in range(len(tmp_data)) if adjs_idx[i]])]

    adjs_Lin, adjs_Par, Set   = np.asarray([[col[0], col[1], col[7]] for col in [adjsens_header[i][-3] 
                        for i in range(len(adjsens_header))]]).T
    
    adjs_Lin = adjs_Lin + math.floor(adjNy/2) + 1 - center_line_y
    adjs_Par = adjs_Par + math.floor(adjNz/2) + 1 - center_line_z

    ## Get x, y, z rotations
    adjread_dir, adjphase_dir, adjslice_dir = np.asarray([np.array([col[-7], col[-6], col[-5]]).T for col in [adjsens_header[i] 
                            for i in range(len(adjsens_header)) if adjs_idx[i]]]).transpose((1,0,2))

    adjdir_rot      = np.array([adjread_dir[0], adjphase_dir[0], adjslice_dir[0]])

    #######################################################
    adjs_64ch   = np.asarray([tmp[i].view(np.complex64).reshape(nC, -1) 
                                    for i in range(0, len(tmp)) if Set[i]==0]).transpose((2,1,0))
    
    adjs_2ch    = np.asarray([tmp[i].view(np.complex64).reshape(adjs_nC, -1) 
                                    for i in range(0, len(tmp)) if Set[i]==1]).transpose((2,1,0))


    data_64ch       = np.zeros((adjHrps[0]*2, nC, *adjHrps[1:]), dtype=np.complex64)
    data_64ch[np.arange(adjNx)[:, None, None], np.arange(nC)[:, None], adjs_Lin[Set==0], adjs_Par[Set==0]] = adjs_64ch
    data_64ch       = remove_ro_oversampling_gpu(data_64ch, chunk_size=10000)
    # data_64ch   = apply_prewhitening(data_64ch.transpose((1,0,2,3)), noise_ws).transpose((1,0,2,3)).astype(np.complex64)
    data_64ch       = ifftn_shift(data_64ch, (3,2,0))

    data_2ch        = np.zeros((adjHrps[0]*2, adjs_nC, *adjHrps[1:]), dtype=np.complex64)
    data_2ch[np.arange(adjNx)[:, None, None], np.arange(adjs_nC)[:, None], adjs_Lin[Set==1], adjs_Par[Set==1]] = adjs_2ch
    data_2ch        = remove_ro_oversampling_gpu(data_2ch, chunk_size=10000)
    data_2ch        = ifftn_shift(data_2ch, (3,2,0))
    
    B1              = data_2ch[:, 0, ...]
    B2              = data_2ch[:, 1, ...]
    phase_diff      = np.angle(np.vdot(B1.flatten(), B2.flatten()))
    B2_corr         = B2 * np.exp(-1j * phase_diff)

    img_body_comb   = 0.5 * (B1 + B2_corr)

    I               = (np.abs(img_body_comb) > .03*np.max(np.abs(img_body_comb)))
    I               = binary_dilation(I, ball(1))#.flatten()

    ## TODO: Remove this hardcoded part
    data_64ch       = data_64ch.transpose((3,1,2,0))
    img_body_comb   = img_body_comb.transpose((2,1,0))
    I               = I.transpose((2,1,0)).flatten()
    #### 

    data_64ch       = np.moveaxis(data_64ch, 1, -1)
    data_64ch       = data_64ch.reshape((np.prod(adjHrps), -1))
    csm_raw         = np.zeros_like(data_64ch, dtype=np.complex64)

    csm_raw         = csm_raw.reshape((np.prod(adjHrps), -1))
    csm_raw[I, :]   = data_64ch[I, :] / (img_body_comb).flatten()[I, np.newaxis]
    csm_raw         = csm_raw.reshape((*adjHrps, nC))

    ## Clean up
    del adjsens_header, adjs_64ch, adjs_2ch, tmp_data, tmp, adjs_idx, adjs_Lin, adjs_Par, data_64ch, data_2ch, B1, B2, phase_diff, B2_corr

    ############################################################################################################
    # snr_threshold = 0.03 * np.max(np.abs(img_body_comb))
    # mask = np.abs(img_body_comb) > snr_threshold

    mask    = np.abs(img_body_comb)
    mask[mask > 0.1 * np.max(mask)] = 1

    # Seuil automatique
    level   = threshold_otsu(mask)
    mask    = mask > level

    # Morphologie 3D pour affiner le masque
    mask    = binary_closing(mask, ball(1))
    # mask    = binary_erosion(mask, ball(1))

    degree = 4
    alpha  = 1e-4

    xx, yy, zz = np.meshgrid(
        np.linspace(0, 1, adjHrps[0]),
        np.linspace(0, 1, adjHrps[1]),
        np.linspace(0, 1, adjHrps[2]),
        indexing='ij'
    )

    indepvar    = np.stack([xx[mask], yy[mask], zz[mask]], axis=-1)
    indep_eval  = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Normalisation des coordonnées
    scaler      = StandardScaler()
    X_fit       = scaler.fit_transform(indepvar)
    X_eval      = scaler.transform(indep_eval)

    # Création des features polynomiales
    poly        = PolynomialFeatures(degree=degree)
    X_fit_poly  = poly.fit_transform(X_fit)
    X_eval_poly = poly.transform(X_eval)

    # Régression régularisée
    reg         = Ridge(alpha=alpha)

    # Stockage des cartes réelles et imaginaires
    poly_re = np.zeros((np.prod(adjHrps), nC), dtype=np.float32)
    poly_im = np.zeros((np.prod(adjHrps), nC), dtype=np.float32)

    for coil in range(nC):
        real    = np.real(csm_raw[..., coil])[mask]
        img     = np.imag(csm_raw[..., coil])[mask]

        reg.fit(X_fit_poly, real)
        poly_re[:, coil] = reg.predict(X_eval_poly)

        reg.fit(X_fit_poly, img)
        poly_im[:, coil] = reg.predict(X_eval_poly)

    # Reconstruction et reshape
    poly_img = (poly_re + 1j * poly_im).reshape((*adjHrps, nC))

    # Nettoyage post-régression : supprime valeurs aberrantes
    # norm        = np.sqrt(np.sum(np.abs(poly_img)**2, axis=-1))
    # norm_med    = np.median(norm[mask])
    # outlier_mask = norm > 10 * norm_med
    # poly_img[outlier_mask] = 0

    ## Clean memory
    del indepvar, indep_eval, scaler, X_fit, X_eval, X_fit_poly, X_eval_poly
    
    ############################################################################################
    
    X, Y, Z = np.meshgrid(*[(0.5*np.linspace(-1, 1, eHrps[idx]+1) * rFOVrps[idx])[:-1] 
                    for idx in range(len(eHrps))], indexing='ij')

    ## figure out why and generalize this
    ZYX_permute = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    mshift              = rotMatDisplacement_mm.copy()
    mshift[2]           = -mshift[2]
    mshift[0]           = -mshift[0]  # Flip x and z for RAS
    
    global_mat          = np.eye(4)
    # global_mat[:3, :3]  = np.linalg.inv(np.diag(hostVoxDim_mm[permutedimss]))   ## dir_rot @ np.linalg.inv(np.diag(hostVoxDim_mm[permutedimss])) @ ZYX_permute    # 
    global_mat[:3, 3]   = mshift

    # global_mat          = global_mat @ ZYX_permute

    pts             = np.stack((X, Y, Z, np.ones_like(X)), axis=-1)
    transformed_pts = np.einsum('ij,xyzj->xyzi', global_mat, pts)
    Xq, Yq, Zq      = transformed_pts[..., :3].transpose(3,0,1,2)

    XX    = (0.5*np.linspace(-1, 1, adjHrps[0]+1) * adjFOVrps[0])[:-1]
    YY    = (0.5*np.linspace(-1, 1, adjHrps[1]+1) * adjFOVrps[1])[:-1]
    ZZ    = (0.5*np.linspace(-1, 1, adjHrps[2]+1) * adjFOVrps[2])[:-1][::-1]

    csm   = np.zeros(np.insert(eHrps, 1, nC), dtype=np.complex64)
    
    with ProcessPoolExecutor(max_workers=nC if nC <= 32 else 32) as executor:
        running_tasks = {executor.submit(interpn, (XX, YY, ZZ), 
                                    poly_img[..., iC], 
                                    np.stack((Xq, Yq, Zq), axis=-1), 
                                    method='cubic', 
                                    bounds_error=False, 
                                    fill_value=0,): iC for iC in range(nC)}

        for running_task in tqdm(as_completed(running_tasks), total=nC, desc="Interpolating CSM"):
            index = running_tasks[running_task]
            
            try:
                csm[:, index, ...] = running_task.result()
                #logging.info(f"Coil channel {index} done !")
            except Exception as e:
                logging.error(f"Error on coil channel {index}: {e}")

    csm_                = np.moveaxis(csm, 1, -1).reshape((eHrps[0]*eHrps[1]*eHrps[2], nC))
    relative_ccm        = np.asarray(np.conj(csm_) @ np.linalg.pinv(psi))

    scale_correction    = np.asarray(np.abs(np.sum(relative_ccm * csm_, axis=1)))

    # Identification des indices non nuls
    nonzero_ind = scale_correction > 0

    # Calcul de `ccm`
    ccm = np.zeros_like(csm_, dtype=complex)
    ccm[nonzero_ind.flatten(), :] = (relative_ccm[nonzero_ind.flatten(), :] / scale_correction[nonzero_ind.flatten(), np.newaxis])

    ## Clean memory
    del csm_, relative_ccm, scale_correction, csm, pts, transformed_pts, Xq, Yq, Zq

    # Reshape pour revenir aux dimensions d'origine
    ccm = ccm.reshape((*eHrps, nC))
    ccm = np.moveaxis(ccm, -1, 1).astype(np.complex64)
    
    tmp = np.abs(np.sum(ccm * np.conj(ccm), axis=1))
    tmp = scalingArray(tmp, np.min(tmp), np.max(tmp))

    saveImage(tmp, Pos_mat, outputDir+f"""sens_map.nii""")
