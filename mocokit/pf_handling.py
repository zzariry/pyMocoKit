import torch


# FFT & iFFT
def ifftnd(kspace : torch.Tensor, axes : list =[-1]) -> torch.Tensor:
    from torch.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img  = fftshift(ifftn(ifftshift(kspace, dim=axes), dim=axes), dim=axes)
    return img


def fftnd(img:torch.Tensor, axes : list =[-1]):
    from torch.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace  = fftshift(fftn(ifftshift(img, dim=axes), dim=axes), dim=axes)
    return kspace

def gaussian_window(M: int, std: float, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor: 
    n       = torch.arange(0, M, dtype=dtype, device=device)
    center  = (M - 1) / 2.
    return torch.exp(-0.5 * ((n - center) / std)**2)


def POCS(kspace:torch.Tensor, dim_enc : list =[0, 2, 3], dim_pf : int = 0, number_of_iterations : int = 5 ) -> torch.Tensor:
    """
    POCS algorithm for partial Fourier reconstruction.
    Args:
        kspace: Input k-space data (torch.Tensor)
        dim_enc: List of encoding dimensions (default: [0, 2, 3])
        dim_pf: Dimension index for partial Fourier (0 for x, 1 for y, or 2 for z)
        number_of_iterations: Number of POCS iterations (default: 5)
    Returns:
        Reconstructed k-space data (torch.Tensor)
    """

    ## dim_pf: 0 for x, 1 for y or 2 for z

    dim_nonpf       = tuple([int(x) for x in range(kspace.ndim) if x != dim_pf])        
    dim_nonpf_enc   = tuple(set(dim_enc) & set(dim_nonpf))

    n_full          = kspace.shape[dim_pf] 

    # mask for partial Fourier dimension taking accelleration into account
    mask            = torch.sum(torch.abs(kspace), dim_nonpf) > 0 # a mask along PF direction, considering acceleration, type: tensor
    
    mask_clone      = mask.clone()
    ind_one         = torch.nonzero(mask == True, as_tuple=True)[0] # index of mask_pf_acc, type: tensor
    acc_pf          = ind_one[1] - ind_one[0] # accelleration in partial Fourier direction

    # partial Fourier is at the beginning or end of the dimension
    ind_samples     = torch.arange(ind_one[-1]+1) # index of samples in PF direction, without acceleration. ind_nopocs does not take accelleration into account, right?

    if ind_one[0] > (mask.numel() - ind_one[-1] - 1): # check which side has more zeros, beginning or end
        ind_samples = torch.arange(ind_one[0], mask.numel())

    # mask if there was no accelleration in PF direction
    mask[ind_samples] = True 

    # vector mask for central region
    mask_sym        = mask & torch.flip(mask, dims=[0])     

    # gaussian mask for central region in partial Fourier dimension
    std             = max(10, n_full * 0.05) 
    gauss_pdf       = gaussian_window(n_full, std=std, device=kspace.device) * mask_sym

    # kspace smoothed with gaussian profile and masked central region
    kspace_symmetric = kspace.clone()
    kspace_symmetric = torch.swapaxes(torch.swapaxes(kspace_symmetric, dim_pf, -1) * gauss_pdf, -1, dim_pf)

    angle_image_symmetric  = ifftnd(kspace_symmetric, axes=dim_enc) # along non-pf encoding directions
    angle_image_symmetric /= torch.abs(angle_image_symmetric) # normalize to unit circle       

    kspace_full         = ifftnd(kspace, axes=dim_nonpf_enc,) # along non-pf encoding directions
    kspace_full_clone   = kspace_full.clone()

    # free memory
    del kspace_symmetric, kspace

    if kspace_full_clone.device.type == 'cuda':
        torch.cuda.empty_cache()

    for _ in range(number_of_iterations):
        image_full  = ifftnd(kspace_full, axes=[dim_pf], )
        image_full  = torch.abs(image_full) * angle_image_symmetric
        kspace_full = fftnd(image_full, axes=[dim_pf], )
        torch.moveaxis(kspace_full, dim_pf, 0)[mask] = torch.moveaxis(kspace_full_clone, dim_pf, 0)[mask] # replace elements of kspace_full from original kspace_full_clone

    kspace_full = fftnd(kspace_full, axes=dim_nonpf_enc,)
    # remove all samples that was not part of the original dataset (e.g. acceleartion)        
    mask = mask_clone
    mask[ind_one[0]%acc_pf::acc_pf] = True
    torch.moveaxis(kspace_full, dim_pf, 0)[~mask] = 0       

    return kspace_full


