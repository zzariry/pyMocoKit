## utils.py
import logging
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.gridspec import SubplotSpec
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import math
import csv
from scipy import signal




def saveImage(data : np.ndarray, affinematrix : np.ndarray, path : str) -> None: 
    '''A nibabel function to save images)''' 
    img_out     = nib.Nifti1Image(data, affinematrix)
    nib.save(img_out, path)

def scalingArray(dataIn : np.ndarray, minIm : float, maxIm : float) -> np.ndarray:
    '''A function for the scaling and the casting of 3D'''
 
    m = maxIm - minIm
    scalingFactor = 4095. / m
    b  = - scalingFactor*minIm
    dataScaled = (1.0*dataIn)*scalingFactor + b 
    dataOut = np.float32(dataScaled.copy())

    return dataOut

def full_extent(ax : plt.Axes, pady : float = -0.0, padx : float = -0.0):

    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    #items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items] )
    
    return bbox.expanded(1.0 + padx, 1.0 + pady)   

def rotmat2vec(POA : np.ndarray, CST: np.ndarray = None) -> list:
    if np.all(CST != None):
        A = CST @ np.linalg.inv(POA) @ np.linalg.inv(CST)
        #return Tx, Ty, Tz, Rx, Ry, Rz
        return [A[0,3], A[1,3], A[2,3], \
            math.degrees(-math.atan2(A[1,2], A[1,1])), \
                math.degrees(-math.atan2(A[2,0], A[0,0])),\
                    math.degrees(math.asin(A[1,0]))]
    
    else: 
        A = POA
        return [A[0,3], A[1,3], A[2,3], \
            math.degrees(-math.atan2(A[1,2], A[1,1])), \
                math.degrees(-math.atan2(A[2,0], A[0,0])),\
                    math.degrees(math.asin(A[1,0]))] ## on Y axis , i added - to A[2,0] to match direction ! i have to check it !)



def plot_curves(mat2plot  : np.ndarray, 
                list_x          : np.ndarray, 
                center_ro       : np.ndarray, 
                outputDir       : str, 
                sys_name        : str,
                smooth_curves   : bool = False) -> None:
    
    Tx, Ty, Tz, Rx, Ry, Rz  = np.transpose(np.array([rotmat2vec(mat) 
                                                     for mat in np.transpose(mat2plot, (2,0,1))]))
    
    with open(os.path.join(outputDir, '{}_curves.csv'.format(sys_name)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (sec)", "Rx", "Ry", "Rz", "Tx", "Ty", "Tz"])
        writer.writerows([[t, rx, ry, rz, tx, ty, tz] for t, rx, ry, rz, tx, ty, tz in zip(list_x, Rx, Ry, Rz, Tx, Ty, Tz)])

    if smooth_curves:
        # Preparing Butterworth filter parameters
        fs      = 1/(30*1e-3)   # Sampling frequency 30ms --> Hz
        fc      = 0.1           # Cut-off frequency - Hz
        w       = fc / (fs/2)   # Normalize the frequency
        forder  = 4
        b, a    = signal.butter(forder, w, 'low')

        ## applying Low pass butterworth filter
        Tx, Ty, Tz, Rx, Ry, Rz = map(lambda x: signal.filtfilt(b, a, x), [Tx, Ty, Tz, Rx, Ry, Rz])

    ### Compute motion score ###
    Radius  = 100 # mm
    Mosco   = np.zeros(1)
    for i in range(1, mat2plot.shape[2]):
        Mat     = mat2plot[..., i] @ np.linalg.inv(mat2plot[..., i-1]) - np.eye(4)

        Mosco   = np.append(Mosco, 
                            math.sqrt(((Radius**2)*1/5)*np.trace(Mat[:3, :3].T@Mat[:3, :3]) \
                                        + Mat[:3, 3].T@Mat[:3, 3]))
        
    #####################################################

    YminR = np.min(np.array([np.min(Rx-Rx[0]), np.min(Ry-Ry[0]), np.min(Rz-Rz[0])]))
    YminT = np.min(np.array([np.min(Tx-Tx[0]), np.min(Ty-Ty[0]), np.min(Tz-Tz[0])]))
    YmaxR = np.max(np.array([np.max(Rx-Rx[0]), np.max(Ry-Ry[0]), np.max(Rz-Rz[0])])) 
    YmaxT = np.max(np.array([np.max(Tx-Tx[0]), np.max(Ty-Ty[0]), np.max(Tz-Tz[0])]))

    MoscoR   = ((Mosco - np.min(Mosco))/(np.max(Mosco)-np.min(Mosco)))*(YmaxR-0) + 0
    MoscoT   = ((Mosco - np.min(Mosco))/(np.max(Mosco)-np.min(Mosco)))*(YmaxT-0) + 0

    fig = plt.figure()
    fig.set_size_inches(25.60, 14.40)

    ax1 = fig.add_subplot(3,1,1, 
                          xlabel=' ', 
                          ylabel='Rotation (Degree)', 
                          title='3D motion curve - {}'.format(sys_name), 
                          facecolor='white')
    
    ax1.plot(list_x, Rx-Rx[0], label='Rx')
    ax1.plot(list_x, Ry-Ry[0], label='Ry')
    ax1.plot(list_x, Rz-Rz[0], label='Rz')
    ax1.fill_between(list_x, MoscoR, label='Motion_score', alpha=0.1, color='grey')
    ax1.fill_between(center_ro, YminR, YmaxR, alpha=0.1)
    
    #ax1.set_yscale(10)
    ax1.grid()
    ax1.legend()
    for tmp in ax1.get_xticklabels()+ax1.get_yticklabels():
        tmp.set_fontsize(16)
    ax1.autoscale(True, 'both', True)
    
    ax2 = fig.add_subplot(3,1,2, 
                          xlabel=' ', 
                          ylabel='Translation (mm)', 
                          title='3D motion curve - {}'.format(sys_name))

    ax2.plot(list_x, Tx-Tx[0], label='Tx')
    ax2.plot(list_x, Ty-Ty[0], label='Ty')
    ax2.plot(list_x, Tz-Tz[0], label='Tz')
    ax2.fill_between(list_x, MoscoT, label='Motion_score', alpha=0.1, color='grey')
    ax2.fill_between(center_ro, YminT, YmaxT, alpha=0.1)
    #ax2.set_yscale(10)
    ax2.grid()
    ax2.legend()
    for tmp in ax2.get_xticklabels()+ax2.get_yticklabels():
        tmp.set_fontsize(16)
    ax2.autoscale(True, 'both', True)
    
    ax3 = fig.add_subplot(3,1,3, 
                          xlabel=' ', 
                          ylabel='Motion score', 
                          title='Motion score - {}'.format(sys_name))

    ax3.plot(list_x, Mosco, label='mm')
    ax3.grid()

    ax3.autoscale(True, 'both', True)

    ## Save
    extent = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(outputDir, 'Rot_{}.png'.format(sys_name)), bbox_inches=extent)
    extent = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(outputDir, 'Trans_{}.png'.format(sys_name)), bbox_inches=extent)
    extent = full_extent(ax3).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(outputDir, '{}_Motion_score.png'.format(sys_name)), bbox_inches=extent)

    logging.info("Motion curves for {} saved !".format(sys_name))
    ## For concat curves
    # to_exp = {
    #     'YminR': YminR, 'YminT': YminT, 'YmaxR':YmaxR, 'YmaxT': YmaxT, 
    #     'list_x': list_x, 
    #     'Rx': Rx, 'Ry': Ry, 'Rz': Rz,
    #     'Tx': Tx, 'Ty': Ty, 'Tz': Tz }
    
    # return to_exp

def plot_concat_curves(dat_basics, working_path):

    raise NotImplementedError("This function must be modified to match the new algorithm structure !")
    # yminR = np.min([dat_basics.moco_sys[var]['YminR'] for var in dat_basics.moco_sys.keys() if var != 'noMoco'])
    # ymaxR = np.max([dat_basics.moco_sys[var]['YmaxR'] for var in dat_basics.moco_sys.keys() if var != 'noMoco'])
    # yminT = np.min([dat_basics.moco_sys[var]['YminT'] for var in dat_basics.moco_sys.keys() if var != 'noMoco'])
    # ymaxT = np.max([dat_basics.moco_sys[var]['YmaxT'] for var in dat_basics.moco_sys.keys() if var != 'noMoco'])

    fig = plt.figure()
    fig.set_size_inches(25.60,14.40)

    ax1 = fig.add_subplot(2,1,1, xlabel=' ', ylabel='Rotation X (Degree)', title='3D motion curve ', facecolor='whitesmoke')
    [ax1.plot(dat_basics.moco_sys[var]['list_x'], 
                dat_basics.moco_sys[var]['Rx']-dat_basics.moco_sys[var]['Rx'][0], 
                label='{}_Rx'.format(dat_basics.moco_sys[var]['name'])) 
                for var in dat_basics.moco_sys.keys() if var != 'noMoco']

    # ax1.fill_between(C_lines, yminR, ymaxR, alpha=1)
    
    #ax1.set_yscale(10)
    ax1.grid()
    ax1.legend()
    for tmp in ax1.get_xticklabels()+ax1.get_yticklabels():
        tmp.set_fontsize(16)
    ax1.autoscale(True, 'both', True)
    
    ax2 = fig.add_subplot(2,1,2, xlabel=' ', ylabel='Translation X (mm)', title='3D motion curve ')
    [ax2.plot(dat_basics.moco_sys[var]['list_x'], 
                dat_basics.moco_sys[var]['Tx']-dat_basics.moco_sys[var]['Tx'][0], 
                label='{}_Tx'.format(dat_basics.moco_sys[var]['name'])) 
                for var in dat_basics.moco_sys.keys() if var != 'noMoco']

    # ax2.fill_between(C_lines, yminT, ymaxT, alpha=1)
    #ax2.set_yscale(10)
    ax2.grid()
    ax2.legend()
    for tmp in ax2.get_xticklabels()+ax2.get_yticklabels():
        tmp.set_fontsize(16)
    ax2.autoscale(True, 'both', True)
    
    ## Save
    extent = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(working_path+'RotX_combined.png', bbox_inches=extent)
    extent = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(working_path+'TransX_combined.png', bbox_inches=extent)

    ###################################
    ######################################################

    fig = plt.figure()
    fig.set_size_inches(25.60,14.40)

    ax1 = fig.add_subplot(2,1,1, xlabel=' ', ylabel='Rotation Y (Degree)', title='3D motion curve ', facecolor='whitesmoke')
    [ax1.plot(dat_basics.moco_sys[var]['list_x'], 
                dat_basics.moco_sys[var]['Ry']-dat_basics.moco_sys[var]['Ry'][0], 
                label='{}_Ry'.format(dat_basics.moco_sys[var]['name'])) 
                for var in dat_basics.moco_sys.keys() if var != 'noMoco']

    # ax1.fill_between(C_lines, yminR, ymaxR, alpha=1)
    
    #ax1.set_yscale(10)
    ax1.grid()
    ax1.legend()
    for tmp in ax1.get_xticklabels()+ax1.get_yticklabels():
        tmp.set_fontsize(16)
    ax1.autoscale(True, 'both', True)
    
    ax2 = fig.add_subplot(2,1,2, xlabel=' ', ylabel='Translation Y (mm)', title='3D motion curve ')
    [ax2.plot(dat_basics.moco_sys[var]['list_x'], 
                dat_basics.moco_sys[var]['Ty']-dat_basics.moco_sys[var]['Ty'][0], 
                label='{}_Ty'.format(dat_basics.moco_sys[var]['name'])) 
                for var in dat_basics.moco_sys.keys() if var != 'noMoco']

    # ax2.fill_between(C_lines, yminT, ymaxT, alpha=1)
    #ax2.set_yscale(10)
    ax2.grid()
    ax2.legend()
    for tmp in ax2.get_xticklabels()+ax2.get_yticklabels():
        tmp.set_fontsize(16)
    ax2.autoscale(True, 'both', True)
    
    ## Save
    extent = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(working_path+'RotY_combined.png', bbox_inches=extent)
    extent = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(working_path+'TransY_combined.png', bbox_inches=extent)

    #####################################
    #######################################################
    fig = plt.figure()
    fig.set_size_inches(25.60,14.40)

    ax1 = fig.add_subplot(2,1,1, xlabel=' ', ylabel='Rotation Z (Degree)', title='3D motion curve ', facecolor='whitesmoke')
    [ax1.plot(dat_basics.moco_sys[var]['list_x'], 
                dat_basics.moco_sys[var]['Rz']-dat_basics.moco_sys[var]['Rz'][0], 
                label='{}_Rz'.format(dat_basics.moco_sys[var]['name'])) 
                for var in dat_basics.moco_sys.keys() if var != 'noMoco']

    # ax1.fill_between(C_lines, yminR, ymaxR, alpha=1)
    
    #ax1.set_yscale(10)
    ax1.grid()
    ax1.legend()
    for tmp in ax1.get_xticklabels()+ax1.get_yticklabels():
        tmp.set_fontsize(16)
    ax1.autoscale(True, 'both', True)
    
    ax2 = fig.add_subplot(2,1,2, xlabel=' ', ylabel='Translation Z (mm)', title='3D motion curve ')
    [ax2.plot(dat_basics.moco_sys[var]['list_x'], 
                dat_basics.moco_sys[var]['Tz']-dat_basics.moco_sys[var]['Tz'][0], 
                label='{}_Tz'.format(dat_basics.moco_sys[var]['name'])) 
                for var in dat_basics.moco_sys.keys() if var != 'noMoco']    

    # ax2.fill_between(C_lines, yminT, ymaxT, alpha=1)
    #ax2.set_yscale(10)
    ax2.grid()
    ax2.legend()
    for tmp in ax2.get_xticklabels()+ax2.get_yticklabels():
        tmp.set_fontsize(16)
    ax2.autoscale(True, 'both', True)
    
    ## Save
    extent = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(working_path+'RotZ_combined.png', bbox_inches=extent)
    extent = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(working_path+'TransZ_combined.png', bbox_inches=extent)


## Some additional functions for plotting kspace

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold', fontsize=15 )
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')


def plot_kspace(data : np.ndarray, 
                kspPos : int,
                dname: str,
                func=fftn) -> None :
    
    """ Plot data in first row and func(data) in second row
    data    : 3D array containing [raw, line, slice], dtype must be np.complex64
    kspPos  : 3D array containing [raw_idx, line_index, slice_index]
    dname   : name of the data
    func    : function to apply on data (default: fftn)
    """

    mcmap   = plt.cm.gray
    grid    = plt.GridSpec(2, 3)
    
    ### Save ksp ###
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 12),)
    fig.set_size_inches(25.60, 14.40)
    
    ax = axes.ravel()
    for mtitle, i in zip(['YZ', 'XZ', 'XY']*2, np.arange(6)):
        #ax[i] = fig.add_subplot(2, 3, i+1)
        ax[i].set_title("Plotting {}".format(mtitle))
        if i > 2:
            ax[i].imshow(
                np.absolute(data.take(kspPos[i-3], axis=(i-3))), 
                    cmap=mcmap)
        else:
            ax[i].imshow(
                np.absolute
                (ifftshift(func(fftshift(data), norm='ortho'))
                    .take(kspPos[i], axis=i)), 
                    cmap=mcmap)            
    
    create_subtitle(fig, grid[0, :], "data")
    create_subtitle(fig, grid[1, :], "func(data)")
    fig.tight_layout()
    fig.savefig(dname+"_plot.png")
