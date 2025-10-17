from __future__ import annotations
from time import sleep
from typing import Optional
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future, wait
from queue import Queue
from threading import Thread

from .data_io import load_dat_basics
from .motion import prepare_tcl_data
from .utils import plot_concat_curves
from .gridding import correct_kspace_and_save
from .ipat import parallel_imaging

def Recon(working_path: str,
        Tcl: bool = False,
        Tcl_dir: Optional[str] = None,
        reverse_moco: bool = False,
        smooth_Tcl: bool = False,
        NavMOCO: bool = False,
        Nav_file: Optional[str] = None,
        noMOCO: bool = True,
        mv2center: bool = False,
        use_pocs: bool = True,
        nthreads: int = 1,
        device: str = "cuda:0",) -> None:
    """
    Main function to call for the reconstruction
    working_path    : path to the folder containing the .dat file
    Tcl             : whether to use Tcl motion correction or not
    Tcl_dir         : path to the folder containing the Tcl files
    reverse_moco    : whether to reverse the motion correction or not (only for Tcl prospective corrected data)
    smooth_Tcl      : whether to smooth the Tcl motion curves or not
    NavMOCO         : whether to use NavMOCO or not (not implemented yet)
    Nav_file        : path to the navigator file (not implemented yet)
    noMOCO          : whether to use the noMOCO (original reacq data) or not
    mv2center       : whether to move the motion to the center of kspace or not
    device          : device to use for the reconstruction (default: 'cuda:0')
    _debug_         : whether to use debug mode or not (more logging)
    """
    ## Load data
    dat_basics      = load_dat_basics(working_path, reverse_moco=reverse_moco, noMOCO=noMOCO)
    

    logging.info("Data loaded with shape: {} ! ".format(dat_basics.data_set.shape))
    
    ## Init moco_sys dict
    dat_basics.moco_sys = {}

    ## load tcl data
    if Tcl      :
        prepare_tcl_data(working_path, Tcl_dir, dat_basics, reverse_moco=reverse_moco, mv2center=mv2center, smooth_Tcl=smooth_Tcl)
        logging.info("Tcl MOCO matrices prepared and saved ! ")

    if NavMOCO  :
        raise NotImplementedError("NavMOCO implementation must be adapted to the new pipeline ! ")
    
    if noMOCO   :
        # create identity matrices for noMoco
        dat_basics.moco_sys['noMoco'] = np.tile(np.eye(4, dtype=np.float32),
                                        (dat_basics.rHrps[1]*dat_basics.rHrps[2]*dat_basics.nAve, 1, 1)
                                       ).reshape(dat_basics.rHrps[1], dat_basics.rHrps[2], dat_basics.nAve, 4, 4)

        logging.info("NoMoco - Identity array of shape {} created ! ".format(dat_basics.moco_sys['noMoco'].shape))

    ##############
    ## For comparison plot motion curves
    if (len(dat_basics.moco_sys) == 2 and 'noMoco' not in dat_basics.moco_sys.keys()) or len(dat_basics.moco_sys) > 2:
        plot_concat_curves(dat_basics, working_path)


    ## TODO: Create function for reconstruction (PF handling / iFFT) without passing trough non-Cartisian gridding
    ## even with identity matrices

    ## Begin iPAT reconstruction
    if nthreads > 1:
        use_multithreading = True
    else:
        use_multithreading = False
    
    to_grappa  = Queue()
    after_ipat = Queue()

    if max(dat_basics.Arps) != 1:
        if not dat_basics.ksp_exist["base"]:
            # first submit moco grappa if needed
            margs = {"dat_basics": dat_basics, "working_path": working_path,
                    "is_nomoco": False, "use_multithreading": use_multithreading,
                    "max_workers": nthreads}

            to_grappa.put(margs)
        
        # second submit noMoco grappa if needed
        if noMOCO and reverse_moco and not dat_basics.ksp_exist["nomoco"]:
            margs = {"dat_basics": dat_basics, "working_path": working_path,
                      "is_nomoco": True, "use_multithreading": use_multithreading,
                      "max_workers": nthreads}
            to_grappa.put(margs)
        
        if dat_basics.ksp_exist["base"]:
            #  base kspace already exists --> send directly to MOCO regrid
            for var in dat_basics.moco_sys:
                if var != 'noMoco':
                    margs = {"dat_basics": dat_basics, "var" : var, "working_path": working_path,
                              "is_nomoco": False, "use_pocs": use_pocs, "device": device}
                    after_ipat.put(margs)
        
        if dat_basics.ksp_exist["nomoco"]:
            #  nomoco kspace already exists --> send directly to MOCO regrid
            margs = {"dat_basics": dat_basics, "var" : 'noMoco', "working_path": working_path,
                      "is_nomoco": True, "use_pocs": use_pocs, "device": device}
            after_ipat.put(margs)

    else:
        #  no iPAT --> send directly to MOCO regrid
        for var in dat_basics.moco_sys:
            if var != 'noMoco':
                margs = {"dat_basics": dat_basics, "var" : var, "working_path": working_path,
                          "is_nomoco": False, "use_pocs": use_pocs, "device": device}
                after_ipat.put(margs)

        if noMOCO and reverse_moco:
            margs = {"dat_basics": dat_basics, "var": 'noMoco', "working_path": working_path,
                      "is_nomoco": True, "use_pocs": use_pocs, "device": device}
            after_ipat.put(margs)

    def grappa_worker():
        while True:
            try:
                args = to_grappa.get(timeout=1)
            except:
                break

            res = parallel_imaging(**args)
            if res == "moco":
                for var in dat_basics.moco_sys:
                    if var != 'noMoco':         
                        margs = {"dat_basics": dat_basics, "var": var, "working_path": working_path,
                                  "is_nomoco": False, "use_pocs": use_pocs, "device": device}
                        after_ipat.put(margs)
            else: ## nomoco
                margs = {"dat_basics": dat_basics, "var": 'noMoco', "working_path": working_path,
                          "is_nomoco": True, "use_pocs": use_pocs, "device": device}
                after_ipat.put(margs)

            to_grappa.task_done()
    
    if not to_grappa.empty():
        logging.info("Starting GRAPPA reconstruction ")
        threads     = []
        n_workers   = min(to_grappa.qsize(), 1)

        for _ in range(n_workers):
            t = Thread(target=grappa_worker)
            t.start()
            threads.append(t)
        
        # Wait for all tasks to be done
        to_grappa.join()

        # Stop workers
        for t in threads:
            t.join()
        
        logging.info("GRAPPA reconstruction done ! ")
    
    ## After ipat - regrid with MOCO
    logging.info("Starting final kspace correction and saving ")
    while not after_ipat.empty() or not to_grappa.empty() or (any(t.is_alive() for t in threads) if 'threads' in locals() else False):
        try:
            args = after_ipat.get(timeout=1)
        except:
            sleep(15)
            continue

        correct_kspace_and_save(**args)

        after_ipat.task_done()

    logging.info("All tasks done ! ")
