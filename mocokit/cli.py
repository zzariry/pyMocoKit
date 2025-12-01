import argparse
from .recon import Recon
from .logconfig import setup_logging
from .runtime import configure_runtime


DESCRIPTION = """
    Version Dec 01st, 2025 - by Z.Z :
    Pipeline to reconstruct MRI image with motion correction (TCL + GRAPPA + NUFFT)\n\n.
        Usage: 
    mocokit i /path/to/folder/dat 
    -tcl -td /path/to/tcl_dir -reverse -smooth 
    -orig 
    -center
    -no_pocs
    -device cuda:0 
    --cuda-visible-devices 0 
    --headless 
    --numpy-precision 6 
    """

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION, 
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("-i", dest="working_path", required=True,
                    help='Path of the input diffusion volume.')
    p.add_argument("-tcl", dest="Tcl", action="store_true",
                    help='Moco using TCL - must specify TCL dir using -td')
    p.add_argument("-td", dest="Tcl_dir",
                    help='Path to the TCL dir for motions')
    p.add_argument("-reverse", dest="reverse_moco", action="store_true",
                    help='Reverse phase-encode')
    p.add_argument("-smooth", dest="smoothTcl", action="store_true",
                    help='Smooth Tcl motion')
    p.add_argument("-nav", dest="NavMOCO", action="store_true",
                    help='Moco using nav - must specify .mat matrix file path using -nd')
    p.add_argument("-nd", dest="Nav_file",
                    help='Path to the nav (.mat) path for motions')
    p.add_argument("-orig", dest="noMOCO", action="store_true",
                    help='Output original image')
    p.add_argument("-orig_noreacq", dest="no_reacq", action="store_true",
                    help='Output noMOCO image without reacquisition - can only be used with -orig and -reverse')
    p.add_argument("-center", dest="mv2center", action="store_true",
                    help='Transform all motion estimates to kspace center sampling - tcl or nav moco must be on')
    p.add_argument("-no_pocs", dest="use_pocs", action="store_false",
                    help='Do not use POCS for PF 0 lines if any - default is to use POCS')
    p.add_argument("-nthreads", dest="nthreads", type=int, default=1,
                    help='Number of threads to use for parallel imaging (GRAPPA) - default is 1 (no multithreading)')
    p.add_argument("-device", dest="device", default="cuda:0",
                    help='Device to use (cpu/gpu_id), e.g., cuda:0')
    
    # General options
    p.add_argument("--cuda-visible-devices", dest="cuda", default=None)
    p.add_argument("--headless", dest="headless", action="store_true")
    p.add_argument("--no-headless", dest="no_headless", action="store_true")
    p.add_argument("--numpy-precision", type=int, default=6)
    p.add_argument("--pydev-warn-timeout", dest="pydev_timeout", default=None)
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    # p.add_argument("-h", action="help", help="Show this help message and exit.")
    return p



def main():
    
    if False:  # For debugging purpose
        setup_logging(verbose=True)
        ## mocokit -i /home/mribbdev/rawdata/sub-201/ses-20221103/test -tcl -td /home/mribbdev/rawdata/sub-201/ses-20221103/test/Tcl_log -reverse -orig -v --cuda-visible-devices 0
        Recon(working_path   = "/home/mribbdev/rawdata/sub-201/ses-20221103/test_t1",
              Tcl            = True,
              Tcl_dir        = "/home/mribbdev/rawdata/sub-201/ses-20221103/test_t1/Tcl_log",
              reverse_moco   = True,
              smooth_Tcl     = False,
              NavMOCO        = False,
              Nav_file       = None,
              noMOCO         = True,
              mv2center      = False,
              use_pocs       = True,
              nthreads       = 20,
              device         = "cuda:0"
              )
        return
    
    args    = buildArgsParser().parse_args()

    headless = True if args.headless else (False if args.no_headless else None)

    configure_runtime(
        cuda_visible_devices= args.cuda,
        headless            = headless,
        numpy_precision     = args.numpy_precision)
    
    setup_logging(verbose=args.verbose)


    Recon(working_path   = args.working_path,
          Tcl            = args.Tcl,
          Tcl_dir        = args.Tcl_dir,
          reverse_moco   = args.reverse_moco,
          smooth_Tcl     = args.smoothTcl,
          NavMOCO        = args.NavMOCO,
          Nav_file       = args.Nav_file,
          noMOCO         = args.noMOCO,
          no_reacq       = args.no_reacq,
          mv2center      = args.mv2center,
          use_pocs       = args.use_pocs,
          nthreads       = args.nthreads,
          device         = args.device)
    
if __name__ == "__main__":
    main()