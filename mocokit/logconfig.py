import logging
import sys

DEFAULT_FMT = ("%(asctime)s | %(levelname)s | %(name)s:%(funcName)s "
               "| pid=%(process)d | %(message)s")

DEFAULT_DATE = "%Y/%m/%d-%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    level   = logging.INFO if verbose else logging.WARNING
    
    logging.basicConfig(
        level   = level,
        format  = DEFAULT_FMT,
        datefmt = DEFAULT_DATE,
        handlers= [logging.StreamHandler(sys.stdout)],
        force   = True,
    )