import sys

if sys.version_info < (3, 10):
	def __recon_placeholder(*args, **kwargs):
		raise RuntimeError(
			"mocokit requires Python >= 3.10. Detected Python {}.{}. "
			"Please run your code/tests with a Python 3.10+ interpreter.".format(
				sys.version_info.major, sys.version_info.minor
			)
		)

	Recon = __recon_placeholder
else:
	from .recon import Recon  # type: ignore
	

__all__ = ["Recon"]
__version__ = "0.1.0"