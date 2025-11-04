r"""EMT Decode"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("deside")
except PackageNotFoundError:
    __version__ = "0.1-dev"
