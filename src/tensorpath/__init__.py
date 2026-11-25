"""The analysis layer does not import PyTorch."""
__version__ = "0.2.1"


def __getattr__(name):
    if name == "Recorder":
        from .recorder import Recorder
        return Recorder
    if name == "capture_forward":
        from .capture import capture_forward
        return capture_forward
    raise AttributeError(name)


__all__ = ["Recorder", "capture_forward", "__version__"]
