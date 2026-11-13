import os
import warnings


def configure(deterministic: bool = True):
    """Configure torch for reproducible runs. Call before CUDA is initialized
    for CUBLAS_WORKSPACE_CONFIG to take effect."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
