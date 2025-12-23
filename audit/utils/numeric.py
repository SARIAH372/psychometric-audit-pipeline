import numpy as np


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    """Divide with protection against zero division."""
    if b == 0:
        return default
    return a / b
