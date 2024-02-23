import numpy as np


# start of a month to the accumulated days in a year
MONTH_TO_DAY = {
    1: 0,
    2: 31,
    3: 59,
    4: 90,
    5: 120,
    6: 151,
    7: 181,
    8: 212,
    9: 243,
    10: 273,
    11: 304,
    12: 334,
}


def compute_r_square(expected: np.ndarray, actual: np.ndarray):
    if expected.shape != actual.shape:
        raise ValueError(
            f"To compute R^2, expected and actual values must have same shape, "
            f"but got {expected.shape} and {actual.shape}"
        )
    deviation = actual - expected
    r_square = 1.0 - np.sum(deviation**2) / (np.sum(expected**2) - np.sum(expected)**2 / expected.shape[0])
    return r_square
