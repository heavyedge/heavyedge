"""Profile preprocessing."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress

__all__ = [
    "preprocess",
]


def preprocess(Y, sigma, std_thres):
    """Preprocess raw profile data.

    Parameters
    ----------
    Y : 1-D array
        1-dimensional profile data.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.
    std_thres : scalar
        Standard deviation threshold to detect contact point.

    Returns
    -------
    Y : 1-D array
        Preprocessed profile data.
    cp : int
        Index of contact point in *Y*.

    Notes
    -----
    Profiles undergo the following steps:

    1. Profile direction is set so that contact point is on the right hand side.
    2. Contact point is detected, and set to have zero height.

    Examples
    --------
    >>> from heavyedge import get_sample_path, RawProfileCsvs
    >>> from heavyedge.api import preprocess
    >>> Y = next(RawProfileCsvs(get_sample_path("Type3")).profiles())
    >>> Y, L = preprocess(Y, 32, 0.01)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(Y[:L])
    """
    if Y[0] < Y[-1]:
        # Make plateau is on the left and cp is on the right
        Y = np.flip(Y)

    X = np.arange(len(Y))
    h_xx = gaussian_filter1d(Y, sigma, order=2, mode="nearest")
    if len(h_xx) > 0:
        peaks, _ = find_peaks(h_xx)
    else:
        peaks = np.empty(0, dtype=int)

    candidates = []
    for i, x in enumerate(peaks):
        x = X[x:]
        if not len(x) > 2:
            continue
        y = Y[x]
        reg = linregress(x, y)
        residuals = y - (reg.intercept + reg.slope * x)
        std = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        if std < std_thres:
            candidates.append(i)

    if candidates:
        cp = peaks[candidates[np.argmax(h_xx[peaks[candidates]])]]
    else:
        cp = len(Y) - 1

    Y = Y - Y[cp]
    return Y, cp
