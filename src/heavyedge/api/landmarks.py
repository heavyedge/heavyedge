"""Landmark detection."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

__all__ = [
    "find_peak",
    "find_peak_trough",
]


def find_peak(Y, sigma):
    """Find heavy edge peak.

    Parameters
    ----------
    Y : 1-D array
        1-dimensional heavy edge profile data.
        The last point must be the contact point.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    peak : int
        Index of heavy edge peak.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import find_peak
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     Y = next(data.profiles())
    >>> peak = find_peak(Y, 32)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(Y)
    ... plt.plot(peak, Y[peak], "o")
    """
    peaks, _ = find_peaks(gaussian_filter1d(Y, sigma))
    return peaks[-1]


def find_peak_trough(Y, sigma):
    """Find heavy edge peak and trough.

    Parameters
    ----------
    Y : 1-D array
        1-dimensional heavy edge profile data with trough.
        The last point must be the contact point.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    peak, trough : int
        Index of heavy edge peak and trough.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import find_peak_trough
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     Y = next(data.profiles())
    >>> peak, trough = find_peak_trough(Y, 32)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(Y)
    ... plt.plot([peak, trough], Y[[peak, trough]], "o")
    """
    Y_smooth = gaussian_filter1d(Y, sigma)
    peaks, _ = find_peaks(Y_smooth)
    peak = peaks[-1]

    troughs, _ = find_peaks(-Y_smooth)
    troughs = troughs[troughs < peak]

    prominences = peak_prominences(-Y_smooth, troughs)[0]
    most_prominent_idx = np.argmax(prominences)
    trough = troughs[most_prominent_idx]
    return peak, trough
