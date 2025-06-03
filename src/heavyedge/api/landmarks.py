"""Landmark detection."""

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

__all__ = [
    "find_peak",
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
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     Y = next(data.profiles())
    >>> peak = find_peak(Y, 32)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(Y)
    ... plt.plot(peak, Y[peak], "o")
    """
    peaks, _ = find_peaks(gaussian_filter1d(Y, sigma))
    return peaks[-1]
