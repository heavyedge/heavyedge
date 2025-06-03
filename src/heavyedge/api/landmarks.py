"""Landmark detection."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

__all__ = [
    "landmarks_type2",
    "landmarks_type3",
]


def landmarks_type2(Y, sigma):
    """Find landmarks for heavy edge profile without trough.

    Parameters
    ----------
    Y : 1-D array
        1-dimensional heavy edge profile data.
        The last point must be the contact point.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    landmarks : (3,) array of int
        Indices of landmarks.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import landmarks_type2
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     Y = next(data.profiles())
    >>> lm = landmarks_type2(Y, 32)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(Y)
    ... plt.plot(lm, Y[lm], "o")
    """
    cp = len(Y) - 1

    Y_smooth = gaussian_filter1d(Y, sigma)
    peaks, _ = find_peaks(Y_smooth)
    peak = peaks[-1]

    Y_ = Y_smooth[:peak]
    pts = np.column_stack([np.arange(len(Y_)), Y_])
    x, y = pts - pts[0], pts[-1] - pts[0]
    dists = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    slope = np.diff(dists)
    (extrema,) = np.nonzero(np.diff(np.sign(slope)))
    K_pos = extrema[slope[extrema] > 0]
    knee = K_pos[np.argmax(np.abs(dists[K_pos]))]

    return np.array([cp, peak, knee])


def landmarks_type3(Y, sigma):
    """Find landmarks for heavy edge profile with trough.

    Parameters
    ----------
    Y : 1-D array
        1-dimensional heavy edge profile data.
        The last point must be the contact point.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    landmarks : (4,) array of int
        Indices of landmarks.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import landmarks_type3
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     Y = next(data.profiles())
    >>> lm = landmarks_type3(Y, 32)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(Y)
    ... plt.plot(lm, Y[lm], "o")
    """
    cp = len(Y) - 1

    Y_smooth = gaussian_filter1d(Y, sigma)
    peaks, _ = find_peaks(Y_smooth)
    peak = peaks[-1]

    troughs, _ = find_peaks(-Y_smooth)
    troughs = troughs[troughs < peak]

    if len(troughs) > 0:
        prominences = peak_prominences(-Y_smooth, troughs)[0]
        most_prominent_idx = np.argmax(prominences)
        trough = troughs[most_prominent_idx]

        Y_ = Y_smooth[: int(trough) + 1]
        pts = np.column_stack([np.arange(len(Y_)), Y_])
        x, y = pts - pts[0], pts[-1] - pts[0]
        dists = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
        slope = np.diff(dists)
        (extrema,) = np.nonzero(np.diff(np.sign(slope)))
        K_neg = extrema[slope[extrema] < 0]
        knee = K_neg[np.argmax(np.abs(dists[K_neg]))]

    else:
        Y_ = Y_smooth[:peak]
        pts = np.column_stack([np.arange(len(Y_)), Y_])
        x, y = pts - pts[0], pts[-1] - pts[0]
        dists = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
        slope = np.diff(dists)
        (extrema,) = np.nonzero(np.diff(np.sign(slope)))
        K_pos = extrema[slope[extrema] > 0]
        knee, trough = K_pos[np.argmax(np.abs(dists[K_pos]))]

    return np.array([cp, peak, trough, knee])
