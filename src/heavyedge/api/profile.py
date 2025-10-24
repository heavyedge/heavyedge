"""Profile preprocessing."""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress

from ..wasserstein import _wmean_old

__all__ = [
    "preprocess",
    "fill_after",
    "outlier",
    "mean",
]


def _deprecated(version, replace):
    removed_version = str(int(version.split(".")[0]) + 1) + ".0"

    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__}() is deprecated since HeavyEdge {version} "
                f"and will be removed in {removed_version}. "
                f"Use {replace} instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


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
    L : int
        Length of *Y* until the contact point.

    Notes
    -----
    Profiles undergo the following steps:

    1. Profile direction is set so that the contact point is on the right hand side.
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
    for i, peak_idx in enumerate(peaks):
        x = X[peak_idx:]
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

    # If any point before cp is lower than the detected contact point,
    # set that as contact point instead.
    cp = np.argmin(Y[: cp + 1])
    Y = Y - Y[cp]
    return Y, cp + 1


def fill_after(Ys, Ls, fill_value):
    """Fill arrays with a constant value after specified lengths.

    The input array *Ys* is modified.

    Parameters
    ----------
    Ys : (N, M) array
        Array of N profiles.
    Ls : (N,) array
        Length of each profile.
    fill_value : scalar
        Value to fill *Ys*.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import fill_after
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> fill_after(Ys, Ls, 0)
    """
    _, M = Ys.shape
    Ys[np.arange(M)[None, :] >= Ls[:, None]] = fill_value


def outlier(values, thres=3.5):
    """Detect outlier from scalar values.

    Parameters
    ----------
    x : array of scalar
        Target data.
    thres : scalar, default=3.5
        Z-score threshold for outlier detection.

    Returns
    -------
    is_outlier : array of bool
        Boolean array where True indicates outlier.

    Notes
    -----
    Outliers are detected by applying modified Z-score method [1]_ on *x*.

    References
    ----------
    .. [1] Boris Iglewicz and David C Hoaglin.
       Volume 16: how to detect and handle outliers. Quality Press, 1993.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import outlier
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x = data.x()
    ...     Ys, _, _ = data[:]
    ...     areas = np.trapezoid(Ys, x, axis=-1)
    >>> is_outlier = outlier(areas, 1.5)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... for Y, skip in zip(Ys, is_outlier):
    ...     if skip:
    ...         plt.plot(Y, color="red")
    ...     else:
    ...         plt.plot(Y, alpha=0.2, color="gray")
    """
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    mod_z = 0.6745 * (values - med) / mad
    return np.abs(mod_z) > thres


@_deprecated("1.6", "heavyedge.api.mean module")
def mean(x, profiles, grid_num):
    """Fréchet mean of profiles using Wasserstein distance.

    Parameters
    ----------
    x : ndarray
        X coordinates of *profiles*.
    profiles : list of array
        Profile data, with last point being the contact point.
    grid_num : int
        Number of sample points in [0, 1] to construct regression results.

    Returns
    -------
    ndarray
        Averaged *Y*.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import mean
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x = data.x()
    ...     profiles = list(data.profiles())
    >>> mean = mean(x, profiles, 100)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... for profile in profiles:
    ...     plt.plot(profile, alpha=0.2, color="gray")
    ... plt.plot(mean)
    """
    xs, areas, pdfs = [], [], []
    for prof in profiles:
        x_ = x[: len(prof)]
        A = np.trapezoid(prof, x_)
        xs.append(x_)
        areas.append(A)
        pdfs.append(prof / A)
    X, F = _wmean_old(xs, pdfs, grid_num)
    # Fix the last point of X to grid
    last_idx = np.argmin(np.abs(x - X[-1]))
    X[-1] = x[last_idx]

    shape = np.interp(x[: last_idx + 1], X, F)
    shape /= np.trapezoid(shape, x[: len(shape)])
    return shape * np.mean(areas)
