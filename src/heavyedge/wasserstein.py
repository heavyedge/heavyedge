"""
Wasserstein distance
--------------------

Wasserstein-related functions.
"""

import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ._wasserstein import optimize_q

__all__ = [
    "quantile",
    "wdist",
    "wmean",
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


def quantile(x, f, t):
    """Convert a probability distribution to a quantile function.

    Parameters
    ----------
    x : ndarray
        Points over which *f* is measured.
    f : ndarray
        The empirical probability density function.
    t : ndarray
        Points over which the quantile function will be measured.
        Must be strictly increasing from 0 to 1.

    Returns
    -------
    ndarray
        Quantile function of *f* over *t*.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.wasserstein import quantile
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     Y = next(data.profiles())
    ...     x = data.x()[:len(Y)]
    >>> f = Y / np.trapezoid(Y, x)
    >>> t = np.linspace(0, 1, 100)
    >>> Q = quantile(x[:len(f)], f, t)
    """
    G = cumulative_trapezoid(f, x, initial=0)
    return interp1d(G, x, bounds_error=False, fill_value=(x[0], x[-1]))(t)


@_deprecated("1.6", "HeavyEdge-Distance package")
def wdist(x1, f1, x2, f2, grid_num):
    r"""Wasserstein distance between two 1D probability distributions.

    .. deprecated:: 1.6
        This function will be removed in HeavyEdge 2.0.
        Use HeavyEdge-Distance package instead.

    .. math::

        d_W(f_1, f_2)^2 = \int^1_0 (Q_1(t) - Q_2(t))^2 dt

    where :math:`Q_i` is the quantile function of :math:`f_i`.

    Parameters
    ----------
    x1, f1 : ndarray
        The first empirical probability density function.
    x2, f2 : ndarray
        The second empirical probability density function.
    grid_num : int
        Number of sample points in [0, 1] to approximate the integral.

    Returns
    -------
    scalar
        Wasserstein distance.
    """
    grid = np.linspace(0, 1, grid_num)
    Q1 = quantile(x1, f1, grid)
    Q2 = quantile(x2, f2, grid)
    return np.trapezoid((Q1 - Q2) ** 2, grid) ** 0.5


def wmean(xs, fs, grid_num):
    """Fréchet mean of probability distrubutions using Wasserstein metric.

    Parameters
    ----------
    xs : list of ndarray
        Points over which each distribution in *fs* is measured.
    fs : list of ndarray
        Empirical probability density functions.
    grid_num : int
        Number of sample points in [0, 1] to approximate the integral.

    Returns
    -------
    x, f : ndarray
        Probability density function.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.wasserstein import wmean
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     (Y1, Y2), (L1, L2), _ = data[:2]
    >>> x1, Y1 = x[:L1], Y1[:L1]
    >>> x2, Y2 = x[:L2] + 3, Y2[:L2]
    >>> f1, f2 = Y1 / np.trapezoid(Y1, x1), Y2 / np.trapezoid(Y2, x2)
    >>> x, f = wmean([x1, x2], [f1, f2], 100)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x1, f1, "--", color="gray")
    ... plt.plot(x2, f2, "--", color="gray")
    ... plt.plot(x, f)
    """
    grid = np.linspace(0, 1, grid_num)
    Q = np.array([quantile(x, f, grid) for x, f in zip(xs, fs)])
    g = np.mean(Q, axis=0)
    if np.all(np.diff(g) >= 0):
        q = g
    else:
        q = optimize_q(g)
    pdf = 1 / np.gradient(q, grid)
    pdf[-1] = 0
    pdf /= np.trapezoid(pdf, q)
    return q, pdf
