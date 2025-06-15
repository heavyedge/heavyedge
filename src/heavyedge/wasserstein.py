"""
Wasserstein distance
--------------------

Wasserstein-related functions.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ._wasserstein import optimize_q

__all__ = [
    "quantile",
    "wdist",
    "wmean",
]


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


def wdist(x1, f1, x2, f2, grid_num):
    r"""Wasserstein distance between two 1D probability distributions.

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

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.wasserstein import wdist
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     (Y1, Y2), (L1, L2), _ = data[:2]
    >>> x1, Y1 = x[:L1], Y1[:L1]
    >>> x2, Y2 = x[:L2], Y2[:L2]
    >>> f1, f2 = Y1 / np.trapezoid(Y1, x1), Y2 / np.trapezoid(Y2, x2)
    >>> d = wdist(x1, f1, x2, f2, 100)
    """
    grid = np.linspace(0, 1, grid_num)
    Q1 = quantile(x1, f1, grid)
    Q2 = quantile(x2, f2, grid)
    return np.trapezoid((Q1 - Q2) ** 2, grid) ** 0.5


def wmean(Y, grid_num):
    """Fréchet mean of probability distrubution functions using Wasserstein metric.

    Parameters
    ----------
    Y : list of array
        Probability distribution functions.
    grid_num : int
        Number of sample points in [0, 1] to construct regression results.

    Returns
    -------
    ndarray
        Averaged *Y*.
    """
    grid = np.linspace(0, 1, grid_num)
    Q = np.array([quantile(np.arange(len(y)), y, grid) for y in Y])
    g = np.mean(Q, axis=0)

    if np.all(np.diff(g) >= 0):
        q = g
    else:
        q = optimize_q(g)
    cdf = np.interp(np.arange(int(q[-1])), q, np.linspace(0, 1, len(q)))
    return np.concatenate([np.diff(cdf), [0]])
