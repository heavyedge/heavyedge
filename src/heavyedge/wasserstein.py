"""
Wasserstein distance
--------------------

Wasserstein-related functions.
"""

import numpy as np

from ._wasserstein import optimize_q

__all__ = [
    "quantile",
    "wdist",
    "wmean",
]


def quantile(x, f, t):
    """Convert probability mass function to quantile function.

    Parameters
    ----------
    x : ndarray
        Random variable over which *f* is measured.
    f : ndarray
        An 1D array of probability mass function.
    t : ndarray
        An 1D array of t values for quantile function. Must be strictly increasing
        from 0 to 1.

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
    ...     x = data.x()
    ...     Y = next(data.profiles())
    >>> f = Y / Y.sum()
    >>> t = np.linspace(0, 1, 100)
    >>> Q = quantile(x[:len(f)], f, t)
    """
    G = np.insert(np.cumsum(f), 0, 0)
    x = np.insert(x, 0, x[0] - (x[1] - x[0]))
    Q = np.interp(t, G, x)
    Q[-1] = x[-1]
    return Q


def wdist(x1, f1, x2, f2, grid_num):
    r"""Wasserstein distance between two 1D probability mass functions.

    .. math::

        d_W(f_1, f_2)^2 = \int^1_0 (Q_1(t) - Q_2(t))^2 dt

    where :math:`Q_i` is the quantile function of :math:`f_i`.

    Parameters
    ----------
    x1, f1 : ndarray
        The first random variable and probability mass function.
    x2, f2 : ndarray
        The second random variable and probability mass function.
    grid_num : int
        Number of sample points in [0, 1] to approximate the integral.

    Returns
    -------
    scalar
        Wasserstein distance.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.wasserstein import wdist
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     (Y1, Y2), (L1, L2), _ = data[:2]
    >>> x1, f1 = x[:L1], Y1[:L1] / Y1[:L1].sum()
    >>> x2, f2 = x[:L2] + 3, Y2[:L2] / Y2[:L2].sum()
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
