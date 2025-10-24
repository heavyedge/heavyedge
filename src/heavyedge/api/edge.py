"""Edge manipulation."""

import numpy as np

from .profile import fill_after

__all__ = [
    "scale_area",
    "scale_plateau",
    "trim",
    "pad",
]


def scale_area(f, batch_size=None, logger=lambda x: None):
    """Scale edge profile by area.

    Parameters
    ----------
    f : heavyedge.ProfileData
        Open h5 file of profiles.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Yields
    ------
    scaled : (batch_size, M) array
        Scaled edge profile.
    Ls : (batch_size,) array
        Lengths of the scaled profiles.
    names : (batch_size,) array
        Names of the scaled profiles.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import scale_area
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as f:
    ...     gen = scale_area(f, batch_size=5)
    ...     Ys = np.concatenate([ys for ys, _, _ in gen], axis=0)
    """
    x = f.x()

    N = len(f)
    if batch_size is None:
        Ys, Ls, names = f[:]
        Ys /= _area(x, Ys, Ls)[:, np.newaxis]
        logger(f"{N}/{N}")
        yield Ys, Ls, names
    else:
        for i in range(0, N, batch_size):
            Ys, Ls, names = f[i : i + batch_size]
            Ys /= _area(x, Ys, Ls)[:, np.newaxis]
            logger(f"{i}/{N}")
            yield Ys, Ls, names


def _area(x, Ys, Ls):
    Ys = Ys.copy()
    fill_after(Ys, Ls, 0)
    return np.trapezoid(Ys, x, axis=1)


def scale_plateau(f, batch_size=None, logger=lambda x: None):
    """Scale edge profile by plateau height.

    Parameters
    ----------
    f : heavyedge.ProfileData
        Open h5 file of profiles.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Yields
    ------
    scaled : (batch_size, M) array
        Scaled edge profile.
    Ls : (batch_size,) array
        Lengths of the scaled profiles.
    names : (batch_size,) array
        Names of the scaled profiles.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import scale_plateau
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as f:
    ...     gen = scale_plateau(f, batch_size=5)
    ...     Ys = np.concatenate([ys for ys, _, _ in gen], axis=0)
    """
    N = len(f)
    if batch_size is None:
        Ys, Ls, names = f[:]
        Ys /= Ys[:, [0]]
        logger(f"{N}/{N}")
        yield Ys, Ls, names
    else:
        for i in range(0, N, batch_size):
            Ys, Ls, names = f[i : i + batch_size]
            Ys /= Ys[:, [0]]
            logger(f"{i}/{N}")
            yield Ys, Ls, names


def trim(f, width, batch_size=None, logger=lambda x: None):
    """Trim edge profile to a specific width.

    Parameters
    ----------
    f : heavyedge.ProfileData
    width : scalar
        Length to trim the profile to.
        Must be of physical length by `f.x()`.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Yields
    ------
    trimmed : (batch_size, width) array
        Trimmed edge profile.
    Ls : (batch_size,) array
        Lengths of the scaled profiles.
    names : (batch_size,) array
        Names of the scaled profiles.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import trim
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as f:
    ...     gen = trim(f, 5, batch_size=10)
    ...     Ys = np.concatenate([ys for ys, _, _ in gen], axis=0)
    """
    N, M = f.shape()
    Ls = f._file["len"][:]
    width_num = int(f.resolution() * width)
    substrate_num = (M - Ls).min()

    if batch_size is None:
        Ys, Ls, names = f[:]
        logger(f"{N}/{N}")
        yield _trim(Ys, Ls, width_num, substrate_num), Ls, names
    else:
        for i in range(0, N, batch_size):
            Ys, Ls, names = f[i : i + batch_size]
            logger(f"{i}/{N}")
            yield _trim(Ys, Ls, width_num, substrate_num), Ls, names


def _trim(Ys, Ls, w1, w2):
    N, M = Ys.shape
    idx_array = np.arange(M)[None, :]
    mask1 = idx_array >= (Ls - w1)[:, None]
    mask2 = idx_array < (Ls + w2)[:, None]
    ret = Ys[mask1 & mask2].reshape(N, w1 + w2)
    return ret


def pad(f, width=None, batch_size=None, logger=lambda x: None):
    """Pad edge profile to a specific width.

    Parameters
    ----------
    f : heavyedge.ProfileData
    width : int
        Length to pad the profile to.
        Must be of physical length by `f.x()`.
        If not passed, set to the length of the shortest profile.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Yields
    ------
    padded : (batch_size, width) array
        Padded edge profile.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import pad
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as f:
    ...     Ys = np.concatenate(list(pad(f, 20, batch_size=10)), axis=0)
    """
    N, M = f.shape()
    Ls = f._file["len"][:]
    if width is None:
        width_num = Ls.max()
    else:
        width_num = int(f.resolution() * width)
    substrate_num = (M - Ls).min()

    if batch_size is None:
        Ys, Ls, _ = f[:]
        logger(f"{N}/{N}")
        yield _pad(Ys, Ls, width_num, substrate_num)
    else:
        for i in range(0, N, batch_size):
            Ys, Ls, _ = f[i : i + batch_size]
            logger(f"{i}/{N}")
            yield _pad(Ys, Ls, width_num, substrate_num)


def _pad(Ys, Ls, w1, w2):
    N, M = Ys.shape
    ret = np.empty((N, w1 + w2), dtype=Ys.dtype)

    ret[:w1] = Ys[:, :1]
    ret_mask = np.arange(w1 + w2)[None, :] >= (w1 - Ls)[:, None]
    Ys_mask = np.arange(M)[None, :] < (w2 + Ls)[:, None]
    ret[ret_mask] = Ys[Ys_mask]
    return ret
