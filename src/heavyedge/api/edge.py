"""Edge manipulation."""

import numpy as np

from .profile import fill_after

__all__ = [
    "scale_area",
    "scale_plateau",
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

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import scale_area
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as f:
    ...     Ys = np.concatenate(list(scale_area(f, batch_size=5)), axis=0)
    """
    x = f.x()

    if batch_size is None:
        Ys, Ls, _ = f[:]
        Ys /= _area(x, Ys, Ls)[:, np.newaxis]
        logger("1/1")
        yield Ys
    else:
        N = len(f)
        for i in range(0, N, batch_size):
            Ys, Ls, _ = f[i : i + batch_size]
            Ys /= _area(x, Ys, Ls)[:, np.newaxis]
            logger(f"{i}/{N}")
            yield Ys


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

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import scale_plateau
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as f:
    ...     Ys = np.concatenate(list(scale_plateau(f, batch_size=5)), axis=0)
    """
    if batch_size is None:
        Ys, _, _ = f[:]
        Ys /= Ys[:, [0]]
        logger("1/1")
        yield Ys
    else:
        N = len(f)
        for i in range(0, N, batch_size):
            Ys, _, _ = f[i : i + batch_size]
            Ys /= Ys[:, [0]]
            logger(f"{i}/{N}")
            yield Ys
