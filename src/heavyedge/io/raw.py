"""Raw data files."""

import abc
import csv

import numpy as np

__all__ = [
    "RawProfileBase",
    "RawProfileCsvs",
]


class RawProfileBase(abc.ABC):
    """Base class to read raw profile data.

    All raw profiles must have the same length.
    """

    def __init__(self, path):
        self.path = path.expanduser()

    @abc.abstractmethod
    def count_profiles(self):
        """Number of raw profiles.

        Returns
        -------
        int
        """

    @abc.abstractmethod
    def profiles(self):
        """Yield raw profiles.

        Yields
        ------
        1-D ndarray
        """

    def all_profiles(self):
        """Return all profiles as 2-D array.

        Returns
        -------
        2-D ndarray
        """
        return np.array([p for p in self.profiles()])

    @abc.abstractmethod
    def names(self):
        """Yield profile names.

        Yields
        ------
        str
        """


class RawProfileCsvs(RawProfileBase):
    """Read raw profile data from a directory containing csv files.

    The raw profile data is stored in a directory with the following structure:

    .. code-block::

        rawdata
        ├── profile1.csv
        ├── profile2.csv
        └── ...

    Here, `rawdata` is the data path, which should be passed to the *dirpath* argument.
    Each CSV file contains height data for a single 1-dimensional profile, written in
    one column with multiple rows. Each row represents a spatial data point. The file
    must have no header.

    Parameters
    ----------
    path : pathlib.Path
        Path to directory containing raw data.

    Examples
    --------
    >>> from heavyedge import get_sample_path, RawProfileCsvs
    >>> profiles = RawProfileCsvs(get_sample_path("Type3")).profiles()
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... for Y in profiles:
    ...     plt.plot(Y)
    """

    def _files(self):
        return sorted(self.path.glob("*.csv"))

    def count_profiles(self):
        return len(list(self._files()))

    def profiles(self):
        for f in self._files():
            with open(f, newline="") as csvfile:
                reader = csv.reader(csvfile)
                prof = np.array([float(row[0]) for row in reader])
            yield prof

    def names(self):
        for f in self._files():
            yield str(f.stem)
