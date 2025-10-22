"""Raw profile data files."""

import abc
import csv
import numbers
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "RawProfileBase",
    "RawProfileCsvs",
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


class RawProfileBase(abc.ABC):
    """Base class to read raw profile data.

    All raw profiles must have the same length.

    Notes
    -----
    ``self[key]`` returns a tuple of profile(s) and profile name(s).
    """

    def __init__(self, path):
        self.path = Path(path).expanduser()

    def __len__(self):
        return self.count_profiles()

    def __getitem__(self, key):
        """Profile and name indexing.

        By default, uses :meth:`all_profiles` and :meth:`profile_names`.
        Subclasses are encouraged to redefine this method for better performance.
        """
        return (self.all_profiles()[key], np.array(self.profile_names())[key])

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

    @_deprecated("1.5", "profiles() method")
    def all_profiles(self):
        """Return all profiles as an 2-D array.

        .. deprecated:: 1.5
            This method will be removed in HeavyEdge 2.0.
            Directly iterate over the generator from profiles() method

        Returns
        -------
        2-D ndarray
        """
        return np.array([p for p in self.profiles()])

    @abc.abstractmethod
    def profile_names(self):
        """Yield profile names.

        Yields
        ------
        str
        """


class RawProfileCsvs(RawProfileBase):
    """Read raw profile data from a directory containing CSV files.

    Directory structure:

    .. code-block::

        rawdata/
        ├── profile1.csv
        ├── profile2.csv
        └── ...

    Parameters
    ----------
    path : pathlike
        Path to the directory containing the raw CSV files.

    Notes
    -----
    - Each CSV file must contain a single column of numeric values (no header).
    - The order of profiles is determined by the sorted filenames.
    - The profile name is derived from the filename stem.

    Examples
    --------
    >>> from heavyedge import get_sample_path, RawProfileCsvs
    >>> profiles = RawProfileCsvs(get_sample_path("Type3")).profiles()
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... for Y in profiles:
    ...     plt.plot(Y)
    """

    def __init__(self, path):
        super().__init__(path)
        self._files = sorted(self.path.glob("*.csv"))

    @staticmethod
    def _read_profile(path):
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            profile = np.array([float(row[0]) for row in reader])
        return profile

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            file = self._files[key]
            return (self._read_profile(file), str(file.stem))
        elif isinstance(key, slice):
            files = self._files[key]
            profiles, names = [], []
            for file in files:
                profiles.append(self._read_profile(file))
                names.append(str(file.stem))
            return (np.array(profiles), np.array(names))
        elif isinstance(key, (Sequence, np.ndarray)):
            profiles, names = [], []
            for k in key:
                file = self._files[k]
                profiles.append(self._read_profile(file))
                names.append(str(file.stem))
            return (np.array(profiles), np.array(names))
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def count_profiles(self):
        return len(list(self._files))

    def profiles(self):
        for file in self._files:
            yield self._read_profile(file)

    def profile_names(self):
        for f in self._files:
            yield str(f.stem)
