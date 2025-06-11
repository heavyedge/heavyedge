import csv

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from heavyedge import ProfileData, RawProfileCsvs
from heavyedge.api import preprocess

np.random.seed(0)


def profile_type1(b0, b1, b2, data_size=None, random_scale=0):
    """Generate artificial Type 1 profile.

    Parameters
    ----------
    b0 : scalar
        Plateau region height.
    b1, b2 : scalar
        Edge region y = -b1 * (x - b2)^2 + b0.
    data_size : int, optional
        If passed, profile is padded with "bare substrate region" to this size.
    random_scale : scalar, default=0
        Scale for standard normal noise.

    Raises
    ------
    ValueError
        If profile lenth is larger than *data_size*.
    """
    x = np.arange(np.ceil(np.sqrt(b0 / b1) + b2).astype(int))
    x_bp = np.ceil(b2).astype(int)

    y1 = np.full(x.shape, b0)
    y1[x_bp:] = 0
    y2 = -b1 * ((x - b2) ** 2) + b0
    y2[:x_bp] = 0
    ret = (y1 + y2).astype(float)

    if data_size is not None:
        if len(ret) > data_size:
            raise ValueError("data_size is too small.")
        ret = np.pad(ret, (0, data_size - len(ret)))
    ret += np.random.standard_normal(ret.shape) * random_scale
    return ret


def profile_type2(b0, b1, b2, b3, data_size=None, random_scale=0):
    """Generate artificial Type 2 profile.

    Parameters
    ----------
    b0 : scalar
        Plateau region height.
    b1, b2, b3 : scalar
        Heavy edge region y = -b1 * (x - b2)^2 + b3.
    data_size : int, optional
        If passed, profile is padded with "bare substrate region" to this size.
    random_scale : scalar, default=0
        Scale for standard normal noise.

    Raises
    ------
    ValueError
        If profile lenth is larger than *data_size*.
    """
    x = np.arange(np.ceil(np.sqrt(b3 / b1) + b2).astype(int))
    x_bp = np.ceil(-np.sqrt((b3 - b0) / b1) + b2).astype(int)

    y1 = np.full(x.shape, b0)
    y1[x_bp:] = 0
    y2 = -b1 * ((x - b2) ** 2) + b3
    y2[:x_bp] = 0
    ret = (y1 + y2).astype(float)

    if data_size is not None:
        if len(ret) > data_size:
            raise ValueError("data_size is too small.")
        ret = np.pad(ret, (0, data_size - len(ret)))
    ret += np.random.standard_normal(ret.shape) * random_scale
    return ret


def profile_type3(b0, b1, b2, b3, b4, data_size=None, random_scale=0):
    """Generate artificial Type 3 profile.

    Parameters
    ----------
    b0 : scalar
        Plateau region height.
    b1, b2, b3, b4 : scalar
        Heavy edge region y = -b1 * (x - b2) * (x - b2 - b3) * (x - b2 + b3) + b4
    data_size : int, optional
        If passed, profile is padded with "bare substrate region" to this size.
    random_scale : scalar, default=0
        Scale for standard normal noise.

    Raises
    ------
    ValueError
        If profile lenth is larger than *data_size*.
    """
    poly = Polynomial(
        [
            b1 * b2**3 - b1 * b2 * b3**2 + b4,
            -3 * b1 * b2**2 + b1 * b3**2,
            3 * b1 * b2,
            -b1,
        ]
    )
    roots1 = poly.roots()
    x_c = np.ceil(roots1[np.isreal(roots1)][-1].real).astype(int)
    poly2 = poly - Polynomial([b0])
    roots2 = poly2.roots()
    x_bp = np.ceil(roots2[np.isreal(roots2)][0].real).astype(int)

    y1 = np.full(x_c, b0)
    y1[x_bp:] = 0
    y2 = np.full(x_c, 0)
    y2[x_bp:] = poly(np.arange(x_bp, x_c))
    ret = (y1 + y2).astype(float)

    if data_size is not None:
        if len(ret) > data_size:
            raise ValueError("data_size is too small.")
        ret = np.pad(ret, (0, data_size - len(ret)))
    ret += np.random.standard_normal(ret.shape) * random_scale
    return ret


class RawDataFactory:
    def __init__(self, path):
        self.path = path

    def mkrawdir(self, dirname):
        path = self.path / dirname
        path.mkdir(parents=True)
        return path

    def mkrawfile(self, rawdir, filename, data):
        path = rawdir / filename
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for y in data:
                writer.writerow([y])
        return path


@pytest.fixture(scope="session")
def tmp_rawdata_type2_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("RawData-")
    rawdata_factory = RawDataFactory(path)

    DATA_SIZE = 150
    RANDOM_SCALE = 5

    rawdir = rawdata_factory.mkrawdir("Type2-00")
    for i in range(5):
        rawdata_factory.mkrawfile(
            rawdir,
            f"{str(i).zfill(2)}.csv",
            profile_type2(
                700, 1, 50, 800, data_size=DATA_SIZE, random_scale=RANDOM_SCALE
            ),
        )
    return rawdir


@pytest.fixture(scope="session")
def tmp_rawdata_type3_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("RawData-")
    rawdata_factory = RawDataFactory(path)

    DATA_SIZE = 70
    RANDOM_SCALE = 5

    rawdir = rawdata_factory.mkrawdir("Type3-00")
    for i in range(5):
        rawdata_factory.mkrawfile(
            rawdir,
            f"{str(i).zfill(2)}.csv",
            profile_type3(
                700, 1, 50, 5, 700, data_size=DATA_SIZE, random_scale=RANDOM_SCALE
            ),
        )
    return rawdir


@pytest.fixture(scope="session")
def tmp_prepdata_type2_path(tmp_rawdata_type2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("PrepData-") / "Type2.h5"
    rawdata = RawProfileCsvs(tmp_rawdata_type2_path)
    M = len(next(rawdata.profiles()))

    with ProfileData(path, "w").create(M, 1, "") as out:
        for profile, name in zip(rawdata.profiles(), rawdata.profile_names()):
            Y, L = preprocess(profile, 32, 0.1)
            out.write_profiles(Y.reshape(1, -1), [L], [name])
    return path
