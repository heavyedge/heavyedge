import os
import subprocess

import numpy as np

from heavyedge import ProfileData


def test_process_commands(tmp_rawdata_type2_path, tmp_path):
    processed_path = tmp_path / "ProcessedProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "prep",
            "--type",
            "csvs",
            "--res=1",
            "--sigma=1",
            "--std-thres=40",
            "--fill-value=0",
            "--z-thres=3.5",
            tmp_rawdata_type2_path,
            "-o",
            processed_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(processed_path)

    merged_path = tmp_path / "MergedProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "merge",
            processed_path,
            processed_path,
            "-o",
            merged_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(merged_path)

    with ProfileData(processed_path) as data:
        N = len(data)
    index_path = tmp_path / "index.npy"
    np.save(index_path, np.arange(N))
    filtered_path = tmp_path / "FilteredProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "filter",
            processed_path,
            index_path,
            "-o",
            filtered_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(filtered_path)


def test_mean_command(tmp_prepdata_type2_path, tmp_path):
    mean_path = tmp_path / "MeanProfile.h5"
    subprocess.run(
        [
            "heavyedge",
            "mean",
            "--wnum",
            "100",
            tmp_prepdata_type2_path,
            "-o",
            mean_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(mean_path)


def test_edge_command(tmp_prepdata_type2_path, tmp_path):
    area_scaled_path = tmp_path / "AreaScaledProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "scale",
            tmp_prepdata_type2_path,
            "--type=area",
            "-o",
            area_scaled_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(area_scaled_path)

    plateau_scaled_path = tmp_path / "PlateauScaledProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "scale",
            tmp_prepdata_type2_path,
            "--type=plateau",
            "-o",
            plateau_scaled_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(plateau_scaled_path)

    trimmed_path = tmp_path / "TrimmedProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "trim",
            tmp_prepdata_type2_path,
            "-o",
            trimmed_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(trimmed_path)

    padded_path = tmp_path / "PaddedProfiles.h5"
    subprocess.run(
        [
            "heavyedge",
            "pad",
            tmp_prepdata_type2_path,
            "-o",
            padded_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(padded_path)
