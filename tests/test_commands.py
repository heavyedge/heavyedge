import os
import subprocess


def test_process_commands(tmp_rawdata_type2_path, tmp_path):
    processed_path = tmp_path / "ProcessedData.h5"
    subprocess.run(
        [
            "heavyedge",
            "prep",
            "--type",
            "csvs",
            "--res=1",
            "--sigma=32",
            "--std-thres=0.1",
            tmp_rawdata_type2_path,
            "-o",
            processed_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(processed_path)

    filtered_path = tmp_path / "FilteredData.h5"
    subprocess.run(
        [
            "heavyedge",
            "outlier",
            "--z",
            "3.5",
            processed_path,
            "-o",
            filtered_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(filtered_path)

    mean_path = tmp_path / "MeanData.h5"
    subprocess.run(
        [
            "heavyedge",
            "mean",
            "--wnum",
            "100",
            processed_path,
            "-o",
            mean_path,
        ],
        capture_output=True,
        check=True,
    )
    assert os.path.exists(mean_path)
