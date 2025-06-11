from heavyedge import ProfileData, RawProfileCsvs


def test_RawData_dunder(tmp_rawdata_type2_path):
    data = RawProfileCsvs(tmp_rawdata_type2_path)
    assert len(data) == data.count_profiles()


def test_ProfileData_dunder(tmp_prepdata_type2_path):
    with ProfileData(tmp_prepdata_type2_path) as data:
        assert len(data) == data.shape()[0]
