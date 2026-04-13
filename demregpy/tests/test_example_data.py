from demregpy.tests.example_data import load_aia_flare_timeseries, load_aia_full_disk_maps


def test_load_aia_full_disk_maps():
    maps = load_aia_full_disk_maps()

    assert len(maps) == 6
    assert [round(amap.wavelength.to_value()) for amap in maps] == [94, 131, 171, 193, 211, 335]


def test_load_aia_flare_timeseries():
    map_rows = load_aia_flare_timeseries()

    assert len(map_rows) == 10
    assert all(len(row) == 6 for row in map_rows)
