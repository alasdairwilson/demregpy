from demregpy._example_utils import load_aia_flare_maps, load_aia_test_maps


def test_load_aia_test_maps_shapes():
    maps = load_aia_test_maps()

    assert len(maps) == 6
    assert [round(amap.wavelength.to_value()) for amap in maps] == [94, 131, 171, 193, 211, 335]


def test_load_aia_flare_maps_shapes():
    map_rows = load_aia_flare_maps()

    assert len(map_rows) == 10
    assert all(len(row) == 6 for row in map_rows)
