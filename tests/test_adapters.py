import numpy as np
import pandas as pd
import pytest

from gdc.adapters.geopandas_pts import PointCloudAdapter
from gdc.adapters.rasterio_dem import RasterDEMAdapter
from gdc.core.sampling import AffineTransform


def test_point_cloud_adapter_dataframe():
    df = pd.DataFrame({
        "E": [100.0, 200.0, 300.0],
        "N": [500.0, 600.0, 700.0],
        "h_te_best_fit": [10.5, 20.3, 30.1],
    })

    adapter = PointCloudAdapter.from_source(df)
    assert adapter.point_count == 3
    assert adapter.z_column_name == "h_te_best_fit"
    assert np.allclose(adapter.x, [100.0, 200.0, 300.0])
    assert np.allclose(adapter.z, [10.5, 20.3, 30.1])


def test_point_cloud_adapter_geodataframe():
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"z": [12.0, 15.0]},
            geometry=[Point(10.0, 20.0), Point(15.0, 25.0)],
            crs="EPSG:32633",
        )

        adapter = PointCloudAdapter.from_source(gdf)
        assert adapter.point_count == 2
        assert adapter.z_column_name == "z"
        assert np.allclose(adapter.x, [10.0, 15.0])
        assert np.allclose(adapter.y, [20.0, 25.0])
    except ImportError:
        pytest.skip("geopandas not installed")


def test_raster_adapter_synthetic():
    grid = np.ones((50, 50))
    trans = AffineTransform(10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0)

    class MockDEM:
        data = grid[np.newaxis, :, :]
        transform = trans
        crs = "EPSG:32633"
        bounds = (1000.0, 4500.0, 1500.0, 5000.0)
        res = (10.0, 10.0)

    adapter = RasterDEMAdapter.from_source(MockDEM())
    assert adapter.grid.shape == (50, 50)
    assert adapter.resolution == (10.0, 10.0)
