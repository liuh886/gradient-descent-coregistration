"""Optional footprint geometry generation for sensor sampling (e.g. ICESat-2 footprints)."""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np


def create_footprint_polygons(
    x: Sequence[float],
    y: Sequence[float],
    width: float = 11.0,
    length: float = 11.0,
    azimuth_deg: Union[float, Sequence[float]] = 0.0,
) -> List[Any]:
    """Generate Shapely Polygon footprint representations for sensor sampling points.

    For circular footprints (width == length), buffers points with radius width / 2.
    For elongated footprints (width < length), generates a rotated capsule/stadium buffer
    aligned along azimuth_deg.

    Parameters
    ----------
    x, y : sequence of float
        Center coordinates of points.
    width : float, default 11.0
        Across-track footprint diameter in meters.
    length : float, default 11.0
        Along-track footprint length in meters.
    azimuth_deg : float or sequence of float, default 0.0
        Azimuth angle of ground track in degrees clockwise from north.

    Returns
    -------
    list of shapely.geometry.Polygon
    """
    from shapely.geometry import LineString, Point

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(x_arr)

    if isinstance(azimuth_deg, (int, float)):
        az_arr = np.full(n, float(azimuth_deg))
    else:
        az_arr = np.asarray(azimuth_deg, dtype=np.float64)

    polys = []
    r_width = float(width) / 2.0
    half_len = float(length) / 2.0

    is_circular = np.isclose(width, length)

    for i in range(n):
        pt_x, pt_y = x_arr[i], y_arr[i]
        if is_circular:
            p = Point(pt_x, pt_y).buffer(r_width, resolution=8)
            polys.append(p)
        else:
            rad = np.radians(az_arr[i])
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            d_len = max(0.0, half_len - r_width)
            p_n = Point(pt_x + d_len * sin_a, pt_y + d_len * cos_a)
            p_s = Point(pt_x - d_len * sin_a, pt_y - d_len * cos_a)
            capsule = LineString([p_s, p_n]).buffer(r_width, resolution=4)
            polys.append(capsule)

    return polys
