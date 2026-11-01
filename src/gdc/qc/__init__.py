"""Quality control and diagnostic modules for DEM validation."""

from gdc.qc.checker import check_dem
from gdc.qc.report import QCReport

__all__ = [
    "QCReport",
    "check_dem",
]
