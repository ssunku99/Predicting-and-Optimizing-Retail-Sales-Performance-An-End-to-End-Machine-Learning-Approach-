"""RetailMind — universal retail sales analytics pipeline.

Designed to ingest any retail sales dataset, map it to a canonical schema,
and run forecasting, anomaly detection, regression-based driver analysis,
and order recommendations end-to-end.
"""

__version__ = "0.1.0"

from retailmind.schema import CanonicalSchema, ColumnRole
from retailmind.ingest import load_file, load_dataset
from retailmind.mapper import SchemaMapper, MappingResult
from retailmind.canonical import canonicalize
from retailmind.pipeline import RetailPipeline
from retailmind.recommend import RecommendationParams
from retailmind.assistant import ask
from retailmind.tuning import tune_lgbm, DEFAULT_GRID, QUICK_GRID, explain_grid
from retailmind.report import save_report, render_report

__all__ = [
    "CanonicalSchema",
    "ColumnRole",
    "load_file",
    "load_dataset",
    "SchemaMapper",
    "MappingResult",
    "canonicalize",
    "RetailPipeline",
    "RecommendationParams",
    "ask",
    "tune_lgbm",
    "DEFAULT_GRID",
    "QUICK_GRID",
    "explain_grid",
    "save_report",
    "render_report",
]
