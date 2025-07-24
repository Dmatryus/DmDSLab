"""
Dataset management utilities for DmDSLab.

Includes:
- UCI Dataset Manager for accessing UCI ML Repository
- Integration with ML data containers
"""

from .ml_data_container import (
    ModelData,
    DataSplit,
    DataInfo,
    create_data_split,
    create_kfold_data,
)

__all__ = [
    "ModelData",
    "DataSplit",
    "DataInfo",
    "create_data_split",
    "create_kfold_data",
]
