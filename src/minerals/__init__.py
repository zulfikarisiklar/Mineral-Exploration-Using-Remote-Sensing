"""Mineral-specific detection modules"""

from .hydroxyl_minerals import HydroxylMineralDetector
from .carbonate_minerals import CarbonateMineralDetector
from .silica_minerals import SilicaMineralizationDetector
from .sulfate_minerals import SulfateMineralDetector

__all__ = ['HydroxylMineralDetector', 'CarbonateMineralDetector',
           'SilicaMineralizationDetector', 'SulfateMineralDetector']
