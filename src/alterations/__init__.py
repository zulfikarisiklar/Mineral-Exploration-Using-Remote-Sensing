"""Alteration detection modules"""

from .iron_oxide import IronOxideDetector
from .argillic import ArgillicDetector
from .phyllic import PhyllicDetector
from .propylitic import PropyliticDetector
from .potassic import PotassicDetector

__all__ = ['IronOxideDetector', 'ArgillicDetector', 'PhyllicDetector',
           'PropyliticDetector', 'PotassicDetector']
