"""
Iron Oxide Alteration Detection
Detects hematite, goethite, and other ferric iron minerals
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices
from analysis.pca_analysis import PCAAnalysis


class IronOxideDetector:
    """
    Detects iron oxide alterations using multiple techniques

    Iron oxides are indicators of:
    - Gossans (oxidized ore deposits)
    - Hydrothermal alteration
    - Weathering processes
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize iron oxide detector

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type
        """
        self.image = image
        self.sensor = sensor
        self.band_ratio = BandRatioAnalysis(image, sensor)
        self.spectral_indices = SpectralIndices(image, sensor)

    def detect_band_ratio_method(self) -> Dict[str, np.ndarray]:
        """
        Detect iron oxide using band ratio method

        Returns:
            Dictionary of iron oxide maps
        """
        results = self.band_ratio.iron_oxide_ratio()
        return results

    def detect_spectral_index_method(self) -> Dict[str, np.ndarray]:
        """
        Detect iron oxide using spectral indices

        Returns:
            Dictionary of iron oxide index maps
        """
        results = self.spectral_indices.iron_oxide_index()
        return results

    def detect_pca_method(self) -> np.ndarray:
        """
        Detect iron oxide using PCA method
        Uses visible and NIR bands

        Returns:
            Iron oxide probability map from PCA
        """
        # Use bands most sensitive to iron oxide
        if self.sensor == 'landsat8':
            iron_bands = [1, 2, 3, 4]  # Blue, Green, Red, NIR
        elif self.sensor == 'aster':
            iron_bands = [0, 1, 2]  # Green, Red, NIR
        else:
            iron_bands = [0, 1, 2, 3]

        pca = PCAAnalysis(self.image)
        components = pca.selective_pca(iron_bands, n_components=3)

        # Iron oxide typically shows up in PC2 or PC3
        # Return the component with highest variance in red/blue ratio areas
        return components[1]  # Usually PC2

    def detect_composite_method(self) -> np.ndarray:
        """
        Composite iron oxide detection combining multiple methods

        Returns:
            Composite iron oxide probability map
        """
        # Get results from different methods
        ratio_results = self.detect_band_ratio_method()
        index_results = self.detect_spectral_index_method()

        # Normalize all results
        normalized_maps = []

        for key, result in {**ratio_results, **index_results}.items():
            normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
            normalized_maps.append(normalized)

        # Combine using weighted average
        if normalized_maps:
            composite = np.mean(normalized_maps, axis=0)
            return composite
        else:
            return np.zeros_like(self.image[0])

    def detect_hematite(self) -> np.ndarray:
        """
        Specifically detect hematite (Fe2O3)
        Hematite has characteristic absorption in blue

        Returns:
            Hematite probability map
        """
        if self.sensor == 'landsat8':
            red = self.image[3]
            blue = self.image[1]
            green = self.image[2]

            # Hematite index: strong red/blue ratio
            hematite_map = (red / (blue + 1e-10)) * (red / (green + 1e-10))

        elif self.sensor == 'aster':
            red = self.image[1]
            green = self.image[0]

            hematite_map = red / (green + 1e-10)

        else:
            # Generic approach
            hematite_map = self.detect_composite_method()

        # Normalize
        hematite_map = (hematite_map - np.nanmin(hematite_map)) / \
                       (np.nanmax(hematite_map) - np.nanmin(hematite_map))

        return hematite_map

    def detect_goethite(self) -> np.ndarray:
        """
        Specifically detect goethite (FeO(OH))
        Goethite has absorption in blue-green

        Returns:
            Goethite probability map
        """
        if self.sensor == 'landsat8':
            red = self.image[3]
            green = self.image[2]
            blue = self.image[1]

            # Goethite shows moderate red/green ratio
            goethite_map = (red / (green + 1e-10)) * (green / (blue + 1e-10))

        elif self.sensor == 'aster':
            red = self.image[1]
            green = self.image[0]

            goethite_map = red / (green + 1e-10)

        else:
            goethite_map = self.detect_composite_method()

        # Normalize
        goethite_map = (goethite_map - np.nanmin(goethite_map)) / \
                       (np.nanmax(goethite_map) - np.nanmin(goethite_map))

        return goethite_map

    def detect_ferric_iron(self) -> np.ndarray:
        """
        Detect general ferric iron (Fe3+) minerals

        Returns:
            Ferric iron probability map
        """
        # Use composite method
        return self.detect_composite_method()

    def detect_with_threshold(self, threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Detect iron oxide with thresholding

        Args:
            threshold: Threshold value (0-1) for detection

        Returns:
            Dictionary with continuous and binary maps
        """
        continuous_map = self.detect_composite_method()

        # Apply threshold
        binary_map = (continuous_map > threshold).astype(np.uint8)

        return {
            'continuous': continuous_map,
            'binary': binary_map,
            'threshold': threshold
        }

    def detect_all_types(self) -> Dict[str, np.ndarray]:
        """
        Detect all iron oxide types

        Returns:
            Dictionary of all iron oxide detection results
        """
        return {
            'general': self.detect_composite_method(),
            'hematite': self.detect_hematite(),
            'goethite': self.detect_goethite(),
            'ferric_iron': self.detect_ferric_iron(),
            'band_ratio': self.detect_band_ratio_method().get('composite', self.image[0] * 0),
            'spectral_index': list(self.detect_spectral_index_method().values())[0]
                             if self.detect_spectral_index_method() else self.image[0] * 0
        }

    def mask_vegetation(self, iron_oxide_map: np.ndarray,
                       ndvi_threshold: float = 0.3) -> np.ndarray:
        """
        Mask out vegetation from iron oxide detection

        Args:
            iron_oxide_map: Iron oxide probability map
            ndvi_threshold: NDVI threshold for vegetation

        Returns:
            Masked iron oxide map
        """
        ndvi = self.spectral_indices.ndvi()

        # Create vegetation mask
        veg_mask = ndvi < ndvi_threshold

        # Apply mask
        masked_map = iron_oxide_map.copy()
        masked_map[~veg_mask] = 0

        return masked_map
