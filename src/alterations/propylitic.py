"""
Propylitic Alteration Detection
Detects chlorite, epidote, and carbonate minerals
"""

import numpy as np
from typing import Dict, Optional
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices
from analysis.pca_analysis import PCAAnalysis


class PropyliticDetector:
    """
    Detects propylitic alteration zones

    Propylitic alteration characterized by:
    - Chlorite, epidote
    - Calcite, other carbonates
    - Fe-Mg-OH absorption features
    - Outer zone of hydrothermal systems
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize propylitic alteration detector

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
        Detect propylitic alteration using band ratios

        Returns:
            Dictionary of propylitic alteration maps
        """
        # Combine ferrous and hydroxyl ratios
        ferrous = self.band_ratio.ferrous_mineral_ratio()
        hydroxyl = self.band_ratio.hydroxyl_ratio()

        results = {**ferrous, **hydroxyl}
        return results

    def detect_spectral_index_method(self) -> Dict[str, np.ndarray]:
        """
        Detect propylitic alteration using spectral indices

        Returns:
            Dictionary of propylitic alteration index maps
        """
        ferrous_indices = self.spectral_indices.ferrous_mineral_index()
        hydroxyl_indices = self.spectral_indices.hydroxyl_index()

        results = {**ferrous_indices, **hydroxyl_indices}
        return results

    def detect_pca_method(self) -> np.ndarray:
        """
        Detect propylitic alteration using PCA
        Uses bands sensitive to Fe-Mg minerals

        Returns:
            Propylitic alteration probability map
        """
        if self.sensor == 'landsat8':
            # Visible to SWIR for ferromagnesian minerals
            prop_bands = [2, 3, 4, 5, 6]  # Green, Red, NIR, SWIR1, SWIR2
        elif self.sensor == 'aster':
            prop_bands = [0, 1, 2, 3, 4]  # VNIR to SWIR
        else:
            prop_bands = [2, 3, 4, 5]

        pca = PCAAnalysis(self.image)
        components = pca.selective_pca(prop_bands, n_components=3)

        # Propylitic minerals in PC2 or PC3
        return components[1]

    def detect_chlorite(self) -> np.ndarray:
        """
        Specifically detect chlorite
        Fe-Mg-OH absorption features

        Returns:
            Chlorite probability map
        """
        if self.sensor == 'landsat8':
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Chlorite index - combination of ferrous and OH features
            chlorite_map = ((swir1 / nir) * (nir / red)) + (swir2 / swir1)

        elif self.sensor == 'aster':
            band3 = self.image[2]  # NIR
            band4 = self.image[3]  # SWIR
            band5 = self.image[4]  # SWIR
            band6 = self.image[5]  # SWIR

            # ASTER chlorite index
            chlorite_map = (band5 / band3) * (band6 / band5)

        else:
            chlorite_map = self.detect_composite_method()

        # Normalize
        chlorite_map = (chlorite_map - np.nanmin(chlorite_map)) / \
                       (np.nanmax(chlorite_map) - np.nanmin(chlorite_map))

        return chlorite_map

    def detect_epidote(self) -> np.ndarray:
        """
        Specifically detect epidote
        Fe3+ and OH absorption

        Returns:
            Epidote probability map
        """
        if self.sensor == 'landsat8':
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]

            # Epidote shows iron absorption in visible and OH in SWIR
            epidote_map = (red / green) * (swir1 / nir)

        elif self.sensor == 'aster':
            band2 = self.image[1]  # Red
            band3 = self.image[2]  # NIR
            band4 = self.image[3]  # SWIR

            epidote_map = (band2 / band3) * (band4 / band3)

        else:
            epidote_map = self.detect_composite_method()

        # Normalize
        epidote_map = (epidote_map - np.nanmin(epidote_map)) / \
                      (np.nanmax(epidote_map) - np.nanmin(epidote_map))

        return epidote_map

    def detect_carbonate(self) -> np.ndarray:
        """
        Detect carbonate minerals in propylitic zones

        Returns:
            Carbonate probability map
        """
        carbonate_results = self.band_ratio.carbonate_ratio()
        carbonate_indices = self.spectral_indices.carbonate_index()

        # Combine all carbonate indicators
        all_results = {**carbonate_results, **carbonate_indices}

        if all_results:
            normalized_maps = []
            for result in all_results.values():
                normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
                normalized_maps.append(normalized)

            carbonate_map = np.mean(normalized_maps, axis=0)
        else:
            carbonate_map = np.zeros_like(self.image[0])

        return carbonate_map

    def detect_actinolite(self) -> np.ndarray:
        """
        Detect actinolite (amphibole in propylitic zones)

        Returns:
            Actinolite probability map
        """
        if self.sensor == 'landsat8':
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Actinolite has Fe-Mg-OH features
            actinolite_map = (swir1 / nir) * (swir1 / swir2)

        elif self.sensor == 'aster':
            band3 = self.image[2]
            band5 = self.image[4]
            band6 = self.image[5]

            actinolite_map = (band5 / band3) * (band5 / band6)

        else:
            actinolite_map = self.detect_composite_method()

        # Normalize
        actinolite_map = (actinolite_map - np.nanmin(actinolite_map)) / \
                         (np.nanmax(actinolite_map) - np.nanmin(actinolite_map))

        return actinolite_map

    def detect_composite_method(self) -> np.ndarray:
        """
        Composite propylitic detection combining multiple methods

        Returns:
            Composite propylitic alteration probability map
        """
        # Get chlorite and epidote as main indicators
        chlorite = self.detect_chlorite()
        epidote = self.detect_epidote()

        # Get other indicators
        ratio_results = self.detect_band_ratio_method()
        index_results = self.detect_spectral_index_method()

        # Normalize all results
        normalized_maps = [chlorite, epidote]

        for key, result in {**ratio_results, **index_results}.items():
            normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
            normalized_maps.append(normalized)

        # Combine using weighted average (chlorite and epidote have higher weight)
        if len(normalized_maps) > 2:
            composite = (2 * chlorite + 2 * epidote + np.sum(normalized_maps[2:], axis=0)) / \
                       (4 + len(normalized_maps) - 2)
        else:
            composite = (chlorite + epidote) / 2

        return composite

    def detect_inner_propylitic(self) -> np.ndarray:
        """
        Detect inner propylitic zone (closer to ore body)
        Higher epidote, actinolite

        Returns:
            Inner propylitic probability map
        """
        epidote = self.detect_epidote()
        actinolite = self.detect_actinolite()

        inner_prop = (epidote + actinolite) / 2

        return inner_prop

    def detect_outer_propylitic(self) -> np.ndarray:
        """
        Detect outer propylitic zone (distal)
        More chlorite, carbonate

        Returns:
            Outer propylitic probability map
        """
        chlorite = self.detect_chlorite()
        carbonate = self.detect_carbonate()

        outer_prop = (chlorite + carbonate) / 2

        return outer_prop

    def detect_with_threshold(self, threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Detect propylitic alteration with thresholding

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
        Detect all propylitic alteration types

        Returns:
            Dictionary of all propylitic detection results
        """
        return {
            'general': self.detect_composite_method(),
            'chlorite': self.detect_chlorite(),
            'epidote': self.detect_epidote(),
            'carbonate': self.detect_carbonate(),
            'actinolite': self.detect_actinolite(),
            'inner_propylitic': self.detect_inner_propylitic(),
            'outer_propylitic': self.detect_outer_propylitic()
        }

    def detect_propylitic_halo(self, phyllic_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect propylitic halo around phyllic/potassic core

        Args:
            phyllic_map: Optional phyllic alteration map

        Returns:
            Propylitic halo probability map
        """
        propylitic = self.detect_composite_method()

        if phyllic_map is not None:
            # Propylitic halo is where propylitic is strong but phyllic is weak
            halo = propylitic * (1 - phyllic_map)

            # Normalize
            halo = (halo - np.nanmin(halo)) / (np.nanmax(halo) - np.nanmin(halo))

            return halo
        else:
            return propylitic
