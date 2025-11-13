"""
Argillic Alteration Detection
Detects clay minerals: kaolinite, dickite, halloysite
"""

import numpy as np
from typing import Dict, Optional
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices
from analysis.pca_analysis import PCAAnalysis


class ArgillicDetector:
    """
    Detects argillic alteration zones

    Argillic alteration characterized by:
    - Kaolinite, dickite, halloysite
    - Al-OH absorption features
    - Associated with acidic hydrothermal fluids
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize argillic alteration detector

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
        Detect argillic alteration using band ratios

        Returns:
            Dictionary of argillic alteration maps
        """
        results = self.band_ratio.clay_mineral_ratio()
        return results

    def detect_spectral_index_method(self) -> Dict[str, np.ndarray]:
        """
        Detect argillic alteration using spectral indices

        Returns:
            Dictionary of argillic alteration index maps
        """
        results = self.spectral_indices.clay_alteration_index()
        return results

    def detect_pca_method(self) -> np.ndarray:
        """
        Detect argillic alteration using PCA
        Uses SWIR bands sensitive to Al-OH absorption

        Returns:
            Argillic alteration probability map
        """
        if self.sensor == 'landsat8':
            # SWIR bands for clay detection
            clay_bands = [4, 5, 6]  # NIR, SWIR1, SWIR2
        elif self.sensor == 'aster':
            clay_bands = [3, 4, 5, 6, 7]  # SWIR bands
        else:
            clay_bands = [3, 4, 5]

        pca = PCAAnalysis(self.image)
        components = pca.selective_pca(clay_bands, n_components=3)

        # Clay minerals typically in PC2 or PC3
        return components[1]

    def detect_kaolinite(self) -> np.ndarray:
        """
        Specifically detect kaolinite
        Strong Al-OH absorption at 2.2 μm (SWIR1)

        Returns:
            Kaolinite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]  # Band 6 - 1.6 μm
            swir2 = self.image[6]  # Band 7 - 2.2 μm
            nir = self.image[4]

            # Kaolinite index
            kaolinite_map = (swir1 * swir1) / (swir2 * nir + 1e-10)

        elif self.sensor == 'aster':
            band5 = self.image[4]  # 2.165 μm
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm

            # ASTER kaolinite index
            kaolinite_map = band6 / ((band5 + band7) / 2 + 1e-10)

        else:
            kaolinite_map = self.detect_composite_method()

        # Normalize
        kaolinite_map = (kaolinite_map - np.nanmin(kaolinite_map)) / \
                        (np.nanmax(kaolinite_map) - np.nanmin(kaolinite_map))

        return kaolinite_map

    def detect_alunite(self) -> np.ndarray:
        """
        Detect alunite (advanced argillic)
        Strong Al-OH and SO4 absorptions

        Returns:
            Alunite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Alunite shows characteristic SWIR ratios
            alunite_map = swir1 / (swir2 + 1e-10)

        elif self.sensor == 'aster':
            band6 = self.image[5]
            band7 = self.image[6]
            band8 = self.image[7]

            # Alunite index for ASTER
            alunite_map = (band6 * band8) / (band7 * band7 + 1e-10)

        else:
            alunite_map = self.detect_composite_method()

        # Normalize
        alunite_map = (alunite_map - np.nanmin(alunite_map)) / \
                      (np.nanmax(alunite_map) - np.nanmin(alunite_map))

        return alunite_map

    def detect_composite_method(self) -> np.ndarray:
        """
        Composite argillic detection combining multiple methods

        Returns:
            Composite argillic alteration probability map
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

    def detect_advanced_argillic(self) -> np.ndarray:
        """
        Detect advanced argillic alteration
        (alunite, pyrophyllite, dickite)

        Returns:
            Advanced argillic probability map
        """
        # Combine kaolinite and alunite
        kaolinite = self.detect_kaolinite()
        alunite = self.detect_alunite()

        # Advanced argillic has both strong signals
        advanced_argillic = np.sqrt(kaolinite * alunite)

        return advanced_argillic

    def detect_intermediate_argillic(self) -> np.ndarray:
        """
        Detect intermediate argillic alteration
        (kaolinite, montmorillonite)

        Returns:
            Intermediate argillic probability map
        """
        # Use composite method with emphasis on kaolinite
        composite = self.detect_composite_method()
        kaolinite = self.detect_kaolinite()

        intermediate = (composite + kaolinite) / 2

        return intermediate

    def detect_with_threshold(self, threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Detect argillic alteration with thresholding

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
        Detect all argillic alteration types

        Returns:
            Dictionary of all argillic detection results
        """
        return {
            'general': self.detect_composite_method(),
            'kaolinite': self.detect_kaolinite(),
            'alunite': self.detect_alunite(),
            'advanced_argillic': self.detect_advanced_argillic(),
            'intermediate_argillic': self.detect_intermediate_argillic()
        }

    def distinguish_from_phyllic(self, phyllic_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Distinguish argillic from phyllic alteration

        Args:
            phyllic_map: Optional phyllic alteration map for comparison

        Returns:
            Argillic-specific map (with phyllic subtracted)
        """
        argillic = self.detect_composite_method()

        if phyllic_map is not None:
            # Argillic is where argillic signal is stronger than phyllic
            argillic_specific = np.maximum(argillic - phyllic_map, 0)
            return argillic_specific
        else:
            return argillic
