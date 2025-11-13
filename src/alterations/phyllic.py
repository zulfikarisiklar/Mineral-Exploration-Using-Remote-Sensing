"""
Phyllic Alteration Detection
Detects sericite, illite, and muscovite (white mica minerals)
"""

import numpy as np
from typing import Dict, Optional
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices
from analysis.pca_analysis import PCAAnalysis


class PhyllicDetector:
    """
    Detects phyllic (sericitic) alteration zones

    Phyllic alteration characterized by:
    - Sericite (fine-grained muscovite)
    - Illite, muscovite
    - Al-OH absorption features
    - Associated with porphyry deposits
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize phyllic alteration detector

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
        Detect phyllic alteration using band ratios

        Returns:
            Dictionary of phyllic alteration maps
        """
        results = self.band_ratio.clay_mineral_ratio()
        return results

    def detect_spectral_index_method(self) -> Dict[str, np.ndarray]:
        """
        Detect phyllic alteration using spectral indices

        Returns:
            Dictionary of phyllic alteration index maps
        """
        results = self.spectral_indices.clay_alteration_index()
        return results

    def detect_pca_method(self) -> np.ndarray:
        """
        Detect phyllic alteration using PCA
        Uses SWIR bands sensitive to mica minerals

        Returns:
            Phyllic alteration probability map
        """
        if self.sensor == 'landsat8':
            # SWIR bands for mica detection
            phyllic_bands = [4, 5, 6]  # NIR, SWIR1, SWIR2
        elif self.sensor == 'aster':
            phyllic_bands = [3, 4, 5, 6, 7]  # SWIR bands
        else:
            phyllic_bands = [3, 4, 5]

        pca = PCAAnalysis(self.image)
        components = pca.selective_pca(phyllic_bands, n_components=3)

        # Phyllic alteration often appears in PC2 or PC3
        return components[1]

    def detect_sericite(self) -> np.ndarray:
        """
        Specifically detect sericite (fine-grained white mica)
        Strong Al-OH absorption at ~2.2 μm

        Returns:
            Sericite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]  # Band 6 - 1.6 μm
            swir2 = self.image[6]  # Band 7 - 2.2 μm
            nir = self.image[4]

            # Sericite index - emphasizes SWIR1 absorption
            sericite_map = swir1 / (swir2 + 1e-10)

            # Enhanced with NIR
            sericite_map = sericite_map * (swir1 / (nir + 1e-10))

        elif self.sensor == 'aster':
            band4 = self.image[3]  # 1.656 μm
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm

            # ASTER sericite index
            sericite_map = (band6 + band7) / (band4 * 2 + 1e-10)

        else:
            sericite_map = self.detect_composite_method()

        # Normalize
        sericite_map = (sericite_map - np.nanmin(sericite_map)) / \
                       (np.nanmax(sericite_map) - np.nanmin(sericite_map))

        return sericite_map

    def detect_muscovite(self) -> np.ndarray:
        """
        Specifically detect muscovite (coarse-grained white mica)

        Returns:
            Muscovite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]
            nir = self.image[4]

            # Muscovite has slightly different absorption position than sericite
            muscovite_map = (swir1 / (swir2 + 1e-10)) * (nir / (swir1 + 1e-10))

        elif self.sensor == 'aster':
            band5 = self.image[4]
            band6 = self.image[5]
            band7 = self.image[6]

            # ASTER muscovite index
            muscovite_map = band6 / ((band5 + band7) / 2 + 1e-10)

        else:
            muscovite_map = self.detect_composite_method()

        # Normalize
        muscovite_map = (muscovite_map - np.nanmin(muscovite_map)) / \
                        (np.nanmax(muscovite_map) - np.nanmin(muscovite_map))

        return muscovite_map

    def detect_illite(self) -> np.ndarray:
        """
        Specifically detect illite

        Returns:
            Illite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Illite has weaker absorption than muscovite
            illite_map = swir2 / (swir1 + 1e-10)

        elif self.sensor == 'aster':
            band5 = self.image[4]
            band6 = self.image[5]

            illite_map = band6 / (band5 + 1e-10)

        else:
            illite_map = self.detect_composite_method()

        # Normalize
        illite_map = (illite_map - np.nanmin(illite_map)) / \
                     (np.nanmax(illite_map) - np.nanmin(illite_map))

        return illite_map

    def detect_composite_method(self) -> np.ndarray:
        """
        Composite phyllic detection combining multiple methods

        Returns:
            Composite phyllic alteration probability map
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

    def detect_with_pyrite(self) -> np.ndarray:
        """
        Detect phyllic alteration with associated pyrite
        (common in porphyry copper deposits)

        Returns:
            Phyllic + pyrite probability map
        """
        phyllic = self.detect_composite_method()

        # Pyrite can show subtle iron absorption
        if self.sensor == 'landsat8':
            red = self.image[3]
            blue = self.image[1]
            # Subtle iron signature
            pyrite_indicator = red / (blue + 1e-10)

            # Normalize
            pyrite_indicator = (pyrite_indicator - np.nanmin(pyrite_indicator)) / \
                             (np.nanmax(pyrite_indicator) - np.nanmin(pyrite_indicator))

            # Combine phyllic with subtle pyrite signal
            combined = phyllic * (1 + 0.3 * pyrite_indicator)
        else:
            combined = phyllic

        # Normalize
        combined = (combined - np.nanmin(combined)) / \
                  (np.nanmax(combined) - np.nanmin(combined))

        return combined

    def detect_potassic_phyllic_transition(self) -> np.ndarray:
        """
        Detect transition zone between potassic and phyllic alteration

        Returns:
            Transition zone probability map
        """
        phyllic = self.detect_composite_method()

        if self.sensor == 'landsat8':
            # Use NIR/SWIR ratios to detect biotite-sericite transition
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            transition = phyllic * (nir / (swir1 + 1e-10)) * (swir2 / (swir1 + 1e-10))

            # Normalize
            transition = (transition - np.nanmin(transition)) / \
                        (np.nanmax(transition) - np.nanmin(transition))
        else:
            transition = phyllic

        return transition

    def detect_with_threshold(self, threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Detect phyllic alteration with thresholding

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
        Detect all phyllic alteration types

        Returns:
            Dictionary of all phyllic detection results
        """
        return {
            'general': self.detect_composite_method(),
            'sericite': self.detect_sericite(),
            'muscovite': self.detect_muscovite(),
            'illite': self.detect_illite(),
            'with_pyrite': self.detect_with_pyrite(),
            'transition_zone': self.detect_potassic_phyllic_transition()
        }

    def distinguish_from_argillic(self, argillic_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Distinguish phyllic from argillic alteration
        Phyllic typically has coarser grain size and different mineral assemblage

        Args:
            argillic_map: Optional argillic alteration map for comparison

        Returns:
            Phyllic-specific map
        """
        phyllic = self.detect_composite_method()

        if argillic_map is not None:
            # Phyllic often has stronger SWIR1 response
            # Argillic has stronger SWIR2 response
            if self.sensor == 'landsat8':
                swir1 = self.image[5]
                swir2 = self.image[6]

                # Phyllic emphasis
                phyllic_emphasis = swir1 / (swir2 + 1e-10)
                phyllic_emphasis = (phyllic_emphasis - np.nanmin(phyllic_emphasis)) / \
                                 (np.nanmax(phyllic_emphasis) - np.nanmin(phyllic_emphasis))

                phyllic_specific = phyllic * phyllic_emphasis
            else:
                phyllic_specific = phyllic

            # Subtract argillic contribution
            phyllic_specific = np.maximum(phyllic_specific - 0.3 * argillic_map, 0)

            return phyllic_specific
        else:
            return phyllic
