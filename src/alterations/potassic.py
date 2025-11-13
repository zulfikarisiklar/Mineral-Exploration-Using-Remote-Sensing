"""
Potassic Alteration Detection
Detects biotite, K-feldspar, and potassic mineral assemblages
"""

import numpy as np
from typing import Dict, Optional
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices
from analysis.pca_analysis import PCAAnalysis


class PotassicDetector:
    """
    Detects potassic alteration zones

    Potassic alteration characterized by:
    - Biotite (dark mica, K-Mg-Fe silicate)
    - K-feldspar (orthoclase, microcline)
    - Magnetite (secondary)
    - Associated with porphyry copper core zones
    - High temperature alteration (>350°C)
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize potassic alteration detector

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type
        """
        self.image = image
        self.sensor = sensor
        self.band_ratio = BandRatioAnalysis(image, sensor)
        self.spectral_indices = SpectralIndices(image, sensor)

    def detect_biotite(self) -> np.ndarray:
        """
        Detect biotite (dark mica with Fe-Mg-OH)
        Biotite has Fe absorption in visible and OH in SWIR

        Returns:
            Biotite probability map
        """
        if self.sensor == 'landsat8':
            blue = self.image[1]
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Biotite has:
            # 1. Low reflectance in visible (dark)
            # 2. Fe absorption (low red/NIR ratio)
            # 3. OH absorption in SWIR

            # Dark feature with Fe absorption
            darkness = 1.0 - ((red + green + blue) / 3)
            fe_absorption = red / (nir + 1e-10)

            # OH absorption
            oh_absorption = swir1 / (swir2 + 1e-10)

            # Combine indicators
            biotite_map = darkness * fe_absorption * oh_absorption

        elif self.sensor == 'aster':
            band1 = self.image[0]  # Green
            band2 = self.image[1]  # Red
            band3 = self.image[2]  # NIR
            band4 = self.image[3]  # SWIR
            band6 = self.image[5]  # SWIR

            # Dark with Fe and OH features
            darkness = 1.0 - ((band1 + band2) / 2)
            fe_absorption = band2 / (band3 + 1e-10)
            oh_feature = band4 / (band6 + 1e-10)

            biotite_map = darkness * fe_absorption * oh_feature

        else:
            # Generic approach
            biotite_map = self.detect_composite_method()

        # Normalize
        biotite_map = (biotite_map - np.nanmin(biotite_map)) / \
                      (np.nanmax(biotite_map) - np.nanmin(biotite_map))

        return biotite_map

    def detect_k_feldspar(self) -> np.ndarray:
        """
        Detect K-feldspar (orthoclase, microcline)
        K-feldspar has minimal absorption features but distinct reflectance

        Returns:
            K-feldspar probability map
        """
        if self.sensor == 'landsat8':
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # K-feldspar characteristics:
            # 1. Moderate reflectance
            # 2. Red edge slope
            # 3. Flat SWIR (no strong absorption)

            # Red edge
            red_edge = nir / (red + 1e-10)

            # SWIR flatness
            swir_flat = 1.0 - np.abs(swir1 - swir2) / (swir1 + swir2 + 1e-10)

            # Moderate brightness
            brightness = (green + red + nir) / 3
            brightness_norm = (brightness - np.nanmin(brightness)) / \
                             (np.nanmax(brightness) - np.nanmin(brightness))

            # K-feldspar index
            kfeld_map = red_edge * swir_flat * brightness_norm

        elif self.sensor == 'aster':
            band2 = self.image[1]  # Red
            band3 = self.image[2]  # NIR
            band5 = self.image[4]  # SWIR
            band6 = self.image[5]  # SWIR

            red_edge = band3 / (band2 + 1e-10)
            swir_flat = 1.0 - np.abs(band5 - band6) / (band5 + band6 + 1e-10)

            kfeld_map = red_edge * swir_flat

        else:
            kfeld_map = self.detect_composite_method()

        # Normalize
        kfeld_map = (kfeld_map - np.nanmin(kfeld_map)) / \
                    (np.nanmax(kfeld_map) - np.nanmin(kfeld_map))

        return kfeld_map

    def detect_magnetite(self) -> np.ndarray:
        """
        Detect secondary magnetite (associated with potassic alteration)
        Magnetite appears dark with low reflectance

        Returns:
            Magnetite probability map
        """
        if self.sensor == 'landsat8':
            blue = self.image[1]
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]

            # Magnetite is very dark across all wavelengths
            total_reflectance = (blue + green + red + nir) / 4

            # Invert to get darkness
            magnetite_map = 1.0 - total_reflectance

            # Slight Fe absorption
            fe_feature = blue / (red + 1e-10)

            magnetite_map = magnetite_map * fe_feature

        elif self.sensor == 'aster':
            band1 = self.image[0]
            band2 = self.image[1]
            band3 = self.image[2]

            total_reflectance = (band1 + band2 + band3) / 3
            magnetite_map = 1.0 - total_reflectance

        else:
            magnetite_map = np.zeros_like(self.image[0])

        # Normalize
        if np.nanmax(magnetite_map) > 0:
            magnetite_map = (magnetite_map - np.nanmin(magnetite_map)) / \
                           (np.nanmax(magnetite_map) - np.nanmin(magnetite_map))

        return magnetite_map

    def detect_composite_method(self) -> np.ndarray:
        """
        Composite potassic alteration detection

        Returns:
            Composite potassic alteration probability map
        """
        # Get individual mineral detections
        biotite = self.detect_biotite()
        k_feldspar = self.detect_k_feldspar()

        # Potassic alteration is combination of biotite and K-feldspar
        # Weight biotite higher as it's more distinctive
        potassic = (biotite * 1.5 + k_feldspar) / 2.5

        return potassic

    def detect_early_potassic(self) -> np.ndarray:
        """
        Detect early/core potassic alteration
        High biotite + K-feldspar + magnetite

        Returns:
            Early potassic probability map
        """
        biotite = self.detect_biotite()
        k_feldspar = self.detect_k_feldspar()
        magnetite = self.detect_magnetite()

        # Strong signal from all three
        early_potassic = np.sqrt(biotite * k_feldspar) * (1 + magnetite * 0.5)

        # Normalize
        early_potassic = (early_potassic - np.nanmin(early_potassic)) / \
                        (np.nanmax(early_potassic) - np.nanmin(early_potassic))

        return early_potassic

    def detect_late_potassic(self) -> np.ndarray:
        """
        Detect late potassic alteration
        K-feldspar dominant, less biotite

        Returns:
            Late potassic probability map
        """
        biotite = self.detect_biotite()
        k_feldspar = self.detect_k_feldspar()

        # K-feldspar stronger than biotite
        late_potassic = k_feldspar * (1 + 0.3 * biotite)

        # Normalize
        late_potassic = (late_potassic - np.nanmin(late_potassic)) / \
                       (np.nanmax(late_potassic) - np.nanmin(late_potassic))

        return late_potassic

    def detect_potassic_phyllic_transition(self) -> np.ndarray:
        """
        Detect transition zone between potassic and phyllic alteration

        Returns:
            Transition zone probability map
        """
        # Get potassic signal
        potassic = self.detect_composite_method()

        # Get phyllic-like signal (sericite indicators)
        clay_ratios = self.band_ratio.clay_mineral_ratio()

        if clay_ratios:
            phyllic_signal = list(clay_ratios.values())[0]
            phyllic_signal = (phyllic_signal - np.nanmin(phyllic_signal)) / \
                            (np.nanmax(phyllic_signal) - np.nanmin(phyllic_signal))

            # Transition has moderate signals of both
            transition = np.minimum(potassic, phyllic_signal)
        else:
            transition = potassic * 0.5

        return transition

    def detect_with_threshold(self, threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Detect potassic alteration with thresholding

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
        Detect all potassic alteration types

        Returns:
            Dictionary of all potassic detection results
        """
        return {
            'general': self.detect_composite_method(),
            'biotite': self.detect_biotite(),
            'k_feldspar': self.detect_k_feldspar(),
            'magnetite': self.detect_magnetite(),
            'early_potassic': self.detect_early_potassic(),
            'late_potassic': self.detect_late_potassic(),
            'transition_zone': self.detect_potassic_phyllic_transition()
        }

    def detect_porphyry_core(self) -> np.ndarray:
        """
        Detect porphyry copper core zone using potassic alteration

        Returns:
            Porphyry core probability map
        """
        # Core zone has strong early potassic alteration
        early_potassic = self.detect_early_potassic()
        biotite = self.detect_biotite()

        # Core typically has highest intensity potassic
        core = early_potassic * (1 + biotite * 0.5)

        # Normalize
        core = (core - np.nanmin(core)) / (np.nanmax(core) - np.nanmin(core))

        return core

    def distinguish_from_mafic_rocks(self) -> np.ndarray:
        """
        Distinguish potassic alteration from primary mafic rocks
        Both can have biotite, but alteration has specific associations

        Returns:
            Potassic alteration (distinguished from mafic rocks)
        """
        potassic = self.detect_composite_method()
        biotite = self.detect_biotite()

        # Mafic rocks have very low reflectance uniformly
        # Potassic alteration has more variability

        if self.sensor == 'landsat8':
            reflectance = np.mean(self.image[1:5], axis=0)
        else:
            reflectance = np.mean(self.image[0:3], axis=0)

        reflectance_norm = (reflectance - np.nanmin(reflectance)) / \
                          (np.nanmax(reflectance) - np.nanmin(reflectance))

        # Alteration should have moderate reflectance, not extremely dark
        # Mask out extremely dark areas (likely primary mafic)
        alteration_mask = reflectance_norm > 0.2

        potassic_alteration = potassic.copy()
        potassic_alteration[~alteration_mask] *= 0.3  # Reduce signal in very dark areas

        return potassic_alteration
