"""
Silica (SiO2) Mineralization Detection
Detects quartz veins, silicification, and silica flooding
"""

import numpy as np
from typing import Dict, Optional
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices
from analysis.pca_analysis import PCAAnalysis


class SilicaMineralizationDetector:
    """
    Detects silica (SiO2) mineralization

    Silica mineralization types:
    - Quartz veins (hydrothermal)
    - Silicification zones
    - Silica caps/sinters
    - Opaline silica
    - Chalcedonic silica

    Methods:
    - Thermal infrared (TIR) band ratios (ASTER)
    - SWIR reflectance characteristics
    - Texture analysis for vein detection
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize silica detector

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type
        """
        self.image = image
        self.sensor = sensor
        self.band_ratio = BandRatioAnalysis(image, sensor)
        self.spectral_indices = SpectralIndices(image, sensor)

    def detect_quartz_vnir_swir(self) -> np.ndarray:
        """
        Detect quartz using VNIR and SWIR bands
        Quartz has high reflectance in SWIR with minimal absorption features

        Returns:
            Quartz probability map
        """
        if self.sensor == 'landsat8':
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Quartz has high SWIR reflectance and flat spectrum
            # Ratio emphasizes high reflectance in both SWIR bands
            quartz_index = (swir1 + swir2) / 2

            # Normalize by NIR to reduce topographic effects
            quartz_map = quartz_index / (nir + 1e-10)

        elif self.sensor == 'aster':
            band3 = self.image[2]  # NIR
            band4 = self.image[3]  # SWIR 1.656
            band5 = self.image[4]  # SWIR 2.165
            band6 = self.image[5]  # SWIR 2.205

            # ASTER quartz index using multiple SWIR bands
            quartz_map = (band4 + band5 + band6) / (band3 * 3 + 1e-10)

        else:
            # Generic approach
            if len(self.image) >= 6:
                quartz_map = (self.image[4] + self.image[5]) / (self.image[3] + 1e-10)
            else:
                quartz_map = np.ones_like(self.image[0])

        # Normalize
        quartz_map = (quartz_map - np.nanmin(quartz_map)) / \
                     (np.nanmax(quartz_map) - np.nanmin(quartz_map))

        return quartz_map

    def detect_silica_tir(self) -> np.ndarray:
        """
        Detect silica using thermal infrared (TIR) bands
        Only works with ASTER TIR or Landsat thermal bands

        Quartz has characteristic emissivity minima at ~8.6 and 9.2 μm

        Returns:
            Silica probability map from TIR
        """
        if self.sensor == 'aster' and self.image.shape[0] >= 14:
            # ASTER TIR bands (10-14)
            band10 = self.image[9]   # 8.125-8.475 μm
            band11 = self.image[10]  # 8.475-8.825 μm
            band12 = self.image[11]  # 8.925-9.275 μm
            band13 = self.image[12]  # 10.25-10.95 μm
            band14 = self.image[13]  # 10.95-11.65 μm

            # Quartz index using TIR bands
            # Band 13 / Band 12 emphasizes quartz emissivity minimum
            quartz_tir = band13 / (band12 + 1e-10)

            # Additional ratio
            silica_index = (band11 * band13) / (band12 * band12 + 1e-10)

            # Combine
            silica_map = (quartz_tir + silica_index) / 2

        elif self.sensor == 'landsat8' and self.image.shape[0] >= 11:
            # Landsat 8 TIR bands
            tir1 = self.image[9]   # Band 10
            tir2 = self.image[10]  # Band 11

            # Simple TIR ratio
            silica_map = tir1 / (tir2 + 1e-10)

        else:
            # No TIR bands available, return zeros
            return np.zeros_like(self.image[0])

        # Normalize
        silica_map = (silica_map - np.nanmin(silica_map)) / \
                     (np.nanmax(silica_map) - np.nanmin(silica_map))

        return silica_map

    def detect_silicification(self) -> np.ndarray:
        """
        Detect silicification (replacement of rock by silica)
        Characterized by high SWIR reflectance and smooth spectrum

        Returns:
            Silicification probability map
        """
        if self.sensor == 'landsat8':
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # High overall reflectance
            total_reflectance = (green + red + nir + swir1 + swir2) / 5

            # Flat SWIR spectrum (minimal absorption)
            swir_flatness = 1.0 - np.abs(swir1 - swir2) / (swir1 + swir2 + 1e-10)

            # Silicification index
            silicification = total_reflectance * swir_flatness

        elif self.sensor == 'aster':
            band1 = self.image[0]
            band2 = self.image[1]
            band3 = self.image[2]
            band4 = self.image[3]
            band5 = self.image[4]
            band6 = self.image[5]

            # High reflectance across VNIR-SWIR
            total_reflectance = (band1 + band2 + band3 + band4 + band5 + band6) / 6

            # SWIR flatness
            swir_flatness = 1.0 - np.std([band4, band5, band6], axis=0) / \
                           (np.mean([band4, band5, band6], axis=0) + 1e-10)

            silicification = total_reflectance * swir_flatness

        else:
            silicification = self.detect_quartz_vnir_swir()

        # Normalize
        silicification = (silicification - np.nanmin(silicification)) / \
                        (np.nanmax(silicification) - np.nanmin(silicification))

        return silicification

    def detect_quartz_veins(self) -> np.ndarray:
        """
        Detect quartz veins using texture and spectral characteristics
        Veins appear as linear high-reflectance features

        Returns:
            Quartz vein probability map
        """
        # Get base quartz detection
        quartz = self.detect_quartz_vnir_swir()

        # Apply edge detection to enhance linear features
        from scipy import ndimage

        # Sobel edge detection
        sx = ndimage.sobel(quartz, axis=0)
        sy = ndimage.sobel(quartz, axis=1)
        edges = np.hypot(sx, sy)

        # Normalize edges
        edges_norm = (edges - np.nanmin(edges)) / (np.nanmax(edges) - np.nanmin(edges))

        # Combine quartz detection with edge enhancement
        # High quartz signal + high edge = likely vein
        vein_map = quartz * (1 + edges_norm)

        # Normalize
        vein_map = (vein_map - np.nanmin(vein_map)) / \
                   (np.nanmax(vein_map) - np.nanmin(vein_map))

        return vein_map

    def detect_silica_cap(self) -> np.ndarray:
        """
        Detect silica cap (residual silica over epithermal deposits)

        Returns:
            Silica cap probability map
        """
        # Silica caps have:
        # 1. Very high silica content
        # 2. Often associated with advanced argillic alteration
        # 3. Resistant to weathering (topographic high)

        quartz = self.detect_quartz_vnir_swir()
        silicification = self.detect_silicification()

        # Strong silica signal
        silica_cap = (quartz + silicification) / 2

        # Enhance with brightness
        if self.sensor == 'landsat8':
            brightness = (self.image[1] + self.image[2] + self.image[3]) / 3
        else:
            brightness = (self.image[0] + self.image[1] + self.image[2]) / 3

        brightness_norm = (brightness - np.nanmin(brightness)) / \
                         (np.nanmax(brightness) - np.nanmin(brightness))

        silica_cap = silica_cap * (1 + brightness_norm)

        # Normalize
        silica_cap = (silica_cap - np.nanmin(silica_cap)) / \
                     (np.nanmax(silica_cap) - np.nanmin(silica_cap))

        return silica_cap

    def detect_opaline_silica(self) -> np.ndarray:
        """
        Detect opaline silica (hydrous amorphous silica)
        Common in hot spring/geothermal environments

        Returns:
            Opaline silica probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Opaline silica has water absorption at 1.9 μm
            # Shows in SWIR1 band
            opal_index = swir2 / (swir1 + 1e-10)

        elif self.sensor == 'aster':
            band4 = self.image[3]
            band6 = self.image[5]

            opal_index = band6 / (band4 + 1e-10)

        else:
            opal_index = self.detect_quartz_vnir_swir()

        # Normalize
        opal_index = (opal_index - np.nanmin(opal_index)) / \
                     (np.nanmax(opal_index) - np.nanmin(opal_index))

        return opal_index

    def detect_composite_method(self) -> np.ndarray:
        """
        Composite silica detection combining multiple methods

        Returns:
            Composite silica probability map
        """
        methods = []

        # VNIR-SWIR quartz detection
        quartz_vnir = self.detect_quartz_vnir_swir()
        methods.append(quartz_vnir)

        # Silicification
        silicification = self.detect_silicification()
        methods.append(silicification)

        # TIR (if available)
        silica_tir = self.detect_silica_tir()
        if np.nanmax(silica_tir) > 0:
            methods.append(silica_tir * 1.5)  # Weight TIR higher if available

        # Combine all methods
        if methods:
            composite = np.mean(methods, axis=0)
        else:
            composite = np.zeros_like(self.image[0])

        return composite

    def detect_with_threshold(self, threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Detect silica with thresholding

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
        Detect all silica mineralization types

        Returns:
            Dictionary of all silica detection results
        """
        return {
            'general': self.detect_composite_method(),
            'quartz_vnir_swir': self.detect_quartz_vnir_swir(),
            'silica_tir': self.detect_silica_tir(),
            'silicification': self.detect_silicification(),
            'quartz_veins': self.detect_quartz_veins(),
            'silica_cap': self.detect_silica_cap(),
            'opaline_silica': self.detect_opaline_silica()
        }

    def detect_epithermal_silica_system(self) -> Dict[str, np.ndarray]:
        """
        Detect silica features characteristic of epithermal gold systems

        Returns:
            Dictionary with epithermal silica indicators
        """
        results = {}

        # Silica cap (high-level)
        results['silica_cap'] = self.detect_silica_cap()

        # Quartz veins (ore-bearing structures)
        results['quartz_veins'] = self.detect_quartz_veins()

        # Opaline silica (hot spring environment)
        results['opaline_silica'] = self.detect_opaline_silica()

        # Composite epithermal indicator
        results['epithermal_composite'] = (
            results['silica_cap'] * 1.5 +
            results['quartz_veins'] * 1.2 +
            results['opaline_silica']
        ) / 3.7

        return results

    def mask_bright_pixels(self, silica_map: np.ndarray,
                          brightness_threshold: float = 0.8) -> np.ndarray:
        """
        Mask out non-silica bright features (carbonates, salts, snow)

        Args:
            silica_map: Silica probability map
            brightness_threshold: Threshold for masking

        Returns:
            Masked silica map
        """
        # Calculate brightness
        if self.sensor == 'landsat8':
            brightness = np.mean(self.image[1:5], axis=0)  # VNIR bands
        else:
            brightness = np.mean(self.image[0:3], axis=0)

        brightness_norm = (brightness - np.nanmin(brightness)) / \
                         (np.nanmax(brightness) - np.nanmin(brightness))

        # Get carbonate signal
        carb_ratios = self.band_ratio.carbonate_ratio()
        if carb_ratios:
            carbonate = list(carb_ratios.values())[0]
            carbonate_norm = (carbonate - np.nanmin(carbonate)) / \
                            (np.nanmax(carbonate) - np.nanmin(carbonate))
        else:
            carbonate_norm = np.zeros_like(brightness_norm)

        # Mask pixels that are bright AND have carbonate signature
        mask = ~((brightness_norm > brightness_threshold) & (carbonate_norm > 0.6))

        masked_silica = silica_map.copy()
        masked_silica[~mask] = 0

        return masked_silica
