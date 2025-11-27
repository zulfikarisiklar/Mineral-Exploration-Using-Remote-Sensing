"""
Sulfate Mineral Detection
Detects gypsum, jarosite, alunite, and other sulfate minerals
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices


class SulfateMineralDetector:
    """
    Detects sulfate minerals

    Sulfate minerals include:
    - Gypsum (CaSO₄·2H₂O)
    - Jarosite (KFe₃(SO₄)₂(OH)₆) - acid sulfate alteration
    - Alunite (KAl₃(SO₄)₂(OH)₆) - advanced argillic
    - Anhydrite (CaSO₄)
    - Barite (BaSO₄)

    These are important indicators of:
    - Acid sulfate alteration
    - Supergene processes
    - Epithermal systems
    - Oxidation zones
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize sulfate mineral detector

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type
        """
        self.image = image
        self.sensor = sensor
        self.band_ratio = BandRatioAnalysis(image, sensor)
        self.spectral_indices = SpectralIndices(image, sensor)

    def detect_gypsum(self) -> np.ndarray:
        """
        Detect gypsum (CaSO₄·2H₂O)
        Gypsum has water absorption at 1.4, 1.9, and 2.2 μm

        Returns:
            Gypsum probability map
        """
        if self.sensor == 'landsat8':
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]  # 1.6 μm
            swir2 = self.image[6]  # 2.2 μm

            # Gypsum has:
            # 1. High reflectance in visible-NIR
            # 2. Water absorption in SWIR
            # 3. Distinct absorption at 1.9 and 2.2 μm

            # High visible reflectance
            brightness = (green + red + nir) / 3

            # Water absorption in SWIR
            water_absorption = nir / (swir1 + 1e-10)

            # Strong 2.2 μm absorption
            swir_absorption = swir1 / (swir2 + 1e-10)

            # Gypsum index
            gypsum_map = brightness * water_absorption * swir_absorption

        elif self.sensor == 'aster':
            band1 = self.image[0]  # Green
            band3 = self.image[2]  # NIR
            band4 = self.image[3]  # SWIR 1.656
            band6 = self.image[5]  # SWIR 2.205

            brightness = (band1 + band3) / 2
            water_abs = band3 / (band4 + 1e-10)
            swir_abs = band4 / (band6 + 1e-10)

            gypsum_map = brightness * water_abs * swir_abs

        else:
            gypsum_map = self.detect_all_sulfates()

        # Normalize
        gypsum_map = (gypsum_map - np.nanmin(gypsum_map)) / \
                     (np.nanmax(gypsum_map) - np.nanmin(gypsum_map))

        return gypsum_map

    def detect_jarosite(self) -> np.ndarray:
        """
        Detect jarosite (KFe₃(SO₄)₂(OH)₆)
        Jarosite has yellow-brown color and Fe³⁺ absorption

        Returns:
            Jarosite probability map
        """
        if self.sensor == 'landsat8':
            blue = self.image[1]
            green = self.image[2]
            red = self.image[3]
            _nir = self.image[4]  # Reserved for future use
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Jarosite characteristics:
            # 1. Yellow color (high red/blue ratio)
            # 2. Fe³⁺ absorption in blue
            # 3. OH and SO₄ absorption in SWIR

            # Yellow color
            yellow_index = (red + green) / (blue + 1e-10)

            # Fe³⁺ absorption
            fe_absorption = red / (blue + 1e-10)

            # SWIR absorption
            sulfate_absorption = swir1 / (swir2 + 1e-10)

            # Jarosite index
            jarosite_map = yellow_index * fe_absorption * sulfate_absorption

        elif self.sensor == 'aster':
            band1 = self.image[0]  # Green
            band2 = self.image[1]  # Red
            _band3 = self.image[2]  # NIR (reserved for future use)
            band5 = self.image[4]  # SWIR
            band6 = self.image[5]  # SWIR

            yellow_index = band2 / (band1 + 1e-10)
            sulfate_abs = band5 / (band6 + 1e-10)

            jarosite_map = yellow_index * sulfate_abs

        else:
            jarosite_map = self.detect_all_sulfates()

        # Normalize
        jarosite_map = (jarosite_map - np.nanmin(jarosite_map)) / \
                       (np.nanmax(jarosite_map) - np.nanmin(jarosite_map))

        return jarosite_map

    def detect_alunite(self) -> np.ndarray:
        """
        Detect alunite (KAl₃(SO₄)₂(OH)₆)
        Alunite has Al-OH and SO₄ absorption features

        Returns:
            Alunite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Alunite has strong absorption at 2.17 μm (Al-OH)
            # and broader SO₄ absorption
            alunite_map = swir1 / (swir2 + 1e-10)

        elif self.sensor == 'aster':
            band5 = self.image[4]  # 2.165 μm
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm

            # Alunite has absorption centered at ~2.17 μm
            # Band 6 is close to this absorption
            alunite_map = (band5 + band7) / (band6 * 2 + 1e-10)

        else:
            alunite_map = self.detect_all_sulfates()

        # Normalize
        alunite_map = (alunite_map - np.nanmin(alunite_map)) / \
                      (np.nanmax(alunite_map) - np.nanmin(alunite_map))

        return alunite_map

    def detect_anhydrite(self) -> np.ndarray:
        """
        Detect anhydrite (CaSO₄)
        Similar to gypsum but without water

        Returns:
            Anhydrite probability map
        """
        if self.sensor == 'landsat8':
            red = self.image[3]
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Anhydrite has:
            # 1. High reflectance
            # 2. Sulfate absorption in SWIR
            # 3. No water absorption (distinguishes from gypsum)

            brightness = (red + nir + swir1) / 3
            sulfate_absorption = swir1 / (swir2 + 1e-10)

            anhydrite_map = brightness * sulfate_absorption

        elif self.sensor == 'aster':
            band3 = self.image[2]
            band5 = self.image[4]
            band6 = self.image[5]

            brightness = (band3 + band5) / 2
            sulfate_absorption = band5 / (band6 + 1e-10)

            anhydrite_map = brightness * sulfate_absorption

        else:
            anhydrite_map = self.detect_all_sulfates()

        # Normalize
        anhydrite_map = (anhydrite_map - np.nanmin(anhydrite_map)) / \
                        (np.nanmax(anhydrite_map) - np.nanmin(anhydrite_map))

        return anhydrite_map

    def detect_all_sulfates(self) -> np.ndarray:
        """
        General sulfate mineral detection

        Returns:
            General sulfate probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # General sulfate absorption in SWIR
            sulfate_map = swir1 / (swir2 + 1e-10)

        elif self.sensor == 'aster':
            band5 = self.image[4]
            band6 = self.image[5]
            band7 = self.image[6]

            # Use multiple SWIR bands
            sulfate_map = (band5 + band7) / (band6 * 2 + 1e-10)

        else:
            sulfate_map = np.ones_like(self.image[0])

        # Normalize
        sulfate_map = (sulfate_map - np.nanmin(sulfate_map)) / \
                      (np.nanmax(sulfate_map) - np.nanmin(sulfate_map))

        return sulfate_map

    def detect_acid_sulfate_alteration(self) -> np.ndarray:
        """
        Detect acid sulfate alteration (jarosite + alunite)
        Common in oxidized epithermal systems

        Returns:
            Acid sulfate alteration probability map
        """
        jarosite = self.detect_jarosite()
        alunite = self.detect_alunite()

        # Acid sulfate has both jarosite and alunite
        acid_sulfate = (jarosite + alunite) / 2

        return acid_sulfate

    def detect_supergene_sulfates(self) -> np.ndarray:
        """
        Detect supergene sulfate minerals (weathering/oxidation)

        Returns:
            Supergene sulfate probability map
        """
        jarosite = self.detect_jarosite()
        gypsum = self.detect_gypsum()

        # Get iron oxide signal (indicates oxidation)
        iron_ratios = self.band_ratio.iron_oxide_ratio()
        if iron_ratios:
            iron_oxide = list(iron_ratios.values())[0]
            iron_oxide_norm = (iron_oxide - np.nanmin(iron_oxide)) / \
                             (np.nanmax(iron_oxide) - np.nanmin(iron_oxide))
        else:
            iron_oxide_norm = np.zeros_like(jarosite)

        # Supergene zone has sulfates + iron oxides
        supergene = (jarosite + gypsum + iron_oxide_norm) / 3

        return supergene

    def detect_epithermal_sulfates(self) -> np.ndarray:
        """
        Detect sulfates in epithermal environment

        Returns:
            Epithermal sulfate probability map
        """
        alunite = self.detect_alunite()
        jarosite = self.detect_jarosite()

        # Epithermal systems have advanced argillic with alunite
        epithermal = (alunite * 1.5 + jarosite) / 2.5

        return epithermal

    def discriminate_gypsum_alunite(self) -> Dict[str, np.ndarray]:
        """
        Discriminate between gypsum and alunite
        Both have SWIR absorption but different characteristics

        Returns:
            Dictionary with gypsum-dominant and alunite-dominant maps
        """
        gypsum = self.detect_gypsum()
        alunite = self.detect_alunite()

        # Gypsum has stronger water absorption
        # Alunite has stronger Al-OH absorption

        # Areas where gypsum dominates
        gypsum_dominant = np.maximum(gypsum - 0.5 * alunite, 0)

        # Areas where alunite dominates
        alunite_dominant = np.maximum(alunite - 0.5 * gypsum, 0)

        # Normalize
        if np.nanmax(gypsum_dominant) > 0:
            gypsum_dominant = gypsum_dominant / np.nanmax(gypsum_dominant)

        if np.nanmax(alunite_dominant) > 0:
            alunite_dominant = alunite_dominant / np.nanmax(alunite_dominant)

        return {
            'gypsum_dominant': gypsum_dominant,
            'alunite_dominant': alunite_dominant,
            'mixed': np.minimum(gypsum, alunite)
        }

    def detect_all_types(self) -> Dict[str, np.ndarray]:
        """
        Detect all sulfate mineral types

        Returns:
            Dictionary of all sulfate detection results
        """
        return {
            'all_sulfates': self.detect_all_sulfates(),
            'gypsum': self.detect_gypsum(),
            'jarosite': self.detect_jarosite(),
            'alunite': self.detect_alunite(),
            'anhydrite': self.detect_anhydrite(),
            'acid_sulfate_alteration': self.detect_acid_sulfate_alteration(),
            'supergene_sulfates': self.detect_supergene_sulfates(),
            'epithermal_sulfates': self.detect_epithermal_sulfates()
        }

    def create_sulfate_composite_map(self) -> np.ndarray:
        """
        Create a composite sulfate map with RGB representation

        Returns:
            RGB composite (height, width, 3) where:
            R = Jarosite (acid sulfate)
            G = Alunite (advanced argillic)
            B = Gypsum (hydrous sulfate)
        """
        jarosite = self.detect_jarosite()
        alunite = self.detect_alunite()
        gypsum = self.detect_gypsum()

        # Create RGB composite
        height, width = jarosite.shape
        composite = np.zeros((height, width, 3))

        composite[:, :, 0] = jarosite  # Red channel
        composite[:, :, 1] = alunite   # Green channel
        composite[:, :, 2] = gypsum    # Blue channel

        return composite

    def mask_false_positives(self, sulfate_map: np.ndarray) -> np.ndarray:
        """
        Mask false positives (carbonates, snow, clouds)

        Args:
            sulfate_map: Sulfate probability map

        Returns:
            Masked sulfate map
        """
        # Get carbonate signal (can be confused with sulfates)
        carb_ratios = self.band_ratio.carbonate_ratio()
        if carb_ratios:
            carbonate = list(carb_ratios.values())[0]
            carbonate_norm = (carbonate - np.nanmin(carbonate)) / \
                            (np.nanmax(carbonate) - np.nanmin(carbonate))

            # Mask high carbonate areas
            mask = carbonate_norm < 0.7
        else:
            mask = np.ones_like(sulfate_map, dtype=bool)

        # Get NDVI to mask vegetation
        ndvi = self.spectral_indices.ndvi()
        veg_mask = ndvi < 0.3

        # Combine masks
        final_mask = mask & veg_mask

        masked_sulfate = sulfate_map.copy()
        masked_sulfate[~final_mask] = 0

        return masked_sulfate
