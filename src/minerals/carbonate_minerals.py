"""
Carbonate Mineral Detection
Detects calcite, dolomite, and other carbonate minerals
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices


class CarbonateMineralDetector:
    """
    Detects carbonate minerals

    Carbonate minerals include:
    - Calcite (CaCO3)
    - Dolomite (CaMg(CO3)2)
    - Siderite (FeCO3)
    - Magnesite (MgCO3)
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize carbonate mineral detector

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type
        """
        self.image = image
        self.sensor = sensor
        self.band_ratio = BandRatioAnalysis(image, sensor)
        self.spectral_indices = SpectralIndices(image, sensor)

    def detect_all_carbonates(self) -> np.ndarray:
        """
        Detect all carbonate minerals (general detection)
        Carbonates have absorption features at 2.3-2.35 μm

        Returns:
            Carbonate probability map
        """
        carb_ratios = self.band_ratio.carbonate_ratio()
        carb_indices = self.spectral_indices.carbonate_index()

        # Combine all carbonate indicators
        all_results = {**carb_ratios, **carb_indices}

        normalized_maps = []
        for result in all_results.values():
            normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
            normalized_maps.append(normalized)

        if normalized_maps:
            carb_map = np.mean(normalized_maps, axis=0)
        else:
            carb_map = np.zeros_like(self.image[0])

        return carb_map

    def detect_calcite(self) -> np.ndarray:
        """
        Specifically detect calcite (CaCO3)
        Strong CO3 absorption at 2.33-2.35 μm

        Returns:
            Calcite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]  # 1.6 μm
            swir2 = self.image[6]  # 2.2 μm

            # Calcite index
            calcite_map = swir2 / (swir1 + 1e-10)

        elif self.sensor == 'aster':
            _band6 = self.image[5]  # 2.205 μm (reserved for future use)
            band7 = self.image[6]  # 2.260 μm
            band8 = self.image[7]  # 2.330 μm
            band9 = self.image[8]  # 2.395 μm

            # ASTER calcite index - strong absorption at band 8
            calcite_map = (band7 + band9) / (band8 + 1e-10)

        else:
            calcite_map = self.detect_all_carbonates()

        # Normalize
        calcite_map = (calcite_map - np.nanmin(calcite_map)) / \
                      (np.nanmax(calcite_map) - np.nanmin(calcite_map))

        return calcite_map

    def detect_dolomite(self) -> np.ndarray:
        """
        Specifically detect dolomite (CaMg(CO3)2)
        Absorption features similar to calcite but shifted

        Returns:
            Dolomite probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]
            nir = self.image[4]

            # Dolomite has additional Mg features
            dolomite_map = (swir2 / swir1) * (swir1 / (nir + 1e-10))

        elif self.sensor == 'aster':
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm
            band8 = self.image[7]  # 2.330 μm

            # Dolomite has broader absorption
            dolomite_map = (band6 + band8) / (band7 + 1e-10)

        else:
            dolomite_map = self.detect_all_carbonates()

        # Normalize
        dolomite_map = (dolomite_map - np.nanmin(dolomite_map)) / \
                       (np.nanmax(dolomite_map) - np.nanmin(dolomite_map))

        return dolomite_map

    def detect_siderite(self) -> np.ndarray:
        """
        Detect siderite (FeCO3) - iron-bearing carbonate

        Returns:
            Siderite probability map
        """
        # Siderite has both iron and carbonate features
        carbonate = self.detect_all_carbonates()

        if self.sensor == 'landsat8':
            red = self.image[3]
            blue = self.image[1]

            # Iron component
            iron_component = red / (blue + 1e-10)
            iron_component = (iron_component - np.nanmin(iron_component)) / \
                           (np.nanmax(iron_component) - np.nanmin(iron_component))

            # Combine carbonate and iron signals
            siderite_map = np.sqrt(carbonate * iron_component)

        elif self.sensor == 'aster':
            band2 = self.image[1]  # Red
            band1 = self.image[0]  # Green

            iron_component = band2 / (band1 + 1e-10)
            iron_component = (iron_component - np.nanmin(iron_component)) / \
                           (np.nanmax(iron_component) - np.nanmin(iron_component))

            siderite_map = np.sqrt(carbonate * iron_component)

        else:
            siderite_map = carbonate

        return siderite_map

    def detect_magnesite(self) -> np.ndarray:
        """
        Detect magnesite (MgCO3) - magnesium carbonate

        Returns:
            Magnesite probability map
        """
        carbonate = self.detect_all_carbonates()

        if self.sensor == 'landsat8':
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Magnesite has Mg features
            mg_component = (swir2 / swir1) * (swir1 / (nir + 1e-10))
            mg_component = (mg_component - np.nanmin(mg_component)) / \
                          (np.nanmax(mg_component) - np.nanmin(mg_component))

            magnesite_map = np.sqrt(carbonate * mg_component)

        else:
            magnesite_map = carbonate

        return magnesite_map

    def detect_carbonate_in_alteration(self) -> np.ndarray:
        """
        Detect carbonate as alteration product
        Common in propylitic and argillic zones

        Returns:
            Carbonate alteration probability map
        """
        carbonate = self.detect_all_carbonates()

        # Enhance areas with strong carbonate signature
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Strong SWIR2/SWIR1 ratio indicates carbonate alteration
            alteration_emphasis = (swir2 / (swir1 + 1e-10)) ** 2
            alteration_emphasis = (alteration_emphasis - np.nanmin(alteration_emphasis)) / \
                                 (np.nanmax(alteration_emphasis) - np.nanmin(alteration_emphasis))

            carbonate_alt = carbonate * alteration_emphasis

        else:
            carbonate_alt = carbonate

        return carbonate_alt

    def detect_marble(self) -> np.ndarray:
        """
        Detect marble (metamorphosed carbonate rocks)
        Strong, pure carbonate signature

        Returns:
            Marble probability map
        """
        calcite = self.detect_calcite()
        dolomite = self.detect_dolomite()

        # Marble is high purity calcite or dolomite
        marble_map = np.maximum(calcite, dolomite)

        # Enhance high values (pure carbonates)
        marble_map = marble_map ** 1.5

        # Normalize
        marble_map = (marble_map - np.nanmin(marble_map)) / \
                     (np.nanmax(marble_map) - np.nanmin(marble_map))

        return marble_map

    def discriminate_calcite_dolomite(self) -> Dict[str, np.ndarray]:
        """
        Discriminate between calcite and dolomite

        Returns:
            Dictionary with calcite-dominated and dolomite-dominated maps
        """
        calcite = self.detect_calcite()
        dolomite = self.detect_dolomite()

        # Areas where calcite dominates
        calcite_dominant = np.maximum(calcite - 0.5 * dolomite, 0)

        # Areas where dolomite dominates
        dolomite_dominant = np.maximum(dolomite - 0.5 * calcite, 0)

        # Normalize
        if np.nanmax(calcite_dominant) > 0:
            calcite_dominant = calcite_dominant / np.nanmax(calcite_dominant)

        if np.nanmax(dolomite_dominant) > 0:
            dolomite_dominant = dolomite_dominant / np.nanmax(dolomite_dominant)

        return {
            'calcite_dominant': calcite_dominant,
            'dolomite_dominant': dolomite_dominant,
            'mixed': np.minimum(calcite, dolomite)
        }

    def detect_gossan_carbonates(self) -> np.ndarray:
        """
        Detect carbonate minerals associated with gossans
        (oxidized sulfide deposits)

        Returns:
            Gossan carbonate probability map
        """
        carbonate = self.detect_all_carbonates()

        # Get iron oxide signal (common in gossans)
        iron_ratios = self.band_ratio.iron_oxide_ratio()

        if iron_ratios:
            iron_signal = list(iron_ratios.values())[0]
            iron_signal = (iron_signal - np.nanmin(iron_signal)) / \
                         (np.nanmax(iron_signal) - np.nanmin(iron_signal))

            # Gossans have both carbonate and iron oxide
            gossan_carb = np.sqrt(carbonate * iron_signal)

        else:
            gossan_carb = carbonate

        return gossan_carb

    def detect_all_types(self) -> Dict[str, np.ndarray]:
        """
        Detect all carbonate mineral types

        Returns:
            Dictionary of all carbonate detection results
        """
        results = {
            'all_carbonates': self.detect_all_carbonates(),
            'calcite': self.detect_calcite(),
            'dolomite': self.detect_dolomite(),
            'siderite': self.detect_siderite(),
            'magnesite': self.detect_magnesite(),
            'carbonate_alteration': self.detect_carbonate_in_alteration(),
            'marble': self.detect_marble(),
            'gossan_carbonates': self.detect_gossan_carbonates()
        }

        # Add discrimination results
        discrimination = self.discriminate_calcite_dolomite()
        results.update(discrimination)

        return results

    def create_carbonate_composite_map(self) -> np.ndarray:
        """
        Create a composite carbonate map with RGB representation

        Returns:
            RGB composite (height, width, 3) where:
            R = Calcite
            G = Dolomite
            B = Iron carbonates (siderite)
        """
        calcite = self.detect_calcite()
        dolomite = self.detect_dolomite()
        siderite = self.detect_siderite()

        # Create RGB composite
        height, width = calcite.shape
        composite = np.zeros((height, width, 3))

        composite[:, :, 0] = calcite   # Red channel
        composite[:, :, 1] = dolomite  # Green channel
        composite[:, :, 2] = siderite  # Blue channel

        return composite

    def mask_vegetation_and_water(self, carbonate_map: np.ndarray) -> np.ndarray:
        """
        Mask out vegetation and water from carbonate detection

        Args:
            carbonate_map: Carbonate probability map

        Returns:
            Masked carbonate map
        """
        ndvi = self.spectral_indices.ndvi()
        ndwi = self.spectral_indices.ndwi()

        # Create masks
        veg_mask = ndvi < 0.3
        water_mask = ndwi < 0.0

        # Combine masks
        valid_mask = veg_mask & water_mask

        # Apply mask
        masked_map = carbonate_map.copy()
        masked_map[~valid_mask] = 0

        return masked_map
