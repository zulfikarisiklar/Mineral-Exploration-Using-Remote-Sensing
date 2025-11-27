"""
Spectral Indices Module
Various spectral indices for mineral and alteration detection
"""

import numpy as np
from typing import Dict


class SpectralIndices:
    """
    Calculate spectral indices for mineral exploration
    Includes standard and specialized indices
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize spectral indices calculator

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type for band configuration
        """
        self.image = image
        self.sensor = sensor
        self.epsilon = 1e-10

    def ndvi(self) -> np.ndarray:
        """
        Normalized Difference Vegetation Index
        For vegetation masking

        Returns:
            NDVI map
        """
        if self.sensor == 'landsat8':
            nir = self.image[4]  # Band 5
            red = self.image[3]  # Band 4
        elif self.sensor == 'landsat7':
            nir = self.image[3]
            red = self.image[2]
        else:
            raise ValueError(f"NDVI not configured for {self.sensor}")

        ndvi = (nir - red) / (nir + red + self.epsilon)
        return ndvi

    def ndwi(self) -> np.ndarray:
        """
        Normalized Difference Water Index
        For water body masking

        Returns:
            NDWI map
        """
        if self.sensor == 'landsat8':
            green = self.image[2]  # Band 3
            nir = self.image[4]    # Band 5
        elif self.sensor == 'landsat7':
            green = self.image[1]
            nir = self.image[3]
        else:
            raise ValueError(f"NDWI not configured for {self.sensor}")

        ndwi = (green - nir) / (green + nir + self.epsilon)
        return ndwi

    def iron_oxide_index(self) -> Dict[str, np.ndarray]:
        """
        Various iron oxide indices

        Returns:
            Dictionary of iron oxide indices
        """
        results = {}

        if self.sensor == 'landsat8':
            red = self.image[3]
            blue = self.image[1]
            _green = self.image[2]  # Reserved for future use
            nir = self.image[4]
            _swir1 = self.image[5]  # Reserved for future use

            # Classic iron oxide ratio
            results['io_ratio'] = red / (blue + self.epsilon)

            # Ferric oxide index
            results['foi'] = (red - blue) / (red + blue + self.epsilon)

            # Iron oxide ratio for Landsat
            results['ior'] = (red / blue) * (red / nir + self.epsilon)

        elif self.sensor == 'aster':
            band1 = self.image[0]
            band2 = self.image[1]
            _band4 = self.image[3]  # Reserved for future use

            results['io_ratio'] = band2 / (band1 + self.epsilon)
            results['foi'] = (band2 - band1) / (band2 + band1 + self.epsilon)

        return results

    def clay_alteration_index(self) -> Dict[str, np.ndarray]:
        """
        Clay alteration indices (argillic/phyllic)

        Returns:
            Dictionary of clay indices
        """
        results = {}

        if self.sensor == 'landsat8':
            swir1 = self.image[5]  # Band 6
            swir2 = self.image[6]  # Band 7
            nir = self.image[4]    # Band 5

            # Alunite-Kaolinite-Pyrophyllite index
            results['akp_index'] = swir1 / swir2

            # Clay minerals ratio
            results['clay_ratio'] = (swir1 * swir1) / (swir2 * nir + self.epsilon)

            # Normalized difference clay index
            results['ndci'] = (swir1 - swir2) / (swir1 + swir2 + self.epsilon)

        elif self.sensor == 'aster':
            _band4 = self.image[3]  # Reserved for future use
            band5 = self.image[4]
            band6 = self.image[5]
            band7 = self.image[6]
            _band8 = self.image[7]  # Reserved for future use

            # ASTER clay index
            results['clay_index'] = (band5 + band7) / band6
            results['aloh_index'] = (band6 / band5) * (band6 / band7 + self.epsilon)

        return results

    def ferrous_mineral_index(self) -> Dict[str, np.ndarray]:
        """
        Ferrous mineral indices

        Returns:
            Dictionary of ferrous mineral indices
        """
        results = {}

        if self.sensor == 'landsat8':
            nir = self.image[4]
            swir1 = self.image[5]
            red = self.image[3]

            # Ferrous minerals
            results['ferrous_index'] = swir1 / nir

            # Enhanced ferrous detection
            results['efi'] = (swir1 / nir) * (nir / (red + self.epsilon))

        elif self.sensor == 'aster':
            band3 = self.image[2]
            _band4 = self.image[3]  # Reserved for future use
            band5 = self.image[4]

            results['ferrous_index'] = band5 / band3

        return results

    def carbonate_index(self) -> Dict[str, np.ndarray]:
        """
        Carbonate mineral indices

        Returns:
            Dictionary of carbonate indices
        """
        results = {}

        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Carbonate index
            results['carb_index'] = swir2 / swir1

            # Normalized carbonate index
            results['nci'] = (swir2 - swir1) / (swir2 + swir1 + self.epsilon)

        elif self.sensor == 'aster':
            _band6 = self.image[5]  # Reserved for future use
            band7 = self.image[6]
            band8 = self.image[7]
            band9 = self.image[8]

            # ASTER carbonate index
            results['carb_index'] = (band7 + band9) / band8

        return results

    def hydroxyl_index(self) -> Dict[str, np.ndarray]:
        """
        Hydroxyl-bearing mineral indices

        Returns:
            Dictionary of hydroxyl indices
        """
        results = {}

        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]
            nir = self.image[4]

            # Hydroxyl index
            results['oh_index'] = swir2 / swir1

            # Enhanced OH detection
            results['eoh'] = (swir2 + swir1) / nir

        elif self.sensor == 'aster':
            band5 = self.image[4]
            band6 = self.image[5]
            band7 = self.image[6]

            results['oh_index'] = (band5 + band7) / band6

        return results

    def silica_index(self) -> Dict[str, np.ndarray]:
        """
        Silica/Quartz indices (requires thermal bands)

        Returns:
            Dictionary of silica indices
        """
        results = {}

        if self.sensor == 'aster' and self.image.shape[0] >= 14:
            # ASTER quartz index using TIR bands
            band11 = self.image[10]
            band12 = self.image[11]
            band13 = self.image[12]

            results['quartz_index'] = (band11 * band11) / (band12 * band13 + self.epsilon)

            # Silica index
            results['silica_index'] = band13 / band12

        return results

    def gossan_index(self) -> Dict[str, np.ndarray]:
        """
        Gossan (oxidized ore) detection indices

        Returns:
            Dictionary of gossan indices
        """
        results = {}

        if self.sensor == 'landsat8':
            red = self.image[3]
            blue = self.image[1]
            _green = self.image[2]  # Reserved for future use
            nir = self.image[4]
            swir1 = self.image[5]

            # Gossan index combining iron oxide and clay
            results['gossan'] = (red / blue) * (swir1 / nir + self.epsilon)

            # Enhanced gossan detection
            results['enhanced_gossan'] = ((red / blue) + (swir1 / nir)) / 2

        return results

    def alteration_index(self) -> Dict[str, np.ndarray]:
        """
        Comprehensive alteration indices

        Returns:
            Dictionary of alteration indices
        """
        results = {}

        if self.sensor == 'landsat8':
            red = self.image[3]
            blue = self.image[1]
            green = self.image[2]
            nir = self.image[4]
            swir1 = self.image[5]
            swir2 = self.image[6]

            # Hydrothermal alteration index
            results['hai'] = ((red + blue) / 2) / green

            # Advanced argillic alteration
            results['aaa'] = (swir1 + swir2) / (nir + red + self.epsilon)

            # Propylitic alteration
            results['prop'] = (green + red) / (blue + nir + self.epsilon)

        return results

    def mineral_composite_index(self) -> Dict[str, np.ndarray]:
        """
        Composite indices combining multiple features

        Returns:
            Dictionary of composite indices
        """
        results = {}

        # Get component indices
        iron = self.iron_oxide_index()
        clay = self.clay_alteration_index()

        # Create composite
        if 'io_ratio' in iron and 'clay_ratio' in clay:
            results['mineral_composite'] = (iron['io_ratio'] + clay['clay_ratio']) / 2

        # Alteration intensity
        results['alteration_intensity'] = np.sqrt(
            iron.get('io_ratio', 0)**2 +
            clay.get('clay_ratio', 0)**2
        )

        return results

    def calculate_all_indices(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate all available spectral indices

        Returns:
            Dictionary of all indices organized by category
        """
        results = {
            'vegetation': {'ndvi': self.ndvi()},
            'water': {'ndwi': self.ndwi()},
            'iron_oxide': self.iron_oxide_index(),
            'clay': self.clay_alteration_index(),
            'ferrous': self.ferrous_mineral_index(),
            'carbonate': self.carbonate_index(),
            'hydroxyl': self.hydroxyl_index(),
            'gossan': self.gossan_index(),
            'alteration': self.alteration_index(),
            'composite': self.mineral_composite_index()
        }

        # Add silica if available
        silica = self.silica_index()
        if silica:
            results['silica'] = silica

        return results
