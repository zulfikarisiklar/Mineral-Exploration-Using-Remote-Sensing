"""
Hydroxyl (OH) Bearing Mineral Detection
Detects various OH-bearing minerals including micas, clays, and amphiboles
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from analysis.band_ratio import BandRatioAnalysis
from analysis.spectral_indices import SpectralIndices


class HydroxylMineralDetector:
    """
    Detects hydroxyl-bearing minerals

    OH-bearing minerals include:
    - Clay minerals (kaolinite, montmorillonite, illite)
    - Micas (muscovite, biotite, phlogopite)
    - Amphiboles (hornblende, actinolite)
    - Chlorite group minerals
    """

    def __init__(self, image: np.ndarray, sensor: str = 'landsat8'):
        """
        Initialize hydroxyl mineral detector

        Args:
            image: Multi-band satellite image (bands, height, width)
            sensor: Sensor type
        """
        self.image = image
        self.sensor = sensor
        self.band_ratio = BandRatioAnalysis(image, sensor)
        self.spectral_indices = SpectralIndices(image, sensor)

    def detect_all_oh_minerals(self) -> np.ndarray:
        """
        Detect all OH-bearing minerals (general detection)

        Returns:
            OH mineral probability map
        """
        oh_ratios = self.band_ratio.hydroxyl_ratio()
        oh_indices = self.spectral_indices.hydroxyl_index()

        # Combine all OH indicators
        all_results = {**oh_ratios, **oh_indices}

        normalized_maps = []
        for result in all_results.values():
            normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
            normalized_maps.append(normalized)

        if normalized_maps:
            oh_map = np.mean(normalized_maps, axis=0)
        else:
            oh_map = np.zeros_like(self.image[0])

        return oh_map

    def detect_aloh_minerals(self) -> np.ndarray:
        """
        Detect Al-OH minerals (clays and white micas)
        Absorption at ~2.2 μm

        Returns:
            Al-OH mineral probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]  # 1.6 μm
            swir2 = self.image[6]  # 2.2 μm

            # Al-OH absorption at SWIR2
            aloh_map = swir1 / (swir2 + 1e-10)

        elif self.sensor == 'aster':
            band5 = self.image[4]  # 2.165 μm
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm

            # Enhanced Al-OH detection
            aloh_map = (band5 + band7) / (band6 + 1e-10)

        else:
            aloh_map = self.detect_all_oh_minerals()

        # Normalize
        aloh_map = (aloh_map - np.nanmin(aloh_map)) / \
                   (np.nanmax(aloh_map) - np.nanmin(aloh_map))

        return aloh_map

    def detect_mgoh_minerals(self) -> np.ndarray:
        """
        Detect Mg-OH minerals (chlorite, talc, serpentine)
        Absorption at ~2.3 μm

        Returns:
            Mg-OH mineral probability map
        """
        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]
            nir = self.image[4]

            # Mg-OH has different absorption position
            mgoh_map = (swir2 / swir1) * (swir1 / (nir + 1e-10))

        elif self.sensor == 'aster':
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm
            band8 = self.image[7]  # 2.330 μm

            # Mg-OH absorption at longer wavelength
            mgoh_map = (band7 + band8) / (band6 + 1e-10)

        else:
            mgoh_map = self.detect_all_oh_minerals()

        # Normalize
        mgoh_map = (mgoh_map - np.nanmin(mgoh_map)) / \
                   (np.nanmax(mgoh_map) - np.nanmin(mgoh_map))

        return mgoh_map

    def detect_feoh_minerals(self) -> np.ndarray:
        """
        Detect Fe-OH minerals (biotite, chlorite with iron)

        Returns:
            Fe-OH mineral probability map
        """
        ferrous_indices = self.spectral_indices.ferrous_mineral_index()
        oh_indices = self.spectral_indices.hydroxyl_index()

        # Combine ferrous and OH signals
        all_maps = []

        for result in list(ferrous_indices.values()) + list(oh_indices.values()):
            normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
            all_maps.append(normalized)

        if all_maps:
            feoh_map = np.mean(all_maps, axis=0)
        else:
            feoh_map = np.zeros_like(self.image[0])

        return feoh_map

    def detect_white_micas(self) -> np.ndarray:
        """
        Detect white micas (muscovite, sericite, illite)

        Returns:
            White mica probability map
        """
        aloh = self.detect_aloh_minerals()

        if self.sensor == 'landsat8':
            swir1 = self.image[5]
            swir2 = self.image[6]

            # White micas have strong sharp Al-OH absorption
            mica_emphasis = (swir1 / (swir2 + 1e-10)) ** 2

            # Normalize
            mica_emphasis = (mica_emphasis - np.nanmin(mica_emphasis)) / \
                           (np.nanmax(mica_emphasis) - np.nanmin(mica_emphasis))

            white_mica = aloh * mica_emphasis

        else:
            white_mica = aloh

        return white_mica

    def detect_dark_micas(self) -> np.ndarray:
        """
        Detect dark micas (biotite, phlogopite)

        Returns:
            Dark mica probability map
        """
        feoh = self.detect_feoh_minerals()
        mgoh = self.detect_mgoh_minerals()

        # Biotite has both Fe and Mg-OH
        dark_mica = (feoh + mgoh) / 2

        return dark_mica

    def detect_amphiboles(self) -> np.ndarray:
        """
        Detect amphibole minerals (hornblende, actinolite, tremolite)

        Returns:
            Amphibole probability map
        """
        mgoh = self.detect_mgoh_minerals()

        if self.sensor == 'landsat8':
            green = self.image[2]
            nir = self.image[4]
            swir1 = self.image[5]

            # Amphiboles show Fe absorption in visible and OH in SWIR
            amphibole_map = (nir / green) * (swir1 / (nir + 1e-10)) * mgoh

            # Normalize
            amphibole_map = (amphibole_map - np.nanmin(amphibole_map)) / \
                           (np.nanmax(amphibole_map) - np.nanmin(amphibole_map))

        else:
            amphibole_map = mgoh

        return amphibole_map

    def detect_clay_minerals(self) -> np.ndarray:
        """
        Detect clay minerals (kaolinite, montmorillonite, illite)

        Returns:
            Clay mineral probability map
        """
        clay_ratios = self.band_ratio.clay_mineral_ratio()
        clay_indices = self.spectral_indices.clay_alteration_index()

        # Combine all clay indicators
        all_results = {**clay_ratios, **clay_indices}

        normalized_maps = []
        for result in all_results.values():
            normalized = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))
            normalized_maps.append(normalized)

        if normalized_maps:
            clay_map = np.mean(normalized_maps, axis=0)
        else:
            clay_map = np.zeros_like(self.image[0])

        return clay_map

    def detect_serpentine(self) -> np.ndarray:
        """
        Detect serpentine minerals

        Returns:
            Serpentine probability map
        """
        mgoh = self.detect_mgoh_minerals()

        if self.sensor == 'landsat8':
            green = self.image[2]
            red = self.image[3]
            nir = self.image[4]

            # Serpentine has characteristic visible-NIR response
            serpentine_map = (nir / red) * (green / (red + 1e-10)) * mgoh

            # Normalize
            serpentine_map = (serpentine_map - np.nanmin(serpentine_map)) / \
                            (np.nanmax(serpentine_map) - np.nanmin(serpentine_map))

        else:
            serpentine_map = mgoh

        return serpentine_map

    def detect_chlorite(self) -> np.ndarray:
        """
        Detect chlorite minerals

        Returns:
            Chlorite probability map
        """
        mgoh = self.detect_mgoh_minerals()
        feoh = self.detect_feoh_minerals()

        # Chlorite has both Mg-OH and Fe-OH
        chlorite_map = np.sqrt(mgoh * feoh)

        return chlorite_map

    def discriminate_oh_minerals(self) -> Dict[str, np.ndarray]:
        """
        Discriminate between different OH-bearing mineral types
        using spectral characteristics

        Returns:
            Dictionary of discriminated mineral maps
        """
        results = {}

        # Get base OH detection (reserved for future enhancement)
        _all_oh = self.detect_all_oh_minerals()

        # Discriminate by absorption wavelength position
        if self.sensor == 'aster' and self.image.shape[0] >= 8:
            band5 = self.image[4]  # 2.165 μm
            band6 = self.image[5]  # 2.205 μm
            band7 = self.image[6]  # 2.260 μm
            band8 = self.image[7]  # 2.330 μm

            # Ratio 6/5 emphasizes 2.2 μm absorption (Al-OH)
            ratio_65 = band6 / (band5 + 1e-10)

            # Ratio 7/6 emphasizes 2.26 μm absorption (Fe-OH, Mg-OH)
            ratio_76 = band7 / (band6 + 1e-10)

            # Ratio 8/7 emphasizes 2.33 μm absorption (Mg-OH)
            ratio_87 = band8 / (band7 + 1e-10)

            # Normalize ratios
            for name, ratio in [('aloh_2.2um', ratio_65),
                               ('feoh_2.26um', ratio_76),
                               ('mgoh_2.33um', ratio_87)]:
                results[name] = (ratio - np.nanmin(ratio)) / \
                               (np.nanmax(ratio) - np.nanmin(ratio))

        elif self.sensor == 'landsat8':
            # Limited discrimination with Landsat
            results['aloh'] = self.detect_aloh_minerals()
            results['mgoh'] = self.detect_mgoh_minerals()

        return results

    def detect_all_types(self) -> Dict[str, np.ndarray]:
        """
        Detect all hydroxyl mineral types

        Returns:
            Dictionary of all OH mineral detection results
        """
        return {
            'all_oh': self.detect_all_oh_minerals(),
            'aloh_minerals': self.detect_aloh_minerals(),
            'mgoh_minerals': self.detect_mgoh_minerals(),
            'feoh_minerals': self.detect_feoh_minerals(),
            'white_micas': self.detect_white_micas(),
            'dark_micas': self.detect_dark_micas(),
            'amphiboles': self.detect_amphiboles(),
            'clay_minerals': self.detect_clay_minerals(),
            'serpentine': self.detect_serpentine(),
            'chlorite': self.detect_chlorite()
        }

    def create_oh_composite_map(self) -> np.ndarray:
        """
        Create a composite OH mineral map with RGB representation

        Returns:
            RGB composite (height, width, 3) where:
            R = Al-OH (white micas, clays)
            G = Mg-OH (chlorite, serpentine)
            B = Fe-OH (biotite, Fe-chlorite)
        """
        aloh = self.detect_aloh_minerals()
        mgoh = self.detect_mgoh_minerals()
        feoh = self.detect_feoh_minerals()

        # Create RGB composite
        height, width = aloh.shape
        composite = np.zeros((height, width, 3))

        composite[:, :, 0] = aloh   # Red channel
        composite[:, :, 1] = mgoh   # Green channel
        composite[:, :, 2] = feoh   # Blue channel

        return composite
