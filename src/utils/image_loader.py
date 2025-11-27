"""
Image Loader Module
Handles loading and reading satellite imagery from various sources
"""

import numpy as np
import rasterio
from typing import Dict, List, Optional
from pathlib import Path


class SatelliteImageLoader:
    """
    Loads and manages satellite imagery data
    Supports multiple satellite sensors: Landsat, ASTER, Sentinel-2
    """

    def __init__(self, image_path: str):
        """
        Initialize the image loader

        Args:
            image_path: Path to the satellite image file
        """
        self.image_path = Path(image_path)
        self.dataset = None
        self.metadata = {}
        self.bands = {}

    def load_image(self) -> np.ndarray:
        """
        Load the satellite image and return as numpy array

        Returns:
            Numpy array of shape (bands, height, width)
        """
        try:
            with rasterio.open(self.image_path) as src:
                self.dataset = src
                self.metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'bounds': src.bounds
                }

                # Read all bands
                image = src.read()
                return image

        except Exception as e:
            raise IOError(f"Error loading image from {self.image_path}: {str(e)}")

    def load_bands(self, band_indices: List[int]) -> np.ndarray:
        """
        Load specific bands from the image

        Args:
            band_indices: List of band indices to load (1-indexed)

        Returns:
            Numpy array of selected bands
        """
        try:
            with rasterio.open(self.image_path) as src:
                bands = []
                for idx in band_indices:
                    band = src.read(idx)
                    bands.append(band)
                return np.array(bands)

        except Exception as e:
            raise IOError(f"Error loading bands {band_indices}: {str(e)}")

    def get_band_names(self, sensor: str = 'landsat8') -> Dict[str, int]:
        """
        Get band names and their indices for different sensors

        Args:
            sensor: Sensor type ('landsat8', 'landsat7', 'aster', 'sentinel2')

        Returns:
            Dictionary mapping band names to indices
        """
        band_configs = {
            'landsat8': {
                'coastal': 1,
                'blue': 2,
                'green': 3,
                'red': 4,
                'nir': 5,
                'swir1': 6,
                'swir2': 7,
                'pan': 8,
                'cirrus': 9,
                'tirs1': 10,
                'tirs2': 11
            },
            'landsat7': {
                'blue': 1,
                'green': 2,
                'red': 3,
                'nir': 4,
                'swir1': 5,
                'tir': 6,
                'swir2': 7,
                'pan': 8
            },
            'aster': {
                'green': 1,
                'red': 2,
                'nir': 3,
                'swir1': 4,
                'swir2': 5,
                'swir3': 6,
                'swir4': 7,
                'swir5': 8,
                'swir6': 9,
                'tir1': 10,
                'tir2': 11,
                'tir3': 12,
                'tir4': 13,
                'tir5': 14
            },
            'sentinel2': {
                'coastal': 1,
                'blue': 2,
                'green': 3,
                'red': 4,
                'rededge1': 5,
                'rededge2': 6,
                'rededge3': 7,
                'nir': 8,
                'rededge4': 9,
                'swir1': 10,
                'swir2': 11
            }
        }

        return band_configs.get(sensor.lower(), band_configs['landsat8'])

    def get_metadata(self) -> Dict:
        """Get image metadata"""
        return self.metadata

    def save_result(self, data: np.ndarray, output_path: str,
                   reference_image: Optional[str] = None):
        """
        Save processed result as GeoTIFF

        Args:
            data: Numpy array to save
            output_path: Output file path
            reference_image: Path to reference image for georeferencing
        """
        ref_path = reference_image or self.image_path

        with rasterio.open(ref_path) as src:
            profile = src.profile.copy()

            # Update profile for single band output
            if len(data.shape) == 2:
                profile.update(count=1, dtype=data.dtype)
            else:
                profile.update(count=data.shape[0], dtype=data.dtype)

            with rasterio.open(output_path, 'w', **profile) as dst:
                if len(data.shape) == 2:
                    dst.write(data, 1)
                else:
                    for i in range(data.shape[0]):
                        dst.write(data[i], i + 1)
