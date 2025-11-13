"""
Preprocessing Module
Handles image preprocessing operations like normalization, enhancement, and filtering
"""

import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Tuple


class ImagePreprocessor:
    """
    Preprocessing utilities for satellite imagery
    """

    @staticmethod
    def normalize(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image data

        Args:
            image: Input image array
            method: Normalization method ('minmax', 'standard', 'percentile')

        Returns:
            Normalized image array
        """
        original_shape = image.shape

        if len(image.shape) > 2:
            # Reshape to (n_pixels, n_bands)
            reshaped = image.reshape(-1, image.shape[0]) if image.shape[0] < image.shape[1] else image.reshape(image.shape[0], -1)
        else:
            reshaped = image.flatten().reshape(-1, 1)

        if method == 'minmax':
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(reshaped.T).T
        elif method == 'standard':
            scaler = StandardScaler()
            normalized = scaler.fit_transform(reshaped.T).T
        elif method == 'percentile':
            p2, p98 = np.percentile(reshaped, (2, 98))
            normalized = np.clip((reshaped - p2) / (p98 - p2), 0, 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.reshape(original_shape)

    @staticmethod
    def remove_nodata(image: np.ndarray, nodata_value: float = 0) -> np.ndarray:
        """
        Replace nodata values with NaN

        Args:
            image: Input image array
            nodata_value: Value representing nodata

        Returns:
            Image with nodata replaced by NaN
        """
        image = image.astype(float)
        image[image == nodata_value] = np.nan
        return image

    @staticmethod
    def mask_invalid(image: np.ndarray, min_val: Optional[float] = None,
                    max_val: Optional[float] = None) -> np.ndarray:
        """
        Mask invalid values outside specified range

        Args:
            image: Input image array
            min_val: Minimum valid value
            max_val: Maximum valid value

        Returns:
            Masked image array
        """
        mask = np.ones_like(image, dtype=bool)

        if min_val is not None:
            mask &= (image >= min_val)
        if max_val is not None:
            mask &= (image <= max_val)

        masked = image.copy()
        masked[~mask] = np.nan
        return masked

    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian smoothing filter

        Args:
            image: Input image array
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Filtered image
        """
        if len(image.shape) == 3:
            # Apply to each band
            filtered = np.zeros_like(image)
            for i in range(image.shape[0]):
                filtered[i] = ndimage.gaussian_filter(image[i], sigma=sigma)
            return filtered
        else:
            return ndimage.gaussian_filter(image, sigma=sigma)

    @staticmethod
    def apply_median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Apply median filter for noise reduction

        Args:
            image: Input image array
            size: Filter window size

        Returns:
            Filtered image
        """
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for i in range(image.shape[0]):
                filtered[i] = ndimage.median_filter(image[i], size=size)
            return filtered
        else:
            return ndimage.median_filter(image, size=size)

    @staticmethod
    def stretch_contrast(image: np.ndarray, percentile: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Apply contrast stretching

        Args:
            image: Input image array
            percentile: Tuple of (min_percentile, max_percentile)

        Returns:
            Contrast stretched image
        """
        p_min, p_max = np.nanpercentile(image, percentile)
        stretched = np.clip((image - p_min) / (p_max - p_min), 0, 1)
        return stretched

    @staticmethod
    def calculate_band_ratio(band1: np.ndarray, band2: np.ndarray,
                            epsilon: float = 1e-10) -> np.ndarray:
        """
        Calculate ratio between two bands

        Args:
            band1: Numerator band
            band2: Denominator band
            epsilon: Small value to prevent division by zero

        Returns:
            Band ratio
        """
        ratio = band1 / (band2 + epsilon)
        return ratio

    @staticmethod
    def calculate_normalized_difference(band1: np.ndarray, band2: np.ndarray,
                                       epsilon: float = 1e-10) -> np.ndarray:
        """
        Calculate normalized difference between two bands
        (band1 - band2) / (band1 + band2)

        Args:
            band1: First band
            band2: Second band
            epsilon: Small value to prevent division by zero

        Returns:
            Normalized difference
        """
        nd = (band1 - band2) / (band1 + band2 + epsilon)
        return nd
