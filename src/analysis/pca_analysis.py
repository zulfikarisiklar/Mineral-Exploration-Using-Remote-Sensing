"""
PCA Analysis Module
Principal Component Analysis for enhancing mineral signatures
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List


class PCAAnalysis:
    """
    Principal Component Analysis for mineral exploration
    Reduces dimensionality and enhances subtle spectral variations
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize PCA analysis

        Args:
            image: Multi-band satellite image (bands, height, width)
        """
        self.image = image
        self.n_bands, self.height, self.width = image.shape
        self.pca_model = None
        self.components = None
        self.explained_variance = None
        self.eigenvalues = None
        self.eigenvectors = None

    def preprocess_for_pca(self) -> np.ndarray:
        """
        Reshape and prepare image data for PCA

        Returns:
            Reshaped data (n_pixels, n_bands)
        """
        # Reshape from (bands, height, width) to (n_pixels, n_bands)
        reshaped = self.image.reshape(self.n_bands, -1).T

        # Remove NaN values
        mask = ~np.isnan(reshaped).any(axis=1)
        clean_data = reshaped[mask]

        return clean_data, mask

    def fit_pca(self, n_components: Optional[int] = None,
               standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA model to image data

        Args:
            n_components: Number of components to compute (None = all)
            standardize: Whether to standardize data before PCA

        Returns:
            Tuple of (principal components, explained variance ratio)
        """
        # Prepare data
        data, mask = self.preprocess_for_pca()

        # Standardize if requested
        if standardize:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            data = (data - mean) / (std + 1e-10)

        # Fit PCA
        n_comp = n_components or self.n_bands
        self.pca_model = PCA(n_components=n_comp)
        transformed = self.pca_model.fit_transform(data)

        # Store results
        self.explained_variance = self.pca_model.explained_variance_ratio_
        self.eigenvalues = self.pca_model.explained_variance_
        self.eigenvectors = self.pca_model.components_

        # Reshape back to image dimensions
        components = []
        for i in range(n_comp):
            component = np.zeros(self.height * self.width)
            component[mask] = transformed[:, i]
            component = component.reshape(self.height, self.width)
            components.append(component)

        self.components = np.array(components)

        return self.components, self.explained_variance

    def selective_pca(self, band_indices: List[int],
                     n_components: Optional[int] = None) -> np.ndarray:
        """
        Perform PCA on selected bands only

        Args:
            band_indices: List of band indices to include
            n_components: Number of components to compute

        Returns:
            Principal components from selected bands
        """
        # Extract selected bands
        selected_image = self.image[band_indices]

        # Create temporary PCA analyzer
        temp_analyzer = PCAAnalysis(selected_image)
        components, _ = temp_analyzer.fit_pca(n_components=n_components)

        return components

    def crosta_technique(self, target_bands: List[int]) -> np.ndarray:
        """
        Crosta Technique (Feature-Oriented PCA) for specific minerals
        Isolates specific spectral features

        Args:
            target_bands: Bands containing target mineral signature

        Returns:
            PC highlighting target mineral
        """
        # Perform PCA on target bands
        components = self.selective_pca(target_bands)

        # Usually the last PC contains the target feature
        # Return the component with highest contrast
        contrasts = [np.std(comp) for comp in components]
        best_pc_idx = np.argmax(contrasts)

        return components[best_pc_idx]

    def decorrelation_stretch(self) -> np.ndarray:
        """
        Apply decorrelation stretch to enhance color differences

        Returns:
            Decorrelated image
        """
        data, mask = self.preprocess_for_pca()

        # Standardize
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        standardized = (data - mean) / (std + 1e-10)

        # PCA transform
        if self.pca_model is None:
            self.fit_pca()

        transformed = self.pca_model.transform(standardized)

        # Stretch each component
        for i in range(transformed.shape[1]):
            pc_min = np.min(transformed[:, i])
            pc_max = np.max(transformed[:, i])
            transformed[:, i] = (transformed[:, i] - pc_min) / (pc_max - pc_min + 1e-10)

        # Inverse transform
        decorrelated = self.pca_model.inverse_transform(transformed)

        # Reshape back to image
        result = np.zeros((self.n_bands, self.height * self.width))
        for i in range(self.n_bands):
            result[i, mask] = decorrelated[:, i]

        result = result.reshape(self.n_bands, self.height, self.width)

        return result

    def mineral_pca_mapping(self) -> dict:
        """
        Apply PCA for different mineral groups using optimal band combinations

        Returns:
            Dictionary of mineral-specific PCA results
        """
        results = {}

        # Iron oxide detection (using visible and NIR bands)
        # Typically bands 1-5 for Landsat
        if self.n_bands >= 5:
            iron_bands = [0, 1, 2, 3, 4]
            results['iron_oxide'] = self.selective_pca(iron_bands, n_components=3)

        # Clay minerals (using SWIR bands)
        # Typically bands 5-7 for Landsat
        if self.n_bands >= 7:
            clay_bands = [4, 5, 6]
            results['clay_minerals'] = self.selective_pca(clay_bands, n_components=3)

        # Ferrous minerals (NIR and SWIR)
        if self.n_bands >= 6:
            ferrous_bands = [3, 4, 5]
            results['ferrous_minerals'] = self.selective_pca(ferrous_bands, n_components=3)

        return results

    def get_pc_image(self, pc_indices: Tuple[int, int, int]) -> np.ndarray:
        """
        Create RGB composite from principal components

        Args:
            pc_indices: Tuple of PC indices for R, G, B

        Returns:
            RGB composite (height, width, 3)
        """
        if self.components is None:
            raise ValueError("Run fit_pca() first")

        rgb = np.zeros((self.height, self.width, 3))

        for i, pc_idx in enumerate(pc_indices):
            component = self.components[pc_idx]
            # Normalize to 0-1
            pc_min = np.nanmin(component)
            pc_max = np.nanmax(component)
            rgb[:, :, i] = (component - pc_min) / (pc_max - pc_min + 1e-10)

        return rgb

    def get_loadings(self) -> np.ndarray:
        """
        Get PCA loadings (eigenvectors)

        Returns:
            Loading matrix (n_components, n_bands)
        """
        if self.eigenvectors is None:
            raise ValueError("Run fit_pca() first")

        return self.eigenvectors

    def get_scree_plot_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for scree plot

        Returns:
            Tuple of (component numbers, explained variance)
        """
        if self.explained_variance is None:
            raise ValueError("Run fit_pca() first")

        component_numbers = np.arange(1, len(self.explained_variance) + 1)
        return component_numbers, self.explained_variance

    def reconstruct_from_pcs(self, n_components: int) -> np.ndarray:
        """
        Reconstruct image using only top N principal components

        Args:
            n_components: Number of components to use

        Returns:
            Reconstructed image
        """
        if self.pca_model is None:
            raise ValueError("Run fit_pca() first")

        data, mask = self.preprocess_for_pca()

        # Transform and select components
        transformed = self.pca_model.transform(data)
        transformed[:, n_components:] = 0

        # Inverse transform
        reconstructed = self.pca_model.inverse_transform(transformed)

        # Reshape back to image
        result = np.zeros((self.n_bands, self.height * self.width))
        for i in range(self.n_bands):
            result[i, mask] = reconstructed[:, i]

        result = result.reshape(self.n_bands, self.height, self.width)

        return result
