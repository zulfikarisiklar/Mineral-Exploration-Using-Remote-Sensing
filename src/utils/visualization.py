"""
Visualization Module
Provides tools for visualizing analysis results and creating maps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List


class ResultVisualizer:
    """
    Visualization utilities for mineral exploration results
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_band_ratio(self, ratio: np.ndarray, title: str = "Band Ratio",
                       cmap: str = 'RdYlBu_r', save_path: Optional[str] = None):
        """
        Plot band ratio result

        Args:
            ratio: Band ratio array
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(ratio, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Ratio Value', rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_alteration_zones(self, alteration_map: np.ndarray,
                             alteration_type: str,
                             threshold: Optional[float] = None,
                             save_path: Optional[str] = None):
        """
        Plot detected alteration zones

        Args:
            alteration_map: Alteration detection result
            alteration_type: Type of alteration
            threshold: Threshold value for binary display
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Continuous map
        im1 = axes[0].imshow(alteration_map, cmap='hot')
        axes[0].set_title(f'{alteration_type} - Continuous',
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Probability', rotation=270, labelpad=20)

        # Binary/thresholded map
        if threshold is not None:
            binary_map = (alteration_map > threshold).astype(int)
        else:
            threshold = np.nanpercentile(alteration_map, 75)
            binary_map = (alteration_map > threshold).astype(int)

        im2 = axes[1].imshow(binary_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_title(f'{alteration_type} - Threshold: {threshold:.3f}',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04,
                            ticks=[0, 1])
        cbar2.set_ticklabels(['No Alteration', 'Alteration'])

        plt.suptitle(f'{alteration_type} Alteration Detection',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_multiple_alterations(self, alteration_maps: dict,
                                  save_path: Optional[str] = None):
        """
        Plot multiple alteration types in a grid

        Args:
            alteration_maps: Dictionary of alteration type -> map
            save_path: Path to save figure
        """
        n_maps = len(alteration_maps)
        ncols = 3
        nrows = (n_maps + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten() if n_maps > 1 else [axes]

        for idx, (alt_type, alt_map) in enumerate(alteration_maps.items()):
            im = axes[idx].imshow(alt_map, cmap='hot')
            axes[idx].set_title(f'{alt_type}', fontsize=11, fontweight='bold')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_maps, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Alteration Zone Detection Results',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_rgb_composite(self, image: np.ndarray,
                          rgb_bands: Tuple[int, int, int] = (0, 1, 2),
                          title: str = "RGB Composite",
                          save_path: Optional[str] = None):
        """
        Plot RGB composite image

        Args:
            image: Multi-band image array (bands, height, width)
            rgb_bands: Tuple of band indices for R, G, B
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Extract RGB bands
        r, g, b = rgb_bands
        rgb = np.dstack([image[r], image[g], image[b]])

        # Normalize to 0-1 range
        rgb_normalized = np.zeros_like(rgb)
        for i in range(3):
            band = rgb[:, :, i]
            p2, p98 = np.nanpercentile(band, (2, 98))
            rgb_normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)

        ax.imshow(rgb_normalized)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pca_components(self, components: List[np.ndarray],
                           n_components: int = 3,
                           save_path: Optional[str] = None):
        """
        Plot PCA components

        Args:
            components: List of PCA component arrays
            n_components: Number of components to plot
            save_path: Path to save figure
        """
        n_plot = min(n_components, len(components))
        fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 5))

        if n_plot == 1:
            axes = [axes]

        for idx in range(n_plot):
            im = axes[idx].imshow(components[idx], cmap='viridis')
            axes[idx].set_title(f'PC {idx + 1}', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.suptitle('Principal Component Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_composite_map(self, alteration_maps: dict,
                           colors: Optional[dict] = None,
                           save_path: Optional[str] = None):
        """
        Create a composite map showing all alteration zones

        Args:
            alteration_maps: Dictionary of alteration type -> map
            colors: Dictionary of alteration type -> RGB color
            save_path: Path to save figure
        """
        if colors is None:
            colors = {
                'Iron Oxide': (1.0, 0.0, 0.0),      # Red
                'Argillic': (0.0, 1.0, 0.0),         # Green
                'Phyllic': (0.0, 0.0, 1.0),          # Blue
                'Propylitic': (1.0, 1.0, 0.0),       # Yellow
                'Hydroxyl': (1.0, 0.0, 1.0),         # Magenta
                'Carbonate': (0.0, 1.0, 1.0)         # Cyan
            }

        # Get image shape from first map
        first_map = list(alteration_maps.values())[0]
        composite = np.zeros((*first_map.shape, 3))

        # Combine all alteration maps
        for alt_type, alt_map in alteration_maps.items():
            color = colors.get(alt_type, (1.0, 1.0, 1.0))

            # Normalize map to 0-1
            normalized = (alt_map - np.nanmin(alt_map)) / (np.nanmax(alt_map) - np.nanmin(alt_map))

            # Add colored layer
            for i in range(3):
                composite[:, :, i] += normalized * color[i]

        # Normalize composite
        composite = np.clip(composite, 0, 1)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(composite)
        ax.set_title('Composite Alteration Map', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Create legend
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=colors.get(alt_type, (1, 1, 1)),
                                   label=alt_type)
                         for alt_type in alteration_maps.keys()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
