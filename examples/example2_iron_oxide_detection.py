"""
Example 2: Iron Oxide Alteration Detection
Focused analysis on iron oxide minerals
"""

import sys
sys.path.append('../src')

import numpy as np
from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from utils.visualization import ResultVisualizer
from alterations.iron_oxide import IronOxideDetector


def main():
    """Run iron oxide focused analysis"""

    # Load image
    print("Loading satellite image...")
    loader = SatelliteImageLoader('path/to/your/landsat8_image.tif')
    image = loader.load_image()

    # Preprocess
    print("Preprocessing...")
    preprocessor = ImagePreprocessor()
    image = preprocessor.normalize(image, method='percentile')

    # Initialize iron oxide detector
    print("\nDetecting iron oxide alterations...")
    detector = IronOxideDetector(image, sensor='landsat8')

    # Detect different types
    print("\n1. General iron oxide detection")
    general = detector.detect_composite_method()

    print("2. Hematite detection")
    hematite = detector.detect_hematite()

    print("3. Goethite detection")
    goethite = detector.detect_goethite()

    print("4. Band ratio method")
    band_ratio_results = detector.detect_band_ratio_method()

    print("5. Spectral index method")
    spectral_index_results = detector.detect_spectral_index_method()

    # Visualize results
    print("\nVisualizing results...")
    visualizer = ResultVisualizer()

    # Plot general iron oxide
    visualizer.plot_alteration_zones(
        general,
        'Iron Oxide (General)',
        threshold=0.7,
        save_path='../results/iron_oxide_general.png'
    )

    # Plot specific types
    iron_oxide_types = {
        'Hematite': hematite,
        'Goethite': goethite,
        'General Iron Oxide': general
    }

    visualizer.plot_multiple_alterations(
        iron_oxide_types,
        save_path='../results/iron_oxide_types.png'
    )

    # Mask vegetation
    print("\nMasking vegetation...")
    masked = detector.mask_vegetation(general, ndvi_threshold=0.3)

    visualizer.plot_band_ratio(
        masked,
        title='Iron Oxide (Vegetation Masked)',
        save_path='../results/iron_oxide_masked.png'
    )

    print("\nIron oxide analysis complete!")


if __name__ == '__main__':
    main()
