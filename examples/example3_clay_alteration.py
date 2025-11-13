"""
Example 3: Clay Alteration Detection (Argillic and Phyllic)
Demonstrates detection and discrimination of clay-bearing alterations
"""

import sys
sys.path.append('../src')

import numpy as np
from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from utils.visualization import ResultVisualizer
from alterations.argillic import ArgillicDetector
from alterations.phyllic import PhyllicDetector


def main():
    """Run clay alteration analysis"""

    # Load and preprocess image
    print("Loading and preprocessing image...")
    loader = SatelliteImageLoader('path/to/your/landsat8_image.tif')
    image = loader.load_image()

    preprocessor = ImagePreprocessor()
    image = preprocessor.normalize(image, method='percentile')

    # Detect argillic alteration
    print("\n=== ARGILLIC ALTERATION DETECTION ===")
    argillic_detector = ArgillicDetector(image, sensor='landsat8')

    print("1. General argillic detection")
    argillic_general = argillic_detector.detect_composite_method()

    print("2. Kaolinite detection")
    kaolinite = argillic_detector.detect_kaolinite()

    print("3. Alunite detection (advanced argillic)")
    alunite = argillic_detector.detect_alunite()

    print("4. Advanced argillic zones")
    advanced_argillic = argillic_detector.detect_advanced_argillic()

    # Detect phyllic alteration
    print("\n=== PHYLLIC ALTERATION DETECTION ===")
    phyllic_detector = PhyllicDetector(image, sensor='landsat8')

    print("1. General phyllic detection")
    phyllic_general = phyllic_detector.detect_composite_method()

    print("2. Sericite detection")
    sericite = phyllic_detector.detect_sericite()

    print("3. Muscovite detection")
    muscovite = phyllic_detector.detect_muscovite()

    print("4. Phyllic with pyrite")
    phyllic_pyrite = phyllic_detector.detect_with_pyrite()

    # Discriminate between argillic and phyllic
    print("\n=== DISCRIMINATION ===")
    argillic_specific = argillic_detector.distinguish_from_phyllic(phyllic_general)
    phyllic_specific = phyllic_detector.distinguish_from_argillic(argillic_general)

    # Visualize results
    print("\nVisualizing results...")
    visualizer = ResultVisualizer()

    # Plot argillic types
    argillic_types = {
        'General Argillic': argillic_general,
        'Kaolinite': kaolinite,
        'Alunite': alunite,
        'Advanced Argillic': advanced_argillic
    }

    visualizer.plot_multiple_alterations(
        argillic_types,
        save_path='../results/argillic_types.png'
    )

    # Plot phyllic types
    phyllic_types = {
        'General Phyllic': phyllic_general,
        'Sericite': sericite,
        'Muscovite': muscovite,
        'Phyllic + Pyrite': phyllic_pyrite
    }

    visualizer.plot_multiple_alterations(
        phyllic_types,
        save_path='../results/phyllic_types.png'
    )

    # Plot discrimination
    discrimination = {
        'Argillic (Specific)': argillic_specific,
        'Phyllic (Specific)': phyllic_specific,
        'Argillic (General)': argillic_general,
        'Phyllic (General)': phyllic_general
    }

    visualizer.plot_multiple_alterations(
        discrimination,
        save_path='../results/clay_discrimination.png'
    )

    print("\nClay alteration analysis complete!")


if __name__ == '__main__':
    main()
