"""
Example 5: Detailed Mineral Mapping
Maps specific minerals using hydroxyl and carbonate detectors
"""

import sys
sys.path.append('../src')

import numpy as np
from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from utils.visualization import ResultVisualizer
from minerals.hydroxyl_minerals import HydroxylMineralDetector
from minerals.carbonate_minerals import CarbonateMineralDetector


def main():
    """Run detailed mineral mapping"""

    # Load and preprocess image
    print("Loading satellite image...")
    loader = SatelliteImageLoader('path/to/your/landsat8_image.tif')
    image = loader.load_image()

    preprocessor = ImagePreprocessor()
    image = preprocessor.normalize(image, method='percentile')

    print("\n=== HYDROXYL MINERAL MAPPING ===")

    # Initialize hydroxyl detector
    oh_detector = HydroxylMineralDetector(image, sensor='landsat8')

    print("1. Al-OH minerals (white micas, clays)")
    aloh = oh_detector.detect_aloh_minerals()
    white_micas = oh_detector.detect_white_micas()
    clay_minerals = oh_detector.detect_clay_minerals()

    print("2. Mg-OH minerals (chlorite, serpentine)")
    mgoh = oh_detector.detect_mgoh_minerals()
    chlorite = oh_detector.detect_chlorite()
    serpentine = oh_detector.detect_serpentine()

    print("3. Fe-OH minerals (biotite, Fe-chlorite)")
    feoh = oh_detector.detect_feoh_minerals()
    dark_micas = oh_detector.detect_dark_micas()

    print("4. Amphiboles")
    amphiboles = oh_detector.detect_amphiboles()

    print("\n=== CARBONATE MINERAL MAPPING ===")

    # Initialize carbonate detector
    carb_detector = CarbonateMineralDetector(image, sensor='landsat8')

    print("1. Calcite")
    calcite = carb_detector.detect_calcite()

    print("2. Dolomite")
    dolomite = carb_detector.detect_dolomite()

    print("3. Siderite (iron carbonate)")
    siderite = carb_detector.detect_siderite()

    print("4. Magnesite")
    magnesite = carb_detector.detect_magnesite()

    print("5. Carbonate discrimination")
    discrimination = carb_detector.discriminate_calcite_dolomite()

    # Visualize hydroxyl minerals
    print("\nVisualizing hydroxyl minerals...")
    visualizer = ResultVisualizer()

    oh_minerals = {
        'Al-OH (Micas/Clays)': aloh,
        'Mg-OH (Chlorite/Serpentine)': mgoh,
        'Fe-OH (Biotite)': feoh,
        'White Micas': white_micas,
        'Chlorite': chlorite,
        'Serpentine': serpentine
    }

    visualizer.plot_multiple_alterations(
        oh_minerals,
        save_path='../results/hydroxyl_minerals.png'
    )

    # Create OH composite RGB
    oh_composite = oh_detector.create_oh_composite_map()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(oh_composite)
    ax.set_title('OH Mineral Composite\n(R=Al-OH, G=Mg-OH, B=Fe-OH)',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('../results/oh_composite_rgb.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualize carbonate minerals
    print("\nVisualizing carbonate minerals...")

    carb_minerals = {
        'Calcite': calcite,
        'Dolomite': dolomite,
        'Siderite': siderite,
        'Magnesite': magnesite
    }

    visualizer.plot_multiple_alterations(
        carb_minerals,
        save_path='../results/carbonate_minerals.png'
    )

    # Plot calcite-dolomite discrimination
    discrimination_maps = {
        'Calcite Dominant': discrimination['calcite_dominant'],
        'Dolomite Dominant': discrimination['dolomite_dominant'],
        'Mixed Carbonate': discrimination['mixed']
    }

    visualizer.plot_multiple_alterations(
        discrimination_maps,
        save_path='../results/carbonate_discrimination.png'
    )

    # Create carbonate composite RGB
    carb_composite = carb_detector.create_carbonate_composite_map()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(carb_composite)
    ax.set_title('Carbonate Mineral Composite\n(R=Calcite, G=Dolomite, B=Siderite)',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('../results/carbonate_composite_rgb.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Generate mineral statistics
    print("\n=== MINERAL STATISTICS ===")
    print("\nHydroxyl Minerals:")
    for name, data in oh_minerals.items():
        coverage = (data > 0.7).sum() / data.size * 100
        mean_value = np.nanmean(data)
        print(f"  {name}: {coverage:.2f}% coverage, mean={mean_value:.3f}")

    print("\nCarbonate Minerals:")
    for name, data in carb_minerals.items():
        coverage = (data > 0.7).sum() / data.size * 100
        mean_value = np.nanmean(data)
        print(f"  {name}: {coverage:.2f}% coverage, mean={mean_value:.3f}")

    print("\nDetailed mineral mapping complete!")


if __name__ == '__main__':
    main()
