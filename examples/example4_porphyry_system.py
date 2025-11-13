"""
Example 4: Porphyry Copper System Detection
Maps alteration zonation typical of porphyry deposits
"""

import sys
sys.path.append('../src')

import numpy as np
from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from utils.visualization import ResultVisualizer
from alterations.phyllic import PhyllicDetector
from alterations.propylitic import PropyliticDetector
from alterations.argillic import ArgillicDetector
from alterations.iron_oxide import IronOxideDetector


def main():
    """Detect porphyry copper alteration zonation"""

    # Load and preprocess image
    print("Loading satellite image...")
    loader = SatelliteImageLoader('path/to/your/landsat8_image.tif')
    image = loader.load_image()

    preprocessor = ImagePreprocessor()
    image = preprocessor.normalize(image, method='percentile')

    print("\n=== PORPHYRY COPPER ALTERATION MAPPING ===")

    # Detect core zone (phyllic/potassic)
    print("\n1. Phyllic alteration (core zone)")
    phyllic_detector = PhyllicDetector(image, sensor='landsat8')
    phyllic = phyllic_detector.detect_with_pyrite()  # Phyllic with pyrite
    sericite = phyllic_detector.detect_sericite()

    # Detect intermediate zone (argillic)
    print("2. Argillic alteration (intermediate zone)")
    argillic_detector = ArgillicDetector(image, sensor='landsat8')
    argillic = argillic_detector.detect_intermediate_argillic()

    # Detect outer zone (propylitic)
    print("3. Propylitic alteration (outer zone)")
    propylitic_detector = PropyliticDetector(image, sensor='landsat8')
    propylitic = propylitic_detector.detect_composite_method()
    inner_prop = propylitic_detector.detect_inner_propylitic()
    outer_prop = propylitic_detector.detect_outer_propylitic()

    # Detect gossan (if present)
    print("4. Iron oxide/Gossan (oxidized cap)")
    iron_detector = IronOxideDetector(image, sensor='landsat8')
    iron_oxide = iron_detector.detect_composite_method()

    # Create alteration zonation map
    print("\nCreating alteration zonation map...")

    # Assign zones with priority (inner zones override outer)
    zonation = np.zeros_like(phyllic, dtype=np.uint8)

    # Zone 5: Outer propylitic (lowest priority)
    zonation[outer_prop > 0.6] = 1

    # Zone 4: Inner propylitic
    zonation[inner_prop > 0.65] = 2

    # Zone 3: Argillic
    zonation[argillic > 0.7] = 3

    # Zone 2: Phyllic (highest priority - core)
    zonation[phyllic > 0.7] = 4

    # Zone 1: Gossan cap (if present)
    zonation[iron_oxide > 0.75] = 5

    # Visualize results
    print("\nVisualizing porphyry zonation...")
    visualizer = ResultVisualizer()

    # Plot individual zones
    alteration_zones = {
        'Phyllic (Core)': phyllic,
        'Argillic (Intermediate)': argillic,
        'Inner Propylitic': inner_prop,
        'Outer Propylitic': outer_prop,
        'Iron Oxide (Gossan)': iron_oxide
    }

    visualizer.plot_multiple_alterations(
        alteration_zones,
        save_path='../results/porphyry_zones.png'
    )

    # Create composite map showing all zones
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors for zones
    colors = ['black', 'lightgreen', 'green', 'yellow', 'red', 'darkred']
    cmap = ListedColormap(colors)

    im = ax.imshow(zonation, cmap=cmap, vmin=0, vmax=5)
    ax.set_title('Porphyry Copper Alteration Zonation', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Background'),
        Patch(facecolor='lightgreen', label='Outer Propylitic'),
        Patch(facecolor='green', label='Inner Propylitic'),
        Patch(facecolor='yellow', label='Argillic'),
        Patch(facecolor='red', label='Phyllic (Core)'),
        Patch(facecolor='darkred', label='Gossan Cap')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('../results/porphyry_zonation_composite.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Generate statistics
    print("\n=== ZONATION STATISTICS ===")
    total_pixels = zonation.size
    for zone_id, zone_name in enumerate([
        'Background',
        'Outer Propylitic',
        'Inner Propylitic',
        'Argillic',
        'Phyllic',
        'Gossan'
    ]):
        count = (zonation == zone_id).sum()
        percentage = count / total_pixels * 100
        print(f"{zone_name}: {percentage:.2f}% ({count} pixels)")

    # Calculate target priority areas
    # High priority: Phyllic and inner propylitic zones
    high_priority = (zonation >= 2) & (zonation <= 4)
    priority_percentage = high_priority.sum() / total_pixels * 100

    print(f"\n=== TARGET PRIORITY ===")
    print(f"High Priority Areas: {priority_percentage:.2f}%")
    print("(Includes: Phyllic, Argillic, Inner Propylitic zones)")

    print("\nPorphyry system analysis complete!")


if __name__ == '__main__':
    main()
