"""
Example 6: Advanced Mineral Detection
Demonstrates new detection capabilities:
- Silica (SiO2) mineralization
- Potassic alteration
- Sulfate minerals
"""

import sys
sys.path.append('../src')

import numpy as np
from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from utils.visualization import ResultVisualizer
from minerals.silica_minerals import SilicaMineralizationDetector
from alterations.potassic import PotassicDetector
from minerals.sulfate_minerals import SulfateMineralDetector


def main():
    """Run advanced mineral detection analysis"""

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Advanced Mineral Detection Example                       ║
    ║  Silica, Potassic, and Sulfate Minerals                  ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Load and preprocess image
    print("\n=== Loading and Preprocessing Image ===")
    loader = SatelliteImageLoader('path/to/your/landsat8_image.tif')
    image = loader.load_image()

    preprocessor = ImagePreprocessor()
    image = preprocessor.normalize(image, method='percentile')

    print(f"Image shape: {image.shape}")

    #=================================================================
    # SILICA MINERALIZATION DETECTION
    #=================================================================

    print("\n" + "="*60)
    print("SILICA (SiO2) MINERALIZATION DETECTION")
    print("="*60)

    silica_detector = SilicaMineralizationDetector(image, sensor='landsat8')

    print("\n1. General quartz detection (VNIR-SWIR)")
    quartz = silica_detector.detect_quartz_vnir_swir()

    print("2. Silicification zones")
    silicification = silica_detector.detect_silicification()

    print("3. Quartz veins (textural analysis)")
    quartz_veins = silica_detector.detect_quartz_veins()

    print("4. Silica cap (epithermal indicator)")
    silica_cap = silica_detector.detect_silica_cap()

    print("5. Opaline silica (hot spring environment)")
    opaline_silica = silica_detector.detect_opaline_silica()

    print("6. Epithermal silica system assessment")
    epithermal_silica = silica_detector.detect_epithermal_silica_system()

    # Visualize silica results
    print("\nVisualizing silica detection results...")
    visualizer = ResultVisualizer()

    silica_types = {
        'General Quartz': quartz,
        'Silicification': silicification,
        'Quartz Veins': quartz_veins,
        'Silica Cap': silica_cap,
        'Opaline Silica': opaline_silica,
        'Epithermal Composite': epithermal_silica['epithermal_composite']
    }

    visualizer.plot_multiple_alterations(
        silica_types,
        save_path='../results/silica_mineralization.png'
    )

    #=================================================================
    # POTASSIC ALTERATION DETECTION
    #=================================================================

    print("\n" + "="*60)
    print("POTASSIC ALTERATION DETECTION")
    print("="*60)

    potassic_detector = PotassicDetector(image, sensor='landsat8')

    print("\n1. Biotite detection")
    biotite = potassic_detector.detect_biotite()

    print("2. K-feldspar detection")
    k_feldspar = potassic_detector.detect_k_feldspar()

    print("3. Magnetite detection")
    magnetite = potassic_detector.detect_magnetite()

    print("4. Early potassic alteration (core zone)")
    early_potassic = potassic_detector.detect_early_potassic()

    print("5. Late potassic alteration")
    late_potassic = potassic_detector.detect_late_potassic()

    print("6. Porphyry core zone")
    porphyry_core = potassic_detector.detect_porphyry_core()

    print("7. Potassic-Phyllic transition")
    transition = potassic_detector.detect_potassic_phyllic_transition()

    # Visualize potassic results
    print("\nVisualizing potassic alteration results...")

    potassic_types = {
        'Biotite': biotite,
        'K-Feldspar': k_feldspar,
        'Magnetite': magnetite,
        'Early Potassic (Core)': early_potassic,
        'Late Potassic': late_potassic,
        'Porphyry Core': porphyry_core
    }

    visualizer.plot_multiple_alterations(
        potassic_types,
        save_path='../results/potassic_alteration.png'
    )

    #=================================================================
    # SULFATE MINERAL DETECTION
    #=================================================================

    print("\n" + "="*60)
    print("SULFATE MINERAL DETECTION")
    print("="*60)

    sulfate_detector = SulfateMineralDetector(image, sensor='landsat8')

    print("\n1. Gypsum (CaSO₄·2H₂O)")
    gypsum = sulfate_detector.detect_gypsum()

    print("2. Jarosite (KFe₃(SO₄)₂(OH)₆) - acid sulfate")
    jarosite = sulfate_detector.detect_jarosite()

    print("3. Alunite (KAl₃(SO₄)₂(OH)₆) - advanced argillic")
    alunite = sulfate_detector.detect_alunite()

    print("4. Anhydrite (CaSO₄)")
    anhydrite = sulfate_detector.detect_anhydrite()

    print("5. Acid sulfate alteration")
    acid_sulfate = sulfate_detector.detect_acid_sulfate_alteration()

    print("6. Supergene sulfates (oxidation zone)")
    supergene = sulfate_detector.detect_supergene_sulfates()

    print("7. Epithermal sulfates")
    epithermal_sulfates = sulfate_detector.detect_epithermal_sulfates()

    # Visualize sulfate results
    print("\nVisualizing sulfate mineral results...")

    sulfate_types = {
        'Gypsum': gypsum,
        'Jarosite': jarosite,
        'Alunite': alunite,
        'Anhydrite': anhydrite,
        'Acid Sulfate': acid_sulfate,
        'Supergene': supergene
    }

    visualizer.plot_multiple_alterations(
        sulfate_types,
        save_path='../results/sulfate_minerals.png'
    )

    # Create RGB composite for sulfates
    print("\nCreating sulfate RGB composite...")
    sulfate_composite = sulfate_detector.create_sulfate_composite_map()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(sulfate_composite)
    ax.set_title('Sulfate Mineral Composite\n(R=Jarosite, G=Alunite, B=Gypsum)',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('../results/sulfate_composite_rgb.png', dpi=300, bbox_inches='tight')
    plt.show()

    #=================================================================
    # INTEGRATED ANALYSIS
    #=================================================================

    print("\n" + "="*60)
    print("INTEGRATED MINERAL SYSTEM ANALYSIS")
    print("="*60)

    # Epithermal Gold System Assessment
    print("\n>>> Epithermal Gold System Indicators:")

    epithermal_score = (
        silica_cap * 1.5 +
        quartz_veins * 1.2 +
        acid_sulfate * 1.3 +
        alunite
    ) / 5.0

    epithermal_score_norm = (epithermal_score - np.nanmin(epithermal_score)) / \
                            (np.nanmax(epithermal_score) - np.nanmin(epithermal_score))

    # Visualize epithermal system
    visualizer.plot_alteration_zones(
        epithermal_score_norm,
        'Epithermal Gold System Indicator',
        threshold=0.7,
        save_path='../results/epithermal_gold_indicator.png'
    )

    # Porphyry Copper System Assessment
    print("\n>>> Porphyry Copper System Indicators:")

    porphyry_score = (
        porphyry_core * 2.0 +
        early_potassic * 1.5 +
        biotite * 1.2
    ) / 4.7

    porphyry_score_norm = (porphyry_score - np.nanmin(porphyry_score)) / \
                          (np.nanmax(porphyry_score) - np.nanmin(porphyry_score))

    visualizer.plot_alteration_zones(
        porphyry_score_norm,
        'Porphyry Copper Core Indicator',
        threshold=0.7,
        save_path='../results/porphyry_core_indicator.png'
    )

    # Supergene Enrichment Zone
    print("\n>>> Supergene Enrichment Zone:")

    supergene_enrichment = (
        supergene * 1.5 +
        jarosite * 1.3 +
        gypsum
    ) / 3.8

    supergene_enrichment_norm = (supergene_enrichment - np.nanmin(supergene_enrichment)) / \
                                (np.nanmax(supergene_enrichment) - np.nanmin(supergene_enrichment))

    visualizer.plot_alteration_zones(
        supergene_enrichment_norm,
        'Supergene Enrichment Zone',
        threshold=0.65,
        save_path='../results/supergene_enrichment.png'
    )

    #=================================================================
    # STATISTICS AND REPORTING
    #=================================================================

    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)

    # Calculate coverage statistics
    threshold = 0.7
    total_pixels = epithermal_score_norm.size

    print("\n=== Silica Mineralization ===")
    for name, data in [('Quartz General', quartz),
                       ('Silicification', silicification),
                       ('Quartz Veins', quartz_veins),
                       ('Silica Cap', silica_cap)]:
        coverage = (data > threshold).sum() / total_pixels * 100
        mean_val = np.nanmean(data)
        print(f"{name:20s}: {coverage:5.2f}% coverage, mean={mean_val:.3f}")

    print("\n=== Potassic Alteration ===")
    for name, data in [('Biotite', biotite),
                       ('K-Feldspar', k_feldspar),
                       ('Early Potassic', early_potassic),
                       ('Porphyry Core', porphyry_core)]:
        coverage = (data > threshold).sum() / total_pixels * 100
        mean_val = np.nanmean(data)
        print(f"{name:20s}: {coverage:5.2f}% coverage, mean={mean_val:.3f}")

    print("\n=== Sulfate Minerals ===")
    for name, data in [('Gypsum', gypsum),
                       ('Jarosite', jarosite),
                       ('Alunite', alunite),
                       ('Acid Sulfate', acid_sulfate)]:
        coverage = (data > threshold).sum() / total_pixels * 100
        mean_val = np.nanmean(data)
        print(f"{name:20s}: {coverage:5.2f}% coverage, mean={mean_val:.3f}")

    print("\n=== Deposit Type Indicators ===")
    for name, data in [('Epithermal Gold', epithermal_score_norm),
                       ('Porphyry Copper', porphyry_score_norm),
                       ('Supergene Enrichment', supergene_enrichment_norm)]:
        coverage = (data > threshold).sum() / total_pixels * 100
        mean_val = np.nanmean(data)
        print(f"{name:20s}: {coverage:5.2f}% coverage, mean={mean_val:.3f}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

    print("""
    \nKey Findings Summary:

    Silica Mineralization:
    - Quartz veins indicate hydrothermal fluid pathways
    - Silica caps suggest epithermal systems
    - Opaline silica indicates hot spring environment

    Potassic Alteration:
    - Early potassic marks porphyry core zones
    - Biotite + K-feldspar + magnetite = high-T alteration
    - Target for porphyry copper exploration

    Sulfate Minerals:
    - Jarosite = acid sulfate alteration (oxidation)
    - Alunite = advanced argillic (epithermal)
    - Gypsum = supergene/weathering products

    All results saved to: ../results/
    """)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Replace 'path/to/your/landsat8_image.tif' with actual image path")
