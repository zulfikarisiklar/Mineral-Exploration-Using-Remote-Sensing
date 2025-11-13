"""
Complete Porphyry Copper Exploration Workflow

This script demonstrates:
1. Downloading satellite imagery from Google Earth Engine
2. Running complete mineral exploration analysis
3. Generating alteration zonation maps
4. Creating exploration reports

Perfect for evaluating a new porphyry copper prospect!
"""

import sys
sys.path.append('../src')

from utils.satellite_downloader import MineralExplorationDownloader
from pipeline import MineralExplorationPipeline
import ee


def main():
    """Complete porphyry exploration workflow"""

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Complete Porphyry Copper Exploration Workflow            ║
    ║  Download → Analyze → Map → Report                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # =================================================================
    # STEP 1: Define Target Area
    # =================================================================

    print("\n" + "="*60)
    print("STEP 1: Define Area of Interest")
    print("="*60)

    # Example: Known porphyry copper deposit in Chile
    # Change these coordinates to your exploration target!

    # Option A: Use bounding box
    target_name = "Escondida_Region_Chile"
    min_lon, min_lat = -69.1, -24.4
    max_lon, max_lat = -68.9, -24.2

    # Option B: Use center point + buffer (uncomment to use)
    # target_name = "My_Prospect"
    # center_lon, center_lat = -70.0, -30.0
    # buffer_km = 15

    print(f"\nTarget: {target_name}")
    print(f"Coordinates: {min_lon}, {min_lat} to {max_lon}, {max_lat}")

    # =================================================================
    # STEP 2: Download Satellite Imagery
    # =================================================================

    print("\n" + "="*60)
    print("STEP 2: Download Satellite Imagery")
    print("="*60)

    downloader = MineralExplorationDownloader()

    # Define AOI
    aoi = downloader.downloader.set_aoi_from_coordinates(
        min_lon, min_lat, max_lon, max_lat
    )

    # Or if using center point:
    # aoi = downloader.downloader.set_aoi_from_point(center_lon, center_lat, buffer_km)

    # Define date range (use dry season for less clouds)
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    output_dir = f'../data/{target_name}'

    print(f"\nDate range: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")

    # Download complete exploration suite
    print("\n>>> Downloading exploration suite (Landsat-8, ASTER, Sentinel-2)...")

    try:
        downloaded_files = downloader.download_porphyry_exploration_suite(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir
        )

        print("\n✓ Downloaded imagery:")
        for satellite, path in downloaded_files.items():
            print(f"  - {satellite}: {path}")

    except Exception as e:
        print(f"Download error: {e}")
        print("\nTrying Landsat-8 only...")

        # Fallback: Just download Landsat-8
        landsat_path = downloader.downloader.download_landsat8(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            output_path=f'{output_dir}/landsat8.tif',
            cloud_cover_max=20
        )

        downloaded_files = {'landsat8': landsat_path}

    # =================================================================
    # STEP 3: Run Mineral Exploration Analysis
    # =================================================================

    print("\n" + "="*60)
    print("STEP 3: Run Mineral Exploration Analysis")
    print("="*60)

    # Use Landsat-8 for analysis (best for porphyry alteration)
    if 'landsat8' in downloaded_files:
        image_path = downloaded_files['landsat8']
        sensor = 'landsat8'
    elif 'sentinel2' in downloaded_files:
        image_path = downloaded_files['sentinel2']
        sensor = 'sentinel2'
    else:
        print("No suitable imagery downloaded!")
        return

    print(f"\nAnalyzing: {image_path}")
    print(f"Sensor: {sensor}")

    # Initialize analysis pipeline
    pipeline = MineralExplorationPipeline(
        image_path=image_path,
        sensor=sensor,
        output_dir=f'../results/{target_name}'
    )

    # Run complete analysis
    print("\n>>> Running complete mineral exploration analysis...")
    results = pipeline.run_complete_analysis()

    # =================================================================
    # STEP 4: Create Porphyry Zonation Map
    # =================================================================

    print("\n" + "="*60)
    print("STEP 4: Create Porphyry Zonation Map")
    print("="*60)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # Get alteration zones
    phyllic = results['alterations']['phyllic']['general']
    argillic = results['alterations']['argillic']['general']
    propylitic = results['alterations']['propylitic']['general']
    iron_oxide = results['alterations']['iron_oxide']['general']

    # Create zonation map
    zonation = np.zeros_like(phyllic, dtype=np.uint8)

    # Assign zones (higher priority zones override lower)
    zonation[propylitic > 0.6] = 1  # Outer propylitic
    zonation[argillic > 0.65] = 2    # Argillic
    zonation[phyllic > 0.7] = 3      # Phyllic (core)
    zonation[iron_oxide > 0.75] = 4  # Gossan

    # Plot zonation
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Zonation map
    colors = ['black', 'lightgreen', 'yellow', 'red', 'darkred']
    cmap = ListedColormap(colors)

    im1 = axes[0].imshow(zonation, cmap=cmap, vmin=0, vmax=4)
    axes[0].set_title(f'{target_name}\nPorphyry Copper Alteration Zonation',
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')

    legend_elements = [
        Patch(facecolor='black', label='Background'),
        Patch(facecolor='lightgreen', label='Propylitic (Outer Zone)'),
        Patch(facecolor='yellow', label='Argillic (Intermediate)'),
        Patch(facecolor='red', label='Phyllic (Core - HIGH PRIORITY)'),
        Patch(facecolor='darkred', label='Gossan (Oxidized Cap)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Individual alterations
    composite_alt = np.zeros((*phyllic.shape, 3))
    composite_alt[:, :, 0] = phyllic      # Red = Phyllic
    composite_alt[:, :, 1] = propylitic   # Green = Propylitic
    composite_alt[:, :, 2] = argillic     # Blue = Argillic

    axes[1].imshow(composite_alt)
    axes[1].set_title('Alteration Composite\n(R=Phyllic, G=Propylitic, B=Argillic)',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'../results/{target_name}/porphyry_zonation.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Zonation map saved")

    # =================================================================
    # STEP 5: Generate Exploration Report
    # =================================================================

    print("\n" + "="*60)
    print("STEP 5: Generate Exploration Report")
    print("="*60)

    # Calculate statistics
    total_pixels = zonation.size
    stats = {
        'Background': (zonation == 0).sum(),
        'Propylitic': (zonation == 1).sum(),
        'Argillic': (zonation == 2).sum(),
        'Phyllic (PRIORITY)': (zonation == 3).sum(),
        'Gossan': (zonation == 4).sum()
    }

    # Generate report
    report = []
    report.append("="*60)
    report.append(f"PORPHYRY COPPER EXPLORATION REPORT")
    report.append(f"Target: {target_name}")
    report.append("="*60)
    report.append(f"\nLocation: {min_lon}, {min_lat} to {max_lon}, {max_lat}")
    report.append(f"Date Range: {start_date} to {end_date}")
    report.append(f"Sensor: {sensor.upper()}")
    report.append(f"\n{'='*60}")
    report.append("ALTERATION ZONATION STATISTICS")
    report.append("="*60)

    for zone, count in stats.items():
        pct = (count / total_pixels) * 100
        km2 = (count * 30 * 30) / 1e6  # Approximate area in km²
        report.append(f"\n{zone}:")
        report.append(f"  Coverage: {pct:.2f}% ({count:,} pixels)")
        report.append(f"  Area: ~{km2:.2f} km²")

    # Priority targets
    high_priority_pct = (stats['Phyllic (PRIORITY)'] / total_pixels) * 100
    report.append(f"\n{'='*60}")
    report.append("EXPLORATION RECOMMENDATIONS")
    report.append("="*60)

    if high_priority_pct > 5:
        report.append("\n✓ HIGH PRIORITY TARGET")
        report.append(f"  Phyllic core zone: {high_priority_pct:.2f}% coverage")
        report.append("  Recommendation: Detailed ground follow-up")
        report.append("  - Soil geochemistry survey")
        report.append("  - Rock chip sampling")
        report.append("  - Structural mapping")
    elif high_priority_pct > 1:
        report.append("\n⚠ MODERATE PRIORITY")
        report.append(f"  Phyllic zone: {high_priority_pct:.2f}% coverage")
        report.append("  Recommendation: Further remote sensing analysis")
    else:
        report.append("\n○ LOW PRIORITY")
        report.append("  Limited phyllic alteration detected")
        report.append("  Recommendation: Consider other targets")

    report.append(f"\n{'='*60}")
    report.append("FILES GENERATED")
    report.append("="*60)
    report.append(f"\nImagery: {output_dir}/")
    report.append(f"Results: ../results/{target_name}/")
    report.append(f"  - porphyry_zonation.png")
    report.append(f"  - alteration maps (GeoTIFF)")
    report.append(f"  - analysis_report.txt")

    report_text = "\n".join(report)
    print("\n" + report_text)

    # Save report
    report_path = f'../results/{target_name}/exploration_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to: {report_path}")

    # =================================================================
    # DONE!
    # =================================================================

    print("\n" + "="*60)
    print("✓ COMPLETE WORKFLOW FINISHED!")
    print("="*60)
    print(f"\nAll outputs saved to: ../results/{target_name}/")
    print("\nNext steps:")
    print("  1. Review zonation maps")
    print("  2. Integrate with geological maps")
    print("  3. Plan field verification")
    print("  4. Conduct detailed geochemical sampling")
    print("="*60)


if __name__ == '__main__':
    print("""
    Before running:
    1. Authenticate: earthengine authenticate
    2. Modify coordinates for your target area
    3. Ensure you have ~100MB free space for downloads
    """)

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
