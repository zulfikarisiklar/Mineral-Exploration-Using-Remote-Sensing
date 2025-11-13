"""
Download ASTER Imagery using Google Earth Engine

ASTER is particularly valuable for mineral exploration due to its
excellent SWIR bands (4-9) which are sensitive to clay minerals,
carbonates, and other alteration minerals.

Before running: earthengine authenticate
"""

import sys
sys.path.append('../src')

from utils.satellite_downloader import SatelliteDownloader


def main():
    """Download ASTER imagery for clay mineral detection"""

    # Initialize downloader
    downloader = SatelliteDownloader()

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  ASTER Download - Optimized for Clay Mineral Detection    ║
    ╚════════════════════════════════════════════════════════════╝

    ASTER advantages for mineral exploration:
    - 5 SWIR bands (1.6-2.4 μm) vs Landsat's 2 SWIR bands
    - Superior clay mineral discrimination
    - Can distinguish kaolinite, alunite, muscovite, etc.
    - 15m resolution in VNIR bands
    """)

    # Example 1: Porphyry copper deposit (Chile - El Teniente region)
    print("\n=== Example 1: Porphyry Copper Exploration (Chile) ===")

    aoi = downloader.set_aoi_from_coordinates(
        min_lon=-70.5,
        min_lat=-34.2,
        max_lon=-70.3,
        max_lat=-34.0
    )

    # Check available images
    info = downloader.get_image_info(
        collection_name='ASTER/AST_L1T_003',
        aoi=aoi,
        start_date='2010-01-01',  # ASTER archive goes back further
        end_date='2023-12-31'
    )

    print(f"Found {info['count']} ASTER images for this location")

    if info['count'] > 0:
        output_file = downloader.download_aster(
            aoi=aoi,
            start_date='2015-01-01',
            end_date='2023-12-31',
            output_path='../data/aster_chile_porphyry.tif',
            scale=30  # Using 30m for all bands
        )
        print(f"Downloaded: {output_file}")
    else:
        print("No ASTER images available for this location/date range")

    # Example 2: Epithermal gold system (Nevada, USA)
    print("\n\n=== Example 2: Epithermal Gold Exploration (Nevada) ===")

    # Comstock district area
    aoi = downloader.set_aoi_from_point(
        lon=-119.65,
        lat=39.30,
        buffer_km=8
    )

    info = downloader.get_image_info(
        collection_name='ASTER/AST_L1T_003',
        aoi=aoi,
        start_date='2010-01-01',
        end_date='2023-12-31'
    )

    print(f"Found {info['count']} ASTER images")

    if info['count'] > 0:
        output_file = downloader.download_aster(
            aoi=aoi,
            start_date='2015-01-01',
            end_date='2023-12-31',
            output_path='../data/aster_nevada_epithermal.tif',
            scale=30
        )
        print(f"Downloaded: {output_file}")

    # Example 3: VMS deposit exploration (Canada)
    print("\n\n=== Example 3: VMS Exploration (Abitibi, Canada) ===")

    # Abitibi greenstone belt
    aoi = downloader.set_aoi_from_coordinates(
        min_lon=-79.5,
        min_lat=48.0,
        max_lon=-79.3,
        max_lat=48.2
    )

    # Preview before downloading
    try:
        Map = downloader.preview_image_rgb(
            satellite='aster',
            aoi=aoi,
            start_date='2010-01-01',
            end_date='2023-12-31',
            output_html='../results/aster_preview_canada.html'
        )
        print("Preview saved!")
    except Exception as e:
        print(f"Preview error: {e}")

    info = downloader.get_image_info(
        collection_name='ASTER/AST_L1T_003',
        aoi=aoi,
        start_date='2010-01-01',
        end_date='2023-12-31'
    )

    print(f"Found {info['count']} ASTER images")

    if info['count'] > 0:
        output_file = downloader.download_aster(
            aoi=aoi,
            start_date='2015-01-01',
            end_date='2023-12-31',
            output_path='../data/aster_canada_vms.tif',
            scale=30
        )
        print(f"Downloaded: {output_file}")

    # Tips for using ASTER data
    print("""
    \n╔════════════════════════════════════════════════════════════╗
    ║  Tips for Using ASTER Data                                 ║
    ╚════════════════════════════════════════════════════════════╝

    Band Configuration:
    - Bands 1-3 (VNIR): 0.52-0.86 μm (15m resolution)
    - Bands 4-9 (SWIR): 1.6-2.43 μm (30m resolution)
    - Bands 10-14 (TIR): 8.1-11.65 μm (90m resolution)

    Best For:
    ✓ Clay mineral mapping (kaolinite, alunite, muscovite)
    ✓ Argillic and phyllic alteration
    ✓ Carbonate detection
    ✓ Silica mapping (using TIR bands)

    Limitations:
    ✗ Less frequent revisit than Landsat/Sentinel-2
    ✗ Smaller spatial coverage
    ✗ SWIR detector stopped collecting data in 2008
      (only VNIR and TIR available after 2008)

    Note: For recent imagery, use Landsat-8 or Sentinel-2
          For historical SWIR data (pre-2008), ASTER is excellent
    """)

    print("\n" + "="*60)
    print("ASTER download complete!")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've authenticated: earthengine authenticate")
