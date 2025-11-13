"""
Download Sentinel-2 Imagery using Google Earth Engine

Sentinel-2 offers:
- Higher spatial resolution (10m for visible/NIR)
- Frequent revisit time (5 days with both satellites)
- Additional red-edge bands useful for vegetation masking
- Free and open access

Before running: earthengine authenticate
"""

import sys
sys.path.append('../src')

from utils.satellite_downloader import SatelliteDownloader


def main():
    """Download Sentinel-2 imagery for mineral exploration"""

    downloader = SatelliteDownloader()

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Sentinel-2 Download - High Resolution Mineral Mapping    ║
    ╚════════════════════════════════════════════════════════════╝

    Sentinel-2 advantages:
    - 10m resolution (B2, B3, B4, B8)
    - 20m resolution (B5, B6, B7, B8A, B11, B12)
    - Frequent revisit (5 days)
    - Excellent for detailed mapping
    - Good SWIR bands (B11, B12) for alteration
    """)

    # Example 1: Copper porphyry (Chuquicamata, Chile)
    print("\n=== Example 1: Chuquicamata Copper Mine (Chile) ===")

    aoi = downloader.set_aoi_from_coordinates(
        min_lon=-68.95,
        min_lat=-22.35,
        max_lon=-68.85,
        max_lat=-22.25
    )

    # Check available images
    info = downloader.get_image_info(
        collection_name='COPERNICUS/S2_SR_HARMONIZED',
        aoi=aoi,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    print(f"Found {info['count']} Sentinel-2 images")
    print("\nSample images with cloud cover:")
    for img in info['sample_images'][:5]:
        print(f"  {img['cloud_cover']}% - {img['id']}")

    output_file = downloader.download_sentinel2(
        aoi=aoi,
        start_date='2023-06-01',
        end_date='2023-08-31',  # Dry season - less clouds
        output_path='../data/sentinel2_chuquicamata.tif',
        cloud_cover_max=10,  # Very low cloud cover
        scale=10  # 10m resolution
    )

    print(f"\nDownloaded: {output_file}")

    # Example 2: Gold exploration (Carlin Trend, Nevada)
    print("\n\n=== Example 2: Carlin Trend Gold District (Nevada) ===")

    aoi = downloader.set_aoi_from_point(
        lon=-116.3,
        lat=40.7,
        buffer_km=12
    )

    # Preview before downloading
    Map = downloader.preview_image_rgb(
        satellite='sentinel2',
        aoi=aoi,
        start_date='2023-07-01',
        end_date='2023-09-30',
        output_html='../results/sentinel2_preview_carlin.html'
    )
    print("Preview saved to ../results/sentinel2_preview_carlin.html")

    output_file = downloader.download_sentinel2(
        aoi=aoi,
        start_date='2023-07-01',
        end_date='2023-09-30',
        output_path='../data/sentinel2_carlin.tif',
        cloud_cover_max=15,
        scale=10
    )

    print(f"Downloaded: {output_file}")

    # Example 3: Iron oxide deposits (Pilbara, Australia)
    print("\n\n=== Example 3: Iron Ore Deposits (Pilbara, Australia) ===")

    aoi = downloader.set_aoi_from_coordinates(
        min_lon=118.5,
        min_lat=-23.0,
        max_lon=118.7,
        max_lat=-22.8
    )

    info = downloader.get_image_info(
        collection_name='COPERNICUS/S2_SR_HARMONIZED',
        aoi=aoi,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    print(f"Found {info['count']} images")

    output_file = downloader.download_sentinel2(
        aoi=aoi,
        start_date='2023-05-01',
        end_date='2023-10-31',
        output_path='../data/sentinel2_pilbara.tif',
        cloud_cover_max=20,
        scale=10
    )

    print(f"Downloaded: {output_file}")

    # Example 4: Seasonal comparison for better results
    print("\n\n=== Example 4: Seasonal Download Strategy ===")

    # Download dry season imagery for better cloud-free coverage
    # Example: Andean copper belt

    aoi = downloader.set_aoi_from_point(
        lon=-70.0,
        lat=-30.5,
        buffer_km=10
    )

    print("\nDownloading DRY SEASON (May-September) - fewer clouds:")
    output_dry = downloader.download_sentinel2(
        aoi=aoi,
        start_date='2023-05-01',
        end_date='2023-09-30',
        output_path='../data/sentinel2_andes_dry.tif',
        cloud_cover_max=10,
        scale=10
    )

    print(f"Dry season image: {output_dry}")

    # Tips for Sentinel-2
    print("""
    \n╔════════════════════════════════════════════════════════════╗
    ║  Tips for Using Sentinel-2 Data                            ║
    ╚════════════════════════════════════════════════════════════╝

    Band Selection for Mineral Exploration:
    - B2 (Blue, 490nm, 10m): Iron oxide detection
    - B3 (Green, 560nm, 10m): Vegetation, alteration
    - B4 (Red, 665nm, 10m): Iron oxide, vegetation
    - B8 (NIR, 842nm, 10m): Vegetation masking, ferrous minerals
    - B11 (SWIR1, 1610nm, 20m): Clay minerals, hydroxyl
    - B12 (SWIR2, 2190nm, 20m): Clay minerals, carbonates

    Best Practices:
    ✓ Use dry season imagery for less cloud cover
    ✓ Set strict cloud cover thresholds (< 15%)
    ✓ Create seasonal composites for best coverage
    ✓ Use 10m resolution for detailed mapping
    ✓ Combine with Landsat-8 for temporal coverage

    Date Ranges by Region:
    - Chile/Peru (Andes): May-September (dry)
    - Nevada/Arizona (SW USA): April-October
    - Australia (Pilbara): May-October
    - Canada: June-August

    Processing Tips:
    - Download at 10m for maximum detail
    - Use B4/B3/B2 for true color
    - Use B11/B8/B4 for false color alteration mapping
    """)

    print("\n" + "="*60)
    print("Sentinel-2 downloads complete!")
    print("Check 'data' directory for imagery.")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've authenticated: earthengine authenticate")
