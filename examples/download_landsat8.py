"""
Download Landsat-8 Imagery using Google Earth Engine

This script demonstrates how to download Landsat-8 imagery for mineral exploration
using Google Earth Engine and geemap.

Before running:
1. Install dependencies: pip install earthengine-api geemap
2. Authenticate: earthengine authenticate
3. Follow the authentication flow
"""

import sys
sys.path.append('../src')

from utils.satellite_downloader import SatelliteDownloader
import ee


def main():
    """Download Landsat-8 imagery for a specific location"""

    # Initialize downloader
    downloader = SatelliteDownloader()

    # Example 1: Define AOI from coordinates (bounding box)
    # This example uses coordinates for a mining area in Chile (Escondida copper mine region)
    print("\n=== Example 1: Download using bounding box ===")

    min_lon, min_lat = -69.1, -24.4
    max_lon, max_lat = -68.9, -24.2

    aoi = downloader.set_aoi_from_coordinates(min_lon, min_lat, max_lon, max_lat)

    # Download Landsat-8 imagery
    # Date range: adjust to your needs
    output_file = downloader.download_landsat8(
        aoi=aoi,
        start_date='2023-01-01',
        end_date='2023-12-31',
        output_path='../data/landsat8_chile.tif',
        cloud_cover_max=15,
        scale=30  # 30m resolution
    )

    print(f"\nDownloaded: {output_file}")

    # Example 2: Define AOI from center point with buffer
    print("\n\n=== Example 2: Download using center point ===")

    # Example: Copper mining region in Arizona (Morenci)
    lon, lat = -109.35, 33.05
    buffer_km = 15  # 15 km radius

    aoi = downloader.set_aoi_from_point(lon, lat, buffer_km)

    output_file = downloader.download_landsat8(
        aoi=aoi,
        start_date='2023-06-01',
        end_date='2023-08-31',
        output_path='../data/landsat8_arizona.tif',
        cloud_cover_max=20,
        scale=30
    )

    print(f"\nDownloaded: {output_file}")

    # Example 3: Get information about available images before downloading
    print("\n\n=== Example 3: Check available images ===")

    # Example: Gold exploration area in Nevada
    aoi = downloader.set_aoi_from_point(-116.5, 40.5, buffer_km=10)

    info = downloader.get_image_info(
        collection_name='LANDSAT/LC08/C02/T1_TOA',
        aoi=aoi,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    print(f"\nFound {info['count']} images")
    print("\nSample images:")
    for img in info['sample_images'][:5]:
        print(f"  - {img['id']}")
        print(f"    Cloud cover: {img['cloud_cover']}%")

    # Example 4: Visualize AOI before downloading
    print("\n\n=== Example 4: Visualize AOI ===")

    # Example: Porphyry copper prospect in Peru
    aoi = downloader.set_aoi_from_coordinates(-76.5, -12.2, -76.3, -12.0)

    # Create interactive map
    Map = downloader.visualize_aoi(aoi, output_html='../results/aoi_preview.html')
    print("AOI map saved to ../results/aoi_preview.html")

    # Example 5: Preview RGB image before downloading
    print("\n\n=== Example 5: Preview RGB composite ===")

    Map = downloader.preview_image_rgb(
        satellite='landsat8',
        aoi=aoi,
        start_date='2023-01-01',
        end_date='2023-12-31',
        output_html='../results/landsat8_preview.html'
    )

    print("Preview saved to ../results/landsat8_preview.html")

    print("\n" + "="*60)
    print("Download complete! Check the 'data' directory for images.")
    print("="*60)


if __name__ == '__main__':
    # First-time setup instructions
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Landsat-8 Download Script - Google Earth Engine          ║
    ╚════════════════════════════════════════════════════════════╝

    Before running this script:
    1. Install Earth Engine: pip install earthengine-api geemap
    2. Authenticate: earthengine authenticate
    3. Follow the browser authentication flow

    This script will download Landsat-8 imagery for several example
    locations. Modify the coordinates for your area of interest.
    """)

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see an authentication error, run: earthengine authenticate")
