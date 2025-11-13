"""
Satellite Image Downloader
Downloads satellite imagery from Google Earth Engine using geemap
"""

import ee
import geemap
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime


class SatelliteDownloader:
    """
    Download satellite imagery from Google Earth Engine

    Supports:
    - Landsat 8 (OLI/TIRS) - 2013-present
    - Landsat 7 (ETM+) - 1999-present
    - Landsat 5 (TM) - 1984-2013 (historical)
    - Sentinel-2 (MSI) - 2015-present
    - ASTER - 2000-present
    """

    def __init__(self):
        """Initialize Earth Engine connection"""
        try:
            ee.Initialize()
            print("Earth Engine initialized successfully!")
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            print("Please run 'earthengine authenticate' first")
            raise

    def set_aoi_from_coordinates(self, min_lon: float, min_lat: float,
                                 max_lon: float, max_lat: float) -> ee.Geometry:
        """
        Set Area of Interest from bounding box coordinates

        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude

        Returns:
            Earth Engine Geometry object
        """
        aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        return aoi

    def set_aoi_from_point(self, lon: float, lat: float, buffer_km: float = 10) -> ee.Geometry:
        """
        Set Area of Interest from a center point with buffer

        Args:
            lon: Longitude of center point
            lat: Latitude of center point
            buffer_km: Buffer distance in kilometers

        Returns:
            Earth Engine Geometry object
        """
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(buffer_km * 1000)  # Convert to meters
        return aoi

    def download_landsat8(self,
                         aoi: ee.Geometry,
                         start_date: str,
                         end_date: str,
                         output_path: str,
                         cloud_cover_max: int = 20,
                         scale: int = 30) -> str:
        """
        Download Landsat-8 imagery

        Args:
            aoi: Area of Interest (ee.Geometry)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Output file path
            cloud_cover_max: Maximum cloud cover percentage
            scale: Resolution in meters (default 30m)

        Returns:
            Path to downloaded file
        """
        print(f"Downloading Landsat-8 imagery...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Max cloud cover: {cloud_cover_max}%")

        # Load Landsat 8 Collection 2 Tier 1 TOA
        collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
                     .sort('CLOUD_COVER'))

        # Get count
        count = collection.size().getInfo()
        print(f"Found {count} images matching criteria")

        if count == 0:
            raise ValueError("No images found for specified criteria")

        # Get median composite or first image
        if count > 1:
            image = collection.median()
            print("Creating median composite from multiple images")
        else:
            image = collection.first()
            print("Using single best image")

        # Select bands (all optical bands)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        image = image.select(bands)

        # Clip to AOI
        image = image.clip(aoi)

        # Download
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading to {output_path}...")
        geemap.ee_export_image(
            image,
            filename=str(output_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )

        print(f"Download complete: {output_path}")
        return str(output_path)

    def download_landsat7(self,
                         aoi: ee.Geometry,
                         start_date: str,
                         end_date: str,
                         output_path: str,
                         cloud_cover_max: int = 20,
                         scale: int = 30) -> str:
        """
        Download Landsat-7 imagery

        Args:
            aoi: Area of Interest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Output file path
            cloud_cover_max: Maximum cloud cover percentage
            scale: Resolution in meters

        Returns:
            Path to downloaded file
        """
        print(f"Downloading Landsat-7 imagery...")

        collection = (ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
                     .sort('CLOUD_COVER'))

        count = collection.size().getInfo()
        print(f"Found {count} images")

        if count == 0:
            raise ValueError("No images found")

        image = collection.median() if count > 1 else collection.first()

        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
        image = image.select(bands).clip(aoi)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        geemap.ee_export_image(
            image,
            filename=str(output_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )

        print(f"Download complete: {output_path}")
        return str(output_path)

    def download_landsat5(self,
                         aoi: ee.Geometry,
                         start_date: str,
                         end_date: str,
                         output_path: str,
                         cloud_cover_max: int = 20,
                         scale: int = 30) -> str:
        """
        Download Landsat-5 TM imagery (historical data 1984-2013)

        Args:
            aoi: Area of Interest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Output file path
            cloud_cover_max: Maximum cloud cover percentage
            scale: Resolution in meters

        Returns:
            Path to downloaded file
        """
        print(f"Downloading Landsat-5 TM imagery...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Note: Landsat-5 operated 1984-2013")

        # Load Landsat 5 Collection 2 Tier 1 TOA
        collection = (ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
                     .sort('CLOUD_COVER'))

        count = collection.size().getInfo()
        print(f"Found {count} images")

        if count == 0:
            raise ValueError("No images found for specified date range")

        image = collection.median() if count > 1 else collection.first()

        # Landsat 5 TM bands: B1-B5, B7 (no B6 thermal in TOA, no B8 pan)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
        image = image.select(bands).clip(aoi)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading to {output_path}...")
        geemap.ee_export_image(
            image,
            filename=str(output_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )

        print(f"Download complete: {output_path}")
        return str(output_path)

    def download_sentinel2(self,
                          aoi: ee.Geometry,
                          start_date: str,
                          end_date: str,
                          output_path: str,
                          cloud_cover_max: int = 20,
                          scale: int = 10) -> str:
        """
        Download Sentinel-2 imagery

        Args:
            aoi: Area of Interest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Output file path
            cloud_cover_max: Maximum cloud cover percentage
            scale: Resolution in meters (default 10m)

        Returns:
            Path to downloaded file
        """
        print(f"Downloading Sentinel-2 imagery...")

        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))

        count = collection.size().getInfo()
        print(f"Found {count} images")

        if count == 0:
            raise ValueError("No images found")

        image = collection.median() if count > 1 else collection.first()

        # Select bands relevant for mineral exploration
        bands = ['B1', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        image = image.select(bands).clip(aoi)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        geemap.ee_export_image(
            image,
            filename=str(output_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )

        print(f"Download complete: {output_path}")
        return str(output_path)

    def download_aster(self,
                      aoi: ee.Geometry,
                      start_date: str,
                      end_date: str,
                      output_path: str,
                      scale: int = 30) -> str:
        """
        Download ASTER imagery

        Args:
            aoi: Area of Interest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Output file path
            scale: Resolution in meters

        Returns:
            Path to downloaded file
        """
        print(f"Downloading ASTER imagery...")

        # ASTER L1T Radiance
        collection = (ee.ImageCollection('ASTER/AST_L1T_003')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .sort('system:time_start'))

        count = collection.size().getInfo()
        print(f"Found {count} images")

        if count == 0:
            raise ValueError("No images found")

        image = collection.median() if count > 1 else collection.first()

        # Select VNIR and SWIR bands (1-9)
        # Band 1-3: VNIR, Band 4-9: SWIR
        bands = ['B01', 'B02', 'B3N', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09']
        image = image.select(bands).clip(aoi)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        geemap.ee_export_image(
            image,
            filename=str(output_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )

        print(f"Download complete: {output_path}")
        return str(output_path)

    def get_image_info(self, collection_name: str, aoi: ee.Geometry,
                      start_date: str, end_date: str) -> Dict:
        """
        Get information about available images

        Args:
            collection_name: Name of image collection
            aoi: Area of Interest
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with image information
        """
        collection = (ee.ImageCollection(collection_name)
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date))

        count = collection.size().getInfo()

        if count == 0:
            return {'count': 0, 'images': []}

        # Get first few images info
        images = collection.limit(10).getInfo()['features']

        info = {
            'count': count,
            'collection': collection_name,
            'date_range': f"{start_date} to {end_date}",
            'sample_images': []
        }

        for img in images:
            props = img['properties']
            info['sample_images'].append({
                'id': img['id'],
                'date': props.get('system:time_start', 'Unknown'),
                'cloud_cover': props.get('CLOUD_COVER', props.get('CLOUDY_PIXEL_PERCENTAGE', 'N/A'))
            })

        return info

    def visualize_aoi(self, aoi: ee.Geometry, output_html: str = 'aoi_map.html'):
        """
        Visualize Area of Interest on interactive map

        Args:
            aoi: Area of Interest
            output_html: Output HTML file path
        """
        Map = geemap.Map()
        Map.centerObject(aoi, zoom=10)
        Map.addLayer(aoi, {'color': 'red'}, 'AOI')

        # Add basemaps
        Map.add_basemap('SATELLITE')

        Map.to_html(output_html)
        print(f"Map saved to {output_html}")

        return Map

    def preview_image_rgb(self,
                         satellite: str,
                         aoi: ee.Geometry,
                         start_date: str,
                         end_date: str,
                         output_html: str = 'preview.html'):
        """
        Preview satellite image as RGB composite

        Args:
            satellite: Satellite type ('landsat8', 'sentinel2', 'aster')
            aoi: Area of Interest
            start_date: Start date
            end_date: End date
            output_html: Output HTML file

        Returns:
            geemap Map object
        """
        Map = geemap.Map()
        Map.centerObject(aoi, zoom=10)

        if satellite.lower() == 'landsat8':
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUD_COVER', 20))
                         .median())
            vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}

        elif satellite.lower() == 'sentinel2':
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                         .median())
            vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}

        elif satellite.lower() == 'aster':
            collection = (ee.ImageCollection('ASTER/AST_L1T_003')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .median())
            vis_params = {'bands': ['B3N', 'B02', 'B01'], 'min': 0, 'max': 255}

        else:
            raise ValueError(f"Unsupported satellite: {satellite}")

        Map.addLayer(collection.clip(aoi), vis_params, f'{satellite} RGB')
        Map.addLayer(aoi, {'color': 'red'}, 'AOI')

        Map.to_html(output_html)
        print(f"Preview saved to {output_html}")

        return Map


class MineralExplorationDownloader:
    """
    Specialized downloader for mineral exploration
    Includes preprocessing and band selection optimized for mineral detection
    """

    def __init__(self):
        """Initialize downloader"""
        self.downloader = SatelliteDownloader()

    def download_for_iron_oxide_detection(self,
                                          aoi: ee.Geometry,
                                          start_date: str,
                                          end_date: str,
                                          output_dir: str,
                                          satellite: str = 'landsat8') -> str:
        """
        Download imagery optimized for iron oxide detection

        Args:
            aoi: Area of Interest
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            satellite: Satellite type

        Returns:
            Path to downloaded image
        """
        output_path = Path(output_dir) / f'{satellite}_iron_oxide.tif'

        if satellite == 'landsat8':
            return self.downloader.download_landsat8(
                aoi, start_date, end_date, str(output_path),
                cloud_cover_max=15  # Strict cloud cover for iron oxide
            )
        elif satellite == 'sentinel2':
            return self.downloader.download_sentinel2(
                aoi, start_date, end_date, str(output_path),
                cloud_cover_max=15
            )
        else:
            raise ValueError(f"Unsupported satellite: {satellite}")

    def download_for_clay_detection(self,
                                    aoi: ee.Geometry,
                                    start_date: str,
                                    end_date: str,
                                    output_dir: str) -> str:
        """
        Download ASTER imagery optimized for clay mineral detection

        ASTER has superior SWIR bands for clay minerals

        Args:
            aoi: Area of Interest
            start_date: Start date
            end_date: End date
            output_dir: Output directory

        Returns:
            Path to downloaded image
        """
        output_path = Path(output_dir) / 'aster_clay_minerals.tif'

        return self.downloader.download_aster(
            aoi, start_date, end_date, str(output_path)
        )

    def download_porphyry_exploration_suite(self,
                                           aoi: ee.Geometry,
                                           start_date: str,
                                           end_date: str,
                                           output_dir: str) -> Dict[str, str]:
        """
        Download complete suite of imagery for porphyry copper exploration

        Downloads:
        - Landsat-8 for general alteration mapping
        - ASTER for detailed clay/SWIR mapping (if available)
        - Sentinel-2 for high-resolution mapping

        Args:
            aoi: Area of Interest
            start_date: Start date
            end_date: End date
            output_dir: Output directory

        Returns:
            Dictionary of downloaded file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        print("\n=== Downloading Porphyry Exploration Suite ===\n")

        # Landsat-8
        try:
            print("1. Downloading Landsat-8...")
            paths['landsat8'] = self.downloader.download_landsat8(
                aoi, start_date, end_date,
                str(output_dir / 'landsat8_porphyry.tif'),
                cloud_cover_max=20
            )
        except Exception as e:
            print(f"Landsat-8 download failed: {e}")

        # ASTER
        try:
            print("\n2. Downloading ASTER...")
            paths['aster'] = self.downloader.download_aster(
                aoi, start_date, end_date,
                str(output_dir / 'aster_porphyry.tif')
            )
        except Exception as e:
            print(f"ASTER download failed: {e}")

        # Sentinel-2
        try:
            print("\n3. Downloading Sentinel-2...")
            paths['sentinel2'] = self.downloader.download_sentinel2(
                aoi, start_date, end_date,
                str(output_dir / 'sentinel2_porphyry.tif'),
                cloud_cover_max=20
            )
        except Exception as e:
            print(f"Sentinel-2 download failed: {e}")

        print(f"\n=== Download complete! Files in {output_dir} ===")

        return paths
