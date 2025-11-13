# Mineral Exploration Using Remote Sensing

A comprehensive Python toolkit for detecting minerals and hydrothermal alteration zones from satellite imagery using multiple analysis techniques.

## Overview

This project provides a complete pipeline for mineral exploration using remote sensing data from satellites like Landsat-8, Landsat-7, ASTER, and Sentinel-2. It implements various spectral analysis techniques to identify different types of hydrothermal alterations and mineral deposits.

## Features

### Analysis Methods
- **Band Ratio Analysis**: Classic ratio techniques for mineral detection
- **Principal Component Analysis (PCA)**: Dimensionality reduction and feature enhancement
- **Spectral Indices**: Specialized indices for different mineral types

### Alteration Detection
Each alteration type is detected separately with multiple sub-type discriminations:

1. **Iron Oxide Alteration**
   - General ferric iron detection
   - Hematite (Fe₂O₃)
   - Goethite (FeO(OH))
   - Gossan (oxidized ore caps)

2. **Argillic Alteration**
   - Kaolinite
   - Dickite
   - Alunite
   - Advanced argillic assemblages
   - Intermediate argillic zones

3. **Phyllic Alteration**
   - Sericite (fine-grained muscovite)
   - Muscovite
   - Illite
   - Phyllic with pyrite
   - Potassic-phyllic transition zones

4. **Propylitic Alteration**
   - Chlorite
   - Epidote
   - Carbonate minerals
   - Actinolite
   - Inner vs outer propylitic zones

### Mineral-Specific Detection

1. **Hydroxyl (OH) Bearing Minerals**
   - Al-OH minerals (white micas, clays)
   - Mg-OH minerals (chlorite, serpentine, talc)
   - Fe-OH minerals (biotite, Fe-chlorite)
   - Amphiboles
   - Mineral discrimination by absorption wavelength

2. **Carbonate Minerals**
   - Calcite (CaCO₃)
   - Dolomite (CaMg(CO₃)₂)
   - Siderite (FeCO₃)
   - Magnesite (MgCO₃)
   - Calcite-dolomite discrimination
   - Marble detection

## Downloading Satellite Imagery

**NEW!** This toolkit now includes automatic satellite image download from **Google Earth Engine** using geemap!

### Supported Data Sources
- **Landsat-8** (OLI/TIRS) - 30m resolution, excellent SWIR bands
- **Landsat-7** (ETM+) - 30m resolution, historical data
- **Sentinel-2** (MSI) - 10m resolution, frequent revisit
- **ASTER** - Superior SWIR bands for clay minerals (note: SWIR stopped in 2008)

### Quick Start - Download Imagery

1. **Authenticate with Google Earth Engine:**
```bash
earthengine authenticate
```
Follow the browser authentication flow.

2. **Download imagery for your area:**
```python
from utils.satellite_downloader import SatelliteDownloader

downloader = SatelliteDownloader()

# Define area of interest
aoi = downloader.set_aoi_from_coordinates(
    min_lon=-70.0, min_lat=-30.0,
    max_lon=-69.8, max_lat=-29.8
)

# Download Landsat-8
downloader.download_landsat8(
    aoi=aoi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    output_path='data/my_area.tif',
    cloud_cover_max=15
)
```

3. **Or use the complete exploration suite:**
```python
from utils.satellite_downloader import MineralExplorationDownloader

downloader = MineralExplorationDownloader()

# Downloads Landsat-8, ASTER, and Sentinel-2 automatically
paths = downloader.download_porphyry_exploration_suite(
    aoi=aoi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    output_dir='data/my_prospect'
)
```

### Download Examples
See the `examples/` directory for detailed download scripts:
- `download_landsat8.py` - Landsat-8 download examples
- `download_aster.py` - ASTER download for clay minerals
- `download_sentinel2.py` - High-resolution Sentinel-2
- `download_and_analyze_porphyry.py` - Complete workflow (download → analyze → report)

### Earth Engine Benefits
- ✓ Automatic cloud filtering
- ✓ Temporal compositing (median, mean)
- ✓ On-the-fly preprocessing
- ✓ Access to massive archive (1970s to present)
- ✓ No manual USGS downloads needed!

## Installation

### Requirements
- Python 3.7+
- GDAL/Rasterio for geospatial processing
- NumPy, SciPy for numerical computing
- scikit-learn for machine learning
- Matplotlib, Seaborn for visualization
- **Google Earth Engine API** and **geemap** for satellite image download

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Earth Engine Setup (for downloading imagery)

1. **Sign up for Google Earth Engine:**
   - Visit: https://earthengine.google.com/
   - Click "Get Started" and sign up (free for research/education)

2. **Authenticate:**
```bash
earthengine authenticate
```
This opens a browser for authentication. Follow the prompts and paste the token.

3. **Verify setup:**
```python
import ee
ee.Initialize()
print("Earth Engine ready!")
```

### GDAL Installation

GDAL can be tricky to install. Platform-specific instructions:

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version)
```

**macOS:**
```bash
brew install gdal
pip install gdal==$(gdal-config --version)
```

**Windows:**
Download pre-built wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

## Project Structure

```
Mineral-Exploration-Using-Remote-Sensing/
├── src/
│   ├── analysis/               # Analysis techniques
│   │   ├── band_ratio.py      # Band ratio analysis
│   │   ├── pca_analysis.py    # Principal component analysis
│   │   └── spectral_indices.py # Spectral indices
│   ├── alterations/           # Alteration detection
│   │   ├── iron_oxide.py      # Iron oxide detection
│   │   ├── argillic.py        # Argillic alteration
│   │   ├── phyllic.py         # Phyllic alteration
│   │   └── propylitic.py      # Propylitic alteration
│   ├── minerals/              # Mineral-specific detection
│   │   ├── hydroxyl_minerals.py  # OH-bearing minerals
│   │   └── carbonate_minerals.py # Carbonate minerals
│   ├── utils/                 # Utility modules
│   │   ├── image_loader.py    # Image loading and I/O
│   │   ├── preprocessing.py   # Preprocessing functions
│   │   └── visualization.py   # Visualization tools
│   └── pipeline.py            # Main processing pipeline
├── examples/                  # Example scripts
│   ├── example1_basic_analysis.py
│   ├── example2_iron_oxide_detection.py
│   ├── example3_clay_alteration.py
│   ├── example4_porphyry_system.py
│   └── example5_mineral_mapping.py
├── data/                      # Input data directory
├── results/                   # Output results directory
├── docs/                      # Documentation
└── requirements.txt           # Python dependencies
```

## Quick Start

### Basic Usage

```python
from pipeline import MineralExplorationPipeline

# Initialize pipeline
pipeline = MineralExplorationPipeline(
    image_path='path/to/landsat8_image.tif',
    sensor='landsat8',
    output_dir='results'
)

# Run complete analysis
results = pipeline.run_complete_analysis()

# Visualize results
pipeline.visualize_results(result_type='alterations', save=True)
pipeline.visualize_results(result_type='minerals', save=True)

# Save results as GeoTIFF
pipeline.save_results(format='geotiff')

# Generate report
pipeline.generate_report()
```

### Detecting Specific Alteration Types

```python
from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from alterations.iron_oxide import IronOxideDetector

# Load and preprocess
loader = SatelliteImageLoader('landsat8_image.tif')
image = loader.load_image()

preprocessor = ImagePreprocessor()
image = preprocessor.normalize(image, method='percentile')

# Detect iron oxide
detector = IronOxideDetector(image, sensor='landsat8')

# Get all iron oxide types
results = detector.detect_all_types()
# Returns: general, hematite, goethite, ferric_iron

# Get specific type
hematite = detector.detect_hematite()
```

### Advanced: Porphyry Copper System Detection

```python
from alterations.phyllic import PhyllicDetector
from alterations.propylitic import PropyliticDetector
from alterations.argillic import ArgillicDetector

# Detect core zone (phyllic)
phyllic_detector = PhyllicDetector(image, sensor='landsat8')
phyllic = phyllic_detector.detect_with_pyrite()

# Detect intermediate zone (argillic)
argillic_detector = ArgillicDetector(image, sensor='landsat8')
argillic = argillic_detector.detect_intermediate_argillic()

# Detect outer zone (propylitic)
propylitic_detector = PropyliticDetector(image, sensor='landsat8')
inner_prop = propylitic_detector.detect_inner_propylitic()
outer_prop = propylitic_detector.detect_outer_propylitic()
```

## Examples

The `examples/` directory contains detailed example scripts:

### Satellite Image Download Examples
1. **download_landsat8.py** - Download Landsat-8 imagery for multiple locations
2. **download_aster.py** - Download ASTER data optimized for clay minerals
3. **download_sentinel2.py** - Download high-resolution Sentinel-2 imagery
4. **download_and_analyze_porphyry.py** - **Complete workflow**: download → analyze → report

### Analysis Examples
5. **example1_basic_analysis.py** - Complete pipeline demonstration
6. **example2_iron_oxide_detection.py** - Focused iron oxide mapping
7. **example3_clay_alteration.py** - Argillic and phyllic discrimination
8. **example4_porphyry_system.py** - Porphyry copper zonation mapping
9. **example5_mineral_mapping.py** - Detailed hydroxyl and carbonate mineral mapping

Run examples:
```bash
cd examples

# Download imagery first
python download_landsat8.py

# Or run complete workflow (download + analyze)
python download_and_analyze_porphyry.py

# Or analyze existing imagery
python example1_basic_analysis.py
```

## Supported Sensors

### Landsat-8 (OLI/TIRS)
- Band 1: Coastal/Aerosol (0.43-0.45 µm)
- Band 2: Blue (0.45-0.51 µm)
- Band 3: Green (0.53-0.59 µm)
- Band 4: Red (0.64-0.67 µm)
- Band 5: NIR (0.85-0.88 µm)
- Band 6: SWIR1 (1.57-1.65 µm)
- Band 7: SWIR2 (2.11-2.29 µm)

### ASTER
- VNIR: Bands 1-3 (0.52-0.86 µm)
- SWIR: Bands 4-9 (1.6-2.43 µm)
- TIR: Bands 10-14 (8.125-11.65 µm)

### Landsat-7 (ETM+)
Similar to Landsat-8 with slight differences

### Sentinel-2 (MSI)
13 spectral bands from visible to SWIR

## Analysis Techniques Explained

### Band Ratio Analysis

Band ratios enhance spectral features by dividing reflectance values of different bands. Common ratios:

- **Iron Oxide Ratio**: Red/Blue - highlights ferric iron minerals
- **Clay Mineral Ratio**: SWIR1/SWIR2 - highlights Al-OH absorption
- **Ferrous Mineral Ratio**: SWIR1/NIR - highlights Fe²⁺ minerals
- **Carbonate Ratio**: SWIR2/SWIR1 - highlights CO₃ absorption

### Principal Component Analysis (PCA)

PCA transforms correlated spectral bands into uncorrelated principal components, enhancing subtle variations and reducing noise. Useful for:
- Crosta Technique (Feature-Oriented PCA)
- Decorrelation stretch
- Noise reduction
- Mineral-specific PCA

### Spectral Indices

Normalized indices that combine multiple bands:
- **NDVI**: Vegetation index for masking
- **NDWI**: Water body detection
- **IOI**: Iron oxide index
- **CAI**: Clay alteration index
- **And many more...**

## Geological Context

### Hydrothermal Alteration Zones

Hydrothermal alteration occurs when hot fluids interact with rocks, creating distinct mineral assemblages:

1. **Core Zone (Phyllic/Potassic)**
   - Highest temperature
   - Sericite, muscovite, biotite
   - Associated with ore mineralization

2. **Intermediate Zone (Argillic)**
   - Moderate temperature
   - Kaolinite, dickite
   - Acidic conditions

3. **Outer Zone (Propylitic)**
   - Lower temperature
   - Chlorite, epidote, carbonate
   - Most extensive zone

4. **Surface Zone (Iron Oxide)**
   - Weathered/oxidized
   - Hematite, goethite, limonite
   - Gossans mark buried deposits

## Output Products

The pipeline generates several output products:

### Raster Outputs (GeoTIFF)
- Individual alteration maps
- Mineral probability maps
- Composite zonation maps
- Principal component images

### Visualizations (PNG)
- False-color composites
- Alteration zone maps
- Comparative plots
- RGB mineral composites

### Reports (TXT)
- Detection statistics
- Coverage percentages
- Zone delineation
- Target prioritization

## Validation and Accuracy

For best results:
1. **Atmospheric Correction**: Use atmospherically corrected imagery
2. **Terrain Correction**: Apply topographic correction in mountainous areas
3. **Validation**: Ground-truth with field data, drill results, or spectral libraries
4. **Integration**: Combine with geological maps, geochemistry, and geophysics

## Limitations

- **Vegetation Cover**: Dense vegetation masks mineral signatures
- **Water Bodies**: Water absorption interferes with detection
- **Atmospheric Effects**: Clouds, haze reduce accuracy
- **Mixed Pixels**: Spatial resolution limits pure mineral detection
- **Weathering**: Surface weathering may not reflect subsurface mineralization

## Contributing

Contributions are welcome! Areas for improvement:
- Additional sensor support
- Machine learning classification
- Spectral unmixing algorithms
- Interactive visualization
- Integration with spectral libraries (USGS, ECOSTRESS)

## References

### Key Papers
1. Sabins, F.F. (1999). Remote sensing for mineral exploration. Ore Geology Reviews.
2. Pour, A.B. & Hashim, M. (2012). The application of ASTER remote sensing data to porphyry copper and epithermal gold deposits. Ore Geology Reviews.
3. Crosta, A.P. & Moore, J.M. (1989). Enhancement of Landsat Thematic Mapper imagery for residual soil mapping in SW Minas Gerais State, Brazil.

### Data Sources
- **Landsat**: https://earthexplorer.usgs.gov/
- **ASTER**: https://search.earthdata.nasa.gov/
- **Sentinel-2**: https://scihub.copernicus.eu/

### Spectral Libraries
- USGS Spectral Library: https://www.usgs.gov/labs/spec-lab
- ECOSTRESS: https://speclib.jpl.nasa.gov/

## License

This project is open source. Please cite appropriately when using in research or commercial applications.

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Contact the development team

## Acknowledgments

This project builds upon decades of remote sensing research in mineral exploration and utilizes open-source libraries including NumPy, scikit-learn, Rasterio, and Matplotlib.

---

**Note**: Replace `'path/to/your/image.tif'` in example scripts with actual paths to your satellite imagery files.
