"""
Main Processing Pipeline
Orchestrates mineral detection workflow
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.image_loader import SatelliteImageLoader
from utils.preprocessing import ImagePreprocessor
from utils.visualization import ResultVisualizer
from analysis.band_ratio import BandRatioAnalysis
from analysis.pca_analysis import PCAAnalysis
from analysis.spectral_indices import SpectralIndices
from alterations.iron_oxide import IronOxideDetector
from alterations.argillic import ArgillicDetector
from alterations.phyllic import PhyllicDetector
from alterations.propylitic import PropyliticDetector
from minerals.hydroxyl_minerals import HydroxylMineralDetector
from minerals.carbonate_minerals import CarbonateMineralDetector


class MineralExplorationPipeline:
    """
    Complete pipeline for mineral exploration using satellite imagery
    """

    def __init__(self, image_path: str, sensor: str = 'landsat8', output_dir: str = 'results'):
        """
        Initialize the pipeline

        Args:
            image_path: Path to satellite image
            sensor: Sensor type ('landsat8', 'landsat7', 'aster', 'sentinel2')
            output_dir: Directory for output results
        """
        self.image_path = Path(image_path)
        self.sensor = sensor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = SatelliteImageLoader(str(self.image_path))
        self.preprocessor = ImagePreprocessor()
        self.visualizer = ResultVisualizer()

        # Image data
        self.image = None
        self.preprocessed_image = None

        # Results storage
        self.results = {
            'alterations': {},
            'minerals': {},
            'analysis': {}
        }

    def load_data(self) -> np.ndarray:
        """
        Load satellite imagery

        Returns:
            Loaded image array
        """
        print(f"Loading image from {self.image_path}...")
        self.image = self.loader.load_image()
        print(f"Image loaded: {self.image.shape}")
        return self.image

    def preprocess(self, normalize: bool = True,
                  remove_nodata: bool = True,
                  apply_filter: bool = False) -> np.ndarray:
        """
        Preprocess the image

        Args:
            normalize: Apply normalization
            remove_nodata: Remove no-data values
            apply_filter: Apply smoothing filter

        Returns:
            Preprocessed image
        """
        print("Preprocessing image...")

        self.preprocessed_image = self.image.copy()

        if remove_nodata:
            self.preprocessed_image = self.preprocessor.remove_nodata(
                self.preprocessed_image
            )

        if normalize:
            self.preprocessed_image = self.preprocessor.normalize(
                self.preprocessed_image, method='percentile'
            )

        if apply_filter:
            self.preprocessed_image = self.preprocessor.apply_gaussian_filter(
                self.preprocessed_image, sigma=1.0
            )

        print("Preprocessing complete.")
        return self.preprocessed_image

    def run_band_ratio_analysis(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run band ratio analysis

        Returns:
            Dictionary of band ratio results
        """
        print("\n=== Running Band Ratio Analysis ===")

        analyzer = BandRatioAnalysis(self.preprocessed_image, self.sensor)
        results = analyzer.get_all_ratios()

        self.results['analysis']['band_ratios'] = results
        print(f"Computed ratios for: {list(results.keys())}")

        return results

    def run_pca_analysis(self, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PCA analysis

        Args:
            n_components: Number of principal components

        Returns:
            Tuple of (components, explained variance)
        """
        print("\n=== Running PCA Analysis ===")

        analyzer = PCAAnalysis(self.preprocessed_image)
        components, variance = analyzer.fit_pca(n_components=n_components)

        self.results['analysis']['pca'] = {
            'components': components,
            'explained_variance': variance
        }

        print(f"Computed {n_components} principal components")
        print(f"Explained variance: {variance}")

        return components, variance

    def run_spectral_indices(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate spectral indices

        Returns:
            Dictionary of spectral indices
        """
        print("\n=== Calculating Spectral Indices ===")

        analyzer = SpectralIndices(self.preprocessed_image, self.sensor)
        results = analyzer.calculate_all_indices()

        self.results['analysis']['spectral_indices'] = results
        print(f"Computed indices for: {list(results.keys())}")

        return results

    def detect_iron_oxide_alteration(self) -> Dict[str, np.ndarray]:
        """
        Detect iron oxide alterations

        Returns:
            Dictionary of iron oxide detection results
        """
        print("\n=== Detecting Iron Oxide Alterations ===")

        detector = IronOxideDetector(self.preprocessed_image, self.sensor)
        results = detector.detect_all_types()

        self.results['alterations']['iron_oxide'] = results
        print(f"Detected iron oxide types: {list(results.keys())}")

        return results

    def detect_argillic_alteration(self) -> Dict[str, np.ndarray]:
        """
        Detect argillic alterations

        Returns:
            Dictionary of argillic detection results
        """
        print("\n=== Detecting Argillic Alterations ===")

        detector = ArgillicDetector(self.preprocessed_image, self.sensor)
        results = detector.detect_all_types()

        self.results['alterations']['argillic'] = results
        print(f"Detected argillic types: {list(results.keys())}")

        return results

    def detect_phyllic_alteration(self) -> Dict[str, np.ndarray]:
        """
        Detect phyllic alterations

        Returns:
            Dictionary of phyllic detection results
        """
        print("\n=== Detecting Phyllic Alterations ===")

        detector = PhyllicDetector(self.preprocessed_image, self.sensor)
        results = detector.detect_all_types()

        self.results['alterations']['phyllic'] = results
        print(f"Detected phyllic types: {list(results.keys())}")

        return results

    def detect_propylitic_alteration(self) -> Dict[str, np.ndarray]:
        """
        Detect propylitic alterations

        Returns:
            Dictionary of propylitic detection results
        """
        print("\n=== Detecting Propylitic Alterations ===")

        detector = PropyliticDetector(self.preprocessed_image, self.sensor)
        results = detector.detect_all_types()

        self.results['alterations']['propylitic'] = results
        print(f"Detected propylitic types: {list(results.keys())}")

        return results

    def detect_hydroxyl_minerals(self) -> Dict[str, np.ndarray]:
        """
        Detect hydroxyl-bearing minerals

        Returns:
            Dictionary of hydroxyl mineral detection results
        """
        print("\n=== Detecting Hydroxyl Minerals ===")

        detector = HydroxylMineralDetector(self.preprocessed_image, self.sensor)
        results = detector.detect_all_types()

        self.results['minerals']['hydroxyl'] = results
        print(f"Detected OH mineral types: {list(results.keys())}")

        return results

    def detect_carbonate_minerals(self) -> Dict[str, np.ndarray]:
        """
        Detect carbonate minerals

        Returns:
            Dictionary of carbonate mineral detection results
        """
        print("\n=== Detecting Carbonate Minerals ===")

        detector = CarbonateMineralDetector(self.preprocessed_image, self.sensor)
        results = detector.detect_all_types()

        self.results['minerals']['carbonate'] = results
        print(f"Detected carbonate types: {list(results.keys())}")

        return results

    def detect_all_alterations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Detect all alteration types

        Returns:
            Dictionary of all alteration results
        """
        alterations = {}
        alterations['iron_oxide'] = self.detect_iron_oxide_alteration()
        alterations['argillic'] = self.detect_argillic_alteration()
        alterations['phyllic'] = self.detect_phyllic_alteration()
        alterations['propylitic'] = self.detect_propylitic_alteration()

        return alterations

    def detect_all_minerals(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Detect all mineral types

        Returns:
            Dictionary of all mineral results
        """
        minerals = {}
        minerals['hydroxyl'] = self.detect_hydroxyl_minerals()
        minerals['carbonate'] = self.detect_carbonate_minerals()

        return minerals

    def run_complete_analysis(self) -> Dict:
        """
        Run complete mineral exploration analysis

        Returns:
            Dictionary of all results
        """
        print("\n" + "="*60)
        print("MINERAL EXPLORATION COMPLETE ANALYSIS")
        print("="*60)

        # Load and preprocess
        self.load_data()
        self.preprocess()

        # Run all analyses
        self.run_band_ratio_analysis()
        self.run_pca_analysis()
        self.run_spectral_indices()

        # Detect alterations
        self.detect_all_alterations()

        # Detect minerals
        self.detect_all_minerals()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        return self.results

    def visualize_results(self, result_type: str = 'alterations',
                         save: bool = True):
        """
        Visualize analysis results

        Args:
            result_type: Type of results to visualize ('alterations', 'minerals', 'analysis')
            save: Whether to save visualizations
        """
        print(f"\nVisualizing {result_type}...")

        if result_type == 'alterations':
            # Visualize main alteration types
            alteration_maps = {
                'Iron Oxide': self.results['alterations']['iron_oxide']['general'],
                'Argillic': self.results['alterations']['argillic']['general'],
                'Phyllic': self.results['alterations']['phyllic']['general'],
                'Propylitic': self.results['alterations']['propylitic']['general']
            }

            save_path = self.output_dir / 'alterations_composite.png' if save else None
            self.visualizer.plot_multiple_alterations(alteration_maps, save_path=save_path)

        elif result_type == 'minerals':
            # Visualize mineral types
            mineral_maps = {
                'Hydroxyl Minerals': self.results['minerals']['hydroxyl']['all_oh'],
                'Carbonate Minerals': self.results['minerals']['carbonate']['all_carbonates']
            }

            save_path = self.output_dir / 'minerals_composite.png' if save else None
            self.visualizer.plot_multiple_alterations(mineral_maps, save_path=save_path)

    def save_results(self, format: str = 'geotiff'):
        """
        Save detection results

        Args:
            format: Output format ('geotiff', 'numpy')
        """
        print(f"\nSaving results in {format} format...")

        if format == 'geotiff':
            # Save main alteration maps
            for alt_type, alt_results in self.results['alterations'].items():
                if 'general' in alt_results:
                    output_path = self.output_dir / f'{alt_type}_alteration.tif'
                    self.loader.save_result(
                        alt_results['general'],
                        str(output_path),
                        str(self.image_path)
                    )

        elif format == 'numpy':
            # Save as numpy arrays
            import pickle
            output_path = self.output_dir / 'results.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(self.results, f)

        print(f"Results saved to {self.output_dir}")

    def generate_report(self) -> str:
        """
        Generate text report of findings

        Returns:
            Report string
        """
        report = []
        report.append("="*60)
        report.append("MINERAL EXPLORATION ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"\nImage: {self.image_path}")
        report.append(f"Sensor: {self.sensor}")
        report.append(f"Image Shape: {self.image.shape}")

        report.append("\n" + "-"*60)
        report.append("DETECTED ALTERATIONS")
        report.append("-"*60)

        for alt_type in self.results['alterations'].keys():
            report.append(f"\n{alt_type.upper()}:")
            for subtype in self.results['alterations'][alt_type].keys():
                data = self.results['alterations'][alt_type][subtype]
                if isinstance(data, np.ndarray):
                    coverage = (data > 0.7).sum() / data.size * 100
                    report.append(f"  - {subtype}: {coverage:.2f}% coverage (>0.7 threshold)")

        report.append("\n" + "-"*60)
        report.append("DETECTED MINERALS")
        report.append("-"*60)

        for min_type in self.results['minerals'].keys():
            report.append(f"\n{min_type.upper()}:")
            for subtype in self.results['minerals'][min_type].keys():
                data = self.results['minerals'][min_type][subtype]
                if isinstance(data, np.ndarray):
                    coverage = (data > 0.7).sum() / data.size * 100
                    report.append(f"  - {subtype}: {coverage:.2f}% coverage (>0.7 threshold)")

        report.append("\n" + "="*60)

        report_text = "\n".join(report)
        print(report_text)

        # Save report
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text
