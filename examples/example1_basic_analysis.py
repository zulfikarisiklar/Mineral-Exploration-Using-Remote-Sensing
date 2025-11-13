"""
Example 1: Basic Mineral Exploration Analysis
Demonstrates basic usage of the pipeline
"""

import sys
sys.path.append('../src')

from pipeline import MineralExplorationPipeline


def main():
    """Run basic analysis"""

    # Initialize pipeline
    # Replace 'path/to/your/image.tif' with actual image path
    pipeline = MineralExplorationPipeline(
        image_path='path/to/your/landsat8_image.tif',
        sensor='landsat8',
        output_dir='../results/basic_analysis'
    )

    # Run complete analysis
    results = pipeline.run_complete_analysis()

    # Visualize results
    pipeline.visualize_results(result_type='alterations', save=True)
    pipeline.visualize_results(result_type='minerals', save=True)

    # Save results
    pipeline.save_results(format='geotiff')

    # Generate report
    pipeline.generate_report()

    print("\nBasic analysis complete!")
    print(f"Results saved to: {pipeline.output_dir}")


if __name__ == '__main__':
    main()
