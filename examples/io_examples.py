"""Examples for using the io module.

This script demonstrates how to use the plottle io module
to load and save data in various formats.

Run this script from the plottle directory:
    python examples/io_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path to import plottle
sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.io import (
    load_pickle, save_pickle,
    load_numpy, save_numpy,
    load_dataframe, save_dataframe,
    load_data, save_data
)

# Use ASCII-safe checkmarks for Windows compatibility
CHECK = '[OK]'
CROSS = '[X]'


def example_pickle_operations():
    """Example 1: Working with pickle files."""
    print("\n" + "="*70)
    print("Example 1: Pickle Operations")
    print("="*70)

    # Create sample data
    experiment_data = {
        'name': 'Thermal Analysis',
        'temperature': 298.15,
        'pressure': 1.013,
        'measurements': [1.2, 1.5, 1.8, 2.1, 2.4],
        'metadata': {
            'date': '2026-02-12',
            'instrument': 'DSC-100'
        }
    }

    # Save to pickle
    output_file = Path('examples/data/experiment.pkl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(experiment_data, output_file)
    print(f"{CHECK} Saved data to {output_file}")

    # Load from pickle
    loaded_data = load_pickle(output_file)
    print(f"{CHECK} Loaded data: {loaded_data['name']}")
    print(f"  Temperature: {loaded_data['temperature']} K")
    print(f"  Measurements: {loaded_data['measurements']}")


def example_numpy_operations():
    """Example 2: Working with NumPy arrays."""
    print("\n" + "="*70)
    print("Example 2: NumPy Array Operations")
    print("="*70)

    # Create sample data
    coordinates = np.random.rand(100, 3)  # 100 points in 3D space
    energies = np.random.rand(100) * 100  # Energy values

    # Save single array
    coords_file = Path('examples/data/coordinates.npy')
    save_numpy(coordinates, coords_file)
    print(f"{CHECK} Saved coordinates to {coords_file}")
    print(f"  Shape: {coordinates.shape}")

    # Save multiple arrays
    arrays_file = Path('examples/data/simulation_data.npz')
    data_dict = {
        'coordinates': coordinates,
        'energies': energies,
        'timestep': np.arange(100)
    }
    save_numpy(data_dict, arrays_file)
    print(f"{CHECK} Saved multiple arrays to {arrays_file}")

    # Load arrays
    loaded_coords = load_numpy(coords_file)
    print(f"{CHECK} Loaded coordinates, shape: {loaded_coords.shape}")

    loaded_dict = load_numpy(arrays_file)
    print(f"{CHECK} Loaded multiple arrays:")
    for key in loaded_dict.files:
        print(f"  - {key}: shape {loaded_dict[key].shape}")


def example_dataframe_operations():
    """Example 3: Working with Pandas DataFrames."""
    print("\n" + "="*70)
    print("Example 3: Pandas DataFrame Operations")
    print("="*70)

    # Create sample DataFrame
    df = pd.DataFrame({
        'Temperature (K)': np.linspace(273, 373, 20),
        'Pressure (bar)': np.random.uniform(0.9, 1.1, 20),
        'Volume (L)': np.random.uniform(22.0, 24.0, 20),
        'Sample': ['A'] * 10 + ['B'] * 10
    })

    # Save to CSV
    csv_file = Path('examples/data/experimental_data.csv')
    save_dataframe(df, csv_file)
    print(f"{CHECK} Saved DataFrame to {csv_file}")
    print(f"  Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    # Save to Excel
    excel_file = Path('examples/data/experimental_data.xlsx')
    save_dataframe(df, excel_file)
    print(f"\n{CHECK} Saved DataFrame to {excel_file}")

    # Load from CSV
    loaded_df = load_dataframe(csv_file)
    print(f"\n{CHECK} Loaded DataFrame from CSV")
    print(f"  Columns: {list(loaded_df.columns)}")
    print(f"  Rows: {len(loaded_df)}")


def example_universal_loader():
    """Example 4: Using the universal load_data/save_data functions."""
    print("\n" + "="*70)
    print("Example 4: Universal Data Loader")
    print("="*70)

    # The universal functions automatically detect format from extension
    print("\nUniversal loader can handle any supported format:")

    # Example with different formats
    formats = {
        'pickle': ({'test': 'data'}, 'examples/data/auto_test.pkl'),
        'numpy': (np.array([1, 2, 3, 4, 5]), 'examples/data/auto_test.npy'),
        'csv': (pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}),
                'examples/data/auto_test.csv')
    }

    for format_name, (data, filepath) in formats.items():
        filepath = Path(filepath)

        # Save using universal saver
        save_data(data, filepath)
        print(f"{CHECK} Saved {format_name} data to {filepath}")

        # Load using universal loader
        loaded = load_data(filepath)
        print("  Loaded back successfully!")

        if format_name == 'numpy':
            assert np.array_equal(data, loaded)
        elif format_name == 'csv':
            assert data.equals(loaded)
        else:
            assert data == loaded


def example_practical_workflow():
    """Example 5: Practical workflow for computational chemistry."""
    print("\n" + "="*70)
    print("Example 5: Practical Workflow - Molecular Dynamics Simulation")
    print("="*70)

    # Simulate a molecular dynamics workflow
    print("\n1. Generating simulation data...")

    # Simulate trajectory data
    n_frames = 50
    n_atoms = 100
    trajectory = {
        'coordinates': np.random.rand(n_frames, n_atoms, 3) * 10,  # Positions
        'velocities': np.random.randn(n_frames, n_atoms, 3) * 0.5,  # Velocities
        'forces': np.random.randn(n_frames, n_atoms, 3) * 2.0,      # Forces
        'time': np.linspace(0, 5000, n_frames)  # Time in ps
    }

    # Save trajectory as compressed npz
    traj_file = Path('examples/data/trajectory.npz')
    save_numpy(trajectory, traj_file, compressed=True)
    print(f"{CHECK} Saved trajectory to {traj_file}")

    # Simulate analysis results
    print("\n2. Analyzing trajectory...")
    energies = np.random.rand(n_frames) * 1000 + 5000  # Total energy
    temperature = np.random.rand(n_frames) * 10 + 295  # Temperature

    results_df = pd.DataFrame({
        'Time (ps)': trajectory['time'],
        'Energy (kJ/mol)': energies,
        'Temperature (K)': temperature,
        'RMSD (Å)': np.random.rand(n_frames) * 2
    })

    # Save analysis results
    results_file = Path('examples/data/md_analysis.csv')
    save_dataframe(results_df, results_file)
    print(f"{CHECK} Saved analysis results to {results_file}")

    # Save summary statistics
    summary = {
        'n_frames': n_frames,
        'n_atoms': n_atoms,
        'avg_energy': float(energies.mean()),
        'avg_temperature': float(temperature.mean()),
        'simulation_time': float(trajectory['time'][-1])
    }

    summary_file = Path('examples/data/md_summary.pkl')
    save_pickle(summary, summary_file)
    print(f"{CHECK} Saved summary to {summary_file}")

    # Load everything back
    print("\n3. Loading results for visualization...")
    loaded_traj = load_data(traj_file)
    loaded_results = load_data(results_file)
    loaded_summary = load_data(summary_file)

    print(f"{CHECK} Loaded trajectory: {len(loaded_traj.files)} arrays")
    print(f"{CHECK} Loaded results: {len(loaded_results)} rows")
    print(f"{CHECK} Loaded summary:")
    for key, value in loaded_summary.items():
        print(f"    {key}: {value}")


def cleanup_examples():
    """Clean up example data files."""
    import shutil
    data_dir = Path('examples/data')
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"\n{CHECK} Cleaned up example data directory")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PLOTTING HELPER - I/O MODULE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates the data I/O capabilities of the")
    print("plottle package for scientific computing.\n")

    try:
        # Run all examples
        example_pickle_operations()
        example_numpy_operations()
        example_dataframe_operations()
        example_universal_loader()
        example_practical_workflow()

        print("\n" + "="*70)
        print("All examples completed successfully! {CHECK}")
        print("="*70)
        print("\nExample data files are saved in: examples/data/")
        print("You can inspect them or use them for testing.")
        print("\nTo clean up example files, run with --cleanup flag")

    except Exception as e:
        print(f"\n{CROSS} Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys

    if '--cleanup' in sys.argv:
        cleanup_examples()
    else:
        main()
