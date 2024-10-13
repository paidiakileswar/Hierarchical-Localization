"""
Script for Generating Image Pairs Based on Geo Data

Author(s):
Kishore Prasad
Paidi Akileswar

Usage:
python3 gen_pairs.py --geo_file <path_to_geo_file> --output_file <path_to_output_pairs_file> --num_matched <number_of_nearest_neighbors>
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def read_geo_positions(file_path):
    """
    Read geo data from a file and return it as a DataFrame.

    Parameters:
    - file_path: Path to the geo.txt file.

    Returns:
    - DataFrame containing the geo data with columns: 'filename', 'x', 'y', 'z'.
    """
    data = pd.read_csv(file_path, delimiter='\t', skiprows=1, header=None)
    data.columns = ['filename', 'x', 'y', 'z']
    print(data)
    return data


def generate_pairs(data, num_matched):
    """
    Generate image pairs based on nearest neighbors in 3D space.

    Parameters:
    - data: DataFrame containing geo data.
    - num_matched: Number of nearest neighbors to consider for each image.

    Returns:
    - List of tuples, each containing a pair of image filenames.
    """
    coords = data[['x', 'y', 'z']].values
    filenames = data['filename'].values
    dist_matrix = distance_matrix(coords, coords)

    pairs = []
    for i, filename in enumerate(filenames):
        nearest_indices = np.argsort(dist_matrix[i])[1:num_matched + 1]
        for j in nearest_indices:
            pairs.append((filename, filenames[j]))

    return pairs


def save_pairs(pairs, output_file):
    """
    Save the generated pairs to a file.

    Parameters:
    - pairs: List of tuples containing image pairs.
    - output_file: Path to the output file where pairs will be saved.
    """
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")


def main(geo_file, output_file, num_matched):
    """
    Main function to generate and save image pairs based on geo data.

    Parameters:
    - geo_file: Path to the geo.txt file.
    - output_file: Path to the output pairs file.
    - num_matched: Number of nearest neighbors to consider for each image.
    """
    geo_data = read_geo_positions(geo_file)
    pairs = generate_pairs(geo_data, num_matched)
    save_pairs(pairs, output_file)
    print(f"Generated {len(pairs)} pairs and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image pairs based on geo data.")
    parser.add_argument(
        '--geo_file',
        type=Path,
        help='Path to the geo.txt file')
    parser.add_argument(
        '--output_file',
        type=Path,
        help='Path to the output pairs-sfm.txt file')
    parser.add_argument(
        '--num_matched',
        type=int,
        help='Number of nearest neighbors to consider for each image')
    args = parser.parse_args()
    args.geo_file, args.output_file, args.num_matched = "./geo.txt" , "./pairs.txt" , 5
    main(args.geo_file, args.output_file, args.num_matched)