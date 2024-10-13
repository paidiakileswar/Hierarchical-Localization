import h5py

def explore_features_h5(file_path):
    """Function to explore the contents of the features.h5 file."""
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}:")
        print(list(f.keys())[:10])  # List first 10 top-level keys (image groups)

        # Exploring contents of the first image group for illustration
        first_image = list(f.keys())[0]
        print(f"\nExploring contents of image group '{first_image}':")
        image_group = f[first_image]

        # List contents of the group (descriptors, image_size, keypoints, scores)
        for dataset_name in image_group.keys():
            dataset = image_group[dataset_name]
            
            if isinstance(dataset, h5py.Dataset):
                print(f"\nDataset '{dataset_name}':")
                print(f"Shape: {dataset.shape}")
                print(f"Data type: {dataset.dtype}")
                print(f"First few values: {dataset[:5]}")  # Print first few values


def explore_matches_h5(file_path):
    """Function to explore the contents of the matches.h5 file."""
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}:")
        print(list(f.keys())[:10])  # List first 10 top-level keys (image groups)

        # Iterate through each image group
        for image_key in f.keys():
            print(f"\nExploring contents of image group '{image_key}':")
            image_group = f[image_key]

            # Explore matches and matching scores within each image group
            for pair_key in image_group.keys():
                pair_group = image_group[pair_key]
                
                if isinstance(pair_group, h5py.Group):
                    print(f"\n'{pair_key}' is a group. It contains:")
                    for dataset_name in pair_group.keys():
                        dataset = pair_group[dataset_name]
                        if isinstance(dataset, h5py.Dataset):
                            print(f"\nDataset '{dataset_name}':")
                            print(f"Shape: {dataset.shape}")
                            print(f"Data type: {dataset.dtype}")
                            print(f"First few values: {dataset[:5]}")  # Print first few values
                            
                            # Count numbers that are not -1 for matches0 datasets
                            if 'matches' in dataset_name:
                                count_not_minus_one = sum(1 for number in dataset if number != -1)
                                print(f"Number of matches (not -1) in '{dataset_name}': {count_not_minus_one}")


# Paths to your .h5 files
features_path = './export/features.h5'
matches_path = './export/matches.h5'

# Explore the features.h5 file
print("\nExploring features.h5")
explore_features_h5(features_path)

# Explore the matches.h5 file and print the number of matches
print("\nExploring matches.h5")
explore_matches_h5(matches_path)
