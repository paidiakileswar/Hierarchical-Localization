"""
Script for Feature Extraction and Matching with SuperPoint and SuperGlue, followed by SfM Reconstruction with PixSfM

Author(s):
Kishore Prasad
Paidi Akileswar

Usage:
python3 run_sp_sg.py <args>
"""

import sys
from pathlib import Path
import os
import time
import subprocess
import logging

from hloc import extract_features, match_features

def setup_logger():
    # Set up logger
    logger = logging.getLogger("SuperPoint-SuperGlue")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def sp_sg(args, workdir, TIMEOUT, logger):
    # Record the start time
    start_time = time.time()
    stage = "RUN SFM SUPERPOINT-SUPERGLUE"
    
    logger.info(f"Starting {stage}")

    # Retrieve configuration settings for SuperPoint and SuperGlue
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superglue"]

    # Define paths based on workdir
    outputs = Path(workdir)  # Using workdir instead of root directory
    images = outputs / 'images'
    sfm_pairs = outputs / 'pairs.txt'
    features = outputs / 'export' / 'features.h5'
    matches = outputs / 'export' / 'matches.h5'

    # List all reference images
    references = [p.relative_to(images).as_posix() for p in images.iterdir() if p.is_file()]
    logger.info(f"{len(references)} images found for mapping")
    logger.info("Referencing completed")

    # Extract features using SuperPoint
    extract_features.main(
        feature_conf,
        images,
        image_list=references,
        feature_path=features
    )
    logger.info("Feature Extraction Completed")

    # Match features using SuperGlue
    match_features.main(
        matcher_conf,
        sfm_pairs,
        features=features,
        matches=matches
    )
    logger.info("Feature Matching Completed")

    # Log overall completion of feature extraction and matching
    logger.info("Feature Extraction & Matching Completed")

    # Log time taken for feature extraction and matching
    time_taken = time.time() - start_time
    logger.info(f"Time taken for SuperPoint Feature extraction and matching: {time_taken:.2f} seconds")


if __name__ == "__main__":
    workdir = "./"
    TIMEOUT = 300  # default timeout

    # Set up logger
    logger = setup_logger()

    # Run the function
    sp_sg(sys.argv, workdir, TIMEOUT, logger)