"""
Inspect mask values across entire dataset to find class 8 sources.

Usage:
    python scripts/inspect_masks.py --data-root /scratch/u6dm/mk25bm.u6dm/ild_dataset_processed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_masks(data_root: str):
    """Scan all mask files for invalid class values."""
    data_root = Path(data_root)
    
    # Find all mask files
    mask_files = list(data_root.glob("**/roi_masks/*.npy"))
    logger.info(f"Found {len(mask_files)} mask files\n")
    
    class_counts = Counter()
    invalid_files = []
    class_8_locations = []
    
    for mask_file in sorted(mask_files):
        mask = np.load(mask_file)
        unique_classes = np.unique(mask)
        
        # Count occurrences
        for cls in unique_classes:
            class_counts[cls] += np.sum(mask == cls)
        
        # Track class 8
        if 8 in unique_classes:
            invalid_files.append(mask_file)
            count_8 = np.sum(mask == 8)
            class_8_locations.append({
                "file": mask_file,
                "count": count_8,
                "total_pixels": mask.size,
                "percentage": 100 * count_8 / mask.size,
                "shape": mask.shape,
                "unique_classes": unique_classes.tolist()
            })
    
    # Print summary
    logger.info("="*70)
    logger.info("CLASS DISTRIBUTION ACROSS ALL MASKS")
    logger.info("="*70 + "\n")
    
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        logger.info(f"Class {cls}: {count:,} pixels")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"INVALID CLASS 8 FOUND IN {len(invalid_files)} FILES")
    logger.info(f"{'='*70}\n")
    
    if class_8_locations:
        for info in sorted(class_8_locations, key=lambda x: x["percentage"], reverse=True)[:10]:
            logger.info(
                f"File: {info['file'].relative_to(data_root)}\n"
                f"  Class 8 pixels: {info['count']} / {info['total_pixels']} ({info['percentage']:.2f}%)\n"
                f"  Shape: {info['shape']}\n"
                f"  Unique classes in file: {info['unique_classes']}\n"
            )
    
    # Check original annotation files (if available)
    logger.info(f"\n{'='*70}")
    logger.info("CHECKING ANNOTATION PIPELINE")
    logger.info(f"{'='*70}\n")
    
    # Look for any JSON/XML annotation files
    annotation_files = list(data_root.glob("**/*.json")) + list(data_root.glob("**/*.xml"))
    logger.info(f"Found {len(annotation_files)} annotation files")
    
    if annotation_files:
        logger.info("\nAnnotation files found:")
        for f in annotation_files[:5]:
            logger.info(f"  - {f.relative_to(data_root)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect mask values")
    parser.add_argument("--data-root", type=str, required=True)
    
    args = parser.parse_args()
    inspect_masks(args.data_root)


if __name__ == "__main__":
    main()