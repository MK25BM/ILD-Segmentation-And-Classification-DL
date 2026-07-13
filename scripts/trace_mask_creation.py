"""
Trace where masks come from in the pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trace_one_file(data_root: str, patient_id: str, slice_id: str):
    """Trace a single mask file through the pipeline."""
    data_root = Path(data_root)
    
    # Find the mask file
    mask_file = data_root / patient_id / "roi_masks" / f"{slice_id}.npy"
    
    if not mask_file.exists():
        logger.error(f"Mask file not found: {mask_file}")
        return
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TRACING MASK: {patient_id} / {slice_id}")
    logger.info(f"{'='*70}\n")
    
    # Load mask
    mask = np.load(mask_file)
    logger.info(f"Loaded mask shape: {mask.shape}")
    logger.info(f"Unique classes: {np.unique(mask).tolist()}")
    logger.info(f"Data type: {mask.dtype}")
    
    # Check for class 8
    if 8 in np.unique(mask):
        logger.warning(f"\n❌ CLASS 8 DETECTED!")
        logger.warning(f"   Pixels with class 8: {np.sum(mask == 8)}")
        logger.warning(f"   Percentage: {100 * np.sum(mask == 8) / mask.size:.2f}%")
        
        # Find spatial location
        locs = np.where(mask == 8)
        logger.warning(f"   Bounding box: rows [{locs[0].min()}, {locs[0].max()}], cols [{locs[1].min()}, {locs[1].max()}]")
    
    # Check for original annotation file
    annotation_candidates = [
        data_root / patient_id / "annotations.json",
        data_root / patient_id / f"{slice_id}.json",
        data_root / patient_id / "labels" / f"{slice_id}.json",
    ]
    
    logger.info(f"\nLooking for source annotations...")
    for candidate in annotation_candidates:
        if candidate.exists():
            logger.info(f"✓ Found: {candidate.relative_to(data_root)}")
            with open(candidate) as f:
                content = f.read()[:200]
                logger.info(f"  Content preview: {content}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patient-id", required=True, help="e.g., 'patient_001'")
    parser.add_argument("--slice-id", required=True, help="e.g., 'slice_00'")
    
    args = parser.parse_args()
    trace_one_file(args.data_root, args.patient_id, args.slice_id)