"""
Validate extracted 2D slices against HRCT physics & MedGIFT structure.

Checks:
- HU value ranges (−950 to +350 for lung window)
- Lung coverage (15–30% typical)
- ROI class distribution vs voxel imbalance
- Alignment (CT, lung, ROI same shape)
- Sparse annotation structure (98.8% unlabeled)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ExtractionPhysicsValidator:
    """Validate extracted slice physics."""
    
    def __init__(self, data_root: str, sample_size: int = 50):
        self.data_root = Path(data_root)
        self.sample_size = sample_size
        self.results = {
            "ct_hu_validation": [],
            "lung_coverage_validation": [],
            "roi_class_validation": [],
            "alignment_validation": [],
            "anomalies": [],
        }
    
    def sample_slices(self) -> list:
        """Randomly sample N patient folders."""
        patient_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        sample_dirs = np.random.choice(
            patient_dirs, 
            size=min(self.sample_size, len(patient_dirs)), 
            replace=False
        )
        return sorted(sample_dirs)
    
    def validate_ct_hu_range(self, ct_array: np.ndarray) -> Dict:
        """
        Validate CT image HU range.
        
        Expected: [−950, +350] for lung window (MedGIFT standard)
        """
        ct_min = float(np.min(ct_array))
        ct_max = float(np.max(ct_array))
        ct_mean = float(np.mean(ct_array))
        ct_std = float(np.std(ct_array))
        
        # Check if already normalized to [0, 1]
        if ct_max <= 1.0 and ct_min >= 0.0:
            normalized = "float32 [0,1]"
            expected_min, expected_max = 0.0, 1.0
            expected_mean_range = [0.2, 0.6]  # Lung tissue in [0,1] space
        else:
            normalized = "HU values (not normalized)"
            expected_min, expected_max = -950, 350
            expected_mean_range = [-600, -300]
        
        # Validate
        in_range = (ct_min >= expected_min * 0.9) and (ct_max <= expected_max * 1.1)
        in_mean_range = (expected_mean_range[0] <= ct_mean <= expected_mean_range[1])
        
        return {
            "min": ct_min,
            "max": ct_max,
            "mean": ct_mean,
            "std": ct_std,
            "normalization": normalized,
            "in_range": in_range,
            "in_mean_range": in_mean_range,
        }
    
    def validate_lung_coverage(self, lung_array: np.ndarray) -> Dict:
        """
        Validate lung mask coverage.
        
        Expected: 15–30% of image is lung tissue for axial HRCT slices
        """
        lung_binary = (lung_array > 0.5).astype(np.uint8) if lung_array.max() <= 1.0 else (lung_array > 128).astype(np.uint8)
        
        total_pixels = np.prod(lung_array.shape)
        lung_pixels = int(np.sum(lung_binary))
        lung_percentage = 100 * lung_pixels / total_pixels
        
        expected_min, expected_max = 15, 30
        in_range = expected_min <= lung_percentage <= expected_max
        
        return {
            "lung_pixels": lung_pixels,
            "total_pixels": int(total_pixels),
            "lung_percentage": lung_percentage,
            "in_range": in_range,
            "warning": "⚠️ Outside 15–30% range" if not in_range else None,
        }
    
    def validate_roi_class_structure(self, roi_array: np.ndarray, lung_array: np.ndarray) -> Dict:
        """
        Validate ROI mask class structure.
        
        Expected (after merging):
        - Class 0: ~98.8% (background + unlabeled lung)
        - Class 1: ~0.3% (healthy annotated)
        - Classes 2–6: disease pathologies
        - Class 7: rare merged (0.06%)
        """
        roi_values = roi_array.astype(np.int64)
        unique_classes = np.unique(roi_values)
        
        total_pixels = np.prod(roi_array.shape)
        lung_binary = (lung_array > 0.5).astype(np.uint8) if lung_array.max() <= 1.0 else (lung_array > 128).astype(np.uint8)
        lung_pixels = np.sum(lung_binary)
        
        class_dist = {}
        for cls in unique_classes:
            count = int(np.sum(roi_values == cls))
            pct = 100 * count / total_pixels
            class_dist[int(cls)] = {
                "count": count,
                "percentage": pct,
                "of_lung_pixels": 100 * count / lung_pixels if lung_pixels > 0 else 0,
            }
        
        # Check sparse annotation structure
        class_0_pct = class_dist.get(0, {}).get("percentage", 0)
        annotated_pct = 100 - class_0_pct
        sparse_annotation = annotated_pct < 10  # <10% annotated = sparse
        
        anomalies = []
        if class_0_pct < 95:
            anomalies.append(f"Class 0 only {class_0_pct:.1f}% (expected ~98.8%)")
        if not sparse_annotation:
            anomalies.append(f"Annotation too dense: {annotated_pct:.1f}% (expected <10%)")
        
        # Check for invalid classes
        invalid = [c for c in unique_classes if c > 17]
        if invalid:
            anomalies.append(f"Invalid class indices: {invalid}")
        
        return {
            "class_distribution": class_dist,
            "sparse_annotation": sparse_annotation,
            "annotated_percentage": annotated_pct,
            "anomalies": anomalies,
        }
    
    def validate_alignment(
        self,
        ct_shape: Tuple[int, int],
        lung_shape: Tuple[int, int],
        roi_shape: Tuple[int, int],
    ) -> Dict:
        """Validate shape alignment."""
        aligned = (ct_shape == lung_shape == roi_shape)
        return {
            "ct_shape": ct_shape,
            "lung_shape": lung_shape,
            "roi_shape": roi_shape,
            "aligned": aligned,
        }
    
    def validate_patient_folder(self, patient_dir: Path) -> Dict:
        """Validate one patient folder."""
        result = {
            "patient_id": patient_dir.name,
            "ct": None,
            "lung": None,
            "roi": None,
            "alignment": None,
            "errors": [],
        }
        
        try:
            images_dir = patient_dir / "images"
            lung_dir = patient_dir / "lung_masks"
            roi_dir = patient_dir / "roi_masks"
            
            if not all([d.exists() for d in [images_dir, lung_dir, roi_dir]]):
                result["errors"].append("Missing subdirectories")
                return result
            
            # Get first slice
            ct_files = sorted(list(images_dir.glob("slice_*.npy")))
            lung_files = sorted(list(lung_dir.glob("slice_*.npy")))
            roi_files = sorted(list(roi_dir.glob("slice_*.npy")))
            
            if not all([ct_files, lung_files, roi_files]):
                result["errors"].append("Missing slice files")
                return result
            
            # Load first slice
            ct_array = np.load(ct_files[0])
            lung_array = np.load(lung_files[0])
            roi_array = np.load(roi_files[0])
            
            # Validate
            result["ct"] = self.validate_ct_hu_range(ct_array)
            result["lung"] = self.validate_lung_coverage(lung_array)
            result["roi"] = self.validate_roi_class_structure(roi_array, lung_array)
            result["alignment"] = self.validate_alignment(
                ct_array.shape, lung_array.shape, roi_array.shape
            )
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        return result
    
    def run(self) -> Dict:
        """Run validation on sample."""
        logger.info(f"\n{'='*70}")
        logger.info("EXTRACTION PHYSICS VALIDATION")
        logger.info(f"{'='*70}\n")
        
        sample_dirs = self.sample_slices()
        logger.info(f"Sampling {len(sample_dirs)} patient folders...\n")
        
        all_results = []
        for patient_dir in sample_dirs:
            result = self.validate_patient_folder(patient_dir)
            all_results.append(result)
        
        # Aggregate results
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: list):
        """Print validation summary."""
        
        # CT HU validation
        logger.info(f"\n{'─'*70}")
        logger.info("1️⃣  CT IMAGE HU VALIDATION")
        logger.info(f"{'─'*70}")
        
        ct_valid = [r["ct"] for r in results if r["ct"] is not None]
        if ct_valid:
            ct_means = [c["mean"] for c in ct_valid]
            ct_ranges_ok = [c["in_range"] for c in ct_valid]
            
            logger.info(f"Mean intensity: {np.mean(ct_means):.1f} ± {np.std(ct_means):.1f}")
            logger.info(f"Range check: {sum(ct_ranges_ok)}/{len(ct_valid)} slices in expected range")
            logger.info(f"Normalization detected: {ct_valid[0]['normalization']}")
            
            if all(ct_ranges_ok):
                logger.info("✓ CT HU values correct")
            else:
                logger.warning("⚠️  Some CT values outside expected range")
        
        # Lung coverage validation
        logger.info(f"\n{'─'*70}")
        logger.info("2️⃣  LUNG MASK COVERAGE VALIDATION")
        logger.info(f"{'─'*70}")
        
        lung_valid = [r["lung"] for r in results if r["lung"] is not None]
        if lung_valid:
            lung_coverage = [l["lung_percentage"] for l in lung_valid]
            logger.info(f"Mean coverage: {np.mean(lung_coverage):.1f}% ± {np.std(lung_coverage):.1f}%")
            logger.info(f"Range: {np.min(lung_coverage):.1f}% – {np.max(lung_coverage):.1f}%")
            
            in_range = sum(l["in_range"] for l in lung_valid)
            logger.info(f"Within 15–30%: {in_range}/{len(lung_valid)} slices")
            
            if in_range == len(lung_valid):
                logger.info("✓ Lung coverage within normal range")
            else:
                logger.warning(f"⚠️  {len(lung_valid) - in_range} slices outside normal coverage")
        
        # ROI class validation
        logger.info(f"\n{'─'*70}")
        logger.info("3️⃣  ROI CLASS STRUCTURE (SPARSE ANNOTATION)")
        logger.info(f"{'─'*70}")
        
        roi_valid = [r["roi"] for r in results if r["roi"] is not None]
        if roi_valid:
            annotated_pcts = [r["annotated_percentage"] for r in roi_valid]
            logger.info(f"Annotated voxels: {np.mean(annotated_pcts):.2f}% ± {np.std(annotated_pcts):.2f}%")
            
            sparse = sum(r["sparse_annotation"] for r in roi_valid)
            logger.info(f"Sparse annotation: {sparse}/{len(roi_valid)} slices (<10% annotated)")
            
            # Aggregate class distribution
            all_class_dist = {}
            for r in roi_valid:
                for cls, data in r["class_distribution"].items():
                    if cls not in all_class_dist:
                        all_class_dist[cls] = {"count": 0, "percentage": 0}
                    all_class_dist[cls]["count"] += data["count"]
            
            total_voxels = sum(d["count"] for d in all_class_dist.values())
            logger.info("\nClass distribution (aggregated):")
            for cls in sorted(all_class_dist.keys()):
                count = all_class_dist[cls]["count"]
                pct = 100 * count / total_voxels
                logger.info(f"  Class {cls}: {pct:6.2f}%")
            
            if sparse == len(roi_valid):
                logger.info("✓ Sparse ROI structure confirmed")
            else:
                logger.warning(f"⚠️  Dense annotation detected in {len(roi_valid) - sparse} slices")
        
        # Alignment validation
        logger.info(f"\n{'─'*70}")
        logger.info("4️⃣  ALIGNMENT CHECK")
        logger.info(f"{'─'*70}")
        
        align_valid = [r["alignment"] for r in results if r["alignment"] is not None]
        if align_valid:
            aligned = sum(a["aligned"] for a in align_valid)
            logger.info(f"Aligned slices: {aligned}/{len(align_valid)}")
            
            if aligned == len(align_valid):
                logger.info("✓ All slices properly aligned")
            else:
                logger.warning("⚠️  Some slices misaligned")
                for a in align_valid:
                    if not a["aligned"]:
                        logger.warning(f"  {a}")
        
        # Errors
        errors = [e for r in results for e in r["errors"]]
        if errors:
            logger.warning(f"\n⚠️  ERRORS ({len(errors)}):")
            for err in errors[:10]:
                logger.warning(f"  {err}")
        else:
            logger.info("\n✓ No errors detected")
        
        logger.info(f"\n{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate extraction physics")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed",
        help="Path to extracted data",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of patient folders to sample",
    )
    
    args = parser.parse_args()
    
    validator = ExtractionPhysicsValidator(args.data_root, args.sample_size)
    validator.run()


if __name__ == "__main__":
    main()