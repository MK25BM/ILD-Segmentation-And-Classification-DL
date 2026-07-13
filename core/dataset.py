"""Dataset loader for ILD HRCT images (manifest-aware, multi-session)."""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
import pandas as pd
import glob
import re
from PIL import Image

from .config import SCRATCH_DIR, NUM_CLASSES, CLASS_MAPPING


class ILDDataset(Dataset):
    """
    Manifest-aware, multi-session patient-based dataset loader for HUG ILD HRCT images.
    
    Uses dataset manifest CSV to:
    - Include only Core_Set patients (aligned data)
    - Map session IDs to patient folders
    - Validate slice alignment (image, lung_mask, roi_mask)
    
    Handles:
    - Numeric patient IDs (e.g., "101")
    - Non-numeric IDs (e.g., "pilot_200")
    - Multi-session patients (e.g., "8_CT-INSPIRIUM-8871", "8_CT-INSPIRIUM-8873")
    - Both .npy and .png files
    """
    
    # Standard size for all images (resize to this)
    STANDARD_SIZE = 512  # ← All images resized to 512x512
    
    def __init__(
        self,
        data_dir: str = SCRATCH_DIR,
        manifest_path: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        transform=None,
        return_metadata: bool = False,
        prefer_npy: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize ILD dataset with manifest validation.
        
        Args:
            data_dir: Path to processed dataset root
            manifest_path: Path to manifest CSV. If None, uses default path.
            session_ids: List of session IDs to load (from manifest Unique_Session_ID).
                        If None, loads all Core_Set sessions.
            transform: Optional torchvision transforms
            return_metadata: If True, return patient_id and slice_id in __getitem__
            prefer_npy: If True, prefer .npy over .png when both exist
            verbose: Print debug info
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_metadata = return_metadata
        self.prefer_npy = prefer_npy
        self.verbose = verbose
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Load manifest
        if manifest_path is None:
            manifest_path = Path(__file__).parent.parent / "archive" / "dataset_manifest.csv"
        
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                f"Run: python archive/generate_manifest.py"
            )
        
        self.manifest = pd.read_csv(manifest_path)
        self._vprint(f"Loaded manifest: {len(self.manifest)} sessions")
        
        # Filter to Core_Set only
        core_set = self.manifest[self.manifest["Cohort"] == "Core_Set"].copy()
        self._vprint(f"Core_Set sessions: {len(core_set)}")
        
        # Filter by session_ids if provided
        if session_ids is not None:
            core_set = core_set[core_set["Unique_Session_ID"].isin(session_ids)]
            self._vprint(f"After filtering by session_ids: {len(core_set)}")
        
        # Map session IDs to patient folders
        self.session_to_folder = {}
        for session_id in core_set["Unique_Session_ID"]:
            folder = self._find_patient_folder(session_id)
            if folder is not None:
                self.session_to_folder[session_id] = folder
        
        self._vprint(f"Found {len(self.session_to_folder)} valid patient folders")
        
        # Build slice index: validate alignment
        self.samples = []  # List of (session_id, slice_id)
        
        for session_id, patient_folder in self.session_to_folder.items():
            valid_slices = self._get_aligned_slices(patient_folder)
            
            if not valid_slices:
                self._vprint(f"⚠️  No aligned slices in {session_id}")
                continue
            
            for slice_id in valid_slices:
                self.samples.append((session_id, slice_id))
        
        if not self.samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")
        
        self._vprint(
            f"✓ ILDDataset initialized: {len(self.session_to_folder)} sessions, "
            f"{len(self.samples)} total aligned slices"
        )
    
    def _vprint(self, msg: str):
        """Verbose print."""
        if self.verbose:
            print(msg)
    
    def _find_patient_folder(self, session_id: str) -> Optional[Path]:
        """Find patient folder matching session ID.
        
        Examples:
        - session_id="101" → patient_101/
        - session_id="8_CT-INSPIRIUM-8871" → patient_8_CT-INSPIRIUM-8871/
        - session_id="pilot_200" → patient_pilot_200/
        """
        expected_folder = self.data_dir / f"patient_{session_id}"
        if expected_folder.exists():
            return expected_folder
        
        # Fallback: search for matching folder (case-insensitive)
        for folder in self.data_dir.iterdir():
            if folder.is_dir() and folder.name.lower() == f"patient_{session_id}".lower():
                return folder
        
        return None
    
    def _get_aligned_slices(self, patient_folder: Path) -> List[str]:
        """Get list of slices that exist in all 3 subfolders (image, lung_mask, roi_mask).
        
        Args:
            patient_folder: Path to patient folder (e.g., patient_101/)
        
        Returns:
            Sorted list of aligned slice IDs (e.g., ["slice_1", "slice_2", ...])
        """
        images_dir = patient_folder / "images"
        lung_dir = patient_folder / "lung_masks"
        roi_dir = patient_folder / "roi_masks"
        
        # Validate all directories exist
        if not all([d.exists() for d in [images_dir, lung_dir, roi_dir]]):
            return []
        
        # Get slice IDs from each directory
        def get_slice_ids(directory: Path) -> set:
            """Extract slice IDs from directory (e.g., slice_1, slice_2, ...)."""
            slice_ids = set()
            for f in directory.glob("slice_*.*"):
                # Extract slice_N from slice_N.npy or slice_N.png
                stem = f.stem
                if stem.startswith("slice_"):
                    slice_ids.add(stem)
            return slice_ids
        
        image_slices = get_slice_ids(images_dir)
        lung_slices = get_slice_ids(lung_dir)
        roi_slices = get_slice_ids(roi_dir)
        
        # Find intersection (aligned slices)
        aligned = sorted(
            list(image_slices & lung_slices & roi_slices),
            key=lambda x: int(x.split("_")[1])  # Sort by slice number
        )
        
        return aligned
    
    def _load_file(self, filepath: Optional[Path]) -> np.ndarray:
        """Load .npy or .png file."""
        if filepath is None:
            return None
        
        try:
            if filepath.suffix == ".npy":
                return np.load(filepath)
            elif filepath.suffix in [".png", ".jpg"]:
                from PIL import Image
                return np.array(Image.open(filepath))
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {e}")
    
    def _get_file_path(
        self,
        directory: Path,
        slice_id: str
    ) -> Optional[Path]:
        """Get file path for slice, preferring .npy over .png."""
        npy_file = directory / f"{slice_id}.npy"
        png_file = directory / f"{slice_id}.png"
        
        if self.prefer_npy:
            if npy_file.exists():
                return npy_file
            elif png_file.exists():
                return png_file
        else:
            if png_file.exists():
                return png_file
            elif npy_file.exists():
                return npy_file
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
        """
        Load image, mask, and lung mask with resize tracking.
        
        Returns:
            (image, mask, lung) tensors if return_metadata=False
            dict with all metadata if return_metadata=True
        """
        session_id, slice_id = self.samples[idx]
        patient_folder = self.session_to_folder[session_id]
        
        # Get file paths
        image_file = self._get_file_path(patient_folder / "images", slice_id)
        mask_file = self._get_file_path(patient_folder / "roi_masks", slice_id)
        lung_file = self._get_file_path(patient_folder / "lung_masks", slice_id)
        
        if image_file is None or mask_file is None or lung_file is None:
            raise FileNotFoundError(f"Missing files for {session_id}/{slice_id}")
        
        # Load arrays
        try:
            image = self._load_file(image_file).astype(np.float32)
            mask = self._load_file(mask_file).astype(np.int64)
            lung = self._load_file(lung_file).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {session_id}/{slice_id}: {e}")
        
        # Store original shape before resize
        original_shape = image.shape  # (H, W)
        
        # Remap classes 7-17 → 7
        mask[mask > 6] = 7
        
        # Validate class range
        unique_classes = np.unique(mask)
        if np.any(unique_classes >= NUM_CLASSES):
            raise ValueError(
                f"Mask contains invalid classes {unique_classes.tolist()} "
                f"(expected [0, {NUM_CLASSES-1}])"
            )
        
        # Validate 2D
        if image.ndim != 2 or mask.ndim != 2 or lung.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays but got: "
                f"image={image.shape}, mask={mask.shape}, lung={lung.shape}"
            )
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # (1, H, W)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()      # (1, H, W)
        mask_tensor = torch.from_numpy(mask).long()         # (H, W)
        lung_tensor = torch.from_numpy(lung).float()        # (H, W)
        
        # ← RESIZE TRACKING: Check if resize is needed
        was_resized = False
        if image_tensor.shape[1:] != (self.STANDARD_SIZE, self.STANDARD_SIZE):
            was_resized = True
            
            # Resize image (BILINEAR for smooth interpolation)
            image_tensor = resize(
                image_tensor,
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            
            # Resize mask (NEAREST to preserve class labels)
            mask_tensor = resize(
                mask_tensor.unsqueeze(0).float(),
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.NEAREST,
            ).squeeze(0).long()
            
            # Resize lung mask (NEAREST to preserve binary values)
            lung_tensor = resize(
                lung_tensor.unsqueeze(0),
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.NEAREST,
            ).squeeze(0)
        
        # Apply transforms if provided
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        if self.return_metadata:
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "lung": lung_tensor,
                "session_id": session_id,
                "slice_id": slice_id,
                "original_shape": original_shape,      # ← TRACK ORIGINAL SIZE
                "was_resized": was_resized,            # ← TRACK IF RESIZED
            }
        else:
            return image_tensor, mask_tensor, lung_tensor
    


class ILDDatasetSplit(ILDDataset):
    """ILD Dataset with train/val/test splitting by session."""
    
    def __init__(
        self,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        empty_mask_strategy: str = "skip",  # ← NEW PARAMETER
        verbose: bool = False,
        return_metadata: bool = False,
        **kwargs
    ):
        """
        Initialize with automatic train/val/test split by session.
        
        Args:
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of sessions for training
            val_ratio: Fraction of sessions for validation
            test_ratio: Fraction of sessions for testing
            seed: Random seed for reproducible splitting
            empty_mask_strategy: 'skip' | 'keep' | 'weighted'
                - 'skip': Remove slices with only class 0 (RECOMMENDED)
                - 'keep': Include all slices
                - 'weighted': Include but downweight
            verbose: Print debug info
            return_metadata: If True, return patient_id and slice_id in __getitem__
            **kwargs: Passed to parent ILDDataset
        """
        # Get all Core_Set sessions from manifest
        manifest_path = kwargs.get(
            "manifest_path",
            Path(__file__).parent.parent / "archive" / "dataset_manifest.csv"
        )
        manifest = pd.read_csv(manifest_path)
        core_set = manifest[manifest["Cohort"] == "Core_Set"].copy()
        all_sessions = sorted(core_set["Unique_Session_ID"].tolist())
        
        # Deterministic split
        rng = np.random.RandomState(seed)
        rng.shuffle(all_sessions)
        
        n_total = len(all_sessions)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_sessions = all_sessions[:n_train]
        val_sessions = all_sessions[n_train : n_train + n_val]
        test_sessions = all_sessions[n_train + n_val :]
        
        split_map = {
            "train": train_sessions,
            "val": val_sessions,
            "test": test_sessions,
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_map.keys())}")
        
        selected_sessions = split_map[split]
        
        # Initialize parent with selected sessions
        super().__init__(
            session_ids=selected_sessions,
            verbose=verbose,
            return_metadata=return_metadata,
            **kwargs
        )
        
        # Apply empty mask strategy AFTER parent init
        self.empty_mask_strategy = empty_mask_strategy
        if empty_mask_strategy == "skip":
            original_count = len (self.samples)
            self.samples = self._filter_empty_masks()
            skipped = original_count - len(self.samples)
            print(f"  Empty mask strategy: {empty_mask_strategy} → skipped {skipped} slices")
        elif empty_mask_strategy == "keep":
            print(f"  Empty mask strategy: {empty_mask_strategy} → keeping all slices")
        else:
            raise ValueError(f"Unknown strategy: {empty_mask_strategy}")
        
        self.split = split
        print(
            f"✓ ILDDatasetSplit: {split} → {len(selected_sessions)} sessions, "
            f"{len(self.samples)} slices "
            f"(train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)})"
        )
    
    def _filter_empty_masks(self) -> List[Tuple[str, str]]:
        """
        Remove slices where mask contains only class 0 (no foreground).
        
        Returns:
            Filtered list of (session_id, slice_id) tuples
        """
        filtered = []
        
        for session_id, slice_id in self.samples:
            patient_folder = self.session_to_folder[session_id]
            mask_file = self._get_file_path(patient_folder / "roi_masks", slice_id)
            
            if mask_file is None:
                continue
            
            # Load mask
            mask = self._load_file(mask_file).astype(np.int64)
            
            # Check if any foreground class (1-7) present
            if np.any(mask > 0):
                filtered.append((session_id, slice_id))
        
        return filtered
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
        """
        Load image, mask, and lung mask.
        
        Returns:
            (image, mask, lung) tensors if return_metadata=False
            dict with all metadata if return_metadata=True
        """
        session_id, slice_id = self.samples[idx]
        patient_folder = self.session_to_folder[session_id]
        
        # Get file paths
        image_file = self._get_file_path(patient_folder / "images", slice_id)
        mask_file = self._get_file_path(patient_folder / "roi_masks", slice_id)
        lung_file = self._get_file_path(patient_folder / "lung_masks", slice_id)
        
        if image_file is None or mask_file is None or lung_file is None:
            raise FileNotFoundError(f"Missing files for {session_id}/{slice_id}")
        
        # Load arrays
        try:
            image = self._load_file(image_file).astype(np.float32)
            mask = self._load_file(mask_file).astype(np.int64)
            lung = self._load_file(lung_file).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {session_id}/{slice_id}: {e}")
        
        # Remap classes 7-17 → 7
        mask[mask > 6] = 7
        
        # Validate class range
        unique_classes = np.unique(mask)
        if np.any(unique_classes >= NUM_CLASSES):
            raise ValueError(
                f"Mask contains invalid classes {unique_classes.tolist()} "
                f"(expected [0, {NUM_CLASSES-1}])"
            )
        
        # Validate 2D
        if image.ndim != 2 or mask.ndim != 2 or lung.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays but got: "
                f"image={image.shape}, mask={mask.shape}, lung={lung.shape}"
            )
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()      # (1, H, W)
        mask_tensor = torch.from_numpy(mask).long()         # (H, W)
        lung_tensor = torch.from_numpy(lung).float()        # (H, W)
        
        # Resize to standard size
        if image_tensor.shape[1:] != (self.STANDARD_SIZE, self.STANDARD_SIZE):
            image_tensor = resize(
                image_tensor,
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            
            mask_tensor = resize(
                mask_tensor.unsqueeze(0).float(),
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.NEAREST,
            ).squeeze(0).long()
            
            lung_tensor = resize(
                lung_tensor.unsqueeze(0),
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.NEAREST,
            ).squeeze(0)
        
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        if self.return_metadata:
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "lung": lung_tensor,
                "session_id": session_id,
                "slice_id": slice_id,
            }
        else:
            return image_tensor, mask_tensor, lung_tensor  # ← RETURN 3 tensors


class ILDDatasetSplit(ILDDataset):
    """ILD Dataset with train/val/test splitting by session."""
    
    def __init__(
        self,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        empty_mask_strategy: str = "skip",  # ← NEW PARAMETER
        verbose: bool = False,
        return_metadata: bool = False,
        **kwargs
    ):
        """
        Initialize with automatic train/val/test split by session.
        
        Args:
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of sessions for training
            val_ratio: Fraction of sessions for validation
            test_ratio: Fraction of sessions for testing
            seed: Random seed for reproducible splitting
            empty_mask_strategy: 'skip' | 'keep' | 'weighted'
                - 'skip': Remove slices with only class 0 (RECOMMENDED)
                - 'keep': Include all slices
                - 'weighted': Include but downweight
            verbose: Print debug info
            return_metadata: If True, return patient_id and slice_id in __getitem__
            **kwargs: Passed to parent ILDDataset
        """
        # Get all Core_Set sessions from manifest
        manifest_path = kwargs.get(
            "manifest_path",
            Path(__file__).parent.parent / "archive" / "dataset_manifest.csv"
        )
        manifest = pd.read_csv(manifest_path)
        core_set = manifest[manifest["Cohort"] == "Core_Set"].copy()
        all_sessions = sorted(core_set["Unique_Session_ID"].tolist())
        
        # Deterministic split
        rng = np.random.RandomState(seed)
        rng.shuffle(all_sessions)
        
        n_total = len(all_sessions)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_sessions = all_sessions[:n_train]
        val_sessions = all_sessions[n_train : n_train + n_val]
        test_sessions = all_sessions[n_train + n_val :]
        
        split_map = {
            "train": train_sessions,
            "val": val_sessions,
            "test": test_sessions,
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_map.keys())}")
        
        selected_sessions = split_map[split]
        
        # Initialize parent with selected sessions
        super().__init__(
            session_ids=selected_sessions,
            verbose=verbose,
            return_metadata=return_metadata,
            **kwargs
        )
        
        # Apply empty mask strategy AFTER parent init
        self.empty_mask_strategy = empty_mask_strategy
        if empty_mask_strategy == "skip":
            original_count = len (self.samples)
            self.samples = self._filter_empty_masks()
            skipped = original_count - len(self.samples)
            print(f"  Empty mask strategy: {empty_mask_strategy} → skipped {skipped} slices")
        elif empty_mask_strategy == "keep":
            print(f"  Empty mask strategy: {empty_mask_strategy} → keeping all slices")
        else:
            raise ValueError(f"Unknown strategy: {empty_mask_strategy}")
        
        self.split = split
        print(
            f"✓ ILDDatasetSplit: {split} → {len(selected_sessions)} sessions, "
            f"{len(self.samples)} slices "
            f"(train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)})"
        )
    
    def _filter_empty_masks(self) -> List[Tuple[str, str]]:
        """
        Remove slices where mask contains only class 0 (no foreground).
        
        Returns:
            Filtered list of (session_id, slice_id) tuples
        """
        filtered = []
        
        for session_id, slice_id in self.samples:
            patient_folder = self.session_to_folder[session_id]
            mask_file = self._get_file_path(patient_folder / "roi_masks", slice_id)
            
            if mask_file is None:
                continue
            
            # Load mask
            mask = self._load_file(mask_file).astype(np.int64)
            
            # Check if any foreground class (1-7) present
            if np.any(mask > 0):
                filtered.append((session_id, slice_id))
        
        return filtered
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
        """
        Load image, mask, and lung mask.
        
        Returns:
            (image, mask, lung) tensors if return_metadata=False
            dict with all metadata if return_metadata=True
        """
        session_id, slice_id = self.samples[idx]
        patient_folder = self.session_to_folder[session_id]
        
        # Get file paths
        image_file = self._get_file_path(patient_folder / "images", slice_id)
        mask_file = self._get_file_path(patient_folder / "roi_masks", slice_id)
        lung_file = self._get_file_path(patient_folder / "lung_masks", slice_id)
        
        if image_file is None or mask_file is None or lung_file is None:
            raise FileNotFoundError(f"Missing files for {session_id}/{slice_id}")
        
        # Load arrays
        try:
            image = self._load_file(image_file).astype(np.float32)
            mask = self._load_file(mask_file).astype(np.int64)
            lung = self._load_file(lung_file).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {session_id}/{slice_id}: {e}")
        
        # Remap classes 7-17 → 7
        mask[mask > 6] = 7
        
        # Validate class range
        unique_classes = np.unique(mask)
        if np.any(unique_classes >= NUM_CLASSES):
            raise ValueError(
                f"Mask contains invalid classes {unique_classes.tolist()} "
                f"(expected [0, {NUM_CLASSES-1}])"
            )
        
        # Validate 2D
        if image.ndim != 2 or mask.ndim != 2 or lung.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays but got: "
                f"image={image.shape}, mask={mask.shape}, lung={lung.shape}"
            )
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()      # (1, H, W)
        mask_tensor = torch.from_numpy(mask).long()         # (H, W)
        lung_tensor = torch.from_numpy(lung).float()        # (H, W)
        
        # Resize to standard size
        if image_tensor.shape[1:] != (self.STANDARD_SIZE, self.STANDARD_SIZE):
            image_tensor = resize(
                image_tensor,
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            
            mask_tensor = resize(
                mask_tensor.unsqueeze(0).float(),
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.NEAREST,
            ).squeeze(0).long()
            
            lung_tensor = resize(
                lung_tensor.unsqueeze(0),
                [self.STANDARD_SIZE, self.STANDARD_SIZE],
                interpolation=transforms.InterpolationMode.NEAREST,
            ).squeeze(0)
        
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        if self.return_metadata:
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "lung": lung_tensor,
                "session_id": session_id,
                "slice_id": slice_id,
            }
        else:
            return image_tensor, mask_tensor, lung_tensor  # ← RETURN 3 tensors
