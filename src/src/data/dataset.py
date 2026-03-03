```python
"""
Dataset class for Alzheimer's Disease progression modeling
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

from .adni_loader import ADNILoader, MRIPreprocessor
from ..config import DataConfig

logger = logging.getLogger(__name__)

class ADProgressionDataset(Dataset):
    """Dataset for Alzheimer's Disease progression modeling"""
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        config: DataConfig,
        split: str = "train",
        transform=None
    ):
        """
        Initialize AD progression dataset
        
        Args:
            data_dir: Directory containing ADNI data
            cache_dir: Cache directory for processed data
            config: Data configuration
            split: Dataset split ("train", "val", "test")
            transform: Data augmentation transforms
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize data loader and preprocessor
        self.adni_loader = ADNILoader(data_dir, cache_dir)
        self.mri_preprocessor = MRIPreprocessor(
            target_shape=config.target_shape,
            voxel_spacing=config.voxel_spacing,
            template_path=config.template_path
        )
        
        # Load dataset
        self.sequences = self._load_sequences()
        
        # Fit scalers for clinical features
        if split == "train":
            self.clinical_scaler = self._fit_clinical_scaler()
        else:
            self.clinical_scaler = self._load_clinical_scaler()
        
        logger.info(f"Loaded {len(self.sequences)} sequences for {split} split")
    
    def _load_sequences(self) -> List[Dict]:
        """Load preprocessed sequences"""
        cache_file = Path(self.cache_dir) / f"{self.split}_sequences.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached sequences from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Create longitudinal dataset
        logger.info("Creating longitudinal sequences...")
        all_sequences = self.adni_loader.create_longitudinal_dataset(
            min_timepoints=self.config.min_sequence_length,
            max_timepoints=self.config.max_sequence_length,
            time_window_months=60
        )
        
        # Filter valid sequences (with MRI files)
        valid_sequences = []
        for seq in all_sequences:
            if self._validate_sequence(seq):
                valid_sequences.append(seq)
        
        # Split dataset
        np.random.seed(self.config.random_seed)
        np.random.shuffle(valid_sequences)
        
        n_total = len(valid_sequences)
        n_train = int(self.config.train_ratio * n_total)
        n_val = int(self.config.val_ratio * n_total)
        
        if self.split == "train":
            split_sequences = valid_sequences[:n_train]
        elif self.split == "val":
            split_sequences = valid_sequences[n_train:n_train + n_val]
        else:  # test
            split_sequences = valid_sequences[n_train + n_val:]
        
        # Cache split
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(split_sequences, f)
        
        return split_sequences
    
    def _validate_sequence(self, sequence: Dict) -> bool:
        """Validate that sequence has valid MRI files"""
        try:
            for timepoint in sequence['sequence']:
                mri_path = timepoint['mri_file']
                if not os.path.exists(mri_path):
                    return False
            return True
        except:
            return False
    
    def _fit_clinical_scaler(self) -> StandardScaler:
        """Fit scaler for clinical features"""
        clinical_features = []
        
        for sequence in self.sequences:
            for timepoint in sequence['sequence']:
                features = self._extract_clinical_features(timepoint)
                if features is not None:
                    clinical_features.append(features)
        
        if clinical_features:
            clinical_array = np.array(clinical_features)
            scaler = StandardScaler()
            scaler.fit(clinical_array)
            
            # Save scaler
            scaler_path = Path(self.cache_dir) / "clinical_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            return scaler
        else:
            return StandardScaler()  # Dummy scaler
    
    def _load_clinical_scaler(self) -> StandardScaler:
        """Load pre-fitted clinical scaler"""
        scaler_path = Path(self.cache_dir) / "clinical_scaler.pkl"
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning("Clinical scaler not found, using dummy scaler")
            return StandardScaler()
    
    def _extract_clinical_features(self, timepoint: Dict) -> Optional[np.ndarray]:
        """Extract clinical features from timepoint"""
        try:
            features = []
            
            # Demographic features
            features.append(timepoint.get('age', 70.0))  # Default age
            features.append(1.0 if timepoint.get('gender') == 'Male' else 0.0)
            features.append(timepoint.get('education', 16.0))  # Default education
            features.append(timepoint.get('apoe4', 0))
            
            # Cognitive scores
            features.append(timepoint.get('mmse', 28.0))  # Default MMSE
            features.append(timepoint.get('adas11', 5.0))  # Default ADAS11
            features.append(timepoint.get('adas13', 8.0))  # Default ADAS13
            features.append(timepoint.get('cdr', 0.0))     # Default CDR
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting clinical features: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample"""
        sequence_data = self.sequences[idx]
        sequence = sequence_data['sequence']
        
        # Load and preprocess MRI volumes
        mri_volumes = []
        clinical_features = []
        timepoints = []
        
        for i, timepoint in enumerate(sequence):
            # Load MRI
            try:
                mri_array = self._load_mri_cached(timepoint['mri_file'])
                
                # Apply transforms if training
                if self.transform and self.split == "train":
                    mri_array = self.transform(mri_array)
                
                mri_volumes.append(mri_array)
                
                # Extract clinical features
                clinical_feat = self._extract_clinical_features(timepoint)
                if clinical_feat is not None:
                    clinical_feat = self.clinical_scaler.transform(clinical_feat.reshape(1, -1))[0]
                    clinical_features.append(clinical_feat)
                else:
                    clinical_features.append(np.zeros(8, dtype=np.float32))
                
                timepoints.append(i)
                
            except Exception as e:
                logger.error(f"Error loading MRI {timepoint['mri_file']}: {e}")
                # Skip this timepoint
                continue
        
        