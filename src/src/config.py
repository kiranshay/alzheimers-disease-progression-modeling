```python
"""
Configuration settings for Alzheimer's Disease Progression Modeling
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class ModelConfig:
    """3D CNN + LSTM model configuration"""
    # 3D ResNet backbone
    resnet_depth: int = 18  # 18, 34, 50, 101
    spatial_dims: int = 3
    in_channels: int = 1
    initial_filters: int = 64
    
    # CNN feature extraction
    cnn_output_dim: int = 512
    dropout_cnn: float = 0.3
    
    # LSTM for temporal modeling
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    bidirectional: bool = True
    
    # Fusion and classification
    fusion_dim: int = 512
    num_classes: int = 4  # CN, SMC, EMCI, LMCI, AD -> 4 progression stages
    
    # Attention mechanism
    use_attention: bool = True
    attention_dim: int = 128

@dataclass
class DataConfig:
    """Data processing configuration"""
    data_dir: str = "data/ADNI/"
    cache_dir: str = "data/cache/"
    
    # MRI preprocessing
    target_shape: Tuple[int, int, int] = (96, 96, 96)  # Standardized brain size
    voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # mm
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    
    # Skull stripping and registration
    skull_strip: bool = True
    register_to_template: bool = True
    template_path: str = "templates/MNI152_T1_2mm.nii.gz"
    
    # Sequence parameters
    max_sequence_length: int = 8  # Maximum timepoints per patient
    min_sequence_length: int = 3  # Minimum timepoints for inclusion
    time_interval_months: int = 6  # Expected interval between scans
    
    # Data augmentation
    augmentation_prob: float = 0.5
    rotation_range: float = 10.0  # degrees
    translation_range: float = 5.0  # mm
    intensity_shift: float = 0.1
    
    # Clinical data integration
    clinical_features: List[str] = None
    cognitive_scores: List[str] = None
    
    # Train/val/test splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    random_seed: int = 42
    
    def __post_init__(self):
        if self.clinical_features is None:
            self.clinical_features = [
                "AGE", "PTGENDER", "PTEDUCAT", "APOE4",
                "MMSE", "CDR", "ADAS11", "ADAS13"
            ]
        
        if self.cognitive_scores is None:
            self.cognitive_scores = ["MMSE", "ADAS11", "ADAS13", "CDR_GLOBAL"]

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4  # Small due to 3D data memory requirements
    learning_rate: float = 1e-4
    num_epochs: int = 150
    weight_decay: float = 1e-4
    
    # Loss configuration
    progression_weight: float = 2.0  # Weight for progression prediction
    clinical_weight: float = 1.0    # Weight for clinical score prediction
    survival_weight: float = 1.5    # Weight for time-to-progression
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, step, reduce_on_plateau
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_delta: float = 0.001
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Evaluation
    eval_every_n_epochs: int = 5
    save_predictions: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```
