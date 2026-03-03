```python
"""
ADNI dataset loader and preprocessor for longitudinal MRI analysis
"""
import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import pickle
from datetime import datetime, timedelta

# Medical imaging libraries
import SimpleITK as sitk
from nilearn import image as nimg
from nilearn.plotting import plot_anat
import ants

logger = logging.getLogger(__name__)

class ADNILoader:
    """Loader for ADNI longitudinal MRI and clinical data"""
    
    def __init__(self, data_dir: str, cache_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected ADNI directory structure
        self.mri_dir = self.data_dir / "MRI"
        self.clinical_dir = self.data_dir / "Clinical"
        
        # Load clinical metadata
        self.clinical_data = self._load_clinical_data()
        self.demographics = self._load_demographics()
        
    def _load_clinical_data(self) -> pd.DataFrame:
        """Load clinical assessment data"""
        clinical_file = self.clinical_dir / "ADNIMERGE.csv"
        cache_file = self.cache_dir / "clinical_data.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached clinical data from {cache_file}")
            return pd.read_pickle(cache_file)
        
        if not clinical_file.exists():
            logger.warning(f"Clinical file not found: {clinical_file}")
            return pd.DataFrame()
        
        logger.info("Loading clinical data from ADNIMERGE.csv...")
        clinical_df = pd.read_csv(clinical_file)
        
        # Convert dates
        clinical_df['EXAMDATE'] = pd.to_datetime(clinical_df['EXAMDATE'])
        clinical_df['SCANDATE'] = pd.to_datetime(clinical_df['SCANDATE'], errors='coerce')
        
        # Create diagnosis mapping
        diagnosis_mapping = {
            'CN': 0,      # Cognitively Normal
            'SMC': 1,     # Significant Memory Concern
            'EMCI': 2,    # Early Mild Cognitive Impairment
            'LMCI': 3,    # Late Mild Cognitive Impairment
            'AD': 4       # Alzheimer's Disease
        }
        
        clinical_df['DX_numeric'] = clinical_df['DX'].map(diagnosis_mapping)
        
        # Cache processed data
        clinical_df.to_pickle(cache_file)
        logger.info(f"Loaded clinical data for {len(clinical_df['PTID'].unique())} patients")
        
        return clinical_df
    
    def _load_demographics(self) -> pd.DataFrame:
        """Load demographic data"""
        demo_file = self.clinical_dir / "PTDEMOG.csv"
        
        if not demo_file.exists():
            logger.warning(f"Demographics file not found: {demo_file}")
            return pd.DataFrame()
        
        demographics_df = pd.read_csv(demo_file)
        return demographics_df
    
    def get_patient_timeline(self, patient_id: str) -> pd.DataFrame:
        """Get longitudinal timeline for a patient"""
        if self.clinical_data.empty:
            return pd.DataFrame()
        
        patient_data = self.clinical_data[self.clinical_data['PTID'] == patient_id].copy()
        patient_data = patient_data.sort_values('EXAMDATE')
        
        return patient_data
    
    def get_mri_files(self, patient_id: str) -> List[Dict]:
        """Get MRI files for a patient"""
        patient_dir = self.mri_dir / patient_id
        
        if not patient_dir.exists():
            return []
        
        mri_files = []
        for scan_dir in patient_dir.iterdir():
            if scan_dir.is_dir():
                # Look for preprocessed files first
                nii_files = list(scan_dir.glob("*.nii")) + list(scan_dir.glob("*.nii.gz"))
                
                for nii_file in nii_files:
                    # Extract scan date from directory name or filename
                    scan_date = self._extract_scan_date(scan_dir.name, nii_file.name)
                    
                    mri_files.append({
                        'file_path': str(nii_file),
                        'scan_date': scan_date,
                        'modality': 'T1w',  # Assume T1-weighted
                        'scan_id': scan_dir.name
                    })
        
        # Sort by scan date
        mri_files.sort(key=lambda x: x['scan_date'] if x['scan_date'] else datetime.min)
        
        return mri_files
    
    def _extract_scan_date(self, dir_name: str, file_name: str) -> Optional[datetime]:
        """Extract scan date from directory or file name"""
        # Try to find date patterns in directory name
        import re
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
            r'(\d{8})',              # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, dir_name)
            if match:
                date_str = match.group(1)
                try:
                    if len(date_str) == 8:  # YYYYMMDD
                        return datetime.strptime(date_str, '%Y%m%d')
                    else:
                        date_str = date_str.replace('_', '-')
                        return datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    continue
        
        return None
    
    def create_longitudinal_dataset(
        self,
        min_timepoints: int = 3,
        max_timepoints: int = 8,
        time_window_months: int = 60
    ) -> List[Dict]:
        """Create longitudinal dataset with sequences of MRI scans"""
        cache_file = self.cache_dir / f"longitudinal_dataset_{min_timepoints}_{max_timepoints}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached longitudinal dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        longitudinal_data = []
        patient_ids = self.clinical_data['PTID'].unique()
        
        logger.info(f"Creating longitudinal dataset for {len(patient_ids)} patients...")
        
        for patient_id in tqdm(patient_ids, desc="Processing patients"):
            # Get patient timeline
            timeline = self.get_patient_timeline(patient_id)
            mri_files = self.get_mri_files(patient_id)
            
            if len(timeline) < min_timepoints or len(mri_files) < min_timepoints:
                continue
            
            # Match MRI files with clinical assessments
            matched_timepoints = self._match_mri_clinical(mri_files, timeline)
            
            if len(matched_timepoints) >= min_timepoints:
                # Create sequences
                sequences = self._create_sequences(
                    matched_timepoints, 
                    max_timepoints, 
                    time_window_months
                )
                
                longitudinal_data.extend(sequences)
        
        # Cache dataset
        with open(cache_file, 'wb') as f:
            pickle.dump(longitudinal_data, f)
        
        logger.info(f"Created longitudinal dataset with {len(longitudinal_data)} sequences")
        return longitudinal_data
    
    def _match_mri_clinical(
        self, 
        mri_files: List[Dict], 
        timeline: pd.DataFrame
    ) -> List[Dict]:
        """Match MRI scans with clinical assessments"""
        matched_timepoints = []
        
        for mri_file in mri_files:
            if mri_file['scan_date'] is None:
                continue
            
            # Find closest clinical assessment within 90 days
            time_diffs = abs((timeline['EXAMDATE'] - mri_file['scan_date']).dt.days)
            closest_idx = time_diffs.idxmin()
            
            if time_diffs.loc[closest_idx] <= 90:  # Within 3 months
                clinical_row = timeline.loc[closest_idx]
                
                timepoint = {
                    'mri_file': mri_file['file_path'],
                    'scan_date': mri_file['scan_date'],
                    'visit_date': clinical_row['EXAMDATE'],
                    'diagnosis': clinical_row['DX'],
                    'diagnosis_numeric': clinical_row['DX_numeric'],
                    'mmse': clinical_row.get('MMSE', np.nan),
                    'adas11': clinical_row.get('ADAS11', np.nan),
                    'adas13': clinical_row.get('ADAS13', np.nan),
                    'cdr': clinical_row.get('CDRSB', np.nan),
                    'age': clinical_row.get('AGE', np.nan),
                    'education': clinical_row.get('PTEDUCAT', np.nan),
                    'apoe4': clinical_row.get('APOE4', 0),
                    'gender': clinical_row.get('PTGENDER', 'Unknown')
                }
                
                matched_timepoints.append(timepoint)
        
        return sorted(matched_timepoints, key=lambda x: x['scan_date'])
    
    def _create_sequences(
        self, 
        timepoints: List[Dict], 
        max_length: int,
        time_window_months: int
    ) -> List[Dict]:
        """Create sequences from matched timepoints"""
        sequences = []
        
        for start_idx in range(len(timepoints)):
            sequence = []
            start_date = timepoints[start_idx]['scan_date']
            
            for tp in timepoints[start_idx:]:
                # Check if within time window
                months_diff = (tp['scan_date'] - start_date).days / 30.44
                
                if months_diff <= time_window_months and len(sequence) < max_length:
                    sequence.append(tp)
                else:
                    break
            
            if len(sequence) >= 3:  # Minimum sequence length
                # Calculate progression information
                progression_info = self._calculate_progression(sequence)
                
                sequence_data = {
                    'patient_id': timepoints[start_idx].get('patient_id', 'unknown'),
                    'sequence': sequence,
                    'length': len(sequence),
                    'progression_info': progression_info
                }
                
                sequences.append(sequence_data)
        
        return sequences
    
    def _calculate_progression(self, sequence: List[Dict]) -> Dict:
        """Calculate progression metrics for a sequence"""
        start_dx = sequence[0]['diagnosis_numeric']
        end_dx = sequence[-1]['diagnosis_numeric']
        
        # Extract cognitive scores
        mmse_scores = [tp['mmse'] for tp in sequence if not np.isnan(tp['mmse'])]
        adas_scores = [tp['adas11'] for tp in sequence if not np.isnan(tp['adas11'])]
        
        progression_info = {
            'start_diagnosis': start_dx,
            'end_diagnosis': end_dx,
            'progression_occurred': end_dx > start_dx,
            'progression_magnitude': end_dx - start_dx,
            'time_span_months': (sequence[-1]['scan_date'] - sequence[0]['scan_date']).days / 30.44,
            'cognitive_decline': {}
        }
        
        # Calculate cognitive decline
        if len(mmse_scores) >= 2:
            progression_info['cognitive_decline']['mmse_change'] = mmse_scores[-1] - mmse_scores[0]
            progression_info['cognitive_decline']['mmse_slope'] = (
                (mmse_scores[-1] - mmse_scores[0]) / 
                max(1, len(mmse_scores) - 1)
            )
        
        if len(adas_scores) >= 2:
            progression_info['cognitive_decline']['adas_change'] = adas_scores[-1] - adas_scores[0]
            progression_info['cognitive_decline']['adas_slope'] = (
                (adas_scores[-1] - adas_scores[0]) / 
                max(1, len(adas_scores) - 1)
            )
        
        return progression_info

class MRIPreprocessor:
    """Preprocessor for MRI images"""
    
    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        template_path: Optional[str] = None
    ):
        self.target_shape = target_shape
        self.voxel_spacing = voxel_spacing
        self.template_path = template_path
        
        # Load template if provided
        self.template = None
        if template_path and os.path.exists(template_path):
            self.template = ants.image_read(template_path)
    
    def preprocess_mri(
        self,
        mri_path: str,
        skull_strip: bool = True,
        register_to_template: bool = True,
        normalize_intensity: bool = True
    ) -> np.ndarray:
        """
        Preprocess MRI image
        
        Args:
            mri_path: Path to MRI file
            skull_strip: Whether to perform skull stripping
            register_to_template: Whether to register to template
            normalize_intensity: Whether to normalize intensity values
            
        Returns:
            Preprocessed MRI array
        """
        try:
            # Load image using ANTs
            img = ants.image_read(mri_path)
            
            # N4 bias field correction
            img = ants.n4_bias_field_correction(img)
            
            # Skull stripping
            if skull_strip:
                brain_mask = ants.get_mask(img)
                img = ants.mask_image(img, brain_mask)
            
            # Registration to template
            if register_to_template and self.template is not None:
                reg = ants.registration(
                    fixed=self.template,
                    moving=img,
                    type_of_transform='SyN'
                )
                img = reg['warpedmovout']
            
            # Resample to target spacing
            img = ants.resample_image(
                img,
                self.voxel_spacing,
                use_voxels=False,
                interp_type=1
            )
            
            # Convert to numpy array
            img_array = img.numpy()
            
            # Crop or pad to target shape
            img_array = self._crop_or_pad(img_array, self.target_shape)
            
            # Intensity normalization
            if normalize_intensity:
                img_array = self._normalize_intensity(img_array)
            
            return img_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing MRI {mri_path}: {e}")
            # Return zero array as fallback
            return np.zeros(self.target_shape, dtype=np.float32)
    
    def _crop_or_pad(self, img_array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Crop or pad image to target shape"""
        current_shape = img_array.shape
        
        # Calculate padding/cropping for each dimension
        padded_array = img_array
        
        for i, (current_size, target_size) in enumerate(zip(current_shape, target_shape)):
            if current_size < target_size:
                # Pad
                pad_before = (target_size - current_size) // 2
                pad_after = target_size - current_size - pad_before
                
                pad_width = [(0, 0)] * len(current_shape)
                pad_width[i] = (pad_before, pad_after)
                
                padded_array = np.pad(padded_array, pad_width, mode='constant', constant_values=0)
            
            elif current_size > target_size:
                # Crop
                start = (current_size - target_size) // 2
                end = start + target_size
                
                if i == 0:
                    padded_array = padded_array[start:end, :, :]
                elif i == 1:
                    padded_array = padded_array[:, start:end, :]
                elif i == 2:
                    padded_array = padded_array[:, :, start:end]
        
        return padded_array
    
    def _normalize_intensity(self, img_array: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1]"""
        # Remove background (assume 0 is background)
        foreground_mask = img_array > 0
        
        if foreground_mask.sum() == 0:
            return img_array
        
        # Robust normalization using percentiles
        p1, p99 = np.percentile(img_array[foreground_mask], [1, 99])
        
        # Clip and normalize
        img_array = np.clip(img_array, p1, p99)
        img_array = (img_array - p1) / (p99 - p1 + 1e-8)
        
        return img_array
```
