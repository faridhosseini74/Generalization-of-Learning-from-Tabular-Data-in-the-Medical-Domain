"""
Universal data utilities for glucose prediction with automatic dataset structure detection.
Works with any CSV/Parquet file containing glucose time series data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
import os
import hashlib
import json
import logging
from typing import Tuple, Dict, List, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlucoseDataset(Dataset):
    """PyTorch dataset for glucose prediction."""
    
    def __init__(self, features, targets):
        """
        Initialize the dataset.
        
        Args:
            features: Tuple of (X_cat, X_cont, cat_mask, con_mask)
            targets: Target values
        """
        if isinstance(features, tuple) and len(features) == 4:
            self.x_cat, self.x_cont, self.cat_mask, self.cont_mask = features
        elif isinstance(features, tuple) and len(features) == 2:
            self.x_cat, self.x_cont = features
            self.cat_mask = torch.ones_like(self.x_cat, dtype=torch.bool)
            self.cont_mask = torch.ones_like(self.x_cont, dtype=torch.bool)
        else:
            raise ValueError("Features should be a tuple of (X_cat, X_cont) or (X_cat, X_cont, cat_mask, con_mask)")
        
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (self.x_cat[idx], self.x_cont[idx], self.cat_mask[idx], self.cont_mask[idx]), self.targets[idx]

def detect_dataset_structure(data: pd.DataFrame, 
                           patient_col: Optional[str] = None,
                           time_col: Optional[str] = None, 
                           glucose_col: Optional[str] = None) -> Dict[str, str]:
    """
    Automatically detect the structure of a glucose dataset.
    
    Args:
        data: The dataset DataFrame
        patient_col: Optional patient ID column name (if known)
        time_col: Optional time column name (if known) 
        glucose_col: Optional glucose column name (if known)
        
    Returns:
        Dictionary with detected column names
    """
    detected = {}
    
    # Detect patient ID column
    if patient_col and patient_col in data.columns:
        detected['patient_col'] = patient_col
    else:
        patient_candidates = []
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['patient', 'id', 'subject', 'user']):
                # Check if it looks like an identifier (not too many unique values relative to dataset size)
                unique_ratio = data[col].nunique() / len(data)
                if 0.001 < unique_ratio < 0.5:  # Between 0.1% and 50% unique values
                    patient_candidates.append((col, unique_ratio))
        
        if patient_candidates:
            # Choose the one with the most reasonable ratio
            detected['patient_col'] = min(patient_candidates, key=lambda x: abs(x[1] - 0.1))[0]
            logger.info(f"Auto-detected patient column: {detected['patient_col']}")
        else:
            # Fallback: use first column if it looks like an ID
            first_col = data.columns[0]
            if data[first_col].nunique() / len(data) < 0.5:
                detected['patient_col'] = first_col
                logger.warning(f"Using first column as patient ID: {first_col}")
            else:
                raise ValueError("Could not detect patient ID column. Please specify patient_col parameter.")
    
    # Detect time column
    if time_col and time_col in data.columns:
        detected['time_col'] = time_col
    else:
        time_candidates = []
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'ts', 'timestamp', 'datetime']):
                time_candidates.append(col)
            elif data[col].dtype in ['datetime64[ns]', 'object']:
                # Try to parse a sample to see if it's a date
                sample = data[col].dropna().iloc[:100] if len(data) > 100 else data[col].dropna()
                try:
                    pd.to_datetime(sample.iloc[0])
                    time_candidates.append(col)
                except:
                    pass
        
        if time_candidates:
            detected['time_col'] = time_candidates[0]
            logger.info(f"Auto-detected time column: {detected['time_col']}")
        else:
            raise ValueError("Could not detect time column. Please specify time_col parameter.")
    
    # Detect glucose column
    if glucose_col and glucose_col in data.columns:
        detected['glucose_col'] = glucose_col
    else:
        glucose_candidates = []
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['glucose', 'bg', 'cgm', 'value', 'level']):
                # Check if it's numeric and in reasonable glucose range (30-600 mg/dL)
                try:
                    numeric_data = pd.to_numeric(data[col], errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        glucose_range = (numeric_data >= 30) & (numeric_data <= 600)
                        if glucose_range.mean() > 0.8:  # 80% of values in reasonable range
                            glucose_candidates.append((col, glucose_range.mean()))
                except:
                    pass
        
        if glucose_candidates:
            # Choose the one with highest percentage in glucose range
            detected['glucose_col'] = max(glucose_candidates, key=lambda x: x[1])[0]
            logger.info(f"Auto-detected glucose column: {detected['glucose_col']}")
        else:
            raise ValueError("Could not detect glucose column. Please specify glucose_col parameter.")
    
    return detected

def load_dataset(file_path: Union[str, Path], 
                patient_col: Optional[str] = None,
                time_col: Optional[str] = None,
                glucose_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a dataset from CSV or Parquet file with automatic structure detection.
    
    Args:
        file_path: Path to the dataset file (CSV or Parquet)
        patient_col: Optional patient ID column name
        time_col: Optional time column name
        glucose_col: Optional glucose column name
        
    Returns:
        Tuple of (data, config) where config contains detected column names
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        data = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.parquet', '.pq']:
        data = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use CSV or Parquet files.")
    
    logger.info(f"Loaded dataset from {file_path}: {data.shape}")
    
    # Detect dataset structure
    config = detect_dataset_structure(data, patient_col, time_col, glucose_col)
    config['name'] = file_path.stem
    config['path'] = str(file_path)
    
    return data, config

def get_available_datasets(base_path: Optional[str] = None) -> List[str]:
    """
    Get list of available dataset files in the given directory.
    
    Args:
        base_path: Directory to search for dataset files
        
    Returns:
        List of available dataset file paths
    """
    if base_path is None:
        base_path = Path(__file__).parent
    else:
        base_path = Path(base_path)
    
    datasets = []
    for ext in ['*.csv', '*.parquet', '*.pq']:
        datasets.extend(base_path.glob(ext))
    
    return [str(f) for f in datasets]

def get_dataset_info(file_path: Union[str, Path]) -> Dict:
    """
    Get basic information about a dataset file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Dictionary with dataset information
    """
    try:
        data, config = load_dataset(file_path)
        return {
            'path': str(file_path),
            'name': config['name'],
            'shape': data.shape,
            'columns': list(data.columns),
            'detected_structure': {
                'patient_col': config['patient_col'],
                'time_col': config['time_col'], 
                'glucose_col': config['glucose_col']
            }
        }
    except Exception as e:
        return {
            'path': str(file_path),
            'error': str(e)
        }

def extract_time_features(df, time_col):
    """Extract time-based features from a timestamp column."""
    df = df.copy()
    
    # Basic time features
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['day_of_month'] = df[time_col].dt.day
    df['month'] = df[time_col].dt.month
    
    # Derived features
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Time of day category
    def time_of_day(hour):
        if 5 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 17:
            return 1  # Afternoon
        elif 17 <= hour < 22:
            return 2  # Evening
        else:
            return 3  # Night
    
    df['time_of_day'] = df['hour'].apply(time_of_day)
    
    # Cyclic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def intelligent_feature_detection(data: pd.DataFrame, exclude_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Intelligently detect categorical and continuous features.
    
    Args:
        data: The dataset DataFrame
        exclude_cols: Columns to exclude from feature detection
        
    Returns:
        Tuple of (categorical_features, continuous_features)
    """
    cat_cols = []
    cont_cols = []
    
    # Process remaining columns
    for col in data.columns:
        if col in exclude_cols:
            continue
            
        # Check data type and cardinality
        if data[col].dtype == 'object':
            if data[col].nunique() < 50:
                cat_cols.append(col)
                logger.info(f"Column '{col}' identified as categorical")
            else:
                logger.info(f"Column '{col}' has high cardinality, excluding")
        else:
            # Numeric column
            if data[col].nunique() < 20:
                cat_cols.append(col)
                logger.info(f"Column '{col}' identified as categorical (low cardinality numeric)")
            else:
                cont_cols.append(col)
                logger.info(f"Column '{col}' identified as continuous")
    
    return cat_cols, cont_cols

def _generate_cache_key(file_path, window_size, prediction_horizon, per_patient, base_path):
    """Generate a unique cache key for the preprocessing parameters."""
    key_string = f"{file_path}_{window_size}_{prediction_horizon}_{per_patient}_{base_path or 'default'}"
    return hashlib.md5(key_string.encode()).hexdigest()

def _get_cache_paths(dataset_name, cache_key, base_path=None):
    """Get the cache file paths for preprocessed data."""
    if base_path:
        cache_dir = Path(base_path) / "preprocessed_cache"
    else:
        cache_dir = Path(__file__).parent / "preprocessed_cache"
    
    cache_dir.mkdir(exist_ok=True)
    
    return {
        'X_cat': cache_dir / f"{dataset_name}_{cache_key}_X_cat.parquet",
        'X_cont': cache_dir / f"{dataset_name}_{cache_key}_X_cont.parquet", 
        'y': cache_dir / f"{dataset_name}_{cache_key}_y.parquet",
        'cat_mask': cache_dir / f"{dataset_name}_{cache_key}_cat_mask.parquet",
        'con_mask': cache_dir / f"{dataset_name}_{cache_key}_con_mask.parquet",
        'metadata': cache_dir / f"{dataset_name}_{cache_key}_metadata.json"
    }

def _tensor_to_dataframe(tensor, prefix="col"):
    """Convert a tensor to a DataFrame for parquet storage."""
    if len(tensor.shape) == 1:
        # 1D tensor
        return pd.DataFrame({f'{prefix}_0': tensor.numpy()})
    elif len(tensor.shape) == 2:
        # 2D tensor
        return pd.DataFrame(tensor.numpy(), columns=[f'{prefix}_{i}' for i in range(tensor.shape[1])])
    elif len(tensor.shape) == 3:
        # 3D tensor - flatten to 2D with multi-index columns
        n_samples, seq_len, n_features = tensor.shape
        flattened = tensor.view(n_samples, seq_len * n_features).numpy()
        columns = [f'{prefix}_{i}_{j}' for i in range(seq_len) for j in range(n_features)]
        return pd.DataFrame(flattened, columns=columns)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def _dataframe_to_tensor(df, original_shape, dtype=torch.float32):
    """Convert a DataFrame back to a tensor with the original shape."""
    if len(original_shape) == 1:
        return torch.tensor(df.iloc[:, 0].values, dtype=dtype)
    elif len(original_shape) == 2:
        return torch.tensor(df.values, dtype=dtype)
    elif len(original_shape) == 3:
        # Reshape flattened data back to 3D
        flattened = torch.tensor(df.values, dtype=dtype)
        return flattened.view(original_shape)
    else:
        raise ValueError(f"Unsupported shape: {original_shape}")

def _save_preprocessed_data(cache_paths, X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler):
    """Save preprocessed data to parquet cache files."""
    
    # Save tensors as parquet files
    _tensor_to_dataframe(X_cat_tensor, "X_cat").to_parquet(cache_paths['X_cat'])
    _tensor_to_dataframe(X_cont_tensor, "X_cont").to_parquet(cache_paths['X_cont'])
    _tensor_to_dataframe(y_tensor, "y").to_parquet(cache_paths['y'])
    _tensor_to_dataframe(cat_mask.int(), "cat_mask").to_parquet(cache_paths['cat_mask'])
    _tensor_to_dataframe(con_mask.int(), "con_mask").to_parquet(cache_paths['con_mask'])
    
    # Save metadata (shapes, cat_dims, scaler) as JSON
    metadata = {
        'X_cat_shape': list(X_cat_tensor.shape),
        'X_cont_shape': list(X_cont_tensor.shape),
        'y_shape': list(y_tensor.shape),
        'cat_mask_shape': list(cat_mask.shape),
        'con_mask_shape': list(con_mask.shape),
        'cat_dims': cat_dims,
        'scaler_data_min_': glucose_scaler.data_min_.tolist(),
        'scaler_data_max_': glucose_scaler.data_max_.tolist(),
        'scaler_data_range_': glucose_scaler.data_range_.tolist(),
        'scaler_scale_': glucose_scaler.scale_.tolist(),
        'scaler_min_': glucose_scaler.min_.tolist()
    }
    
    with open(cache_paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Preprocessed data cached to parquet files in: {cache_paths['X_cat'].parent}")

def _load_preprocessed_data(cache_paths):
    """Load preprocessed data from parquet cache files."""
    # Load metadata first
    with open(cache_paths['metadata'], 'r') as f:
        metadata = json.load(f)
    
    # Load tensors from parquet files
    X_cat_df = pd.read_parquet(cache_paths['X_cat'])
    X_cont_df = pd.read_parquet(cache_paths['X_cont'])
    y_df = pd.read_parquet(cache_paths['y'])
    cat_mask_df = pd.read_parquet(cache_paths['cat_mask'])
    con_mask_df = pd.read_parquet(cache_paths['con_mask'])
    
    # Convert back to tensors
    X_cat_tensor = _dataframe_to_tensor(X_cat_df, metadata['X_cat_shape'], torch.long)
    X_cont_tensor = _dataframe_to_tensor(X_cont_df, metadata['X_cont_shape'], torch.float32)
    y_tensor = _dataframe_to_tensor(y_df, metadata['y_shape'], torch.float32)
    cat_mask = _dataframe_to_tensor(cat_mask_df, metadata['cat_mask_shape'], torch.bool)
    con_mask = _dataframe_to_tensor(con_mask_df, metadata['con_mask_shape'], torch.bool)
    
    # Reconstruct glucose scaler
    glucose_scaler = MinMaxScaler()
    glucose_scaler.data_min_ = np.array(metadata['scaler_data_min_'])
    glucose_scaler.data_max_ = np.array(metadata['scaler_data_max_'])
    glucose_scaler.data_range_ = np.array(metadata['scaler_data_range_'])
    glucose_scaler.scale_ = np.array(metadata['scaler_scale_'])
    glucose_scaler.min_ = np.array(metadata['scaler_min_'])
    
    cat_dims = metadata['cat_dims']
    
    logger.info(f"Loaded preprocessed data from cache")
    logger.info(f"X_cat shape: {X_cat_tensor.shape}, X_cont shape: {X_cont_tensor.shape}, y shape: {y_tensor.shape}")
    
    return X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler

def prepare_glucose_data_from_file(file_path: Union[str, Path],
                        window_size: int = 12,
                        prediction_horizon: int = 3,
                        per_patient: bool = True,
                        use_cache: bool = True,
                        patient_col: Optional[str] = None,
                        time_col: Optional[str] = None,
                        glucose_col: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], MinMaxScaler]:
    """
    Prepare glucose data for training with automatic structure detection.
    
    Args:
        file_path: Path to the dataset file
        window_size: Size of the sliding window
        prediction_horizon: Number of steps ahead to predict
        per_patient: Whether to create sequences separately for each patient
        use_cache: Whether to use cached preprocessed data
        patient_col: Optional patient ID column name
        time_col: Optional time column name  
        glucose_col: Optional glucose column name
        
    Returns:
        Tuple of (X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler)
    """
    # Generate cache key based on file path and parameters
    file_path = str(file_path)
    cache_key = _generate_cache_key(file_path, window_size, prediction_horizon, per_patient, None)
    
    # Check cache first if enabled
    if use_cache:
        cache_paths = _get_cache_paths(os.path.basename(file_path), cache_key, None)
        
        # Check if all cache files exist
        cache_exists = all(path.exists() for path in cache_paths.values())
        
        if cache_exists:
            logger.info(f"Loading preprocessed data from cache: {cache_paths['X_cat'].parent}")
            return _load_preprocessed_data(cache_paths)
        else:
            logger.info(f"Cache not found, will preprocess and save to: {cache_paths['X_cat'].parent}")
    else:
        logger.info("Cache disabled, preprocessing data...")
        cache_paths = None
        
    logger.info(f"=== Processing {file_path} ===")
    
    # Load dataset with automatic structure detection
    data, config = load_dataset(file_path, patient_col, time_col, glucose_col)
    
    patient_id_col = config['patient_col']
    time_col = config['time_col']
    glucose_col = config['glucose_col']
    
    logger.info(f"Using columns - Patient: {patient_id_col}, Time: {time_col}, Glucose: {glucose_col}")
    
    # Generic data filtering - remove obvious non-glucose records
    initial_len = len(data)
    if 'RecordType' in data.columns:
        # Filter for glucose-related records
        glucose_mask = data['RecordType'].str.contains('glucose|cgm|bg', case=False, na=False)
        data = data[glucose_mask]
        logger.info(f"Filtered by RecordType: {len(data)} rows remaining from {initial_len}")
    
    if 'Units' in data.columns:
        # Filter for glucose units
        units_mask = data['Units'].str.contains('mg/dl|mmol', case=False, na=False)
        data = data[units_mask]
        logger.info(f"Filtered by Units: {len(data)} rows remaining")
    
    # Remove header rows that may be present
    logger.info("Cleaning data and removing header rows...")
    initial_len = len(data)
    data = data[data[time_col] != time_col]
    data = data[data[patient_id_col] != patient_id_col] 
    data = data[data[glucose_col] != glucose_col]
    final_len = len(data)
    if initial_len != final_len:
        logger.info(f"Removed {initial_len - final_len} header rows")
    
    # Convert timestamp to datetime
    logger.info(f"Converting {time_col} to datetime...")
    logger.info(f"Original timestamp dtype: {data[time_col].dtype}")
    logger.info(f"Sample timestamps: {data[time_col].head(5).values}")
    
    if data[time_col].dtype != 'datetime64[ns]':
        # Try multiple common formats
        formats_to_try = [
            '%Y-%m-%dT%H:%M:%S',      # ISO format (HUPA)
            '%Y-%m-%dT%H:%M:%S.%f',   # ISO with microseconds
            '%m/%d/%Y %H:%M',         # US format (Ohio)
            '%m/%d/%Y %H:%M:%S',      # US format with seconds
            '%Y-%m-%d %H:%M:%S',      # Standard format
            '%Y-%m-%d %H:%M:%S.%f',   # Standard with microseconds
            '%d/%m/%Y %H:%M:%S',      # European format
            None                       # Let pandas infer
        ]
        
        converted = None
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    converted = pd.to_datetime(data[time_col], infer_datetime_format=True, errors='coerce')
                    fmt_name = "inferred"
                else:
                    converted = pd.to_datetime(data[time_col], format=fmt, errors='coerce')
                    fmt_name = fmt
                
                valid_count = converted.count()
                logger.info(f"Format '{fmt_name}': {valid_count}/{len(data)} valid conversions")
                
                if valid_count > len(data) * 0.5:  # If more than 50% convert successfully
                    data[time_col] = converted
                    logger.info(f"Successfully used format: {fmt_name}")
                    break
            except Exception as e:
                logger.info(f"Format '{fmt_name}' failed: {e}")
        else:
            logger.warning("All timestamp conversion attempts failed!")
    
    # Remove rows with invalid timestamps
    before_time_filter = len(data)
    data = data.dropna(subset=[time_col])
    after_time_filter = len(data)
    if before_time_filter != after_time_filter:
        logger.info(f"Removed {before_time_filter - after_time_filter} rows with invalid timestamps")
    
    # Extract time features
    logger.info("Extracting time features...")
    data = extract_time_features(data, time_col)
    
    # Sort by patient and timestamp
    data = data.sort_values([patient_id_col, time_col]).reset_index(drop=True)
    
    # Convert glucose to numeric
    logger.info(f"Converting {glucose_col} to numeric...")
    data[glucose_col] = pd.to_numeric(data[glucose_col], errors='coerce')
    
    # Define categorical and continuous columns
    categorical_time_features = ['hour', 'minute', 'day_of_week', 'day_of_month', 'month', 
                                'is_weekend', 'time_of_day']
    continuous_time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                               'day_of_week_sin', 'day_of_week_cos']
    
    cat_cols = []
    cont_cols = []
    
    # Add time features
    for col in categorical_time_features:
        if col in data.columns:
            cat_cols.append(col)
    
    for col in continuous_time_features:
        if col in data.columns:
            cont_cols.append(col)
    
    # Exclude columns we don't want as features
    exclude_cols = [patient_id_col, time_col, glucose_col]
    
    # Intelligently detect additional categorical and continuous features
    additional_cat_cols, additional_cont_cols = intelligent_feature_detection(data, exclude_cols + cat_cols + cont_cols)
    cat_cols.extend(additional_cat_cols)
    cont_cols.extend(additional_cont_cols)
    
    logger.info(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    logger.info(f"Continuous columns ({len(cont_cols)}): {cont_cols}")
    
    # Create clean dataframe
    logger.info("Creating clean dataframe...")
    clean_df = pd.DataFrame()
    
    # Copy essential columns
    clean_df[patient_id_col] = data[patient_id_col]
    clean_df[glucose_col] = data[glucose_col]
    
    # Process categorical columns
    for col in cat_cols:
        if col in data.columns:
            clean_df[col] = data[col].astype(str).fillna('unknown')
            le = LabelEncoder()
            clean_df[col] = le.fit_transform(clean_df[col])
    
    # Process continuous columns
    for col in cont_cols:
        if col in data.columns:
            clean_df[col] = pd.to_numeric(data[col], errors='coerce')
            if clean_df[col].isna().sum() > 0:
                logger.warning(f"Column '{col}' has {clean_df[col].isna().sum()} NaN values")
            clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
    
    # Create sequences
    logger.info(f"Creating sliding window sequences with window={window_size}")
    
    all_X_cat = []
    all_X_cont = []
    all_y = []
    
    if per_patient:
        # Process data by patient
        patient_groups = clean_df.groupby(patient_id_col)
        print(f"Found {len(patient_groups)} different patients")
        
        for patient_id, patient_df in patient_groups:
            patient_df = patient_df.reset_index(drop=True)
            n_sequences = max(0, len(patient_df) - window_size)
            
            if n_sequences <= 0:
                continue
                
            # Create arrays for this patient
            if cat_cols:
                patient_X_cat = np.zeros((n_sequences, window_size, len(cat_cols)), dtype=np.int64)
            else:
                patient_X_cat = np.zeros((n_sequences, window_size, 1), dtype=np.int64)
                
            if cont_cols:
                patient_X_cont = np.zeros((n_sequences, window_size, len(cont_cols)), dtype=np.float32)
            else:
                patient_X_cont = np.zeros((n_sequences, window_size, 1), dtype=np.float32)
                
            patient_y = np.zeros(n_sequences, dtype=np.float32)
            
            # Generate sequences
            for i in range(n_sequences):
                window = patient_df.iloc[i:i+window_size]
                if cat_cols:
                    patient_X_cat[i] = window[cat_cols].values
                if cont_cols:
                    patient_X_cont[i] = window[cont_cols].values
                # Predict the next value
                patient_y[i] = patient_df.iloc[i+window_size][glucose_col]
            
            all_X_cat.append(patient_X_cat)
            all_X_cont.append(patient_X_cont)
            all_y.append(patient_y)
        
        # Combine all sequences
        if all_X_cat:
            X_cat = np.vstack(all_X_cat)
            X_cont = np.vstack(all_X_cont)
            y = np.concatenate(all_y)
        else:
            X_cat = np.zeros((0, window_size, max(1, len(cat_cols))), dtype=np.int64)
            X_cont = np.zeros((0, window_size, max(1, len(cont_cols))), dtype=np.float32)
            y = np.zeros(0, dtype=np.float32)
    
    else:
        # Process without patient splitting
        n_sequences = max(0, len(clean_df) - window_size)
        
        if cat_cols:
            X_cat = np.zeros((n_sequences, window_size, len(cat_cols)), dtype=np.int64)
        else:
            X_cat = np.zeros((n_sequences, window_size, 1), dtype=np.int64)
            
        if cont_cols:
            X_cont = np.zeros((n_sequences, window_size, len(cont_cols)), dtype=np.float32)
        else:
            X_cont = np.zeros((n_sequences, window_size, 1), dtype=np.float32)
            
        y = np.zeros(n_sequences, dtype=np.float32)
        
        for i in range(n_sequences):
            window = clean_df.iloc[i:i+window_size]
            if cat_cols:
                X_cat[i] = window[cat_cols].values
            if cont_cols:
                X_cont[i] = window[cont_cols].values
            y[i] = clean_df.iloc[i+window_size][glucose_col]
    
    print(f"Created {len(y)} sequences")
    print(f"X_cat shape: {X_cat.shape}")
    print(f"X_cont shape: {X_cont.shape}")
    
    # FIXED: Proper handling of category dimensions
    if cat_cols and X_cat.shape[2] > 0 and X_cat.size > 0:
        # Only calculate cat_dims if we actually have categorical data and sequences
        cat_dims = []
        for i in range(X_cat.shape[2]):
            try:
                max_val = int(X_cat[:, :, i].max())
                cat_dims.append(max_val + 1)
            except ValueError:  # Empty array
                cat_dims.append(2)  # Default for empty arrays
    else:
        cat_dims = [2]  # Dummy category with 2 values for no categorical features
        # Ensure X_cat has the right shape even when empty
        if X_cat.shape[2] == 0:
            X_cat = np.zeros((X_cat.shape[0], X_cat.shape[1], 1), dtype=np.int64)
    
    print(f"Category dimensions: {cat_dims}")
    
    # Normalize continuous features
    if cont_cols and X_cont.shape[2] > 0 and X_cont.size > 0:
        scaler = MinMaxScaler()
        original_shape = X_cont.shape
        X_cont_flat = X_cont.reshape(-1, X_cont.shape[2])
        X_cont_flat = scaler.fit_transform(X_cont_flat)
        X_cont = X_cont_flat.reshape(original_shape)
        print("Continuous features normalized using MinMaxScaler")
    elif X_cont.shape[2] == 0:
        # Ensure X_cont has the right shape even when empty
        X_cont = np.zeros((X_cont.shape[0], X_cont.shape[1], 1), dtype=np.float32)
    
    # Normalize target values
    glucose_scaler = MinMaxScaler()
    if y.size > 0:
        y_normalized = glucose_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        print(f"Target values normalized. Original range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Normalized range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
        y = y_normalized
    else:
        print("Warning: No target values to normalize (empty dataset)")
        # Fit scaler with dummy data to avoid errors later
        glucose_scaler.fit([[0], [1]])
    
    # Convert to tensors
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_cont_tensor = torch.tensor(X_cont, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Create masks
    cat_mask = torch.ones_like(X_cat_tensor, dtype=torch.bool)
    con_mask = torch.ones_like(X_cont_tensor, dtype=torch.bool)
    
    print(f"Final tensor shapes:")
    print(f"X_cat: {X_cat_tensor.shape}")
    print(f"X_cont: {X_cont_tensor.shape}")
    print(f"y: {y_tensor.shape}")
    
    # Save to cache if enabled and cache_paths was set
    if use_cache and cache_paths is not None:
        _save_preprocessed_data(cache_paths, X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler)
    
    return X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler

def _generate_cache_key(dataset_key, window_size, prediction_horizon, per_patient, base_path):
    """Generate a unique cache key for the preprocessing parameters."""
    key_string = f"{dataset_key}_{window_size}_{prediction_horizon}_{per_patient}_{base_path or 'default'}"
    return hashlib.md5(key_string.encode()).hexdigest()

def _get_cache_paths(dataset_key, cache_key, base_path=None):
    """Get the cache file paths for preprocessed data."""
    if base_path:
        cache_dir = Path(base_path) / "preprocessed_cache"
    else:
        cache_dir = Path(__file__).parent / "preprocessed_cache"
    
    cache_dir.mkdir(exist_ok=True)
    
    return {
        'X_cat': cache_dir / f"{dataset_key}_{cache_key}_X_cat.parquet",
        'X_cont': cache_dir / f"{dataset_key}_{cache_key}_X_cont.parquet", 
        'y': cache_dir / f"{dataset_key}_{cache_key}_y.parquet",
        'cat_mask': cache_dir / f"{dataset_key}_{cache_key}_cat_mask.parquet",
        'con_mask': cache_dir / f"{dataset_key}_{cache_key}_con_mask.parquet",
        'metadata': cache_dir / f"{dataset_key}_{cache_key}_metadata.json"
    }

def _tensor_to_dataframe(tensor, prefix="col"):
    """Convert a tensor to a DataFrame for parquet storage."""
    if len(tensor.shape) == 1:
        # 1D tensor
        return pd.DataFrame({f'{prefix}_0': tensor.numpy()})
    elif len(tensor.shape) == 2:
        # 2D tensor
        return pd.DataFrame(tensor.numpy(), columns=[f'{prefix}_{i}' for i in range(tensor.shape[1])])
    elif len(tensor.shape) == 3:
        # 3D tensor - flatten to 2D with multi-index columns
        n_samples, seq_len, n_features = tensor.shape
        flattened = tensor.view(n_samples, seq_len * n_features).numpy()
        columns = [f'{prefix}_{i}_{j}' for i in range(seq_len) for j in range(n_features)]
        return pd.DataFrame(flattened, columns=columns)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def _dataframe_to_tensor(df, original_shape, dtype=torch.float32):
    """Convert a DataFrame back to a tensor with the original shape."""
    if len(original_shape) == 1:
        return torch.tensor(df.iloc[:, 0].values, dtype=dtype)
    elif len(original_shape) == 2:
        return torch.tensor(df.values, dtype=dtype)
    elif len(original_shape) == 3:
        # Reshape flattened data back to 3D
        flattened = torch.tensor(df.values, dtype=dtype)
        return flattened.view(original_shape)
    else:
        raise ValueError(f"Unsupported shape: {original_shape}")

def _save_preprocessed_data(cache_paths, X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler):
    """Save preprocessed data to parquet cache files."""
    
    # Save tensors as parquet files
    _tensor_to_dataframe(X_cat_tensor, "X_cat").to_parquet(cache_paths['X_cat'])
    _tensor_to_dataframe(X_cont_tensor, "X_cont").to_parquet(cache_paths['X_cont'])
    _tensor_to_dataframe(y_tensor, "y").to_parquet(cache_paths['y'])
    _tensor_to_dataframe(cat_mask.int(), "cat_mask").to_parquet(cache_paths['cat_mask'])
    _tensor_to_dataframe(con_mask.int(), "con_mask").to_parquet(cache_paths['con_mask'])
    
    # Save metadata (shapes, cat_dims, scaler) as JSON
    metadata = {
        'X_cat_shape': list(X_cat_tensor.shape),
        'X_cont_shape': list(X_cont_tensor.shape),
        'y_shape': list(y_tensor.shape),
        'cat_mask_shape': list(cat_mask.shape),
        'con_mask_shape': list(con_mask.shape),
        'cat_dims': cat_dims,
        'scaler_data_min_': glucose_scaler.data_min_.tolist(),
        'scaler_data_max_': glucose_scaler.data_max_.tolist(),
        'scaler_data_range_': glucose_scaler.data_range_.tolist(),
        'scaler_scale_': glucose_scaler.scale_.tolist(),
        'scaler_min_': glucose_scaler.min_.tolist()
    }
    
    with open(cache_paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Preprocessed data cached to parquet files in: {cache_paths['X_cat'].parent}")

def _load_preprocessed_data(cache_paths):
    """Load preprocessed data from parquet cache files."""
    
    # Load metadata
    with open(cache_paths['metadata'], 'r') as f:
        metadata = json.load(f)
    
    # Load tensors from parquet files
    X_cat_df = pd.read_parquet(cache_paths['X_cat'])
    X_cont_df = pd.read_parquet(cache_paths['X_cont'])
    y_df = pd.read_parquet(cache_paths['y'])
    cat_mask_df = pd.read_parquet(cache_paths['cat_mask'])
    con_mask_df = pd.read_parquet(cache_paths['con_mask'])
    
    # Convert back to tensors
    X_cat_tensor = _dataframe_to_tensor(X_cat_df, metadata['X_cat_shape'], torch.long)
    X_cont_tensor = _dataframe_to_tensor(X_cont_df, metadata['X_cont_shape'], torch.float32)
    y_tensor = _dataframe_to_tensor(y_df, metadata['y_shape'], torch.float32)
    cat_mask = _dataframe_to_tensor(cat_mask_df, metadata['cat_mask_shape'], torch.bool)
    con_mask = _dataframe_to_tensor(con_mask_df, metadata['con_mask_shape'], torch.bool)
    
    # Reconstruct scaler
    glucose_scaler = MinMaxScaler()
    glucose_scaler.data_min_ = np.array(metadata['scaler_data_min_'])
    glucose_scaler.data_max_ = np.array(metadata['scaler_data_max_'])
    glucose_scaler.data_range_ = np.array(metadata['scaler_data_range_'])
    glucose_scaler.scale_ = np.array(metadata['scaler_scale_'])
    glucose_scaler.min_ = np.array(metadata['scaler_min_'])
    
    return (
        X_cat_tensor,
        X_cont_tensor,
        y_tensor,
        cat_mask,
        con_mask,
        metadata['cat_dims'],
        glucose_scaler
    )

def prepare_glucose_data(dataset_key, window_size=6, prediction_horizon=24, per_patient=True, base_path=None, use_cache=True):
    """
    Prepare data for glucose prediction from any of the 4 parquet datasets.
    
    Args:
        dataset_key: Dataset identifier ('ohio', 'hupa', 'dexcom', 'iobp2', 'combined_ohio_and_hupa')
        window_size: Size of the sliding window
        prediction_horizon: Prediction horizon (not used in sliding window implementation)
        per_patient: Whether to split data by patient ID before creating sequences
        base_path: Base path to override default dataset paths
        use_cache: Whether to use cached preprocessed data if available
        
    Returns:
        Tuple of (X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler)
    """
    # Check cache first if enabled
    if use_cache:
        cache_key = _generate_cache_key(dataset_key, window_size, prediction_horizon, per_patient, base_path)
        cache_paths = _get_cache_paths(dataset_key, cache_key, base_path)
        
        # Check if all cache files exist
        cache_exists = all(path.exists() for path in cache_paths.values())
        
        if cache_exists:
            print(f"Loading preprocessed data from cache: {cache_paths['X_cat'].parent}")
            return _load_preprocessed_data(cache_paths)
        else:
            print(f"Cache not found, will preprocess and save to: {cache_paths['X_cat'].parent}")
    else:
        print("Cache disabled, preprocessing data...")
        cache_paths = None
    print(f"=== Processing {dataset_key} dataset ===")
    
    # Load dataset
    data, config = load_dataset(dataset_key, base_path)
    
    patient_id_col = config['patient_col']
    time_col = config['time_col']
    glucose_col = config['glucose_col']
    
    logger.info(f"Using columns - Patient: {patient_id_col}, Time: {time_col}, Glucose: {glucose_col}")
    
    # Generic data filtering - remove obvious non-glucose records
    initial_len = len(data)
    if 'RecordType' in data.columns:
        # Filter for glucose-related records
        glucose_mask = data['RecordType'].str.contains('glucose|cgm|bg', case=False, na=False)
        data = data[glucose_mask]
        logger.info(f"Filtered by RecordType: {len(data)} rows remaining from {initial_len}")
    
    if 'Units' in data.columns:
        # Filter for glucose units
        units_mask = data['Units'].str.contains('mg/dl|mmol', case=False, na=False)
        data = data[units_mask]
        logger.info(f"Filtered by Units: {len(data)} rows remaining")
    
    # Remove header rows that may be present
    logger.info("Cleaning data and removing header rows...")
    initial_len = len(data)
    data = data[data[time_col] != time_col]
    data = data[data[patient_id_col] != patient_id_col] 
    data = data[data[glucose_col] != glucose_col]
    final_len = len(data)
    if initial_len != final_len:
        logger.info(f"Removed {initial_len - final_len} header rows")
    
    # Convert timestamp to datetime
    logger.info(f"Converting {time_col} to datetime...")
    logger.info(f"Original timestamp dtype: {data[time_col].dtype}")
    logger.info(f"Sample timestamps: {data[time_col].head(5).values}")
    
    if data[time_col].dtype != 'datetime64[ns]':
        # Try multiple common formats
        formats_to_try = [
            '%Y-%m-%dT%H:%M:%S',      # ISO format (HUPA)
            '%Y-%m-%dT%H:%M:%S.%f',   # ISO with microseconds
            '%m/%d/%Y %H:%M',         # US format (Ohio)
            '%m/%d/%Y %H:%M:%S',      # US format with seconds
            '%Y-%m-%d %H:%M:%S',      # Standard format
            '%Y-%m-%d %H:%M:%S.%f',   # Standard with microseconds
            '%d/%m/%Y %H:%M:%S',      # European format
            None                       # Let pandas infer
        ]
        
        converted = None
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    converted = pd.to_datetime(data[time_col], infer_datetime_format=True, errors='coerce')
                    fmt_name = "inferred"
                else:
                    converted = pd.to_datetime(data[time_col], format=fmt, errors='coerce')
                    fmt_name = fmt
                
                valid_count = converted.count()
                logger.info(f"Format '{fmt_name}': {valid_count}/{len(data)} valid conversions")
                
                if valid_count > len(data) * 0.5:  # If more than 50% convert successfully
                    data[time_col] = converted
                    logger.info(f"Successfully used format: {fmt_name}")
                    break
            except Exception as e:
                logger.info(f"Format '{fmt_name}' failed: {e}")
        else:
            logger.warning("All timestamp conversion attempts failed!")
    
    # Remove rows with invalid timestamps
    before_time_filter = len(data)
    data = data.dropna(subset=[time_col])
    after_time_filter = len(data)
    if before_time_filter != after_time_filter:
        logger.info(f"Removed {before_time_filter - after_time_filter} rows with invalid timestamps")
    
    # Extract time features
    logger.info("Extracting time features...")
    data = extract_time_features(data, time_col)
    
    # Sort by patient and timestamp
    data = data.sort_values([patient_id_col, time_col]).reset_index(drop=True)
    
    # Convert glucose to numeric
    logger.info(f"Converting {glucose_col} to numeric...")
    data[glucose_col] = pd.to_numeric(data[glucose_col], errors='coerce')
    
    # Define categorical and continuous columns
    categorical_time_features = ['hour', 'minute', 'day_of_week', 'day_of_month', 'month', 
                                'is_weekend', 'time_of_day']
    continuous_time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                               'day_of_week_sin', 'day_of_week_cos']
    
    cat_cols = []
    cont_cols = []
    
    # Add time features
    for col in categorical_time_features:
        if col in data.columns:
            cat_cols.append(col)
    
    for col in continuous_time_features:
        if col in data.columns:
            cont_cols.append(col)
    
    # Exclude columns we don't want as features
    exclude_cols = [patient_id_col, time_col, glucose_col]
    
    # Intelligently detect additional categorical and continuous features
    additional_cat_cols, additional_cont_cols = intelligent_feature_detection(data, exclude_cols + cat_cols + cont_cols)
    cat_cols.extend(additional_cat_cols)
    cont_cols.extend(additional_cont_cols)
    
    logger.info(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    logger.info(f"Continuous columns ({len(cont_cols)}): {cont_cols}")
    
    # Create clean dataframe
    logger.info("Creating clean dataframe...")
    clean_df = pd.DataFrame()
    
    # Copy essential columns
    clean_df[patient_id_col] = data[patient_id_col]
    clean_df[glucose_col] = data[glucose_col]
    
    # Process categorical columns
    for col in cat_cols:
        if col in data.columns:
            clean_df[col] = data[col].astype(str).fillna('unknown')
            le = LabelEncoder()
            clean_df[col] = le.fit_transform(clean_df[col])
    
    # Process continuous columns
    for col in cont_cols:
        if col in data.columns:
            clean_df[col] = pd.to_numeric(data[col], errors='coerce')
            if clean_df[col].isna().sum() > 0:
                logger.warning(f"Column '{col}' has {clean_df[col].isna().sum()} NaN values")
            clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
    
    # Create sequences
    logger.info(f"Creating sliding window sequences with window={window_size}")
    
    all_X_cat = []
    all_X_cont = []
    all_y = []
    
    if per_patient:
        # Process data by patient
        patient_groups = clean_df.groupby(patient_id_col)
        print(f"Found {len(patient_groups)} different patients")
        
        for patient_id, patient_df in patient_groups:
            patient_df = patient_df.reset_index(drop=True)
            n_sequences = max(0, len(patient_df) - window_size)
            
            if n_sequences <= 0:
                continue
                
            # Create arrays for this patient
            if cat_cols:
                patient_X_cat = np.zeros((n_sequences, window_size, len(cat_cols)), dtype=np.int64)
            else:
                patient_X_cat = np.zeros((n_sequences, window_size, 1), dtype=np.int64)
                
            if cont_cols:
                patient_X_cont = np.zeros((n_sequences, window_size, len(cont_cols)), dtype=np.float32)
            else:
                patient_X_cont = np.zeros((n_sequences, window_size, 1), dtype=np.float32)
                
            patient_y = np.zeros(n_sequences, dtype=np.float32)
            
            # Generate sequences
            for i in range(n_sequences):
                window = patient_df.iloc[i:i+window_size]
                if cat_cols:
                    patient_X_cat[i] = window[cat_cols].values
                if cont_cols:
                    patient_X_cont[i] = window[cont_cols].values
                # Predict the next value
                patient_y[i] = patient_df.iloc[i+window_size][glucose_col]
            
            all_X_cat.append(patient_X_cat)
            all_X_cont.append(patient_X_cont)
            all_y.append(patient_y)
        
        # Combine all sequences
        if all_X_cat:
            X_cat = np.vstack(all_X_cat)
            X_cont = np.vstack(all_X_cont)
            y = np.concatenate(all_y)
        else:
            X_cat = np.zeros((0, window_size, max(1, len(cat_cols))), dtype=np.int64)
            X_cont = np.zeros((0, window_size, max(1, len(cont_cols))), dtype=np.float32)
            y = np.zeros(0, dtype=np.float32)
    
    else:
        # Process without patient splitting
        n_sequences = max(0, len(clean_df) - window_size)
        
        if cat_cols:
            X_cat = np.zeros((n_sequences, window_size, len(cat_cols)), dtype=np.int64)
        else:
            X_cat = np.zeros((n_sequences, window_size, 1), dtype=np.int64)
            
        if cont_cols:
            X_cont = np.zeros((n_sequences, window_size, len(cont_cols)), dtype=np.float32)
        else:
            X_cont = np.zeros((n_sequences, window_size, 1), dtype=np.float32)
            
        y = np.zeros(n_sequences, dtype=np.float32)
        
        for i in range(n_sequences):
            window = clean_df.iloc[i:i+window_size]
            if cat_cols:
                X_cat[i] = window[cat_cols].values
            if cont_cols:
                X_cont[i] = window[cont_cols].values
            y[i] = clean_df.iloc[i+window_size][glucose_col]
    
    print(f"Created {len(y)} sequences")
    print(f"X_cat shape: {X_cat.shape}")
    print(f"X_cont shape: {X_cont.shape}")
    
    # FIXED: Proper handling of category dimensions
    if cat_cols and X_cat.shape[2] > 0 and X_cat.size > 0:
        # Only calculate cat_dims if we actually have categorical data and sequences
        cat_dims = []
        for i in range(X_cat.shape[2]):
            try:
                max_val = int(X_cat[:, :, i].max())
                cat_dims.append(max_val + 1)
            except ValueError:  # Empty array
                cat_dims.append(2)  # Default for empty arrays
    else:
        cat_dims = [2]  # Dummy category with 2 values for no categorical features
        # Ensure X_cat has the right shape even when empty
        if X_cat.shape[2] == 0:
            X_cat = np.zeros((X_cat.shape[0], X_cat.shape[1], 1), dtype=np.int64)
    
    print(f"Category dimensions: {cat_dims}")
    
    # Normalize continuous features
    if cont_cols and X_cont.shape[2] > 0 and X_cont.size > 0:
        scaler = MinMaxScaler()
        original_shape = X_cont.shape
        X_cont_flat = X_cont.reshape(-1, X_cont.shape[2])
        X_cont_flat = scaler.fit_transform(X_cont_flat)
        X_cont = X_cont_flat.reshape(original_shape)
        print("Continuous features normalized using MinMaxScaler")
    elif X_cont.shape[2] == 0:
        # Ensure X_cont has the right shape even when empty
        X_cont = np.zeros((X_cont.shape[0], X_cont.shape[1], 1), dtype=np.float32)
    
    # Normalize target values
    glucose_scaler = MinMaxScaler()
    if y.size > 0:
        y_normalized = glucose_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        print(f"Target values normalized. Original range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Normalized range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
        y = y_normalized
    else:
        print("Warning: No target values to normalize (empty dataset)")
        # Fit scaler with dummy data to avoid errors later
        glucose_scaler.fit([[0], [1]])
    
    # Convert to tensors
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
    X_cont_tensor = torch.tensor(X_cont, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Create masks
    cat_mask = torch.ones_like(X_cat_tensor, dtype=torch.bool)
    con_mask = torch.ones_like(X_cont_tensor, dtype=torch.bool)
    
    print(f"Final tensor shapes:")
    print(f"X_cat: {X_cat_tensor.shape}")
    print(f"X_cont: {X_cont_tensor.shape}")
    print(f"y: {y_tensor.shape}")
    
    # Save to cache if enabled and cache_paths was set
    if use_cache and cache_paths is not None:
        _save_preprocessed_data(cache_paths, X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler)
    
    return X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler


def prepare_data_for_model(file_path: Union[str, Path],
                          model_name: str,
                          window_size: int = 12,
                          prediction_horizon: int = 3,
                          per_patient: bool = True,
                          use_cache: bool = True,
                          patient_col: Optional[str] = None,
                          time_col: Optional[str] = None,
                          glucose_col: Optional[str] = None,
                          **model_kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], MinMaxScaler]:
    """
    Prepare data optimized for specific model architectures.
    
    Args:
        file_path: Path to the dataset file
        model_name: Name of the model ('qisn', 'saint', 'ft-transformer', etc.)
        window_size: Size of the sliding window
        prediction_horizon: Number of steps ahead to predict
        per_patient: Whether to create sequences separately for each patient
        use_cache: Whether to use cached preprocessed data
        patient_col: Optional patient ID column name
        time_col: Optional time column name  
        glucose_col: Optional glucose column name
        **model_kwargs: Additional model-specific parameters
        
    Returns:
        Tuple of (X_cat_tensor, X_cont_tensor, y_tensor, cat_mask, con_mask, cat_dims, glucose_scaler)
    """
    model_name = model_name.lower()
    
    # Get base data preparation
    X_cat, X_cont, y, cat_mask, con_mask, cat_dims, glucose_scaler = prepare_glucose_data_from_file(
        file_path=file_path,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        per_patient=per_patient,
        use_cache=use_cache,
        patient_col=patient_col,
        time_col=time_col,
        glucose_col=glucose_col
    )
    
    # Apply model-specific transformations
    if model_name == 'qisn':
        # QISN works with the standard format
        logger.info("Data prepared for QISN model")
        
    elif model_name in ['saint', 'ft-transformer', 'fttransformer']:
        # These models work well with the standard format but may benefit from different masking
        logger.info(f"Data prepared for {model_name.upper()} model")
        
        # For SAINT, we might want to ensure proper intersample attention setup
        if model_name == 'saint':
            # SAINT can benefit from specific attention patterns
            pass
            
    elif model_name == 'tabpfn':
        # TabPFN might work better with shorter sequences for meta-learning
        logger.info("Data prepared for TabPFN model")
        # TabPFN typically works with smaller datasets and shorter sequences
        
    elif model_name == 'patchtst':
        # PatchTST needs specific patch size considerations
        patch_len = model_kwargs.get('patch_len', 3)
        if window_size < patch_len:
            logger.warning(f"Window size ({window_size}) smaller than patch length ({patch_len}). Consider increasing window_size.")
        logger.info(f"Data prepared for PatchTST model with patch_len={patch_len}")
        
    elif model_name == 'tst':
        # TST works with standard time series format
        logger.info("Data prepared for TST (Time Series Transformer) model")
        
    elif model_name == 'informer':
        # Informer is designed for longer sequences
        if window_size < 20:
            logger.warning(f"Informer typically works better with longer sequences. Current window_size: {window_size}")
        logger.info("Data prepared for Informer model")
        
    elif model_name in ['tabnet', 'tabnetregressor']:
        # TabNet typically works on flattened tabular data rather than sequences
        logger.info("Data prepared for TabNet model")
        # Note: TabNet in our implementation adapts to sequence format
        
    else:
        logger.warning(f"Unknown model name: {model_name}. Using standard data preparation.")
    
    return X_cat, X_cont, y, cat_mask, con_mask, cat_dims, glucose_scaler


def create_dataset_for_model(features: Tuple, targets: torch.Tensor, model_name: str) -> Dataset:
    """
    Create a PyTorch Dataset optimized for specific model architectures.
    
    Args:
        features: Tuple of feature tensors
        targets: Target tensor
        model_name: Name of the model
        
    Returns:
        PyTorch Dataset
    """
    model_name = model_name.lower()
    
    if model_name in ['qisn', 'saint', 'ft-transformer', 'fttransformer', 'tabpfn', 
                      'patchtst', 'tst', 'informer', 'tabnet', 'tabnetregressor']:
        # All these models use the same dataset format
        return GlucoseDataset(features, targets)
    else:
        logger.warning(f"Unknown model name: {model_name}. Using standard dataset.")
        return GlucoseDataset(features, targets)


def get_model_data_requirements(model_name: str) -> Dict[str, Any]:
    """
    Get data requirements and recommendations for specific models.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with data requirements and recommendations
    """
    model_name = model_name.lower()
    
    requirements = {
        'qisn': {
            'min_window_size': 6,
            'max_window_size': 100,
            'recommended_window_size': 12,
            'handles_missing_values': True,
            'handles_variable_length': True,
            'optimal_sequence_length': 'medium',
            'memory_efficiency': 'high'
        },
        'saint': {
            'min_window_size': 5,
            'max_window_size': 200,
            'recommended_window_size': 20,
            'handles_missing_values': True,
            'handles_variable_length': True,
            'optimal_sequence_length': 'medium_to_long',
            'memory_efficiency': 'medium'
        },
        'ft-transformer': {
            'min_window_size': 5,
            'max_window_size': 150,
            'recommended_window_size': 15,
            'handles_missing_values': True,
            'handles_variable_length': True,
            'optimal_sequence_length': 'medium',
            'memory_efficiency': 'medium'
        },
        'tabpfn': {
            'min_window_size': 3,
            'max_window_size': 50,
            'recommended_window_size': 10,
            'handles_missing_values': True,
            'handles_variable_length': False,
            'optimal_sequence_length': 'short',
            'memory_efficiency': 'high',
            'notes': 'Works best with smaller datasets due to meta-learning approach'
        },
        'patchtst': {
            'min_window_size': 6,
            'max_window_size': 500,
            'recommended_window_size': 48,
            'handles_missing_values': True,
            'handles_variable_length': True,
            'optimal_sequence_length': 'long',
            'memory_efficiency': 'high',
            'special_params': {'patch_len': 3, 'stride': 1}
        },
        'tst': {
            'min_window_size': 5,
            'max_window_size': 200,
            'recommended_window_size': 24,
            'handles_missing_values': True,
            'handles_variable_length': True,
            'optimal_sequence_length': 'medium_to_long',
            'memory_efficiency': 'medium'
        },
        'informer': {
            'min_window_size': 20,
            'max_window_size': 1000,
            'recommended_window_size': 96,
            'handles_missing_values': True,
            'handles_variable_length': True,
            'optimal_sequence_length': 'very_long',
            'memory_efficiency': 'high',
            'special_params': {'factor': 5}
        },
        'tabnet': {
            'min_window_size': 1,
            'max_window_size': 100,
            'recommended_window_size': 12,
            'handles_missing_values': True,
            'handles_variable_length': False,
            'optimal_sequence_length': 'short_to_medium',
            'memory_efficiency': 'high',
            'notes': 'Primarily designed for tabular data, adapted for sequences'
        }
    }
    
    return requirements.get(model_name, {
        'min_window_size': 5,
        'max_window_size': 100,
        'recommended_window_size': 12,
        'handles_missing_values': False,
        'handles_variable_length': False,
        'optimal_sequence_length': 'medium',
        'memory_efficiency': 'medium',
        'notes': 'Unknown model - using default recommendations'
    })