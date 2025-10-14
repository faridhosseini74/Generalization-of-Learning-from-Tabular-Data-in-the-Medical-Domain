import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from data_utils import prepare_glucose_data, prepare_data_for_model, detect_dataset_structure, load_dataset

logger = logging.getLogger(__name__)

class UniversalModelWrapper:
    """Universal wrapper to make any model compatible with SHAP."""
    
    def __init__(self, model, device, model_name='unknown'):
        self.model = model
        self.device = device
        self.model_name = model_name.lower()
        
        # Determine if model expects patient_mask
        self.expects_patient_mask = hasattr(model, 'forward') and 'patient_mask' in str(model.forward.__code__.co_varnames)
        
    def __call__(self, X):
        """Make predictions for SHAP analysis - handles all model types."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            
            X = X.to(self.device)
            
            # Handle traditional models separately
            if (hasattr(self.model, 'model') and hasattr(self.model.model, 'predict')) or \
               (hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'TraditionalModelWrapper'):
                # This is a traditional model wrapper (sklearn models)
                batch_size = X.shape[0]
                
                # For traditional models, we need to prepare features properly
                if len(X.shape) == 2:
                    # Assume X is flattened features
                    # Create dummy categorical and continuous features
                    # This is a simplified approach - in practice you'd want proper feature splitting
                    X_cat = torch.zeros((batch_size, 1, 0), dtype=torch.long, device=self.device)
                    X_cont = X.view(batch_size, 1, -1)
                    
                    # Use the model's forward method
                    predictions = self.model.forward(X_cat, X_cont)
                    return predictions.cpu().numpy()
                else:
                    # Handle 3D input
                    X_cat = torch.zeros((batch_size, X.shape[1], 0), dtype=torch.long, device=self.device) 
                    X_cont = X
                    
                    predictions = self.model.forward(X_cat, X_cont)
                    return predictions.cpu().numpy()
            
            # Handle transformer models
            if hasattr(self.model, '__class__') and self.model.__class__.__name__ in ['FTTransformer']:
                # FT-Transformer and similar models
                batch_size = X.shape[0]
                
                if len(X.shape) == 2:
                    # Add sequence dimension
                    X_cont = X.unsqueeze(1)
                else:
                    X_cont = X
                    
                # Create categorical features based on model's expected categories
                if hasattr(self.model, 'categories') and len(self.model.categories) > 0:
                    # Create dummy categorical data
                    num_cat = len(self.model.categories)
                    X_cat = torch.zeros((batch_size, X_cont.shape[1], num_cat), dtype=torch.long, device=self.device)
                else:
                    # No categorical features
                    X_cat = torch.zeros((batch_size, X_cont.shape[1], 0), dtype=torch.long, device=self.device)
                
                predictions = self.model(X_cat, X_cont)
                return predictions.cpu().numpy()
            
            # Handle different input formats based on model type
            batch_size = X.shape[0]
            
            if len(X.shape) == 2:
                # 2D input: [batch_size, features] - reshape for sequence models
                if self.model_name in ['patchtst', 'tst', 'informer']:
                    # These models expect longer sequences
                    seq_len = min(X.shape[1], 12)  # Use reasonable sequence length
                    X_cont = X.view(batch_size, seq_len, -1)
                elif self.model_name in ['tabnet', 'tabnetregressor']:
                    # TabNet works with flattened features
                    X_cont = X.unsqueeze(1)  # Add sequence dimension
                else:
                    # Default: add sequence dimension
                    X_cont = X.unsqueeze(1)
                    
                X_cat = torch.zeros((batch_size, X_cont.shape[1], 0), dtype=torch.long, device=self.device)
                
            elif len(X.shape) == 3:
                # 3D input: [batch_size, seq_len, features] - already in sequence format
                X_cont = X
                X_cat = torch.zeros((batch_size, X_cont.shape[1], 0), dtype=torch.long, device=self.device)
            else:
                raise ValueError(f"Unsupported input shape: {X.shape}")
            
            # Create masks
            cat_mask = torch.ones_like(X_cat, dtype=torch.bool)
            cont_mask = torch.ones_like(X_cont, dtype=torch.bool)
            
            # Call model with appropriate arguments
            try:
                if self.expects_patient_mask:
                    # Models that expect patient_mask
                    patient_mask = torch.ones((batch_size, X_cont.shape[1]), dtype=torch.bool, device=self.device)
                    output = self.model(X_cat, X_cont, cat_mask, cont_mask, patient_mask)
                else:
                    # Standard models
                    output = self.model(X_cat, X_cont, cat_mask, cont_mask)
                    
                return output.cpu().numpy()
                
            except Exception as e:
                logger.warning(f"Model call failed with standard arguments, trying simplified call: {e}")
                # Fallback: try with minimal arguments
                try:
                    output = self.model(X_cat, X_cont)
                    return output.cpu().numpy()
                except Exception as e2:
                    logger.error(f"All model call attempts failed: {e2}")
                    # Final fallback: return dummy predictions
                    return np.zeros((batch_size, 1))


# Keep backward compatibility
ModelWrapper = UniversalModelWrapper

class UniversalSHAPAnalyzer:
    """Universal SHAP analysis for all glucose prediction models."""
    
    def __init__(self, model_path: str, data_path: str, model_name: str = None, device: int = 0):
        """
        Initialize SHAP analyzer.
        
        Args:
            model_path: Path to trained model
            data_path: Path to dataset
            model_name: Model architecture name (auto-detect if None)
            device: GPU device ID
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        self.model = None
        self.model_wrapper = None
        self.shap_values = None
        self.feature_names = []
        self.background_data = None
        self.test_data = None
        
    def load_model_and_data(self, n_background: int = 1000, n_test: int = 1000, 
                           window_size: int = 12, prediction_horizon: int = 3,
                           patient_col: Optional[str] = None, time_col: Optional[str] = None,
                           glucose_col: Optional[str] = None):
        """Load model and prepare data for SHAP analysis."""
        logger.info("Loading model...")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model information from checkpoint
        model_config = checkpoint.get('model_config', {})
        cat_dims = checkpoint.get('cat_dims', [])
        model_name = self.model_name or checkpoint.get('model_name', 'qisn')
        
        logger.info(f"Detected model: {model_name}")
        
        # Import model factory
        try:
            from model import ModelFactory
            
            # Create model using the factory
            self.model = ModelFactory.create_model(
                model_name=model_name,
                categories=cat_dims,
                num_continuous=model_config.get('num_continuous', 6),
                config=model_config
            )
            
        except ImportError:
            logger.warning("Model factory not available, falling back to QISN")
            from model import QISNForGlucose
            self.model = QISNForGlucose(
                categories=cat_dims,
                num_continuous=model_config.get('num_continuous', 6),
                **model_config
            )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Loading and preprocessing data...")
        
        # Load and preprocess data using model-specific preparation
        try:
            from data_utils import prepare_data_for_model
            X_cat, X_cont, y, cat_mask, cont_mask, cat_dims_data, scaler = prepare_data_for_model(
                file_path=self.data_path,
                model_name=model_name,
                window_size=window_size,
                prediction_horizon=prediction_horizon,
                patient_col=patient_col,
                time_col=time_col,
                glucose_col=glucose_col,
                use_cache=False  # Don't use cache for SHAP analysis
            )
        except ImportError:
            # Fallback to standard data preparation
            from data_utils import prepare_glucose_data
            if isinstance(self.data_path, Path):
                # Convert to dataset key for backward compatibility
                dataset_key = self.data_path.stem
            else:
                dataset_key = self.data_path
            X_cat, X_cont, y, cat_mask, cont_mask, cat_dims_data, scaler = prepare_glucose_data(
                dataset_key, window_size=window_size, prediction_horizon=prediction_horizon
            )
        
        # Prepare data for SHAP
        self._prepare_shap_data(X_cont, n_background, n_test)
        
        # Create model wrapper
        self.model_wrapper = UniversalModelWrapper(self.model, self.device, model_name)
        
        # Generate feature names
        self._generate_feature_names(X_cont.shape[-1], window_size)
        
        logger.info(f"Model and data loaded successfully. Background samples: {len(self.background_data)}, Test samples: {len(self.test_data)}")
        
    def _prepare_shap_data(self, X_cont: torch.Tensor, n_background: int, n_test: int):
        """Prepare background and test data for SHAP analysis."""
        # Convert to numpy and reshape for SHAP
        X_numpy = X_cont.cpu().numpy()
        
        # Flatten sequences for SHAP analysis (SHAP works better with 2D data)
        if len(X_numpy.shape) == 3:
            batch_size, seq_len, n_features = X_numpy.shape
            X_flat = X_numpy.reshape(batch_size, seq_len * n_features)
        else:
            X_flat = X_numpy
        
        # Sample background and test data
        total_samples = len(X_flat)
        indices = np.random.choice(total_samples, min(n_background + n_test, total_samples), replace=False)
        
        self.background_data = X_flat[indices[:n_background]]
        self.test_data = X_flat[indices[n_background:n_background + n_test]]
        
        logger.info(f"Prepared SHAP data - Background: {self.background_data.shape}, Test: {self.test_data.shape}")
        
    def _generate_feature_names(self, n_features: int, window_size: int):
        """Generate meaningful feature names for visualization."""
        base_features = [
            'glucose_t-1', 'glucose_t-2', 'glucose_t-3',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'meal_indicator', 'exercise_indicator', 'sleep_indicator'
        ]
        
        # Extend or trim base features to match actual number
        if n_features <= len(base_features):
            base_features = base_features[:n_features]
        else:
            # Add generic feature names
            for i in range(len(base_features), n_features):
                base_features.append(f'feature_{i}')
        
        # Generate names for each timestep
        self.feature_names = []
        for t in range(window_size):
            for feature in base_features:
                self.feature_names.append(f'{feature}_t{t}')
                
        # Trim to match actual flattened feature count
        expected_features = len(self.test_data[0]) if len(self.test_data) > 0 else window_size * n_features
        self.feature_names = self.feature_names[:expected_features]
        
        # Fill any missing names
        while len(self.feature_names) < expected_features:
            self.feature_names.append(f'feature_{len(self.feature_names)}')

    def compute_shap_values(self, max_evals: int = 2000):
        """Compute SHAP values for the test data."""
        if self.model_wrapper is None:
            raise ValueError("Model not loaded. Call load_model_and_data() first.")
            
        logger.info("Computing SHAP values...")
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(self.model_wrapper, self.background_data)
        
        # Compute SHAP values
        self.shap_values = explainer.shap_values(
            self.test_data, 
            nsamples=max_evals,
            silent=False
        )
        
        logger.info(f"Computed SHAP values for {len(self.test_data)} samples")
        
    def plot_feature_importance(self, output_dir: str, top_k: int = 20):
        """Plot feature importance based on SHAP values."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Get top k features
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]
        top_values = mean_shap[top_indices]
        top_names = [self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}' 
                    for i in top_indices]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_values)), top_values)
        plt.yticks(range(len(top_values)), top_names)
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title(f'Top {top_k} Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_dir / 'feature_importance.png'}")
        
    def plot_summary(self, output_dir: str):
        """Plot SHAP summary plot."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        
        # Prepare feature names (limit to reasonable number for visibility)
        feature_names_display = self.feature_names[:min(len(self.feature_names), 30)]
        shap_values_display = self.shap_values[:, :len(feature_names_display)]
        test_data_display = self.test_data[:, :len(feature_names_display)]
        
        shap.summary_plot(
            shap_values_display, 
            test_data_display,
            feature_names=feature_names_display,
            show=False
        )
        
        # Save plot
        plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {output_dir / 'shap_summary.png'}")
        
    def save_feature_importance_csv(self, output_dir: str):
        """Save feature importance rankings to CSV."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        std_shap = np.std(np.abs(self.shap_values), axis=0)
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'feature_name': [self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}' 
                           for i in range(len(mean_shap))],
            'mean_abs_shap': mean_shap,
            'std_abs_shap': std_shap,
            'importance_rank': range(1, len(mean_shap) + 1)
        })
        
        # Sort by importance
        results_df = results_df.sort_values('mean_abs_shap', ascending=False)
        results_df['importance_rank'] = range(1, len(results_df) + 1)
        
        # Save to CSV
        csv_path = output_dir / 'feature_importance.csv'
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"Feature importance CSV saved to {csv_path}")
        return results_df


# Keep backward compatibility
SHAPAnalyzer = UniversalSHAPAnalyzer


def analyze_traditional_model_with_shap(
    model_path: str,
    data_path: str, 
    output_dir: str,
    model_name: str,
    n_background: int = 1000,
    n_test: int = 1000,
    max_evals: int = 2000,
    window_size: int = 12,
    prediction_horizon: int = 3,
    patient_col: Optional[str] = None,
    time_col: Optional[str] = None,
    glucose_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Specialized SHAP analysis for traditional models (sklearn-based).
    
    Args:
        model_path: Path to trained traditional model (joblib file)
        data_path: Path to dataset file  
        output_dir: Directory to save SHAP analysis results
        model_name: Traditional model name (randomforest, xgboost, etc.)
        n_background: Number of background samples for SHAP
        n_test: Number of test samples for SHAP analysis
        max_evals: Maximum SHAP evaluations
        window_size: Input sequence window size
        prediction_horizon: Prediction horizon
        patient_col: Patient ID column name (auto-detect if None)
        time_col: Time column name (auto-detect if None)
        glucose_col: Glucose column name (auto-detect if None)
        
    Returns:
        Feature importance DataFrame
    """
    
    logger.info(f"Starting SHAP analysis for traditional model: {model_name}")
    
    try:
        import joblib
        
        # Load the traditional model
        model = joblib.load(model_path)
        logger.info(f"Loaded traditional model from {model_path}")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        
        # Load dataset
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            # Try using the data_utils function
            data, _ = load_dataset(data_path)
        
        # Auto-detect columns if not provided
        if not all([patient_col, time_col, glucose_col]):
            detected = detect_dataset_structure(data)
            patient_col = patient_col or detected['patient_col']
            time_col = time_col or detected['time_col'] 
            glucose_col = glucose_col or detected['glucose_col']
        
        # Prepare data using the main data utilities
        from data_utils import prepare_glucose_data, prepare_data_for_model
        
        # Get data in the format expected by traditional models
        cat_data, cont_data, cat_dims, num_continuous = prepare_data_for_model(
            data, patient_col, time_col, glucose_col, window_size, prediction_horizon
        )
        
        # For traditional models, we need flattened features
        n_samples = len(cat_data)
        features = []
        feature_names = []
        
        for i, sample in enumerate(cat_data):
            if i >= n_test + n_background:  # Limit samples for efficiency
                break
                
            # Flatten categorical features
            if sample['cat_features'].numel() > 0:
                cat_flat = sample['cat_features'].view(-1).numpy()
                if i == 0:  # Only add feature names once
                    for j in range(len(cat_flat)):
                        feature_names.append(f'cat_feature_{j}')
            else:
                cat_flat = np.array([])
                
            # Flatten continuous features  
            if sample['cont_features'].numel() > 0:
                cont_flat = sample['cont_features'].view(-1).numpy()
                if i == 0:
                    for j in range(len(cont_flat)):
                        feature_names.append(f'cont_feature_{j}')
            else:
                cont_flat = np.array([])
                
            # Flatten statistics if present
            if 'statistics' in sample and sample['statistics'].numel() > 0:
                stats_flat = sample['statistics'].view(-1).numpy()
                if i == 0:
                    for j in range(len(stats_flat)):
                        feature_names.append(f'stat_feature_{j}')
            else:
                stats_flat = np.array([])
            
            # Combine all features
            feature_parts = [part for part in [cat_flat, cont_flat, stats_flat] if len(part) > 0]
            if feature_parts:
                combined_features = np.concatenate(feature_parts)
            else:
                combined_features = np.array([0.0])  # Fallback
                
            features.append(combined_features)
        
        # Convert to numpy array
        X_combined = np.array(features)
        
        # Handle NaN and remove constant features
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=1.0, neginf=0.0)
        if X_combined.shape[1] > 1:
            feature_variance = np.var(X_combined, axis=0)
            non_constant_mask = feature_variance > 1e-8
            if np.any(non_constant_mask):
                X_combined = X_combined[:, non_constant_mask]
                feature_names = [name for i, name in enumerate(feature_names) if non_constant_mask[i]]
        
        logger.info(f"Prepared {X_combined.shape[0]} samples with {X_combined.shape[1]} features")
        
        # Split data for SHAP
        n_samples = X_combined.shape[0]
        background_end = min(n_background, n_samples//2)
        test_end = min(background_end + n_test, n_samples)
        
        X_background = X_combined[:background_end]
        X_test = X_combined[background_end:test_end]
        
        logger.info(f"Using {len(X_background)} background samples and {len(X_test)} test samples")
        
        # Choose appropriate SHAP explainer based on model type
        model_name_lower = model_name.lower()
        
        if model_name_lower in ['randomforest', 'rf']:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        elif model_name_lower in ['xgboost', 'xgb', 'lightgbm', 'lgb', 'catboost', 'cat']:
            # Use TreeExplainer for gradient boosting models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # Use KernelExplainer for other models (MLP, etc.)
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_values = explainer.shap_values(X_test, nsamples=min(max_evals, 1000))
        
        logger.info("SHAP values computed successfully")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        logger.info("Generating SHAP plots...")
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        results_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Save results
        results_df.to_csv(output_dir / f'{model_name}_shap_feature_importance.csv', index=False)
        
        logger.info(f"SHAP analysis completed for {model_name}. Results saved to {output_dir}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error in traditional model SHAP analysis: {e}")
        raise


def run_shap_analysis(model_path: str, data_path: str, output_dir: str,
                     model_name: str = None, n_background: int = 1000, 
                     n_test: int = 1000, max_evals: int = 2000,
                     window_size: int = 12, prediction_horizon: int = 3,
                     patient_col: Optional[str] = None, time_col: Optional[str] = None,
                     glucose_col: Optional[str] = None, device: int = 0):
    """
    Run complete SHAP analysis for any model architecture.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to dataset file
        output_dir: Directory to save SHAP analysis results
        model_name: Model architecture name (auto-detect if None)
        n_background: Number of background samples for SHAP
        n_test: Number of test samples for SHAP analysis
        max_evals: Maximum SHAP evaluations
        window_size: Input sequence window size
        prediction_horizon: Prediction horizon
        patient_col: Patient ID column name (auto-detect if None)
        time_col: Time column name (auto-detect if None)
        glucose_col: Glucose column name (auto-detect if None)
        device: GPU device ID
        
    Returns:
        Feature importance DataFrame
    """
    
    logger.info("Starting SHAP analysis...")
    
    # Create analyzer
    analyzer = UniversalSHAPAnalyzer(model_path, data_path, model_name, device)
    
    # Load model and data
    analyzer.load_model_and_data(
        n_background=n_background,
        n_test=n_test,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        patient_col=patient_col,
        time_col=time_col,
        glucose_col=glucose_col
    )
    
    # Compute SHAP values
    analyzer.compute_shap_values(max_evals)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots and save results
    analyzer.plot_feature_importance(output_dir)
    analyzer.plot_summary(output_dir)
    results_df = analyzer.save_feature_importance_csv(output_dir)
    
    logger.info(f"SHAP analysis completed. Results saved to {output_dir}")
    
    return results_df
