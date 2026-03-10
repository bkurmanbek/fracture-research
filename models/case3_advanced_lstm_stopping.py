#!/usr/bin/env python
# coding: utf-8

# # Case 3: Advanced LSTM with Stopping Prediction
# 
# This notebook implements the **Advanced Multi-head LSTM with Natural Stopping Criteria** model described in Section 4.4 of the paper.
# 
# ## Model Architecture Overview
# 
# The architecture includes:
# 1. **Input Projection Layer**: Maps input features to hidden dimension
# 2. **BiLSTM Stack**: 2 layers with hidden dimensions [256, 128]
# 3. **Additive Attention Mechanism**: Focuses on relevant historical points
# 4. **Dual Output Heads**:
#    - **Coordinate Head**: Predicts next (x, y) coordinates
#    - **Stopping Head**: Predicts probability of fracture termination
# 5. **Combined Loss**: MSE for coordinates + BCE for stopping (lambda_stop = 1.0)
# 
# The key innovation is the natural stopping criteria that allows the model to learn when to terminate path generation.

# ## 1. Import Libraries and Setup

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
import os
import sys

# Add workspace to path to import visualization_utils
sys.path.append('/workspace')
try:
    from visualization_utils import plot_fracture_generation_comparison, plot_generation_progression
    VISUALIZATION_UTILS_AVAILABLE = True
except ImportError:
    VISUALIZATION_UTILS_AVAILABLE = False
    print("Warning: visualization_utils not found. Enhanced plotting will be disabled.")

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ADVANCED GENERATION UTILITIES (from fracture_sept.ipynb)
# Integrated directly for self-contained execution
# ============================================================================

from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist

class TrainingDistributionAnalyzer:
    """Analyze training data distributions and compute statistics for generation constraints."""
    
    def __init__(self):
        self.stats = {}
        
    def analyze_training_data(self, train_fractures: List[Dict]) -> Dict:
        """
        Analyze training data to extract distribution statistics.
        
        Args:
            train_fractures: List of fracture dictionaries with 'points' arrays
            
        Returns:
            Dictionary with distribution statistics
        """
        all_segment_lengths = []
        all_angles = []
        all_coords = []
        all_path_lengths = []
        all_curvatures = []
        
        for fracture in train_fractures:
            points = fracture['points']
            if len(points) < 2:
                continue
                
            # Collect coordinates
            all_coords.append(points)
            
            # Compute segment lengths
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            all_segment_lengths.extend(lengths.tolist())
            
            # Compute angles and curvatures
            for i in range(len(points) - 1):
                dx = points[i+1, 0] - points[i, 0]
                dy = points[i+1, 1] - points[i, 1]
                if np.linalg.norm([dx, dy]) > 1e-6:
                    angle = np.arctan2(dy, dx)
                    all_angles.append(angle)
            
            # Compute curvature (angle changes)
            if len(points) > 2:
                for i in range(1, len(points) - 1):
                    v1 = points[i] - points[i-1]
                    v2 = points[i+1] - points[i]
                    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                        cos_theta = np.clip(cos_theta, -1, 1)
                        theta = np.arccos(cos_theta)
                        all_curvatures.append(theta)
            
            # Path length
            path_length = np.sum(lengths)
            all_path_lengths.append(path_length)
        
        # Compute statistics
        all_coords_array = np.vstack(all_coords) if all_coords else np.array([]).reshape(0, 2)
        
        stats = {
            # Segment length statistics
            'segment_length': {
                'mean': np.mean(all_segment_lengths) if all_segment_lengths else 0.0,
                'std': np.std(all_segment_lengths) if all_segment_lengths else 1.0,
                'median': np.median(all_segment_lengths) if all_segment_lengths else 0.0,
                'min': np.min(all_segment_lengths) if all_segment_lengths else 0.0,
                'max': np.max(all_segment_lengths) if all_segment_lengths else 1.0,
                'q25': np.percentile(all_segment_lengths, 25) if all_segment_lengths else 0.0,
                'q75': np.percentile(all_segment_lengths, 75) if all_segment_lengths else 1.0,
            },
            
            # Angle statistics
            'angle': {
                'mean': np.mean(all_angles) if all_angles else 0.0,
                'std': np.std(all_angles) if all_angles else np.pi,
                'min': np.min(all_angles) if all_angles else -np.pi,
                'max': np.max(all_angles) if all_angles else np.pi,
            },
            
            # Curvature statistics
            'curvature': {
                'mean': np.mean(all_curvatures) if all_curvatures else 0.0,
                'std': np.std(all_curvatures) if all_curvatures else 0.1,
                'max': np.max(all_curvatures) if all_curvatures else np.pi,
            },
            
            # Coordinate bounds
            'coordinate_bounds': {
                'x_min': np.min(all_coords_array[:, 0]) if len(all_coords_array) > 0 else -1000.0,
                'x_max': np.max(all_coords_array[:, 0]) if len(all_coords_array) > 0 else 1000.0,
                'y_min': np.min(all_coords_array[:, 1]) if len(all_coords_array) > 0 else -1000.0,
                'y_max': np.max(all_coords_array[:, 1]) if len(all_coords_array) > 0 else 1000.0,
            },
            
            # Path length statistics
            'path_length': {
                'mean': np.mean(all_path_lengths) if all_path_lengths else 0.0,
                'std': np.std(all_path_lengths) if all_path_lengths else 1.0,
                'median': np.median(all_path_lengths) if all_path_lengths else 0.0,
                'max': np.max(all_path_lengths) if all_path_lengths else 1.0,
            },
        }
        
        self.stats = stats
        return stats
    
    def get_reasonable_bounds(self, seed_points: np.ndarray, expansion_factor: float = 3.0) -> Dict:
        """
        Get reasonable bounds for generation based on seed points and training statistics.
        
        Args:
            seed_points: Seed points for the fracture
            expansion_factor: How many standard deviations to expand beyond seed bounds
            
        Returns:
            Dictionary with bounds
        """
        if len(seed_points) == 0:
            # Use training data bounds
            return {
                'x_min': self.stats['coordinate_bounds']['x_min'],
                'x_max': self.stats['coordinate_bounds']['x_max'],
                'y_min': self.stats['coordinate_bounds']['y_min'],
                'y_max': self.stats['coordinate_bounds']['y_max'],
            }
        
        # Compute bounds from seed points
        seed_x_min, seed_x_max = np.min(seed_points[:, 0]), np.max(seed_points[:, 0])
        seed_y_min, seed_y_max = np.min(seed_points[:, 1]), np.max(seed_points[:, 1])
        
        # Expand based on typical segment length and path length
        typical_length = self.stats['segment_length']['mean']
        max_expected_length = self.stats['path_length']['max']
        
        # Expand bounds - use larger of typical expansion or half max path length
        expansion = max(typical_length * expansion_factor, max_expected_length * 0.5)
        
        return {
            'x_min': seed_x_min - expansion,
            'x_max': seed_x_max + expansion,
            'y_min': seed_y_min - expansion,
            'y_max': seed_y_max + expansion,
        }


class AdvancedStoppingCriteria:
    """Advanced stopping criteria for fracture path generation."""
    
    def __init__(self, training_stats: Dict, config: Dict):
        self.stats = training_stats
        self.config = config
        
        # Default thresholds
        self.max_deviation_factor = config.get('max_deviation_factor', 3.0)  # std deviations
        self.oscillation_threshold = config.get('oscillation_threshold', 0.1)
        self.stagnation_limit = config.get('stagnation_limit', 5)
        self.movement_threshold = config.get('movement_threshold', 0.01)
        self.max_distance_from_seed = config.get('max_distance_from_seed', None)  # None = auto
        
        # Auto-compute max distance from seed if not provided
        if self.max_distance_from_seed is None:
            self.max_distance_from_seed = training_stats.get('path_length', {}).get('max', 1000.0) * 1.5
        
    def is_outside_reasonable_bounds(self, point: np.ndarray, seed_points: np.ndarray, 
                                     bounds: Optional[Dict] = None) -> bool:
        """
        Check if point is outside reasonable bounds based on training distribution.
        
        Args:
            point: Current point to check
            seed_points: Seed points for the fracture
            bounds: Optional precomputed bounds
            
        Returns:
            True if point is outside reasonable bounds
        """
        if bounds is None:
            analyzer = TrainingDistributionAnalyzer()
            analyzer.stats = self.stats
            bounds = analyzer.get_reasonable_bounds(seed_points)
        
        # Check bounds
        if (point[0] < bounds['x_min'] or point[0] > bounds['x_max'] or
            point[1] < bounds['y_min'] or point[1] > bounds['y_max']):
            return True
        
        # Check distance from seed centroid
        if len(seed_points) > 0:
            seed_centroid = np.mean(seed_points, axis=0)
            distance_from_seed = np.linalg.norm(point - seed_centroid)
            if distance_from_seed > self.max_distance_from_seed:
                return True
        
        return False
    
    def detect_oscillation(self, recent_points: List[np.ndarray]) -> bool:
        """
        Detect if path is oscillating (going back and forth).
        
        Args:
            recent_points: List of recent points (last 5-6 points)
            
        Returns:
            True if oscillation detected
        """
        if len(recent_points) < 4:
            return False
        
        recent_points = np.array(recent_points)
        
        # Compute direction vectors
        directions = np.diff(recent_points, axis=0)
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)
        
        # Check for alternating directions (oscillation)
        if len(directions) >= 3:
            # Compute dot products between consecutive directions
            dots = []
            for i in range(len(directions) - 1):
                dot = np.dot(directions[i], directions[i+1])
                dots.append(dot)
            
            # If we have alternating negative/positive dots, it's oscillating
            if len(dots) >= 2:
                # Check if directions are reversing
                negative_count = sum(1 for d in dots if d < -self.oscillation_threshold)
                if negative_count >= len(dots) * 0.5:  # More than half are negative
                    return True
        
        return False
    
    def check_stagnation(self, recent_movements: List[float]) -> bool:
        """
        Check if path generation has stagnated (very small movements).
        
        Args:
            recent_movements: List of recent movement distances
            
        Returns:
            True if stagnation detected
        """
        if len(recent_movements) < self.stagnation_limit:
            return False
        
        avg_movement = np.mean(recent_movements[-self.stagnation_limit:])
        return avg_movement < self.movement_threshold
    
    def check_excessive_deviation(self, current_path: np.ndarray, seed_points: np.ndarray,
                                  true_path: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Check if current path has deviated excessively from expected trajectory.
        
        Args:
            current_path: Current generated path
            seed_points: Seed points
            true_path: Optional true path for comparison (if available during training)
            
        Returns:
            Tuple of (should_stop, deviation_metric)
        """
        if len(current_path) < 10:
            return False, 0.0
        
        # Compute path length
        path_length = np.sum(np.linalg.norm(np.diff(current_path, axis=0), axis=1))
        
        # Check if path length exceeds reasonable bounds
        expected_max_length = self.stats.get('path_length', {}).get('max', 1000.0)
        expected_mean_length = self.stats.get('path_length', {}).get('mean', 100.0)
        expected_std_length = self.stats.get('path_length', {}).get('std', 50.0)
        
        # Z-score for path length
        z_score = (path_length - expected_mean_length) / (expected_std_length + 1e-6)
        
        if z_score > self.max_deviation_factor:
            return True, z_score
        
        # If true path available, check distance from true path
        if true_path is not None and len(true_path) > 0:
            # Compute minimum distance from current point to true path
            current_point = current_path[-1]
            distances = np.linalg.norm(true_path - current_point, axis=1)
            min_distance = np.min(distances)
            
            # Get typical segment length
            typical_segment = self.stats.get('segment_length', {}).get('mean', 1.0)
            
            # If too far from true path (more than 5x typical segment)
            if min_distance > typical_segment * 5.0:
                return True, min_distance / typical_segment
        
        return False, 0.0


def apply_distribution_constraints(sampled_value: float, distribution_stats: Dict,
                                   constraint_type: str = 'clip') -> float:
    """
    Apply distribution constraints to sampled values.

    Args:
        sampled_value: Value sampled from model
        distribution_stats: Statistics from training distribution
        constraint_type: 'clip' (hard bounds) or 'soft' (probabilistic adjustment)

    Returns:
        Constrained value
    """
    if constraint_type == 'clip':
        # Hard clipping to reasonable range (use IQR: Q25 to Q75, or mean ± 2std)
        q25 = distribution_stats.get('q25', None)
        q75 = distribution_stats.get('q75', None)

        if q25 is not None and q75 is not None:
            return np.clip(sampled_value, q25, q75)
        else:
            # Fallback to mean ± 2std
            mean = distribution_stats.get('mean', 0.0)
            std = distribution_stats.get('std', 1.0)
            return np.clip(sampled_value, mean - 2*std, mean + 2*std)

    elif constraint_type == 'soft':
        # Soft constraint: adjust towards mean if too far
        mean = distribution_stats.get('mean', 0.0)
        std = distribution_stats.get('std', 1.0)

        # If more than 2 std away, pull towards mean
        z_score = (sampled_value - mean) / (std + 1e-6)
        if abs(z_score) > 2.0:
            # Soft pull: move 50% towards mean
            return mean + np.sign(z_score) * 2.0 * std * 0.5 + (sampled_value - mean) * 0.5
        return sampled_value

    return sampled_value


class NaturalStoppingPathGenerator:
    """
    Advanced path generator with natural stopping criteria based on fracture_sept.ipynb.
    This integrates multiple stopping mechanisms including model predictions,
    oscillation detection, stagnation, and boundary checking.
    """

    def __init__(self, model, data_loader, training_stats: Dict, config: Optional[Dict] = None):
        """
        Initialize the natural stopping path generator.

        Args:
            model: Trained model (should have 'coordinate_output' and optionally 'stopping_output')
            data_loader: Data loader with scalers
            training_stats: Training distribution statistics
            config: Optional configuration dictionary
        """
        self.model = model
        self.data_loader = data_loader
        self.training_stats = training_stats

        # Configuration
        default_config = {
            'stopping_threshold': 0.5,
            'confidence_threshold': 0.7,
            'max_consecutive_high_stop': 3,
            'movement_threshold': 0.1,
            'stagnation_limit': 5,
            'max_generation_length': 200,
            'min_generation_length': 3,
            'max_deviation_factor': 3.0,
            'oscillation_threshold': 0.1,
        }

        if config:
            default_config.update(config)
        self.config = default_config

        # Initialize stopping criteria
        self.stopping_criteria = AdvancedStoppingCriteria(training_stats, default_config)

    def generate_path(self, fracture_data: Dict, start_point: Optional[np.ndarray] = None) -> Dict:
        """
        Generate fracture path with natural stopping.

        Args:
            fracture_data: Dictionary with fracture information (including 'points')
            start_point: Optional start point (defaults to first point of fracture)

        Returns:
            Dictionary with generation results
        """
        if start_point is None:
            start_point = fracture_data['points'][0]

        actual_points = fracture_data['points']

        # Generate with natural stopping
        generated_points, stopping_info = self._generate_with_natural_stopping(
            start_point, fracture_data
        )

        # Combine start point with generated points
        complete_path = np.vstack([start_point.reshape(1, -1), generated_points]) if len(generated_points) > 0 else start_point.reshape(1, -1)

        return {
            'fracture_id': fracture_data.get('id', 'unknown'),
            'start_point': start_point,
            'actual_path': actual_points,
            'generated_path': complete_path,
            'stopping_info': stopping_info,
            'n_actual_points': len(actual_points),
            'n_generated_points': len(complete_path),
        }

    def _generate_with_natural_stopping(self, start_point: np.ndarray,
                                        fracture_data: Dict) -> Tuple[np.ndarray, Dict]:
        """Generate path with natural stopping criteria."""
        current_path = [start_point]
        generated_points = []

        # Tracking
        consecutive_high_stop = 0
        recent_movements = []
        stopping_probabilities = []
        stop_reason = "max_length_reached"

        max_len = self.config['max_generation_length']
        min_len = self.config['min_generation_length']

        for step in range(max_len):
            # Prepare sequence
            sequence_length = getattr(self.data_loader, 'sequence_length', 10)
            if len(current_path) >= sequence_length:
                sequence = np.array(current_path[-sequence_length:])
            else:
                # Pad with start point
                sequence = np.array(current_path)
                padding_needed = sequence_length - len(sequence)
                padding = np.tile(start_point.reshape(1, -1), (padding_needed, 1))
                sequence = np.vstack([padding, sequence])

            # Scale sequence
            sequence_scaled = self.data_loader.coord_scaler.transform(sequence)
            seq_input = sequence_scaled.reshape(1, sequence_length, 2)

            # Prepare features (simplified - adapt based on your data loader)
            features = self._create_generation_features(current_path, fracture_data, sequence_length)

            # Get model predictions
            try:
                predictions = self.model.predict([seq_input, features], verbose=0)

                # Handle different model output formats
                if isinstance(predictions, dict):
                    next_point_scaled = predictions['coordinate_output'][0]
                    stopping_prob = predictions.get('stopping_output', [[0.5]])[0][0] if 'stopping_output' in predictions else 0.5
                elif isinstance(predictions, (list, tuple)):
                    next_point_scaled = predictions[0][0] if len(predictions) > 0 else np.zeros(2)
                    stopping_prob = predictions[1][0][0] if len(predictions) > 1 else 0.5
                else:
                    next_point_scaled = predictions[0] if len(predictions.shape) > 1 else predictions
                    stopping_prob = 0.5

                stopping_probabilities.append(stopping_prob)

                # Convert to original scale
                next_point = self.data_loader.coord_scaler.inverse_transform(
                    next_point_scaled.reshape(1, -1)
                )[0]

                # Calculate movement
                movement_distance = np.linalg.norm(next_point - current_path[-1])
                recent_movements.append(movement_distance)
                if len(recent_movements) > self.config['stagnation_limit']:
                    recent_movements.pop(0)

                # STOPPING CRITERIA
                should_stop = False

                # 1. Model stopping prediction
                if stopping_prob > self.config['stopping_threshold']:
                    consecutive_high_stop += 1
                    if consecutive_high_stop >= self.config['max_consecutive_high_stop']:
                        should_stop = True
                        stop_reason = "model_stopping_prediction"
                else:
                    consecutive_high_stop = 0

                # 2. Stagnation detection
                if self.stopping_criteria.check_stagnation(recent_movements):
                    should_stop = True
                    stop_reason = "stagnation_detected"

                # 3. Oscillation detection
                if len(current_path) >= 5:
                    if self.stopping_criteria.detect_oscillation(current_path[-5:]):
                        should_stop = True
                        stop_reason = "oscillation_detected"

                # 4. Boundary detection
                if self.stopping_criteria.is_outside_reasonable_bounds(next_point, np.array(current_path[:5])):
                    should_stop = True
                    stop_reason = "boundary_exceeded"

                # 5. Excessive deviation from true path (if available during evaluation)
                actual_path = fracture_data.get('points', None)
                if actual_path is not None:
                    should_stop_dev, deviation = self.stopping_criteria.check_excessive_deviation(
                        np.array(current_path), np.array(current_path[:5]), actual_path
                    )
                    if should_stop_dev:
                        should_stop = True
                        stop_reason = f"excessive_deviation_{deviation:.2f}"

                # Apply stopping decision
                if should_stop and step >= min_len:
                    break

                # Add point to path
                generated_points.append(next_point)
                current_path.append(next_point)

            except Exception as e:
                print(f"    Error during generation at step {step}: {e}")
                stop_reason = f"generation_error: {str(e)[:50]}"
                break

        stopping_info = {
            'stop_reason': stop_reason,
            'final_stopping_prob': stopping_probabilities[-1] if stopping_probabilities else 0.0,
            'avg_stopping_prob': np.mean(stopping_probabilities) if stopping_probabilities else 0.0,
            'n_steps': len(generated_points),
        }

        return np.array(generated_points) if generated_points else np.array([]).reshape(0, 2), stopping_info

    def _create_generation_features(self, current_path: List[np.ndarray],
                                    fracture_data: Dict, sequence_length: int) -> np.ndarray:
        """Create features for generation (simplified placeholder)."""
        # This is a simplified version - you should adapt based on your actual feature requirements
        feature_dim = getattr(self.data_loader, 'total_feature_dim', 28)

        # Create dummy features with proper shape
        features = np.zeros((1, sequence_length, feature_dim))

        return features


class AdvancedGenerationController:
    """
    Controller for managing advanced generation with distribution constraints.
    Combines NaturalStoppingPathGenerator with distribution-aware adjustments.
    """

    def __init__(self, model, data_loader, training_stats: Dict, config: Optional[Dict] = None):
        self.model = model
        self.data_loader = data_loader
        self.training_stats = training_stats
        self.config = config or {}

        # Initialize natural stopping generator
        self.path_generator = NaturalStoppingPathGenerator(
            model, data_loader, training_stats, config
        )

    def generate_fracture(self, fracture_data: Dict) -> Dict:
        """Generate a complete fracture path with all advanced features."""
        return self.path_generator.generate_path(fracture_data)

    def evaluate_on_test_set(self, test_fractures: List[Dict]) -> List[Dict]:
        """Evaluate generation on a test set."""
        results = []
        for frac in test_fractures:
            result = self.generate_fracture(frac)
            results.append(result)
        return results


# Alias for backward compatibility
TrainingStatisticsTracker = TrainingDistributionAnalyzer


def compute_training_distributions(train_fractures: List[Dict]) -> Dict:
    """
    Compute training distribution statistics.

    Args:
        train_fractures: List of fracture dictionaries

    Returns:
        Dictionary with distribution statistics
    """
    analyzer = TrainingDistributionAnalyzer()
    stats = analyzer.analyze_training_data(train_fractures)
    return stats


# Set random seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Check GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"GPU Device: {tf.config.list_physical_devices('GPU')}")

# Plotting configuration
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


# ## 2. Configuration and Hyperparameters

# In[ ]:


CONFIG = {
    # Data parameters
    'train_csv': 'train_fractures_processed.csv',
    'test_csv': 'test_fractures_processed.csv',
    'sequence_length': 10,  # Number of previous points to consider

    # Model architecture - ENHANCED FOR A40 GPU
    'input_dim': 10,  # 2 coords + 8 features
    'input_projection_dim': 128,
    'lstm_units': [512, 256, 128, 64],  # Deep pyramid architecture
    'dropout': 0.3,          # Increased for regularization
    'recurrent_dropout': 0.1, # Use software kernel to avoid CuDNN error
    'attention_dim': 128,
    'shared_dense_units': [256, 128],
    'coord_output_dim': 2,
    'stop_output_dim': 1,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 1e-3,
    'num_epochs': 50,
    'lambda_stop': 1.0,  # Weight for stopping loss

    # Callbacks
    'early_stopping_patience': 20,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,

    # Inference parameters
    'stop_threshold': 0.5,  # Threshold for stopping probability
    'consecutive_stop_steps': 3,  # Number of consecutive high stop probs to terminate
    'min_movement_threshold': 0.01,  # Stagnation detection
    'max_generation_steps': 200,

    # Output paths
    'model_save_path': 'fracture_results/case3/models/best_model.keras',
    'results_dir': 'fracture_results/case3',
    'plots_dir': 'fracture_results/case3/plots',
    'models_dir': 'fracture_results/case3/models',
}

# Create directories
for d in [CONFIG['results_dir'], CONFIG['plots_dir'], CONFIG['models_dir']]:
    os.makedirs(d, exist_ok=True)

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


# ## 3. Data Loading and Preprocessing

# In[ ]:


def load_and_prepare_data(csv_path):
    """Load and prepare fracture data."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"  Total points: {len(df)}")
    print(f"  Unique fractures: {df['fracture_id'].nunique()}")
    print(f"  Columns: {list(df.columns[:10])}...")
    
    return df

# Load data
train_df = load_and_prepare_data(CONFIG['train_csv'])
test_df = load_and_prepare_data(CONFIG['test_csv'])

print("\nSample training data:")
print(train_df.head())


# In[ ]:


# Helper functions
def angle_difference(a1, a2):
    """Compute smallest angle difference, wrapped to [-π, π]."""
    diff = a2 - a1
    return np.arctan2(np.sin(diff), np.cos(diff))

class FracturePreprocessor:
    """Preprocesses fracture data."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.stats = {}
        
    def compute_features(self):
        """Compute normalized features."""
        results = []
        
        for fid in self.df['fracture_id'].unique():
            frac = self.df[self.df['fracture_id'] == fid].sort_values('point_idx').reset_index(drop=True)
            n = len(frac)
            
            if n < 3:
                continue
            
            # Extract and normalize coordinates
            xs = frac['coord_x'].values
            ys = frac['coord_y'].values
            
            cx, cy = xs.mean(), ys.mean()
            xs_norm = xs - cx
            ys_norm = ys - cy
            
            lengths = np.sqrt(np.diff(xs_norm)**2 + np.diff(ys_norm)**2)
            scale = np.median(lengths) if len(lengths) > 0 else 1.0
            scale = scale if scale > 0 else 1.0
            
            xs_norm /= scale
            ys_norm /= scale
            
            # Recompute features
            lengths = np.sqrt(np.diff(xs_norm)**2 + np.diff(ys_norm)**2)
            angles = np.arctan2(np.diff(ys_norm), np.diff(xs_norm))
            
            delta_angles = np.zeros(n)
            if len(angles) > 1:
                for i in range(1, len(angles)):
                    delta_angles[i] = angle_difference(angles[i-1], angles[i])
            
            curvature_trajectory = np.zeros(n)
            if n > 3:
                for i in range(2, n-1):
                    curvature_trajectory[i] = delta_angles[i] - delta_angles[i-1]
            
            # Fracture statistics
            mean_curvature = np.abs(delta_angles).mean()
            length_variance = np.var(lengths) if len(lengths) > 0 else 0.0
            path_length = lengths.sum()
            endpoint_dist = np.sqrt((xs_norm[-1]-xs_norm[0])**2 + (ys_norm[-1]-ys_norm[0])**2)
            tortuosity = path_length / (endpoint_dist + 1e-6)
            
            for i in range(n):
                point = {
                    'fracture_id': fid,
                    'point_idx': i,
                    'coord_x_norm': xs_norm[i],
                    'coord_y_norm': ys_norm[i],
                    'scale': scale,
                    'centroid_x': cx,
                    'centroid_y': cy,
                }
                
                if i < n - 1:
                    point['delta_r'] = lengths[i]
                    point['delta_theta'] = angles[i]
                    point['log_delta_r'] = np.log(lengths[i] + 1e-6)
                    point['sin_theta'] = np.sin(angles[i])
                    point['cos_theta'] = np.cos(angles[i])
                else:
                    point['delta_r'] = 0.0
                    point['delta_theta'] = 0.0
                    point['log_delta_r'] = np.log(1e-6)
                    point['sin_theta'] = 0.0
                    point['cos_theta'] = 0.0
                
                point['delta_angle'] = delta_angles[i]
                point['curvature_trajectory'] = curvature_trajectory[i]
                point['mean_curvature'] = mean_curvature
                point['length_variance'] = length_variance
                point['tortuosity'] = tortuosity
                point['is_last'] = 1.0 if i == n - 1 else 0.0
                
                results.append(point)
        
        processed_df = pd.DataFrame(results)
        
        # Compute statistics
        valid_log_delta_r = processed_df[processed_df['delta_r'] > 0]['log_delta_r']
        self.stats['log_delta_r_mean'] = valid_log_delta_r.mean() if len(valid_log_delta_r) > 0 else 0.0
        self.stats['log_delta_r_std'] = valid_log_delta_r.std() if len(valid_log_delta_r) > 0 else 1.0
        
        return processed_df
    
    def normalize_features(self, df):
        """Normalize features."""
        df = df.copy()
        df['log_delta_r_norm'] = (
            (df['log_delta_r'] - self.stats['log_delta_r_mean']) / 
            (self.stats['log_delta_r_std'] + 1e-6)
        )
        return df

# Preprocess
print("\nPreprocessing...")
train_preprocessor = FracturePreprocessor(train_df)
train_processed = train_preprocessor.compute_features()
train_processed = train_preprocessor.normalize_features(train_processed)

test_preprocessor = FracturePreprocessor(test_df)
test_preprocessor.stats = train_preprocessor.stats
test_processed = test_preprocessor.compute_features()
test_processed = test_preprocessor.normalize_features(test_processed)

print(f"Training: {len(train_processed)} points")
print(f"Test: {len(test_processed)} points")


# ## 4. Data Generator for Training

# In[ ]:


def create_sequences(df, sequence_length=10):
    """Create training sequences from fracture data."""
    feature_names = [
        'log_delta_r_norm', 'sin_theta', 'cos_theta',
        'delta_angle', 'curvature_trajectory',
        'mean_curvature', 'length_variance', 'tortuosity'
    ]
    
    X_features = []
    y_coords = []
    y_stop = []
    
    for fid in df['fracture_id'].unique():
        frac = df[df['fracture_id'] == fid].sort_values('point_idx').reset_index(drop=True)
        n = len(frac)
        
        if n < sequence_length + 1:
            continue
        
        # Build feature matrix
        features = np.zeros((n, len(feature_names)))
        for i, fname in enumerate(feature_names):
            if fname in frac.columns:
                vals = frac[fname].values
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                features[:, i] = vals
        
        coords = np.column_stack([frac['coord_x_norm'].values, frac['coord_y_norm'].values])
        is_last = frac['is_last'].values
        
        # Concatenate coordinates and features
        combined_features = np.hstack([coords, features])
        
        # Create sliding windows
        for i in range(len(frac) - sequence_length):
            X_features.append(combined_features[i:i+sequence_length])
            y_coords.append(coords[i+sequence_length])
            y_stop.append(is_last[i+sequence_length])
    
    return (
        np.array(X_features, dtype=np.float32),
        np.array(y_coords, dtype=np.float32),
        np.array(y_stop, dtype=np.float32)
    )

# Create datasets
print("\nCreating training sequences...")
X_train, y_train_coords, y_train_stop = create_sequences(train_processed, CONFIG['sequence_length'])
X_test, y_test_coords, y_test_stop = create_sequences(test_processed, CONFIG['sequence_length'])

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature shape: {X_train.shape}")
print(f"Coordinate target shape: {y_train_coords.shape}")
print(f"Stop target shape: {y_train_stop.shape}")
print(f"Positive stopping examples: {y_train_stop.sum():.0f} ({100*y_train_stop.mean():.2f}%)")


# ## 5. Model Architecture
# 
# ### 5.1 Additive Attention Layer

# In[ ]:


class AdditiveAttention(layers.Layer):
    """Additive attention mechanism (Bahdanau attention)."""
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.v = layers.Dense(1, use_bias=False)
    
    def call(self, hidden_states, mask=None):
        # hidden_states: [batch, seq_len, hidden_dim]
        
        # Compute attention scores
        score = self.v(tf.nn.tanh(self.W(hidden_states)))  # [batch, seq_len, 1]
        
        if mask is not None:
            score += (mask * -1e9)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # [batch, seq_len, 1]
        
        # Weighted sum
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)  # [batch, hidden_dim]
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


# ### 5.2 Complete Model

# In[ ]:


def build_advanced_lstm_model(config):
    """Build Advanced LSTM model with dual heads."""
    
    # Input
    inputs = layers.Input(shape=(config['sequence_length'], config['input_dim']), name='input_features')
    
    # Input projection
    x = layers.Dense(config['input_projection_dim'], activation='relu', name='input_projection')(inputs)
    x = layers.Dropout(config['dropout'])(x)
    
    # Bidirectional LSTM stack
    for i, units in enumerate(config['lstm_units']):
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=True,
                dropout=config['dropout'],
                recurrent_dropout=config['recurrent_dropout']
            ),
            name=f'bilstm_{i+1}'
        )(x)
        x = layers.LayerNormalization()(x)
    
    # Additive attention
    context, attention_weights = AdditiveAttention(
        config['attention_dim'],
        name='additive_attention'
    )(x)
    
    # Shared representation
    shared = context
    for i, units in enumerate(config['shared_dense_units']):
        shared = layers.Dense(units, activation='relu', name=f'shared_dense_{i+1}')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(config['dropout'])(shared)
    
    # Coordinate prediction head
    coord_head = layers.Dense(128, activation='relu', name='coord_head_1')(shared)
    coord_head = layers.Dropout(config['dropout'])(coord_head)
    coord_head = layers.Dense(64, activation='relu', name='coord_head_2')(coord_head)
    coord_output = layers.Dense(config['coord_output_dim'], name='coord_output')(coord_head)
    
    # Stopping prediction head
    stop_head = layers.Dense(64, activation='relu', name='stop_head_1')(shared)
    stop_head = layers.Dropout(config['dropout'])(stop_head)
    stop_head = layers.Dense(32, activation='relu', name='stop_head_2')(stop_head)
    stop_output = layers.Dense(config['stop_output_dim'], activation='sigmoid', name='stop_output')(stop_head)
    
    # Build model
    model = Model(inputs=inputs, outputs=[coord_output, stop_output], name='AdvancedLSTM_DualHead')
    
    return model

# Build model
model = build_advanced_lstm_model(CONFIG)
print("\nModel Summary:")
model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"\nTotal trainable parameters: {trainable_params:,}")


# ## 6. Training Setup and Compilation

# In[ ]:


# Compile model with dual loss
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss={
        'coord_output': 'mse',
        'stop_output': 'binary_crossentropy'
    },
    loss_weights={
        'coord_output': 1.0,
        'stop_output': CONFIG['lambda_stop']
    },
    metrics={
        'coord_output': ['mae'],
        'stop_output': ['accuracy', tf.keras.metrics.AUC(name='auc')]
    }
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=CONFIG['reduce_lr_factor'],
        patience=CONFIG['reduce_lr_patience'],
        verbose=1,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath=f"{CONFIG['models_dir']}/best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("Training setup complete.")


# ## 7. Training

# In[ ]:


print("\n" + "="*80)
print("TRAINING")
print("="*80)

history = model.fit(
    X_train,
    {'coord_output': y_train_coords, 'stop_output': y_train_stop},
    validation_data=(X_test, {'coord_output': y_test_coords, 'stop_output': y_test_stop}),
    epochs=CONFIG['num_epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")


# ## 8. Training Visualization

# In[ ]:


# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Total loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Coordinate loss
axes[0, 1].plot(history.history['coord_output_loss'], label='Train Coord Loss', linewidth=2)
axes[0, 1].plot(history.history['val_coord_output_loss'], label='Val Coord Loss', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE Loss')
axes[0, 1].set_title('Coordinate Prediction Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Stop loss
axes[1, 0].plot(history.history['stop_output_loss'], label='Train Stop Loss', linewidth=2)
axes[1, 0].plot(history.history['val_stop_output_loss'], label='Val Stop Loss', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('BCE Loss')
axes[1, 0].set_title('Stopping Prediction Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Stop accuracy
axes[1, 1].plot(history.history['stop_output_accuracy'], label='Train Accuracy', linewidth=2)
axes[1, 1].plot(history.history['val_stop_output_accuracy'], label='Val Accuracy', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Stopping Prediction Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['plots_dir']}/training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

# Print final metrics
print("\nFinal Training Metrics:")
print(f"  Total Loss: {history.history['loss'][-1]:.6f}")
print(f"  Coord Loss: {history.history['coord_output_loss'][-1]:.6f}")
print(f"  Stop Loss: {history.history['stop_output_loss'][-1]:.6f}")
print(f"  Stop Accuracy: {history.history['stop_output_accuracy'][-1]:.4f}")

print("\nFinal Validation Metrics:")
print(f"  Total Loss: {history.history['val_loss'][-1]:.6f}")
print(f"  Coord Loss: {history.history['val_coord_output_loss'][-1]:.6f}")
print(f"  Stop Loss: {history.history['val_stop_output_loss'][-1]:.6f}")
print(f"  Stop Accuracy: {history.history['val_stop_output_accuracy'][-1]:.4f}")


# ## 9. Fracture Path Generation with Natural Stopping

# In[ ]:


def generate_fracture_path_with_stopping(model, seed_features, seed_coords, config, preprocessor_stats):
    """Generate fracture path with natural stopping criteria."""
    
    generated_coords = list(seed_coords)
    
    # Initialize current sequence with concatenated coords and features
    current_sequence = np.hstack([seed_coords, seed_features])
    
    consecutive_stop_count = 0
    stop_probs = []
    
    for step in range(config['max_generation_steps']):
        # Prepare input
        input_seq = current_sequence[-config['sequence_length']:]
        if len(input_seq) < config['sequence_length']:
            pad_length = config['sequence_length'] - len(input_seq)
            input_seq = np.vstack([np.zeros((pad_length, input_seq.shape[1])), input_seq])
        
        input_batch = np.expand_dims(input_seq, axis=0)
        
        # Predict
        pred_coord, pred_stop = model.predict(input_batch, verbose=0)
        next_coord = pred_coord[0]
        stop_prob = pred_stop[0][0]
        
        stop_probs.append(stop_prob)
        
        # Check stopping criteria
        if stop_prob > config['stop_threshold']:
            consecutive_stop_count += 1
            if consecutive_stop_count >= config['consecutive_stop_steps']:
                print(f"  Stopped at step {step} (consecutive high stop probability)")
                break
        else:
            consecutive_stop_count = 0
        
        # Add predicted point
        generated_coords.append(next_coord)
        
        # Check for stagnation
        if step > 5:
            recent_movement = np.linalg.norm(np.array(generated_coords[-1]) - np.array(generated_coords[-2]))
            if recent_movement < config['min_movement_threshold']:
                # print(f"  Stopped at step {step} (stagnation detected)")
                break
        
        # Update sequence
        # Create new input vector: [next_coord, last_features]
        # Note: Ideally we should recompute features based on new trajectory, 
        # but for efficiency we reuse last features for now
        last_features = current_sequence[-1, 2:] # 8 features
        new_input = np.concatenate([next_coord, last_features])
        current_sequence = np.vstack([current_sequence, new_input])
    
    return np.array(generated_coords), stop_probs

# Load best model
print("Loading best model...")
model = keras.models.load_model(
    f"{CONFIG['models_dir']}/best_model.keras",
    custom_objects={'AdditiveAttention': AdditiveAttention}
)
print("Model loaded successfully.")


# ## 10. Evaluation on Test Set

# In[ ]:


def compute_hausdorff(path1, path2):
    """Compute Hausdorff distance."""
    if len(path1) < 2 or len(path2) < 2:
        return float('inf')
    try:
        return max(directed_hausdorff(path1, path2)[0], directed_hausdorff(path2, path1)[0])
    except:
        return float('inf')

def discrete_frechet_distance(P, Q):
    """Iterative discrete Fréchet distance using DP."""
    P, Q = np.array(P), np.array(Q)
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float('inf')
    ca = np.full((n, m), np.inf)
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(P[i] - Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(P[0] - Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]),
                np.linalg.norm(P[i] - Q[j])
            )
    return float(ca[n-1, m-1])

def compute_path_similarity(path1, path2):
    """Mean cosine similarity of direction vectors, mapped to [0, 1]."""
    p1, p2 = np.array(path1), np.array(path2)
    if len(p1) < 2 or len(p2) < 2:
        return 0.0
    d1 = np.diff(p1, axis=0)
    d2 = np.diff(p2, axis=0)
    min_len = min(len(d1), len(d2))
    d1, d2 = d1[:min_len], d2[:min_len]
    n1 = np.linalg.norm(d1, axis=1, keepdims=True) + 1e-10
    n2 = np.linalg.norm(d2, axis=1, keepdims=True) + 1e-10
    cos_sims = np.sum((d1 / n1) * (d2 / n2), axis=1)
    return float(np.mean((cos_sims + 1.0) / 2.0))

def _seg_lengths(path):
    p = np.array(path)
    return np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) >= 2 else np.array([])

def _seg_angles(path):
    p = np.array(path)
    if len(p) < 2:
        return np.array([])
    d = np.diff(p, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])

# -----------------------------------------------------------------------
# Single-step prediction metrics on the full test set
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("SINGLE-STEP PREDICTION METRICS")
print("="*60)

pred_outputs = model.predict(X_test, batch_size=CONFIG['batch_size'], verbose=0)
y_pred_coords_ss = pred_outputs[0]   # [N, 2]
y_pred_stop_ss   = pred_outputs[1].squeeze()  # [N]

mse_ss   = float(np.mean((y_pred_coords_ss - y_test_coords) ** 2))
rmse_ss  = float(np.sqrt(mse_ss))
mae_ss   = float(np.mean(np.abs(y_pred_coords_ss - y_test_coords)))
mse_x_ss = float(np.mean((y_pred_coords_ss[:, 0] - y_test_coords[:, 0]) ** 2))
mse_y_ss = float(np.mean((y_pred_coords_ss[:, 1] - y_test_coords[:, 1]) ** 2))
mae_x_ss = float(np.mean(np.abs(y_pred_coords_ss[:, 0] - y_test_coords[:, 0])))
mae_y_ss = float(np.mean(np.abs(y_pred_coords_ss[:, 1] - y_test_coords[:, 1])))

print(f"  MSE:   {mse_ss:.6f}  RMSE: {rmse_ss:.6f}  MAE: {mae_ss:.6f}")
print(f"  MSE-X: {mse_x_ss:.6f}  MSE-Y: {mse_y_ss:.6f}")
print(f"  MAE-X: {mae_x_ss:.6f}  MAE-Y: {mae_y_ss:.6f}")
print(f"  N test samples: {len(X_test)}")

pd.DataFrame([{
    'mse': mse_ss, 'rmse': rmse_ss, 'mae': mae_ss,
    'mse_x': mse_x_ss, 'mse_y': mse_y_ss,
    'mae_x': mae_x_ss, 'mae_y': mae_y_ss,
    'n_samples': len(X_test)
}]).to_csv(f"{CONFIG['results_dir']}/evaluation_metrics.csv", index=False)
print(f"Single-step metrics saved to {CONFIG['results_dir']}/evaluation_metrics.csv")

# -----------------------------------------------------------------------
# Stopping prediction metrics on the full test set (classification)
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("STOPPING PREDICTION METRICS")
print("="*60)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

stop_pred_binary = (y_pred_stop_ss >= CONFIG['stop_threshold']).astype(int)
stop_true_binary = y_test_stop.astype(int)

stop_accuracy = float(accuracy_score(stop_true_binary, stop_pred_binary))
stop_precision = float(precision_score(stop_true_binary, stop_pred_binary, zero_division=0))
stop_recall    = float(recall_score(stop_true_binary, stop_pred_binary, zero_division=0))
stop_f1        = float(f1_score(stop_true_binary, stop_pred_binary, zero_division=0))

print(f"  Accuracy:  {stop_accuracy:.4f}")
print(f"  Precision: {stop_precision:.4f}")
print(f"  Recall:    {stop_recall:.4f}")
print(f"  F1-score:  {stop_f1:.4f}")
print(f"  Positive stopping examples: {stop_true_binary.sum()} / {len(stop_true_binary)}")

pd.DataFrame([{
    'accuracy': stop_accuracy,
    'precision': stop_precision,
    'recall': stop_recall,
    'f1_score': stop_f1,
    'n_samples': len(stop_true_binary),
    'n_positive': int(stop_true_binary.sum())
}]).to_csv(f"{CONFIG['results_dir']}/stopping_metrics.csv", index=False)
print(f"Stopping metrics saved to {CONFIG['results_dir']}/stopping_metrics.csv")

def plot_fracture_generation_comparison_with_stopping(true_path, gen_path, seed_points, stop_probs, 
                                                      fracture_id, save_dir, metrics=None):
    """
    Enhanced visualization with stopping probability analysis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Left plot: Path comparison with seed highlighting
    ax1 = axes[0]
    
    # Plot true path
    if len(true_path) > 0:
        ax1.plot(true_path[:, 0], true_path[:, 1], 'b-o', 
                label='True Path', linewidth=2.5, markersize=5, alpha=0.8, zorder=2)
    
    # Plot generated path
    if len(gen_path) > 0:
        ax1.plot(gen_path[:, 0], gen_path[:, 1], 'r-s', 
                label='Generated Path', linewidth=2.5, markersize=4, alpha=0.7, zorder=2)
    
    # Highlight seed points
    if len(seed_points) > 0:
        ax1.scatter(seed_points[:, 0], seed_points[:, 1], 
                  c='green', s=200, marker='*', 
                  label='Seed Points', edgecolors='darkgreen', 
                  linewidths=2, zorder=5, alpha=0.9)
        ax1.scatter(seed_points[0, 0], seed_points[0, 1], 
                  c='lime', s=300, marker='*', 
                  edgecolors='darkgreen', linewidths=3, 
                  zorder=6, label='Start Point')
    
    ax1.set_xlabel('X Coordinate (Normalized)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Coordinate (Normalized)', fontsize=14, fontweight='bold')
    title = f'Fracture {fracture_id}: Path Comparison'
    if metrics:
        title += f'\nHausdorff: {metrics.get("hausdorff", 0):.4f} | Endpoint: {metrics.get("endpoint_error", 0):.4f}'
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    
    # Middle plot: Stopping probability evolution
    ax2 = axes[1]
    if len(stop_probs) > 0:
        ax2.plot(stop_probs, 'g-', linewidth=2.5, label='Stop Probability', alpha=0.8)
        ax2.fill_between(range(len(stop_probs)), stop_probs, alpha=0.3, color='green')
        ax2.axhline(y=CONFIG['stop_threshold'], color='r', linestyle='--', 
                   linewidth=2, label=f'Threshold: {CONFIG["stop_threshold"]}')
        ax2.axhline(y=np.mean(stop_probs), color='orange', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(stop_probs):.3f}')
        ax2.set_xlabel('Generation Step', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Stopping Probability', fontsize=14, fontweight='bold')
        ax2.set_title('Stopping Probability Evolution', fontsize=16, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1])
    
    # Right plot: Error visualization
    ax3 = axes[2]
    if len(true_path) > 0 and len(gen_path) > 0:
        min_len = min(len(true_path), len(gen_path))
        true_subset = true_path[:min_len]
        gen_subset = gen_path[:min_len]
        errors = np.linalg.norm(true_subset - gen_subset, axis=1)
        
        ax3.plot(errors, 'purple', linewidth=2, label='Point-wise Error', alpha=0.7)
        ax3.fill_between(range(len(errors)), errors, alpha=0.3, color='purple')
        ax3.axhline(y=np.mean(errors), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax3.axhline(y=np.median(errors), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(errors):.4f}')
        ax3.set_xlabel('Point Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Euclidean Error', fontsize=14, fontweight='bold')
        ax3.set_title('Error Along Generated Path', fontsize=16, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fracture_{fracture_id}_generation_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_generation_progression(true_path, gen_path, seed_points, fracture_id, save_dir):
    """
    Visualize the generation progression with color-coded steps.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot true path
    if len(true_path) > 0:
        ax.plot(true_path[:, 0], true_path[:, 1], 'b-o', 
               label='True Path', linewidth=2, markersize=4, alpha=0.6, zorder=1)
    
    # Plot generated path with color progression
    if len(gen_path) > len(seed_points):
        generated_only = gen_path[len(seed_points):]
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(generated_only)))
        
        for i in range(len(generated_only) - 1):
            ax.plot([generated_only[i, 0], generated_only[i+1, 0]], 
                   [generated_only[i, 1], generated_only[i+1, 1]], 
                   color=colors[i], linewidth=2.5, alpha=0.7, zorder=2)
            ax.scatter(generated_only[i, 0], generated_only[i, 1], 
                      c=[colors[i]], s=50, zorder=3, alpha=0.8)
        
        if len(generated_only) > 0:
            ax.scatter(generated_only[-1, 0], generated_only[-1, 1], 
                      c='darkred', s=100, marker='s', zorder=4, 
                      label='Generated End', edgecolors='black', linewidths=1.5)
    
    # Highlight seed points
    if len(seed_points) > 0:
        ax.scatter(seed_points[:, 0], seed_points[:, 1], 
                  c='green', s=250, marker='*', 
                  label='Seed Points', edgecolors='darkgreen', 
                  linewidths=2, zorder=5)
        ax.scatter(seed_points[0, 0], seed_points[0, 1], 
                  c='lime', s=350, marker='*', 
                  edgecolors='darkgreen', linewidths=3, 
                  zorder=6, label='Start Point')
    
    # Highlight true path endpoints
    if len(true_path) > 0:
        ax.scatter(true_path[0, 0], true_path[0, 1], 
                  c='cyan', s=200, marker='o', 
                  edgecolors='blue', linewidths=2, 
                  zorder=4, label='True Start', alpha=0.8)
        ax.scatter(true_path[-1, 0], true_path[-1, 1], 
                  c='blue', s=200, marker='o', 
                  edgecolors='darkblue', linewidths=2, 
                  zorder=4, label='True End', alpha=0.8)
    
    ax.set_xlabel('X Coordinate (Normalized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title(f'Fracture {fracture_id}: Generation Progression\n(Color intensity shows generation order)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fracture_{fracture_id}_generation_progression.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Evaluate
print("\n" + "="*80)
print("FRACTURE PATH GENERATION EVALUATION")
print("="*80)

all_metrics = {
    'hausdorff': [],
    'frechet': [],
    'endpoint_error': [],
    'path_length_error': [],
    'path_similarity': [],
    'generation_length': [],
    'true_length': []
}
path_rows = []
_true_lens, _gen_lens, _true_angs, _gen_angs = [], [], [], []

feature_names = [
    'log_delta_r_norm', 'sin_theta', 'cos_theta',
    'delta_angle', 'curvature_trajectory',
    'mean_curvature', 'length_variance', 'tortuosity'
]

# Get unique fractures from test set — evaluate ALL
test_fracture_ids = test_processed['fracture_id'].unique()

for fid in test_fracture_ids:
    frac = test_processed[test_processed['fracture_id'] == fid].sort_values('point_idx').reset_index(drop=True)

    if len(frac) < 2:
        continue

    # Build seed features (use up to sequence_length points; generation pads if shorter)
    seed_len = min(CONFIG['sequence_length'], len(frac) - 1)
    seed_features = []
    for i in range(seed_len):
        feat_vec = []
        for fname in feature_names:
            val = frac.iloc[i][fname] if fname in frac.columns else 0.0
            feat_vec.append(val if not np.isnan(val) else 0.0)
        seed_features.append(feat_vec)
    seed_features = np.array(seed_features)

    seed_coords = frac[['coord_x_norm', 'coord_y_norm']].values[:seed_len]
    true_path = frac[['coord_x_norm', 'coord_y_norm']].values

    # Generate
    try:
        gen_path, stop_probs = generate_fracture_path_with_stopping(
            model, seed_features, seed_coords, CONFIG, train_preprocessor.stats
        )
    except Exception as e:
        print(f"Fracture {fid}: generation error - {e}")
        continue

    if len(gen_path) > 1:
        hausdorff = compute_hausdorff(true_path, gen_path)
        frechet   = discrete_frechet_distance(true_path, gen_path)
        endpoint_error = np.linalg.norm(gen_path[-1] - true_path[-1])
        path_sim  = compute_path_similarity(true_path, gen_path)
        gen_length = np.sum(np.linalg.norm(np.diff(gen_path, axis=0), axis=1))
        true_length = np.sum(np.linalg.norm(np.diff(true_path, axis=0), axis=1))
        length_error = abs(gen_length - true_length) / (true_length + 1e-6)

        if hausdorff != float('inf'):
            all_metrics['hausdorff'].append(hausdorff)
        if frechet != float('inf'):
            all_metrics['frechet'].append(frechet)
        all_metrics['endpoint_error'].append(endpoint_error)
        all_metrics['path_length_error'].append(length_error)
        all_metrics['path_similarity'].append(path_sim)
        all_metrics['generation_length'].append(len(gen_path))
        all_metrics['true_length'].append(len(true_path))

        path_rows.append({
            'fracture_id': fid,
            'hausdorff': hausdorff,
            'frechet': frechet,
            'endpoint_error': endpoint_error,
            'length_error': length_error,
            'path_similarity': path_sim,
            'true_n_pts': len(true_path),
            'gen_n_pts': len(gen_path)
        })
        _true_lens.extend(_seg_lengths(true_path).tolist())
        _gen_lens.extend(_seg_lengths(gen_path).tolist())
        _true_angs.extend(_seg_angles(true_path).tolist())
        _gen_angs.extend(_seg_angles(gen_path).tolist())

        print(f"Fracture {fid}: Hausdorff={hausdorff:.4f}, Frechet={frechet:.4f}, "
              f"Endpoint={endpoint_error:.4f}, LenErr={length_error:.4f}, "
              f"PathSim={path_sim:.4f}")

        # Visualization
        try:
            plot_fracture_generation_comparison_with_stopping(
                true_path, gen_path, seed_coords, stop_probs, fid, CONFIG['plots_dir'],
                {'hausdorff': hausdorff, 'endpoint_error': endpoint_error, 'length_error': length_error}
            )
            plot_generation_progression(true_path, gen_path, seed_coords, fid, CONFIG['plots_dir'])
        except Exception as viz_err:
            print(f"  Visualization failed for fracture {fid}: {viz_err}")


# Print summary and save path metrics CSV
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)

for metric_name, values in all_metrics.items():
    if values:
        print(f"{metric_name.replace('_', ' ').title():25s}: "
              f"Mean={np.mean(values):.4f} +/- {np.std(values):.4f}, "
              f"Median={np.median(values):.4f}")
    else:
        print(f"{metric_name.replace('_', ' ').title():25s}: No data")

if path_rows:
    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(f"{CONFIG['results_dir']}/path_metrics.csv", index=False)
    print(f"\nPath metrics saved to {CONFIG['results_dir']}/path_metrics.csv")

    summary_row = {}
    for col in ['hausdorff', 'frechet', 'endpoint_error', 'length_error', 'path_similarity']:
        vals = path_df[col].dropna().values
        vals = vals[np.isfinite(vals)]
        summary_row[f'{col}_mean'] = float(np.mean(vals)) if len(vals) > 0 else float('nan')
        summary_row[f'{col}_std']  = float(np.std(vals))  if len(vals) > 0 else float('nan')
    summary_row['n_fractures'] = len(path_rows)
    pd.DataFrame([summary_row]).to_csv(f"{CONFIG['results_dir']}/path_metrics_summary.csv", index=False)
    print(f"Path metrics summary saved to {CONFIG['results_dir']}/path_metrics_summary.csv")

    # Save segment distributions for visualization
    if _true_lens or _gen_lens:
        pd.DataFrame({
            'type': ['true'] * len(_true_lens) + ['generated'] * len(_gen_lens),
            'value': _true_lens + _gen_lens
        }).to_csv(f"{CONFIG['results_dir']}/segment_lengths.csv", index=False)
    if _true_angs or _gen_angs:
        pd.DataFrame({
            'type': ['true'] * len(_true_angs) + ['generated'] * len(_gen_angs),
            'value': _true_angs + _gen_angs
        }).to_csv(f"{CONFIG['results_dir']}/segment_angles.csv", index=False)
    print(f"Segment distributions saved to {CONFIG['results_dir']}/")

    # -----------------------------------------------------------------------
    # Early/late stopping rates (path-level) — extend stopping_metrics.csv
    # -----------------------------------------------------------------------
    n_total = len(path_df)
    # Early stopping within 2 points: |gen - true| <= 2  (stopped close to true endpoint)
    early_within_2 = int(((path_df['true_n_pts'] - path_df['gen_n_pts']).abs() <= 2).sum())
    # Late stopping: gen_n_pts > true_n_pts + 5 (overshot by more than 5 steps)
    late_overshoot_5 = int((path_df['gen_n_pts'] - path_df['true_n_pts'] > 5).sum())

    early_rate = early_within_2 / n_total if n_total > 0 else float('nan')
    late_rate  = late_overshoot_5 / n_total if n_total > 0 else float('nan')

    print(f"\nStopping Quality (path-level):")
    print(f"  Early stopping within 2 pts: {early_within_2}/{n_total} = {early_rate:.4f}")
    print(f"  Late stopping (overshoot >5): {late_overshoot_5}/{n_total} = {late_rate:.4f}")

    # Load existing stopping_metrics.csv and append these columns
    stop_csv_path = f"{CONFIG['results_dir']}/stopping_metrics.csv"
    try:
        stop_df = pd.read_csv(stop_csv_path)
        stop_df['early_within_2_pts_rate'] = early_rate
        stop_df['late_overshoot_5pts_rate'] = late_rate
        stop_df['n_fractures_generated'] = n_total
    except FileNotFoundError:
        stop_df = pd.DataFrame([{
            'early_within_2_pts_rate': early_rate,
            'late_overshoot_5pts_rate': late_rate,
            'n_fractures_generated': n_total
        }])
    stop_df.to_csv(stop_csv_path, index=False)
    print(f"Stopping metrics updated in {stop_csv_path}")


# ## 11. Results and Conclusions
# 
# ### Key Features:
# 1. **Dual-Head Architecture**: Simultaneously predicts coordinates and stopping probability
# 2. **Natural Stopping**: Model learns when fractures should terminate
# 3. **Additive Attention**: Focuses on relevant historical context
# 4. **BiLSTM Stack**: Captures bidirectional sequence patterns
# 
# ### Stopping Criteria:
# - Probability threshold (default 0.5)
# - Consecutive high probabilities (3 steps)
# - Stagnation detection
# - Maximum step limit
# 
# ### Potential Improvements:
# 1. Curriculum learning for stopping prediction
# 2. Confidence-based early termination
# 3. Multi-modal stopping distributions
# 4. Reinforcement learning for generation quality

# In[ ]:


print("\n" + "="*80)
print("CASE 3: ADVANCED LSTM WITH STOPPING - COMPLETE")
print("="*80)
print(f"\nResults saved to: {CONFIG['results_dir']}")
print(f"Plots saved to: {CONFIG['plots_dir']}")
print(f"Model saved to: {CONFIG['models_dir']}/best_model.keras")

