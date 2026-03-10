#!/usr/bin/env python
# coding: utf-8

# # Case 4: CNN-GRU with Mixture Density Network (MDN)
# 
# This notebook implements the **CNN-GRU with Mixture Density Network** model described in Section 4.5 of the paper.
# 
# This is adapted from the existing `fracture_network_oct_nov_2025.py` implementation.
# 
# ## Model Architecture Overview
# 
# The architecture includes:
# 1. **Dilated CNN Encoder**: 3 convolutional blocks with dilations [1, 2, 4] and channels [64, 128, 128]
# 2. **Bidirectional GRU**: Hidden dimension 128 for sequence modeling
# 3. **Additive Attention**: Aggregates GRU outputs
# 4. **MDN Head**: Outputs mixture parameters for K=3 Gaussian components
#    - Mixture weights π_k
#    - Means μ_r, μ_θ for step length and angle
#    - Standard deviations σ_r, σ_θ
#    - Stopping probability p_stop
# 5. **Probabilistic Generation**: Samples from predicted mixture distributions
# 
# The key innovation is probabilistic prediction via MDN, capturing multi-modal uncertainty in fracture propagation.

# ## 1. Import Libraries and Setup

# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add workspace to path to import visualization_utils
sys.path.append('/workspace')
try:
    from visualization_utils import plot_fracture_generation_comparison, plot_generation_progression
    VISUALIZATION_UTILS_AVAILABLE = True
except ImportError:
    VISUALIZATION_UTILS_AVAILABLE = False
    print("Warning: visualization_utils not found. Enhanced plotting will be disabled.")

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
import os
from typing import Tuple, List, Dict
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
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Plotting configuration
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


# ## 2. Configuration

# In[ ]:


CONFIG = {
    'seed': 42,
    'epsilon': 1e-6,
    'device': device,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'num_epochs': 50,
    'history_window': 10,  # k previous steps
    'num_mixtures': 3,     # Model architecture - ENHANCED FOR A40 GPU
    'input_dim': 8,
    'cnn_filters': [64, 128, 256, 512],  # Added 4th block
    'cnn_kernel_size': 3,
    'cnn_dropout': 0.2,
    'gru_hidden_dim': 256,    # Increased from 128
    'gru_layers': 2,
    'gru_dropout': 0.2,
    'dense_units': 128,
    'num_mixtures': 5,        # Increased from 3
    
    # Training parameters
    'lambda_stop': 1.0,
    'max_steps': 200,
    'stop_threshold': 0.5,
    
    # Paths
    'train_csv': 'train_fractures_processed.csv',
    'test_csv': 'test_fractures_processed.csv',
    'results_dir': 'fracture_results/case4',
    'plots_dir': 'fracture_results/case4/plots',
    'models_dir': 'fracture_results/case4/models',
}

print("Configuration:")
for key, value in CONFIG.items():
    if key != 'device':
        print(f"  {key}: {value}")


# ## 3. Data Preprocessing Utilities

# In[ ]:


def compute_angle(x1, y1, x2, y2):
    """Compute angle in radians from point 1 to point 2."""
    return np.arctan2(y2 - y1, x2 - x1)

def compute_length(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_difference(a1, a2):
    """Compute the smallest angle difference, wrapped to [-π, π]."""
    diff = a2 - a1
    return np.arctan2(np.sin(diff), np.cos(diff))

class FracturePreprocessor:
    """Preprocesses fracture data and computes derived features."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.stats = {}
        
    def compute_deltas_and_features(self):
        """Compute Δr, Δθ, curvature trajectory, and global stats."""
        results = []
        
        for fid in self.df['fracture_id'].unique():
            frac = self.df[self.df['fracture_id'] == fid].sort_values('point_idx').reset_index(drop=True)
            n = len(frac)
            
            if n < 3:
                continue
            
            # Extract coordinates
            xs = frac['coord_x'].values
            ys = frac['coord_y'].values
            
            # Normalize by fracture centroid
            cx, cy = xs.mean(), ys.mean()
            xs_norm = xs - cx
            ys_norm = ys - cy
            
            # Compute segment lengths
            lengths = np.sqrt(np.diff(xs_norm)**2 + np.diff(ys_norm)**2)
            median_length = np.median(lengths) if len(lengths) > 0 else 1.0
            
            # Scale coordinates
            scale = median_length if median_length > 0 else 1.0
            xs_norm /= scale
            ys_norm /= scale
            
            # Recompute lengths after normalization
            lengths = np.sqrt(np.diff(xs_norm)**2 + np.diff(ys_norm)**2)
            
            # Compute angles
            angles = np.arctan2(np.diff(ys_norm), np.diff(xs_norm))
            
            # Compute Δθ (angle changes)
            delta_angles = np.zeros(n)
            if len(angles) > 1:
                for i in range(1, len(angles)):
                    delta_angles[i] = angle_difference(angles[i-1], angles[i])
            
            # Compute curvature trajectory (second derivative of angle)
            curvature_trajectory = np.zeros(n)
            if n > 3:
                for i in range(2, n-1):
                    curvature_trajectory[i] = delta_angles[i] - delta_angles[i-1]
            
            # Compute fracture-level statistics (fingerprint)
            mean_curvature = np.abs(delta_angles).mean()
            length_variance = np.var(lengths) if len(lengths) > 0 else 0.0
            path_length = lengths.sum()
            endpoint_dist = np.sqrt((xs_norm[-1]-xs_norm[0])**2 + (ys_norm[-1]-ys_norm[0])**2)
            tortuosity = path_length / (endpoint_dist + 1e-6)
            
            # Process each point
            for i in range(n):
                point = frac.iloc[i].to_dict()
                point['coord_x_norm'] = xs_norm[i]
                point['coord_y_norm'] = ys_norm[i]
                point['scale'] = scale
                point['centroid_x'] = cx
                point['centroid_y'] = cy
                
                # Current and next deltas
                if i < n - 1:
                    point['delta_r'] = lengths[i]
                    point['delta_theta'] = angles[i]
                    point['log_delta_r'] = np.log(lengths[i] + CONFIG['epsilon'])
                    point['sin_theta'] = np.sin(angles[i])
                    point['cos_theta'] = np.cos(angles[i])
                else:
                    point['delta_r'] = 0.0
                    point['delta_theta'] = 0.0
                    point['log_delta_r'] = np.log(CONFIG['epsilon'])
                    point['sin_theta'] = 0.0
                    point['cos_theta'] = 0.0
                
                point['delta_angle'] = delta_angles[i]
                point['curvature_trajectory'] = curvature_trajectory[i]
                
                # Fracture fingerprint
                point['mean_curvature'] = mean_curvature
                point['length_variance'] = length_variance
                point['tortuosity'] = tortuosity
                
                results.append(point)
        
        processed_df = pd.DataFrame(results)
        
        # Compute global statistics for normalization
        valid_log_delta_r = processed_df[processed_df['delta_r'] > 0]['log_delta_r']
        self.stats['log_delta_r_mean'] = valid_log_delta_r.mean() if len(valid_log_delta_r) > 0 else 0.0
        self.stats['log_delta_r_std'] = valid_log_delta_r.std() if len(valid_log_delta_r) > 0 else 1.0
        self.stats['delta_theta_mean'] = processed_df['delta_theta'].mean()
        self.stats['delta_theta_std'] = processed_df['delta_theta'].std()
        self.stats['delta_r_quantiles'] = np.quantile(valid_log_delta_r, [0.01, 0.99]) if len(valid_log_delta_r) > 0 else [-5, 2]
        
        return processed_df
    
    def normalize_features(self, df: pd.DataFrame):
        """Standardize features using training statistics."""
        df = df.copy()
        
        # Standardize log_delta_r
        df['log_delta_r_norm'] = (
            (df['log_delta_r'] - self.stats['log_delta_r_mean']) / 
            (self.stats['log_delta_r_std'] + CONFIG['epsilon'])
        )
        
        return df


# ## 4. Dataset Class

# In[ ]:


class FractureDataset(Dataset):
    """PyTorch Dataset for fracture sequences."""
    
    def __init__(self, df: pd.DataFrame, history_window: int = 10, is_train: bool = True):
        self.df = df
        self.history_window = history_window
        self.is_train = is_train
        
        # Group by fracture_id
        self.fractures = []
        for fid in df['fracture_id'].unique():
            frac = df[df['fracture_id'] == fid].sort_values('point_idx').reset_index(drop=True)
            if len(frac) >= 2:  # Need seed + at least 1 target
                self.fractures.append(frac)
    
    def __len__(self):
        return len(self.fractures)
    
    def __getitem__(self, idx):
        frac = self.fractures[idx]
        n = len(frac)
        
        # Build input features
        features = self._build_features(frac)
        
        # Build target for the point after history window
        target_idx = min(self.history_window, n - 1)
        target = {
            'delta_r': float(frac.iloc[target_idx]['delta_r']),
            'log_delta_r': float(frac.iloc[target_idx]['log_delta_r']),
            'delta_theta': float(frac.iloc[target_idx]['delta_theta']),
            'sin_theta': float(frac.iloc[target_idx]['sin_theta']),
            'cos_theta': float(frac.iloc[target_idx]['cos_theta']),
            'is_last': 1.0 if target_idx == n - 1 else 0.0
        }
        
        return {
            'features': torch.FloatTensor(features),
            'target': target,
            'fracture_id': int(frac['fracture_id'].iloc[0]),
            'n_points': n
        }
    
    def _build_features(self, frac: pd.DataFrame):
        """Build feature matrix for a fracture."""
        n = len(frac)
        feature_names = [
            'log_delta_r_norm', 'sin_theta', 'cos_theta',
            'delta_angle', 'curvature_trajectory',
            'mean_curvature', 'length_variance', 'tortuosity'
        ]
        
        features = np.zeros((n, len(feature_names)))
        for i, fname in enumerate(feature_names):
            if fname in frac.columns:
                vals = frac[fname].values
                # Replace NaN with 0
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                features[:, i] = vals
            else:
                # Fill with zeros if feature is missing
                features[:, i] = 0.0
        
        return features

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Find max sequence length in batch
    max_len = max([item['features'].shape[0] for item in batch])
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[1]
    
    # Pad sequences
    features_padded = torch.zeros(batch_size, max_len, feature_dim)
    lengths = []
    targets = []
    fracture_ids = []
    n_points_list = []
    
    for i, item in enumerate(batch):
        seq_len = item['features'].shape[0]
        features_padded[i, :seq_len, :] = item['features']
        lengths.append(seq_len)
        targets.append(item['target'])
        fracture_ids.append(item['fracture_id'])
        n_points_list.append(item['n_points'])
    
    return {
        'features': features_padded,
        'lengths': torch.LongTensor(lengths),
        'targets': targets,
        'fracture_ids': fracture_ids,
        'n_points': n_points_list
    }


# ## 5. Model Architecture
# 
# ### 5.1 Dilated 1D Convolutional Block

# In[ ]:


class DilatedConv1DBlock(nn.Module):
    """1D Convolutional block with dilation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        if self.residual is not None:
            x = self.residual(x)
        
        return out + x


# ### 5.2 Additive Attention

# In[ ]:


class AdditiveAttention(nn.Module):
    """Additive attention mechanism."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden_states, mask=None):
        # hidden_states: [batch, seq_len, hidden_dim]
        scores = self.v(torch.tanh(self.W(hidden_states)))  # [batch, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), hidden_states)  # [batch, 1, hidden_dim]
        
        return context.squeeze(1), weights


# ### 5.3 MDN Output Head

# In[ ]:


class MDNHead(nn.Module):
    """Mixture Density Network output head."""
    
    def __init__(self, input_dim, num_mixtures):
        super().__init__()
        self.num_mixtures = num_mixtures
        
        # Output: pi (mixture weights), mu (means), sigma (stds) for Δr and Δθ
        self.pi_layer = nn.Linear(input_dim, num_mixtures)
        self.mu_r_layer = nn.Linear(input_dim, num_mixtures)
        self.sigma_r_layer = nn.Linear(input_dim, num_mixtures)
        self.mu_theta_layer = nn.Linear(input_dim, num_mixtures)
        self.sigma_theta_layer = nn.Linear(input_dim, num_mixtures)
        
        # Stop probability
        self.stop_layer = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # Mixture weights (softmax to ensure sum to 1)
        pi = F.softmax(self.pi_layer(x), dim=-1)
        
        # Means (unbounded)
        mu_r = self.mu_r_layer(x)
        mu_theta = self.mu_theta_layer(x)
        
        # Standard deviations (positive via softplus)
        sigma_r = F.softplus(self.sigma_r_layer(x)) + CONFIG['epsilon']
        sigma_theta = F.softplus(self.sigma_theta_layer(x)) + CONFIG['epsilon']
        
        # Stop probability
        p_stop = torch.sigmoid(self.stop_layer(x))
        
        return pi, mu_r, sigma_r, mu_theta, sigma_theta, p_stop


# ### 5.4 Complete Model

# In[ ]:


class FractureGeneratorModel(nn.Module):
    """Complete Case 4 model: 1D CNN + GRU + Attention + MDN."""
    
    def __init__(self, input_dim, hidden_dim, num_mixtures, cnn_channels, kernel_size):
        super().__init__()
        
        # 1D CNN encoder with dilated convolutions
        self.conv_blocks = nn.ModuleList()
        in_ch = input_dim
        # Dynamic dilation rates: 1, 2, 4, 8, ...
        dilations = [2**i for i in range(len(cnn_channels))]
        
        for out_ch, dilation in zip(cnn_channels, dilations):
            self.conv_blocks.append(DilatedConv1DBlock(in_ch, out_ch, kernel_size, dilation))
            in_ch = out_ch
        
        # Bidirectional GRU
        self.gru = nn.GRU(cnn_channels[-1], hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = AdditiveAttention(hidden_dim * 2)
        
        # MDN head
        self.mdn_head = MDNHead(hidden_dim * 2, num_mixtures)
    
    def forward(self, x, lengths=None):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # CNN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Back to [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # GRU
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_dim*2]
        
        # Attention
        context, attn_weights = self.attention(gru_out)
        
        # MDN head
        pi, mu_r, sigma_r, mu_theta, sigma_theta, p_stop = self.mdn_head(context)
        
        return pi, mu_r, sigma_r, mu_theta, sigma_theta, p_stop, attn_weights


# ## 6. Loss Functions

# In[ ]:


def mdn_loss(pi, mu_r, sigma_r, mu_theta, sigma_theta, target_r, target_theta):
    """Negative log-likelihood loss for MDN."""
    # target_r, target_theta: [batch_size]
    # pi, mu_r, sigma_r, mu_theta, sigma_theta: [batch_size, num_mixtures]
    
    # Expand targets to [batch_size, num_mixtures]
    target_r = target_r.unsqueeze(-1)  # [batch_size, 1]
    target_theta = target_theta.unsqueeze(-1)  # [batch_size, 1]
    
    # For Δr
    z_r = (target_r - mu_r) / (sigma_r + CONFIG['epsilon'])
    prob_r = torch.exp(-0.5 * z_r ** 2) / (sigma_r * np.sqrt(2 * np.pi) + CONFIG['epsilon'])
    
    # For Δθ
    z_theta = (target_theta - mu_theta) / (sigma_theta + CONFIG['epsilon'])
    prob_theta = torch.exp(-0.5 * z_theta ** 2) / (sigma_theta * np.sqrt(2 * np.pi) + CONFIG['epsilon'])
    
    # Joint probability (assuming independence)
    prob = prob_r * prob_theta
    
    # Weighted sum over mixtures
    weighted_prob = (pi * prob).sum(dim=-1)  # [batch_size]
    
    # Negative log-likelihood
    nll = -torch.log(weighted_prob + CONFIG['epsilon'])
    
    return nll.mean()


# ## 7. Data Loading

# In[ ]:


print("\n" + "="*80)
print("DATA LOADING AND PREPROCESSING")
print("="*80)

# Load data
print("\nLoading data...")
try:
    train_df = pd.read_csv(CONFIG['train_csv'])
    test_df = pd.read_csv(CONFIG['test_csv'])
    print(f"Training samples: {len(train_df)}, Fractures: {train_df['fracture_id'].nunique()}")
    print(f"Test samples: {len(test_df)}, Fractures: {test_df['fracture_id'].nunique()}")
except FileNotFoundError:
    print("ERROR: CSV files not found. Please ensure the following files exist:")
    print(f"  - {CONFIG['train_csv']}")
    print(f"  - {CONFIG['test_csv']}")
    raise

# Preprocess
print("\nPreprocessing training data...")
train_preprocessor = FracturePreprocessor(train_df)
train_processed = train_preprocessor.compute_deltas_and_features()
train_processed = train_preprocessor.normalize_features(train_processed)

print("Preprocessing test data...")
test_preprocessor = FracturePreprocessor(test_df)
test_preprocessor.stats = train_preprocessor.stats  # Use training stats
test_processed = test_preprocessor.compute_deltas_and_features()
test_processed = test_preprocessor.normalize_features(test_processed)

print(f"\nTraining processed: {len(train_processed)} points")
print(f"Test processed: {len(test_processed)} points")

# Create datasets
print("\nCreating datasets...")
train_dataset = FractureDataset(train_processed, history_window=CONFIG['history_window'], is_train=True)
test_dataset = FractureDataset(test_processed, history_window=CONFIG['history_window'], is_train=False)

print(f"Training fractures: {len(train_dataset)}")
print(f"Test fractures: {len(test_dataset)}")

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")


# ## 8. Model Initialization

# In[ ]:


print("\n" + "="*80)
print("MODEL INITIALIZATION")
print("="*80)

input_dim = 8  # Number of features per point
model = FractureGeneratorModel(
    input_dim=8,  # Fixed input dim based on features
    hidden_dim=CONFIG['gru_hidden_dim'],
    num_mixtures=CONFIG['num_mixtures'],
    cnn_channels=CONFIG['cnn_filters'],
    kernel_size=CONFIG['cnn_kernel_size']
).to(CONFIG['device'])

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("\nModel architecture:")
print(model)


# ## 9. Training Loop

# In[ ]:


def train_epoch(model, dataloader, optimizer, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mdn_loss = 0.0
    total_stop_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        features = batch['features'].to(CONFIG['device'])
        targets = batch['targets']
        batch_size = len(targets)
        
        optimizer.zero_grad()
        
        # Forward pass
        pi, mu_r, sigma_r, mu_theta, sigma_theta, p_stop, _ = model(features)
        
        # Prepare target tensors
        target_r_list = []
        target_theta_list = []
        target_stop_list = []
        
        for i in range(batch_size):
            target_r_list.append(targets[i]['log_delta_r'])
            target_theta_list.append(targets[i]['delta_theta'])
            target_stop_list.append(targets[i]['is_last'])
        
        target_r = torch.tensor(target_r_list, dtype=torch.float32).to(CONFIG['device'])
        target_theta = torch.tensor(target_theta_list, dtype=torch.float32).to(CONFIG['device'])
        target_stop = torch.tensor(target_stop_list, dtype=torch.float32).to(CONFIG['device'])
        
        # MDN loss
        batch_mdn_loss = mdn_loss(
            pi, mu_r, sigma_r, mu_theta, sigma_theta, target_r, target_theta
        )
        
        # Stop loss
        batch_stop_loss = F.binary_cross_entropy(
            p_stop.squeeze(), target_stop, reduction='mean'
        )
        
        # Total loss
        loss = batch_mdn_loss + CONFIG['lambda_stop'] * batch_stop_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mdn_loss += batch_mdn_loss.item()
        total_stop_loss += batch_stop_loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    
    avg_loss = total_loss / num_batches
    avg_mdn = total_mdn_loss / num_batches
    avg_stop = total_stop_loss / num_batches
    
    return avg_loss, avg_mdn, avg_stop

# Training
print("\n" + "="*80)
print("TRAINING")
print("="*80)

best_loss = float('inf')
train_losses = []
mdn_losses = []
stop_losses = []

for epoch in range(CONFIG['num_epochs']):
    avg_loss, avg_mdn, avg_stop = train_epoch(
        model, train_loader, optimizer, epoch, CONFIG['num_epochs']
    )
    
    train_losses.append(avg_loss)
    mdn_losses.append(avg_mdn)
    stop_losses.append(avg_stop)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | "
              f"Loss: {avg_loss:.4f} | MDN: {avg_mdn:.4f} | Stop: {avg_stop:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': CONFIG,
            'preprocessor_stats': train_preprocessor.stats
        }, f"{CONFIG['models_dir']}/best_model.pt")
        if (epoch + 1) % 10 == 0:
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

print("\nTraining complete!")


# ## 10. Training Visualization

# In[ ]:


# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, linewidth=2, color='b')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Total Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(mdn_losses, linewidth=2, color='g')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MDN Loss (NLL)')
axes[1].set_title('MDN Component Loss')
axes[1].grid(True, alpha=0.3)

axes[2].plot(stop_losses, linewidth=2, color='r')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Stop Loss (BCE)')
axes[2].set_title('Stopping Component Loss')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['results_dir']}/training_loss.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal loss: {train_losses[-1]:.6f}")
print(f"Best loss: {best_loss:.6f}")


# ## 11. Inference and Sampling

# In[ ]:


def sample_from_mdn(pi, mu_r, sigma_r, mu_theta, sigma_theta, temperature=1.0, r_quantiles=None):
    """Sample from MDN with temperature and clamping."""
    # Sample mixture component
    pi_np = pi.cpu().detach().numpy()
    k = np.random.choice(len(pi_np), p=pi_np)
    
    # Sample from selected Gaussian
    r_sample = torch.normal(mu_r[k], sigma_r[k] * temperature).item()
    theta_sample = torch.normal(mu_theta[k], sigma_theta[k] * temperature).item()
    
    # Clamp log(Δr) to training quantiles
    if r_quantiles is not None:
        r_sample = np.clip(r_sample, r_quantiles[0], r_quantiles[1])
    
    # Convert back to actual Δr
    delta_r = np.exp(r_sample)
    
    # Wrap angle to [-π, π]
    delta_theta = np.arctan2(np.sin(theta_sample), np.cos(theta_sample))
    
    return delta_r, delta_theta

def generate_fracture(model, seed_fracture_df, preprocessor_stats, max_steps=200, 
                     stop_threshold=0.5, temperature=1.0):
    """Generate a complete fracture path from seed points."""
    model.eval()
    
    # Extract seed points
    n_seed = min(CONFIG['history_window'], len(seed_fracture_df))
    seed_df = seed_fracture_df.iloc[:n_seed].copy()
    
    # Get normalized coordinates from seed
    xs_seed = seed_df['coord_x_norm'].values
    ys_seed = seed_df['coord_y_norm'].values
    
    generated_x = list(xs_seed)
    generated_y = list(ys_seed)
    generated_lengths = []
    generated_angles = []
    
    # Start from last seed point
    current_x = xs_seed[-1]
    current_y = ys_seed[-1]
    
    # Track current angle
    if len(seed_df) >= 2:
        last_angle = seed_df.iloc[-1]['delta_theta']
        if np.isnan(last_angle) or last_angle == 0:
            last_angle = np.arctan2(ys_seed[-1] - ys_seed[-2], xs_seed[-1] - xs_seed[-2])
    else:
        last_angle = 0.0
    
    current_angle = last_angle
    
    # Get fracture-level statistics
    mean_curvature = seed_df['mean_curvature'].iloc[0]
    length_variance = seed_df['length_variance'].iloc[0]
    tortuosity = seed_df['tortuosity'].iloc[0]
    
    # Build feature names
    feature_names = [
        'log_delta_r_norm', 'sin_theta', 'cos_theta',
        'delta_angle', 'curvature_trajectory',
        'mean_curvature', 'length_variance', 'tortuosity'
    ]
    
    # Initialize history with seed features
    history_features = []
    for i in range(n_seed):
        feature_vec = []
        for fname in feature_names:
            if fname in seed_df.columns:
                val = seed_df.iloc[i][fname]
                feature_vec.append(val if not np.isnan(val) else 0.0)
            else:
                feature_vec.append(0.0)
        history_features.append(feature_vec)
    
    # Keep track of previous delta_angle for curvature
    prev_delta_angle = 0.0
    if n_seed > 0 and 'delta_angle' in seed_df.columns:
        prev_delta_angle = seed_df.iloc[-1]['delta_angle']
        if np.isnan(prev_delta_angle):
            prev_delta_angle = 0.0
    
    with torch.no_grad():
        for step in range(max_steps):
            # Prepare input
            input_features = np.zeros((CONFIG['history_window'], len(feature_names)))
            actual_history_len = len(history_features)
            
            # Fill with actual history
            start_idx = max(0, actual_history_len - CONFIG['history_window'])
            for i in range(start_idx, actual_history_len):
                input_features[i - start_idx] = history_features[i]
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(CONFIG['device'])
            
            # Forward pass
            pi, mu_r, sigma_r, mu_theta, sigma_theta, p_stop, _ = model(features_tensor)
            
            # Check stopping condition
            if p_stop.item() > stop_threshold and step > 5:
                break
            
            # Sample next step
            delta_r, delta_theta = sample_from_mdn(
                pi[0], mu_r[0], sigma_r[0], mu_theta[0], sigma_theta[0],
                temperature=temperature,
                r_quantiles=preprocessor_stats.get('delta_r_quantiles')
            )
            
            # Update angle
            current_angle = delta_theta
            
            # Update position
            current_x += delta_r * np.cos(current_angle)
            current_y += delta_r * np.sin(current_angle)
            
            generated_x.append(current_x)
            generated_y.append(current_y)
            generated_lengths.append(delta_r)
            generated_angles.append(current_angle)
            
            # Compute delta_angle
            if len(generated_angles) >= 2:
                delta_angle = angle_difference(generated_angles[-2], generated_angles[-1])
            else:
                delta_angle = 0.0
            
            # Compute curvature
            curvature = delta_angle - prev_delta_angle
            prev_delta_angle = delta_angle
            
            # Normalize log_delta_r
            log_r = np.log(delta_r + CONFIG['epsilon'])
            log_r_norm = (log_r - preprocessor_stats['log_delta_r_mean']) / (preprocessor_stats['log_delta_r_std'] + CONFIG['epsilon'])
            
            # Build new feature vector
            new_features = [
                log_r_norm,
                np.sin(current_angle),
                np.cos(current_angle),
                delta_angle,
                curvature,
                mean_curvature,
                length_variance,
                tortuosity
            ]
            
            # Add to history
            history_features.append(new_features)
            
            if len(generated_x) >= max_steps + n_seed:
                break
    
    generated_path = np.column_stack([generated_x, generated_y])
    return generated_path, generated_lengths, generated_angles


# ## 12. Evaluation

# In[ ]:


def compute_hausdorff_distance(path1, path2):
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

def extract_segment_lengths(path):
    """Return segment lengths of a path."""
    p = np.array(path)
    return np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) >= 2 else np.array([])

def extract_segment_angles(path):
    """Return segment angles (radians) of a path."""
    p = np.array(path)
    if len(p) < 2:
        return np.array([])
    d = np.diff(p, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])

def compute_kl_divergence(p_samples, q_samples, bins=50):
    """KL divergence D(P||Q) via histogram approximation."""
    if len(p_samples) == 0 or len(q_samples) == 0:
        return float('nan')
    combined = np.concatenate([p_samples, q_samples])
    edges = np.linspace(combined.min() - 1e-6, combined.max() + 1e-6, bins + 1)
    ph, _ = np.histogram(p_samples, bins=edges, density=True)
    qh, _ = np.histogram(q_samples, bins=edges, density=True)
    eps = 1e-10
    ph = (ph + eps) / (ph + eps).sum()
    qh = (qh + eps) / (qh + eps).sum()
    return float(np.sum(ph * np.log(ph / qh)))

# Visualization functions preserved in visualization_utils.py
# plot_fracture_generation_comparison and plot_generation_progression are imported

# Load best model and evaluate
print("\n" + "="*80)
print("EVALUATION")
print("="*80)

print("\nLoading best model...")
checkpoint = torch.load(f"{CONFIG['models_dir']}/best_model.pt", map_location=CONFIG['device'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# -----------------------------------------------------------------------
# Single-step prediction metrics on the full test set
# For MDN: use expected value (weighted mean of mixture components)
# and convert from polar to Cartesian coordinates
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("SINGLE-STEP PREDICTION METRICS")
print("="*60)

ss_true_xy = []
ss_pred_xy = []

model.eval()
with torch.no_grad():
    for frac_df in test_dataset.fractures:
        n = len(frac_df)
        hw = CONFIG['history_window']
        if n < hw + 1:
            continue

        # Current position: last point of the seed window
        cur_x = float(frac_df.iloc[hw - 1]['coord_x_norm'])
        cur_y = float(frac_df.iloc[hw - 1]['coord_y_norm'])

        # True next position
        true_x_next = float(frac_df.iloc[hw]['coord_x_norm'])
        true_y_next = float(frac_df.iloc[hw]['coord_y_norm'])

        # Build feature tensor (seed window)
        feat_names = [
            'log_delta_r_norm', 'sin_theta', 'cos_theta',
            'delta_angle', 'curvature_trajectory',
            'mean_curvature', 'length_variance', 'tortuosity'
        ]
        feats = np.zeros((n, len(feat_names)))
        for fi, fname in enumerate(feat_names):
            if fname in frac_df.columns:
                vals = frac_df[fname].values
                feats[:, fi] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad to full length (as collate_fn does) and feed model
        feats_tensor = torch.FloatTensor(feats).unsqueeze(0).to(CONFIG['device'])
        pi_out, mu_r_out, sigma_r_out, mu_theta_out, sigma_theta_out, p_stop_out, _ = model(feats_tensor)
        pi_np      = pi_out[0].cpu().numpy()        # [K]
        mu_r_np    = mu_r_out[0].cpu().numpy()      # [K] (log_delta_r)
        mu_theta_np = mu_theta_out[0].cpu().numpy() # [K]

        # Expected values
        exp_log_r = float(np.sum(pi_np * mu_r_np))
        exp_delta_r = np.exp(np.clip(exp_log_r, -10, 10))
        exp_theta = float(np.sum(pi_np * mu_theta_np))

        # Predicted next Cartesian position
        pred_x_next = cur_x + exp_delta_r * np.cos(exp_theta)
        pred_y_next = cur_y + exp_delta_r * np.sin(exp_theta)

        ss_true_xy.append([true_x_next, true_y_next])
        ss_pred_xy.append([pred_x_next, pred_y_next])

if ss_true_xy:
    y_true_ss = np.array(ss_true_xy)
    y_pred_ss = np.array(ss_pred_xy)

    mse_ss   = float(np.mean((y_pred_ss - y_true_ss) ** 2))
    rmse_ss  = float(np.sqrt(mse_ss))
    mae_ss   = float(np.mean(np.abs(y_pred_ss - y_true_ss)))
    mse_x_ss = float(np.mean((y_pred_ss[:, 0] - y_true_ss[:, 0]) ** 2))
    mse_y_ss = float(np.mean((y_pred_ss[:, 1] - y_true_ss[:, 1]) ** 2))
    mae_x_ss = float(np.mean(np.abs(y_pred_ss[:, 0] - y_true_ss[:, 0])))
    mae_y_ss = float(np.mean(np.abs(y_pred_ss[:, 1] - y_true_ss[:, 1])))

    print(f"  MSE:   {mse_ss:.6f}  RMSE: {rmse_ss:.6f}  MAE: {mae_ss:.6f}")
    print(f"  MSE-X: {mse_x_ss:.6f}  MSE-Y: {mse_y_ss:.6f}")
    print(f"  MAE-X: {mae_x_ss:.6f}  MAE-Y: {mae_y_ss:.6f}")
    print(f"  N test samples: {len(y_true_ss)}")

    pd.DataFrame([{
        'mse': mse_ss, 'rmse': rmse_ss, 'mae': mae_ss,
        'mse_x': mse_x_ss, 'mse_y': mse_y_ss,
        'mae_x': mae_x_ss, 'mae_y': mae_y_ss,
        'n_samples': len(y_true_ss)
    }]).to_csv(f"{CONFIG['results_dir']}/evaluation_metrics.csv", index=False)
    print(f"Single-step metrics saved to {CONFIG['results_dir']}/evaluation_metrics.csv")

print("\nGenerating fractures on ALL test fractures...")

all_metrics = {
    'hausdorff': [],
    'frechet': [],
    'endpoint_error': [],
    'length_error': [],
    'path_similarity': []
}
path_rows = []
gen_segment_lengths_all = []
true_segment_lengths_all = []
gen_segment_angles_all = []
true_segment_angles_all = []
# For stratified analysis: per-fracture records
strat_records = []  # {fid, n_pts, mean_curvature, hausdorff}

for idx in range(len(test_dataset)):
    try:
        test_frac = test_dataset.fractures[idx]
        fracture_id = int(test_frac['fracture_id'].iloc[0])

        # Extract true path
        true_x = test_frac['coord_x_norm'].values
        true_y = test_frac['coord_y_norm'].values
        true_path = np.column_stack([true_x, true_y])

        # Generate fracture
        gen_path, gen_lengths, gen_angles = generate_fracture(
            model,
            test_frac,
            train_preprocessor.stats,
            max_steps=CONFIG['max_steps'],
            stop_threshold=CONFIG['stop_threshold'],
            temperature=1.0
        )

        # Compute metrics
        if len(gen_path) > 1:
            hausdorff = compute_hausdorff_distance(true_path, gen_path)
            if hausdorff != float('inf'):
                all_metrics['hausdorff'].append(hausdorff)

            frechet = discrete_frechet_distance(true_path, gen_path)
            if frechet != float('inf'):
                all_metrics['frechet'].append(frechet)

            if len(gen_path) > 0 and len(true_path) > 0:
                endpoint_error = np.linalg.norm(gen_path[-1] - true_path[-1])
                all_metrics['endpoint_error'].append(endpoint_error)
            else:
                endpoint_error = float('inf')

            path_sim = compute_path_similarity(true_path, gen_path)
            all_metrics['path_similarity'].append(path_sim)

            gen_total_length = sum(gen_lengths) if gen_lengths else 0
            true_lengths_col = test_frac['delta_r'].values
            true_lengths_valid = true_lengths_col[true_lengths_col > 0]
            true_total_length = float(sum(true_lengths_valid))
            if true_total_length > 0:
                length_error = abs(gen_total_length - true_total_length) / true_total_length
                all_metrics['length_error'].append(length_error)
            else:
                length_error = float('nan')

            path_rows.append({
                'fracture_id': fracture_id,
                'hausdorff': hausdorff,
                'frechet': frechet,
                'endpoint_error': endpoint_error,
                'length_error': length_error,
                'path_similarity': path_sim,
                'true_n_pts': len(true_path),
                'gen_n_pts': len(gen_path)
            })

            # Collect segment distributions for Wasserstein/KL
            gen_segment_lengths_all.extend(extract_segment_lengths(gen_path).tolist())
            true_segment_lengths_all.extend(extract_segment_lengths(true_path).tolist())
            gen_segment_angles_all.extend(extract_segment_angles(gen_path).tolist())
            true_segment_angles_all.extend(extract_segment_angles(true_path).tolist())

            # Stratification record: use n_pts and mean_curvature from test_frac
            mean_kappa = float(test_frac['mean_curvature'].mean()) if 'mean_curvature' in test_frac.columns else 0.0
            strat_records.append({
                'fracture_id': fracture_id,
                'n_pts': len(true_path),
                'mean_curvature': mean_kappa,
                'hausdorff': hausdorff if hausdorff != float('inf') else float('nan')
            })

            # Extract seed points (first history_window points)
            seed_points = true_path[:min(CONFIG['history_window'], len(true_path))]
            metrics = {
                'hausdorff': hausdorff,
                'endpoint_error': endpoint_error,
                'length_error': length_error
            }
            
            try:
                if VISUALIZATION_UTILS_AVAILABLE:
                    plot_fracture_generation_comparison(true_path, gen_path, seed_points,
                                                      fracture_id, CONFIG['plots_dir'], metrics)
                    plot_generation_progression(true_path, gen_path, seed_points,
                                               fracture_id, CONFIG['plots_dir'])
            except Exception as viz_error:
                print(f"  Warning: Visualization failed for fracture {fracture_id}: {viz_error}")

            print(f"Fracture {fracture_id}: "
                  f"Hausdorff={hausdorff:.4f}, Frechet={frechet:.4f}, "
                  f"Endpoint={endpoint_error:.4f}, LenErr={length_error:.4f}, "
                  f"PathSim={path_sim:.4f}, "
                  f"Generated {len(gen_path)} pts (True: {len(true_path)})")
        else:
            print(f"Fracture {fracture_id}: Generation failed")
    except Exception as e:
        print(f"Error processing fracture {idx}: {e}")
        continue

# -----------------------------------------------------------------------
# Summary + save path metrics CSV
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Distributional metrics: Wasserstein (Length, Angle) + KL Divergence
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("DISTRIBUTIONAL METRICS")
print("="*60)

true_seg_len = np.array(true_segment_lengths_all)
gen_seg_len  = np.array(gen_segment_lengths_all)
true_seg_ang = np.array(true_segment_angles_all)
gen_seg_ang  = np.array(gen_segment_angles_all)

# Save raw segment distributions for visualization scripts
if len(true_seg_len) > 0 or len(gen_seg_len) > 0:
    pd.DataFrame({
        'type': ['true'] * len(true_seg_len) + ['generated'] * len(gen_seg_len),
        'value': np.concatenate([true_seg_len, gen_seg_len]) if len(true_seg_len) > 0 and len(gen_seg_len) > 0 else (true_seg_len if len(true_seg_len) > 0 else gen_seg_len)
    }).to_csv(f"{CONFIG['results_dir']}/segment_lengths.csv", index=False)
if len(true_seg_ang) > 0 or len(gen_seg_ang) > 0:
    pd.DataFrame({
        'type': ['true'] * len(true_seg_ang) + ['generated'] * len(gen_seg_ang),
        'value': np.concatenate([true_seg_ang, gen_seg_ang]) if len(true_seg_ang) > 0 and len(gen_seg_ang) > 0 else (true_seg_ang if len(true_seg_ang) > 0 else gen_seg_ang)
    }).to_csv(f"{CONFIG['results_dir']}/segment_angles.csv", index=False)
print(f"Segment distributions saved to {CONFIG['results_dir']}/")

w_length = float(wasserstein_distance(true_seg_len, gen_seg_len)) if len(true_seg_len) > 0 and len(gen_seg_len) > 0 else float('nan')
w_angle  = float(wasserstein_distance(true_seg_ang, gen_seg_ang)) if len(true_seg_ang) > 0 and len(gen_seg_ang)  > 0 else float('nan')
kl_div   = compute_kl_divergence(gen_seg_len, true_seg_len)

print(f"  Wasserstein (Length): {w_length:.4f}")
print(f"  Wasserstein (Angle):  {w_angle:.4f}")
print(f"  KL Divergence:        {kl_div:.4f}")

pd.DataFrame([{
    'model': 'CNN-GRU-MDN',
    'wasserstein_length': w_length,
    'wasserstein_angle': w_angle,
    'kl_divergence': kl_div
}]).to_csv(f"{CONFIG['results_dir']}/distributional_metrics.csv", index=False)
print(f"Distributional metrics saved to {CONFIG['results_dir']}/distributional_metrics.csv")

# -----------------------------------------------------------------------
# Stratified Hausdorff by fracture LENGTH
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("LENGTH-STRATIFIED HAUSDORFF")
print("="*60)

if strat_records:
    strat_df = pd.DataFrame(strat_records)

    length_bins = [
        ('Short (3-5 pts)',      3,  5),
        ('Medium (6-15 pts)',    6, 15),
        ('Long (16-30 pts)',    16, 30),
        ('Very Long (>30 pts)', 31, 9999),
    ]

    length_rows = []
    for label, lo, hi in length_bins:
        mask = (strat_df['n_pts'] >= lo) & (strat_df['n_pts'] <= hi)
        subset = strat_df.loc[mask, 'hausdorff'].dropna()
        subset = subset[np.isfinite(subset)]
        count = int(mask.sum())
        h_mean = float(np.mean(subset)) if len(subset) > 0 else float('nan')
        h_std  = float(np.std(subset))  if len(subset) > 0 else float('nan')
        length_rows.append({
            'category': label, 'n_fractures': count,
            'hausdorff_mean': h_mean, 'hausdorff_std': h_std
        })
        print(f"  {label:25s}: N={count:3d}, Hausdorff={h_mean:.4f} ± {h_std:.4f}")

    pd.DataFrame(length_rows).to_csv(f"{CONFIG['results_dir']}/length_stratified_metrics.csv", index=False)
    print(f"Length-stratified metrics saved to {CONFIG['results_dir']}/length_stratified_metrics.csv")

    # -----------------------------------------------------------------------
    # Stratified Hausdorff by fracture CURVATURE
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("CURVATURE-STRATIFIED HAUSDORFF")
    print("="*60)

    curvature_bins = [
        ('Low (kappa < 0.01)',          0.0,   0.01),
        ('Medium (0.01 <= kappa < 0.05)', 0.01,  0.05),
        ('High (kappa >= 0.05)',         0.05, 9999.0),
    ]

    curvature_rows = []
    for label, lo, hi in curvature_bins:
        mask = (strat_df['mean_curvature'] >= lo) & (strat_df['mean_curvature'] < hi)
        subset = strat_df.loc[mask, 'hausdorff'].dropna()
        subset = subset[np.isfinite(subset)]
        count = int(mask.sum())
        h_mean = float(np.mean(subset)) if len(subset) > 0 else float('nan')
        h_std  = float(np.std(subset))  if len(subset) > 0 else float('nan')
        curvature_rows.append({
            'category': label, 'n_fractures': count,
            'hausdorff_mean': h_mean, 'hausdorff_std': h_std
        })
        print(f"  {label:35s}: N={count:3d}, Hausdorff={h_mean:.4f} ± {h_std:.4f}")

    pd.DataFrame(curvature_rows).to_csv(f"{CONFIG['results_dir']}/curvature_stratified_metrics.csv", index=False)
    print(f"Curvature-stratified metrics saved to {CONFIG['results_dir']}/curvature_stratified_metrics.csv")


# ## 13. Results and Conclusions
# 
# ### Key Features of CNN-GRU-MDN Model:
# 
# 1. **Dilated CNN Encoder**: Captures multi-scale patterns with expanding receptive field
# 2. **Bidirectional GRU**: Models temporal dependencies in both directions
# 3. **Additive Attention**: Focuses on relevant historical context
# 4. **MDN Output**: Probabilistic predictions with K=3 Gaussian mixtures
#    - Captures uncertainty and multi-modality
#    - Enables diverse path sampling
# 5. **Stopping Prediction**: Joint prediction of coordinates and termination
# 
# ### Advantages:
# - **Probabilistic**: Models inherent uncertainty in fracture propagation
# - **Multi-modal**: Can represent multiple possible next steps
# - **Hierarchical**: CNN extracts features, GRU models sequence
# - **Flexible**: Temperature parameter controls sampling diversity
# 
# ### Potential Improvements:
# 1. Adaptive mixture components (K)
# 2. Physics-informed constraints on sampling
# 3. Beam search for best paths
# 4. Conditional generation based on fracture properties
# 5. Variational inference for better uncertainty quantification

# In[ ]:


print("\n" + "="*80)
print("CASE 4: CNN-GRU-MDN - COMPLETE")
print("="*80)
print(f"\nResults saved to: {CONFIG['results_dir']}/")
print(f"Plots saved to: {CONFIG['plots_dir']}/")
print(f"Best model saved to: {CONFIG['models_dir']}/best_model.pt")
print("\nThis implementation is adapted from fracture_network_oct_nov_2025.py")

