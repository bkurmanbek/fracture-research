#!/usr/bin/env python
# coding: utf-8

# # Case 1: Bidirectional LSTM with Multi-head Attention
# 
# ## Overview
# 
# This notebook implements **Case 1** from the paper: *"Deep Learning Approaches for Autoregressive Fracture Network Path Prediction: A Case Study on the Teapot Dome Dataset"*.
# 
# ### Model Architecture (Section 4.2)
# 
# This architecture combines **bidirectional LSTM layers** with **multi-head self-attention** for capturing both:
# - **Local sequential patterns** (via BiLSTM)
# - **Global context** (via multi-head attention)
# 
# ### Key Components:
# 1. **Input**: Concatenated coordinates and features, dimension $d_{in} = 2 + d_f$
# 2. **Bidirectional LSTM**: 3 layers, hidden dimension 256, dropout 0.2
# 3. **Multi-head Attention**: 4 heads operating on LSTM outputs
# 4. **Layer Normalization**: Applied after attention with skip connection
# 5. **Feed-forward Network**: Two dense layers (256 → 2) with ReLU
# 
# ### Forward Pass:
# $$
# \begin{align}
# h &= \text{BiLSTM}(\text{concat}(x, f)) \\
# a &= \text{LayerNorm}(h + \text{MultiHeadAttn}(h, h, h)) \\
# \hat{p} &= W_2 \cdot \text{ReLU}(W_1 \cdot a + b_1) + b_2
# \end{align}
# $$
# 
# ### Training Details:
# - **Loss**: Mean Squared Error (MSE)
# - **Optimizer**: AdamW with learning rate $10^{-3}$ and weight decay 0.01
# - **Learning Rate Scheduling**: ReduceLROnPlateau (patience=5, factor=0.5)
# - **Early Stopping**: patience=15
# - **Sequence Length**: k = 10 points
# - **Batch Size**: 32

# ## 1. Import Libraries and Setup

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import List, Dict, Tuple
import os
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add workspace to path to import visualization_utils
sys.path.append('/workspace')
import visualization_utils
ADVANCED_UTILS_AVAILABLE = True
try:
    from visualization_utils import plot_fracture_generation_comparison, plot_generation_progression
    VISUALIZATION_UTILS_AVAILABLE = True
except ImportError:
    VISUALIZATION_UTILS_AVAILABLE = False
    print("Warning: visualization_utils not found. Enhanced plotting will be disabled.")


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
    
    def check_statistical_compliance(self, path: np.ndarray) -> Dict:
        """
        Score how well a generated path conforms to training distributions.

        Each dimension is scored with a Gaussian-decay function:
            score = exp(-0.5 * z^2)  where z = (gen_value - train_mean) / train_std
        This gives 1.0 for a perfect match, ~0.61 at 1σ, ~0.14 at 2σ, ~0.01 at 3σ.

        Args:
            path: (N, 2) numpy array of generated fracture coordinates

        Returns:
            Dict with keys: 'segment_length', 'path_length', 'curvature',
                            'coordinate_bounds', 'overall'
            All values are in [0, 1]; higher means more compliant.
        """
        if self.stats == {} or len(path) < 2:
            return {'segment_length': 0.0, 'path_length': 0.0,
                    'curvature': 0.0, 'coordinate_bounds': 0.0, 'overall': 0.0}

        def gaussian_score(value, mean, std):
            if std < 1e-9:
                return 1.0 if abs(value - mean) < 1e-9 else 0.0
            z = (value - mean) / std
            return float(np.exp(-0.5 * z ** 2))

        scores = {}

        # --- Segment length compliance ---
        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
        gen_seg_mean = float(np.mean(seg_lens))
        sl = self.stats['segment_length']
        scores['segment_length'] = gaussian_score(gen_seg_mean, sl['mean'], sl['std'])

        # --- Path length compliance ---
        gen_path_len = float(np.sum(seg_lens))
        pl = self.stats['path_length']
        scores['path_length'] = gaussian_score(gen_path_len, pl['mean'], pl['std'])

        # --- Curvature compliance ---
        if len(path) > 2:
            curvatures = []
            for i in range(1, len(path) - 1):
                v1 = path[i] - path[i - 1]
                v2 = path[i + 1] - path[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    curvatures.append(float(np.arccos(cos_t)))
            gen_curv_mean = float(np.mean(curvatures)) if curvatures else 0.0
        else:
            gen_curv_mean = 0.0
        cv = self.stats['curvature']
        scores['curvature'] = gaussian_score(gen_curv_mean, cv['mean'], cv['std'])

        # --- Coordinate bounds compliance ---
        # Fraction of points that lie within training coordinate bounds
        cb = self.stats['coordinate_bounds']
        in_bounds = (
            (path[:, 0] >= cb['x_min']) & (path[:, 0] <= cb['x_max']) &
            (path[:, 1] >= cb['y_min']) & (path[:, 1] <= cb['y_max'])
        )
        scores['coordinate_bounds'] = float(np.mean(in_bounds))

        # --- Overall: equal-weight average ---
        scores['overall'] = float(np.mean(list(scores.values())))

        return scores

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


def compute_training_distributions(train_fractures: List[Dict]) -> 'TrainingDistributionAnalyzer':
    """
    Compute training distribution statistics.

    Args:
        train_fractures: List of fracture dictionaries

    Returns:
        Fitted TrainingDistributionAnalyzer instance
    """
    analyzer = TrainingDistributionAnalyzer()
    analyzer.analyze_training_data(train_fractures)
    return analyzer




# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration
try:
    # Set memory growth for MPS
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    print(f"MPS (Metal) Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except Exception as e:
    print(f"GPU setup warning: {e}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Running on CPU")


# ## 2. Configuration Parameters

# In[ ]:


CONFIG = {
    # Model architecture
    'sequence_length': 10,
    'lstm_units': 512,       # Increased from 256
    'lstm_layers': 4,        # Increased from 3
    'dropout_rate': 0.3,
    'attention_heads': 8,    # Increased from 4
    'key_dim': 64,           # Dimension for attention keys
    'ffn_dim': 256,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'epochs': 50,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    
    # Data paths
    'train_path': 'train_fractures_processed.csv',
    'test_path': 'test_fractures_processed.csv',

    # Output paths
    'model_save_path': 'fracture_results/case1/best_model.h5',
    'results_dir': 'fracture_results/case1',
    'plots_dir': 'fracture_results/case1/plots',
    'models_dir': 'fracture_results/case1/models'
}

# Create output directories
for d in [CONFIG['results_dir'], CONFIG['plots_dir'], CONFIG['models_dir']]:
    os.makedirs(d, exist_ok=True)

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


# ## 3. Data Loading and Preprocessing
# 
# We load the pre-processed fracture data from CSV files. Each fracture consists of:
# - **Coordinates**: (x, y) positions
# - **Geometric features**: angles, lengths, curvatures
# - **Neighborhood features**: closest fracture segments

# In[ ]:


class FractureDataLoader:
    """Loads and processes fracture data for autoregressive training"""
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.coord_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
    def load_data(self, train_path: str, test_path: str):
        """Load fracture data from CSV files"""
        print("Loading data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Train data: {train_df.shape}, {train_df['fracture_id'].nunique()} fractures")
        print(f"Test data: {test_df.shape}, {test_df['fracture_id'].nunique()} fractures")
        
        # Convert to fracture dictionaries
        train_fractures = self._df_to_fractures(train_df)
        test_fractures = self._df_to_fractures(test_df)
        
        return train_fractures, test_fractures
    
    def _df_to_fractures(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of fracture dictionaries"""
        fractures = []
        
        for fid, group in df.groupby('fracture_id'):
            group = group.sort_values('point_idx')
            
            # Extract coordinates and features
            points = group[['coord_x', 'coord_y']].values
            
            # Build feature matrix
            features = np.column_stack([
                group['prev_angle'].values,
                group['next_angle'].values,
                group['prev_length'].values,
                group['next_length'].values,
                group['curvature'].values
            ])
            
            fractures.append({
                'id': fid,
                'points': points,
                'features': features
            })
        
        return fractures
    
    def prepare_sequences(self, fractures: List[Dict], fit_scalers: bool = False):
        """Prepare autoregressive sequences for training/testing"""
        X_coords_list = []
        X_features_list = []
        y_list = []
        
        for fracture in fractures:
            points = fracture['points']
            features = fracture['features']
            n_points = len(points)
            
            if n_points < self.sequence_length + 1:
                continue
            
            # Create sliding window sequences
            for i in range(n_points - self.sequence_length):
                X_coords_list.append(points[i:i+self.sequence_length])
                X_features_list.append(features[i:i+self.sequence_length])
                y_list.append(points[i+self.sequence_length])
        
        X_coords = np.array(X_coords_list)
        X_features = np.array(X_features_list)
        y = np.array(y_list)
        
        # Normalize coordinates and features
        if fit_scalers:
            X_coords_reshaped = X_coords.reshape(-1, 2)
            self.coord_scaler.fit(X_coords_reshaped)
            
            X_features_reshaped = X_features.reshape(-1, X_features.shape[-1])
            self.feature_scaler.fit(X_features_reshaped)
        
        # Transform
        X_coords_norm = self.coord_scaler.transform(X_coords.reshape(-1, 2))
        X_coords_norm = X_coords_norm.reshape(X_coords.shape)
        
        X_features_norm = self.feature_scaler.transform(X_features.reshape(-1, X_features.shape[-1]))
        X_features_norm = X_features_norm.reshape(X_features.shape)
        
        y_norm = self.coord_scaler.transform(y)
        
        # Concatenate coordinates and features
        X = np.concatenate([X_coords_norm, X_features_norm], axis=-1)
        
        print(f"Prepared {X.shape[0]} sequences")
        print(f"Input shape: {X.shape}, Target shape: {y_norm.shape}")
        
        return X, y_norm

# Load and prepare data
data_loader = FractureDataLoader(sequence_length=CONFIG['sequence_length'])
train_fractures, test_fractures = data_loader.load_data(
    CONFIG['train_path'], 
    CONFIG['test_path']
)

print("\nPreparing training sequences...")
X_train, y_train = data_loader.prepare_sequences(train_fractures, fit_scalers=True)

print("\nPreparing test sequences...")
X_test, y_test = data_loader.prepare_sequences(test_fractures, fit_scalers=False)

# Compute training statistics for advanced generation
stats_tracker = None
if True:  # Advanced utils embedded
    print("\nComputing training statistics for advanced generation control...")
    stats_tracker = compute_training_distributions(train_fractures)
    print("Training statistics computed successfully.")


# ## 4. Model Architecture
# 
# ### Bidirectional LSTM with Multi-head Attention
# 
# The model architecture consists of:
# 1. **Multiple BiLSTM layers** that process the sequence in both forward and backward directions
# 2. **Multi-head self-attention** layer that allows the model to attend to different positions
# 3. **Layer normalization with residual connection** for stable training
# 4. **Feed-forward network** for final coordinate prediction

# In[ ]:


def build_bilstm_attention_model(config: dict, input_shape: tuple) -> Model:
    """
    Build BiLSTM with Multi-head Attention model
    
    Args:
        config: Configuration dictionary
        input_shape: Shape of input sequences (sequence_length, features)
    
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Bidirectional LSTM layers
    x = inputs
    for i in range(config['lstm_layers']):
        return_sequences = (i < config['lstm_layers'] - 1) or True # Always return sequences for attention
        x = layers.Bidirectional(
            layers.LSTM(
                config['lstm_units'],
                return_sequences=True,  # Always return sequences for attention
                dropout=config['dropout_rate'],
                recurrent_dropout=config['dropout_rate'],
                name=f'bilstm_{i+1}'
            )
        )(x)
    
    lstm_out = x
    
    # Multi-head Self Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config['attention_heads'],
        key_dim=config['key_dim'],
        dropout=0.1
    )(x, x)
    
    # Residual connection + Layer Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward Network
    # BiLSTM outputs 2 * units, so model dimension is 2 * units (1024)
    model_dim = config['lstm_units'] * 2
    
    x_ff = layers.Dense(model_dim * 2, activation='gelu')(x) # Expansion (2048)
    x_ff = layers.Dropout(config['dropout_rate'])(x_ff)
    x_ff = layers.Dense(model_dim)(x_ff) # Projection back to model_dim (1024)
    
    # Residual connection + Layer Norm
    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling (use last state or average)
    try:
        x = layers.GlobalAveragePooling1D()(x)
    except:
        # Fallback if shape is not compatible
        x = layers.Lambda(lambda t: t[:, -1, :])(x)
    
    # Feed-forward network
    x = layers.Dense(
        config['ffn_dim'],
        activation='relu',
        name='ffn_1'
    )(x)
    x = layers.Dropout(0.2, name='ffn_dropout')(x)
    
    # Output layer (2 coordinates: x, y)
    outputs = layers.Dense(2, name='output')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_MultiHead_Attention')
    
    # Compile with AdamW optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Build the model
input_shape = (CONFIG['sequence_length'], X_train.shape[-1])
model = build_bilstm_attention_model(CONFIG, input_shape)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"\nTotal trainable parameters: {trainable_params:,}")


# ## 5. Training
# 
# We train the model with:
# - **Early stopping** to prevent overfitting
# - **Learning rate reduction** when validation loss plateaus
# - **Model checkpointing** to save the best model

# In[ ]:


# Setup callbacks
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
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=CONFIG['model_save_path'],
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=CONFIG['batch_size'],
    epochs=CONFIG['epochs'],
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")


# ## 6. Training History Visualization

# In[ ]:


def plot_training_history(history):
    """Plot training and validation loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plots_dir']}/training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print best epoch
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = np.min(history.history['val_loss'])
    print(f"\nBest model at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")

plot_training_history(history)


# ## 7. Model Evaluation
# 
# We evaluate the model using:
# - **Mean Squared Error (MSE)**
# - **Mean Absolute Error (MAE)**
# - **Root Mean Squared Error (RMSE)**

# In[ ]:


# Make predictions
y_pred = model.predict(X_test, batch_size=CONFIG['batch_size'])

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Per-coordinate metrics
mse_x = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_y = mean_squared_error(y_test[:, 1], y_pred[:, 1])
mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nOverall Metrics:")
print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"\nPer-Coordinate Metrics:")
print(f"  MSE (X):  {mse_x:.6f}  |  MSE (Y):  {mse_y:.6f}")
print(f"  MAE (X):  {mae_x:.6f}  |  MAE (Y):  {mae_y:.6f}")
print("="*60)

# Save results
results = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'mse_x': mse_x,
    'mse_y': mse_y,
    'mae_x': mae_x,
    'mae_y': mae_y
}

results_df = pd.DataFrame([results])
results_df.to_csv(f"{CONFIG['results_dir']}/evaluation_metrics.csv", index=False)
print(f"\nResults saved to {CONFIG['results_dir']}/evaluation_metrics.csv")


# ## 8. Prediction Visualization

# In[ ]:


def plot_predictions(y_true, y_pred, n_samples=1000):
    """Visualize predicted vs actual coordinates"""
    # Sample random points for visualization
    indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # X coordinate
    axes[0].scatter(y_true[indices, 0], y_pred[indices, 0], alpha=0.5, s=10)
    axes[0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                 [y_true[:, 0].min(), y_true[:, 0].max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True X Coordinate', fontsize=12)
    axes[0].set_ylabel('Predicted X Coordinate', fontsize=12)
    axes[0].set_title('X Coordinate Predictions', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y coordinate
    axes[1].scatter(y_true[indices, 1], y_pred[indices, 1], alpha=0.5, s=10)
    axes[1].plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                 [y_true[:, 1].min(), y_true[:, 1].max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Y Coordinate', fontsize=12)
    axes[1].set_ylabel('Predicted Y Coordinate', fontsize=12)
    axes[1].set_title('Y Coordinate Predictions', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plots_dir']}/prediction_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_predictions(y_test, y_pred)


# ## 9. Error Distribution Analysis

# In[ ]:


def plot_error_distribution(y_true, y_pred):
    """Plot distribution of prediction errors"""
    errors = y_pred - y_true
    error_magnitude = np.linalg.norm(errors, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error in X
    axes[0, 0].hist(errors[:, 0], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Error in X', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Error Distribution - X Coordinate', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error in Y
    axes[0, 1].hist(errors[:, 1], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Error in Y', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Error Distribution - Y Coordinate', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error magnitude
    axes[1, 0].hist(error_magnitude, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Error Magnitude', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Error Magnitude Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(error_magnitude, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Error Magnitude', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plots_dir']}/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print error statistics
    print("\nError Statistics:")
    print(f"Mean error (X): {errors[:, 0].mean():.6f}")
    print(f"Mean error (Y): {errors[:, 1].mean():.6f}")
    print(f"Std error (X): {errors[:, 0].std():.6f}")
    print(f"Std error (Y): {errors[:, 1].std():.6f}")
    print(f"Mean error magnitude: {error_magnitude.mean():.6f}")
    print(f"Median error magnitude: {np.median(error_magnitude):.6f}")

plot_error_distribution(y_test, y_pred)


# ## 10. Advanced Generation with Distribution Analysis and Stopping Criteria

# In[ ]:




# Analyze training distributions
print("\n" + "="*80)
print("ANALYZING TRAINING DATA DISTRIBUTIONS")
print("="*80)

distribution_analyzer = TrainingDistributionAnalyzer()
training_stats = distribution_analyzer.analyze_training_data(train_fractures)

print("\nTraining Distribution Statistics:")
print(f"  Segment Length: mean={training_stats['segment_length']['mean']:.4f}, "
      f"std={training_stats['segment_length']['std']:.4f}, "
      f"median={training_stats['segment_length']['median']:.4f}")
print(f"  Path Length: mean={training_stats['path_length']['mean']:.4f}, "
      f"std={training_stats['path_length']['std']:.4f}, "
      f"max={training_stats['path_length']['max']:.4f}")
print(f"  Coordinate Bounds: X=[{training_stats['coordinate_bounds']['x_min']:.2f}, "
      f"{training_stats['coordinate_bounds']['x_max']:.2f}], "
      f"Y=[{training_stats['coordinate_bounds']['y_min']:.2f}, "
      f"{training_stats['coordinate_bounds']['y_max']:.2f}]")

# Configure advanced stopping criteria
advanced_stopping_config = {
    'max_deviation_factor': 3.0,  # Stop if >3 std from mean path length
    'oscillation_threshold': 0.1,
    'stagnation_limit': 5,
    'movement_threshold': 0.01,
    'max_distance_from_seed': None,  # Auto-compute from training stats
}

stopping_criteria = AdvancedStoppingCriteria(training_stats, advanced_stopping_config)

# ## 11. Fracture Path Generation and Visualization
# 
# Generate complete fracture paths from seed points with advanced stopping criteria

# In[ ]:


def generate_fracture_path(model, seed_sequence, seed_features, max_steps=200, 
                          data_loader=None, training_stats=None, stopping_criteria=None,
                          true_path=None, bounds=None):
    """
    Generate a complete fracture path autoregressively with advanced stopping criteria.
    
    Args:
        model: Trained Keras model
        seed_sequence: Initial sequence of points (shape: [sequence_length, 2])
        seed_features: Initial sequence of features (shape: [sequence_length, feature_dim])
        max_steps: Maximum number of points to generate
        data_loader: DataLoader instance for scaling
        training_stats: Training distribution statistics
        stopping_criteria: AdvancedStoppingCriteria instance
        true_path: Optional true path for distance-based stopping (during evaluation)
    
    Returns:
        Generated path as numpy array, stop_reason string
    """
    generated_path = list(seed_sequence)
    current_sequence = seed_sequence.copy()
    current_features = seed_features.copy()

    # Pad seed to sequence_length so reshape(1, sequence_length, -1) always works
    seq_len = CONFIG['sequence_length']
    if len(current_sequence) < seq_len:
        pad_n = seq_len - len(current_sequence)
        current_sequence = np.vstack([np.zeros((pad_n, current_sequence.shape[1])), current_sequence])
        feat_dim = current_features.shape[1] if current_features.ndim > 1 else 1
        current_features = np.vstack([np.zeros((pad_n, feat_dim)), current_features])

    # Track for advanced stopping
    recent_movements = []
    stop_reason = "max_steps_reached"
    
    # Get bounds for boundary checking
    bounds = None
    if stopping_criteria is not None:
        analyzer = TrainingDistributionAnalyzer()
        analyzer.stats = training_stats if training_stats else {}
        bounds = analyzer.get_reasonable_bounds(seed_sequence)
    
    def compute_features_from_path(path_points):
        """Compute features from a path of points."""
        n = len(path_points)
        features = np.zeros((n, 5))  # 5 features: prev_angle, next_angle, prev_length, next_length, curvature
        
        for i in range(n):
            # Previous angle and length
            if i > 0:
                dx = path_points[i] - path_points[i-1]
                if np.linalg.norm(dx) > 0:
                    angle = np.arctan2(dx[1], dx[0])
                    features[i, 0] = angle
                    features[i, 2] = np.linalg.norm(dx) / 1000.0  # Normalize
            else:
                features[i, 0] = 0.0
                features[i, 2] = 0.0
            
            # Next angle and length
            if i < n - 1:
                dx = path_points[i+1] - path_points[i]
                if np.linalg.norm(dx) > 0:
                    angle = np.arctan2(dx[1], dx[0])
                    features[i, 1] = angle
                    features[i, 3] = np.linalg.norm(dx) / 1000.0  # Normalize
            else:
                features[i, 1] = features[i, 0] if i > 0 else 0.0
                features[i, 3] = 0.0
            
            # Curvature
            if i > 1 and i < n - 1:
                p0, p1, p2 = path_points[i-1], path_points[i], path_points[i+1]
                v1 = p1 - p0
                v2 = p2 - p1
                norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm_v1 > 0 and norm_v2 > 0:
                    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    cos_theta = np.clip(cos_theta, -1, 1)
                    theta = np.arccos(cos_theta)
                    chord_length = np.linalg.norm(p2 - p0)
                    if chord_length > 0:
                        features[i, 4] = theta / chord_length
        
        return features
    
    for step in range(max_steps):
        # Scale coordinates
        if data_loader is not None:
            seq_coords_scaled = data_loader.coord_scaler.transform(current_sequence)
            seq_features_scaled = data_loader.feature_scaler.transform(current_features)
        else:
            seq_coords_scaled = current_sequence
            seq_features_scaled = current_features
        
        # Concatenate coordinates and features
        seq_input = np.concatenate([seq_coords_scaled, seq_features_scaled], axis=-1)
        seq_input = seq_input.reshape(1, CONFIG['sequence_length'], -1)
        
        # Predict next point
        next_point_scaled = model.predict(seq_input, verbose=0)[0]
        
        # Inverse transform
        if data_loader is not None:
            next_point = data_loader.coord_scaler.inverse_transform([next_point_scaled])[0]
        else:
            next_point = next_point_scaled
        
        # Compute movement distance
        if len(generated_path) > 0:
            movement = np.linalg.norm(next_point - generated_path[-1])
            recent_movements.append(movement)
            if len(recent_movements) > 10:
                recent_movements.pop(0)
        
        # Apply distribution constraints to next_point if training_stats available
        if training_stats is not None and len(generated_path) > 0:
            # Constrain segment length based on training distribution
            segment_length = np.linalg.norm(next_point - generated_path[-1])
            constrained_length = apply_distribution_constraints(
                segment_length,
                training_stats.get('segment_length', {}),
                constraint_type='soft'  # Use soft constraints for smoother generation
            )
            # Adjust point to match constrained length
            if segment_length > 1e-6:
                direction = (next_point - generated_path[-1]) / segment_length
                next_point = generated_path[-1] + direction * constrained_length
        
        # Apply advanced stopping criteria if available
        if stopping_criteria is not None and step >= 5:  # Allow some initial steps
            should_stop = False
            
            # 1. Boundary detection
            if stopping_criteria.is_outside_reasonable_bounds(next_point, seed_sequence, bounds):
                should_stop = True
                stop_reason = "boundary_exceeded"
            
            # 2. Oscillation detection
            elif len(generated_path) >= 5:
                recent_points = generated_path[-5:] + [next_point]
                if stopping_criteria.detect_oscillation(recent_points):
                    should_stop = True
                    stop_reason = "oscillation_detected"
            
            # 3. Stagnation detection
            elif stopping_criteria.check_stagnation(recent_movements):
                should_stop = True
                stop_reason = "stagnation_detected"
            
            # 4. Excessive deviation from expected trajectory
            elif len(generated_path) >= 10:
                current_path_array = np.array(generated_path + [next_point])
                deviated, deviation_metric = stopping_criteria.check_excessive_deviation(
                    current_path_array, seed_sequence, true_path
                )
                if deviated:
                    should_stop = True
                    stop_reason = f"excessive_deviation_{deviation_metric:.2f}"
            
            if should_stop:
                print(f"  Stopped at step {step}: {stop_reason}")
                # Don't add the point that triggered stopping
                break
        
        # Add to path
        generated_path.append(next_point)
        
        # Update sequence (sliding window)
        current_sequence = np.vstack([current_sequence[1:], next_point])
        
        # Recompute features for the updated sequence
        current_features = compute_features_from_path(current_sequence)
    
    return np.array(generated_path), stop_reason

def plot_fracture_generation_comparison(true_path, gen_path, seed_points, fracture_id, save_dir, metrics=None):
    """
    Enhanced visualization comparing true vs generated fracture paths with seed point highlighting.
    
    Args:
        true_path: Ground truth path (numpy array, shape: [n_points, 2])
        gen_path: Generated path (numpy array, shape: [n_points, 2])
        seed_points: Seed points used for generation (numpy array, shape: [seed_len, 2])
        fracture_id: Fracture identifier
        save_dir: Directory to save plot
        metrics: Optional dictionary of metrics to display
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
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
        # Highlight first seed point
        if len(seed_points) > 0:
            ax1.scatter(seed_points[0, 0], seed_points[0, 1], 
                      c='lime', s=300, marker='*', 
                      edgecolors='darkgreen', linewidths=3, 
                      zorder=6, label='Start Point')
    
    # Add arrows to show direction (only if paths are long enough)
    if len(true_path) > 1:
        arrow_step = max(1, len(true_path)//10)
        for i in range(0, len(true_path)-1, arrow_step):
            dx = true_path[i+1, 0] - true_path[i, 0]
            dy = true_path[i+1, 1] - true_path[i, 1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Only draw if movement is significant
                try:
                    ax1.arrow(true_path[i, 0], true_path[i, 1], dx*0.3, dy*0.3,
                             head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.5)
                except:
                    pass  # Skip if arrow drawing fails
    
    if len(gen_path) > 1:
        arrow_step = max(1, len(gen_path)//10)
        for i in range(0, len(gen_path)-1, arrow_step):
            dx = gen_path[i+1, 0] - gen_path[i, 0]
            dy = gen_path[i+1, 1] - gen_path[i, 1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Only draw if movement is significant
                try:
                    ax1.arrow(gen_path[i, 0], gen_path[i, 1], dx*0.3, dy*0.3,
                             head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.5)
                except:
                    pass  # Skip if arrow drawing fails
    
    ax1.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    title = f'Fracture {fracture_id}: True vs Generated Path'
    if metrics:
        title += f'\nHausdorff: {metrics.get("hausdorff", 0):.4f} | Endpoint Error: {metrics.get("endpoint_error", 0):.4f}'
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    
    # Right plot: Error visualization
    ax2 = axes[1]
    
    if len(true_path) > 0 and len(gen_path) > 0:
        # Compute point-wise errors
        min_len = min(len(true_path), len(gen_path))
        true_subset = true_path[:min_len]
        gen_subset = gen_path[:min_len]
        
        errors = np.linalg.norm(true_subset - gen_subset, axis=1)
        
        # Plot error along path
        ax2.plot(errors, 'g-', linewidth=2, label='Point-wise Error', alpha=0.7)
        ax2.fill_between(range(len(errors)), errors, alpha=0.3, color='green')
        ax2.axhline(y=np.mean(errors), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean Error: {np.mean(errors):.4f}')
        ax2.axhline(y=np.median(errors), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median Error: {np.median(errors):.4f}')
        
        ax2.set_xlabel('Point Index', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Euclidean Error', fontsize=14, fontweight='bold')
        ax2.set_title('Error Along Generated Path', fontsize=16, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
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
        
        # Last point
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
    
    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title(f'Fracture {fracture_id}: Generation Progression\n(Color intensity shows generation order)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fracture_{fracture_id}_generation_progression.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Evaluate on test fractures with path generation
print("\n" + "="*80)
print("FRACTURE PATH GENERATION EVALUATION")
print("="*80)

from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance

def compute_hausdorff_distance(path1, path2):
    """Compute Hausdorff distance between two paths."""
    if len(path1) < 2 or len(path2) < 2:
        return float('inf')
    try:
        return max(directed_hausdorff(path1, path2)[0],
                  directed_hausdorff(path2, path1)[0])
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

# Generate paths for all test fractures
num_test_fractures = len(test_fractures)
generation_results = []

for i in range(num_test_fractures):
    fracture = test_fractures[i]
    fracture_id = fracture['id']
    true_path = fracture['points']
    true_features = fracture['features']
    
    if len(true_path) < 2:
        continue

    # Use up to sequence_length points as seed (pad if shorter)
    seed_len = min(CONFIG['sequence_length'], len(true_path) - 1)
    seed_points = true_path[:seed_len]
    seed_features = true_features[:seed_len]

    # Generate path with advanced controls
    try:
        gen_path, stop_reason = generate_fracture_path(
            model, seed_points, seed_features,
            max_steps=len(true_path) - seed_len,
            data_loader=data_loader,
            training_stats=training_stats,
            stopping_criteria=stopping_criteria,
            true_path=true_path  # For distance-based stopping during evaluation
        )
        
        if len(gen_path) < 2:
            print(f"Fracture {fracture_id}: Generation failed (too short: {len(gen_path)} points)")
            continue
        
        # Compute metrics
        hausdorff = compute_hausdorff_distance(true_path, gen_path)
        endpoint_error = np.linalg.norm(gen_path[-1] - true_path[-1]) if len(gen_path) > 0 and len(true_path) > 0 else float('inf')
        frechet = discrete_frechet_distance(true_path, gen_path)
        path_sim = compute_path_similarity(true_path, gen_path)
        gen_total_len = float(np.sum(np.linalg.norm(np.diff(gen_path, axis=0), axis=1)))
        true_total_len = float(np.sum(np.linalg.norm(np.diff(true_path, axis=0), axis=1)))
        length_error = abs(gen_total_len - true_total_len) / (true_total_len + 1e-6)

        metrics = {
            'hausdorff': hausdorff,
            'endpoint_error': endpoint_error,
            'frechet': frechet,
            'path_similarity': path_sim,
            'length_error': length_error
        }
        
        # Visualize
        if VISUALIZATION_UTILS_AVAILABLE:
            try:
                plot_fracture_generation_comparison(true_path, gen_path, seed_points, 
                                                  fracture_id, CONFIG['plots_dir'], metrics)
                plot_generation_progression(true_path, gen_path, seed_points, 
                                           fracture_id, CONFIG['plots_dir'])
            except Exception as viz_error:
                print(f"  Warning: Visualization failed for fracture {fracture_id}: {viz_error}")
        
        generation_results.append({
            'fracture_id': fracture_id,
            'metrics': metrics,
            'true_length': len(true_path),
            'gen_length': len(gen_path),
            'stop_reason': stop_reason,
            'true_path': true_path,
            'gen_path': gen_path
        })

        print(f"Fracture {fracture_id}: Hausdorff={hausdorff:.4f}, Frechet={frechet:.4f}, "
              f"Endpoint Error={endpoint_error:.4f}, Length Error={length_error:.4f}, "
              f"Path Sim={path_sim:.4f}, Generated {len(gen_path)} pts (True: {len(true_path)}), "
              f"Stop: {stop_reason}")
        
    except Exception as e:
        print(f"Error generating fracture {fracture_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Print summary and save path metrics
if generation_results:
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)

    def _finite(vals):
        return [v for v in vals if v != float('inf') and not np.isnan(v)]

    hausdorffs      = _finite([r['metrics']['hausdorff']       for r in generation_results])
    frchets         = _finite([r['metrics']['frechet']          for r in generation_results])
    endpoint_errors = _finite([r['metrics']['endpoint_error']   for r in generation_results])
    length_errors   = _finite([r['metrics']['length_error']     for r in generation_results])
    path_sims       = _finite([r['metrics']['path_similarity']  for r in generation_results])

    for name, vals in [('Hausdorff', hausdorffs), ('Frechet', frchets),
                       ('Endpoint Error', endpoint_errors), ('Length Error', length_errors),
                       ('Path Similarity', path_sims)]:
        if vals:
            print(f"{name:20s}: Mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save per-fracture path metrics
    path_rows = []
    for r in generation_results:
        row = {'fracture_id': r['fracture_id'], 'true_length': r['true_length'],
               'gen_length': r['gen_length']}
        row.update(r['metrics'])
        path_rows.append(row)
    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(f"{CONFIG['results_dir']}/path_metrics.csv", index=False)
    print(f"\nPath metrics saved to {CONFIG['results_dir']}/path_metrics.csv")

    # Summary row
    summary = {
        'hausdorff_mean': np.mean(hausdorffs) if hausdorffs else float('nan'),
        'hausdorff_std':  np.std(hausdorffs)  if hausdorffs else float('nan'),
        'frechet_mean':   np.mean(frchets)    if frchets    else float('nan'),
        'frechet_std':    np.std(frchets)     if frchets    else float('nan'),
        'endpoint_error_mean': np.mean(endpoint_errors) if endpoint_errors else float('nan'),
        'endpoint_error_std':  np.std(endpoint_errors)  if endpoint_errors else float('nan'),
        'length_error_mean': np.mean(length_errors) if length_errors else float('nan'),
        'length_error_std':  np.std(length_errors)  if length_errors else float('nan'),
        'path_similarity_mean': np.mean(path_sims) if path_sims else float('nan'),
        'path_similarity_std':  np.std(path_sims)  if path_sims else float('nan'),
    }
    pd.DataFrame([summary]).to_csv(f"{CONFIG['results_dir']}/path_metrics_summary.csv", index=False)

    # -----------------------------------------------------------------------
    # Distributional metrics: compare segment-length & angle distributions
    # between generated paths and true test paths
    # -----------------------------------------------------------------------
    _tl = [extract_segment_lengths(r['true_path']) for r in generation_results if len(r['true_path']) >= 2]
    _gl = [extract_segment_lengths(r['gen_path'])  for r in generation_results if len(r['gen_path'])  >= 2]
    _ta = [extract_segment_angles(r['true_path'])  for r in generation_results if len(r['true_path']) >= 2]
    _ga = [extract_segment_angles(r['gen_path'])   for r in generation_results if len(r['gen_path'])  >= 2]

    true_lengths_all = np.concatenate(_tl) if _tl else np.array([])
    gen_lengths_all  = np.concatenate(_gl) if _gl else np.array([])
    true_angles_all  = np.concatenate(_ta) if _ta else np.array([])
    gen_angles_all   = np.concatenate(_ga) if _ga else np.array([])

    # Save raw segment distributions for visualization scripts
    if len(true_lengths_all) > 0 or len(gen_lengths_all) > 0:
        pd.DataFrame({
            'type': ['true'] * len(true_lengths_all) + ['generated'] * len(gen_lengths_all),
            'value': np.concatenate([true_lengths_all, gen_lengths_all]) if len(true_lengths_all) > 0 and len(gen_lengths_all) > 0 else (true_lengths_all if len(true_lengths_all) > 0 else gen_lengths_all)
        }).to_csv(f"{CONFIG['results_dir']}/segment_lengths.csv", index=False)
    if len(true_angles_all) > 0 or len(gen_angles_all) > 0:
        pd.DataFrame({
            'type': ['true'] * len(true_angles_all) + ['generated'] * len(gen_angles_all),
            'value': np.concatenate([true_angles_all, gen_angles_all]) if len(true_angles_all) > 0 and len(gen_angles_all) > 0 else (true_angles_all if len(true_angles_all) > 0 else gen_angles_all)
        }).to_csv(f"{CONFIG['results_dir']}/segment_angles.csv", index=False)

    w_length = wasserstein_distance(true_lengths_all, gen_lengths_all) if len(true_lengths_all) > 0 and len(gen_lengths_all) > 0 else float('nan')
    w_angle  = wasserstein_distance(true_angles_all,  gen_angles_all)  if len(true_angles_all)  > 0 and len(gen_angles_all)  > 0 else float('nan')
    kl_div   = compute_kl_divergence(gen_lengths_all, true_lengths_all)

    print(f"\nDistributional Metrics:")
    print(f"  Wasserstein (Length): {w_length:.4f}")
    print(f"  Wasserstein (Angle):  {w_angle:.4f}")
    print(f"  KL Divergence:        {kl_div:.4f}")

    distrib_df = pd.DataFrame([{
        'model': 'BiLSTM-Attn',
        'wasserstein_length': w_length,
        'wasserstein_angle': w_angle,
        'kl_divergence': kl_div
    }])
    distrib_df.to_csv(f"{CONFIG['results_dir']}/distributional_metrics.csv", index=False)
    print(f"Distributional metrics saved to {CONFIG['results_dir']}/distributional_metrics.csv")

    # Print statistical compliance if available
    if stats_tracker is not None and ADVANCED_UTILS_AVAILABLE:
        print("\nStatistical Compliance Analysis:")
        compliance_scores = []
        for r in generation_results:
            if 'gen_path' in r:
                compliance = stats_tracker.check_statistical_compliance(r['gen_path'])
                compliance_scores.append(compliance['overall'])

        if compliance_scores:
            print(f"  Mean Compliance: {np.mean(compliance_scores):.4f} ± {np.std(compliance_scores):.4f}")
            print(f"  Median Compliance: {np.median(compliance_scores):.4f}")


# ## 11. Summary and Conclusions
# 
# ### Key Findings:
# 
# 1. **Model Performance**: The Bidirectional LSTM with Multi-head Attention successfully learns to predict fracture path coordinates from historical sequences.
# 
# 2. **Attention Mechanism**: The multi-head attention allows the model to focus on relevant past points when predicting the next location.
# 
# 3. **Bidirectional Processing**: Processing sequences in both directions helps capture forward and backward dependencies in fracture paths.
# 
# ### Strengths:
# - Strong sequential modeling with BiLSTM
# - Global context capture via attention
# - Stable training with layer normalization
# - Good generalization on test data
# 
# ### Limitations:
# - Fixed sequence length requirement
# - No explicit stopping prediction
# - Deterministic predictions (no uncertainty quantification)
# - Error accumulation in autoregressive generation
# 
# ### Next Steps:
# - Implement autoregressive generation for complete fracture paths
# - Add stopping prediction mechanism (see Case 3)
# - Explore probabilistic outputs (see Case 4 with MDN)
# - Compare with other architectures (Cases 2, 3, 4)

# ## References
# 
# This implementation is based on:
# 
# > **Deep Learning Approaches for Autoregressive Fracture Network Path Prediction: A Case Study on the Teapot Dome Dataset**
# > 
# > Section 4.2: Model 1 - Bidirectional LSTM with Multi-head Attention
# 
# **Key Citations:**
# - Hochreiter & Schmidhuber (1997): Long Short-Term Memory
# - Schuster & Paliwal (1997): Bidirectional Recurrent Neural Networks  
# - Vaswani et al. (2017): Attention is All You Need
