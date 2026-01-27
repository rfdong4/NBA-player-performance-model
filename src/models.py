"""
Model Training and Management Module for NBA Player Performance Prediction.
Supports multiple model types and prediction targets for sports betting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import MODEL_PARAMS, MODEL_DIR, PREDICTION_TARGETS


class ModelTrainer:
    """Trains and manages prediction models for different betting targets."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost')
        """
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, Dict] = {}

    def _create_model(self) -> Any:
        """Create a new model instance based on model type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(**MODEL_PARAMS['random_forest'])
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**MODEL_PARAMS['gradient_boosting'])
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(**MODEL_PARAMS['xgboost'])
            except ImportError:
                print("XGBoost not installed, falling back to Random Forest")
                return RandomForestRegressor(**MODEL_PARAMS['random_forest'])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        targets: List[str] = None,
        test_size: float = 0.2,
        use_time_split: bool = True,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Train models for specified prediction targets.

        Args:
            df: DataFrame with features and target columns
            feature_cols: List of feature column names
            targets: List of target column names to train models for
            test_size: Fraction of data for testing
            use_time_split: Use time-based split instead of random
            verbose: Print training progress

        Returns:
            Dict of metrics for each target
        """
        self.feature_columns = feature_cols

        if targets is None:
            targets = [c for c in df.columns if c.startswith('TARGET_')]

        # Prepare data
        df_clean = df.dropna(subset=feature_cols + targets)

        if len(df_clean) < 100:
            raise ValueError("Not enough data for training (need at least 100 samples)")

        X = df_clean[feature_cols]
        self.metrics = {}

        for target in targets:
            if verbose:
                print(f"\nTraining model for {target}...")

            y = df_clean[target]

            # Split data
            if use_time_split and 'GAME_DATE' in df_clean.columns:
                # Time-based split
                df_sorted = df_clean.sort_values('GAME_DATE')
                split_idx = int(len(df_sorted) * (1 - test_size))
                X_train = df_sorted[feature_cols].iloc[:split_idx]
                X_test = df_sorted[feature_cols].iloc[split_idx:]
                y_train = df_sorted[target].iloc[:split_idx]
                y_test = df_sorted[target].iloc[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target] = scaler

            # Train model
            model = self._create_model()
            model.fit(X_train_scaled, y_train)
            self.models[target] = model

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'mean_actual': y_test.mean(),
                'std_actual': y_test.std()
            }
            self.metrics[target] = metrics

            if verbose:
                print(f"  MAE: {metrics['mae']:.2f}")
                print(f"  RMSE: {metrics['rmse']:.2f}")
                print(f"  R²: {metrics['r2']:.3f}")

        return self.metrics

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target: str,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation.

        Args:
            df: DataFrame with features and target
            feature_cols: Feature column names
            target: Target column name
            n_splits: Number of CV splits

        Returns:
            Dict with CV metrics
        """
        df_clean = df.dropna(subset=feature_cols + [target])
        df_sorted = df_clean.sort_values('GAME_DATE')

        X = df_sorted[feature_cols].values
        y = df_sorted[target].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = self._create_model()

        scores = cross_val_score(
            model, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error'
        )

        return {
            'cv_mae_mean': -scores.mean(),
            'cv_mae_std': scores.std()
        }

    def predict(
        self,
        X: pd.DataFrame,
        target: str
    ) -> np.ndarray:
        """
        Make predictions for a target.

        Args:
            X: Feature DataFrame
            target: Target to predict

        Returns:
            Array of predictions
        """
        if target not in self.models:
            raise ValueError(f"No model trained for target: {target}")

        # Ensure correct feature order
        X_ordered = X[self.feature_columns].copy()

        # Handle missing values
        X_ordered = X_ordered.fillna(X_ordered.median())

        # Scale
        X_scaled = self.scalers[target].transform(X_ordered)

        return self.models[target].predict(X_scaled)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        target: str,
        n_estimators: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (for tree-based models).

        Args:
            X: Feature DataFrame
            target: Target to predict
            n_estimators: Number of estimators to use for interval

        Returns:
            Tuple of (predictions, std_deviations)
        """
        if target not in self.models:
            raise ValueError(f"No model trained for target: {target}")

        model = self.models[target]

        if not hasattr(model, 'estimators_'):
            # Not a forest model, return point prediction only
            pred = self.predict(X, target)
            return pred, np.zeros_like(pred)

        X_ordered = X[self.feature_columns].copy()
        X_ordered = X_ordered.fillna(X_ordered.median())
        X_scaled = self.scalers[target].transform(X_ordered)

        # Get predictions from individual trees
        predictions = np.array([
            tree.predict(X_scaled) for tree in model.estimators_
        ])

        return predictions.mean(axis=0), predictions.std(axis=0)

    def get_feature_importance(self, target: str) -> pd.DataFrame:
        """
        Get feature importance for a target model.

        Args:
            target: Target model name

        Returns:
            DataFrame with feature importances
        """
        if target not in self.models:
            raise ValueError(f"No model trained for target: {target}")

        model = self.models[target]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str = None):
        """
        Save all models and scalers.

        Args:
            path: Directory to save models
        """
        if path is None:
            path = MODEL_DIR

        os.makedirs(path, exist_ok=True)

        # Save models
        for target, model in self.models.items():
            model_path = os.path.join(path, f'model_{target}.joblib')
            joblib.dump(model, model_path)

        # Save scalers
        for target, scaler in self.scalers.items():
            scaler_path = os.path.join(path, f'scaler_{target}.joblib')
            joblib.dump(scaler, scaler_path)

        # Save feature columns and metrics
        meta_path = os.path.join(path, 'model_metadata.joblib')
        joblib.dump({
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'model_type': self.model_type,
            'trained_at': datetime.now().isoformat()
        }, meta_path)

        print(f"Models saved to {path}")

    def load(self, path: str = None):
        """
        Load models and scalers.

        Args:
            path: Directory containing saved models
        """
        if path is None:
            path = MODEL_DIR

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        # Load metadata
        meta_path = os.path.join(path, 'model_metadata.joblib')
        if os.path.exists(meta_path):
            metadata = joblib.load(meta_path)
            self.feature_columns = metadata['feature_columns']
            self.metrics = metadata.get('metrics', {})
            self.model_type = metadata.get('model_type', 'random_forest')

        # Load models
        for filename in os.listdir(path):
            if filename.startswith('model_') and filename.endswith('.joblib'):
                target = filename[6:-7]  # Remove 'model_' prefix and '.joblib' suffix
                model_path = os.path.join(path, filename)
                self.models[target] = joblib.load(model_path)

            elif filename.startswith('scaler_') and filename.endswith('.joblib'):
                target = filename[7:-7]  # Remove 'scaler_' prefix and '.joblib' suffix
                scaler_path = os.path.join(path, filename)
                self.scalers[target] = joblib.load(scaler_path)

        print(f"Loaded {len(self.models)} models from {path}")

    def get_available_targets(self) -> List[str]:
        """Get list of targets with trained models."""
        return list(self.models.keys())


class BettingAnalyzer:
    """Analyzes predictions in the context of sports betting."""

    def __init__(self, model_trainer: ModelTrainer):
        """
        Initialize betting analyzer.

        Args:
            model_trainer: Trained ModelTrainer instance
        """
        self.trainer = model_trainer

    def analyze_line(
        self,
        X: pd.DataFrame,
        target: str,
        line: float
    ) -> Dict[str, Any]:
        """
        Analyze a betting line.

        Args:
            X: Feature DataFrame (single row)
            target: Target stat to analyze
            line: Betting line (e.g., 25.5 for over/under)

        Returns:
            Dict with betting analysis
        """
        pred, std = self.trainer.predict_with_confidence(X, target)
        pred = pred[0]
        std = std[0] if std[0] > 0 else self.trainer.metrics[target]['mae']

        # Calculate probabilities using normal distribution approximation
        from scipy import stats

        diff = pred - line
        z_score = diff / std if std > 0 else 0

        prob_over = 1 - stats.norm.cdf(-z_score)
        prob_under = 1 - prob_over

        # Edge calculation (assuming -110 odds)
        implied_prob = 0.5238  # -110 implied probability

        over_edge = prob_over - implied_prob
        under_edge = prob_under - implied_prob

        # Confidence level
        confidence = min(abs(z_score) / 2, 1.0)  # Normalize to 0-1

        return {
            'prediction': pred,
            'std_dev': std,
            'line': line,
            'edge_over_line': diff,
            'probability_over': prob_over,
            'probability_under': prob_under,
            'over_edge': over_edge,
            'under_edge': under_edge,
            'recommendation': 'OVER' if over_edge > under_edge else 'UNDER',
            'confidence': confidence,
            'bet_strength': 'strong' if abs(max(over_edge, under_edge)) > 0.1 else
                           'moderate' if abs(max(over_edge, under_edge)) > 0.05 else 'weak'
        }

    def get_value_bets(
        self,
        X: pd.DataFrame,
        lines: Dict[str, float],
        min_edge: float = 0.05
    ) -> List[Dict]:
        """
        Find value bets across multiple props.

        Args:
            X: Feature DataFrame
            lines: Dict of target -> line
            min_edge: Minimum edge to consider a value bet

        Returns:
            List of value bet opportunities
        """
        value_bets = []

        for target, line in lines.items():
            if target not in self.trainer.models:
                continue

            analysis = self.analyze_line(X, target, line)

            if max(analysis['over_edge'], analysis['under_edge']) >= min_edge:
                value_bets.append({
                    'prop': target.replace('TARGET_', ''),
                    'line': line,
                    **analysis
                })

        # Sort by edge
        value_bets.sort(
            key=lambda x: max(x['over_edge'], x['under_edge']),
            reverse=True
        )

        return value_bets
