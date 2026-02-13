"""
Model Training and Management Module for NBA Player Performance Prediction.
Supports multiple model types, ensembling, and prediction targets for sports betting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
import warnings

from .config import MODEL_PARAMS, MODEL_DIR, PREDICTION_TARGETS

warnings.filterwarnings('ignore')


class FeatureSelector:
    """Intelligent feature selection to reduce noise and improve generalization."""

    def __init__(self, method: str = 'importance', threshold: float = 0.01):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('importance', 'correlation', 'recursive')
            threshold: Importance threshold for feature selection
        """
        self.method = method
        self.threshold = threshold
        self.selected_features: List[str] = []
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> List[str]:
        """
        Fit the feature selector and return selected features.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names

        Returns:
            List of selected feature names
        """
        if self.method == 'importance':
            # Use Random Forest importance
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            importances = rf.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Select features above threshold
            cumulative_importance = importance_df['importance'].cumsum()
            n_features = (cumulative_importance < 0.95).sum() + 1  # Keep 95% of importance
            n_features = max(n_features, 20)  # Keep at least 20 features

            self.selected_features = importance_df.head(n_features)['feature'].tolist()

        elif self.method == 'correlation':
            # Remove highly correlated features
            corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()

            # Select upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find features with correlation > 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            self.selected_features = [f for f in feature_names if f not in to_drop]

        elif self.method == 'recursive':
            # Recursive feature elimination with CV
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            self.selector = RFECV(rf, step=5, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            self.selector.fit(X, y)
            self.selected_features = [f for f, s in zip(feature_names, self.selector.support_) if s]

        return self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to selected subset."""
        return X[self.selected_features]


class EnsembleModel:
    """Ensemble model combining multiple algorithms for better predictions."""

    def __init__(self, use_xgboost: bool = True):
        """
        Initialize ensemble model.

        Args:
            use_xgboost: Whether to include XGBoost in ensemble
        """
        self.use_xgboost = use_xgboost
        self.models = {}
        self.weights = {}
        self.ensemble = None

    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """Create base models for ensemble."""
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                min_samples_split=5,
                random_state=42
            )),
            ('ridge', Ridge(alpha=1.0)),
        ]

        if self.use_xgboost:
            try:
                from xgboost import XGBRegressor
                # Test that XGBoost actually works (not just importable)
                test_model = XGBRegressor(n_estimators=1)
                models.append(('xgb', XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )))
            except (ImportError, Exception) as e:
                # XGBoost not available or broken (e.g., missing libomp)
                print(f"  Note: XGBoost unavailable ({type(e).__name__}), using other models")
                pass

        return models

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Fit ensemble model with optional validation-based weighting.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (for weight optimization)
            y_val: Validation target
        """
        base_models = self._create_base_models()

        # Fit individual models and calculate weights based on validation performance
        if X_val is not None and y_val is not None:
            val_scores = []
            for name, model in base_models:
                model.fit(X, y)
                self.models[name] = model
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                val_scores.append((name, mae))

            # Weight inversely proportional to MAE
            total_inv_mae = sum(1/s[1] for s in val_scores)
            self.weights = {name: (1/mae) / total_inv_mae for name, mae in val_scores}
        else:
            # Equal weights
            for name, model in base_models:
                model.fit(X, y)
                self.models[name] = model
                self.weights[name] = 1.0 / len(base_models)

        # Create voting ensemble with optimized weights
        weighted_models = [(name, self.models[name]) for name in self.models.keys()]
        self.ensemble = VotingRegressor(
            estimators=weighted_models,
            weights=[self.weights[name] for name, _ in weighted_models]
        )
        self.ensemble.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble."""
        return self.ensemble.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Returns:
            Tuple of (predictions, uncertainty estimates)
        """
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = np.average(predictions, axis=0, weights=list(self.weights.values()))
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred


class QuantileModel:
    """Quantile regression for prediction intervals."""

    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        Initialize quantile model.

        Args:
            quantiles: List of quantiles to predict
        """
        self.quantiles = quantiles
        self.models = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile regression models."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            for q in self.quantiles:
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=q,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X, y)
                self.models[q] = model
        except Exception as e:
            print(f"Quantile model fitting failed: {e}")

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict quantiles."""
        return {q: model.predict(X) for q, model in self.models.items()}

    def get_prediction_interval(self, X: np.ndarray, confidence: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction interval.

        Args:
            X: Features
            confidence: Confidence level (0.8 = 80% interval)

        Returns:
            Tuple of (lower bound, upper bound)
        """
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q

        predictions = self.predict(X)

        # Find closest quantiles
        lower_key = min(self.quantiles, key=lambda x: abs(x - lower_q))
        upper_key = min(self.quantiles, key=lambda x: abs(x - upper_q))

        return predictions[lower_key], predictions[upper_key]


class ModelTrainer:
    """Trains and manages prediction models for different betting targets."""

    def __init__(self, model_type: str = 'ensemble', use_feature_selection: bool = True):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'ensemble')
            use_feature_selection: Whether to use automatic feature selection
        """
        self.model_type = model_type
        self.use_feature_selection = use_feature_selection
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_selectors: Dict[str, FeatureSelector] = {}
        self.quantile_models: Dict[str, QuantileModel] = {}
        self.feature_columns: List[str] = []
        self.selected_features: Dict[str, List[str]] = {}
        self.metrics: Dict[str, Dict] = {}

    def _create_model(self) -> Any:
        """Create a new model instance based on model type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                # Test that XGBoost actually works
                test_model = XGBRegressor(n_estimators=1)
                return XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            except (ImportError, Exception) as e:
                print(f"XGBoost not available ({type(e).__name__}), falling back to Gradient Boosting")
                return GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    random_state=42
                )
        elif self.model_type == 'ensemble':
            return EnsembleModel(use_xgboost=True)
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

        self.metrics = {}

        for target in targets:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training model for {target}...")
                print(f"{'='*50}")

            y = df_clean[target]

            # Split data with validation set for ensemble weighting
            if use_time_split and 'GAME_DATE' in df_clean.columns:
                df_sorted = df_clean.sort_values('GAME_DATE')
                n = len(df_sorted)
                train_idx = int(n * 0.7)
                val_idx = int(n * 0.85)

                X_train = df_sorted[feature_cols].iloc[:train_idx]
                X_val = df_sorted[feature_cols].iloc[train_idx:val_idx]
                X_test = df_sorted[feature_cols].iloc[val_idx:]
                y_train = df_sorted[target].iloc[:train_idx]
                y_val = df_sorted[target].iloc[train_idx:val_idx]
                y_test = df_sorted[target].iloc[val_idx:]
            else:
                # Two-stage split
                X_temp, X_test, y_temp, y_test = train_test_split(
                    df_clean[feature_cols], y, test_size=test_size, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.2, random_state=42
                )

            if verbose:
                print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Feature selection
            if self.use_feature_selection:
                if verbose:
                    print(f"  Selecting features from {len(feature_cols)}...")

                selector = FeatureSelector(method='importance')
                selected = selector.fit(X_train.values, y_train.values, feature_cols)
                self.feature_selectors[target] = selector
                self.selected_features[target] = selected

                X_train = X_train[selected]
                X_val = X_val[selected]
                X_test = X_test[selected]

                if verbose:
                    print(f"  Selected {len(selected)} features")
            else:
                self.selected_features[target] = feature_cols

            # Scale features using RobustScaler (better for outliers)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target] = scaler

            # Train main model
            if verbose:
                print(f"  Training {self.model_type} model...")

            model = self._create_model()

            if self.model_type == 'ensemble':
                model.fit(X_train_scaled, y_train.values, X_val_scaled, y_val.values)
            else:
                model.fit(X_train_scaled, y_train.values)

            self.models[target] = model

            # Train quantile model for better intervals
            if verbose:
                print(f"  Training quantile models for prediction intervals...")

            quantile_model = QuantileModel()
            quantile_model.fit(X_train_scaled, y_train.values)
            self.quantile_models[target] = quantile_model

            # Evaluate on test set
            if self.model_type == 'ensemble':
                y_pred, y_std = model.predict_with_uncertainty(X_test_scaled)
            else:
                y_pred = model.predict(X_test_scaled)
                y_std = np.zeros_like(y_pred)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Calculate coverage of prediction intervals
            if target in self.quantile_models:
                lower, upper = self.quantile_models[target].get_prediction_interval(X_test_scaled, 0.8)
                coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
            else:
                coverage = 0.0

            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'coverage_80': coverage,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'n_features': len(self.selected_features[target]),
                'mean_actual': y_test.mean(),
                'std_actual': y_test.std(),
                'mean_uncertainty': y_std.mean() if y_std.sum() > 0 else mae
            }
            self.metrics[target] = metrics

            if verbose:
                print(f"\n  Results:")
                print(f"    MAE:  {metrics['mae']:.2f}")
                print(f"    RMSE: {metrics['rmse']:.2f}")
                print(f"    R²:   {metrics['r2']:.3f}")
                print(f"    80% Interval Coverage: {metrics['coverage_80']*100:.1f}%")

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

        mae_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = self._create_model()
            if self.model_type == 'ensemble':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled) if not isinstance(model, EnsembleModel) else model.predict(X_test_scaled)
            mae_scores.append(mean_absolute_error(y_test, y_pred))

        return {
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores)
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

        # Use selected features
        selected = self.selected_features.get(target, self.feature_columns)
        X_ordered = X[selected].copy()

        # Handle missing values
        X_ordered = X_ordered.fillna(X_ordered.median())

        # Scale
        X_scaled = self.scalers[target].transform(X_ordered)

        model = self.models[target]
        if isinstance(model, EnsembleModel):
            return model.predict(X_scaled)
        return model.predict(X_scaled)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        target: str,
        confidence: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.

        Args:
            X: Feature DataFrame
            target: Target to predict
            confidence: Confidence level for interval

        Returns:
            Tuple of (predictions, std_deviations)
        """
        if target not in self.models:
            raise ValueError(f"No model trained for target: {target}")

        # Use selected features
        selected = self.selected_features.get(target, self.feature_columns)
        X_ordered = X[selected].copy()
        X_ordered = X_ordered.fillna(X_ordered.median())
        X_scaled = self.scalers[target].transform(X_ordered)

        model = self.models[target]

        # Get point prediction and uncertainty
        if isinstance(model, EnsembleModel):
            pred, std = model.predict_with_uncertainty(X_scaled)
        elif hasattr(model, 'estimators_'):
            # Random Forest or similar
            predictions = np.array([tree.predict(X_scaled) for tree in model.estimators_])
            pred = predictions.mean(axis=0)
            std = predictions.std(axis=0)
        else:
            pred = model.predict(X_scaled)
            std = np.full_like(pred, self.metrics.get(target, {}).get('mae', 5.0))

        # Improve std estimate using quantile models if available
        if target in self.quantile_models and std.mean() < 0.1:
            lower, upper = self.quantile_models[target].get_prediction_interval(X_scaled, confidence)
            std = (upper - lower) / 3.29  # Convert 80% interval to approximate std

        return pred, std

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
        selected = self.selected_features.get(target, self.feature_columns)

        if isinstance(model, EnsembleModel):
            # Average importance across ensemble models
            importances = []
            for name, m in model.models.items():
                if hasattr(m, 'feature_importances_'):
                    importances.append(m.feature_importances_)

            if importances:
                avg_importance = np.mean(importances, axis=0)
                return pd.DataFrame({
                    'feature': selected,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)

        elif hasattr(model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': selected,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

        return pd.DataFrame()

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

        # Save quantile models
        for target, qmodel in self.quantile_models.items():
            qmodel_path = os.path.join(path, f'quantile_{target}.joblib')
            joblib.dump(qmodel, qmodel_path)

        # Save feature columns and metrics
        meta_path = os.path.join(path, 'model_metadata.joblib')
        joblib.dump({
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
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
            self.selected_features = metadata.get('selected_features', {})
            self.metrics = metadata.get('metrics', {})
            self.model_type = metadata.get('model_type', 'ensemble')

        # Load models
        for filename in os.listdir(path):
            if filename.startswith('model_') and filename.endswith('.joblib'):
                target = filename[6:-7]
                model_path = os.path.join(path, filename)
                self.models[target] = joblib.load(model_path)

            elif filename.startswith('scaler_') and filename.endswith('.joblib'):
                target = filename[7:-7]
                scaler_path = os.path.join(path, filename)
                self.scalers[target] = joblib.load(scaler_path)

            elif filename.startswith('quantile_') and filename.endswith('.joblib'):
                target = filename[9:-7]
                qmodel_path = os.path.join(path, filename)
                self.quantile_models[target] = joblib.load(qmodel_path)

        # Ensure selected_features has entries for all targets
        for target in self.models.keys():
            if target not in self.selected_features:
                self.selected_features[target] = self.feature_columns

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
        std = std[0] if std[0] > 0.1 else self.trainer.metrics.get(target, {}).get('mae', 5.0)

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
        confidence = min(abs(z_score) / 2, 1.0)

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

        value_bets.sort(
            key=lambda x: max(x['over_edge'], x['under_edge']),
            reverse=True
        )

        return value_bets
