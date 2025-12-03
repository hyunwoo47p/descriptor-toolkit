"""
ChemDescriptorML (CDML) - Track 2: ML Model Training

Optimal ML Ensemble for Molecular Property Prediction

Supported Models:
- RandomForest, ExtraTrees (Tree-based)
- XGBoost, LightGBM (Boosting)
- Ridge, Lasso, ElasticNet (Linear)
- GPR (Gaussian Process Regressor)

Features:
- K-Fold Cross-Validation (reliability)
- Hold-Out Test Set (real-world performance)
- Strong regularization (overfitting prevention)
- Extreme value evaluation (important regions)
- Cluster-aware descriptor selection (sequential, representative, random_alternative, mixed)
"""

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    HAS_GPR = True
except ImportError:
    HAS_GPR = False

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class OptimalMLEnsemble:
    """
    Optimal ML Ensemble System for Molecular Property Prediction

    Features:
    - K-Fold Cross-Validation (reliability)
    - Hold-Out Test Set (real-world performance)
    - Strong regularization (overfitting prevention)
    - Extreme value evaluation (important regions)
    - Cluster-aware descriptor selection from final_cluster_info.json
    """

    # Metadata columns to exclude from descriptors
    METADATA_COLS = [
        'CID', 'isomeric_smiles', 'standardized_smiles', 'SMILES',
        'parse_source', 'standardization_status', 'source_file',
        'InChI', 'InChIKey', 'mol_id', 'id', 'name'
    ]

    def __init__(
        self,
        data_path: Union[str, Path],
        target_col: str = 'pLeach',
        cluster_info_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        use_regularization: bool = False
    ):
        """
        Initialize ML Ensemble.

        Args:
            data_path: Path to training data (.parquet or .csv)
            target_col: Target column name for prediction
            cluster_info_path: Path to final_cluster_info.json (optional)
            test_data_path: Path to separate test data (.parquet or .csv).
                           If provided, test_size is ignored and this file is used as test set.
            test_size: Hold-out test ratio (default 0.2). Ignored if test_data_path is provided.
            cv_folds: K-Fold CV folds (default 5)
            random_state: Random seed for reproducibility
            use_regularization: If True, apply strong regularization to models (default False)
        """
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.cluster_info_path = Path(cluster_info_path) if cluster_info_path else None
        self.test_data_path = Path(test_data_path) if test_data_path else None
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_regularization = use_regularization
        self.use_external_test = test_data_path is not None

        # Results storage
        self.results = []
        self.best_models = {}
        self.cluster_info = None

        # Load data
        self._load_data()

        # Load cluster info if provided
        if self.cluster_info_path and self.cluster_info_path.exists():
            self._load_cluster_info()

        print("=" * 80)
        print("OPTIMAL ML ENSEMBLE INITIALIZED")
        print("=" * 80)
        print(f"Train samples: {len(self.df)}")
        if self.use_external_test:
            print(f"Test samples: {len(self.df_test)} (external file)")
        else:
            print(f"Test split: {test_size*100:.0f}% (random)")
        print(f"Total descriptors: {len(self.descriptor_cols)}")
        print(f"Target: {self.target_col}")
        print(f"CV Folds: {cv_folds}")
        print(f"Regularization: {'ON (strong)' if use_regularization else 'OFF (default params)'}")
        if self.cluster_info:
            print(f"Cluster info: {len(self.cluster_info['descriptors'])} representative descriptors")
        print("=" * 80 + "\n")

    def _load_data(self):
        """Load training data (and optionally separate test data) from parquet or CSV"""
        # Load training data
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        # Load external test data if provided
        self.df_test = None
        if self.test_data_path:
            if self.test_data_path.suffix == '.parquet':
                self.df_test = pd.read_parquet(self.test_data_path)
            elif self.test_data_path.suffix == '.csv':
                self.df_test = pd.read_csv(self.test_data_path)
            else:
                raise ValueError(f"Unsupported test file format: {self.test_data_path.suffix}")

        # Identify descriptor columns (exclude metadata and target)
        exclude_cols = set(self.METADATA_COLS) | {self.target_col}
        # Also exclude any column ending with common non-descriptor suffixes
        self.descriptor_cols = [
            col for col in self.df.columns
            if col not in exclude_cols and col.lower() not in [c.lower() for c in exclude_cols]
        ]

        # Filter to only numeric columns
        numeric_cols = self.df[self.descriptor_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.descriptor_cols = numeric_cols

        if self.df_test is not None:
            print(f"Train data loaded: {len(self.df)} samples, {len(self.descriptor_cols)} descriptors")
            print(f"Test data loaded: {len(self.df_test)} samples (external file)")
        else:
            print(f"Data loaded: {len(self.df)} samples, {len(self.descriptor_cols)} descriptors")

    def _load_cluster_info(self):
        """Load cluster info from JSON"""
        with open(self.cluster_info_path, 'r') as f:
            self.cluster_info = json.load(f)

        # Extract representative descriptors (ordered by cluster size)
        descriptors_info = self.cluster_info.get('descriptors', {})

        # Sort by cluster size (descending)
        sorted_descs = sorted(
            descriptors_info.items(),
            key=lambda x: x[1].get('cluster_size', 0),
            reverse=True
        )

        self.representative_descriptors = [d[0] for d in sorted_descs]
        print(f"Cluster info loaded: {len(self.representative_descriptors)} representative descriptors")

    def get_descriptors_from_clusters(
        self,
        n_descriptors: int,
        mode: str = 'sequential',
        random_per_cluster: int = 0
    ) -> List[str]:
        """
        Get descriptors based on cluster information.

        Args:
            n_descriptors: Number of descriptors to select
            mode: Selection mode
                - 'sequential': Use first n descriptors in original column order (default)
                - 'representative': Use representative descriptors from cluster info
                - 'random_alternative': For each representative, randomly pick from cluster
                - 'mixed': Mix of representatives and random alternatives
            random_per_cluster: Number of random alternatives per cluster (for 'mixed' mode)

        Returns:
            List of selected descriptor names
        """
        # Sequential mode: use original column order (no cluster info needed)
        if mode == 'sequential':
            return self.descriptor_cols[:n_descriptors]

        if not self.cluster_info:
            # No cluster info, fall back to sequential
            return self.descriptor_cols[:n_descriptors]

        descriptors_info = self.cluster_info.get('descriptors', {})
        selected = []

        if mode == 'representative':
            # Simply use top N representatives
            for desc in self.representative_descriptors:
                if desc in self.descriptor_cols:
                    selected.append(desc)
                if len(selected) >= n_descriptors:
                    break

        elif mode == 'random_alternative':
            # For each cluster, randomly pick one member
            rng = random.Random(self.random_state)

            for desc in self.representative_descriptors:
                if len(selected) >= n_descriptors:
                    break

                info = descriptors_info.get(desc, {})
                alternatives = info.get('all_cluster_members', [desc])

                # Filter to available descriptors
                available = [d for d in alternatives if d in self.descriptor_cols]

                if available:
                    chosen = rng.choice(available)
                    if chosen not in selected:
                        selected.append(chosen)

        elif mode == 'mixed':
            # Mix representatives with random alternatives
            rng = random.Random(self.random_state)

            for desc in self.representative_descriptors:
                if len(selected) >= n_descriptors:
                    break

                # Add representative
                if desc in self.descriptor_cols and desc not in selected:
                    selected.append(desc)

                # Add random alternatives
                if random_per_cluster > 0:
                    info = descriptors_info.get(desc, {})
                    alternatives = info.get('alternative_descriptors', [])
                    available = [d for d in alternatives if d in self.descriptor_cols and d not in selected]

                    n_pick = min(random_per_cluster, len(available))
                    if n_pick > 0:
                        picked = rng.sample(available, n_pick)
                        selected.extend(picked)

        return selected[:n_descriptors]

    def _get_model(self, model_name: str):
        """
        Create model with or without regularization based on self.use_regularization.
        """
        if self.use_regularization:
            # Strong regularization for small datasets
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=3,
                    min_samples_split=5,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=3,
                    min_samples_split=5,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'Lasso': Lasso(alpha=0.1, max_iter=10000, random_state=self.random_state),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=self.random_state),
            }

            if HAS_XGB:
                models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    n_jobs=-1
                )

            if HAS_LGB:
                models['LightGBM'] = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
        else:
            # Default parameters (no strong regularization)
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1, max_iter=10000, random_state=self.random_state),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=self.random_state),
            }

            if HAS_XGB:
                models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )

            if HAS_LGB:
                models['LightGBM'] = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )

        # Add GPR (same for both modes)
        if HAS_GPR:
            models['GPR'] = GaussianProcessRegressor(
                kernel=C(1.0) * RBF(1.0) + WhiteKernel(1.0),
                random_state=self.random_state,
                n_restarts_optimizer=3
            )

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

        return models[model_name]

    def _prepare_data(self, descriptors: List[str]) -> Tuple:
        """Prepare data: NaN handling + Scaling + Train/Test Split

        If external test data is provided (self.use_external_test), use it directly.
        Otherwise, perform train_test_split on the training data.
        """
        X_train = self.df[descriptors].values
        y_train = self.df[self.target_col].values

        # Handle NaN in training data (mean imputation)
        mask_train = np.isnan(X_train)
        col_means = np.nanmean(X_train, axis=0)
        if mask_train.any():
            for col_idx in range(X_train.shape[1]):
                X_train[mask_train[:, col_idx], col_idx] = col_means[col_idx]

        if self.use_external_test and self.df_test is not None:
            # Use external test data
            X_test = self.df_test[descriptors].values
            y_test = self.df_test[self.target_col].values

            # Handle NaN in test data (use training data means)
            mask_test = np.isnan(X_test)
            if mask_test.any():
                for col_idx in range(X_test.shape[1]):
                    X_test[mask_test[:, col_idx], col_idx] = col_means[col_idx]
        else:
            # Random train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=self.test_size, random_state=self.random_state
            )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _compute_extreme_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute extreme value metrics.
        Top/Bottom 20% MAE.
        """
        # Top 20% (High extremes)
        high_threshold = np.percentile(y_true, 80)
        high_mask = y_true > high_threshold

        # Bottom 20% (Low extremes)
        low_threshold = np.percentile(y_true, 20)
        low_mask = y_true < low_threshold

        results = {}

        if high_mask.sum() > 0:
            results['extreme_high_mae'] = float(mean_absolute_error(
                y_true[high_mask], y_pred[high_mask]
            ))
            results['extreme_high_n'] = int(high_mask.sum())
        else:
            results['extreme_high_mae'] = None
            results['extreme_high_n'] = 0

        if low_mask.sum() > 0:
            results['extreme_low_mae'] = float(mean_absolute_error(
                y_true[low_mask], y_pred[low_mask]
            ))
            results['extreme_low_n'] = int(low_mask.sum())
        else:
            results['extreme_low_mae'] = None
            results['extreme_low_n'] = 0

        return results

    def train_single_model(
        self,
        model_name: str,
        descriptors: List[str],
        verbose: bool = True
    ) -> Tuple[Dict, object]:
        """
        Train and evaluate a single model.

        Returns:
            result: Dict with all metrics
            model: Fitted model object
        """
        n_desc = len(descriptors)

        if verbose:
            print(f"\n{'='*80}")
            print(f"{model_name} with {n_desc} Descriptors")
            print(f"{'='*80}")

        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(descriptors)

        # Create model
        model = self._get_model(model_name)

        # === 1. K-Fold Cross-Validation ===
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        cv_r2_scores = cross_val_score(
            model, X_train, y_train,
            cv=kfold,
            scoring='r2',
            n_jobs=-1
        )

        cv_r2_mean = cv_r2_scores.mean()
        cv_r2_std = cv_r2_scores.std()

        if verbose:
            print(f"  K-Fold CV R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

        # === 2. Full Training + Hold-Out Test ===
        model.fit(X_train, y_train)

        # Training set performance
        y_pred_train = model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)

        # Test set performance
        y_pred_test = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Overfitting metric
        overfitting_gap = train_r2 - test_r2

        if verbose:
            print(f"  Train R² = {train_r2:.4f}, RMSE = {train_rmse:.3f}")
            print(f"  Test  R² = {test_r2:.4f}, RMSE = {test_rmse:.3f}")
            print(f"  Overfitting Gap = {overfitting_gap:.4f}")

        # === 3. Extreme value evaluation ===
        extreme_metrics = self._compute_extreme_metrics(y_test, y_pred_test)

        if verbose and extreme_metrics['extreme_high_mae'] is not None:
            print(f"  Extreme High MAE = {extreme_metrics['extreme_high_mae']:.3f} "
                  f"(n={extreme_metrics['extreme_high_n']})")
        if verbose and extreme_metrics['extreme_low_mae'] is not None:
            print(f"  Extreme Low MAE = {extreme_metrics['extreme_low_mae']:.3f} "
                  f"(n={extreme_metrics['extreme_low_n']})")

        # === 4. Feature Importance ===
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                desc: float(imp)
                for desc, imp in zip(descriptors, model.feature_importances_)
            }
        elif hasattr(model, 'coef_'):
            feature_importance = {
                desc: float(abs(coef))
                for desc, coef in zip(descriptors, model.coef_)
            }

        # Store result
        result = {
            'model_name': model_name,
            'n_descriptors': n_desc,
            'descriptors': descriptors,

            # K-Fold CV
            'kfold_r2_mean': float(cv_r2_mean),
            'kfold_r2_std': float(cv_r2_std),

            # Training set
            'train_r2': float(train_r2),
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),

            # Test set (Hold-out)
            'holdout_r2': float(test_r2),
            'holdout_rmse': float(test_rmse),
            'holdout_mae': float(test_mae),

            # Overfitting
            'overfitting_gap': float(overfitting_gap),

            # Extreme values
            'extreme_high_mae': extreme_metrics['extreme_high_mae'],
            'extreme_high_n': extreme_metrics['extreme_high_n'],
            'extreme_low_mae': extreme_metrics['extreme_low_mae'],
            'extreme_low_n': extreme_metrics['extreme_low_n'],

            # Feature importance
            'feature_importance': feature_importance
        }

        self.results.append(result)

        return result, model

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = ['RandomForest', 'ExtraTrees', 'Ridge', 'Lasso', 'ElasticNet']
        if HAS_XGB:
            models.append('XGBoost')
        if HAS_LGB:
            models.append('LightGBM')
        if HAS_GPR:
            models.append('GPR')
        return models

    def train_all_models(
        self,
        descriptor_sizes: List[int] = [5, 10, 15, 20, 30, 40, 50],
        descriptor_mode: str = 'representative',
        models: Optional[List[str]] = None
    ):
        """
        Train all models with various descriptor counts.

        Args:
            descriptor_sizes: List of descriptor counts to try
            descriptor_mode: 'representative', 'random_alternative', or 'mixed'
            models: List of model names (default: all available)
        """
        if models is None:
            models = self.get_available_models()

        # Filter to valid sizes
        max_desc = len(self.descriptor_cols)
        descriptor_sizes = [s for s in descriptor_sizes if s <= max_desc]

        print("\n" + "=" * 80)
        print("TRAINING ALL MODELS")
        print("=" * 80)
        print(f"Models: {len(models)} - {models}")
        print(f"Descriptor sizes: {descriptor_sizes}")
        print(f"Descriptor mode: {descriptor_mode}")
        print(f"Total experiments: {len(models) * len(descriptor_sizes)}")
        print("=" * 80)

        total = len(models) * len(descriptor_sizes)
        count = 0

        for model_name in models:
            for n_desc in descriptor_sizes:
                count += 1
                print(f"\n[{count}/{total}] {model_name} - {n_desc}D")

                # Get descriptors
                descriptors = self.get_descriptors_from_clusters(
                    n_desc, mode=descriptor_mode
                )

                # Train
                try:
                    result, model = self.train_single_model(
                        model_name, descriptors, verbose=False
                    )

                    # Track best model
                    key = f"{model_name}_{n_desc}D"
                    self.best_models[key] = {
                        'model': model,
                        'result': result
                    }

                    print(f"   K-Fold R²={result['kfold_r2_mean']:.4f}, "
                          f"Hold-Out R²={result['holdout_r2']:.4f}, "
                          f"Gap={result['overfitting_gap']:.4f}")
                except Exception as e:
                    print(f"   ERROR: {e}")

        print(f"\n{'='*80}")
        print("ALL MODELS TRAINED")
        print(f"{'='*80}\n")

    def find_best_model(self, metric: str = 'holdout_r2') -> Dict:
        """Find best performing model"""
        if not self.results:
            raise ValueError("No results yet. Run train_all_models first.")

        best_result = max(self.results, key=lambda x: x[metric])

        print(f"\n{'='*80}")
        print(f"BEST MODEL (by {metric})")
        print(f"{'='*80}")
        print(f"Model: {best_result['model_name']}")
        print(f"Descriptors: {best_result['n_descriptors']}")
        print(f"K-Fold R² = {best_result['kfold_r2_mean']:.4f} ± {best_result['kfold_r2_std']:.4f}")
        print(f"Hold-Out R² = {best_result['holdout_r2']:.4f}")
        print(f"Hold-Out RMSE = {best_result['holdout_rmse']:.3f}")
        print(f"Overfitting Gap = {best_result['overfitting_gap']:.4f}")
        if best_result['extreme_high_mae']:
            print(f"Extreme High MAE = {best_result['extreme_high_mae']:.3f}")
        print(f"{'='*80}\n")

        return best_result

    def save_results(self, output_dir: Union[str, Path] = 'output'):
        """Save results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save all results
        results_file = output_path / 'ml_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save best model info
        if self.results:
            best = self.find_best_model(metric='holdout_r2')
            best_file = output_path / 'best_model.json'
            with open(best_file, 'w') as f:
                json.dump(best, f, indent=2)

        print(f"Results saved to {output_path}")

    def plot_model_comparison(self, output_dir: Union[str, Path] = 'output'):
        """Generate model comparison plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("No results to plot")
            return

        # Get available models from results
        models = list(set(r['model_name'] for r in self.results))

        # Find best result per model
        best_by_model = {}
        for model_name in models:
            model_results = [r for r in self.results if r['model_name'] == model_name]
            if model_results:
                best = max(model_results, key=lambda x: x['holdout_r2'])
                best_by_model[model_name] = best

        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        # === 1. Test R² comparison ===
        fig, ax = plt.subplots(figsize=(12, 8))

        test_r2_values = [best_by_model[m]['holdout_r2'] for m in models]
        bars = ax.bar(range(len(models)), test_r2_values,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=11, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('Test R²', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Comparison - Test R²',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)

        # Add value labels
        for i, m in enumerate(models):
            v = best_by_model[m]['holdout_r2']
            ax.text(i, v + 0.02, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison_test_r2.png', dpi=300, bbox_inches='tight')
        plt.close()

        # === 2. K-Fold vs Hold-Out comparison ===
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(models))
        width = 0.35

        kfold_r2 = [best_by_model[m]['kfold_r2_mean'] for m in models]
        holdout_r2 = [best_by_model[m]['holdout_r2'] for m in models]

        bars1 = ax.bar(x - width/2, kfold_r2, width, label='K-Fold CV',
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, holdout_r2, width, label='Hold-Out',
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel('R²', fontsize=14, fontweight='bold')
        ax.set_title('K-Fold CV vs Hold-Out Test Performance',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig(output_path / 'kfold_vs_holdout.png', dpi=300, bbox_inches='tight')
        plt.close()

        # === 3. Descriptor count vs R² heatmap ===
        # Get unique descriptor sizes and models
        desc_sizes = sorted(set(r['n_descriptors'] for r in self.results))

        if len(desc_sizes) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Create matrix
            matrix = np.zeros((len(models), len(desc_sizes)))
            for i, model in enumerate(models):
                for j, size in enumerate(desc_sizes):
                    matching = [r for r in self.results
                               if r['model_name'] == model and r['n_descriptors'] == size]
                    if matching:
                        matrix[i, j] = matching[0]['holdout_r2']

            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(range(len(desc_sizes)))
            ax.set_xticklabels(desc_sizes, fontsize=11)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=11)
            ax.set_xlabel('Number of Descriptors', fontsize=14, fontweight='bold')
            ax.set_ylabel('Model', fontsize=14, fontweight='bold')
            ax.set_title('Hold-Out R² by Model and Descriptor Count',
                        fontsize=16, fontweight='bold', pad=20)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('R²', fontsize=12)

            # Add text annotations
            for i in range(len(models)):
                for j in range(len(desc_sizes)):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                  ha='center', va='center', fontsize=9,
                                  color='white' if matrix[i, j] < 0.5 else 'black')

            plt.tight_layout()
            plt.savefig(output_path / 'descriptor_count_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Plots saved to {output_path}")

    def plot_best_model_analysis(self, output_dir: Union[str, Path] = 'output'):
        """
        Generate detailed analysis plots for the best model.
        Includes: Predicted vs Actual, Residual plot, Feature importance.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("No results to plot")
            return

        # Find best model
        best_result = max(self.results, key=lambda x: x['holdout_r2'])
        model_name = best_result['model_name']
        n_desc = best_result['n_descriptors']
        descriptors = best_result['descriptors']

        # Get the trained model
        key = f"{model_name}_{n_desc}D"
        if key not in self.best_models:
            print(f"Model {key} not found in best_models")
            return

        model = self.best_models[key]['model']

        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(descriptors)

        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # === 1. Predicted vs Actual (Main plot) ===
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot train and test data
        ax.scatter(y_train, y_pred_train, alpha=0.6, s=80, c='blue',
                   edgecolors='darkblue', linewidth=1, label=f'Train (n={len(y_train)})')
        ax.scatter(y_test, y_pred_test, alpha=0.8, s=100, c='red',
                   edgecolors='darkred', linewidth=1.5, label=f'Test (n={len(y_test)})')

        # Perfect prediction line
        all_values = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
        min_val, max_val = all_values.min(), all_values.max()
        margin = (max_val - min_val) * 0.1
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'k--', linewidth=2, label='Perfect Prediction')

        # Calculate metrics for display
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Add metrics text box
        textstr = f'Test R² = {test_r2:.4f}\nTest RMSE = {test_rmse:.4f}\nTrain R² = {train_r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')

        ax.set_xlabel(f'Actual {self.target_col}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Predicted {self.target_col}', fontsize=14, fontweight='bold')
        ax.set_title(f'{model_name} ({n_desc}D) - Predicted vs Actual',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'pred_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()

        # === 2. Residual Plot ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Residuals vs Predicted
        residuals_test = y_test - y_pred_test
        axes[0].scatter(y_pred_test, residuals_test, alpha=0.7, s=80, c='red',
                        edgecolors='darkred', linewidth=1)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Value', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
        axes[0].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Residual Distribution
        axes[1].hist(residuals_test, bins=15, alpha=0.7, color='steelblue',
                     edgecolor='black', linewidth=1.2)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[1].axvline(x=residuals_test.mean(), color='orange', linestyle='-',
                        linewidth=2, label=f'Mean: {residuals_test.mean():.3f}')
        axes[1].set_xlabel('Residual', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'{model_name} ({n_desc}D) - Residual Analysis',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # === 3. Feature Importance (Top 15) ===
        if best_result['feature_importance']:
            fig, ax = plt.subplots(figsize=(12, 8))

            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in best_result['feature_importance'].items()
            ]).sort_values('importance', ascending=True)

            # Top 15
            top_n = min(15, len(importance_df))
            importance_df = importance_df.tail(top_n)

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
            bars = ax.barh(importance_df['feature'], importance_df['importance'],
                           color=colors, edgecolor='black', linewidth=1)

            ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            ax.set_ylabel('Descriptor', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name} ({n_desc}D) - Top {top_n} Feature Importance',
                         fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, axis='x', alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, importance_df['importance']):
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        # === 4. Actual vs Predicted with Error Bars (Test only) ===
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sort by actual value for better visualization
        sorted_idx = np.argsort(y_test)
        y_test_sorted = y_test[sorted_idx]
        y_pred_sorted = y_pred_test[sorted_idx]

        x_pos = np.arange(len(y_test))
        ax.scatter(x_pos, y_test_sorted, s=120, c='green', marker='o',
                   label='Actual', edgecolors='darkgreen', linewidth=1.5, zorder=3)
        ax.scatter(x_pos, y_pred_sorted, s=80, c='red', marker='s',
                   label='Predicted', edgecolors='darkred', linewidth=1, zorder=2)

        # Connect actual and predicted with lines
        for i in range(len(x_pos)):
            ax.plot([x_pos[i], x_pos[i]], [y_test_sorted[i], y_pred_sorted[i]],
                    'gray', alpha=0.5, linewidth=1, zorder=1)

        ax.set_xlabel('Sample Index (sorted by actual value)', fontsize=12, fontweight='bold')
        ax.set_ylabel(self.target_col, fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} ({n_desc}D) - Test Set Prediction Comparison',
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'test_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Best model analysis plots saved to {output_path}")

    def plot_all_models_comparison(self, output_dir: Union[str, Path] = 'output'):
        """
        Generate a grid plot comparing all models' predictions vs actual values.
        Shows 8 models in a 2x4 grid for easy visual comparison.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("No results to plot")
            return

        # Get unique models and find best result for each
        model_results = {}
        for result in self.results:
            model_name = result['model_name']
            if model_name not in model_results or result['holdout_r2'] > model_results[model_name]['holdout_r2']:
                model_results[model_name] = result

        # Sort by holdout R² (best first)
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['holdout_r2'], reverse=True)

        # Create 2x4 grid
        n_models = len(sorted_models)
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, result) in enumerate(sorted_models):
            ax = axes[idx]
            n_desc = result['n_descriptors']
            descriptors = result['descriptors']

            # Get model key
            key = f"{model_name}_{n_desc}D"
            if key not in self.best_models:
                ax.text(0.5, 0.5, f'{model_name}\nModel not found',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}')
                continue

            model = self.best_models[key]['model']

            # Prepare data and get predictions
            X_train, X_test, y_train, y_test = self._prepare_data(descriptors)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Plot
            ax.scatter(y_train, y_pred_train, alpha=0.5, s=40, c='blue', label='Train')
            ax.scatter(y_test, y_pred_test, alpha=0.8, s=60, c='red', label='Test')

            # Perfect prediction line
            all_values = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
            min_val, max_val = all_values.min(), all_values.max()
            margin = (max_val - min_val) * 0.1
            ax.plot([min_val - margin, max_val + margin],
                    [min_val - margin, max_val + margin],
                    'k--', linewidth=1.5)

            # Metrics
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            ax.set_xlabel('Actual', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.set_title(f'{model_name} ({n_desc}D)\nR²={test_r2:.3f}, RMSE={test_rmse:.2f}',
                         fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(sorted_models), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('All Models Comparison - Predicted vs Actual', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'all_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All models comparison plot saved to {output_path / 'all_models_comparison.png'}")

    def generate_all_plots(self, output_dir: Union[str, Path] = 'output'):
        """Generate all available plots"""
        self.plot_model_comparison(output_dir)
        self.plot_best_model_analysis(output_dir)
        self.plot_all_models_comparison(output_dir)
