"""
Optimal ML Ensemble for Molecular Property Prediction

Features:
- K-Fold Cross-Validation (reliability)
- Hold-Out Test Set (real-world performance)
- Strong regularization (overfitting prevention)
- Extreme value evaluation (important regions)
- Cluster-aware descriptor selection
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
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        use_regularization: bool = False
    ):
        """
        Initialize ML Ensemble.

        Args:
            data_path: Path to labeled data (.parquet or .csv)
            target_col: Target column name for prediction
            cluster_info_path: Path to final_cluster_info.json (optional)
            test_size: Hold-out test ratio (default 0.2)
            cv_folds: K-Fold CV folds (default 5)
            random_state: Random seed for reproducibility
            use_regularization: If True, apply strong regularization to models (default False)
        """
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.cluster_info_path = Path(cluster_info_path) if cluster_info_path else None
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_regularization = use_regularization

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
        print(f"Total samples: {len(self.df)}")
        print(f"Total descriptors: {len(self.descriptor_cols)}")
        print(f"Target: {self.target_col}")
        print(f"CV Folds: {cv_folds}")
        print(f"Test size: {test_size*100:.0f}%")
        print(f"Regularization: {'ON (strong)' if use_regularization else 'OFF (default params)'}")
        if self.cluster_info:
            print(f"Cluster info: {len(self.cluster_info['descriptors'])} representative descriptors")
        print("=" * 80 + "\n")

    def _load_data(self):
        """Load data from parquet or CSV"""
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

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
        """Prepare data: NaN handling + Scaling + Train/Test Split"""
        X = self.df[descriptors].values
        y = self.df[self.target_col].values

        # Handle NaN (mean imputation)
        mask = np.isnan(X)
        if mask.any():
            col_means = np.nanmean(X, axis=0)
            for col_idx in range(X.shape[1]):
                X[mask[:, col_idx], col_idx] = col_means[col_idx]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
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
