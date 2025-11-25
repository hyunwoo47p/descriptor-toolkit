"""
최적화된 ML Ensemble 코드
- 이전 ml_ensemble의 강점: K-fold CV, 정규화, 극단값 평가
- 현재 코드의 강점: 간단함, 깔끔한 시각화
- 77 샘플 작은 데이터셋에 최적화
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class OptimalMLEnsemble:
    """
    최적화된 ML Ensemble 시스템
    
    핵심 기능:
    - K-Fold Cross-Validation (신뢰성)
    - Hold-Out Test Set (현실성)
    - 강력한 정규화 (Overfitting 방지)
    - 극단값 평가 (중요 영역 성능)
    - 깔끔한 시각화 (Publication-ready)
    """
    
    def __init__(self, data_path, cluster_path=None, test_size=0.2, cv_folds=5, random_state=42):
        """
        Args:
            data_path: 라벨링된 데이터 경로
            cluster_path: 클러스터 구조 (옵션)
            test_size: Hold-out 테스트 비율
            cv_folds: K-Fold CV folds 수
            random_state: 재현성을 위한 시드
        """
        self.data_path = data_path
        self.cluster_path = cluster_path
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # 결과 저장
        self.results = []
        self.best_models = {}
        
        # 데이터 로드
        self._load_data()
        
        print("="*80)
        print("OPTIMAL ML ENSEMBLE INITIALIZED")
        print("="*80)
        print(f"Total samples: {len(self.df)}")
        print(f"Total descriptors: {len(self.descriptor_cols)}")
        print(f"Target: {self.target_col}")
        print(f"CV Folds: {cv_folds}")
        print(f"Test size: {test_size*100:.0f}%")
        print("="*80 + "\n")
    
    def _load_data(self):
        """데이터 로드 및 기본 처리"""
        self.df = pd.read_parquet(self.data_path)
        
        # Target과 descriptor 분리
        self.target_col = 'pLeach'
        self.descriptor_cols = [col for col in self.df.columns if col != self.target_col]
        
        print(f"✅ Data loaded: {len(self.df)} samples, {len(self.descriptor_cols)} descriptors")
    
    def _get_model_with_regularization(self, model_name):
        """
        강력한 정규화가 적용된 모델 생성
        77 샘플 작은 데이터셋에 최적화
        """
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,           # 얕은 트리
                min_samples_leaf=3,    # 강한 정규화
                min_samples_split=5,   # 추가 정규화
                max_features='sqrt',   # Feature 샘플링
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
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_child_weight=3,    # 강한 정규화 ✨
                subsample=0.8,         # Row sampling ✨
                colsample_bytree=0.8,  # Column sampling ✨
                reg_alpha=0.1,         # L1 정규화 ✨
                reg_lambda=1.0,        # L2 정규화 ✨
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_child_samples=5,   # 강한 정규화 ✨
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,         # L1 정규화 ✨
                reg_lambda=1.0,        # L2 정규화 ✨
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'Ridge': Ridge(
                alpha=1.0,             # L2 정규화
                random_state=self.random_state
            ),
            'Lasso': Lasso(
                alpha=0.1,             # L1 정규화
                max_iter=10000,
                random_state=self.random_state
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,          # L1+L2 혼합
                max_iter=10000,
                random_state=self.random_state
            ),
            'GPR': GaussianProcessRegressor(
                kernel=C(1.0) * RBF(1.0) + WhiteKernel(1.0),
                random_state=self.random_state,
                n_restarts_optimizer=3
            )
        }
        return models[model_name]
    
    def _prepare_data(self, descriptors):
        """데이터 준비: NaN 처리 + Scaling + Train/Test Split"""
        X = self.df[descriptors].values
        y = self.df[self.target_col].values
        
        # NaN 처리 (평균값 대체)
        mask = np.isnan(X)
        if mask.any():
            col_means = np.nanmean(X, axis=0)
            for col_idx in range(X.shape[1]):
                X[mask[:, col_idx], col_idx] = col_means[col_idx]
        
        # Train/Test Split (고정)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _compute_extreme_metrics(self, y_true, y_pred):
        """
        극단값 영역별 성능 계산
        상위 20% / 하위 20% MAE
        """
        # 상위 20% (High extremes)
        high_threshold = np.percentile(y_true, 80)
        high_mask = y_true > high_threshold
        
        # 하위 20% (Low extremes)
        low_threshold = np.percentile(y_true, 20)
        low_mask = y_true < low_threshold
        
        results = {}
        
        if high_mask.sum() > 0:
            results['extreme_high_mae'] = mean_absolute_error(
                y_true[high_mask], y_pred[high_mask]
            )
            results['extreme_high_n'] = high_mask.sum()
        else:
            results['extreme_high_mae'] = None
            results['extreme_high_n'] = 0
        
        if low_mask.sum() > 0:
            results['extreme_low_mae'] = mean_absolute_error(
                y_true[low_mask], y_pred[low_mask]
            )
            results['extreme_low_n'] = low_mask.sum()
        else:
            results['extreme_low_mae'] = None
            results['extreme_low_n'] = 0
        
        return results
    
    def train_single_model(self, model_name, descriptors, verbose=True):
        """
        단일 모델 학습 및 평가
        - K-Fold CV (신뢰성)
        - Hold-Out Test (현실성)
        - 극단값 평가 (중요 영역)
        """
        n_desc = len(descriptors)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"{model_name} with {n_desc} Descriptors")
            print(f"{'='*80}")
        
        # 데이터 준비
        X_train, X_test, y_train, y_test = self._prepare_data(descriptors)
        
        # 모델 생성 (강력한 정규화)
        model = self._get_model_with_regularization(model_name)
        
        # === 1. K-Fold Cross-Validation (Training set) ===
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Negative MSE를 R²로 변환하기 위해 직접 계산
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
        
        # Training set 성능
        y_pred_train = model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        
        # Test set 성능
        y_pred_test = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Overfitting 지표
        overfitting_gap = train_r2 - test_r2
        
        if verbose:
            print(f"  Train R² = {train_r2:.4f}, RMSE = {train_rmse:.3f}")
            print(f"  Test  R² = {test_r2:.4f}, RMSE = {test_rmse:.3f}")
            print(f"  Overfitting Gap = {overfitting_gap:.4f}")
        
        # === 3. 극단값 평가 ===
        extreme_metrics = self._compute_extreme_metrics(y_test, y_pred_test)
        
        if verbose and extreme_metrics['extreme_high_mae'] is not None:
            print(f"  Extreme High MAE = {extreme_metrics['extreme_high_mae']:.3f} "
                  f"(n={extreme_metrics['extreme_high_n']})")
        if verbose and extreme_metrics['extreme_low_mae'] is not None:
            print(f"  Extreme Low MAE = {extreme_metrics['extreme_low_mae']:.3f} "
                  f"(n={extreme_metrics['extreme_low_n']})")
        
        # === 4. Feature Importance (tree-based 모델) ===
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                desc: float(imp) 
                for desc, imp in zip(descriptors, model.feature_importances_)
            }
        elif hasattr(model, 'coef_'):  # Linear 모델
            feature_importance = {
                desc: float(abs(coef)) 
                for desc, coef in zip(descriptors, model.coef_)
            }
        
        # 결과 저장
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
            
            # 극단값
            'extreme_high_mae': extreme_metrics['extreme_high_mae'],
            'extreme_high_n': extreme_metrics['extreme_high_n'],
            'extreme_low_mae': extreme_metrics['extreme_low_mae'],
            'extreme_low_n': extreme_metrics['extreme_low_n'],
            
            # Feature importance
            'feature_importance': feature_importance
        }
        
        self.results.append(result)
        
        return result, model
    
    def train_all_models(self, descriptor_sizes=[5, 10, 15, 20, 30, 40, 50, 64]):
        """
        모든 모델과 descriptor 조합 학습
        """
        models = ['XGBoost', 'RandomForest', 'ExtraTrees', 'ElasticNet', 
                  'GPR', 'Ridge', 'Lasso', 'LightGBM']
        
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        print(f"Models: {len(models)}")
        print(f"Descriptor sizes: {descriptor_sizes}")
        print(f"Total experiments: {len(models) * len(descriptor_sizes)}")
        print("="*80)
        
        total = len(models) * len(descriptor_sizes)
        count = 0
        
        for model_name in models:
            for n_desc in descriptor_sizes:
                count += 1
                print(f"\n[{count}/{total}] {model_name} - {n_desc}D")
                
                # Descriptor 선택 (상위 n개)
                descriptors = self.descriptor_cols[:n_desc]
                
                # 학습
                result, model = self.train_single_model(
                    model_name, descriptors, verbose=False
                )
                
                # Best 모델 추적
                key = f"{model_name}_{n_desc}D"
                self.best_models[key] = {
                    'model': model,
                    'result': result
                }
        
        print(f"\n{'='*80}")
        print("✅ ALL MODELS TRAINED")
        print(f"{'='*80}\n")
    
    def find_best_model(self, metric='holdout_r2'):
        """최고 성능 모델 찾기"""
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
    
    def save_results(self, output_dir='output'):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # JSON 저장
        with open(output_path / 'optimal_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved to {output_path / 'optimal_results.json'}")
    
    def plot_model_comparison(self, output_dir='output'):
        """모델 비교 시각화"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 모델별 최고 성능 추출
        models = ['XGBoost', 'RandomForest', 'ExtraTrees', 'ElasticNet',
                  'GPR', 'Ridge', 'Lasso', 'LightGBM']
        
        best_by_model = {}
        for model_name in models:
            model_results = [r for r in self.results if r['model_name'] == model_name]
            if model_results:
                best = max(model_results, key=lambda x: x['holdout_r2'])
                best_by_model[model_name] = best
        
        # 색상
        colors = {
            'XGBoost': '#8dd3c7',
            'RandomForest': '#ffffb3',
            'ExtraTrees': '#fb8072',
            'ElasticNet': '#fdb462',
            'GPR': '#b3de69',
            'Ridge': '#fccde5',
            'Lasso': '#bc80bd',
            'LightGBM': '#ffed6f'
        }
        
        # === 1. Test R² 비교 ===
        fig, ax = plt.subplots(figsize=(12, 8))
        
        test_r2_values = [best_by_model[m]['holdout_r2'] for m in models]
        bars = ax.bar(range(len(models)), test_r2_values,
                     color=[colors[m] for m in models], alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=13, fontweight='bold')
        ax.set_ylabel('Test R²', fontsize=16, fontweight='bold')
        ax.set_title('Model Performance Comparison - Test R²', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=13)
        
        # 막대 위 라벨
        for i, m in enumerate(models):
            v = best_by_model[m]['holdout_r2']
            ax.text(i, v + 0.02, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison_test_r2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved: {output_path / 'model_comparison_test_r2.png'}")
        
        # === 2. K-Fold vs Hold-Out 비교 ===
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
        ax.set_xticklabels(models, fontsize=13, fontweight='bold')
        ax.set_ylabel('R²', fontsize=16, fontweight='bold')
        ax.set_title('K-Fold CV vs Hold-Out Test Performance', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=13)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=13)
        
        plt.tight_layout()
        plt.savefig(output_path / 'kfold_vs_holdout.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved: {output_path / 'kfold_vs_holdout.png'}")


def main():
    """메인 실행 함수"""
    
    # 초기화
    ensemble = OptimalMLEnsemble(
        data_path='/mnt/user-data/uploads/Labeled_descriptors.parquet',
        test_size=0.2,
        cv_folds=5,
        random_state=42
    )
    
    # 모든 모델 학습
    ensemble.train_all_models(descriptor_sizes=[5, 10, 15, 20, 30, 40, 50, 64])
    
    # 최고 모델 찾기
    best_model = ensemble.find_best_model(metric='holdout_r2')
    
    # 결과 저장
    ensemble.save_results(output_dir='/mnt/user-data/outputs')
    
    # 시각화
    ensemble.plot_model_comparison(output_dir='/mnt/user-data/outputs')
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
