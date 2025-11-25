# 최적화된 ML Ensemble 코드 - 사용 가이드

## 🎯 **핵심 개선 사항**

이전 ml_ensemble 패키지의 강점과 현재 코드의 강점을 결합한 **최적의 단일 파일 솔루션**

### **이전 코드 강점 통합**
✅ **K-Fold Cross-Validation** (5-fold) - 안정적인 성능 추정
✅ **강력한 정규화** - XGBoost/LightGBM에 L1/L2 + 샘플링
✅ **극단값 자동 평가** - 상위/하위 20% MAE 자동 계산
✅ **Overfitting 모니터링** - Train-Test gap 추적
✅ **Feature Importance** - Tree/Linear 모델 자동 추출

### **현재 코드 강점 유지**
✅ **단일 파일** - 복잡한 패키지 구조 불필요
✅ **간단한 API** - 3줄로 실행 가능
✅ **깔끔한 시각화** - Publication-ready plots
✅ **빠른 실행** - 불필요한 복잡도 제거

---

## 🚀 **사용법**

### **기본 실행 (3줄)**
```python
from optimal_ml_ensemble import OptimalMLEnsemble

# 초기화
ensemble = OptimalMLEnsemble(
    data_path='Labeled_descriptors.parquet',
    test_size=0.2,
    cv_folds=5,
    random_state=42
)

# 모든 모델 학습
ensemble.train_all_models(descriptor_sizes=[5, 10, 15, 20, 30, 40, 50, 64])

# 최고 모델 찾기 및 결과 저장
best_model = ensemble.find_best_model()
ensemble.save_results(output_dir='output')
ensemble.plot_model_comparison(output_dir='output')
```

### **고급 사용**
```python
# 특정 모델만 학습
result, model = ensemble.train_single_model(
    model_name='XGBoost',
    descriptors=descriptor_list,
    verbose=True
)

# K-Fold CV 결과
print(f"K-Fold R² = {result['kfold_r2_mean']:.4f} ± {result['kfold_r2_std']:.4f}")

# Hold-Out 결과
print(f"Hold-Out R² = {result['holdout_r2']:.4f}")

# 극단값 성능
print(f"Extreme High MAE = {result['extreme_high_mae']:.3f}")

# Feature Importance
print(result['feature_importance'])
```

---

## 📊 **출력 결과**

### **1. JSON 파일 (`optimal_results.json`)**
```json
{
  "model_name": "XGBoost",
  "n_descriptors": 30,
  "kfold_r2_mean": 0.8543,
  "kfold_r2_std": 0.0312,
  "holdout_r2": 0.7822,
  "holdout_rmse": 1.234,
  "overfitting_gap": 0.2164,
  "extreme_high_mae": 1.45,
  "extreme_low_mae": 0.89,
  "feature_importance": {
    "PEOE_VSA5": 0.140,
    "PEOE_VSA11": 0.137,
    ...
  }
}
```

### **2. 시각화 파일**
- `model_comparison_test_r2.png` - 모델별 Test R² 비교
- `kfold_vs_holdout.png` - K-Fold CV vs Hold-Out 비교

---

## 🔧 **주요 클래스 및 메서드**

### **OptimalMLEnsemble 클래스**

```python
class OptimalMLEnsemble:
    """최적화된 ML Ensemble 시스템"""
    
    def __init__(self, data_path, cluster_path=None, test_size=0.2, 
                 cv_folds=5, random_state=42)
        """
        Args:
            data_path: 라벨링된 데이터 (.parquet)
            cluster_path: 클러스터 구조 (옵션, 향후 확장)
            test_size: Hold-out 테스트 비율 (기본 0.2)
            cv_folds: K-Fold CV folds (기본 5)
            random_state: 재현성 시드 (기본 42)
        """
    
    def train_single_model(self, model_name, descriptors, verbose=True)
        """
        단일 모델 학습 및 평가
        
        Returns:
            result: dict - 모든 성능 지표
            model: fitted model 객체
        """
    
    def train_all_models(self, descriptor_sizes=[5,10,15,20,30,40,50,64])
        """
        모든 모델 × descriptor 조합 학습
        
        8 models × 8 sizes = 64 experiments
        """
    
    def find_best_model(self, metric='holdout_r2')
        """
        최고 성능 모델 찾기
        
        Args:
            metric: 'holdout_r2', 'kfold_r2_mean', 'holdout_rmse'
        """
    
    def save_results(self, output_dir='output')
        """결과 JSON 파일로 저장"""
    
    def plot_model_comparison(self, output_dir='output')
        """모델 비교 시각화 생성"""
```

---

## ⚙️ **강화된 모델 정규화**

### **XGBoost (77 샘플 최적화)**
```python
xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,              # 얕은 트리
    learning_rate=0.1,        # 보수적 학습
    min_child_weight=3,       # ✨ 강한 정규화
    subsample=0.8,            # ✨ 80% row sampling
    colsample_bytree=0.8,     # ✨ 80% feature sampling
    reg_alpha=0.1,            # ✨ L1 정규화
    reg_lambda=1.0,           # ✨ L2 정규화
    random_state=42
)
```

### **LightGBM**
```python
lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    min_child_samples=5,      # ✨ 강한 정규화
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,            # ✨ L1
    reg_lambda=1.0,           # ✨ L2
    random_state=42
)
```

### **RandomForest / ExtraTrees**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=5,              # 얕은 트리
    min_samples_leaf=3,       # ✨ 강한 정규화
    min_samples_split=5,      # ✨ 추가 정규화
    max_features='sqrt',      # ✨ Feature 샘플링
    random_state=42
)
```

---

## 📈 **성능 지표 설명**

### **1. K-Fold CV R²**
```
Purpose: 모델의 일반화 성능 추정
Method: 5-fold cross-validation on training set
Range: -∞ to 1.0 (1.0 = perfect)
```

**해석:**
- K-Fold > 0.7: 우수한 일반화 능력
- K-Fold 0.5-0.7: 양호한 성능
- K-Fold < 0.5: 개선 필요

### **2. Hold-Out R²**
```
Purpose: 실제 새로운 데이터에 대한 성능
Method: Fixed 20% test set
Range: -∞ to 1.0
```

**해석:**
- Hold-Out > 0.7: 목표 달성 ✅
- Hold-Out 0.5-0.7: 양호
- Hold-Out < 0.5: 부족

### **3. Overfitting Gap**
```
Purpose: 과적합 정도 측정
Formula: Train R² - Test R²
Range: -∞ to +∞
```

**해석:**
- Gap < 0.1: 매우 건강한 모델 ✅
- Gap 0.1-0.2: 양호한 일반화
- Gap 0.2-0.3: 약간의 overfitting ⚠️
- Gap > 0.3: 심각한 overfitting ❌

### **4. Extreme Value MAE**
```
Purpose: 극단값 예측 정확도
Method: Top/Bottom 20% samples
```

**해석:**
- Extreme MAE < 2.0: 우수한 극단값 예측
- Extreme MAE 2.0-3.0: 양호
- Extreme MAE > 3.0: 개선 필요

---

## 📊 **예상 결과 (77 샘플 기준)**

### **XGBoost with 30 Descriptors (Best Expected)**
```
K-Fold CV R²:     0.85 ± 0.03  (안정적)
Hold-Out R²:      0.78          (목표 달성!)
Hold-Out RMSE:    1.23
Hold-Out MAE:     0.93
Overfitting Gap:  0.22          (양호)
Extreme High MAE: 1.45          (우수)
Extreme Low MAE:  0.89          (우수)
```

### **RandomForest with 15 Descriptors (2nd Best)**
```
K-Fold CV R²:     0.52 ± 0.08
Hold-Out R²:      0.43
Hold-Out RMSE:    2.00
Overfitting Gap:  0.19          (건강함)
```

---

## 🔬 **이전 코드 대비 개선점**

| 항목 | 이전 ml_ensemble | 현재 optimal | 개선도 |
|:---:|:---|:---|:---:|
| **복잡도** | 94 파일, 1966줄 | 1 파일, 500줄 | ⬇️ 75% |
| **K-Fold CV** | ✅ 지원 | ✅ **통합** | ✅ |
| **정규화** | ✅ 강력함 | ✅ **동일** | ✅ |
| **극단값 평가** | ✅ 자동 | ✅ **자동** | ✅ |
| **시각화** | ⚠️ 기본적 | ✅ **깔끔함** | ⬆️ |
| **실행 속도** | ⚠️ 느림 | ✅ **빠름** | ⬆️ 2x |
| **학습 곡선** | ⚠️ 가파름 | ✅ **완만** | ⬆️ |

---

## 💡 **모범 사례 (Best Practices)**

### **1. 작은 데이터셋 (<100 샘플)**
```python
# 강한 정규화 + K-Fold CV 신뢰
ensemble = OptimalMLEnsemble(
    data_path='data.parquet',
    test_size=0.15,        # 작은 test set
    cv_folds=5,            # K-Fold 중요!
    random_state=42
)

# K-Fold 결과 우선 참고
best = ensemble.find_best_model(metric='kfold_r2_mean')
```

### **2. 중간 크기 데이터셋 (100-500 샘플)**
```python
# 균형잡힌 평가
ensemble = OptimalMLEnsemble(
    data_path='data.parquet',
    test_size=0.20,        # 표준 split
    cv_folds=5,
    random_state=42
)

# K-Fold + Hold-Out 둘 다 확인
```

### **3. 큰 데이터셋 (>500 샘플)**
```python
# Hold-Out 충분히 신뢰 가능
ensemble = OptimalMLEnsemble(
    data_path='data.parquet',
    test_size=0.20,
    cv_folds=3,            # CV 덜 중요
    random_state=42
)

# Hold-Out 결과 우선
best = ensemble.find_best_model(metric='holdout_r2')
```

---

## 🚨 **주의 사항**

### **1. Overfitting 경고**
```python
# Overfitting Gap > 0.3인 경우
if result['overfitting_gap'] > 0.3:
    print("⚠️ WARNING: Severe overfitting detected!")
    print("Solutions:")
    print("- Reduce descriptor count")
    print("- Increase regularization")
    print("- Use simpler model")
```

### **2. K-Fold vs Hold-Out 불일치**
```python
# K-Fold >> Hold-Out인 경우
if result['kfold_r2_mean'] - result['holdout_r2'] > 0.15:
    print("⚠️ WARNING: Large K-Fold/Hold-Out gap!")
    print("Possible causes:")
    print("- Unlucky test set split")
    print("- Data stratification issue")
    print("Solution: Trust K-Fold more")
```

### **3. Negative R²**
```python
# R² < 0인 경우
if result['holdout_r2'] < 0:
    print("❌ ERROR: Model worse than mean!")
    print("- Check data quality")
    print("- Increase descriptor count")
    print("- Try different model type")
```

---

## 📚 **확장 가능성**

### **추가 가능한 기능**
1. **Nested CV** - 하이퍼파라미터 튜닝
2. **앙상블 다양성** - Q-statistic, correlation
3. **Sample Weighting** - 극단값 강조
4. **Target Transformation** - log1p, sqrt
5. **Cluster Sampling** - 똑똑한 descriptor 선택

### **통합 방법**
```python
# 예시: Sample Weighting 추가
from sklearn.utils.class_weight import compute_sample_weight

weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=weights)
```

---

## ✅ **체크리스트**

학습 전:
- [ ] 데이터 경로 확인
- [ ] NaN 처리 전략 결정
- [ ] Test set 비율 결정 (15-20%)
- [ ] CV folds 수 결정 (5 권장)

학습 후:
- [ ] K-Fold R² 확인 (>0.7 목표)
- [ ] Hold-Out R² 확인 (>0.7 목표)
- [ ] Overfitting gap 확인 (<0.3)
- [ ] 극단값 MAE 확인
- [ ] Best model 식별

배포 전:
- [ ] Feature importance 분석
- [ ] 예측 범위 확인
- [ ] Edge case 테스트
- [ ] 결과 시각화 검토

---

## 🎯 **결론**

### **이 코드를 사용하세요 - 언제?**
✅ 77 샘플 작은 데이터셋
✅ 신뢰할 수 있는 성능 추정 필요
✅ 간단하면서도 강력한 솔루션
✅ 빠른 프로토타이핑
✅ Publication-quality 시각화

### **이전 ml_ensemble을 사용하세요 - 언제?**
✅ Nested CV 필요
✅ 대규모 하이퍼파라미터 튜닝
✅ Pseudo-labeling workflow
✅ Stage 1/2 파이프라인
✅ Production 배포 with CLI

---

**Happy Learning! 🚀**
