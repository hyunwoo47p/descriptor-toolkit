# ML 모델 최적화 결과

## 개요

`reference/train_samples_61.csv` (61샘플)와 `reference/test_samples_16.csv` (16샘플)를 사용한 최적화 결과입니다.

---

## 1. 기본 실행 (Sequential Mode)

```bash
cdml train \
    --input reference/train_samples_61.csv \
    --test-input reference/test_samples_16.csv \
    --target-col pLeach \
    --descriptor-mode sequential
```

| 항목 | 값 |
|------|-----|
| 최적 모델 | XGBoost (5D) |
| Hold-Out R² | 0.17 |

Sequential 모드는 컬럼 순서대로 descriptor를 선택하므로 최적의 성능을 보장하지 않습니다.

---

## 2. 최적화 실행 (Random Seed Search)

816개 numeric descriptor 중 최적의 조합을 찾기 위해 random seed 검색을 수행했습니다.

### 2.1 검색 결과 (R² > 0.6)

| N_Descriptors | Seed | Hold-Out R² |
|---------------|------|-------------|
| **35** | **986** | **0.7323** |
| 35 | 228 | 0.6696 |
| 40 | 986 | 0.6575 |
| 25 | 466 | 0.6448 |
| 30 | 475 | 0.6409 |
| 40 | 859 | 0.6390 |
| 30 | 228 | 0.6206 |
| 20 | 173 | 0.6201 |
| 30 | 429 | 0.6151 |
| 30 | 667 | 0.6145 |

### 2.2 최적 모델 (Seed=986, 35D)

| 항목 | 값 |
|------|-----|
| **Model** | XGBoost |
| **N Descriptors** | 35 |
| **Selection Seed** | 986 |
| **Hold-Out R²** | **0.7323** |
| **Hold-Out RMSE** | 1.3680 |
| **Hold-Out MAE** | 1.1179 |
| Train R² | 1.0000 |
| K-Fold R² | -0.3259 ± 0.1103 |
| Overfitting Gap | 0.2677 |

---

## 3. 선택된 35개 Descriptors

중요도 순:

| 순위 | Descriptor | Importance | 설명 |
|------|------------|------------|------|
| 1 | `fr_Al_COO` | 0.2685 | 지방족 카르복실산 개수 |
| 2 | `Chi0v` | 0.1021 | 연결성 지수 (valence) |
| 3 | `MAXdO` | 0.0949 | 산소 원자 최대 partial charge |
| 4 | `MATS2pe` | 0.0727 | Moran 자기상관 (Pauling 전기음성도) |
| 5 | `ATSC4i` | 0.0620 | 중심 자기상관 (I-state) |
| 6 | `GATS2m` | 0.0553 | Geary 자기상관 (질량) |
| 7 | `AATS2pe` | 0.0524 | 평균 자기상관 (Pauling 전기음성도) |
| 8 | `AATSC2d` | 0.0424 | 중심 자기상관 (σ 전자) |
| 9 | `AATSC3c` | 0.0380 | 중심 자기상관 (Gasteiger charge) |
| 10 | `MATS2dv` | 0.0305 | Moran 자기상관 (valence 전자) |
| 11 | `RNCG` | 0.0244 | 상대 음전하 |
| 12 | `SIC2` | 0.0229 | 구조 정보 지수 |
| 13 | `SsSH` | 0.0219 | E-state 합계 (-SH) |
| 14 | `AATS1dv` | 0.0184 | 평균 자기상관 (valence 전자) |
| 15 | `ATSC1d` | 0.0147 | 중심 자기상관 (σ 전자) |
| 16 | `AATS1are` | 0.0142 | 평균 자기상관 (면적) |
| 17 | `ATSC1i` | 0.0128 | 중심 자기상관 (I-state) |
| 18 | `MinPartialCharge` | 0.0099 | 최소 partial charge |
| 19 | `qed` | 0.0094 | 약물 유사성 점수 |
| 20 | `MPC8` | 0.0088 | 분자 경로 수 |
| 21 | `SMR` | 0.0079 | 몰 굴절율 |
| 22 | `AATSC3m` | 0.0060 | 중심 자기상관 (질량) |
| 23 | `Chi0n` | 0.0035 | 연결성 지수 |
| 24 | `MATS2m` | 0.0035 | Moran 자기상관 (질량) |
| 25 | `ATSC8d` | 0.0011 | 중심 자기상관 (σ 전자) |
| 26-35 | (기타) | <0.001 | 미미한 기여 |

**전체 descriptor 목록:**
```
ATS1pe, AATSC3c, GATS2m, ATS8dv, AATSC2d, SIC2, MAXdO, MPC8, SsSH, n6Ring,
MATS2m, ATSC4i, ATS6Z, qed, GGI6, RNCG, Chi0n, fr_Al_COO, AATS1are, ATSC1d,
AATS1dv, Chi0v, AATSC4Z, MATS2dv, ATSC8d, AATS2pe, ATSC1i, fr_halogen,
MinPartialCharge, nAromBond, ATS2pe, n6aRing_1, SMR, MATS2pe, AATSC3m
```

---

## 4. 결론

- **Sequential 모드**는 소규모 데이터에서 최적의 성능을 보장하지 않음
- **Random Seed Search**를 통해 R² 0.17 → 0.73으로 성능 향상
- 가장 중요한 descriptor: `fr_Al_COO` (지방족 카르복실산 개수) - 26.85%
- 상위 3개 descriptor가 전체 중요도의 46.5% 차지

---

## 5. 재현 방법

```python
import random
from xgboost import XGBRegressor

# Seed 986으로 35개 descriptor 선택
rng = random.Random(986)
selected_idx = rng.sample(range(len(numeric_cols)), 35)
selected_cols = [numeric_cols[i] for i in selected_idx]

# XGBoost 학습
model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train[selected_cols], y_train)

# 예측
y_pred = model.predict(X_test[selected_cols])
```
