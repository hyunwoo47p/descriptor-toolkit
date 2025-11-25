# Cluster Backtracking 사용 가이드

## 📋 개요

`surviving_descriptors_clusters.json` 파일은 최종 surviving descriptors의 클러스터 구조를 담고 있습니다. 각 descriptor에 대해 Pass 4 → 3 → 2 → 1 순서로 역추적하여 **재귀적으로** 모든 클러스터 멤버를 찾아냅니다.

---

## 🎯 핵심 개념

### 재귀적 역추적 예시

```
Pass 3: A가 대표, B 제거 (A-B 클러스터)
Pass 2: B가 대표, C 제거 (B-C 클러스터)
→ 최종: A의 all_cluster_members = {A, B, C}
```

**설명:**
1. A가 최종 생존 (Pass 4까지 통과)
2. Pass 3에서 A-B 클러스터, A가 대표 → B 추적 목록에 추가
3. Pass 2에서 B-C 클러스터, B가 대표 → C도 A의 클러스터에 포함
4. 결과: A의 alternative_descriptors = [B, C]

---

## 🚀 사용 방법

### 방법 1: Pipeline에서 자동 생성 (권장)

```python
from descriptor_pipeline.core.pipeline import DescriptorPipeline
from descriptor_pipeline.config.settings import PipelineConfig

config = PipelineConfig(
    parquet_glob="data/*.parquet",
    output_dir="output/results",
    checkpoint=True,  # 중요: checkpoint를 켜야 생성됨
    verbose=True
)

pipeline = DescriptorPipeline(config)
results = pipeline.run()

# surviving_descriptors_clusters.json 자동 생성됨
```

**출력 파일:**
- `output/results/surviving_descriptors_clusters.json`

---

### 방법 2: 독립 실행 (이미 checkpoint가 있는 경우)

```python
from cluster_backtracker import create_cluster_structure

structure = create_cluster_structure(
    output_dir='output/results',
    verbose=True
)

print(f"Total descriptors: {structure['metadata']['total_descriptors']}")
print(f"With alternatives: {structure['metadata']['descriptors_with_alternatives']}")
```

**CLI 사용:**
```bash
python cluster_backtracker.py \
    --output-dir output/results \
    --final-descriptors output/results/final_descriptors.txt \
    --output-file output/results/surviving_descriptors_clusters.json
```

---

## 📊 JSON 파일 구조

```json
{
  "metadata": {
    "description": "Cluster structure for 337 surviving descriptors",
    "total_descriptors": 337,
    "descriptors_with_alternatives": 92,
    "standalone_descriptors": 245,
    "total_alternative_descriptors": 772
  },
  
  "statistics": {
    "cluster_size_mean": 3.29,
    "cluster_size_median": 1,
    "cluster_size_min": 1,
    "cluster_size_max": 400,
    "size_distribution": {
      "1": 245,  # 245개 descriptors가 클러스터 크기 1
      "2": 39,   # 39개 descriptors가 클러스터 크기 2
      "3": 37,
      ...
    }
  },
  
  "descriptors": {
    "AATS8Z": {
      "cluster_size": 2,
      "is_representative": true,
      "alternative_descriptors": ["AATS8m"],
      "all_cluster_members": ["AATS8Z", "AATS8m"],
      "removal_history": {
        "pass3": ["AATS8m"]  # Pass 3에서 AATS8m이 제거됨
      },
      "total_alternatives": 1
    },
    
    "AATSC7v": {
      "cluster_size": 3,
      "is_representative": true,
      "alternative_descriptors": ["MATS7v", "SomeOther"],
      "all_cluster_members": ["AATSC7v", "MATS7v", "SomeOther"],
      "removal_history": {
        "pass4": ["MATS7v"],
        "pass2": ["SomeOther"]
      },
      "total_alternatives": 2
    }
  }
}
```

---

## 🔍 필드 설명

### Metadata
- `total_descriptors`: 최종 생존한 descriptors 수
- `descriptors_with_alternatives`: alternative가 있는 descriptors 수
- `standalone_descriptors`: 혼자인 descriptors 수 (클러스터 크기 1)
- `total_alternative_descriptors`: 모든 alternative descriptors의 합

### Statistics
- `cluster_size_*`: 클러스터 크기 통계
- `size_distribution`: 크기별 분포 (크기: 개수)

### Descriptor Fields
- `cluster_size`: 이 descriptor의 클러스터 크기
- `is_representative`: 항상 true (최종 생존자이므로)
- `alternative_descriptors`: 대체 가능한 descriptors 리스트
- `all_cluster_members`: 모든 클러스터 멤버 (본인 포함)
- `removal_history`: Pass별 제거 히스토리
  - Key: pass 이름 (pass2, pass3, pass4)
  - Value: 해당 pass에서 제거된 descriptors 리스트
- `total_alternatives`: alternative 개수

---

## 💡 활용 예시

### 1. Alternative Descriptors 찾기

```python
import json

with open('output/results/surviving_descriptors_clusters.json', 'r') as f:
    data = json.load(f)

# "AATS8Z" descriptor의 alternative 찾기
descriptor = "AATS8Z"
info = data['descriptors'][descriptor]

print(f"Representative: {descriptor}")
print(f"Alternatives: {info['alternative_descriptors']}")
print(f"All members: {info['all_cluster_members']}")
```

### 2. 큰 클러스터 찾기

```python
# 클러스터 크기가 10 이상인 descriptors 찾기
large_clusters = {
    desc: info 
    for desc, info in data['descriptors'].items() 
    if info['cluster_size'] >= 10
}

print(f"Found {len(large_clusters)} large clusters")
for desc, info in sorted(large_clusters.items(), 
                         key=lambda x: x[1]['cluster_size'], 
                         reverse=True):
    print(f"  {desc}: {info['cluster_size']} members")
```

### 3. 제거 히스토리 분석

```python
# Pass별 제거 통계
removal_stats = {'pass2': 0, 'pass3': 0, 'pass4': 0}

for desc, info in data['descriptors'].items():
    for pass_name, removed_list in info['removal_history'].items():
        removal_stats[pass_name] += len(removed_list)

print("Removal statistics:")
for pass_name, count in removal_stats.items():
    print(f"  {pass_name}: {count} descriptors removed")
```

### 4. Descriptor 교체 추천

```python
def get_alternatives(descriptor, data):
    """특정 descriptor의 alternatives 반환"""
    if descriptor not in data['descriptors']:
        return []
    return data['descriptors'][descriptor]['alternative_descriptors']

# 사용 예시
alternatives = get_alternatives("AATS8Z", data)
print(f"If AATS8Z is problematic, use: {alternatives}")
```

---

## 🔧 고급 사용법

### Custom Backtracker 클래스

```python
from cluster_backtracker import ClusterBacktracker

backtracker = ClusterBacktracker(
    output_dir='output/results',
    verbose=True
)

# Checkpoint 로드
backtracker.load_checkpoints()

# 특정 descriptors만 추적
descriptors_to_track = ['AATS8Z', 'AATSC7v']
cluster_info = backtracker.backtrack_clusters(descriptors_to_track)

# 결과 확인
for desc, info in cluster_info.items():
    print(f"\n{desc}:")
    print(f"  Total members: {len(info['all_cluster_members'])}")
    print(f"  Alternatives: {info['alternative_descriptors']}")
```

---

## 📝 주의사항

### 1. Checkpoint 파일 필수
```python
config = PipelineConfig(
    checkpoint=True  # 반드시 True!
)
```

**필요한 checkpoint 파일:**
- `pass2_spearman.json`
- `pass3_vif.json`
- `pass4_nonlinear.json`

### 2. Pass 1은 클러스터 정보 없음
- Pass 1은 variance filtering만 수행 (클러스터링 없음)
- 따라서 Pass 2부터 역추적 시작

### 3. 재귀적 추적 깊이
- 모든 Pass를 거슬러 올라가며 완전 추적
- 메모리에 유의 (매우 큰 클러스터의 경우)

---

## 🐛 문제 해결

### Q1: JSON 파일이 생성되지 않아요
**A**: `checkpoint=True`로 설정했는지 확인하세요.

```python
config = PipelineConfig(
    checkpoint=True,  # 이것이 없으면 생성 안됨
    ...
)
```

### Q2: Alternative가 비어있어요
**A**: 
- 해당 descriptor가 standalone일 수 있습니다 (클러스터 크기 1)
- 모든 Pass에서 혼자 남았다는 의미입니다

### Q3: 제거 히스토리가 비어있어요
**A**:
- 정상입니다 - 모든 descriptor가 제거 히스토리를 가지는 것은 아닙니다
- 이 descriptor가 어떤 Pass에서도 다른 descriptor를 제거하지 않았다는 의미

### Q4: 클러스터 크기가 너무 큽니다 (수백 개)
**A**:
- 정상일 수 있습니다 - 매우 상관관계가 높은 descriptor 그룹
- 재귀적 추적으로 여러 Pass를 거쳐 누적된 결과
- `removal_history`를 확인하여 어느 Pass에서 추가되었는지 확인

---

## 📚 관련 문서

- Pipeline 사용 가이드: `IMPLEMENTATION_GUIDE.md`
- Checkpoint 파일 구조: 각 Pass의 JSON 파일 참조
- 메모리 관리: `MEMORY_LEAK_COMPREHENSIVE_DIAGNOSIS.md`

---

## ✅ 체크리스트

생성 전:
- [ ] `checkpoint=True` 설정
- [ ] Pass 2, 3, 4 모두 실행 완료
- [ ] Checkpoint 파일 존재 확인

생성 후:
- [ ] `surviving_descriptors_clusters.json` 파일 확인
- [ ] Metadata의 total_descriptors 수 확인
- [ ] 큰 클러스터 확인 (size_distribution)
- [ ] Alternative descriptors 활용

---

🎉 **완료!**

이제 surviving descriptors의 완전한 클러스터 구조를 확인할 수 있습니다!
