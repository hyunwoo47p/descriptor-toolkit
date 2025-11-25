# Descriptor Pipeline 개선사항 요약

## 🔴 발견된 주요 문제점

### 1. 메모리 관리 문제
- **문제**: `max_gpu_memory_gb` 설정이 Config에 없어 기본값만 사용
- **해결**: PipelineConfig에 `max_gpu_memory_gb` 필드 추가 (기본값 40.0 GB)
- **영향**: RTX 6000 Ada (48GB)에 최적화된 메모리 관리 가능

### 2. 체크포인트 기능 미완성
- **문제**: JSON 저장만 하고 재시작 시 불러오는 로직 없음
- **해결**: 
  - 각 Pass별 완료 체크 기능 추가
  - 중간 결과 파일 저장/로드 기능 구현
  - 재시작 시 이미 완료된 Pass 자동 스킵
- **영향**: 긴 작업 중 중단되어도 처음부터 다시 시작할 필요 없음

### 3. GPU 메모리 정리 불완전
- **문제**: 일부 구간에서 명시적 메모리 정리 누락
- **해결**: Pass 간 전환 시 torch.cuda.empty_cache() 호출 추가
- **영향**: 메모리 누적 방지, OOM 에러 감소

## ✅ 주요 개선사항

### 1. Config 수정 (settings.py)
```python
# 추가된 설정
max_gpu_memory_gb: float = 40.0  # GPU 메모리 제한 (GB)
```

### 2. Pipeline 개선 (pipeline_improved.py)
**새로운 기능:**
- ✅ 체크포인트 재개 (Checkpoint Resume)
  - `_check_checkpoint()`: 완료된 Pass 확인
  - `_load_checkpoint()`: 저장된 상태 로드
  - `_save_intermediate_columns()`: 중간 컬럼 리스트 저장
  - `_load_intermediate_columns()`: 중간 컬럼 리스트 로드

**저장되는 체크포인트 파일:**
```
output_dir/
├── pass1_variance_filtering.json    # Pass1 완료 정보
├── pass1_columns.txt                # Pass1 결과 컬럼
├── pass1_stats.npz                  # Pass1 통계
├── pass2_spearman.json              # Pass2 완료 정보
├── pass2_columns.txt                # Pass2 결과 컬럼
├── pass2_spearman_matrix.npy        # Spearman 상관행렬
├── pass3_vif.json                   # Pass3 완료 정보
├── pass3_columns.txt                # Pass3 결과 컬럼
├── pass4_nonlinear.json             # Pass4 완료 정보
└── final_descriptors.txt            # 최종 결과
```

**재시작 동작:**
```python
# Pass1이 완료된 경우
if checkpoint exists:
    columns_p1 = load_from_file("pass1_columns.txt")
    stats_p1 = load_from_file("pass1_stats.npz")
    # Pass1 스킵, Pass2부터 시작
else:
    # Pass1 실행
```

### 3. 메모리 관리 개선
```python
# Pass 전환 시 GPU 메모리 정리
if self.using_gpu:
    import torch
    del large_matrix
    torch.cuda.empty_cache()
```

## 📝 명령어 사용법

### 기존 명령어 (문제 없음)
```bash
python -u -m descriptor_pipeline.cli.run_pipeline \
  --parquet-glob "/home/ML_data/pubchem/Compound/Filtered_Descriptor/*.parquet" \
  --output-dir "/home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3" \
  --n-metadata 6 \
  --gpu \
  --verbose \
  --batch-rows 50000 \
  --variance-threshold 0.002 \
  --range-mode trimmed \
  --trim-lower 2.5 \
  --trim-upper 97.5 \
  --max-missing-ratio 0.9 \
  --spearman-threshold 0.90 \
  --vif-threshold 10 \
  --nonlinear-threshold 0.7 \
  --w-hsic 0.3 \
  --w-rdc 0.7 \
  --m 8192 \
  --r 5 \
  --hsic-D 8 \
  --rdc-d 12 \
  --rdc-seeds 2 \
  --topk 40 \
  --leiden-resolution 1.0 \
  --n-consensus 10 \
  --random-seed 1557 \
  --checkpoint \
  2>&1 | tee /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/output.log
```

### ✅ 모든 인자가 올바르게 지원됨

**확인된 인자:**
- ✅ `--parquet-glob`: 입력 파일 패턴
- ✅ `--output-dir`: 출력 디렉토리
- ✅ `--n-metadata`: 메타데이터 컬럼 수 (기본값 6)
- ✅ `--gpu`: GPU 사용
- ✅ `--verbose`: 상세 로그
- ✅ `--batch-rows`: 배치 크기 (기본값 10000, 명령어에서 50000)
- ✅ `--variance-threshold`: 분산 임계값
- ✅ `--range-mode`: 범위 계산 모드
- ✅ `--trim-lower`, `--trim-upper`: Trimmed 백분위수
- ✅ `--max-missing-ratio`: 최대 결측치 비율
- ✅ `--spearman-threshold`: Spearman 상관 임계값
- ✅ `--vif-threshold`: VIF 임계값
- ✅ `--nonlinear-threshold`: 비선형 유사도 임계값
- ✅ `--w-hsic`, `--w-rdc`: HSIC/RDC 가중치
- ✅ `--m`, `--r`: CountSketch 파라미터
- ✅ `--hsic-D`: HSIC 차원
- ✅ `--rdc-d`: RDC 차원
- ✅ `--rdc-seeds`: RDC 시드 수
- ✅ `--topk`: k-NN 그래프 k값
- ✅ `--leiden-resolution`: Leiden 해상도
- ✅ `--n-consensus`: Consensus 반복 횟수
- ✅ `--random-seed`: 랜덤 시드
- ✅ `--checkpoint`: 체크포인트 활성화

## 🚀 실행 방법

### 1. 파일 교체
```bash
# 기존 파일 백업
cp descriptor_pipeline/config/settings.py descriptor_pipeline/config/settings.py.backup

# 개선된 파일로 교체
cp settings.py descriptor_pipeline/config/settings.py
cp pipeline_improved.py descriptor_pipeline/core/pipeline.py
```

### 2. 실행
```bash
# 기존 명령어 그대로 사용 가능
python -u -m descriptor_pipeline.cli.run_pipeline \
  [기존 옵션들...]
```

### 3. 재시작 (중단된 경우)
```bash
# 같은 명령어로 재실행하면 자동으로 체크포인트부터 재개
python -u -m descriptor_pipeline.cli.run_pipeline \
  [동일한 옵션들...]
```

## 📊 체크포인트 동작 예시

### 시나리오 1: Pass2에서 중단된 경우
```
실행 1: Pass0 → Pass1 → Pass2 (중단)
저장된 파일:
  ✓ pass1_columns.txt
  ✓ pass1_stats.npz
  ✓ pass1_variance_filtering.json

실행 2 (재시작):
  ✓ Pass0: 건너뜀 (sampled_data.parquet 존재)
  ✓ Pass1: 건너뜀 (체크포인트 로드)
  → Pass2: 여기서부터 재개
  → Pass3, Pass4: 계속 진행
```

### 시나리오 2: 완전히 새로 시작
```bash
# 체크포인트 삭제
rm -rf /output/dir/*

# 처음부터 실행
python -u -m descriptor_pipeline.cli.run_pipeline [옵션...]
```

## ⚠️ 주의사항

### 1. 메모리 설정
- RTX 6000 Ada (48GB)를 사용하므로 `max_gpu_memory_gb=40.0`이 적절
- 필요시 config에서 조정 가능

### 2. 배치 크기
- `--batch-rows 50000`은 48GB GPU에 적합
- OOM 발생 시 30000~40000으로 줄이기

### 3. 체크포인트
- `--checkpoint` 옵션 필수 (재시작 기능 사용하려면)
- 체크포인트 파일은 자동으로 output_dir에 저장됨
- 체크포인트 무시하고 새로 시작하려면 output_dir 비우기

### 4. 진행 상황 확인
```bash
# 로그 파일 모니터링
tail -f /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/output.log

# 체크포인트 파일 확인
ls -lh /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/*.json
```

## 🐛 문제 해결

### OOM (Out of Memory) 에러
```bash
# 1. 배치 크기 줄이기
--batch-rows 30000

# 2. GPU 메모리 제한 줄이기 (settings.py에서)
max_gpu_memory_gb: float = 35.0
```

### 체크포인트 오류
```bash
# 손상된 체크포인트 파일 삭제 후 재시작
rm /output/dir/pass*.json
rm /output/dir/pass*.txt
rm /output/dir/pass*.npz
```

### 변수 미지정 에러
```python
# settings.py에서 max_gpu_memory_gb 확인
max_gpu_memory_gb: float = 40.0  # 이 줄이 있어야 함
```

## 📈 예상 효과

1. **메모리 안정성**: OOM 에러 감소, 안정적인 대용량 데이터 처리
2. **시간 절약**: 중단 후 재시작 시 완료된 Pass 스킵 (수 시간 절약)
3. **디스크 사용**: 체크포인트 파일 약 5-10GB 추가 사용
4. **복원력**: 예상치 못한 중단에도 데이터 손실 최소화

## 🔄 업데이트 내역

- **2025-11-04**: 
  - PipelineConfig에 max_gpu_memory_gb 추가
  - 체크포인트 재개 기능 구현
  - GPU 메모리 정리 개선
  - 중간 결과 저장/로드 기능 추가
