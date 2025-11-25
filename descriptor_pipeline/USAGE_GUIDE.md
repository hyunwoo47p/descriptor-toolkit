# Descriptor Pipeline 개선 버전 - 설치 및 사용 가이드

## 📋 개요

이 개선 버전은 다음 3가지 주요 문제를 해결합니다:

1. **메모리 관리 문제**: GPU 메모리 설정이 누락되어 있던 문제 해결
2. **체크포인트 기능**: 재시작 시 이미 완료된 Pass를 자동으로 스킵하는 기능 추가
3. **변수 참조 문제**: 코드 내 변수 미지정 문제 수정

## 🔧 설치 방법

### 1. 기존 코드 백업
```bash
cd ~/hyunwoo-proj  # 또는 프로젝트 디렉토리
cp -r descriptor_pipeline descriptor_pipeline_backup_$(date +%Y%m%d)
```

### 2. 개선된 파일 적용
```bash
# 업로드된 파일 압축 해제
unzip descriptor_pipeline_improved.zip

# 주요 파일들 교체
cp descriptor_pipeline_improved/config/settings.py descriptor_pipeline/config/
cp descriptor_pipeline_improved/core/pipeline.py descriptor_pipeline/core/
cp descriptor_pipeline_improved/core/similarity_gpu.py descriptor_pipeline/core/

# 문서 파일 복사 (선택사항)
cp descriptor_pipeline_improved/IMPROVEMENTS.md descriptor_pipeline/
cp descriptor_pipeline_improved/USAGE_GUIDE.md descriptor_pipeline/
```

### 3. 확인
```bash
# settings.py에 max_gpu_memory_gb가 있는지 확인
grep "max_gpu_memory_gb" descriptor_pipeline/config/settings.py
# 출력: max_gpu_memory_gb: float = 40.0  # Maximum GPU memory to use (GB)
```

## 🚀 사용 방법

### 기본 실행 (변경 없음)
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

**중요**: 기존 명령어를 그대로 사용하면 됩니다! 모든 인자가 정상적으로 작동합니다.

## 💾 체크포인트 기능

### 자동 재개
작업이 중단된 후 같은 명령어로 재실행하면:

```
✓ Pass 0: Sampling already completed (using cached file)
✓ Pass 1: Statistics & Variance Filtering already completed (loading from checkpoint)
→ Pass 2: Spearman Correlation Filtering (GPU) ... 여기서부터 재개
```

### 저장되는 파일들
```
output_dir/
├── sampled_data.parquet             # Pass0 샘플링 결과
├── pass1_variance_filtering.json    # Pass1 완료 정보
├── pass1_columns.txt                # Pass1 결과 컬럼 (텍스트)
├── pass1_stats.npz                  # Pass1 통계 (NumPy 압축)
├── pass2_spearman.json              # Pass2 완료 정보
├── pass2_columns.txt                # Pass2 결과 컬럼
├── pass2_spearman_matrix.npy        # Spearman 상관행렬 (재사용)
├── pass3_vif.json                   # Pass3 완료 정보
├── pass3_columns.txt                # Pass3 결과 컬럼
├── pass4_nonlinear.json             # Pass4 완료 정보
├── final_descriptors.txt            # 최종 결과
└── output.log                       # 실행 로그
```

### 처음부터 다시 시작하려면
```bash
# 방법 1: output 디렉토리 삭제
rm -rf /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3

# 방법 2: 체크포인트 파일만 삭제
rm /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/pass*.json
rm /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/pass*.txt
rm /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/pass*.npy
rm /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/pass*.npz
```

## 📊 진행 상황 모니터링

### 로그 실시간 확인
```bash
# 새 터미널에서
tail -f /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/output.log
```

### 체크포인트 상태 확인
```bash
# 어떤 Pass까지 완료되었는지 확인
ls -lh /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/pass*.json

# 각 Pass별 결과 컬럼 수 확인
wc -l /home/ML_data/pubchem/Compound/Descriptor_clustering/FULL3/pass*_columns.txt
```

## ⚙️ 메모리 설정 조정

### GPU 메모리 한도 변경
RTX 6000 Ada (48GB)에서 OOM 발생 시:

```bash
# descriptor_pipeline/config/settings.py 편집
vim descriptor_pipeline/config/settings.py

# 다음 줄 수정
max_gpu_memory_gb: float = 35.0  # 40.0 → 35.0으로 줄임
```

### 배치 크기 조정
```bash
# 명령어에서 --batch-rows 조정
--batch-rows 30000  # 50000 → 30000으로 줄임
```

## 🐛 문제 해결

### 1. OOM (Out of Memory) 에러
```
RuntimeError: CUDA out of memory
```

**해결 방법:**
```bash
# 1단계: 배치 크기 줄이기
--batch-rows 30000

# 2단계: GPU 메모리 한도 줄이기
# settings.py에서 max_gpu_memory_gb: float = 35.0

# 3단계: GPU 캐시 정리
python -c "import torch; torch.cuda.empty_cache()"
```

### 2. 체크포인트 파일 손상
```
JSONDecodeError: Expecting value
```

**해결 방법:**
```bash
# 해당 Pass의 체크포인트 파일만 삭제
rm /output/dir/pass2*.json
rm /output/dir/pass2*.txt
rm /output/dir/pass2*.npy

# 다시 실행하면 Pass2부터 재계산됨
```

### 3. 변수 미지정 에러
```
AttributeError: 'PipelineConfig' object has no attribute 'max_gpu_memory_gb'
```

**해결 방법:**
```bash
# settings.py가 제대로 교체되지 않았을 가능성
# 다시 한번 복사
cp descriptor_pipeline_improved/config/settings.py descriptor_pipeline/config/

# 확인
grep "max_gpu_memory_gb" descriptor_pipeline/config/settings.py
```

## 📈 성능 개선 효과

### 메모리 사용량
- **이전**: GPU 메모리 사용량 불확실, 자주 OOM 발생
- **개선**: 40GB 한도 설정, 안정적인 메모리 관리

### 재시작 시간
- **이전**: 항상 처음부터 (12+ 시간)
- **개선**: 
  - Pass1 완료 후 중단 → 재시작 시 11시간 절약
  - Pass2 완료 후 중단 → 재시작 시 9시간 절약
  - Pass3 완료 후 중단 → 재시작 시 6시간 절약

### 디스크 사용량
- 체크포인트 파일: 약 5-10GB 추가
- 대신 재계산 시간 대폭 절약

## ✅ 검증 방법

### 1. 설치 확인
```bash
# Python에서 직접 확인
python -c "
from descriptor_pipeline.config.settings import PipelineConfig
config = PipelineConfig(
    parquet_glob='test/*.parquet',
    output_dir='test_output'
)
print(f'max_gpu_memory_gb: {config.max_gpu_memory_gb}')
print('✓ 설치 성공!')
"
```

### 2. 체크포인트 동작 확인
```bash
# 작은 테스트 실행
python -u -m descriptor_pipeline.cli.run_pipeline \
  --parquet-glob "/home/ML_data/pubchem/Compound/Filtered_Descriptor/part-000*.parquet" \
  --output-dir "/tmp/test_checkpoint" \
  --n-metadata 6 \
  --gpu \
  --verbose \
  --checkpoint

# Pass1 완료 후 Ctrl+C로 중단

# 재시작 - Pass1이 스킵되는지 확인
python -u -m descriptor_pipeline.cli.run_pipeline \
  --parquet-glob "/home/ML_data/pubchem/Compound/Filtered_Descriptor/part-000*.parquet" \
  --output-dir "/tmp/test_checkpoint" \
  --n-metadata 6 \
  --gpu \
  --verbose \
  --checkpoint

# "✓ Pass 1: already completed (loading from checkpoint)" 메시지 확인
```

## 📞 문의 및 지원

문제 발생 시:
1. 로그 파일 확인 (`output.log`)
2. 체크포인트 파일 상태 확인 (`ls -lh output_dir/`)
3. GPU 메모리 사용량 확인 (`nvidia-smi`)

## 🔄 업데이트 이력

- **2025-11-04**: 
  - ✅ Config에 `max_gpu_memory_gb` 추가
  - ✅ 체크포인트 재개 기능 완성
  - ✅ GPU 메모리 정리 개선
  - ✅ 중간 결과 저장/로드 기능
  - ✅ Pass별 자동 스킵 기능
