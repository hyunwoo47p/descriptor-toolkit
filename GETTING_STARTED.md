# Getting Started - Molecular Descriptor Toolkit v1.0

**빠르게 시작하는 가이드**

---

## 📦 설치

```bash
# 1. 압축 해제
tar -xzf molecular_descriptor_toolkit_v1.0.tar.gz
cd molecular_descriptor_toolkit

# 2. Python 경로 설정
export PYTHONPATH=$(pwd):$PYTHONPATH

# 3. 의존성 설치 (필요시)
pip install -r requirements.txt
```

---

## ✅ 검증

```bash
# 설치 확인
python -c "from molecular_descriptor_toolkit.cli import main; print('✓ Installation OK')"

# GPU 확인
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 🚀 실행 (A to Z)

### 방법 1: 자동 스크립트 (권장)

```bash
./test_pipeline.sh
```

이 스크립트는 다음을 자동으로 실행합니다:
1. Preprocessing: XML → Descriptors
2. Filtering: Pass 0-4
3. 결과 검증

**예상 시간**: 40-90분

### 방법 2: 수동 실행

#### Step 1: Preprocessing (XML → Descriptors)

```bash
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/preprocessing/descriptors.parquet \
    --n-jobs 16 \
    --verbose
```

**예상 시간**: 30-60분  
**출력**: `test_output/preprocessing/descriptors.parquet` (약 500K compounds × 1000 descriptors)

#### Step 2: Filtering (Pass 0-4)

```bash
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering \
    --config config/test_settings.yaml \
    --passes 0,1,2,3,4
```

**예상 시간**: 10-30분  
**출력**: `test_output/filtering/pass*_results/` (각 Pass별 결과)

---

## 📊 결과 확인

```bash
# 파이프라인 요약
cat test_output/filtering/pipeline_summary.json | python -m json.tool

# 최종 descriptor 개수
python -c "
import pyarrow.parquet as pq
table = pq.read_table('test_output/filtering/pass4_results/descriptors.parquet')
print(f'Final descriptors: {len(table.column_names)}')
print(f'Compounds: {table.num_rows}')
"
```

**예상 결과**:
- Input: ~1,000 descriptors
- Pass 0: ~500 descriptors (50% reduction)
- Pass 1: ~100 descriptors (80% reduction)
- Pass 2: ~50 descriptors (50% reduction)
- Pass 3: ~20 descriptors (60% reduction)
- Pass 4: ~10-15 descriptors (40% reduction)

---

## 🎯 핵심 명령어 요약

```bash
# 환경 설정
export PYTHONPATH=$(pwd):$PYTHONPATH

# 전체 파이프라인 (자동)
./test_pipeline.sh

# 또는 수동으로
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/preprocessing/descriptors.parquet \
    --n-jobs 16 --verbose

python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering \
    --config config/test_settings.yaml \
    --passes 0,1,2,3,4
```

---

## 📚 추가 문서

- **[QUICKSTART.md](QUICKSTART.md)**: 완전한 A-Z 가이드
- **[COMMANDS.md](COMMANDS.md)**: 명령어 참조
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)**: 설정 파라미터 가이드
- **[VERSION_SUMMARY.md](VERSION_SUMMARY.md)**: v1.0 요약

---

## 🔧 문제 해결

### GPU Out of Memory
```yaml
# config/test_settings.yaml 수정
filtering:
  correlation_batch_size: 5000    # 줄이기
  nonlinear_batch_size: 2500      # 줄이기
```

### Import Error
```bash
# PYTHONPATH 재설정
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## 📞 지원

문제가 발생하면 다음 문서를 참조하세요:
- [QUICKSTART.md](QUICKSTART.md) - 상세 가이드
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 설정 옵션

---

**Version**: 1.0.0  
**Last Updated**: 2024-11-10
