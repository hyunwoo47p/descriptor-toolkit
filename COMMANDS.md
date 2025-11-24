# Quick Command Reference

실제로 사용할 명령어들만 간단하게 정리한 문서입니다.

---

## 1. 환경 설정

```bash
# 디렉토리 이동
cd /home/claude/molecular_descriptor_toolkit

# Python 경로 설정
export PYTHONPATH=$(pwd):$PYTHONPATH

# 환경 확인
python -c "from molecular_descriptor_toolkit.cli import main; print('✓ OK')"
```

---

## 2. 전체 파이프라인 실행 (A to Z)

### 방법 1: 자동 스크립트 사용 (권장)

```bash
# 전체 파이프라인 실행 (preprocessing + filtering)
./test_pipeline.sh
```

### 방법 2: 수동 명령어

```bash
# Step 1: Preprocessing (XML → Descriptors)
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/preprocessing/descriptors.parquet \
    --n-jobs 16 \
    --verbose

# Step 2: Filtering (Pass 0-4)
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering \
    --config config/test_settings.yaml \
    --passes 0,1,2,3,4
```

---

## 3. 빠른 테스트 (Preprocessing만)

```bash
# 빠른 검증용
./quick_test.sh
```

또는:

```bash
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/quick/descriptors.parquet \
    --n-jobs 16
```

---

## 4. 개별 Pass 실행

```bash
# Pass 0만 실행
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/pass0_only \
    --config config/test_settings.yaml \
    --passes 0

# Pass 0-2만 실행 (GPU 테스트)
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/pass012 \
    --config config/test_settings.yaml \
    --passes 0,1,2
```

---

## 5. 설정 커스터마이징

```bash
# 설정 파일 편집
vi config/test_settings.yaml

# 주요 파라미터:
# - variance_threshold: 0.01 (낮을수록 더 많이 유지)
# - spearman_threshold: 0.95 (낮을수록 더 적극적 필터링)
# - correlation_threshold: 0.90
# - vif_threshold: 10.0 (낮을수록 더 적극적 필터링)
# - w_hsic: 0.3 (비선형 가중치)
# - w_rdc: 0.3
```

---

## 6. GPU 메모리 문제 해결

설정 파일에서 배치 크기 조정:

```yaml
filtering:
  correlation_batch_size: 5000    # 기본값 10000에서 감소
  nonlinear_batch_size: 2500      # 기본값 5000에서 감소
```

---

## 7. 결과 확인

```bash
# 파이프라인 요약 보기
cat test_output/filtering/pipeline_summary.json | python -m json.tool

# 최종 descriptor 개수 확인
python -c "
import pyarrow.parquet as pq
table = pq.read_table('test_output/filtering/pass4_results/descriptors.parquet')
print(f'Final descriptors: {len(table.column_names)}')
print(f'Compounds: {table.num_rows}')
"

# 모든 Pass 결과 비교
for pass in pass0 pass1 pass2 pass3 pass4; do
    python -c "
import pyarrow.parquet as pq
table = pq.read_table('test_output/filtering/${pass}_results/descriptors.parquet')
print(f'${pass}: {len(table.column_names):>4} descriptors')
" 2>/dev/null || echo "${pass}: (not found)"
done
```

---

## 8. GPU 모니터링

```bash
# 별도 터미널에서 실행
watch -n 1 nvidia-smi

# 또는 로그 저장
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv -l 1 > gpu_log.csv &
```

---

## 9. 성능 측정

```bash
# 시간 측정
time ./test_pipeline.sh

# 메모리 사용량 모니터링
/usr/bin/time -v python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering \
    --config config/test_settings.yaml
```

---

## 10. 정리

```bash
# 테스트 결과 삭제
rm -rf test_output/

# 특정 Pass 결과만 삭제
rm -rf test_output/filtering/pass*_results/
```

---

## 예상 실행 시간

| 단계 | 예상 시간 | 메모리 |
|-----|---------|--------|
| Preprocessing | 30-60분 | 8-16 GB |
| Pass 0 | 30-60초 | 4-8 GB |
| Pass 1 | 2-5분 | 8-16 GB |
| Pass 2 | 1-3분 | 20-30 GB |
| Pass 3 | 3-8분 | 8-16 GB |
| Pass 4 | 5-15분 | 20-40 GB |
| **Total** | **40-90분** | **40 GB peak** |

---

## 일반적인 Workflow

```bash
# 1. 첫 실행 (전체 테스트)
./test_pipeline.sh

# 2. 설정 조정
vi config/test_settings.yaml

# 3. Filtering만 다시 실행 (preprocessing 재사용)
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering_v2 \
    --config config/test_settings.yaml

# 4. 결과 비교
python compare_results.py test_output/filtering test_output/filtering_v2
```

---

**Last Updated**: 2024-11-10  
**Version**: 1.0.0
