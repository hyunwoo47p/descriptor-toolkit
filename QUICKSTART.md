# Quick Start Guide - Complete A-Z Testing

이 가이드는 실제 PubChem XML 파일을 사용하여 전체 파이프라인을 처음부터 끝까지 테스트하는 방법을 안내합니다.

---

## Test File Information

**Test File**: `/home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz`

- **Compound Range**: 50,000,001 - 50,500,000
- **Expected Count**: ~500,000 compounds
- **File Size**: ~2-3 GB (compressed)
- **Format**: PubChem XML.gz

---

## Step-by-Step Testing

### Step 1: Environment Setup

```bash
# 1. Navigate to toolkit directory
cd /home/claude/molecular_descriptor_toolkit

# 2. Set Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# 3. Verify Python environment
which python
# Expected: /home/hyunwoo/miniconda3/envs/descriptor/bin/python

# 4. Check Python version
python --version
# Expected: Python 3.11.x
```

### Step 2: Verify Dependencies

```bash
# Check core dependencies
python -c "
import numpy as np
import torch
import pyarrow as pa
import scipy
import igraph
import leidenalg
import statsmodels
print('✓ All core dependencies available')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Expected output:
```
✓ All core dependencies available
PyTorch: 2.5.1+cu124
CUDA available: True
GPU: NVIDIA RTX 6000 Ada Generation
```

### Step 3: Create Configuration File

```bash
# Create test configuration
cat > config/test_settings.yaml << 'YAML'
system:
  verbose: true
  n_jobs: -1

device:
  use_gpu: true
  device_id: 0
  max_gpu_memory_gb: 40.0

filtering:
  # Pass 0: Variance
  variance_threshold: 0.01
  min_variance: 1e-10
  
  # Pass 1: Spearman Clustering
  spearman_threshold: 0.95
  resolution: 1.0
  n_iterations: 5
  
  # Pass 2: Correlation (GPU)
  correlation_threshold: 0.90
  correlation_batch_size: 10000
  
  # Pass 3: VIF
  vif_threshold: 10.0
  n_jobs_vif: 8
  
  # Pass 4: Nonlinearity
  w_hsic: 0.3
  w_rdc: 0.3
  nonlinear_threshold: 0.5
  nonlinear_batch_size: 5000

data:
  batch_size: 100000
  chunk_size: 500000
  use_float32: true

output:
  save_intermediate: true
  checkpoint_frequency: 1
YAML

echo "✓ Configuration file created: config/test_settings.yaml"
```

### Step 4: Test Configuration Validation

```bash
# Validate configuration
python -m molecular_descriptor_toolkit.cli validate-config \
    --config config/test_settings.yaml
```

Expected output:
```
✓ Configuration validated successfully
Using device: cuda:0
GPU Memory: 48.0 GB available
```

### Step 5: Preprocessing (XML → Descriptors)

```bash
# Create output directory
mkdir -p test_output/preprocessing

# Run preprocessing pipeline
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/preprocessing/descriptors.parquet \
    --n-jobs 16 \
    --verbose
```

**Expected Duration**: 30-60 minutes (depends on system)

**Expected Output**:
- `test_output/preprocessing/descriptors.parquet` - Molecular descriptors
- `test_output/preprocessing/metadata.parquet` - Compound metadata
- Log showing ~500K compounds processed

**Key Log Messages**:
```
Parsing XML: 100%|████████████| 500000/500000
Calculating descriptors: 100%|████████████| 500000/500000
Saved descriptors: 500000 compounds × ~1000 descriptors
```

### Step 6: Run Complete Filtering Pipeline

```bash
# Create filtering output directory
mkdir -p test_output/filtering

# Run complete pipeline (Pass 0 → Pass 4)
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering \
    --config config/test_settings.yaml \
    --passes 0,1,2,3,4
```

**Expected Duration**: 10-30 minutes (depends on GPU)

**Expected Outputs**:
```
test_output/filtering/
├── pass0_results/
│   ├── descriptors.parquet       # After variance filtering
│   └── metadata.json
├── pass1_results/
│   ├── descriptors.parquet       # After Spearman clustering
│   ├── clusters.parquet
│   └── metadata.json
├── pass2_results/
│   ├── descriptors.parquet       # After correlation (GPU)
│   ├── correlation_matrix.npy
│   └── metadata.json
├── pass3_results/
│   ├── descriptors.parquet       # After VIF
│   ├── vif_scores.parquet
│   └── metadata.json
├── pass4_results/
│   ├── descriptors.parquet       # Final filtered set
│   ├── scores.parquet
│   └── metadata.json
└── pipeline_summary.json
```

### Step 7: Verify Results

```bash
# Check pipeline summary
python -c "
import json
with open('test_output/filtering/pipeline_summary.json', 'r') as f:
    summary = json.load(f)

print('Pipeline Summary')
print('=' * 50)
for pass_name, stats in summary.items():
    if isinstance(stats, dict) and 'n_descriptors_out' in stats:
        print(f'{pass_name}: {stats[\"n_descriptors_out\"]} descriptors')
        print(f'  Reduction: {stats.get(\"reduction_pct\", 0):.1f}%')
        print(f'  Duration: {stats.get(\"duration_sec\", 0):.1f}s')
        print()
"
```

**Expected Output**:
```
Pipeline Summary
==================================================
pass0: ~50000 descriptors
  Reduction: 50.0%
  Duration: 30.0s

pass1: ~10000 descriptors
  Reduction: 80.0%
  Duration: 120.0s

pass2: ~5000 descriptors
  Reduction: 50.0%
  Duration: 45.0s

pass3: ~2000 descriptors
  Reduction: 60.0%
  Duration: 180.0s

pass4: ~1000 descriptors
  Reduction: 50.0%
  Duration: 300.0s
```

### Step 8: Inspect Final Results

```bash
# Load and inspect final descriptors
python -c "
import pyarrow.parquet as pq

# Read final descriptors
table = pq.read_table('test_output/filtering/pass4_results/descriptors.parquet')
df = table.to_pandas()

print('Final Descriptor Set')
print('=' * 50)
print(f'Compounds: {len(df):,}')
print(f'Descriptors: {len(df.columns):,}')
print()
print('Sample descriptors:')
print(df.columns[:10].tolist())
print()
print('Data shape:', df.shape)
print('Memory usage:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')
"
```

---

## Advanced Testing Options

### Test Individual Passes

```bash
# Test only Pass 0 (Variance)
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/pass0_only \
    --config config/test_settings.yaml \
    --passes 0

# Test Pass 1-3 (skip Pass 4)
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/pass123 \
    --config config/test_settings.yaml \
    --passes 0,1,2,3
```

### Test with Different Configurations

```bash
# More aggressive filtering
cat > config/aggressive_settings.yaml << 'YAML'
filtering:
  variance_threshold: 0.05        # Higher variance requirement
  spearman_threshold: 0.90        # Lower correlation threshold
  correlation_threshold: 0.85
  vif_threshold: 5.0              # Stricter VIF
  nonlinear_threshold: 0.7        # More selective
YAML

python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/aggressive \
    --config config/aggressive_settings.yaml
```

### Test with CPU Only

```bash
# Disable GPU
cat > config/cpu_settings.yaml << 'YAML'
device:
  use_gpu: false
  
filtering:
  # Use smaller batches for CPU
  correlation_batch_size: 1000
  nonlinear_batch_size: 500
YAML

python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/cpu_test \
    --config config/cpu_settings.yaml
```

---

## Performance Monitoring

### Monitor GPU Usage

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Monitor Memory Usage

```bash
# During pipeline execution
while true; do
    ps aux | grep python | grep molecular_descriptor
    sleep 5
done
```

### Profile Execution Time

```bash
# Use time command
time python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/profiled \
    --config config/test_settings.yaml
```

---

## Troubleshooting

### Issue 1: GPU Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```yaml
# Reduce batch sizes in config
filtering:
  correlation_batch_size: 5000   # Reduce from 10000
  nonlinear_batch_size: 2500     # Reduce from 5000
```

### Issue 2: Import Errors

**Symptom**:
```
ModuleNotFoundError: No module named 'molecular_descriptor_toolkit'
```

**Solution**:
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/home/claude/molecular_descriptor_toolkit:$PYTHONPATH
```

### Issue 3: Preprocessing Hangs

**Symptom**: Preprocessing stuck at certain compound

**Solution**:
```bash
# Use timeout and error handling
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/preprocessing/descriptors.parquet \
    --skip-errors \
    --timeout 30
```

### Issue 4: Numerical Warnings

**Symptom**:
```
RuntimeWarning: invalid value encountered in divide
```

**Solution**: These warnings are usually harmless. To suppress:
```bash
# Add to Python command
python -W ignore -m molecular_descriptor_toolkit.cli run ...
```

---

## Complete Test Script

Create `run_complete_test.sh`:

```bash
#!/bin/bash
set -e

echo "Starting Complete Pipeline Test"
echo "================================"

# Setup
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p test_output/{preprocessing,filtering}

# Step 1: Preprocessing
echo "[1/3] Running preprocessing..."
python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input /home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz \
    --output test_output/preprocessing/descriptors.parquet \
    --n-jobs 16 \
    --verbose

# Step 2: Filtering pipeline
echo "[2/3] Running filtering pipeline..."
python -m molecular_descriptor_toolkit.cli run \
    --input test_output/preprocessing/descriptors.parquet \
    --output test_output/filtering \
    --config config/test_settings.yaml \
    --passes 0,1,2,3,4

# Step 3: Verify results
echo "[3/3] Verifying results..."
python -c "
import json
import pyarrow.parquet as pq

# Check summary
with open('test_output/filtering/pipeline_summary.json', 'r') as f:
    summary = json.load(f)
    
# Check final output
table = pq.read_table('test_output/filtering/pass4_results/descriptors.parquet')

print('\n✓ Test completed successfully!')
print(f'Final descriptors: {len(table.column_names)}')
print(f'Compounds: {table.num_rows}')
"

echo "================================"
echo "Test completed! Check test_output/ directory"
```

Run it:
```bash
chmod +x run_complete_test.sh
./run_complete_test.sh
```

---

## Expected Timeline

| Stage | Duration | Memory | GPU Usage |
|-------|----------|--------|-----------|
| Preprocessing | 30-60 min | 8-16 GB | 0% |
| Pass 0 | 30-60 sec | 4-8 GB | 0% |
| Pass 1 | 2-5 min | 8-16 GB | 0% |
| Pass 2 | 1-3 min | 20-30 GB | 80-90% |
| Pass 3 | 3-8 min | 8-16 GB | 0% |
| Pass 4 | 5-15 min | 20-40 GB | 80-90% |
| **Total** | **40-90 min** | **40 GB peak** | **GPU: ~50%** |

---

## Success Criteria

✅ **Preprocessing Complete**:
- `descriptors.parquet` file created
- ~500K compounds processed
- ~1000 descriptors calculated

✅ **Filtering Complete**:
- All 5 passes completed
- Final descriptor count: 500-2000
- No errors or crashes

✅ **Results Valid**:
- Pipeline summary JSON exists
- All parquet files readable
- Descriptor reduction follows expected pattern

---

**Last Updated**: 2024-11-10  
**Version**: 1.0.0
