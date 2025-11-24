#!/bin/bash
# Complete Pipeline Test Script for v1.0
# Tests PubChem file: Compound_050000001_050500000.xml.gz

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Molecular Descriptor Toolkit v1.0${NC}"
echo -e "${GREEN}Complete Pipeline Test${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# Configuration
INPUT_FILE="/home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz"
OUTPUT_DIR="test_output"
CONFIG_FILE="config/test_settings.yaml"

# Step 0: Setup
echo -e "${YELLOW}[0/3] Setting up environment...${NC}"
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p ${OUTPUT_DIR}/{preprocessing,filtering}

# Verify input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Verify Python environment
echo "Python: $(which python)"
python --version

# Check GPU
echo ""
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo ""

# Step 1: Preprocessing
echo -e "${YELLOW}[1/3] Running preprocessing (XML → Descriptors)...${NC}"
echo "This may take 30-60 minutes..."
echo ""

time python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/preprocessing/descriptors.parquet" \
    --n-jobs 16 \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Preprocessing completed${NC}"
    
    # Show preprocessing results
    python -c "
import pyarrow.parquet as pq
table = pq.read_table('${OUTPUT_DIR}/preprocessing/descriptors.parquet')
print(f'\nPreprocessing Results:')
print(f'  Compounds: {table.num_rows:,}')
print(f'  Descriptors: {len(table.column_names):,}')
print(f'  File size: {pq.ParquetFile(\"${OUTPUT_DIR}/preprocessing/descriptors.parquet\").metadata.serialized_size / 1024**2:.1f} MB')
"
else
    echo -e "${RED}✗ Preprocessing failed${NC}"
    exit 1
fi
echo ""

# Step 2: Filtering Pipeline
echo -e "${YELLOW}[2/3] Running filtering pipeline (Pass 0-4)...${NC}"
echo "This may take 10-30 minutes..."
echo ""

time python -m molecular_descriptor_toolkit.cli run \
    --input "${OUTPUT_DIR}/preprocessing/descriptors.parquet" \
    --output "${OUTPUT_DIR}/filtering" \
    --config "$CONFIG_FILE" \
    --passes 0,1,2,3,4

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Filtering pipeline completed${NC}"
else
    echo -e "${RED}✗ Filtering pipeline failed${NC}"
    exit 1
fi
echo ""

# Step 3: Verify Results
echo -e "${YELLOW}[3/3] Verifying results...${NC}"

python -c "
import json
import pyarrow.parquet as pq

print('Pipeline Summary')
print('=' * 60)

# Load summary
try:
    with open('${OUTPUT_DIR}/filtering/pipeline_summary.json', 'r') as f:
        summary = json.load(f)
    
    for pass_name in ['pass0', 'pass1', 'pass2', 'pass3', 'pass4']:
        if pass_name in summary and isinstance(summary[pass_name], dict):
            stats = summary[pass_name]
            n_desc = stats.get('n_descriptors_out', 'N/A')
            reduction = stats.get('reduction_pct', 0)
            duration = stats.get('duration_sec', 0)
            print(f'{pass_name.upper()}: {n_desc:>6} descriptors | Reduction: {reduction:>5.1f}% | Time: {duration:>6.1f}s')
except FileNotFoundError:
    print('Warning: pipeline_summary.json not found')

print('=' * 60)

# Check final output
try:
    table = pq.read_table('${OUTPUT_DIR}/filtering/pass4_results/descriptors.parquet')
    print(f'\nFinal Results:')
    print(f'  Compounds: {table.num_rows:,}')
    print(f'  Descriptors: {len(table.column_names):,}')
    print(f'  Memory: {table.nbytes / 1024**2:.1f} MB')
    
    print(f'\nSample descriptors:')
    print(f'  {table.column_names[:5]}')
    print(f'  ... ({len(table.column_names) - 10} more)')
    print(f'  {table.column_names[-5:]}')
except Exception as e:
    print(f'Error reading final results: {e}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}✓ Test completed successfully!${NC}"
    echo -e "${GREEN}=====================================${NC}"
    echo ""
    echo "Results saved in: ${OUTPUT_DIR}/"
    echo ""
    echo "Output structure:"
    ls -lh ${OUTPUT_DIR}/preprocessing/ 2>/dev/null || true
    ls -lh ${OUTPUT_DIR}/filtering/pass*/ 2>/dev/null || true
else
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi
