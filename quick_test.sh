#!/bin/bash
# Quick Test - Preprocessing Only
# For fast validation of the toolkit

set -e

echo "======================================"
echo "Quick Test - Preprocessing Only"
echo "======================================"
echo ""

# Setup
export PYTHONPATH=$(pwd):$PYTHONPATH

# Input file
INPUT_FILE="/home/ML_data/pubchem/Compound/XML/Compound_050000001_050500000.xml.gz"
OUTPUT_DIR="test_output/quick_test"

mkdir -p "$OUTPUT_DIR"

# Verify
echo "Checking environment..."
python -c "
from molecular_descriptor_toolkit.cli import main
print('✓ Toolkit imported successfully')
"
echo ""

# Run preprocessing
echo "Running preprocessing..."
time python -m molecular_descriptor_toolkit.preprocessing.pipeline \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/descriptors.parquet" \
    --n-jobs 16 \
    --verbose

# Verify results
echo ""
echo "Results:"
python -c "
import pyarrow.parquet as pq
table = pq.read_table('${OUTPUT_DIR}/descriptors.parquet')
print(f'Compounds: {table.num_rows:,}')
print(f'Descriptors: {len(table.column_names):,}')
"

echo ""
echo "✓ Quick test completed!"
echo "Results: ${OUTPUT_DIR}/descriptors.parquet"
