#!/bin/bash
# GPU Mode Quick Test Script

echo "=== GPU Mode Descriptor Pipeline Test ==="
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "❌ Error: 'data' directory not found"
    echo "   Please create a 'data' directory with parquet files"
    exit 1
fi

# Check for parquet files
parquet_count=$(find data -name "*.parquet" | wc -l)
if [ $parquet_count -eq 0 ]; then
    echo "❌ Error: No parquet files found in 'data' directory"
    exit 1
fi

echo "✅ Found $parquet_count parquet file(s)"
echo ""

# Test 1: Basic GPU run
echo "Test 1: Basic GPU run (sampled)"
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/*.parquet" \
    --output-dir ./test_basic \
    --gpu \
    --sample-per-file 1000 \
    --verbose

if [ $? -eq 0 ]; then
    echo "✅ Test 1 passed"
else
    echo "❌ Test 1 failed"
    exit 1
fi

echo ""
echo "Test 2: GPU with custom thresholds"
python -m descriptor_pipeline.cli.run_pipeline \
    --parquet-glob "data/*.parquet" \
    --output-dir ./test_custom \
    --gpu \
    --sample-per-file 1000 \
    --variance-threshold 0.001 \
    --spearman-threshold 0.98 \
    --vif-threshold 5.0 \
    --nonlinear-threshold 0.2 \
    --verbose

if [ $? -eq 0 ]; then
    echo "✅ Test 2 passed"
else
    echo "❌ Test 2 failed"
    exit 1
fi

echo ""
echo "=== All tests passed! ==="
echo ""
echo "Results saved to:"
echo "  - test_basic/"
echo "  - test_custom/"
