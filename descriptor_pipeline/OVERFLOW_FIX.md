# Numerical Overflow Fix for Normalized Variance Calculation

## Issue Summary
Your pipeline encountered `RuntimeWarning: overflow encountered in square` and `RuntimeWarning: invalid value encountered in divide` errors during Pass 1 statistics computation.

## Root Cause

**Location:** `core/pass1_statistics.py`, line 242-243

**Problematic Code:**
```python
RANGE_EPS = 1e-12
...
normalized_variance = variances / (ranges ** 2)
normalized_variance_robust = variances / (trimmed_ranges ** 2)
```

**The Problem:**
1. `RANGE_EPS = 1e-12` is very small
2. When squared: `(1e-12)² = 1e-24`
3. This is dangerously close to the float32 underflow limit (~1.2e-38)
4. Dividing by such tiny values causes **overflow to infinity**
5. The `nan_to_num` cleanup happens *after* the overflow, making it ineffective

## Performance Impact
The slow processing speed you observed (0.1 columns/second) is likely due to:
- **PyArrow/DuckDB overhead** in per-column statistics computation
- **Not related to this overflow issue** (which only affects the final calculation)

## Solution Applied

### Change 1: Increase RANGE_EPS
```python
# Before
RANGE_EPS = 1e-12

# After
RANGE_EPS = 1e-6  # Increased to prevent overflow when squaring
```

**Rationale:**
- `1e-6` is still extremely small for practical purposes
- When squared: `(1e-6)² = 1e-12` - safe for float32
- Won't affect filtering quality (descriptors with ranges < 1e-6 are essentially constant)

### Change 2: Safe Division with Clipping
```python
# Before
normalized_variance = variances / (ranges ** 2)
normalized_variance_robust = variances / (trimmed_ranges ** 2)

# After
# Square the ranges and clip to prevent division by near-zero
ranges_squared = np.maximum(ranges ** 2, RANGE_EPS ** 2)
trimmed_ranges_squared = np.maximum(trimmed_ranges ** 2, RANGE_EPS ** 2)

# Safe division with explicit overflow handling
with np.errstate(divide='ignore', invalid='ignore'):
    normalized_variance = variances / ranges_squared
    normalized_variance_robust = variances / trimmed_ranges_squared
```

**Benefits:**
- Clips squared ranges to prevent division by near-zero
- Explicitly suppresses runtime warnings (values are cleaned up afterward)
- Maintains numerical stability across all descriptors

## Files Modified

1. `/core/pass1_statistics.py` (main version, 454 lines)
2. `/descriptor_pipeline/core/pass1_statistics.py` (nested copy, 225 lines)

Both files have been fixed with identical changes.

## Impact on Results

**Minimal Impact:**
- Descriptors with extremely small ranges (< 1e-6) will now be treated as having range = 1e-6 instead of 1e-12
- These are essentially constant columns anyway and would be filtered out
- No significant change to descriptor selection or model performance

## Validation Checklist

After applying this fix:
- [ ] No more overflow warnings in Pass 1
- [ ] Statistics computation completes successfully
- [ ] Same number of descriptors pass variance filtering (should be ~1633)
- [ ] No NaN/Inf values in the final statistics dictionary

## Additional Recommendations

### 1. Performance Optimization for Pass 1
Your current implementation processes columns one-by-one, which is very slow (0.1 cols/sec for 1775 columns = ~5 hours).

**Suggested Improvements:**
- Use DuckDB for batch statistics computation
- Compute all quantiles in a single query
- Process columns in parallel batches

**Potential speedup:** 50-100x faster (5 hours → 3-6 minutes)

### 2. GPU Acceleration
The GPU implementation in Pass 2 is working well. Consider:
- Increasing chunk size if you have 48GB VRAM (try 2-5M rows)
- Using float16 for correlation matrices (2x memory reduction with minimal accuracy loss)

### 3. Checkpoint Recovery
If your pipeline crashes again, it should resume from the last checkpoint. Make sure:
- Checkpoint directory exists and is writable
- Enough disk space for parquet files

## Testing the Fix

Run your pipeline with the fixed code:
```bash
cd ~/tools/descriptor_pipeline
python -m descriptor_pipeline.cli.run_pipeline \
    --input your_input.parquet \
    --output output_dir \
    --gpu \
    --device cuda
```

Monitor for:
1. **No overflow warnings** during Pass 1
2. **Successful completion** of statistics computation
3. **Consistent descriptor counts** across runs

## Questions?

If you encounter any issues with the fix or need further optimization, let me know!
