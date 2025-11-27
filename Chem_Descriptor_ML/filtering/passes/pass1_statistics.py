"""
ChemDescriptorML (CDML) - Track 1, Pass 1: Statistics + Variance Filtering

GPU-accelerated statistics computation and low-variance filtering:
1. Global statistics computation (mean, std, quantiles)
2. Low variance filtering (excluding metadata columns)
3. Binary column detection
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.special import ndtri
import torch

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

from Chem_Descriptor_ML.config.settings import FilteringConfig, IOConfig, SystemConfig
from Chem_Descriptor_ML.filtering.utils.logging import log
from Chem_Descriptor_ML.filtering.io.parquet_reader import iter_batches


# Numerical constants
VAR_EPS = 1e-12
RANGE_EPS = 1e-6  # Increased from 1e-12 to prevent overflow when squaring


class StatisticsAndVarianceFilter:
    """
    Pass1: 통계 계산 및 저분산 필터링
    
    GPU를 활용한 초고속 통계 계산 후 normalized variance 기반 필터링 수행.
    메타데이터 컬럼(처음 n개)은 필터링에서 제외되나 통계는 계산됨.
    """
    
    def __init__(
        self,
        filtering_cfg: FilteringConfig,
        io_cfg: IOConfig,
        system_cfg: SystemConfig,
        device: torch.device
    ):
        self.filtering_cfg = filtering_cfg
        self.io_cfg = io_cfg
        self.system_cfg = system_cfg
        self.device = device
        self.verbose = system_cfg.verbose
        self.using_gpu = (device.type == 'cuda')
    
    def _log(self, msg: str):
        log(msg, self.verbose)
    
    def compute(self, parquet_paths: List[str], columns: List[str]) -> Tuple[List[str], Dict, np.ndarray]:
        """
        통계 계산 및 저분산 필터링 수행
        
        Args:
            parquet_paths: Parquet 파일 경로 리스트
            columns: 컬럼 이름 리스트 (이미 메타데이터 제외됨)
            
        Returns:
            (columns_kept, stats, indices_kept)
            - columns_kept: 필터링 후 남은 컬럼 이름
            - stats: 통계 정보 딕셔너리
            - indices_kept: 남은 컬럼의 원본 인덱스
        """
        self._log("\n" + "="*70)
        self._log("Pass1: Statistics + Variance + Missing Data Filtering")
        self._log("="*70)
        self._log(f"  Descriptor columns: {len(columns)} (metadata already excluded)")
        self._log(f"  Variance threshold: {self.filtering_cfg.variance_threshold}")
        self._log(f"  Max missing ratio: {self.filtering_cfg.max_missing_ratio}")
        self._log(f"  Min effective N: {self.filtering_cfg.min_effective_n}")
        self._log(f"  Using {'GPU' if self.using_gpu else 'CPU'}")
        
        # Check for cached statistics
        cache_file = Path(self.io_cfg.output_dir) / "pass1_statistics_cache.npz"
        stats_from_cache = False
        
        if cache_file.exists() and not self.filtering_cfg.force_recompute:
            try:
                self._log(f"\n  Found cached statistics: {cache_file.name}")
                stats, cached_columns = self._load_statistics_cache(cache_file)
                
                # Verify cache is valid (same columns)
                if cached_columns == columns:
                    self._log(f"  ✓ Cache is valid, loading statistics...")
                    stats_from_cache = True
                else:
                    self._log(f"  ⚠️ Cache columns mismatch, recomputing...")
            except Exception as e:
                self._log(f"  ⚠️ Failed to load cache: {str(e)}")
        
        t_start = time.time()
        
        if not stats_from_cache:
            # Step 1: Compute global statistics (PyArrow + GPU)
            self._log("\n  [1/3] Computing global statistics...")
            stats = self._compute_statistics_pyarrow(parquet_paths, columns)
            t_stats = time.time() - t_start
            self._log(f"        ✓ Statistics computed in {t_stats:.1f}s")
            
            # Save cache
            if self.system_cfg.checkpoint:
                self._save_statistics_cache(cache_file, stats, columns)
                self._log(f"        ✓ Statistics cached to {cache_file.name}")
        else:
            t_stats = time.time() - t_start
            self._log(f"        ✓ Statistics loaded from cache in {t_stats:.1f}s")
        
        # Step 2: Variance filtering
        self._log("\n  [2/3] Filtering by variance, missing ratio, and effective N...")
        columns_kept, indices_kept = self._filter_low_variance(columns, stats)
        t_filter = time.time() - t_start - t_stats
        self._log(f"        ✓ Filtering completed in {t_filter:.1f}s")
        
        # Step 3: Update stats for kept columns
        self._log("\n  [3/3] Updating statistics for kept columns...")
        stats_kept = self._filter_stats(stats, indices_kept)
        
        n_removed = len(columns) - len(columns_kept)
        self._log(f"\n✅ Pass1 completed in {time.time() - t_start:.1f}s")
        self._log(f"  Initial descriptors: {len(columns)}")
        self._log(f"  Removed (variance/missing/effective N): {n_removed}")
        self._log(f"  Remaining: {len(columns_kept)}")
        
        return columns_kept, stats_kept, indices_kept
    
    def _compute_statistics_pyarrow(self, parquet_paths: List[str], columns: List[str]) -> Dict:
        """PyArrow 기반 초고속 통계 계산"""
        import time
        stat_start_time = time.time()
        
        p = len(columns)
        dset = ds.dataset(parquet_paths, format="parquet")
        
        # 결과 배열
        clip_lower = np.zeros(p, dtype=np.float64)
        clip_upper = np.zeros(p, dtype=np.float64)
        min_vals = np.zeros(p, dtype=np.float64)
        max_vals = np.zeros(p, dtype=np.float64)
        means = np.zeros(p, dtype=np.float64)
        stds = np.zeros(p, dtype=np.float64)
        variances = np.zeros(p, dtype=np.float64)
        
        # Trimmed range for normalized variance
        trimmed_ranges = np.zeros(p, dtype=np.float64)
        
        # Missing data info
        n_missing = np.zeros(p, dtype=np.int64)
        n_valid = np.zeros(p, dtype=np.int64)
        
        use_percentile = False
        
        # Prepare all quantile points in one list to minimize data scans
        # Always need: min(0), max(1), trim_low, trim_high
        trim_low = self.filtering_cfg.trim_lower
        trim_high = self.filtering_cfg.trim_upper
        
        # Initialize p_low and p_high (not used when use_percentile is False)
        p_low = 0.0
        p_high = 1.0
        
        if use_percentile:
            # Combine all unique quantiles: [0, p_low, trim_low, trim_high, p_high, 1.0]
            all_q_points = sorted(set([0.0, p_low, trim_low, trim_high, p_high, 1.0]))
        else:
            # Just need: [0, trim_low, trim_high, 1.0]
            all_q_points = sorted(set([0.0, trim_low, trim_high, 1.0]))
        
        # Create mapping for extracting specific quantiles
        q_indices = {q: all_q_points.index(q) for q in all_q_points}
        
        # Get total row count
        total_count = len(dset.to_table(columns=[columns[0]])[columns[0]])
        
        # Compute statistics for each column
        for idx, col_name in enumerate(columns):
            col = dset.to_table(columns=[col_name])[col_name]
            
            # Count valid (non-null, finite) values
            n_valid[idx] = pc.count(col, mode='only_valid').as_py()
            n_missing[idx] = total_count - n_valid[idx]
            
            # Skip columns with no valid data
            if n_valid[idx] == 0:
                min_vals[idx] = 0.0
                max_vals[idx] = 0.0
                clip_lower[idx] = 0.0
                clip_upper[idx] = 0.0
                means[idx] = 0.0
                stds[idx] = 0.0
                variances[idx] = 0.0
                trimmed_ranges[idx] = RANGE_EPS
                continue
            
            # ALL quantiles in ONE call (major optimization!)
            try:
                all_quantiles = pc.quantile(col, q=all_q_points, skip_nulls=True).to_pylist()
                # Handle None values in quantiles
                all_quantiles = [q if q is not None else 0.0 for q in all_quantiles]
                
                # Extract specific quantiles from results
                min_vals[idx] = all_quantiles[q_indices[0.0]]
                max_vals[idx] = all_quantiles[q_indices[1.0]]
                
                if use_percentile:
                    clip_lower[idx] = all_quantiles[q_indices[p_low]]
                    clip_upper[idx] = all_quantiles[q_indices[p_high]]
                else:
                    clip_lower[idx] = min_vals[idx]
                    clip_upper[idx] = max_vals[idx]
                
                # Trimmed range (already computed in same call!)
                trim_lower_val = all_quantiles[q_indices[trim_low]]
                trim_upper_val = all_quantiles[q_indices[trim_high]]
                trimmed_ranges[idx] = max(trim_upper_val - trim_lower_val, RANGE_EPS)
                
            except Exception as e:
                min_vals[idx] = 0.0
                max_vals[idx] = 0.0
                clip_lower[idx] = 0.0
                clip_upper[idx] = 0.0
                trimmed_ranges[idx] = RANGE_EPS
            
            # Mean and std (separate efficient operations)
            try:
                means[idx] = pc.mean(col, skip_nulls=True).as_py() or 0.0
                stds[idx] = pc.stddev(col, skip_nulls=True).as_py() or 0.0
                variances[idx] = stds[idx] ** 2
            except:
                means[idx] = 0.0
                stds[idx] = 0.0
                variances[idx] = 0.0
            
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - stat_start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (p - idx - 1) / rate if rate > 0 else 0
                self._log(f"        Progress: {idx+1}/{p} columns ({100*(idx+1)/p:.1f}%) | "
                         f"{rate:.1f} cols/s | ETA: {eta/60:.1f}min")
        
        # Compute normalized variance with safe division
        ranges = np.maximum(max_vals - min_vals, RANGE_EPS)
        
        # Square the ranges and clip to prevent division by near-zero
        ranges_squared = np.maximum(ranges ** 2, RANGE_EPS ** 2)
        trimmed_ranges_squared = np.maximum(trimmed_ranges ** 2, RANGE_EPS ** 2)
        
        # Safe division with explicit overflow handling
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_variance = variances / ranges_squared
            normalized_variance_robust = variances / trimmed_ranges_squared
        
        # Handle NaN/Inf values in all statistics
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
        variances = np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)
        min_vals = np.nan_to_num(min_vals, nan=0.0, posinf=0.0, neginf=0.0)
        max_vals = np.nan_to_num(max_vals, nan=0.0, posinf=0.0, neginf=0.0)
        clip_lower = np.nan_to_num(clip_lower, nan=0.0, posinf=0.0, neginf=0.0)
        clip_upper = np.nan_to_num(clip_upper, nan=0.0, posinf=0.0, neginf=0.0)
        ranges = np.nan_to_num(ranges, nan=RANGE_EPS, posinf=0.0, neginf=0.0)
        trimmed_ranges = np.nan_to_num(trimmed_ranges, nan=RANGE_EPS, posinf=0.0, neginf=0.0)
        normalized_variance = np.nan_to_num(normalized_variance, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_variance_robust = np.nan_to_num(normalized_variance_robust, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute missing ratios
        missing_ratio = n_missing.astype(np.float64) / total_count
        
        # ===== CDF Lookup Table Generation (for Spearman/RDC) =====
        cdf_start = time.time()
        self._log("\n        Computing CDF lookup tables for Spearman correlation...")
        
        # Use config resolution (default: 1000)
        n_points = getattr(self.filtering_cfg, 'rdc_cdf_resolution', 1000)
        self._log(f"        Quantile points per descriptor: {n_points}")
        self._log(f"        Estimated time: {p * n_points / 100000:.1f} minutes")
        
        cdf_lookups = []
        
        # Progress tracking (5% intervals)
        progress_interval = max(1, p // 20)
        
        for idx, col_name in enumerate(columns):
            col = dset.to_table(columns=[col_name])[col_name]
            
            # Quantile 계산 (PyArrow C++)
            q_vals = [i / (n_points - 1) for i in range(n_points)]
            try:
                lookup_vals = pc.quantile(col, q=q_vals, skip_nulls=True).to_pylist()
                
                # Handle None values
                lookup_vals = [v if v is not None else 0.0 for v in lookup_vals]
                
                # 단조성 보정 (필수)
                lookup_vals = np.array(lookup_vals, dtype=np.float64)
                lookup_vals = np.maximum.accumulate(lookup_vals)
                
                # CDF 값 (float64)
                lookup_cdfs = np.array(q_vals, dtype=np.float64)
                
                cdf_lookups.append((lookup_vals, lookup_cdfs))
                
            except Exception as e:
                # Fallback: use min/max with linear interpolation
                lookup_vals = np.linspace(min_vals[idx], max_vals[idx], n_points, dtype=np.float64)
                lookup_cdfs = np.array(q_vals, dtype=np.float64)
                cdf_lookups.append((lookup_vals, lookup_cdfs))
            
            # Progress update
            if (idx + 1) % progress_interval == 0 or idx == p - 1:
                elapsed = time.time() - cdf_start
                progress_pct = (idx + 1) / p * 100
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (p - idx - 1) / rate if rate > 0 else 0
                self._log(f"        Progress: {progress_pct:.0f}% ({idx+1}/{p}) | "
                         f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        cdf_elapsed = time.time() - cdf_start
        self._log(f"        ✓ Created {len(cdf_lookups)} CDF lookup tables in {cdf_elapsed:.1f}s")
        self._log(f"          ({n_points} points per table, float64 precision)")
        
        return {
            'mean': means,
            'std': stds,
            'variance': variances,
            'min': min_vals,
            'max': max_vals,
            'range': ranges,
            'clip_lower': clip_lower,
            'clip_upper': clip_upper,
            'trimmed_range': trimmed_ranges,
            'normalized_variance': normalized_variance,
            'normalized_variance_robust': normalized_variance_robust,
            'n_total': total_count,
            'n_valid': n_valid,
            'n_missing': n_missing,
            'missing_ratio': missing_ratio,
            'count': total_count,
            'total_rows': total_count,
            'cdf_lookups': cdf_lookups  # ← 추가!
        }
    
    def _filter_low_variance(self, columns: List[str], stats: Dict) -> Tuple[List[str], np.ndarray]:
        """
        저분산 및 결측 데이터가 많은 디스크립터 필터링
        
        Returns:
            (columns_kept, indices_kept)
        """
        p = len(columns)
        
        # Use robust normalized variance if enabled
        if False:
            norm_var = stats['normalized_variance_robust']
        else:
            norm_var = stats['normalized_variance']
        
        # Filter criteria
        variance_mask = norm_var >= self.filtering_cfg.variance_threshold
        missing_mask = stats['missing_ratio'] <= self.filtering_cfg.max_missing_ratio
        effective_n_mask = stats['n_valid'] >= self.filtering_cfg.min_effective_n
        
        # Combined mask
        mask = variance_mask & missing_mask & effective_n_mask
        
        indices_kept = np.where(mask)[0]
        columns_kept = [columns[i] for i in indices_kept]
        
        # Count removal reasons
        n_removed_variance = np.sum(~variance_mask)
        n_removed_missing = np.sum(~missing_mask & variance_mask)
        n_removed_effective_n = np.sum(~effective_n_mask & variance_mask & missing_mask)
        n_total_removed = p - len(columns_kept)
        
        self._log(f"        Removed by low variance: {n_removed_variance}/{p}")
        self._log(f"        Removed by high missing ratio: {n_removed_missing}/{p}")
        self._log(f"        Removed by low effective N: {n_removed_effective_n}/{p}")
        self._log(f"        Total removed: {n_total_removed}/{p}")
        self._log(f"        Remaining: {len(columns_kept)}/{p}")
        
        return columns_kept, indices_kept
    
    def _filter_stats(self, stats: Dict, indices: np.ndarray) -> Dict:
        """필터링된 인덱스에 맞춰 통계 업데이트"""
        stats_filtered = {}
        for key, value in stats.items():
            if key == 'cdf_lookups':
                # Special handling for CDF lookups (list of tuples)
                stats_filtered[key] = [value[i] for i in indices]
            elif isinstance(value, np.ndarray) and len(value) == len(indices) + (len(stats['mean']) - len(indices)):
                # Array-type stats
                try:
                    stats_filtered[key] = value[indices]
                except:
                    stats_filtered[key] = value
            else:
                # Scalar stats
                stats_filtered[key] = value
        return stats_filtered
    
    def _save_statistics_cache(self, cache_file: Path, stats: Dict, columns: List[str]):
        """통계 결과를 캐시 파일로 저장"""
        cache_data = {
            'columns': columns,
            'config_hash': self._get_config_hash()
        }
        
        # Add all numpy arrays from stats
        for key, value in stats.items():
            if key == 'cdf_lookups':
                # Special handling for CDF lookups (list of tuples)
                cache_data['cdf_lookups'] = np.array(value, dtype=object)
            elif isinstance(value, np.ndarray):
                cache_data[f'stats_{key}'] = value
            else:
                cache_data[f'stats_{key}'] = np.array([value])
        
        np.savez_compressed(cache_file, **cache_data)
    
    def _load_statistics_cache(self, cache_file: Path) -> Tuple[Dict, List[str]]:
        """캐시 파일에서 통계 결과 로드"""
        cache = np.load(cache_file, allow_pickle=True)
        
        # Verify config hasn't changed
        cached_hash = cache['config_hash'].item()
        current_hash = self._get_config_hash()
        if cached_hash != current_hash:
            raise ValueError("Configuration changed, cache invalid")
        
        # Reconstruct stats dictionary
        stats = {}
        for key in cache.keys():
            if key == 'cdf_lookups':
                # Special handling for CDF lookups
                stats['cdf_lookups'] = cache['cdf_lookups'].tolist()
            elif key.startswith('stats_'):
                stat_name = key[6:]  # Remove 'stats_' prefix
                value = cache[key]
                if len(value) == 1 and stat_name in ['count', 'total_rows', 'n_total']:
                    stats[stat_name] = value.item()
                else:
                    stats[stat_name] = value
        
        columns = cache['columns'].tolist()
        
        return stats, columns
    
    def _get_config_hash(self) -> str:
        """캐시 검증을 위한 설정 해시 생성"""
        import hashlib
        config_str = f"{self.filtering_cfg.variance_threshold}_{self.filtering_cfg.max_missing_ratio}_" \
                    f"{self.filtering_cfg.min_effective_n}_{False}_" \
                    f"{self.filtering_cfg.trim_lower}_{self.filtering_cfg.trim_upper}"
        return hashlib.md5(config_str.encode()).hexdigest()
