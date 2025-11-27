"""
Pass0: Data Sampling

Creates a unified sampled parquet file if sampling is enabled.
"""

import time
from pathlib import Path
from typing import List, Optional

from Chem_Descriptor_ML.config.settings import FilteringConfig, IOConfig, SystemConfig
from Chem_Descriptor_ML.filtering.utils.logging import log
from Chem_Descriptor_ML.filtering.io.parquet_reader import create_sampled_parquet_file


class SamplingPass:
    """Pass0: Unified sampling"""
    
    def __init__(
        self,
        filtering_cfg: FilteringConfig,
        io_cfg: IOConfig,
        system_cfg: SystemConfig
    ):
        self.filtering_cfg = filtering_cfg
        self.io_cfg = io_cfg
        self.system_cfg = system_cfg
        self.verbose = system_cfg.verbose
    
    def _log(self, msg: str):
        log(msg, self.verbose)
    
    def run(self, parquet_paths: List[str], columns: List[str], output_dir: Path) -> Optional[str]:
        """
        샘플링 수행 (모든 컬럼 포함, 메타데이터 포함)
        
        Args:
            parquet_paths: 입력 파일 경로들
            columns: 컬럼 이름 (전체, 메타데이터 포함)
            output_dir: 출력 디렉토리
            
        Returns:
            sampled_file_path or None if no sampling
        """
        if self.filtering_cfg.sample_per_file is None:
            return None
        
        self._log("\n" + "="*70)
        self._log("Pass0: Creating Sampled Dataset")
        self._log("="*70)
        self._log(f"  Sample per file: {self.filtering_cfg.sample_per_file:,}")
        self._log(f"  Random seed: {self.system_cfg.random_seed}")
        self._log(f"  Total columns: {len(columns)} (including metadata)")
        
        t_start = time.time()
        
        sampled_file_path = output_dir / "sampled_data.parquet"
        
        if sampled_file_path.exists():
            self._log(f"\n  ✓ Found existing sampled file")
            file_size_mb = sampled_file_path.stat().st_size / (1024 * 1024)
            self._log(f"    Size: {file_size_mb:.1f} MB")
            return str(sampled_file_path)
        
        # Create sampled file (with all columns including metadata)
        create_sampled_parquet_file(
            parquet_paths,
            columns,  # Include all columns
            self.filtering_cfg.sample_per_file,
            sampled_file_path,
            self.system_cfg.random_seed,
            verbose=self.verbose
        )
        
        t_elapsed = time.time() - t_start
        file_size_mb = sampled_file_path.stat().st_size / (1024 * 1024)
        
        self._log(f"\n✅ Pass0 completed in {t_elapsed:.1f}s ({t_elapsed/60:.1f}min)")
        self._log(f"  Sampled file: {sampled_file_path}")
        self._log(f"  File size: {file_size_mb:.1f} MB")
        
        return str(sampled_file_path)
