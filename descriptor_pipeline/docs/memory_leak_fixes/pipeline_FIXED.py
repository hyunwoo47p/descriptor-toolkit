"""
Main Pipeline - Integrated descriptor filtering pipeline (Memory Safe Version)

메모리 누수 수정사항:
- Pass 2 함수 호출 인자 수정 (graph_builder, leiden 제거)
- NumPy view → .copy()로 변경하여 원본 참조 제거
- 각 Pass 후 명시적 메모리 정리
- GPU 메모리 명시적 해제
"""

import glob
import time
import json
import numpy as np
import gc
from pathlib import Path
from typing import List, Dict, Optional, Any
import pyarrow.dataset as ds

from descriptor_pipeline.config.settings import PipelineConfig
from descriptor_pipeline.utils.logging import log
from descriptor_pipeline.utils.gpu import get_optimal_device
from descriptor_pipeline.core.pass0_sampling import SamplingPass
from descriptor_pipeline.core.pass1_statistics import StatisticsAndVarianceFilter
from descriptor_pipeline.core.similarity_gpu import SpearmanComputerGPU
from descriptor_pipeline.core.advanced_filtering_gpu import VIFFilteringPassGPU, NonlinearDetectionPassGPU
from descriptor_pipeline.core.graph_builder import GraphBuilder, LeidenClustering
from descriptor_pipeline.core.seed_manager import SeedManager
from descriptor_pipeline.io import iter_batches


class DescriptorPipeline:
    """
    통합 디스크립터 필터링 파이프라인
    
    Pipeline:
    - Pass 0: Sampling (optional)
    - Pass 1: Statistics + Low Variance Filtering
    - Pass 2: Spearman Correlation (GPU)
    - Pass 3: VIF Multicollinearity (GPU)
    - Pass 4: HSIC + RDC Nonlinear (GPU)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # GPU setup
        self.device = get_optimal_device(config.prefer_gpu)
        self.using_gpu = (self.device.type == 'cuda')
        
        # Seed manager
        self.seed_mgr = SeedManager(config.random_seed)
        
        # Results
        self.columns: Optional[List[str]] = None
        self.final_columns: Optional[List[str]] = None
        self.stats: Optional[Dict] = None
        
        # Passes
        self.pass0 = SamplingPass(config, config.verbose)
        self.pass1 = StatisticsAndVarianceFilter(config, self.device, config.verbose)
        self.graph_builder = GraphBuilder(config, config.verbose)
        self.leiden = LeidenClustering(config, self.seed_mgr, config.verbose)
    
    def _log(self, msg: str):
        log(msg, self.config.verbose)
    
    def _cleanup_memory(self):
        """명시적 메모리 정리"""
        gc.collect()
        if self.using_gpu:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def run(self) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Returns:
            Dict: 결과 정보
        """
        self._log("="*70)
        self._log("Descriptor Pipeline - Unified GPU-Accelerated Filtering")
        self._log("="*70)
        if self.using_gpu:
            import torch
            self._log(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._log("Device: CPU")
        self._log(f"Output: {self.output_dir}")
        self._log("="*70)
        
        # Get input files
        parquet_paths = sorted(glob.glob(self.config.parquet_glob))
        if not parquet_paths:
            raise RuntimeError(f"No parquet files found: {self.config.parquet_glob}")
        self._log(f"\nInput files: {len(parquet_paths)}")
        
        # Get columns
        self.columns = self._get_columns(parquet_paths)
        p_initial = len(self.columns)
        self._log(f"Initial descriptors: {p_initial}")
        
        # ===== Pass 0: Sampling =====
        sampled_file = self.pass0.run(parquet_paths, self.columns, self.output_dir)
        if sampled_file:
            parquet_paths = [sampled_file]
        
        self._cleanup_memory()
        
        # ===== Pass 1: Statistics + Variance Filtering =====
        columns_p1, stats_p1, indices_p1 = self.pass1.compute(parquet_paths, self.columns)
        self.stats = stats_p1
        
        if self.config.checkpoint:
            self._save_checkpoint("pass1_variance_filtering.json", {
                'removed_count': p_initial - len(columns_p1),
                'remaining_count': len(columns_p1),
                'removed_columns': [c for i, c in enumerate(self.columns) if i not in indices_p1]
            })
        
        self._cleanup_memory()
        
        # Load data for remaining passes
        data = self._load_data(parquet_paths, columns_p1)
        
        # ===== Pass 2: Spearman Correlation =====
        self._log("\n" + "="*70)
        self._log("Pass2: Spearman Correlation Filtering (GPU)")
        self._log("="*70)
        t_start_p2 = time.time()
        
        spearman_gpu = SpearmanComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
        G_spearman = spearman_gpu.compute(parquet_paths, columns_p1, stats_p1)
        
        self._cleanup_memory()
        
        # Cluster and filter
        from descriptor_pipeline.core.advanced_filtering import SpearmanClusteringPass
        spearman_pass = SpearmanClusteringPass(self.config, self.config.verbose)
        
        # FIXED: 4개 인자만 전달 (graph_builder, leiden 제거)
        columns_p2, spearman_info, indices_p2 = spearman_pass.process(
            data, columns_p1, G_spearman, stats_p1
        )
        
        t_p2 = time.time() - t_start_p2
        self._log(f"\n✅ Pass2 completed in {t_p2:.1f}s")
        self._log(f"  Removed: {len(columns_p1) - len(columns_p2)}")
        self._log(f"  Remaining: {len(columns_p2)}")
        
        if self.config.checkpoint:
            self._save_checkpoint("pass2_spearman.json", spearman_info)
        
        self._cleanup_memory()
        
        # ===== Pass 3: VIF =====
        self._log("\n" + "="*70)
        self._log("Pass3: VIF Multicollinearity Filtering (GPU)")
        self._log("="*70)
        t_start_p3 = time.time()
        
        # Update data and stats - FIXED: .copy() 추가
        data_p2 = data[:, indices_p2].copy()
        G_spearman_p2 = G_spearman[indices_p2][:, indices_p2].copy()
        stats_p2 = self._filter_stats_by_indices(stats_p1, indices_p2)
        
        # 원본 삭제하여 메모리 확보
        del data, G_spearman
        self._cleanup_memory()
        
        vif_pass = VIFFilteringPassGPU(self.config, self.config.verbose, self.device)
        columns_p3, vif_info, indices_p3_sub = vif_pass.process_from_correlation(
            G_spearman_p2, columns_p2, stats_p2
        )
        
        # Remap to original space
        indices_p3 = indices_p2[np.asarray(indices_p3_sub, dtype=int)]
        
        t_p3 = time.time() - t_start_p3
        self._log(f"\n✅ Pass3 completed in {t_p3:.1f}s")
        self._log(f"  Removed: {len(columns_p2) - len(columns_p3)}")
        self._log(f"  Remaining: {len(columns_p3)}")
        
        if self.config.checkpoint:
            self._save_checkpoint("pass3_vif.json", vif_info)
        
        # 이전 데이터 삭제
        del G_spearman_p2
        self._cleanup_memory()
        
        # ===== Pass 4: Nonlinear (HSIC + RDC) =====
        self._log("\n" + "="*70)
        self._log("Pass4: HSIC + RDC Nonlinear Detection (GPU)")
        self._log("="*70)
        t_start_p4 = time.time()
        
        # Update data and stats - FIXED: .copy() 추가
        data_p3 = data_p2[:, indices_p3_sub].copy()
        stats_p3 = self._filter_stats_by_indices(stats_p1, indices_p3)
        
        # 이전 데이터 삭제
        del data_p2, stats_p1, stats_p2
        self._cleanup_memory()
        
        # Compute HSIC and RDC
        from descriptor_pipeline.core.similarity_gpu import HSICComputerGPU, RDCComputerGPU
        
        w_hsic, w_rdc = self.config.w_hsic, self.config.w_rdc
        
        if w_hsic > 0:
            hsic_gpu = HSICComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
            G_hsic = hsic_gpu.compute(parquet_paths, columns_p3, stats_p3)
            self._cleanup_memory()
        else:
            G_hsic = np.eye(len(columns_p3))
        
        if w_rdc > 0:
            rdc_gpu = RDCComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
            G_rdc = rdc_gpu.compute(parquet_paths, columns_p3, stats_p3)
            self._cleanup_memory()
        else:
            G_rdc = np.eye(len(columns_p3))
        
        # Apply nonlinear filtering
        nonlinear_pass = NonlinearDetectionPassGPU(self.config, self.config.verbose, self.device)
        columns_final, nonlinear_info, indices_final = nonlinear_pass.process(
            data_p3, columns_p3, G_hsic, G_rdc, stats_p3,
            self.graph_builder, self.leiden
        )
        
        self.final_columns = columns_final
        
        # 최종 메모리 정리
        del data_p3, G_hsic, G_rdc, stats_p3
        self._cleanup_memory()
        
        t_p4 = time.time() - t_start_p4
        self._log(f"\n✅ Pass4 completed in {t_p4:.1f}s")
        self._log(f"  Removed: {len(columns_p3) - len(columns_final)}")
        self._log(f"  Remaining: {len(columns_final)}")
        
        if self.config.checkpoint:
            self._save_checkpoint("pass4_nonlinear.json", nonlinear_info)
        
        # ===== Summary =====
        self._log("\n" + "="*70)
        self._log("Pipeline Completed Successfully!")
        self._log("="*70)
        self._log(f"Initial descriptors:      {p_initial:>6}")
        self._log(f"After Pass1 (Variance):   {len(columns_p1):>6}  (-{p_initial - len(columns_p1)})")
        self._log(f"After Pass2 (Spearman):   {len(columns_p2):>6}  (-{len(columns_p1) - len(columns_p2)})")
        self._log(f"After Pass3 (VIF):        {len(columns_p3):>6}  (-{len(columns_p2) - len(columns_p3)})")
        self._log(f"After Pass4 (Nonlinear):  {len(columns_final):>6}  (-{len(columns_p3) - len(columns_final)})")
        self._log(f"")
        self._log(f"Final descriptors:        {len(columns_final):>6}")
        self._log(f"Total removed:            {p_initial - len(columns_final):>6}")
        self._log(f"Reduction:                {100 * (1 - len(columns_final) / p_initial):>5.1f}%")
        self._log("="*70)
        
        # Save final results
        self._save_final_results(columns_final)
        
        # 최종 정리
        self._cleanup_memory()
        
        return {
            'initial_count': p_initial,
            'final_count': len(columns_final),
            'final_columns': columns_final,
            'pass1_removed': p_initial - len(columns_p1),
            'pass2_removed': len(columns_p1) - len(columns_p2),
            'pass3_removed': len(columns_p2) - len(columns_p3),
            'pass4_removed': len(columns_p3) - len(columns_final),
        }
    
    def _get_columns(self, parquet_paths: List[str]) -> List[str]:
        """Get column names from dataset"""
        if self.config.descriptor_columns is not None:
            return self.config.descriptor_columns
        
        try:
            dataset = ds.dataset(parquet_paths, format="parquet")
            return dataset.schema.names
        except:
            dataset = ds.dataset(parquet_paths[0], format="parquet")
            return dataset.schema.names
    
    def _load_data(self, parquet_paths: List[str], columns: List[str]) -> np.ndarray:
        """Load data into memory with explicit cleanup"""
        batches = []
        batch_generator = None
        
        try:
            batch_generator = iter_batches(parquet_paths, columns, self.config.batch_rows)
            
            for batch_data, offset in batch_generator:
                # 명시적 복사로 generator 참조 제거
                batches.append(batch_data.copy())
                del batch_data
        
        finally:
            # Generator cleanup
            if batch_generator is not None:
                try:
                    batch_generator.close()
                except:
                    pass
            
            gc.collect()
        
        result = np.vstack(batches)
        
        # 중간 리스트 삭제
        del batches
        gc.collect()
        
        return result
    
    def _filter_stats_by_indices(self, stats: Dict, indices: np.ndarray) -> Dict:
        """Filter statistics by indices with explicit copy"""
        stats_filtered = {}
        
        for key, value in stats.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                try:
                    # FIXED: .copy() 추가하여 독립 배열 생성
                    stats_filtered[key] = value[indices].copy()
                except:
                    stats_filtered[key] = value
            elif isinstance(value, list):
                # 리스트도 필터링
                try:
                    stats_filtered[key] = [value[i] for i in indices]
                except:
                    stats_filtered[key] = value
            else:
                stats_filtered[key] = value
        
        return stats_filtered
    
    def _save_checkpoint(self, filename: str, data: Dict):
        """Save checkpoint file"""
        with open(self.output_dir / filename, 'w') as f:
            # Convert numpy types to native Python types for JSON
            data_json = self._convert_to_json_serializable(data)
            json.dump(data_json, f, indent=2)
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(x) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _save_final_results(self, columns: List[str]):
        """Save final descriptor list"""
        with open(self.output_dir / "final_descriptors.txt", 'w') as f:
            f.write("\n".join(columns))
        self._log(f"\nSaved final descriptor list to: {self.output_dir / 'final_descriptors.txt'}")
