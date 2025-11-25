"""
Main Pipeline - Integrated descriptor filtering pipeline
WITH CHECKPOINT RESUME SUPPORT
"""

import glob
import time
import json
import numpy as np
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
from descriptor_pipeline.io.parquet_reader import iter_batches


class DescriptorPipeline:
    """
    통합 디스크립터 필터링 파이프라인
    
    Pipeline:
    - Pass 0: Sampling (optional)
    - Pass 1: Statistics + Low Variance + Missing Data Filtering
    - Pass 2: Spearman Correlation (GPU)
    - Pass 3: VIF Multicollinearity (GPU)
    - Pass 4: HSIC + RDC Nonlinear (GPU)
    
    NEW: Checkpoint resume support - automatically skip completed passes
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
    
    def _check_checkpoint(self, filename: str) -> bool:
        """Check if a checkpoint file exists"""
        checkpoint_path = self.output_dir / filename
        return checkpoint_path.exists()
    
    def _load_checkpoint(self, filename: str) -> Optional[Dict]:
        """Load checkpoint data if exists"""
        checkpoint_path = self.output_dir / filename
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_intermediate_columns(self, filename: str, columns: List[str]):
        """Save intermediate column list"""
        with open(self.output_dir / filename, 'w') as f:
            f.write("\n".join(columns))
    
    def _load_intermediate_columns(self, filename: str) -> Optional[List[str]]:
        """Load intermediate column list"""
        filepath = self.output_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            except:
                return None
        return None
    
    def run(self) -> Dict[str, Any]:
        """
        전체 파이프라인 실행 (체크포인트 재개 지원)
        
        Returns:
            Dict: 결과 정보
        """
        self._log("="*70)
        self._log("Descriptor Pipeline - Unified GPU-Accelerated Filtering")
        self._log("WITH CHECKPOINT RESUME SUPPORT")
        self._log("="*70)
        if self.using_gpu:
            import torch
            self._log(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            self._log(f"GPU Memory: {gpu_mem:.1f} GB")
        else:
            self._log("Device: CPU")
        self._log(f"Output: {self.output_dir}")
        self._log(f"Checkpoint: {'Enabled' if self.config.checkpoint else 'Disabled'}")
        self._log("="*70)
        
        # Get input files
        parquet_paths = sorted(glob.glob(self.config.parquet_glob))
        if not parquet_paths:
            raise RuntimeError(f"No parquet files found: {self.config.parquet_glob}")
        self._log(f"\nInput files: {len(parquet_paths)}")
        
        # Get all columns (for Pass 0 sampling)
        all_columns = self._get_all_columns(parquet_paths)
        
        # Get descriptor columns (metadata excluded)
        self.columns = self._get_columns(parquet_paths)
        p_initial = len(self.columns)
        self._log(f"Descriptor columns (after metadata exclusion): {p_initial}")
        
        # ===== Pass 0: Sampling =====
        sampled_file_path = self.output_dir / "sampled_data.parquet"
        if self.config.checkpoint and sampled_file_path.exists():
            self._log("\n✓ Pass 0: Sampling already completed (using cached file)")
            parquet_paths = [str(sampled_file_path)]
            self.columns = self._get_columns([str(sampled_file_path)])
        else:
            # Pass all columns to Pass 0 (including metadata)
            sampled_file = self.pass0.run(parquet_paths, all_columns, self.output_dir)
            if sampled_file:
                parquet_paths = [sampled_file]
                # After sampling, update descriptor columns from sampled file (exclude metadata)
                self.columns = self._get_columns([sampled_file])
        
        # ===== Pass 1: Statistics + Variance Filtering =====
        pass1_checkpoint = self._load_checkpoint("pass1_variance_filtering.json")
        pass1_columns_file = self.output_dir / "pass1_columns.txt"
        pass1_stats_file = self.output_dir / "pass1_stats.npz"
        
        if (self.config.checkpoint and pass1_checkpoint is not None 
            and pass1_columns_file.exists() and pass1_stats_file.exists()):
            self._log("\n✓ Pass 1: Statistics & Variance Filtering already completed (loading from checkpoint)")
            columns_p1 = self._load_intermediate_columns("pass1_columns.txt")
            stats_p1 = dict(np.load(str(pass1_stats_file), allow_pickle=True))
            # Reconstruct indices
            indices_p1 = np.array([i for i, c in enumerate(self.columns) if c in columns_p1])
        else:
            columns_p1, stats_p1, indices_p1 = self.pass1.compute(parquet_paths, self.columns)
            self.stats = stats_p1
            
            if self.config.checkpoint:
                self._save_checkpoint("pass1_variance_filtering.json", {
                    'removed_count': p_initial - len(columns_p1),
                    'remaining_count': len(columns_p1),
                    'removed_columns': [c for i, c in enumerate(self.columns) if i not in indices_p1]
                })
                self._save_intermediate_columns("pass1_columns.txt", columns_p1)
                # Save stats with proper handling of object arrays
                stats_to_save = {}
                for key, value in stats_p1.items():
                    if isinstance(value, (list, tuple)):
                        stats_to_save[key] = np.array(value)
                    else:
                        stats_to_save[key] = value
                np.savez(str(pass1_stats_file), **stats_to_save)
        
        self._log(f"Pass 1: {len(columns_p1)} columns remaining (removed {p_initial - len(columns_p1)})")
        
        # ===== Pass 2: Spearman Correlation =====
        pass2_checkpoint = self._load_checkpoint("pass2_spearman.json")
        pass2_columns_file = self.output_dir / "pass2_columns.txt"
        pass2_graph_file = self.output_dir / "pass2_spearman_matrix.npy"
        
        if (self.config.checkpoint and pass2_checkpoint is not None 
            and pass2_columns_file.exists() and pass2_graph_file.exists()):
            self._log("\n✓ Pass 2: Spearman Correlation already completed (loading from checkpoint)")
            columns_p2 = self._load_intermediate_columns("pass2_columns.txt")
            # Reconstruct indices
            indices_p2 = np.array([i for i, c in enumerate(columns_p1) if c in columns_p2])
        else:
            self._log("\n" + "="*70)
            self._log("Pass2: Spearman Correlation Filtering (GPU)")
            self._log("="*70)
            t_start_p2 = time.time()
            
            spearman_gpu = SpearmanComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
            G_spearman = spearman_gpu.compute(parquet_paths, columns_p1, stats_p1)
            
            # Save Spearman matrix for checkpoint
            if self.config.checkpoint:
                np.save(str(pass2_graph_file), G_spearman)
            
            # Cluster and filter (no data loading needed - uses stats)
            from descriptor_pipeline.core.advanced_filtering import SpearmanClusteringPass
            spearman_pass = SpearmanClusteringPass(self.config, self.config.verbose)
            columns_p2, spearman_info, indices_p2 = spearman_pass.process(
                None, columns_p1, G_spearman, stats_p1  # Pass None for data - uses stats instead
            )
            
            t_p2 = time.time() - t_start_p2
            self._log(f"\n✅ Pass2 completed in {t_p2:.1f}s")
            self._log(f"  Removed: {len(columns_p1) - len(columns_p2)}")
            self._log(f"  Remaining: {len(columns_p2)}")
            
            if self.config.checkpoint:
                self._save_checkpoint("pass2_spearman.json", spearman_info)
                self._save_intermediate_columns("pass2_columns.txt", columns_p2)
            
            # Clear GPU memory
            if self.using_gpu:
                import torch
                del G_spearman
                torch.cuda.empty_cache()
        
        self._log(f"Pass 2: {len(columns_p2)} columns remaining")
        
        # ===== Pass 3: VIF =====
        pass3_checkpoint = self._load_checkpoint("pass3_vif.json")
        pass3_columns_file = self.output_dir / "pass3_columns.txt"
        
        if self.config.checkpoint and pass3_checkpoint is not None and pass3_columns_file.exists():
            self._log("\n✓ Pass 3: VIF Multicollinearity already completed (loading from checkpoint)")
            columns_p3 = self._load_intermediate_columns("pass3_columns.txt")
            # Reconstruct indices
            indices_p3_in_p1 = np.array([i for i, c in enumerate(columns_p1) if c in columns_p3])
            indices_p3 = indices_p3_in_p1  # Already in Pass1 space
        else:
            self._log("\n" + "="*70)
            self._log("Pass3: VIF Multicollinearity Filtering (GPU)")
            self._log("="*70)
            t_start_p3 = time.time()
            
            # Load or recompute Spearman matrix
            if pass2_graph_file.exists():
                G_spearman = np.load(str(pass2_graph_file))
            else:
                # Need to recompute
                spearman_gpu = SpearmanComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
                G_spearman = spearman_gpu.compute(parquet_paths, columns_p1, stats_p1)
            
            # Update stats and correlation matrix
            G_spearman_p2 = G_spearman[indices_p2][:, indices_p2]
            stats_p2 = self._filter_stats_by_indices(stats_p1, indices_p2)
            
            vif_pass = VIFFilteringPassGPU(self.config, self.config.verbose, self.device)
            columns_p3, vif_info, indices_p3_sub = vif_pass.process_from_correlation(
                G_spearman_p2, columns_p2, stats_p2
            )
            
            # Remap to Pass1 space
            indices_p3 = indices_p2[np.asarray(indices_p3_sub, dtype=int)]
            
            t_p3 = time.time() - t_start_p3
            self._log(f"\n✅ Pass3 completed in {t_p3:.1f}s")
            self._log(f"  Removed: {len(columns_p2) - len(columns_p3)}")
            self._log(f"  Remaining: {len(columns_p3)}")
            
            if self.config.checkpoint:
                self._save_checkpoint("pass3_vif.json", vif_info)
                self._save_intermediate_columns("pass3_columns.txt", columns_p3)
            
            # Clear GPU memory
            if self.using_gpu:
                import torch
                del G_spearman, G_spearman_p2
                torch.cuda.empty_cache()
        
        self._log(f"Pass 3: {len(columns_p3)} columns remaining")
        
        # ===== Pass 4: Nonlinear (HSIC + RDC) =====
        pass4_checkpoint = self._load_checkpoint("pass4_nonlinear.json")
        final_descriptors_file = self.output_dir / "final_descriptors.txt"
        
        if self.config.checkpoint and pass4_checkpoint is not None and final_descriptors_file.exists():
            self._log("\n✓ Pass 4: HSIC + RDC Nonlinear already completed (loading from checkpoint)")
            columns_final = self._load_intermediate_columns("final_descriptors.txt")
            self.final_columns = columns_final
        else:
            self._log("\n" + "="*70)
            self._log("Pass4: HSIC + RDC Nonlinear Detection (GPU)")
            self._log("="*70)
            t_start_p4 = time.time()
            
            # Update stats only (data not needed)
            stats_p3 = self._filter_stats_by_indices(stats_p1, indices_p3)
            
            # Compute HSIC and RDC
            from descriptor_pipeline.core.similarity_gpu import HSICComputerGPU, RDCComputerGPU
            
            w_hsic, w_rdc = self.config.w_hsic, self.config.w_rdc
            
            if w_hsic > 0:
                hsic_gpu = HSICComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
                G_hsic = hsic_gpu.compute(parquet_paths, columns_p3, stats_p3)
            else:
                G_hsic = np.eye(len(columns_p3))
            
            if w_rdc > 0:
                rdc_gpu = RDCComputerGPU(self.config, self.seed_mgr, self.config.verbose, self.device)
                G_rdc = rdc_gpu.compute(parquet_paths, columns_p3, stats_p3)
            else:
                G_rdc = np.eye(len(columns_p3))
            
            # Apply nonlinear filtering (pass None for data - not used)
            nonlinear_pass = NonlinearDetectionPassGPU(self.config, self.config.verbose, self.device)
            columns_final, nonlinear_info, indices_final = nonlinear_pass.process(
                None, columns_p3, G_hsic, G_rdc, stats_p3,
                self.graph_builder, self.leiden
            )
            
            self.final_columns = columns_final
            
            t_p4 = time.time() - t_start_p4
            self._log(f"\n✅ Pass4 completed in {t_p4:.1f}s")
            self._log(f"  Removed: {len(columns_p3) - len(columns_final)}")
            self._log(f"  Remaining: {len(columns_final)}")
            
            if self.config.checkpoint:
                self._save_checkpoint("pass4_nonlinear.json", nonlinear_info)
            
            # Save final results
            self._save_final_results(columns_final)
            
            # Clear GPU memory
            if self.using_gpu:
                import torch
                del G_hsic, G_rdc
                torch.cuda.empty_cache()
        
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
        """
        Get descriptor column names from dataset (excluding metadata)
        
        Returns:
            List of descriptor column names (metadata excluded)
        """
        if self.config.descriptor_columns is not None:
            return self.config.descriptor_columns
        
        try:
            dataset = ds.dataset(parquet_paths, format="parquet")
            all_columns = dataset.schema.names
        except:
            dataset = ds.dataset(parquet_paths[0], format="parquet")
            all_columns = dataset.schema.names
        
        # Exclude metadata columns
        n_meta = self.config.n_metadata_cols
        if n_meta > 0:
            self._log(f"  Total columns in file: {len(all_columns)}")
            self._log(f"  Metadata columns (excluded): {n_meta}")
            self._log(f"  Metadata column names: {all_columns[:n_meta]}")
            descriptor_columns = all_columns[n_meta:]
            return descriptor_columns
        else:
            return all_columns
    
    def _get_all_columns(self, parquet_paths: List[str]) -> List[str]:
        """Get all column names including metadata (for Pass 0)"""
        try:
            dataset = ds.dataset(parquet_paths, format="parquet")
            return dataset.schema.names
        except:
            dataset = ds.dataset(parquet_paths[0], format="parquet")
            return dataset.schema.names
    
    def _load_data(self, parquet_paths: List[str], columns: List[str]) -> np.ndarray:
        """Load data into memory"""
        batches = []
        for batch_data, offset in iter_batches(parquet_paths, columns, self.config.batch_rows):
            batches.append(batch_data)
        return np.vstack(batches)
    
    def _filter_stats_by_indices(self, stats: Dict, indices: np.ndarray) -> Dict:
        """Filter statistics by indices"""
        stats_filtered = {}
        for key, value in stats.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                try:
                    stats_filtered[key] = value[indices]
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
