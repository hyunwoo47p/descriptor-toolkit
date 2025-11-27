"""
ChemDescriptorML (CDML) - Track 1: Descriptor Filtering Pipeline

5-Stage filtering pipeline with checkpoint resume support:
- Pass 0: Sampling (for large datasets)
- Pass 1: Variance Filtering (remove low-variance descriptors)
- Pass 2: Spearman Correlation Clustering
- Pass 3: VIF (Variance Inflation Factor) Filtering
- Pass 4: Nonlinear Analysis (HSIC + RDC)

Memory optimization:
- Pass 2에서 data=None 전달 (stats 사용)
- .copy() 사용하여 NumPy view 참조 제거
- 각 Pass 후 명시적 메모리 정리
- GPU 메모리 명시적 해제
"""

import glob
import time
import json
import numpy as np
import gc
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import pyarrow.dataset as ds

from Chem_Descriptor_ML.config.settings import Config
from Chem_Descriptor_ML.filtering.utils.logging import log
from Chem_Descriptor_ML.filtering.utils.gpu import get_optimal_device
from Chem_Descriptor_ML.filtering.passes.pass0_sampling import SamplingPass
from Chem_Descriptor_ML.filtering.passes.pass1_statistics import StatisticsAndVarianceFilter
from Chem_Descriptor_ML.filtering.passes.pass2_correlation import SpearmanComputerGPU
from Chem_Descriptor_ML.filtering.passes.pass4_nonlinear import NonlinearDetectionPassGPU
from Chem_Descriptor_ML.filtering.passes.pass3_vif import VIFFilteringPassGPUWithClusters
from Chem_Descriptor_ML.filtering.passes.graph_builder import GraphBuilder, LeidenClustering
from Chem_Descriptor_ML.filtering.passes.seed_manager import SeedManager
from Chem_Descriptor_ML.filtering.io.parquet_reader import iter_batches

# Import new iterative VIF (conditional for backward compatibility)
try:
    from Chem_Descriptor_ML.filtering.passes.vif_iterative import IterativeVIFFiltering
    ITERATIVE_VIF_AVAILABLE = True
except ImportError:
    ITERATIVE_VIF_AVAILABLE = False

# Backward compatibility
PipelineConfig = Config


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
        
        # Extract sections
        self.device_cfg = config.device
        self.io_cfg = config.io
        self.filtering_cfg = config.filtering
        self.system_cfg = config.system
        
        # Setup output directory
        self.output_dir = Path(self.io_cfg.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # GPU setup
        self.device = get_optimal_device(self.device_cfg.prefer_gpu)
        self.using_gpu = (self.device.type == 'cuda')
        
        # Seed manager
        self.seed_mgr = SeedManager(self.system_cfg.random_seed)
        
        # Results
        self.columns: Optional[List[str]] = None
        self.final_columns: Optional[List[str]] = None
        self.stats: Optional[Dict] = None
        
        # Passes - now use section-based configs
        self.pass0 = SamplingPass(
            filtering_cfg=self.filtering_cfg,
            io_cfg=self.io_cfg,
            system_cfg=self.system_cfg
        )
        self.pass1 = StatisticsAndVarianceFilter(
            filtering_cfg=self.filtering_cfg,
            io_cfg=self.io_cfg,
            system_cfg=self.system_cfg,
            device=self.device
        )
        self.graph_builder = GraphBuilder(
            filtering_cfg=self.filtering_cfg,
            system_cfg=self.system_cfg
        )
        self.leiden = LeidenClustering(
            filtering_cfg=self.filtering_cfg,
            seed_mgr=self.seed_mgr,
            system_cfg=self.system_cfg
        )
    
    def _log(self, msg: str):
        log(msg, self.system_cfg.verbose)
    
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
    
    def _cleanup_memory(self):
        """명시적 메모리 정리"""
        gc.collect()
        if self.using_gpu:
            import torch
            torch.cuda.empty_cache()
    
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
        self._log(f"Checkpoint: {'Enabled' if self.system_cfg.checkpoint else 'Disabled'}")
        self._log("="*70)
        
        # Get input files
        parquet_paths = sorted(glob.glob(self.io_cfg.parquet_glob))
        if not parquet_paths:
            raise RuntimeError(f"No parquet files found: {self.io_cfg.parquet_glob}")
        self._log(f"\nInput files: {len(parquet_paths)}")
        
        # Get all columns (for Pass 0 sampling)
        all_columns = self._get_all_columns(parquet_paths)
        
        # Get descriptor columns (metadata excluded)
        self.columns = self._get_columns(parquet_paths)
        p_initial = len(self.columns)
        self._log(f"Descriptor columns (after metadata exclusion): {p_initial}")
        
        # ===== Pass 0: Sampling =====
        sampled_file_path = self.output_dir / "sampled_data.parquet"
        if self.system_cfg.checkpoint and sampled_file_path.exists():
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
        
        self._cleanup_memory()
        
        # ===== Pass 1: Statistics + Variance Filtering =====
        pass1_checkpoint = self._load_checkpoint("pass1_variance_filtering.json")
        pass1_columns_file = self.output_dir / "pass1_columns.txt"
        pass1_stats_file = self.output_dir / "pass1_stats.npz"
        
        if (self.system_cfg.checkpoint and pass1_checkpoint is not None 
            and pass1_columns_file.exists() and pass1_stats_file.exists()):
            # Load checkpoint
            columns_p1_cached = self._load_intermediate_columns("pass1_columns.txt")
            
            # CRITICAL: Validate checkpoint against current dataset
            # Check if all cached columns exist in current self.columns
            cached_set = set(columns_p1_cached)
            current_set = set(self.columns)
            
            if cached_set.issubset(current_set):
                # Checkpoint is valid
                self._log("\n✓ Pass 1: Statistics & Variance Filtering already completed (loading from checkpoint)")
                columns_p1 = columns_p1_cached
                stats_p1 = dict(np.load(str(pass1_stats_file), allow_pickle=True))
                # Reconstruct indices
                indices_p1 = np.array([i for i, c in enumerate(self.columns) if c in columns_p1])
            else:
                # Checkpoint is invalid - columns don't match
                invalid_cols = cached_set - current_set
                self._log(f"\n⚠ Pass 1 checkpoint invalid: {len(invalid_cols)} columns not in current dataset")
                self._log(f"  Re-computing Pass 1 from scratch...")
                columns_p1, stats_p1, indices_p1 = self.pass1.compute(parquet_paths, self.columns)
                self.stats = stats_p1
                
                if self.system_cfg.checkpoint:
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
        else:
            columns_p1, stats_p1, indices_p1 = self.pass1.compute(parquet_paths, self.columns)
            self.stats = stats_p1
            
            if self.system_cfg.checkpoint:
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
        
        if (self.system_cfg.checkpoint and pass2_checkpoint is not None 
            and pass2_columns_file.exists() and pass2_graph_file.exists()):
            # Load and validate checkpoint
            columns_p2_cached = self._load_intermediate_columns("pass2_columns.txt")
            
            # Validate: columns_p2 should be subset of columns_p1
            if set(columns_p2_cached).issubset(set(columns_p1)):
                self._log("\n✓ Pass 2: Spearman Correlation already completed (loading from checkpoint)")
                columns_p2 = columns_p2_cached
                indices_p2 = np.array([i for i, c in enumerate(columns_p1) if c in columns_p2])
            else:
                self._log("\n⚠ Pass 2 checkpoint invalid: columns don't match")
                self._log(f"  Re-computing Pass 2 from scratch...")
                columns_p2, indices_p2 = None, None  # Will compute below
        else:
            columns_p2, indices_p2 = None, None
        
        if columns_p2 is None:
            self._log("\n" + "="*70)
            self._log("Pass2: Spearman Correlation Filtering (GPU)")
            self._log("="*70)
            t_start_p2 = time.time()
            
            spearman_gpu = SpearmanComputerGPU(self.filtering_cfg, self.io_cfg, self.system_cfg, self.seed_mgr, self.device)
            G_spearman = spearman_gpu.compute(parquet_paths, columns_p1, stats_p1)
            
            # Save Spearman matrix for checkpoint
            if self.system_cfg.checkpoint:
                np.save(str(pass2_graph_file), G_spearman)
            
            # Cluster and filter (no data loading needed - uses stats)
            from Chem_Descriptor_ML.filtering.passes.spearman_clustering import SpearmanClusteringPass
            spearman_pass = SpearmanClusteringPass(self.config, self.system_cfg.verbose)
            columns_p2, spearman_info, indices_p2 = spearman_pass.process(
                None, columns_p1, G_spearman, stats_p1  # Pass None for data - uses stats instead
            )
            
            t_p2 = time.time() - t_start_p2
            self._log(f"\n✅ Pass2 completed in {t_p2:.1f}s")
            self._log(f"  Removed: {len(columns_p1) - len(columns_p2)}")
            self._log(f"  Remaining: {len(columns_p2)}")
            
            if self.system_cfg.checkpoint:
                self._save_checkpoint("pass2_spearman.json", spearman_info)
                self._save_intermediate_columns("pass2_columns.txt", columns_p2)
            
            # Keep G_spearman for Pass3 (don't delete yet)
            # Will be deleted after Pass3
        else:
            # Load spearman_info and G_spearman from checkpoint
            spearman_info = pass2_checkpoint
            if pass2_graph_file.exists():
                G_spearman = np.load(str(pass2_graph_file))
            else:
                self._log("⚠ Warning: Pass2 correlation matrix not found, will recompute in Pass3")
                G_spearman = None
        
        self._log(f"Pass 2: {len(columns_p2)} columns remaining")
        
        # ===== Pass 3: VIF =====
        pass3_checkpoint = self._load_checkpoint("pass3_vif.json")
        pass3_columns_file = self.output_dir / "pass3_columns.txt"
        
        if self.system_cfg.checkpoint and pass3_checkpoint is not None and pass3_columns_file.exists():
            # Load and validate checkpoint
            columns_p3_cached = self._load_intermediate_columns("pass3_columns.txt")
            
            # Validate: columns_p3 should be subset of columns_p1
            if set(columns_p3_cached).issubset(set(columns_p1)):
                self._log("\n✓ Pass 3: VIF Multicollinearity already completed (loading from checkpoint)")
                columns_p3 = columns_p3_cached
                indices_p3_in_p1 = np.array([i for i, c in enumerate(columns_p1) if c in columns_p3])
                indices_p3 = indices_p3_in_p1  # Already in Pass1 space
            else:
                self._log("\n⚠ Pass 3 checkpoint invalid: columns don't match")
                self._log(f"  Re-computing Pass 3 from scratch...")
                columns_p3, indices_p3 = None, None
        else:
            columns_p3, indices_p3 = None, None
        
        if columns_p3 is None:
            self._log("\n" + "="*70)
            self._log("Pass3: Iterative VIF Multicollinearity Filtering")
            self._log("="*70)
            t_start_p3 = time.time()
            
            # Load or recompute G_spearman if needed
            if 'G_spearman' not in locals() or G_spearman is None:
                self._log("Pass2 correlation matrix not found, recomputing...")
                if pass2_graph_file.exists():
                    G_spearman = np.load(str(pass2_graph_file))
                else:
                    spearman_gpu = SpearmanComputerGPU(self.filtering_cfg, self.io_cfg, self.system_cfg, self.seed_mgr, self.device)
                    G_spearman = spearman_gpu.compute(parquet_paths, columns_p1, stats_p1)
            
            # Get Pass2 correlation matrix subset
            self._log("Extracting Pass2 correlation submatrix...")
            G_spearman_p2 = G_spearman[np.ix_(indices_p2, indices_p2)].copy()
            
            # Determine VIF method based on sample size
            # Quick sample size check
            self._log("Checking sample size for VIF method selection...")
            first_batch_data = None
            for batch_data, offset in iter_batches(parquet_paths, columns_p2, batch_rows=50000):
                first_batch_data = batch_data
                break
            
            if first_batch_data is None:
                raise RuntimeError("No data available for VIF calculation")
            
            n_samples = first_batch_data.shape[0]
            p_descriptors = len(columns_p2)
            use_direct_vif = n_samples > 2 * p_descriptors
            
            self._log(f"Samples: {n_samples}, Descriptors: {p_descriptors}")
            self._log(f"Sample/Descriptor ratio: {n_samples/p_descriptors:.2f}")
            
            # Load pass2_info if available
            pass2_info_to_use = None
            if 'spearman_info' in locals():
                pass2_info_to_use = spearman_info
            elif pass2_checkpoint is not None:
                pass2_info_to_use = pass2_checkpoint
            
            if use_direct_vif:
                # Method 1: Direct VIF calculation with actual data
                self._log("→ Using DIRECT VIF calculation (n > 2p)")
                
                # Load full data (reuse first batch, then load rest)
                self._log("Loading Pass2 data...")
                data_p2 = first_batch_data.copy()
                del first_batch_data
                batch_count = 1
                skip_first = True
                
                for batch_data, offset in iter_batches(parquet_paths, columns_p2, batch_rows=50000):
                    if skip_first:  # Skip first batch (already loaded)
                        skip_first = False
                        continue
                    batch_count += 1
                    data_p2 = np.vstack([data_p2, batch_data])
                    
                    if batch_count % 10 == 0:
                        gc.collect()
                        self._log(f"  Loaded {batch_count} batches")
                
                self._log(f"Loaded {data_p2.shape[0]} samples x {data_p2.shape[1]} descriptors")
                
                if pass2_info_to_use is None:
                    self._log("⚠ Warning: Pass2 info not available, cluster tracking may be incomplete")
                
                # Run iterative VIF filtering with direct calculation
                vif_filter = IterativeVIFFiltering(
                    vif_threshold=self.filtering_cfg.vif_threshold,
                    correlation_threshold=0.7,
                    verbose=self.system_cfg.verbose
                )
                
                columns_p3, vif_info = vif_filter.process(
                    data=data_p2,
                    columns=columns_p2,
                    pass2_info=pass2_info_to_use,
                    correlation_matrix=G_spearman_p2,
                    output_dir=self.output_dir
                )
                
                # Cleanup
                del data_p2
                gc.collect()
                
            else:
                # Method 2: Correlation-based VIF (no data loading needed)
                self._log("→ Using CORRELATION-BASED VIF calculation (n ≤ 2p)")
                self._log("   (More stable for small sample sizes)")
                
                # Update stats for Pass2 subset
                stats_p2 = self._filter_stats_by_indices(stats_p1, indices_p2)
                
                if pass2_info_to_use is None:
                    self._log("⚠ Warning: Pass2 info not available, cluster tracking may be incomplete")
                
                # Use GPU-based correlation VIF
                vif_pass = VIFFilteringPassGPUWithClusters(
                    self.filtering_cfg,
                    self.io_cfg,
                    self.system_cfg,
                    self.device
                )
                columns_p3, vif_info, indices_p3_sub = vif_pass.process_from_correlation(
                    G_spearman_p2, columns_p2, stats_p2, pass2_info_to_use
                )
                
                # For correlation-based, we get indices directly
                indices_p3 = indices_p2[np.asarray(indices_p3_sub, dtype=int)]
                
                # Save checkpoint for correlation-based method
                if self.system_cfg.checkpoint:
                    self._save_checkpoint("pass3_vif.json", vif_info)
                
                # Cleanup
                del first_batch_data
                del G_spearman_p2
                gc.collect()
            
            # Calculate indices for direct method
            if use_direct_vif:
                indices_p3_in_p2 = np.array([i for i, c in enumerate(columns_p2) if c in columns_p3])
                indices_p3 = indices_p2[indices_p3_in_p2]
            
            t_p3 = time.time() - t_start_p3
            self._log(f"\n✅ Pass3 completed in {t_p3:.1f}s")
            self._log(f"  Removed: {len(columns_p2) - len(columns_p3)}")
            self._log(f"  Remaining: {len(columns_p3)}")
            
            if self.system_cfg.checkpoint:
                self._save_checkpoint("pass3_vif.json", vif_info)
                self._save_intermediate_columns("pass3_columns.txt", columns_p3)
            
            # Clear memory
            if 'data_p2' in locals():
                del data_p2
            if 'first_batch_data' in locals():
                del first_batch_data
            if 'G_spearman' in locals():
                del G_spearman
            if 'G_spearman_p2' in locals():
                del G_spearman_p2
            self._cleanup_memory()
        
        self._log(f"Pass 3: {len(columns_p3)} columns remaining")
        
        # ===== Pass 4: Nonlinear (HSIC + RDC) =====
        pass4_checkpoint = self._load_checkpoint("pass4_nonlinear.json")
        final_descriptors_file = self.output_dir / "final_descriptors.txt"
        
        if self.system_cfg.checkpoint and pass4_checkpoint is not None and final_descriptors_file.exists():
            # Load and validate checkpoint
            columns_final_cached = self._load_intermediate_columns("final_descriptors.txt")
            
            # Validate: final columns should be subset of columns_p3
            if set(columns_final_cached).issubset(set(columns_p3)):
                self._log("\n✓ Pass 4: HSIC + RDC Nonlinear already completed (loading from checkpoint)")
                columns_final = columns_final_cached
                self.final_columns = columns_final
            else:
                self._log("\n⚠ Pass 4 checkpoint invalid: columns don't match")
                self._log(f"  Re-computing Pass 4 from scratch...")
                columns_final = None
        else:
            columns_final = None
        
        if columns_final is None:
            self._log("\n" + "="*70)
            self._log("Pass4: HSIC + RDC Nonlinear Detection (GPU)")
            self._log("="*70)
            t_start_p4 = time.time()
            
            # Update stats only (data not needed)
            stats_p3 = self._filter_stats_by_indices(stats_p1, indices_p3)
            
            # Compute HSIC and RDC
            from Chem_Descriptor_ML.filtering.passes.pass2_correlation import HSICComputerGPU, RDCComputerGPU
            
            w_hsic, w_rdc = self.filtering_cfg.w_hsic, self.filtering_cfg.w_rdc
            
            if w_hsic > 0:
                hsic_gpu = HSICComputerGPU(
                    filtering_cfg=self.filtering_cfg,
                    io_cfg=self.io_cfg,
                    system_cfg=self.system_cfg,
                    seed_mgr=self.seed_mgr,
                    verbose=self.system_cfg.verbose,
                    device=self.device
                )
                G_hsic = hsic_gpu.compute(parquet_paths, columns_p3, stats_p3)
            else:
                G_hsic = np.eye(len(columns_p3))
            
            if w_rdc > 0:
                rdc_gpu = RDCComputerGPU(
                    filtering_cfg=self.filtering_cfg,
                    io_cfg=self.io_cfg,
                    system_cfg=self.system_cfg,
                    seed_mgr=self.seed_mgr,
                    verbose=self.system_cfg.verbose,
                    device=self.device
                )
                G_rdc = rdc_gpu.compute(parquet_paths, columns_p3, stats_p3)
            else:
                G_rdc = np.eye(len(columns_p3))
            
            # Apply nonlinear filtering (pass None for data - not used)
            nonlinear_pass = NonlinearDetectionPassGPU(self.filtering_cfg, self.io_cfg, self.system_cfg, self.device)
            columns_final, nonlinear_info, indices_final = nonlinear_pass.process(
                None, columns_p3, G_hsic, G_rdc, stats_p3,
                self.graph_builder, self.leiden
            )
            
            self.final_columns = columns_final
            
            t_p4 = time.time() - t_start_p4
            self._log(f"\n✅ Pass4 completed in {t_p4:.1f}s")
            self._log(f"  Removed: {len(columns_p3) - len(columns_final)}")
            self._log(f"  Remaining: {len(columns_final)}")
            
            if self.system_cfg.checkpoint:
                self._save_checkpoint("pass4_nonlinear.json", nonlinear_info)
            
            # Save final results
            self._save_final_results(columns_final)
            
            # Clear memory
            del G_hsic, G_rdc
            self._cleanup_memory()
        
        # ===== Generate Final Integrated Cluster Info =====
        # Load pass info for merging
        pass2_info = self._load_checkpoint("pass2_spearman.json") or {}
        pass3_info = self._load_checkpoint("pass3_vif.json") or {}
        pass4_info = self._load_checkpoint("pass4_nonlinear.json") or {}

        final_cluster_info = self._generate_final_cluster_info(
            columns_final,
            columns_p1, columns_p2, columns_p3,
            pass2_info, pass3_info, pass4_info
        )

        if self.system_cfg.checkpoint:
            self._save_checkpoint("final_cluster_info.json", final_cluster_info)
            self._log(f"\nSaved final cluster info to: {self.output_dir / 'final_cluster_info.json'}")

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
            'final_cluster_info': final_cluster_info,
        }
    
    def _get_columns(self, parquet_paths: List[str]) -> List[str]:
        """
        Get descriptor column names from dataset (excluding metadata)
        
        Returns:
            List of descriptor column names (metadata excluded)
        """
        if self.io_cfg.descriptor_columns is not None:
            return self.io_cfg.descriptor_columns
        
        try:
            dataset = ds.dataset(parquet_paths, format="parquet")
            all_columns = dataset.schema.names
        except:
            dataset = ds.dataset(parquet_paths[0], format="parquet")
            all_columns = dataset.schema.names
        
        # Exclude metadata columns
        n_meta = self.io_cfg.n_metadata
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
        """Load data into memory with explicit cleanup"""
        batches = []
        batch_generator = None
        
        try:
            batch_generator = iter_batches(parquet_paths, columns, self.io_cfg.batch_rows)
            
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

    def _generate_final_cluster_info(
        self,
        final_columns: List[str],
        columns_p1: List[str],
        columns_p2: List[str],
        columns_p3: List[str],
        pass2_info: Dict,
        pass3_info: Dict,
        pass4_info: Dict
    ) -> Dict:
        """
        Generate integrated cluster info for final descriptors.

        This tracks which descriptors each final representative absorbed through
        Pass 2 (Spearman), Pass 3 (VIF), and Pass 4 (Nonlinear) filtering.

        Output format matches cluster_structure.json:
        {
            "metadata": {...},
            "statistics": {...},
            "descriptors": {
                "DESC_NAME": {
                    "cluster_size": N,
                    "is_representative": true,
                    "alternative_descriptors": [...],
                    "all_cluster_members": [...],
                    "removal_history": {
                        "pass2": [...],
                        "pass3": [...]
                    },
                    "total_alternatives": N
                },
                ...
            }
        }
        """
        # Build reverse mapping: removed_descriptor -> survivor
        # This tracks the chain of absorption through passes

        # Pass 2: removed -> representative (from Spearman clustering)
        pass2_removed = pass2_info.get('removed', {})

        # Pass 3: removed -> survivor (from VIF filtering)
        # Note: pass3 'removed' can be a list or a dict
        pass3_removed_raw = pass3_info.get('removed', {})
        pass3_removed = {}

        if isinstance(pass3_removed_raw, dict):
            pass3_removed = pass3_removed_raw
        else:
            # 'removed' is a list, need to build mapping from clusters or removed_details
            pass3_clusters = pass3_info.get('clusters', {})
            pass3_removed_details = pass3_info.get('removed_details', [])

            # Method 1: Use clusters which has 'survivor' field
            if isinstance(pass3_clusters, dict):
                for removed_desc, cluster_info in pass3_clusters.items():
                    survivor = cluster_info.get('survivor')
                    if survivor:
                        pass3_removed[removed_desc] = survivor
                        # Also add members to the mapping
                        members = cluster_info.get('members', [])
                        for member in members:
                            if member != removed_desc and member not in pass3_removed:
                                pass3_removed[member] = survivor

            # Method 2: Use removed_details which has 'assigned_to' or 'survivor' field
            for detail in pass3_removed_details:
                removed_desc = detail.get('descriptor')
                survivor = detail.get('assigned_to') or detail.get('survivor')
                if removed_desc and survivor:
                    pass3_removed[removed_desc] = survivor
                    # Also add pass2_cluster_members if available
                    pass2_members = detail.get('pass2_cluster_members', [])
                    for member in pass2_members:
                        if member not in pass3_removed:
                            pass3_removed[member] = survivor

        # Also check representatives structure (for absorbed_descriptors format)
        pass3_representatives = pass3_info.get('representatives', {})
        if isinstance(pass3_representatives, dict):
            for rep, rep_info in pass3_representatives.items():
                if isinstance(rep_info, dict):
                    absorbed = rep_info.get('absorbed_descriptors', [])
                    for absorbed_desc in absorbed:
                        if absorbed_desc not in pass3_removed:
                            pass3_removed[absorbed_desc] = rep

        # Pass 4: removed -> representative (from nonlinear clustering)
        # Note: pass4 'removed' can be a list or a dict
        pass4_removed_raw = pass4_info.get('removed', {})
        pass4_removed = {}
        if isinstance(pass4_removed_raw, dict):
            pass4_removed = pass4_removed_raw
        elif isinstance(pass4_removed_raw, list):
            # If it's a list, we need to find the mapping from clusters
            pass4_clusters = pass4_info.get('clusters', [])
            pass4_representatives = pass4_info.get('representatives', [])
            # Build mapping from cluster members to representative
            for i, cluster in enumerate(pass4_clusters):
                members = cluster.get('members', [])
                if len(members) > 0:
                    # First member is typically the representative
                    rep = pass4_representatives[i] if i < len(pass4_representatives) else members[0]
                    for member in members:
                        if member != rep:
                            pass4_removed[member] = rep

        # Chain absorption helper - trace A->B->C chains
        def trace_chain_to_final(removed_map, final_set, max_depth=20):
            """
            Trace which descriptors ultimately end up at each final descriptor.
            Returns: Dict[final_desc] -> set of absorbed descriptors
            """
            absorbed_by = {f: set() for f in final_set}

            # Build reverse: for each removed, find its final destination
            for removed in removed_map:
                current = removed
                visited = set()
                for _ in range(max_depth):
                    if current in visited:
                        break
                    visited.add(current)

                    if current in final_set:
                        absorbed_by[current].add(removed)
                        break

                    next_survivor = removed_map.get(current)
                    if next_survivor is None:
                        # Check if it maps to a final via another pass
                        break
                    current = next_survivor

            return absorbed_by

        # Get Pass 2 representatives info for tracking pass2 clusters
        pass2_representatives = pass2_info.get('representatives', {})

        # Build complete absorption chains
        final_set = set(final_columns)

        # Merge all removal mappings for chain tracing
        all_removed = {}
        all_removed.update(pass2_removed)
        all_removed.update(pass3_removed)
        all_removed.update(pass4_removed)

        # Track which pass removed each descriptor
        removed_in_pass2 = set(pass2_removed.keys())
        removed_in_pass3 = set(pass3_removed.keys())
        removed_in_pass4 = set(pass4_removed.keys())

        # Build descriptors dict in cluster_structure.json format
        descriptors_dict = {}
        cluster_sizes = []
        total_alternatives = 0

        for final_desc in final_columns:
            # Find all descriptors absorbed by this final descriptor
            all_absorbed = set()
            current_check = {final_desc}

            for _ in range(30):  # max depth
                new_absorbed = set()
                for removed, survivor in all_removed.items():
                    if survivor in current_check and removed not in all_absorbed:
                        new_absorbed.add(removed)
                        all_absorbed.add(removed)

                if not new_absorbed:
                    break
                current_check = new_absorbed

            # Determine removal history
            pass2_history = sorted([d for d in all_absorbed if d in removed_in_pass2])
            pass3_history = sorted([d for d in all_absorbed if d in removed_in_pass3])

            # All cluster members (including representative)
            all_members = sorted(list(all_absorbed | {final_desc}))
            alternative_descs = sorted(list(all_absorbed))

            cluster_size = len(all_members)
            cluster_sizes.append(cluster_size)
            total_alternatives += len(alternative_descs)

            descriptors_dict[final_desc] = {
                'cluster_size': cluster_size,
                'is_representative': True,
                'alternative_descriptors': alternative_descs,
                'all_cluster_members': all_members,
                'removal_history': {
                    'pass2': pass2_history,
                    'pass3': pass3_history
                },
                'total_alternatives': len(alternative_descs)
            }

        # Calculate cluster size distribution
        from collections import Counter
        size_distribution = dict(Counter(cluster_sizes))

        # Find largest clusters
        sorted_by_size = sorted(
            [(d, info['cluster_size'], info['total_alternatives'])
             for d, info in descriptors_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )
        largest_clusters = [
            {'descriptor': d, 'cluster_size': s, 'alternatives': a}
            for d, s, a in sorted_by_size[:20]
        ]

        # Calculate removal distribution
        pass2_only = 0
        pass3_only = 0
        both_passes = 0
        for desc, info in descriptors_dict.items():
            has_p2 = len(info['removal_history']['pass2']) > 0
            has_p3 = len(info['removal_history']['pass3']) > 0
            if has_p2 and has_p3:
                both_passes += 1
            elif has_p2:
                pass2_only += 1
            elif has_p3:
                pass3_only += 1

        # Count standalone vs with alternatives
        standalone = sum(1 for info in descriptors_dict.values() if info['total_alternatives'] == 0)
        with_alternatives = len(final_columns) - standalone

        result = {
            'metadata': {
                'description': 'Final cluster structure with complete Pass2 expansion - includes Pass2 clusters of both alternatives AND final representatives',
                'version': '6.0',
                'total_descriptors': len(final_columns),
                'descriptors_with_alternatives': with_alternatives,
                'standalone_descriptors': standalone,
                'total_alternative_descriptors': total_alternatives
            },
            'statistics': {
                'total_descriptors': len(final_columns),
                'descriptors_with_alternatives': with_alternatives,
                'standalone_descriptors': standalone,
                'total_alternative_descriptors': total_alternatives,
                'cluster_size_distribution': {str(k): v for k, v in sorted(size_distribution.items())},
                'removal_distribution': {
                    'pass2_only': pass2_only,
                    'pass3_only': pass3_only,
                    'both_passes': both_passes
                },
                'largest_clusters': largest_clusters
            },
            'descriptors': descriptors_dict
        }

        return result

    # ========================================
    # Independent Pass Execution Methods
    # ========================================
    
    def run_pass0(self):
        """
        Execute Pass 0: Sampling only
        
        Returns:
            Path to sampled parquet file
        """
        import glob
        
        self._log("="*70)
        self._log("Pass 0: Sampling")
        self._log("="*70)
        self._log(self.config.get_device_info())
        self._log(f"Output: {self.output_dir}")
        self._log("="*70)
        
        # Get input files
        parquet_paths = sorted(glob.glob(self.io_cfg.parquet_glob))
        if not parquet_paths:
            raise RuntimeError(f"No parquet files found: {self.io_cfg.parquet_glob}")
        self._log(f"\nInput files: {len(parquet_paths)}")
        
        # Get all columns (including metadata)
        all_columns = self._get_all_columns(parquet_paths)
        
        # Run Pass 0
        sampled_file_path = self.output_dir / "sampled_data.parquet"
        if self.system_cfg.checkpoint and sampled_file_path.exists():
            self._log("\n✓ Pass 0: Sampling already completed (using cached file)")
            return str(sampled_file_path)
        
        sampled_file = self.pass0.run(parquet_paths, all_columns, self.output_dir)
        self._cleanup_memory()
        
        self._log(f"\n✅ Pass 0 completed: {sampled_file}")
        return sampled_file
    
    def run_pass1(self):
        """
        Execute Pass 1: Statistics + Variance Filtering
        
        Automatically loads Pass 0 result if available
        
        Returns:
            Tuple of (filtered_columns, stats, indices)
        """
        import glob
        
        self._log("="*70)
        self._log("Pass 1: Statistics + Variance Filtering")
        self._log("="*70)
        self._log(self.config.get_device_info())
        self._log(f"Output: {self.output_dir}")
        self._log("="*70)
        
        # Determine input files (check for Pass 0 output first)
        sampled_file_path = self.output_dir / "sampled_data.parquet"
        if sampled_file_path.exists():
            self._log("\nUsing Pass 0 output")
            parquet_paths = [str(sampled_file_path)]
        else:
            parquet_paths = sorted(glob.glob(self.io_cfg.parquet_glob))
            if not parquet_paths:
                raise RuntimeError(f"No parquet files found: {self.io_cfg.parquet_glob}")
            self._log(f"\nUsing original input files: {len(parquet_paths)}")
        
        # Get descriptor columns
        self.columns = self._get_columns(parquet_paths)
        p_initial = len(self.columns)
        self._log(f"Descriptor columns (after metadata exclusion): {p_initial}")
        
        # Check for checkpoint
        pass1_checkpoint = self._load_checkpoint("pass1_variance_filtering.json")
        pass1_columns_file = self.output_dir / "pass1_columns.txt"
        pass1_stats_file = self.output_dir / "pass1_stats.npz"
        
        if (self.system_cfg.checkpoint and pass1_checkpoint is not None 
            and pass1_columns_file.exists() and pass1_stats_file.exists()):
            # Load checkpoint
            columns_p1 = self._load_intermediate_columns("pass1_columns.txt")
            stats_p1 = dict(np.load(str(pass1_stats_file), allow_pickle=True))
            indices_p1 = np.array([i for i, c in enumerate(self.columns) if c in columns_p1])
            
            # Validate checkpoint
            cached_set = set(columns_p1)
            current_set = set(self.columns)
            
            if not cached_set.issubset(current_set):
                self._log("\n⚠ Pass 1 checkpoint invalid, re-computing...")
                columns_p1, stats_p1, indices_p1 = self.pass1.compute(parquet_paths, self.columns)
                self._save_pass1_checkpoint(columns_p1, stats_p1, indices_p1, p_initial)
            else:
                self._log("\n✓ Pass 1: Loaded from checkpoint")
        else:
            columns_p1, stats_p1, indices_p1 = self.pass1.compute(parquet_paths, self.columns)
            self._save_pass1_checkpoint(columns_p1, stats_p1, indices_p1, p_initial)
        
        self.stats = stats_p1
        self._cleanup_memory()
        
        self._log(f"\n✅ Pass 1 completed: {len(columns_p1)} columns remaining (removed {p_initial - len(columns_p1)})")
        return columns_p1, stats_p1, indices_p1
    
    def run_pass234(self):
        """
        Execute Pass 2+3+4: Correlation + VIF + Nonlinear Filtering
        
        Automatically loads Pass 1 result
        
        Returns:
            Final filtered columns
        """
        self._log("="*70)
        self._log("Pass 2+3+4: Correlation + VIF + Nonlinear Filtering")
        self._log("="*70)
        self._log(self.config.get_device_info())
        self._log(f"Output: {self.output_dir}")
        self._log("="*70)
        
        # Load Pass 1 results
        pass1_columns_file = self.output_dir / "pass1_columns.txt"
        pass1_stats_file = self.output_dir / "pass1_stats.npz"
        
        if not (pass1_columns_file.exists() and pass1_stats_file.exists()):
            raise RuntimeError("Pass 1 results not found. Please run Pass 1 first with: cdml filter pass1")
        
        self._log(f"\nLoading Pass 1 results...")
        
        # This will use the full run() method but skip Pass 0 and Pass 1
        # since they're already completed and checkpointed
        result = self.run()
        
        return result['final_columns']
    
    def run_all(self):
        """
        Execute full pipeline (Pass 0 through Pass 4)
        
        This is an alias for run()
        """
        return self.run()
    
    def _save_pass1_checkpoint(self, columns_p1, stats_p1, indices_p1, p_initial):
        """Helper to save Pass 1 checkpoint"""
        if self.system_cfg.checkpoint:
            self._save_checkpoint("pass1_variance_filtering.json", {
                'removed_count': p_initial - len(columns_p1),
                'remaining_count': len(columns_p1),
                'removed_columns': [c for i, c in enumerate(self.columns) if i not in indices_p1]
            })
            self._save_intermediate_columns("pass1_columns.txt", columns_p1)
            # Save stats
            stats_to_save = {}
            for key, value in stats_p1.items():
                if isinstance(value, (list, tuple)):
                    stats_to_save[key] = np.array(value)
                else:
                    stats_to_save[key] = value
            np.savez(str(self.output_dir / "pass1_stats.npz"), **stats_to_save)

