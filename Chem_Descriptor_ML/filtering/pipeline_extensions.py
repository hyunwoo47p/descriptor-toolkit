"""
Extension methods for DescriptorPipeline to enable independent pass execution
These should be added to the DescriptorPipeline class in pipeline.py
"""

def run_pass0(self):
    """
    Execute Pass 0: Sampling only
    
    Returns:
        Path to sampled parquet file
    """
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
        raise RuntimeError("Pass 1 results not found. Please run Pass 1 first.")
    
    columns_p1 = self._load_intermediate_columns("pass1_columns.txt")
    stats_p1 = dict(np.load(str(pass1_stats_file), allow_pickle=True))
    
    self._log(f"\nLoaded Pass 1 results: {len(columns_p1)} columns")
    
    # Determine input files
    sampled_file_path = self.output_dir / "sampled_data.parquet"
    if sampled_file_path.exists():
        parquet_paths = [str(sampled_file_path)]
    else:
        parquet_paths = sorted(glob.glob(self.io_cfg.parquet_glob))
    
    # Run Pass 2, 3, 4 (extracted from main run() method)
    # This is a simplified version - the full implementation would include
    # all the checkpoint logic from the original run() method
    
    # For now, call the main run() method and extract Pass 2-4
    # A full refactoring would separate this logic more cleanly
    result = self.run()
    
    return result['final_columns']


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


def run_all(self):
    """
    Execute full pipeline (Pass 0 through Pass 4)
    
    This is the original run() method
    """
    return self.run()
