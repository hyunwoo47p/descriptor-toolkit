"""Filtering IO module - Lazy imports"""

__all__ = [
    "iter_batches",
    "create_sampled_parquet_file",
]

def __getattr__(name):
    if name == "iter_batches":
        from Chem_Descriptor_ML.filtering.io.parquet_reader import iter_batches
        return iter_batches
    elif name == "create_sampled_parquet_file":
        from Chem_Descriptor_ML.filtering.io.parquet_reader import create_sampled_parquet_file
        return create_sampled_parquet_file
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
