#!/usr/bin/env python3
"""
Example usage of Molecular Descriptor Toolkit
==============================================

This script demonstrates the basic usage of MDT for both
API and CLI approaches.
"""

from Chem_Descriptor_ML.config import (
    Config,
    IOConfig,
    DeviceConfig,
    FilteringConfig,
    SystemConfig,
)
from Chem_Descriptor_ML.filtering import DescriptorPipeline


def example_full_pipeline():
    """Run full pipeline with default GPU acceleration"""

    print("="*70)
    print("Example 1: Full Pipeline (GPU accelerated)")
    print("="*70)

    config = Config(
        io=IOConfig(
            parquet_glob="data/*.parquet",
            output_dir="results/full_pipeline",
        ),
        device=DeviceConfig(
            prefer_gpu=True,  # GPU by default
        ),
        filtering=FilteringConfig(
            sample_per_file=100000,
            variance_threshold=0.002,
            spearman_threshold=0.95,
            vif_threshold=10.0,
            nonlinear_threshold=0.3,
        ),
        system=SystemConfig(
            checkpoint=True,
            verbose=True,
        ),
    )

    # Validate and finalize (auto-detect device)
    config.validate_and_finalize()

    # Device info
    print(f"\n{config.get_device_info()}")
    print(f"Checkpoint: {'Enabled' if config.system.checkpoint else 'Disabled'}")

    # Run pipeline
    pipeline = DescriptorPipeline(config)
    result = pipeline.run()

    print(f"\n✅ Pipeline completed!")
    print(f"📊 Initial descriptors: {result['initial_count']}")
    print(f"📊 Final descriptors: {len(result['final_columns'])}")
    print(f"📁 Results: {config.io.output_dir}")


def example_step_by_step():
    """Run pipeline step by step"""

    print("\n" + "="*70)
    print("Example 2: Step-by-Step Execution")
    print("="*70)

    config = Config(
        io=IOConfig(
            parquet_glob="data/*.parquet",
            output_dir="results/step_by_step",
        ),
        device=DeviceConfig(
            prefer_gpu=True,
        ),
        system=SystemConfig(
            checkpoint=True,
            verbose=True,
        ),
    )

    # Validate and finalize
    config.validate_and_finalize()

    pipeline = DescriptorPipeline(config)

    # Pass 0: Sampling
    print("\n📊 Running Pass 0: Sampling")
    sampled_file = pipeline.run_pass0()
    print(f"✅ Sampled data: {sampled_file}")

    # Pass 1: Statistics + Variance
    print("\n📊 Running Pass 1: Statistics + Variance Filtering")
    columns_p1, stats_p1, indices_p1 = pipeline.run_pass1()
    print(f"✅ Pass 1: {len(columns_p1)} columns remaining")

    # Pass 2+3+4: Correlation + VIF + Nonlinear
    print("\n📊 Running Pass 2+3+4: Advanced Filtering")
    final_columns = pipeline.run_pass234()
    print(f"✅ Final: {len(final_columns)} descriptors")


def example_cpu_mode():
    """Run in CPU mode (no GPU)"""

    print("\n" + "="*70)
    print("Example 3: CPU Mode")
    print("="*70)

    config = Config(
        io=IOConfig(
            parquet_glob="data/*.parquet",
            output_dir="results/cpu_mode",
        ),
        device=DeviceConfig(
            prefer_gpu=False,  # Force CPU
        ),
        system=SystemConfig(
            checkpoint=True,
            verbose=True,
        ),
    )

    # Validate and finalize
    config.validate_and_finalize()

    print(f"\n{config.get_device_info()}")

    pipeline = DescriptorPipeline(config)
    result = pipeline.run()

    print(f"\n✅ Pipeline completed in CPU mode!")
    print(f"📊 Final descriptors: {len(result['final_columns'])}")


def example_custom_parameters():
    """Run with custom filtering parameters"""

    print("\n" + "="*70)
    print("Example 4: Custom Parameters")
    print("="*70)

    config = Config(
        io=IOConfig(
            parquet_glob="data/*.parquet",
            output_dir="results/custom_params",
        ),
        device=DeviceConfig(
            prefer_gpu=True,
        ),
        filtering=FilteringConfig(
            # Custom filtering parameters
            variance_threshold=0.001,  # More lenient
            max_missing_ratio=0.3,     # Stricter on missing data
            spearman_threshold=0.90,   # More lenient correlation
            vif_threshold=5.0,         # Stricter VIF
            nonlinear_threshold=0.2,   # More lenient nonlinear
        ),
        system=SystemConfig(
            checkpoint=True,
            verbose=True,
        ),
    )

    # Validate and finalize
    config.validate_and_finalize()

    print(f"\n{config.get_device_info()}")
    print(f"Custom parameters:")
    print(f"  - Variance threshold: {config.filtering.variance_threshold}")
    print(f"  - Missing ratio: {config.filtering.max_missing_ratio}")
    print(f"  - Spearman threshold: {config.filtering.spearman_threshold}")
    print(f"  - VIF threshold: {config.filtering.vif_threshold}")
    print(f"  - Nonlinear threshold: {config.filtering.nonlinear_threshold}")

    pipeline = DescriptorPipeline(config)
    result = pipeline.run()

    print(f"\n✅ Pipeline completed with custom parameters!")
    print(f"📊 Final descriptors: {len(result['final_columns'])}")


if __name__ == "__main__":
    import sys

    print("🧬 Molecular Descriptor Toolkit - Examples")
    print("="*70)

    # Check if data exists
    import glob
    if not glob.glob("data/*.parquet"):
        print("\n⚠️  Warning: No parquet files found in 'data/' directory")
        print("Please prepare your data first using:")
        print("  mdt preprocess xml-to-parquet ...")
        print("  mdt preprocess calculate-descriptors ...")
        sys.exit(1)

    # Run examples
    try:
        # Example 1: Full pipeline
        example_full_pipeline()

        # Example 2: Step by step
        example_step_by_step()

        # Example 3: CPU mode
        example_cpu_mode()

        # Example 4: Custom parameters
        example_custom_parameters()

        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
