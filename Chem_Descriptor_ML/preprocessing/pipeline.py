"""
ChemDescriptorML (CDML) - Preprocessing Pipeline

Config-based interface to preprocessing modules:
- XML Parser: PubChem XML → Parquet conversion
- Schema Generator: Auto-detect descriptor schema
- Descriptor Calculator: RDKit + Mordred descriptor calculation
"""

from pathlib import Path
from typing import List, Optional
import logging

from Chem_Descriptor_ML.config.settings import (
    PreprocessingConfig,
    DescriptorConfig,
    IOConfig,
    SystemConfig
)


class PreprocessingPipeline:
    """
    Preprocessing pipeline wrapper that uses hierarchical Config
    
    This wraps the existing preprocessing modules (xml_parser, descriptor_calculator)
    and provides a Config-based interface.
    """
    
    def __init__(
        self,
        preprocessing_cfg: PreprocessingConfig,
        descriptor_cfg: DescriptorConfig,
        io_cfg: IOConfig,
        system_cfg: SystemConfig
    ):
        self.preprocessing_cfg = preprocessing_cfg
        self.descriptor_cfg = descriptor_cfg
        self.io_cfg = io_cfg
        self.system_cfg = system_cfg
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger based on SystemConfig"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.system_cfg.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def parse_xml_to_csv(
        self,
        xml_files: List[str],
        output_dir: str,
        columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Parse PubChem XML files to CSV
        
        Args:
            xml_files: List of XML file paths (.xml or .xml.gz)
            output_dir: Output directory for CSV files
            columns: Columns to extract (None = default columns)
        
        Returns:
            List of output CSV file paths
        """
        from Chem_Descriptor_ML.preprocessing import xml_parser
        
        self.logger.info(f"Parsing {len(xml_files)} XML files to CSV...")
        
        # Use xml_parser module with appropriate settings
        # This would call the actual xml_parser functions
        # For now, just a placeholder
        
        self.logger.warning("XML parsing not yet implemented in wrapper")
        return []
    
    def calculate_descriptors(
        self,
        input_files: List[str],
        output_parquet: str
    ) -> str:
        """
        Calculate molecular descriptors from input files
        
        Args:
            input_files: Input CSV/Parquet files with molecules
            output_parquet: Output Parquet file path
        
        Returns:
            Output Parquet file path
        """
        from Chem_Descriptor_ML.preprocessing import descriptor_calculator
        
        self.logger.info(f"Calculating descriptors for {len(input_files)} files...")
        
        # Build configuration dict from Config objects
        calc_config = {
            # Preprocessing settings
            'std_core': self.preprocessing_cfg.std_core,
            'use_normalizer': self.preprocessing_cfg.use_normalizer,
            'use_reionizer': self.preprocessing_cfg.use_reionizer,
            'metal_disconnector': self.preprocessing_cfg.use_metal_disconnector,
            'largest_fragment': self.preprocessing_cfg.keep_largest_fragment,
            
            # Descriptor settings
            'descriptor_set': self.descriptor_cfg.descriptor_set,
            'descriptor_include': self.descriptor_cfg.descriptor_include,
            'descriptor_exclude': self.descriptor_cfg.descriptor_exclude,
            'timeout': self.descriptor_cfg.per_molecule_timeout_sec,
            'workers': self.descriptor_cfg.get_workers(),
            
            # IO settings
            'output_dir': self.io_cfg.output_dir,
            'row_group_size': self.io_cfg.row_group_size,
            
            # System settings
            'random_seed': self.system_cfg.random_seed,
            'verbose': self.system_cfg.verbose,
        }
        
        self.logger.info(f"Config: {calc_config}")
        self.logger.warning("Descriptor calculation not yet fully implemented in wrapper")
        
        return output_parquet
    
    def run_full_pipeline(
        self,
        xml_files: List[str],
        output_parquet: str
    ) -> str:
        """
        Run complete preprocessing pipeline: XML → CSV → Descriptors
        
        Args:
            xml_files: Input PubChem XML files
            output_parquet: Final output Parquet file
        
        Returns:
            Output Parquet file path
        """
        self.logger.info("Running full preprocessing pipeline...")
        
        # Step 1: Parse XML to CSV
        temp_dir = Path(self.io_cfg.output_dir) / "temp_csv"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        csv_files = self.parse_xml_to_csv(
            xml_files=xml_files,
            output_dir=str(temp_dir)
        )
        
        # Step 2: Calculate descriptors
        result = self.calculate_descriptors(
            input_files=csv_files,
            output_parquet=output_parquet
        )
        
        self.logger.info(f"Pipeline complete: {result}")
        return result


# Convenience function
def run_preprocessing(
    preprocessing_cfg: PreprocessingConfig,
    descriptor_cfg: DescriptorConfig,
    io_cfg: IOConfig,
    system_cfg: SystemConfig,
    xml_files: List[str],
    output_parquet: str
) -> str:
    """
    Convenience function to run preprocessing pipeline
    
    Example:
        from Chem_Descriptor_ML.config import load_config
        from Chem_Descriptor_ML.preprocessing.pipeline import run_preprocessing
        
        config = load_config("settings.yaml")
        result = run_preprocessing(
            preprocessing_cfg=config.preprocessing,
            descriptor_cfg=config.descriptor,
            io_cfg=config.io,
            system_cfg=config.system,
            xml_files=["compound1.xml.gz", "compound2.xml.gz"],
            output_parquet="descriptors.parquet"
        )
    """
    pipeline = PreprocessingPipeline(
        preprocessing_cfg=preprocessing_cfg,
        descriptor_cfg=descriptor_cfg,
        io_cfg=io_cfg,
        system_cfg=system_cfg
    )
    
    return pipeline.run_full_pipeline(xml_files, output_parquet)
