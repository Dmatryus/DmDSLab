#!/usr/bin/env python3
"""
UCI Database Initialization Script (File-based)

This script manages UCI dataset metadata through JSON files instead of hardcoded data.
It provides a flexible system for adding, updating, and organizing dataset metadata.

Directory structure:
    datasets_metadata/
    ├── predefined/          # Manually curated datasets
    │   ├── mushroom_73.json
    │   ├── adult_2.json
    │   └── ...
    ├── discovered/          # Auto-discovered datasets
    │   └── dataset_XXX.json
    ├── categories.json      # Dataset categories/collections
    └── templates/           # Templates for new datasets
        └── dataset_template.json

Usage:
    # Initialize database from metadata files
    python initialize_uci_database.py
    
    # Process unknown datasets and generate metadata files
    python initialize_uci_database.py --process-unknown
    
    # Generate metadata file for specific dataset
    python initialize_uci_database.py --generate-metadata 186
    
    # Validate all metadata files
    python initialize_uci_database.py --validate
    
    # Show statistics
    python initialize_uci_database.py --stats

Author: DmDSLab Team
License: Apache 2.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from dmdslab.datasets.uci_dataset_manager import (
        DatasetInfo,
        Domain,
        TaskType,
        UCIDatasetManager,
    )
except ImportError:
    logger.error("DmDSLab not installed. Please install it first.")
    sys.exit(1)


class MetadataManager:
    """Manages dataset metadata files."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize metadata manager.
        
        Args:
            base_path: Base path for metadata files. If None, uses default.
        """
        if base_path is None:
            # Default to datasets_metadata/ in the same directory as this script
            self.base_path = Path(__file__).parent / "datasets_metadata"
        else:
            self.base_path = Path(base_path)
        
        # Create directory structure if it doesn't exist
        self.predefined_dir = self.base_path / "predefined"
        self.discovered_dir = self.base_path / "discovered"
        self.templates_dir = self.base_path / "templates"
        
        for dir_path in [self.predefined_dir, self.discovered_dir, self.templates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create default template if it doesn't exist
        self._create_default_template()
    
    def _create_default_template(self):
        """Create default dataset template."""
        template_path = self.templates_dir / "dataset_template.json"
        if not template_path.exists():
            template = {
                "id": 0,
                "name": "Dataset Name",
                "url": "https://archive.ics.uci.edu/dataset/{id}/",
                "n_instances": 0,
                "n_features": 0,
                "task_type": "binary_classification",
                "domain": "unknown",
                "description": "Dataset description",
                "year": None,
                "has_missing_values": False,
                "is_imbalanced": False,
                "imbalance_ratio": None,
                "class_balance": None,
                "feature_names": None,
                "target_name": None,
                "tags": [],
                "metadata": {}
            }
            
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2)
            
            logger.info(f"Created template: {template_path}")
    
    def load_dataset_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load dataset metadata from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with metadata or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def save_dataset_metadata(self, metadata: Dict[str, Any], file_path: Path):
        """Save dataset metadata to JSON file.
        
        Args:
            metadata: Dataset metadata dictionary
            file_path: Path to save the file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
            raise
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate dataset metadata.
        
        Args:
            metadata: Dataset metadata dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = ['id', 'name', 'url', 'n_instances', 'n_features', 
                          'task_type', 'domain']
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
        
        # Validate ID
        if 'id' in metadata:
            if not isinstance(metadata['id'], int) or metadata['id'] <= 0:
                errors.append("ID must be a positive integer")
        
        # Validate task_type
        if 'task_type' in metadata:
            valid_task_types = [t.value for t in TaskType]
            if metadata['task_type'] not in valid_task_types:
                errors.append(f"Invalid task_type. Valid values: {valid_task_types}")
        
        # Validate domain
        if 'domain' in metadata:
            valid_domains = [d.value for d in Domain]
            if metadata['domain'] not in valid_domains:
                errors.append(f"Invalid domain. Valid values: {valid_domains}")
        
        # Validate numeric fields
        numeric_fields = ['n_instances', 'n_features']
        for field in numeric_fields:
            if field in metadata:
                if not isinstance(metadata[field], int) or metadata[field] < 0:
                    errors.append(f"{field} must be a non-negative integer")
        
        # Validate imbalance ratio
        if metadata.get('is_imbalanced') and 'imbalance_ratio' in metadata:
            if metadata['imbalance_ratio'] is not None:
                if not isinstance(metadata['imbalance_ratio'], (int, float)) or metadata['imbalance_ratio'] <= 1:
                    errors.append("imbalance_ratio must be a number greater than 1")
        
        return len(errors) == 0, errors
    
    def get_all_metadata_files(self) -> Dict[str, List[Path]]:
        """Get all metadata files organized by directory.
        
        Returns:
            Dictionary with keys 'predefined' and 'discovered' containing file paths
        """
        files = {
            'predefined': list(self.predefined_dir.glob('*.json')),
            'discovered': list(self.discovered_dir.glob('*.json'))
        }
        return files
    
    def create_metadata_from_unknown(self, dataset_id: int, 
                                   manager: UCIDatasetManager) -> Optional[Dict[str, Any]]:
        """Create metadata by fetching unknown dataset.
        
        Args:
            dataset_id: UCI dataset ID
            manager: UCIDatasetManager instance
            
        Returns:
            Metadata dictionary or None if failed
        """
        try:
            logger.info(f"Fetching dataset {dataset_id} from UCI...")
            
            # Load dataset
            model_data = manager.load_dataset(dataset_id, allow_unknown=True)
            
            # Extract basic info
            metadata = {
                "id": dataset_id,
                "name": model_data.info.name,
                "url": f"https://archive.ics.uci.edu/dataset/{dataset_id}/",
                "n_instances": model_data.n_samples,
                "n_features": model_data.n_features,
                "description": f"Auto-discovered dataset. Please update this description.",
                "auto_generated": True,
                "discovered_date": datetime.now().isoformat(),
                "needs_review": True,
                "tags": ["auto-discovered", "needs-review"]
            }
            
            # Try to determine task type
            if model_data.target is None:
                metadata["task_type"] = "clustering"
            else:
                import numpy as np
                unique_values = len(np.unique(model_data.target))
                if unique_values == 2:
                    metadata["task_type"] = "binary_classification"
                elif unique_values < 20:
                    metadata["task_type"] = "multiclass_classification"
                else:
                    metadata["task_type"] = "regression"
            
            # Set domain as unknown
            metadata["domain"] = "unknown"
            
            # Add feature names if available
            if model_data.feature_names:
                metadata["feature_names"] = model_data.feature_names
            
            # Check for missing values
            if hasattr(model_data.features, 'isnull'):
                metadata["has_missing_values"] = model_data.features.isnull().any().any()
            else:
                metadata["has_missing_values"] = False
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch dataset {dataset_id}: {e}")
            return None


def load_metadata_to_database(metadata_manager: MetadataManager, 
                            db_manager: UCIDatasetManager,
                            validate: bool = True) -> Tuple[int, int]:
    """Load all metadata files into the database.
    
    Args:
        metadata_manager: MetadataManager instance
        db_manager: UCIDatasetManager instance
        validate: Whether to validate metadata before loading
        
    Returns:
        Tuple of (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    # Get all metadata files
    all_files = metadata_manager.get_all_metadata_files()
    
    # Process predefined datasets first
    logger.info(f"\nLoading predefined datasets...")
    for file_path in sorted(all_files['predefined']):
        logger.info(f"Processing {file_path.name}...")
        
        metadata = metadata_manager.load_dataset_metadata(file_path)
        if metadata is None:
            error_count += 1
            continue
        
        # Validate if requested
        if validate:
            is_valid, errors = metadata_manager.validate_metadata(metadata)
            if not is_valid:
                logger.error(f"Validation failed for {file_path.name}:")
                for error in errors:
                    logger.error(f"  - {error}")
                error_count += 1
                continue
        
        # Convert to DatasetInfo
        try:
            dataset_info = DatasetInfo(
                id=metadata['id'],
                name=metadata['name'],
                url=metadata['url'],
                n_instances=metadata['n_instances'],
                n_features=metadata['n_features'],
                task_type=TaskType(metadata['task_type']),
                domain=Domain(metadata['domain']),
                class_balance=metadata.get('class_balance'),
                description=metadata.get('description', ''),
                year=metadata.get('year'),
                has_missing_values=metadata.get('has_missing_values', False),
                is_imbalanced=metadata.get('is_imbalanced', False),
                imbalance_ratio=metadata.get('imbalance_ratio'),
                feature_names=metadata.get('feature_names'),
                target_name=metadata.get('target_name')
            )
            
            # Add to database
            db_manager.add_dataset(dataset_info)
            success_count += 1
            logger.info(f"  ✓ Added: {dataset_info.name} (ID: {dataset_info.id})")
            
        except Exception as e:
            logger.error(f"Failed to add {file_path.name}: {e}")
            error_count += 1
    
    # Process discovered datasets
    if all_files['discovered']:
        logger.info(f"\nLoading discovered datasets...")
        for file_path in sorted(all_files['discovered']):
            # Same process as above
            metadata = metadata_manager.load_dataset_metadata(file_path)
            if metadata is None:
                error_count += 1
                continue
            
            try:
                dataset_info = DatasetInfo(
                    id=metadata['id'],
                    name=metadata['name'],
                    url=metadata['url'],
                    n_instances=metadata['n_instances'],
                    n_features=metadata['n_features'],
                    task_type=TaskType(metadata['task_type']),
                    domain=Domain(metadata.get('domain', 'unknown')),
                    description=metadata.get('description', ''),
                    has_missing_values=metadata.get('has_missing_values', False)
                )
                
                db_manager.add_dataset(dataset_info)
                success_count += 1
                logger.info(f"  ✓ Added discovered: {dataset_info.name} (ID: {dataset_info.id})")
                
            except Exception as e:
                logger.error(f"Failed to add {file_path.name}: {e}")
                error_count += 1
    
    return success_count, error_count


def process_unknown_datasets(metadata_manager: MetadataManager,
                           db_manager: UCIDatasetManager,
                           interactive: bool = False):
    """Process unknown datasets and generate metadata files.
    
    Args:
        metadata_manager: MetadataManager instance
        db_manager: UCIDatasetManager instance
        interactive: Whether to run in interactive mode
    """
    # Load unknown IDs
    unknown_info = db_manager._load_unknown_ids()
    
    if not unknown_info:
        logger.info("No unknown datasets found.")
        return
    
    logger.info(f"\nFound {len(unknown_info)} unknown dataset IDs")
    
    generated_count = 0
    failed_count = 0
    
    for dataset_id_str, info in unknown_info.items():
        dataset_id = int(dataset_id_str)
        
        # Check if metadata file already exists
        existing_file = metadata_manager.discovered_dir / f"dataset_{dataset_id}.json"
        if existing_file.exists():
            logger.info(f"Metadata file already exists for dataset {dataset_id}")
            continue
        
        logger.info(f"\nProcessing dataset {dataset_id}...")
        logger.info(f"  Load count: {info.get('load_count', 0)}")
        
        if interactive:
            response = input("Generate metadata file? (y/n/s[kip all]): ").lower()
            if response == 's':
                break
            elif response != 'y':
                continue
        
        # Generate metadata
        metadata = metadata_manager.create_metadata_from_unknown(dataset_id, db_manager)
        
        if metadata:
            # Save to file
            output_path = metadata_manager.discovered_dir / f"dataset_{dataset_id}.json"
            metadata_manager.save_dataset_metadata(metadata, output_path)
            generated_count += 1
            
            logger.info(f"  ✓ Generated metadata file: {output_path.name}")
            
            if interactive:
                edit = input("Edit metadata file now? (y/n): ").lower()
                if edit == 'y':
                    import subprocess
                    import os
                    editor = os.environ.get('EDITOR', 'nano')
                    subprocess.call([editor, str(output_path)])
        else:
            failed_count += 1
            logger.error(f"  ✗ Failed to generate metadata for dataset {dataset_id}")
    
    logger.info(f"\nSummary: {generated_count} generated, {failed_count} failed")


def show_statistics(metadata_manager: MetadataManager, db_manager: UCIDatasetManager):
    """Show statistics about metadata files and database.
    
    Args:
        metadata_manager: MetadataManager instance
        db_manager: UCIDatasetManager instance
    """
    # File statistics
    all_files = metadata_manager.get_all_metadata_files()
    
    print("\n" + "=" * 60)
    print("METADATA FILES STATISTICS")
    print("=" * 60)
    
    print(f"\nMetadata directory: {metadata_manager.base_path}")
    print(f"Predefined datasets: {len(all_files['predefined'])}")
    print(f"Discovered datasets: {len(all_files['discovered'])}")
    
    # List files by category
    if all_files['predefined']:
        print("\nPredefined datasets:")
        for f in sorted(all_files['predefined'])[:10]:
            print(f"  - {f.name}")
        if len(all_files['predefined']) > 10:
            print(f"  ... and {len(all_files['predefined']) - 10} more")
    
    if all_files['discovered']:
        print("\nDiscovered datasets:")
        for f in sorted(all_files['discovered'])[:10]:
            print(f"  - {f.name}")
        if len(all_files['discovered']) > 10:
            print(f"  ... and {len(all_files['discovered']) - 10} more")
    
    # Database statistics
    stats = db_manager.get_statistics()
    
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal datasets in database: {stats['total_datasets']}")
    
    if stats['by_task_type']:
        print("\nBy task type:")
        for task_type, count in sorted(stats['by_task_type'].items()):
            print(f"  - {task_type}: {count}")
    
    if stats['by_domain']:
        print("\nTop domains:")
        for domain, count in sorted(stats['by_domain'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {domain}: {count}")
    
    print(f"\nImbalanced datasets: {stats['imbalanced_datasets']}")
    print(f"Average dataset size: {stats['avg_instances']} instances × {stats['avg_features']} features")
    
    # Unknown datasets
    if stats['unknown_datasets_requested'] > 0:
        print(f"\nUnknown datasets requested: {stats['unknown_datasets_requested']}")
        print(f"Unknown IDs: {stats['unknown_ids'][:10]}")
        if len(stats['unknown_ids']) > 10:
            print(f"  ... and {len(stats['unknown_ids']) - 10} more")


def validate_all_files(metadata_manager: MetadataManager):
    """Validate all metadata files.
    
    Args:
        metadata_manager: MetadataManager instance
    """
    all_files = metadata_manager.get_all_metadata_files()
    total_files = len(all_files['predefined']) + len(all_files['discovered'])
    
    logger.info(f"\nValidating {total_files} metadata files...")
    
    valid_count = 0
    invalid_count = 0
    
    for category, files in all_files.items():
        if files:
            logger.info(f"\nValidating {category} datasets...")
            
            for file_path in sorted(files):
                metadata = metadata_manager.load_dataset_metadata(file_path)
                
                if metadata is None:
                    invalid_count += 1
                    continue
                
                is_valid, errors = metadata_manager.validate_metadata(metadata)
                
                if is_valid:
                    valid_count += 1
                    logger.info(f"  ✓ {file_path.name}")
                else:
                    invalid_count += 1
                    logger.error(f"  ✗ {file_path.name}")
                    for error in errors:
                        logger.error(f"    - {error}")
    
    logger.info(f"\nValidation complete: {valid_count} valid, {invalid_count} invalid")


def generate_single_metadata(metadata_manager: MetadataManager,
                           db_manager: UCIDatasetManager,
                           dataset_id: int,
                           force: bool = False):
    """Generate metadata file for a single dataset.
    
    Args:
        metadata_manager: MetadataManager instance
        db_manager: UCIDatasetManager instance
        dataset_id: UCI dataset ID
        force: Whether to overwrite existing file
    """
    output_path = metadata_manager.discovered_dir / f"dataset_{dataset_id}.json"
    
    if output_path.exists() and not force:
        logger.warning(f"Metadata file already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check if already in database
    existing = db_manager.get_dataset_info(dataset_id)
    if existing:
        logger.info(f"Dataset {dataset_id} already in database: {existing.name}")
        
        # Convert to metadata dict
        metadata = existing.to_dict()
        metadata['tags'] = ['from-database']
        metadata['exported_date'] = datetime.now().isoformat()
        
        # Save to discovered directory
        metadata_manager.save_dataset_metadata(metadata, output_path)
        logger.info(f"Exported metadata to: {output_path}")
    else:
        # Generate from UCI
        logger.info(f"Fetching dataset {dataset_id} from UCI...")
        metadata = metadata_manager.create_metadata_from_unknown(dataset_id, db_manager)
        
        if metadata:
            metadata_manager.save_dataset_metadata(metadata, output_path)
            logger.info(f"Generated metadata file: {output_path}")
        else:
            logger.error(f"Failed to generate metadata for dataset {dataset_id}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="UCI Database Initialization (File-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database from metadata files
  python initialize_uci_database.py
  
  # Process unknown datasets
  python initialize_uci_database.py --process-unknown
  
  # Generate metadata for specific dataset
  python initialize_uci_database.py --generate-metadata 186
  
  # Validate all metadata files
  python initialize_uci_database.py --validate
  
  # Show statistics
  python initialize_uci_database.py --stats
        """
    )
    
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Base directory for metadata files (default: ./datasets_metadata)"
    )
    
    parser.add_argument(
        "--process-unknown",
        action="store_true",
        help="Process unknown datasets and generate metadata files"
    )
    
    parser.add_argument(
        "--generate-metadata",
        type=int,
        metavar="ID",
        help="Generate metadata file for specific dataset ID"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all metadata files"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about metadata files and database"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation when loading metadata"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear database before loading"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize managers
        metadata_manager = MetadataManager(args.metadata_dir)
        db_manager = UCIDatasetManager()
        
        # Handle different commands
        if args.validate:
            validate_all_files(metadata_manager)
        
        elif args.stats:
            show_statistics(metadata_manager, db_manager)
        
        elif args.generate_metadata:
            generate_single_metadata(
                metadata_manager, 
                db_manager, 
                args.generate_metadata,
                force=args.force
            )
        
        elif args.process_unknown:
            process_unknown_datasets(
                metadata_manager,
                db_manager,
                interactive=args.interactive
            )
        
        else:
            # Default: load metadata to database
            logger.info("UCI Database Initialization")
            logger.info("=" * 50)
            
            # Check existing data
            stats = db_manager.get_statistics()
            if stats["total_datasets"] > 0 and not args.clear:
                logger.info(f"Database contains {stats['total_datasets']} datasets")
                response = input("Clear and reload? (y/N): ")
                if response.lower() == 'y':
                    count = db_manager.delete_all_datasets()
                    logger.info(f"Cleared {count} datasets")
                else:
                    logger.info("Keeping existing data. Loading new datasets...")
            elif args.clear:
                count = db_manager.delete_all_datasets()
                logger.info(f"Cleared {count} datasets")
            
            # Load metadata files
            success, errors = load_metadata_to_database(
                metadata_manager,
                db_manager,
                validate=not args.no_validation
            )
            
            logger.info(f"\nComplete: {success} loaded, {errors} errors")
            
            # Show final statistics
            final_stats = db_manager.get_statistics()
            logger.info(f"\nDatabase now contains {final_stats['total_datasets']} datasets")
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
