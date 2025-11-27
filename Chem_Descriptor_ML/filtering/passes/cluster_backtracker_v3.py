"""Cluster Backtracking v3.0 - Proper Recursive Tracking"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class ImprovedClusterBacktrackerV3:
    def __init__(self, output_dir: Path, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.checkpoint_files = {
            'pass1': self.output_dir / 'pass1_variance_filtering.json',
            'pass2': self.output_dir / 'pass2_spearman.json',
            'pass3': self.output_dir / 'pass3_vif.json',
            'pass4': self.output_dir / 'pass4_nonlinear.json',
        }
        self.checkpoints = {}
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def load_checkpoints(self):
        self._log("\n📂 Loading checkpoints...")
        for pass_name, file_path in self.checkpoint_files.items():
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.checkpoints[pass_name] = json.load(f)
                self._log(f"  ✓ {pass_name}")
            else:
                self.checkpoints[pass_name] = None
    
    def _extract_cluster_info_pass2(self, checkpoint: Dict) -> Dict[str, Set[str]]:
        """Extract Pass2 cluster information: representative -> removed members"""
        cluster_map = {}
        
        if 'clusters' in checkpoint:
            clusters = checkpoint['clusters']
            if isinstance(clusters, list):
                for cluster in clusters:
                    rep = cluster.get('representative')
                    members = cluster.get('members', [])
                    if rep and members:
                        # Members include representative, so remove it
                        removed = set(members) - {rep}
                        if removed:
                            cluster_map[rep] = removed
            else:
                for rep, info in clusters.items():
                    members = info.get('members', [])
                    if members:
                        removed = set(members) - {rep}
                        if removed:
                            cluster_map[rep] = removed
        
        if 'representatives' in checkpoint and not cluster_map:
            for rep, info in checkpoint['representatives'].items():
                members = info.get('members', [])
                if members:
                    removed = set(members) - {rep}
                    if removed:
                        cluster_map[rep] = removed
        
        return cluster_map
    
    def _extract_cluster_info_pass3(self, checkpoint: Dict) -> Dict[str, Set[str]]:
        """Extract Pass3 cluster information: surviving -> removed members"""
        cluster_map = {}
        
        if 'clusters' in checkpoint:
            clusters = checkpoint['clusters']
            if isinstance(clusters, dict):
                for removed_desc, info in clusters.items():
                    members = info.get('members', [])
                    if members:
                        cluster_map[removed_desc] = set(members)
        
        if 'representatives' in checkpoint and not cluster_map:
            representatives = checkpoint['representatives']
            if isinstance(representatives, dict):
                for removed_desc, info in representatives.items():
                    members = info.get('members', [])
                    if members:
                        cluster_map[removed_desc] = set(members)
        
        if 'removed_details' in checkpoint and not cluster_map:
            for detail in checkpoint['removed_details']:
                removed_desc = detail['descriptor']
                correlated = detail.get('correlated_descriptors', [])
                members = [removed_desc] + correlated
                cluster_map[removed_desc] = set(members)
        
        return cluster_map
    
    def _build_pass2_structure(self) -> Dict[str, Dict]:
        """
        Build Pass2 cluster structure.
        Returns: {representative: {'alternatives': [...], 'cluster_size': N}}
        """
        self._log("\n🔍 Building Pass2 structure...")
        
        cluster_map = self._extract_cluster_info_pass2(self.checkpoints.get('pass2', {}))
        
        pass2_structure = {}
        for rep, removed in cluster_map.items():
            pass2_structure[rep] = {
                'alternative_descriptors': sorted(removed),
                'cluster_size': len(removed) + 1,  # +1 for representative itself
                'removal_history': {
                    'pass2': sorted(removed)
                }
            }
            self._log(f"  Pass2: {rep} -> {len(removed)} alternatives")
        
        self._log(f"✓ Pass2: {len(pass2_structure)} clusters built")
        return pass2_structure
    
    def _build_pass3_structure(self, pass2_input: List[str]) -> Dict[str, Dict]:
        """
        Build Pass3 cluster structure from Pass2 representatives.
        Returns: {surviving: {'alternatives': [...], 'cluster_size': N}}
        """
        self._log("\n🔍 Building Pass3 structure...")
        
        checkpoint = self.checkpoints.get('pass3', {})
        
        # Extract what was removed in Pass3
        # The input to Pass3 was all Pass2 representatives
        # Some survived, some were removed
        
        # Get surviving descriptors from pass3_columns.txt or from final_descriptors
        pass3_file = self.output_dir / 'pass3_columns.txt'
        if pass3_file.exists():
            with open(pass3_file, 'r') as f:
                surviving = set(line.strip() for line in f if line.strip())
        else:
            # If no pass3_columns.txt, assume all pass2 survived
            surviving = set(pass2_input)
        
        # Determine what was removed in Pass3
        removed_in_pass3 = set(pass2_input) - surviving
        
        # Build structure: for each surviving descriptor, find what it absorbed
        pass3_structure = {}
        
        # Method 1: Use checkpoint clusters if available
        cluster_map = self._extract_cluster_info_pass3(checkpoint)
        
        if cluster_map:
            # Build inverse mapping: which survivor absorbed which removed descriptors
            survivor_to_removed = defaultdict(set)
            
            for removed_desc in removed_in_pass3:
                # This descriptor was removed, find its cluster
                if removed_desc in cluster_map:
                    # This removed descriptor had its own cluster
                    # Find which surviving descriptor it's correlated with
                    # (This is tricky - we need to check VIF removal logic)
                    pass
            
            # Actually, VIF removal works differently
            # In VIF, we remove high-VIF descriptors one by one
            # The checkpoint might have: which descriptor was removed and why
            
            if 'removed_details' in checkpoint:
                for detail in checkpoint['removed_details']:
                    removed = detail['descriptor']
                    correlated = detail.get('correlated_descriptors', [])
                    
                    # Find which of the correlated descriptors survived
                    surviving_correlated = [d for d in correlated if d in surviving]
                    
                    if surviving_correlated:
                        # Assign to first surviving correlated descriptor
                        survivor = surviving_correlated[0]
                        if survivor not in pass3_structure:
                            pass3_structure[survivor] = {
                                'alternative_descriptors': [],
                                'cluster_size': 1,
                                'removal_history': {'pass3': []}
                            }
                        pass3_structure[survivor]['alternative_descriptors'].append(removed)
                        pass3_structure[survivor]['removal_history']['pass3'].append(removed)
        
        # Ensure all surviving descriptors are in the structure
        for desc in surviving:
            if desc not in pass3_structure:
                pass3_structure[desc] = {
                    'alternative_descriptors': [],
                    'cluster_size': 1,
                    'removal_history': {'pass3': []}
                }
        
        for desc, info in pass3_structure.items():
            info['alternative_descriptors'] = sorted(set(info['alternative_descriptors']))
            info['removal_history']['pass3'] = sorted(set(info['removal_history']['pass3']))
            info['cluster_size'] = len(info['alternative_descriptors']) + 1
        
        self._log(f"✓ Pass3: {len(pass3_structure)} clusters built")
        return pass3_structure
    
    def _merge_structures(self, pass2_structure: Dict, pass3_structure: Dict) -> Dict[str, Dict]:
        """
        Merge Pass2 and Pass3 structures with proper propagation.
        
        Logic:
        - For each final surviving descriptor from Pass3
        - If it was removed in Pass3, find what removed it from Pass2
        - Propagate Pass2 removal_history to the final survivor
        """
        self._log("\n🔗 Merging Pass2 and Pass3 structures...")
        
        final_structure = {}
        
        for final_desc, pass3_info in pass3_structure.items():
            # Initialize final structure for this descriptor
            final_structure[final_desc] = {
                'alternative_descriptors': [],
                'all_cluster_members': [final_desc],
                'removal_history': {
                    'pass2': [],
                    'pass3': []
                },
                'cluster_size': 1
            }
            
            # Step 1: Add Pass3 removed descriptors
            pass3_removed = set(pass3_info['removal_history']['pass3'])
            final_structure[final_desc]['removal_history']['pass3'] = sorted(pass3_removed)
            final_structure[final_desc]['all_cluster_members'].extend(pass3_removed)
            
            # Step 2: For each Pass3 removed descriptor, check if it has Pass2 alternatives
            pass2_removed = set()
            
            # First, check if final_desc itself has Pass2 alternatives
            if final_desc in pass2_structure:
                pass2_removed.update(pass2_structure[final_desc]['removal_history']['pass2'])
                final_structure[final_desc]['all_cluster_members'].extend(
                    pass2_structure[final_desc]['removal_history']['pass2']
                )
            
            # Second, for each Pass3 removed descriptor, propagate its Pass2 alternatives
            for removed_desc in pass3_removed:
                if removed_desc in pass2_structure:
                    # This removed descriptor was a Pass2 representative
                    # Propagate its Pass2 alternatives to the final descriptor
                    propagated = pass2_structure[removed_desc]['removal_history']['pass2']
                    pass2_removed.update(propagated)
                    final_structure[final_desc]['all_cluster_members'].extend(propagated)
                    
                    self._log(f"  Propagating {removed_desc} -> {final_desc}: {len(propagated)} descriptors")
            
            # Finalize
            final_structure[final_desc]['removal_history']['pass2'] = sorted(pass2_removed)
            final_structure[final_desc]['all_cluster_members'] = sorted(set(
                final_structure[final_desc]['all_cluster_members']
            ))
            
            all_alternatives = pass2_removed | pass3_removed
            final_structure[final_desc]['alternative_descriptors'] = sorted(all_alternatives)
            final_structure[final_desc]['cluster_size'] = len(final_structure[final_desc]['all_cluster_members'])
            final_structure[final_desc]['total_alternatives'] = len(all_alternatives)
        
        self._log(f"✓ Merged: {len(final_structure)} final clusters")
        return final_structure
    
    def build_cluster_structure(self, final_descriptors_file: Optional[str] = None) -> Dict:
        """Build complete cluster structure with proper recursive tracking"""
        self._log("\n" + "="*70)
        self._log("Building Cluster Structure v3.0 - Recursive Tracking")
        self._log("="*70)
        
        # Load checkpoints
        self.load_checkpoints()
        
        # Load final descriptors
        if final_descriptors_file is None:
            final_descriptors_file = self.output_dir / 'final_descriptors.txt'
        else:
            final_descriptors_file = Path(final_descriptors_file)
        
        if not final_descriptors_file.exists():
            raise FileNotFoundError(f"Not found: {final_descriptors_file}")
        
        with open(final_descriptors_file, 'r') as f:
            final_descriptors = [line.strip() for line in f if line.strip()]
        
        self._log(f"\n📊 Final descriptors: {len(final_descriptors)}")
        
        # Step 1: Build Pass2 structure
        pass2_structure = self._build_pass2_structure()
        
        # Step 2: Build Pass3 structure (input = Pass2 representatives)
        pass2_representatives = list(pass2_structure.keys())
        
        # Add any descriptors that passed Pass1 but weren't in any Pass2 cluster
        pass2_file = self.output_dir / 'pass2_columns.txt'
        if pass2_file.exists():
            with open(pass2_file, 'r') as f:
                all_pass2_input = [line.strip() for line in f if line.strip()]
        else:
            all_pass2_input = pass2_representatives
        
        pass3_structure = self._build_pass3_structure(all_pass2_input)
        
        # Step 3: Merge structures with proper propagation
        final_structure = self._merge_structures(pass2_structure, pass3_structure)
        
        # Step 4: Calculate statistics
        stats = self._calculate_statistics(final_structure)
        
        # Step 5: Build output structure
        output = {
            'metadata': {
                'description': 'Cluster structure v3.0 with recursive tracking',
                'version': '3.0',
                'total_descriptors': len(final_descriptors),
                'descriptors_with_alternatives': sum(
                    1 for info in final_structure.values() if info['total_alternatives'] > 0
                ),
                'standalone_descriptors': sum(
                    1 for info in final_structure.values() if info['total_alternatives'] == 0
                ),
                'total_alternative_descriptors': sum(
                    info['total_alternatives'] for info in final_structure.values()
                ),
            },
            'statistics': stats,
            'descriptors': {}
        }
        
        for descriptor, info in final_structure.items():
            output['descriptors'][descriptor] = {
                'cluster_size': info['cluster_size'],
                'is_representative': True,
                'alternative_descriptors': info['alternative_descriptors'],
                'all_cluster_members': info['all_cluster_members'],
                'removal_history': info['removal_history'],
                'total_alternatives': info['total_alternatives']
            }
        
        self._log(f"\n✅ Done! Total: {len(final_descriptors)} descriptors")
        self._log(f"   - With alternatives: {output['metadata']['descriptors_with_alternatives']}")
        self._log(f"   - Standalone: {output['metadata']['standalone_descriptors']}")
        self._log(f"   - Total alternatives: {output['metadata']['total_alternative_descriptors']}")
        
        return output
    
    def _calculate_statistics(self, cluster_results: Dict) -> Dict:
        """Calculate statistics about cluster sizes"""
        cluster_sizes = [info['cluster_size'] for info in cluster_results.values()]
        size_distribution = defaultdict(int)
        for size in cluster_sizes:
            size_distribution[size] += 1
        
        if cluster_sizes:
            return {
                'cluster_size_mean': float(np.mean(cluster_sizes)),
                'cluster_size_median': float(np.median(cluster_sizes)),
                'cluster_size_min': int(np.min(cluster_sizes)),
                'cluster_size_max': int(np.max(cluster_sizes)),
                'cluster_size_std': float(np.std(cluster_sizes)),
                'size_distribution': {str(k): v for k, v in sorted(size_distribution.items())}
            }
        return {
            'cluster_size_mean': 0,
            'cluster_size_median': 0,
            'cluster_size_min': 0,
            'cluster_size_max': 0,
            'cluster_size_std': 0,
            'size_distribution': {}
        }
    
    def save_to_json(self, structure: Dict, output_file: Optional[str] = None):
        """Save cluster structure to JSON"""
        if output_file is None:
            output_file = self.output_dir / 'surviving_descriptors_clusters_v3.json'
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        self._log(f"\n💾 Saved: {output_file}")


def create_cluster_structure_v3(
    output_dir: str,
    final_descriptors_file: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """Main function to create cluster structure v3"""
    backtracker = ImprovedClusterBacktrackerV3(output_dir, verbose)
    structure = backtracker.build_cluster_structure(final_descriptors_file)
    backtracker.save_to_json(structure, output_file)
    return structure


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster Backtracking v3.0')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--final-descriptors', type=str, default=None,
                       help='Path to final_descriptors.txt (default: output-dir/final_descriptors.txt)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output JSON file (default: output-dir/surviving_descriptors_clusters_v3.json)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    structure = create_cluster_structure_v3(
        args.output_dir,
        args.final_descriptors,
        args.output_file,
        not args.quiet
    )
    
    print(f"\n✅ {len(structure['descriptors'])} clusters created")
    print(f"   Total alternatives: {structure['metadata']['total_alternative_descriptors']}")
