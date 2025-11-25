"""Cluster Backtracking v2.0"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class ImprovedClusterBacktracker:
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
        cluster_map = {}
        if 'clusters' in checkpoint:
            clusters = checkpoint['clusters']
            if isinstance(clusters, list):
                for cluster in clusters:
                    rep = cluster.get('representative')
                    members = cluster.get('members', [])
                    if rep and members:
                        cluster_map[rep] = set(members)
            else:
                for rep, info in clusters.items():
                    members = info.get('members', [])
                    if members:
                        cluster_map[rep] = set(members)
        if 'representatives' in checkpoint and not cluster_map:
            for rep, info in checkpoint['representatives'].items():
                members = info.get('members', [])
                if members:
                    cluster_map[rep] = set(members)
        return cluster_map
    
    def _extract_cluster_info_pass3(self, checkpoint: Dict) -> Dict[str, Set[str]]:
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
    
    def _extract_cluster_info(self, pass_name: str) -> Dict[str, Set[str]]:
        checkpoint = self.checkpoints.get(pass_name)
        if checkpoint is None:
            return {}
        if pass_name == 'pass2':
            return self._extract_cluster_info_pass2(checkpoint)
        elif pass_name == 'pass3':
            return self._extract_cluster_info_pass3(checkpoint)
        return {}
    
    def _build_pass_clusters(self, pass_name: str, input_descriptors: List[str]) -> Dict[str, Dict]:
        cluster_map = self._extract_cluster_info(pass_name)
        result = {}
        for desc in input_descriptors:
            if desc in cluster_map:
                members = cluster_map[desc]
                alternatives = sorted(members - {desc})
                result[desc] = {'alternatives': alternatives, 'removed': alternatives.copy()}
            else:
                result[desc] = {'alternatives': [], 'removed': []}
        return result
    
    def _merge_pass_results(self, pass3_results: Dict, pass2_results: Dict) -> Dict[str, Dict]:
        merged = {}
        for final_desc, pass3_info in pass3_results.items():
            pass3_removed = set(pass3_info['removed'])
            pass2_removed = set()
            if final_desc in pass2_results:
                pass2_removed.update(pass2_results[final_desc]['removed'])
            for removed_desc in pass3_removed:
                if removed_desc in pass2_results:
                    pass2_removed.update(pass2_results[removed_desc]['removed'])
            all_alternatives = pass3_removed | pass2_removed
            merged[final_desc] = {
                'all_cluster_members': sorted([final_desc] + list(all_alternatives)),
                'alternative_descriptors': sorted(all_alternatives),
                'removal_history': {'pass2': sorted(pass2_removed), 'pass3': sorted(pass3_removed)},
                'total_alternatives': len(all_alternatives)
            }
        return merged
    
    def backtrack_clusters(self, surviving_descriptors: List[str]) -> Dict[str, Dict]:
        self._log(f"\n🔍 Backtracking {len(surviving_descriptors)} descriptors...")
        pass2_checkpoint = self.checkpoints.get('pass2')
        if pass2_checkpoint and 'clusters' in pass2_checkpoint:
            clusters = pass2_checkpoint['clusters']
            if isinstance(clusters, list):
                pass2_all_reps = [c['representative'] for c in clusters]
            else:
                pass2_all_reps = list(clusters.keys())
        else:
            pass2_all_reps = []
        pass2_results = self._build_pass_clusters('pass2', pass2_all_reps)
        pass3_results = self._build_pass_clusters('pass3', surviving_descriptors)
        merged_results = self._merge_pass_results(pass3_results, pass2_results)
        return merged_results
    
    def build_cluster_structure(self, final_descriptors_file: Optional[str] = None) -> Dict:
        self._log("\n" + "="*70)
        self._log("Building Cluster Structure v2.0")
        self._log("="*70)
        self.load_checkpoints()
        if final_descriptors_file is None:
            final_descriptors_file = self.output_dir / 'final_descriptors.txt'
        else:
            final_descriptors_file = Path(final_descriptors_file)
        if not final_descriptors_file.exists():
            raise FileNotFoundError(f"Not found: {final_descriptors_file}")
        with open(final_descriptors_file, 'r') as f:
            surviving_descriptors = [line.strip() for line in f if line.strip()]
        cluster_results = self.backtrack_clusters(surviving_descriptors)
        stats = self._calculate_statistics(cluster_results)
        structure = {
            'metadata': {
                'description': 'Cluster structure v2.0',
                'total_descriptors': len(surviving_descriptors),
                'descriptors_with_alternatives': sum(1 for info in cluster_results.values() if info['total_alternatives'] > 0),
                'standalone_descriptors': sum(1 for info in cluster_results.values() if info['total_alternatives'] == 0),
                'total_alternative_descriptors': sum(info['total_alternatives'] for info in cluster_results.values()),
            },
            'statistics': stats,
            'descriptors': {}
        }
        for descriptor, info in cluster_results.items():
            structure['descriptors'][descriptor] = {
                'cluster_size': len(info['all_cluster_members']),
                'is_representative': True,
                'alternative_descriptors': info['alternative_descriptors'],
                'all_cluster_members': info['all_cluster_members'],
                'removal_history': info['removal_history'],
                'total_alternatives': info['total_alternatives']
            }
        self._log(f"\n✅ Done! Total: {len(surviving_descriptors)}")
        return structure
    
    def _calculate_statistics(self, cluster_results: Dict) -> Dict:
        cluster_sizes = [len(info['all_cluster_members']) for info in cluster_results.values()]
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
        return {'cluster_size_mean': 0, 'cluster_size_median': 0, 'cluster_size_min': 0, 'cluster_size_max': 0, 'cluster_size_std': 0, 'size_distribution': {}}
    
    def save_to_json(self, structure: Dict, output_file: Optional[str] = None):
        if output_file is None:
            output_file = self.output_dir / 'surviving_descriptors_clusters.json'
        else:
            output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(structure, f, indent=2)
        self._log(f"\n💾 Saved: {output_file}")


def create_cluster_structure_improved(output_dir: str, final_descriptors_file: Optional[str] = None, output_file: Optional[str] = None, verbose: bool = True) -> Dict:
    backtracker = ImprovedClusterBacktracker(output_dir, verbose)
    structure = backtracker.build_cluster_structure(final_descriptors_file)
    backtracker.save_to_json(structure, output_file)
    return structure


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--final-descriptors', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    structure = create_cluster_structure_improved(args.output_dir, args.final_descriptors, args.output_file, not args.quiet)
    print(f"\n✅ {len(structure['descriptors'])} clusters")
