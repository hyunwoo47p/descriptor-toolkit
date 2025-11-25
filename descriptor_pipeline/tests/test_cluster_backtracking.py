#!/usr/bin/env python3
"""
Cluster Backtracking 테스트 스크립트

surviving_descriptors_clusters.json 생성을 테스트합니다.
"""

import json
from pathlib import Path
from cluster_backtracker import create_cluster_structure


def test_cluster_structure(output_dir: str):
    """클러스터 구조 생성 테스트"""
    
    print("="*70)
    print("Cluster Backtracking Test")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    # 1. Checkpoint 파일 확인
    print("\n1️⃣ Checking checkpoint files...")
    checkpoint_files = [
        'pass2_spearman.json',
        'pass3_vif.json',
        'pass4_nonlinear.json',
        'final_descriptors.txt'
    ]
    
    all_exist = True
    for filename in checkpoint_files:
        file_path = output_dir / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some checkpoint files are missing!")
        print("Run the pipeline with checkpoint=True first.")
        return False
    
    # 2. 클러스터 구조 생성
    print("\n2️⃣ Generating cluster structure...")
    try:
        structure = create_cluster_structure(
            output_dir=str(output_dir),
            verbose=True
        )
    except Exception as e:
        print(f"\n❌ Failed to generate cluster structure: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 결과 검증
    print("\n3️⃣ Validating results...")
    
    # Metadata 확인
    metadata = structure['metadata']
    print(f"\n  Metadata:")
    print(f"    Total descriptors: {metadata['total_descriptors']}")
    print(f"    With alternatives: {metadata['descriptors_with_alternatives']}")
    print(f"    Standalone: {metadata['standalone_descriptors']}")
    print(f"    Total alternatives: {metadata['total_alternative_descriptors']}")
    
    # Statistics 확인
    stats = structure['statistics']
    print(f"\n  Statistics:")
    print(f"    Cluster size (mean): {stats['cluster_size_mean']:.2f}")
    print(f"    Cluster size (median): {stats['cluster_size_median']}")
    print(f"    Cluster size (min/max): {stats['cluster_size_min']} / {stats['cluster_size_max']}")
    
    # 큰 클러스터 찾기
    large_clusters = {
        desc: info 
        for desc, info in structure['descriptors'].items() 
        if info['cluster_size'] >= 10
    }
    
    print(f"\n  Large clusters (size >= 10): {len(large_clusters)}")
    if large_clusters:
        print(f"    Top 5 largest:")
        for desc, info in sorted(large_clusters.items(), 
                                key=lambda x: x[1]['cluster_size'], 
                                reverse=True)[:5]:
            print(f"      {desc}: {info['cluster_size']} members")
    
    # 샘플 descriptor 확인
    print(f"\n  Sample descriptors:")
    sample_descs = list(structure['descriptors'].keys())[:3]
    for desc in sample_descs:
        info = structure['descriptors'][desc]
        print(f"\n    {desc}:")
        print(f"      Cluster size: {info['cluster_size']}")
        print(f"      Alternatives: {len(info['alternative_descriptors'])}")
        if info['alternative_descriptors']:
            print(f"      Examples: {info['alternative_descriptors'][:3]}")
        if info['removal_history']:
            print(f"      Removal history: {list(info['removal_history'].keys())}")
    
    # 4. 파일 저장 확인
    print("\n4️⃣ Checking saved file...")
    output_file = output_dir / 'surviving_descriptors_clusters.json'
    if output_file.exists():
        file_size = output_file.stat().st_size / 1024
        print(f"  ✓ File saved: {output_file}")
        print(f"    Size: {file_size:.1f} KB")
    else:
        print(f"  ✗ File not saved!")
        return False
    
    # 5. JSON 로드 테스트
    print("\n5️⃣ Testing JSON load...")
    try:
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        print(f"  ✓ JSON loaded successfully")
        print(f"    Descriptors in file: {len(loaded['descriptors'])}")
    except Exception as e:
        print(f"  ✗ Failed to load JSON: {e}")
        return False
    
    # 최종 결과
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print(f"\nCluster structure saved to:")
    print(f"  {output_file}")
    print(f"\nSummary:")
    print(f"  - {metadata['total_descriptors']} surviving descriptors")
    print(f"  - {metadata['total_alternative_descriptors']} alternative descriptors")
    print(f"  - Average cluster size: {stats['cluster_size_mean']:.2f}")
    print(f"  - Largest cluster: {stats['cluster_size_max']} members")
    
    return True


def analyze_alternatives(output_dir: str, top_n: int = 10):
    """Alternative descriptors 분석"""
    
    output_file = Path(output_dir) / 'surviving_descriptors_clusters.json'
    
    if not output_file.exists():
        print(f"File not found: {output_file}")
        return
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print(f"Top {top_n} Descriptors by Alternative Count")
    print("="*70)
    
    # Alternative 개수로 정렬
    sorted_descs = sorted(
        data['descriptors'].items(),
        key=lambda x: x[1]['total_alternatives'],
        reverse=True
    )
    
    for i, (desc, info) in enumerate(sorted_descs[:top_n], 1):
        print(f"\n{i}. {desc}")
        print(f"   Total alternatives: {info['total_alternatives']}")
        print(f"   Cluster size: {info['cluster_size']}")
        
        if info['removal_history']:
            print(f"   Removal history:")
            for pass_name, removed in info['removal_history'].items():
                print(f"     {pass_name}: {len(removed)} removed")
        
        if info['alternative_descriptors']:
            print(f"   Sample alternatives: {info['alternative_descriptors'][:5]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test cluster backtracking functionality"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Also analyze alternatives after generation'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top descriptors to show in analysis'
    )
    
    args = parser.parse_args()
    
    # 메인 테스트
    success = test_cluster_structure(args.output_dir)
    
    # 분석
    if success and args.analyze:
        analyze_alternatives(args.output_dir, args.top_n)
