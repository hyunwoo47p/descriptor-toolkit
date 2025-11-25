#!/usr/bin/env python3
"""
메모리 누수 자동 수정 스크립트
Memory Leak Auto-Fix Script

이 스크립트는 다음을 수행합니다:
1. 백업 생성
2. parquet_reader_duckdb.py 수정
3. similarity_gpu.py 수정
4. pipeline.py 검증 (수동 수정 필요)

Usage:
    python fix_memory_leaks_auto.py
"""

import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


class MemoryLeakFixer:
    """메모리 누수 자동 수정 클래스"""
    
    def __init__(self, base_dir: str = "descriptor_pipeline"):
        self.base_dir = Path(base_dir)
        self.backup_dir = None
        self.fixes_applied = []
        self.manual_fixes_needed = []
    
    def create_backup(self) -> Path:
        """백업 디렉토리 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"descriptor_pipeline_backup_{timestamp}")
        
        print(f"📦 Creating backup: {backup_dir}")
        shutil.copytree(self.base_dir, backup_dir, dirs_exist_ok=True)
        self.backup_dir = backup_dir
        print(f"✓ Backup created successfully")
        
        return backup_dir
    
    def fix_parquet_reader_duckdb(self) -> bool:
        """parquet_reader_duckdb.py 수정"""
        file_path = self.base_dir / "io" / "parquet_reader_duckdb.py"
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return False
        
        print(f"\n🔧 Fixing {file_path.name}...")
        
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes_count = 0
        
        # Fix 1: .values.astype → .values.copy().astype
        pattern1 = r'\.values\.astype\(np\.float64\)'
        replacement1 = r'.values.copy().astype(np.float64)'
        
        new_content, count1 = re.subn(pattern1, replacement1, content)
        if count1 > 0:
            content = new_content
            fixes_count += count1
            print(f"  ✓ Added .copy() to {count1} locations")
        
        # Fix 2: 중복 함수 제거 (라인 226부터)
        lines = content.split('\n')
        second_def_line = -1
        
        for i, line in enumerate(lines):
            if i >= 225 and 'def iter_batches_duckdb' in line:
                second_def_line = i
                break
        
        if second_def_line > 0:
            lines = lines[:second_def_line]
            content = '\n'.join(lines)
            fixes_count += 1
            print(f"  ✓ Removed duplicate function definition (line {second_def_line + 1})")
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            self.fixes_applied.append(f"{file_path.name}: {fixes_count} fixes")
            print(f"✓ {file_path.name} fixed successfully ({fixes_count} changes)")
            return True
        else:
            print(f"⚠ No changes needed for {file_path.name}")
            return False
    
    def fix_similarity_gpu(self) -> bool:
        """similarity_gpu.py 수정"""
        file_path = self.base_dir / "core" / "similarity_gpu.py"
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return False
        
        print(f"\n🔧 Fixing {file_path.name}...")
        
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Fix: .cpu().numpy() → .detach().cpu().numpy().copy()
        # 단, 이미 .detach()가 있는 경우는 건너뛰기
        pattern = r'(?<!\.detach\(\))\.cpu\(\)\.numpy\(\)'
        replacement = r'.detach().cpu().numpy().copy()'
        
        new_content, count = re.subn(pattern, replacement, content)
        
        if count > 0:
            content = new_content
            file_path.write_text(content, encoding='utf-8')
            self.fixes_applied.append(f"{file_path.name}: {count} fixes")
            print(f"✓ {file_path.name} fixed successfully ({count} changes)")
            return True
        else:
            print(f"⚠ No changes needed for {file_path.name}")
            return False
    
    def check_pipeline_py(self) -> List[str]:
        """pipeline.py 검증 (수동 수정 필요)"""
        file_path = self.base_dir / "core" / "pipeline.py"
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return []
        
        print(f"\n🔍 Checking {file_path.name}...")
        
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        issues = []
        
        # Check 1: import gc
        if 'import gc' not in content:
            issues.append("Missing 'import gc' statement")
        
        # Check 2: spearman_pass.process 호출
        for i, line in enumerate(lines):
            if 'spearman_pass.process' in line:
                # 다음 몇 줄 확인
                context = '\n'.join(lines[i:min(i+5, len(lines))])
                if 'self.graph_builder' in context or 'self.leiden' in context:
                    issues.append(
                        f"Line {i+1}: spearman_pass.process() has incorrect arguments"
                    )
        
        # Check 3: NumPy slicing without .copy()
        view_patterns = [
            r'data\[:,\s*indices_\w+\](?!\.copy\(\))',
            r'G_\w+\[indices_\w+\].*\[indices_\w+\](?!\.copy\(\))',
        ]
        
        for pattern in view_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(
                    f"Line {line_num}: NumPy slicing without .copy() - {match.group()}"
                )
        
        # Check 4: _cleanup_memory 메서드
        if '_cleanup_memory' not in content:
            issues.append("Missing _cleanup_memory() method")
        
        if issues:
            print(f"⚠ Found {len(issues)} issues requiring manual fixes:")
            for issue in issues:
                print(f"  - {issue}")
            self.manual_fixes_needed.extend(issues)
        else:
            print(f"✓ {file_path.name} looks good")
        
        return issues
    
    def print_summary(self):
        """수정 요약 출력"""
        print("\n" + "="*70)
        print("수정 요약 (Fix Summary)")
        print("="*70)
        
        if self.backup_dir:
            print(f"\n📦 Backup location: {self.backup_dir}")
        
        if self.fixes_applied:
            print(f"\n✅ Automatically fixed ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"  - {fix}")
        
        if self.manual_fixes_needed:
            print(f"\n⚠️  Manual fixes needed ({len(self.manual_fixes_needed)}):")
            for fix in self.manual_fixes_needed:
                print(f"  - {fix}")
            print("\n📖 Please refer to IMPLEMENTATION_GUIDE.md for details")
        else:
            print("\n✓ No manual fixes needed")
        
        print("\n" + "="*70)
        print("Next steps:")
        print("  1. Review changes in modified files")
        print("  2. Apply manual fixes to pipeline.py")
        print("  3. Run tests: python test_memory_fix.py")
        print("  4. If issues occur, restore from backup")
        print("="*70)
    
    def run(self):
        """전체 수정 프로세스 실행"""
        print("="*70)
        print("Memory Leak Auto-Fix Script")
        print("="*70)
        
        # 1. 백업
        self.create_backup()
        
        # 2. 자동 수정
        try:
            self.fix_parquet_reader_duckdb()
            self.fix_similarity_gpu()
            self.check_pipeline_py()
        except Exception as e:
            print(f"\n✗ Error during fix: {e}")
            print(f"Backup is available at: {self.backup_dir}")
            return False
        
        # 3. 요약
        self.print_summary()
        
        return True


def main():
    """메인 함수"""
    import sys
    
    # 작업 디렉토리 확인
    if not Path("descriptor_pipeline").exists():
        print("✗ Error: descriptor_pipeline directory not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # 수정 실행
    fixer = MemoryLeakFixer()
    success = fixer.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
