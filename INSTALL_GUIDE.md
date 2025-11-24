# 설치 가이드 (Installation Guide)

## 🖥️ 시스템 요구사항

- **Python**: 3.11.x (3.12+ 비권장 - PyTorch nightly 호환성)
- **OS**: Windows 10/11, Linux (Ubuntu 22.04+)
- **GPU**: NVIDIA GPU with CUDA support

---

## 🎯 환경별 설치 가이드

이 패키지는 두 가지 환경에서 테스트되었습니다:

| 환경 | GPU | CUDA | PyTorch |
|------|-----|------|---------|
| **서버** (UES-ML) | RTX 6000 Ada (sm_89) | 12.4 | stable |
| **노트북** | RTX 5070 (sm_120 Blackwell) | 13.0 | nightly |

---

## 🖥️ 서버 환경 설치 (RTX 6000 Ada, RTX 40xx 등)

### 한 번에 설치 (복사-붙여넣기)

```bash
# 1. 환경 생성
conda create -n descriptor python=3.11 -c conda-forge --override-channels -y
conda activate descriptor

# 2. Conda 패키지 (ABI 호환성을 위해 conda-forge 통일)
conda install -c conda-forge \
    rdkit \
    mordred \
    numpy=1.26.4 \
    pandas \
    pyarrow \
    scipy \
    tqdm \
    duckdb \
    lxml \
    requests \
    -y

# 3. PyTorch + CUDA 12.4 (stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 추가 pip 패키지
pip install igraph leidenalg statsmodels tdigest openpyxl

# 5. 패키지 설치
cd molecular_descriptor_toolkit
pip install -e .
```

---

## 💻 노트북 환경 설치 (RTX 5070, RTX 50xx Blackwell)

### ⚠️ 중요: RTX 50 시리즈는 PyTorch Nightly 필수!

RTX 5070/5080/5090은 Blackwell 아키텍처(sm_120)로, stable PyTorch에서 지원하지 않습니다.

### 한 번에 설치 (복사-붙여넣기)

```bash
# 1. 환경 생성 (Python 3.11 필수!)
conda create -n descriptor python=3.11 -c conda-forge --override-channels -y
conda activate descriptor

# 2. Conda 패키지
conda install -c conda-forge \
    rdkit \
    mordred \
    numpy=1.26.4 \
    pandas \
    pyarrow \
    scipy \
    tqdm \
    duckdb \
    lxml \
    requests \
    -y

# 3. PyTorch Nightly + CUDA 13.0 (RTX 5070용)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 4. 추가 pip 패키지
pip install igraph leidenalg statsmodels tdigest openpyxl

# 5. 패키지 설치
cd molecular_descriptor_toolkit
pip install -e .
```

---

## 🔬 CPU 전용 설치 (GPU 없는 경우)

```bash
# 1. 환경 생성
conda create -n descriptor python=3.11 -c conda-forge --override-channels -y
conda activate descriptor

# 2. Conda 패키지
conda install -c conda-forge \
    rdkit mordred numpy=1.26.4 pandas pyarrow scipy tqdm duckdb lxml requests -y

# 3. PyTorch CPU 버전
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 추가 pip 패키지
pip install igraph leidenalg statsmodels tdigest openpyxl

# 5. 패키지 설치
cd molecular_descriptor_toolkit
pip install -e .
```

---

## ✅ 설치 확인

### 전체 환경 검증 스크립트

```bash
python << 'EOF'
import sys
print("="*60)
print("🔍 환경 검증")
print("="*60)

# Python
print(f"\n[1] Python: {sys.version.split()[0]}")

# PyTorch + CUDA
try:
    import torch
    print(f"\n[2] PyTorch: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"    Supported archs: {torch.cuda.get_arch_list()}")
except ImportError as e:
    print(f"\n[2] PyTorch: ❌ {e}")

# Core packages
packages = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("pyarrow", "pa"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
]
print("\n[3] Core packages:")
for name, alias in packages:
    try:
        mod = __import__(name)
        print(f"    ✅ {name}: {mod.__version__}")
    except ImportError:
        print(f"    ❌ {name}: NOT INSTALLED")

# Chemistry packages
print("\n[4] Chemistry packages:")
try:
    from rdkit import Chem
    import rdkit
    print(f"    ✅ rdkit: {rdkit.__version__}")
except ImportError:
    print(f"    ❌ rdkit: NOT INSTALLED")

try:
    from mordred import Calculator
    print(f"    ✅ mordred: OK")
except ImportError:
    print(f"    ❌ mordred: NOT INSTALLED")

# Clustering packages
print("\n[5] Clustering packages:")
for name in ["igraph", "leidenalg", "statsmodels", "tdigest"]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "OK")
        print(f"    ✅ {name}: {ver}")
    except ImportError:
        print(f"    ❌ {name}: NOT INSTALLED")

# Toolkit
print("\n[6] Toolkit:")
try:
    from molecular_descriptor_toolkit import Config
    from molecular_descriptor_toolkit import __version__
    print(f"    ✅ molecular_descriptor_toolkit: {__version__}")
except ImportError as e:
    print(f"    ❌ molecular_descriptor_toolkit: {e}")

print("\n" + "="*60)
print("검증 완료!")
print("="*60)
EOF
```

### 예상 출력 (서버: RTX 6000 Ada)
```
[2] PyTorch: 2.9.0
    CUDA available: True
    GPU: NVIDIA RTX 6000 Ada Generation
    Compute capability: (8, 9)
```

### 예상 출력 (노트북: RTX 5070)
```
[2] PyTorch: 2.x.x.dev...+cu130
    CUDA available: True
    GPU: NVIDIA GeForce RTX 5070
    Compute capability: (12, 0)
    Supported archs: [..., 'sm_120', ...]
```

---

## 📦 패키지별 버전 요약 (서버 기준)

| 패키지 | 서버 버전 | 설치 방법 | 용도 |
|--------|----------|----------|------|
| **Python** | 3.11.13 | conda | 기본 |
| **PyTorch** | 2.9.0+cu124 | pip | GPU 가속 |
| **RDKit** | 2025.9.1 | conda/pip | 분자 처리 |
| **numpy** | 1.26.4 | conda | 수치 계산 (ABI 호환) |
| **pandas** | 2.3.3 | conda/pip | 데이터프레임 |
| **pyarrow** | 22.0.0 | conda/pip | Parquet I/O |
| **scipy** | 1.16.3 | pip | 통계/수치 |
| **tqdm** | 4.67.1 | conda | 진행바 |
| **igraph** | 1.0.0 | pip | 그래프 |
| **leidenalg** | 0.11.0 | pip | Leiden 클러스터링 |
| **statsmodels** | 0.14.5 | pip | VIF 계산 |
| **tdigest** | 0.5.2.2 | pip | CDF 근사 |
| **mordred** | 1.2.0 | conda | Descriptor 계산 |
| **duckdb** | 1.4.1 | conda | 대용량 쿼리 |

---

## ⚠️ 문제 해결

### Q1: RTX 5070/5080/5090에서 CUDA 오류

```
UserWarning: NVIDIA GeForce RTX 5070 with CUDA capability sm_120 
is not compatible with the current PyTorch installation.
```

**해결**: PyTorch Nightly 설치 필요
```bash
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Q2: Conda 약관 동의 오류

```
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

**해결**: 약관 동의 또는 conda-forge만 사용
```bash
# 방법 1: 약관 동의
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 방법 2: conda-forge만 사용 (권장)
conda create -n descriptor python=3.11 -c conda-forge --override-channels -y
```

### Q3: CUDA 버전 확인

```bash
nvidia-smi
# CUDA Version 확인 (우측 상단)
# 예: CUDA Version: 12.4 → cu124 사용
# 예: CUDA Version: 13.0 → cu130 사용 (RTX 50 시리즈)
```

### Q4: RDKit import 오류

```bash
# conda 환경 활성화 확인
conda activate descriptor
python -c "from rdkit import Chem; print('OK')"
```

### Q5: NumPy ABI 불일치 경고

```bash
# pip numpy 제거 후 conda로 재설치
pip uninstall numpy -y
conda install -c conda-forge numpy=1.26.4 -y
```

### Q6: `leidenalg` 설치 실패 (Windows)

```bash
# C++ 빌드 도구 필요할 수 있음
# Visual Studio Build Tools 설치 후 재시도
pip install leidenalg
```

### Q5: GPU 인식 안 됨

```bash
# 1. NVIDIA 드라이버 확인
nvidia-smi

# 2. PyTorch CUDA 재설치
conda uninstall pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

---

## 🔧 환경 내보내기/가져오기

### 환경 내보내기

```bash
conda activate descriptor
conda env export > environment.yml
```

### 환경 가져오기 (다른 컴퓨터에서)

```bash
conda env create -f environment.yml
conda activate descriptor
```

---

## 📝 요약 명령어 (빠른 참조)

```bash
# 새 환경 설치
conda create -n descriptor python=3.11 -y
conda activate descriptor
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c conda-forge rdkit -y
pip install numpy pandas pyarrow scipy tqdm igraph leidenalg statsmodels mordred
pip install -e .

# 확인
python -c "import torch; from rdkit import Chem; from molecular_descriptor_toolkit import Config; print('All OK!')"
```
