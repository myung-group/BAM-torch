# CG (Coarse-Grained) Water Simulation Workflow

이 문서는 BAM-torch를 이용한 CG water 시뮬레이션의 전체 워크플로우와 관련 코드를 설명합니다.

---

## 디렉토리 구조

```
example-CG/
├── force_auto/              # regress_forces='auto' 훈련 (권장)
├── md/water/
│   ├── aa_md/               # All-Atom MD 결과
│   └── cg_md/
│       ├── v2/              # regress_forces='direct' (문제 있음)
│       └── v3_auto/         # regress_forces='auto' (정상 작동)
│           └── 49epochs/    # 최종 MD 결과
├── *.py                     # 유틸리티 스크립트
├── *.json                   # 훈련 설정 파일
└── *.npz, *.pkl, *.pt       # 데이터 및 모델 파일
```

---

## 전체 워크플로우

```
[1. 데이터 준비]
    AA trajectory (water.traj)
           ↓
    preprocess_cg.py (AA → CG 매핑)
           ↓
    water_cg.npz (CG 훈련 데이터)

[2. 모델 훈련]
    water_cg.npz + input_cg_auto.json
           ↓
    main.py (BAM-torch 훈련)
           ↓
    model_cg_auto.pkl (훈련된 모델)

[3. LAMMPS 변환]
    model_cg_auto.pkl
           ↓
    make_pt_cg.py (JIT 컴파일)
           ↓
    model_cg_auto-lammps.pt (LAMMPS용 모델)

[4. MD 시뮬레이션]
    water_cg.data + model_cg_auto-lammps.pt + in.lammps
           ↓
    lmp (LAMMPS 실행)
           ↓
    dump.lammpstrj + log.lammps

[5. 후처리 및 분석]
    dump.lammpstrj + log.lammps
           ↓
    lammpsout_to_traj.py
           ↓
    lammps_out.traj
           ↓
    plot_rdf.py (RDF 분석)
```

---

## 1단계: 데이터 준비

### preprocess_cg.py
**위치**: `example-CG/preprocess_cg.py`
**기능**: AA trajectory를 CG로 매핑하여 NPZ 파일로 저장

```bash
python preprocess_cg.py -i water.traj -o water_cg.npz -m water
```

**입력**: `water.traj` (768 atoms = 256 H2O × 3)
**출력**: `water_cg.npz` (256 CG beads)

**NPZ 파일 구조**:
| 키 | 형태 | 설명 |
|----|------|------|
| positions | (frames, beads, 3) | CG bead COM 좌표 |
| forces | (frames, beads, 3) | 매핑된 힘 |
| energies | (frames,) | AA 에너지 |
| types | (beads,) | bead type |
| cells | (frames, 3, 3) | 시뮬레이션 셀 |

---

### aa_to_cg_data.py
**위치**: `md/water/cg_md/v3_auto/aa_to_cg_data.py`
**기능**: AA LAMMPS data 파일을 CG LAMMPS data 파일로 변환

```bash
python aa_to_cg_data.py
```

**입력**: AA LAMMPS data (w256_rst)
**출력**: `water_cg_from_rst.data` (CG LAMMPS data)

**주요 함수**:
- `read_lammps_data()`: LAMMPS data 파일 읽기
- `cg_mapping()`: H2O → COM 매핑 (minimum image convention 적용)
- `write_lammps_data()`: CG LAMMPS data 파일 쓰기

---

## 2단계: 모델 훈련

### main.py
**위치**: `example-CG/main.py` 또는 `force_auto/main.py`
**기능**: BAM-torch 모델 훈련 실행

```bash
python main.py input_cg_auto.json
```

### input_cg_auto.json (권장 설정)
**위치**: `force_auto/input_cg_auto.json`
**핵심 설정**:
```json
{
  "regress_forces": "auto",  // F = -dE/dR (에너지 보존 보장)
  "data_path": "water_cg.npz",
  "model_path": "model_cg_auto.pkl"
}
```

**중요**: `regress_forces` 옵션
| 값 | 설명 | MD 안정성 |
|----|------|-----------|
| `"direct"` | 모델이 직접 힘 예측 | 불안정 (F ≠ -dE/dR) |
| `"auto"` | autograd로 힘 계산 | 안정 (F = -dE/dR 보장) |

**출력**: `model_cg_auto.pkl`

---

### run.sh
**위치**: `force_auto/run.sh`
**기능**: 훈련 실행 스크립트

```bash
#!/bin/bash
python main.py input_cg_auto.json > loss_cg_auto.out 2>&1
```

---

## 3단계: LAMMPS 변환

### make_pt_cg.py
**위치**: `md/water/cg_md/v3_auto/49epochs/make_pt_cg.py`
**기능**: PKL 모델을 LAMMPS용 TorchScript (.pt)로 변환

```bash
python make_pt_cg.py
```

**입력**: `model_cg_auto.pkl`
**출력**: `model_cg_auto-lammps.pt`

**핵심 코드**:
```python
from bam_torch.lammps import lammps_bam
lammps_bam.deploy_model("model_cg_auto.pkl", "model_cg_auto-lammps.pt")
```

---

### create_lammps_cg.py
**위치**: `md/water/cg_md/v3_auto/49epochs/create_lammps_cg.py`
**기능**: NPZ 파일에서 LAMMPS data 파일 생성

```bash
python create_lammps_cg.py
```

**입력**: `water_cg.npz`
**출력**: `water_cg.data`

---

### npz_to_lammps_data.py
**위치**: `example-CG/npz_to_lammps_data.py`
**기능**: NPZ → LAMMPS data 변환 (CLI 버전)

```bash
python npz_to_lammps_data.py water_cg.npz water_cg.data --frame 0
```

---

## 4단계: MD 시뮬레이션

### in.lammps
**위치**: `md/water/cg_md/v3_auto/49epochs/in.lammps`
**기능**: LAMMPS MD 시뮬레이션 입력 파일

**주요 명령어**:
```lammps
units           metal
atom_style      atomic
read_data       water_cg_from_rst.data

pair_style      bam
pair_coeff      * * model_cg_auto-lammps.pt water

fix             1 all nvt temp 50 50 0.1
run             100000
```

**실행**:
```bash
lmp -in in.lammps
```

**출력**:
- `dump.lammpstrj`: 궤적 파일
- `log.lammps`: 에너지/온도 로그

---

## 5단계: 후처리 및 분석

### lammpsout_to_traj.py
**위치**: `md/water/cg_md/v3_auto/49epochs/lammpsout_to_traj.py`
**기능**: LAMMPS dump + log → ASE trajectory 변환

```bash
python lammpsout_to_traj.py
```

**입력**: `dump.lammpstrj`, `log.lammps`
**출력**: `lammps_out.traj`

**주요 처리**:
- `'water'` → `'Ar'` 치환 (ASE가 'water' 원소 인식 못함)
- log.lammps에서 에너지 추출
- SinglePointCalculator로 에너지/힘 저장

---

### plot_rdf.py
**위치**: `md/water/cg_md/v3_auto/49epochs/plot_rdf.py`
**기능**: CG trajectory에서 RDF 계산 및 플롯

```bash
python plot_rdf.py
```

**입력**: `lammps_out.traj`
**출력**: `rdf_cg.png`, `rdf_cg.dat`

**파라미터**:
- `rmax = 9.5 Å` (셀 크기의 절반 이하)
- `nbins = 100`

---

### plot_rdf_aa_to_cg.py
**위치**: `md/water/aa_md/plot_rdf_aa_to_cg.py`
**기능**: AA trajectory를 CG로 매핑하여 RDF 계산 후 CG MD와 비교

```bash
python plot_rdf_aa_to_cg.py
```

**입력**: AA `lammps_out.traj`, CG `rdf_cg.dat`
**출력**: `rdf_comparison.png`, `rdf_aa_to_cg.dat`

**핵심 함수**:
- `map_aa_to_cg()`: AA atoms → CG beads (COM 계산, minimum image convention)

---

### plot_rdf_oo.py
**위치**: `md/water/aa_md/plot_rdf_oo.py`
**기능**: AA trajectory에서 O-O RDF 계산

```bash
python plot_rdf_oo.py
```

**입력**: `lammps_out.traj`
**출력**: `rdf_oo.png`, `rdf_oo.dat`

---

## 디버깅/검증 스크립트

### evaluate_cg_model.py
**위치**: `example-CG/evaluate_cg_model.py`
**기능**: 훈련된 CG 모델의 에너지/힘 예측 평가

### compare_forces.py / compare_forces_v2.py
**위치**: `md/water/cg_md/v2/`
**기능**: forces_local vs forces (autograd) 비교
- `regress_forces='direct'`일 때 두 값이 다름
- `regress_forces='auto'`일 때 두 값이 같음

### check_cg_mapping.py
**위치**: `md/water/cg_md/v2/check_cg_mapping.py`
**기능**: CG 매핑 검증

### debug_lammps_conversion.py
**위치**: `md/water/cg_md/v2/` 및 `v3_auto/`
**기능**: LAMMPS 변환 디버깅

---

## 파일 형식 요약

| 확장자 | 설명 |
|--------|------|
| `.traj` | ASE trajectory (positions, energies, forces) |
| `.npz` | NumPy compressed (CG 훈련 데이터) |
| `.pkl` | Python pickle (BAM-torch 모델) |
| `.pt` | TorchScript (LAMMPS용 모델) |
| `.data` | LAMMPS data 파일 (초기 구조) |
| `.lammpstrj` | LAMMPS dump 파일 (궤적) |
| `.json` | 훈련 설정 파일 |

---

## 핵심 교훈

1. **regress_forces='auto' 사용 필수**
   - `direct`: 모델이 직접 힘 예측 → F ≠ -dE/dR → MD 불안정
   - `auto`: autograd로 힘 계산 → F = -dE/dR 보장 → MD 안정

2. **RDF 비교 방법**
   - CG MD RDF: CG bead 간 거리
   - AA→CG RDF: AA trajectory를 COM으로 매핑 후 거리 계산
   - 두 RDF가 일치하면 CG 모델이 AA 구조를 잘 재현

3. **에너지 보존 확인**
   - `log.lammps`의 `econserve` 열 확인
   - 값이 일정하면 에너지 보존됨
