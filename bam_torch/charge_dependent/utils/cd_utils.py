"""
Charge-dependent data utilities for BAM-torch.

QM9star 데이터를 charge 정보와 함께 PyG Data 객체로 변환하는 유틸리티.
Wiggle150 테스트셋 로딩도 지원.

데이터 소스:
  - qm9star_preprocessor.py로 변환된 extended xyz 파일
  - 또는 charge info가 포함된 일반 xyz/traj 파일
"""

import numpy as np
from ase.io import read
from matscipy.neighbours import neighbour_list
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from bam_torch.utils.utils import (
    get_enr_avg_per_element,
    get_relative_vector,
)

# =============================================================
# 단위 변환 상수
# =============================================================
HARTREE_TO_EV = 27.211386245988       # eV/Hartree
BOHR_TO_ANGSTROM = 0.529177249        # Å/bohr
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # ≈ 51.4221


def _safe_get_forces(atoms):
    """atoms 객체에서 안전하게 forces를 가져오기.

    ASE 3.27+ 에서 'forces' 는 arrays 가 아닌 내부 calculator 에 저장됨.
    """
    # (1) arrays에서 직접 읽기 (구버전 ASE)
    if 'forces' in atoms.arrays:
        return atoms.arrays['forces']
    # (2) ASE 3.27+: get_forces() 직접 호출
    try:
        f = atoms.get_forces()
        if f is not None:
            return f
    except Exception:
        pass
    # (3) calculator results에서 읽기
    if atoms.calc is not None:
        try:
            return atoms.get_forces()
        except Exception:
            pass
    return np.zeros((len(atoms), 3))


def _safe_get_energy(atoms):
    """atoms 객체에서 안전하게 energy를 가져오기.

    ASE 3.27+ 에서 'energy' 는 info 가 아닌 get_potential_energy() 로 접근.
    """
    # (1) info에서 직접 읽기 (구버전 ASE)
    if 'energy' in atoms.info:
        return float(atoms.info['energy'])
    # (2) ASE 3.27+: get_potential_energy() 직접 호출
    try:
        e = atoms.get_potential_energy()
        if e is not None:
            return float(e)
    except Exception:
        pass
    # (3) calculator results에서 읽기
    if atoms.calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    raise ValueError("Energy를 찾을 수 없습니다. "
                     "Extended xyz 또는 calculator가 필요합니다.")


def _safe_get_stress(atoms):
    """atoms 객체에서 안전하게 stress를 가져오기."""
    if atoms.calc is not None:
        try:
            if 'stress' in atoms.calc.results:
                return atoms.get_stress()
        except Exception:
            pass
    return np.zeros(6)


def _safe_get_volume(atoms):
    """atoms 객체에서 안전하게 volume을 가져오기."""
    try:
        vol = atoms.get_volume()
        if vol > 0:
            return vol
    except Exception:
        pass
    return np.zeros(1)


def _extract_charges(atoms, charge_key="charges",
                     total_charge_key="total_charge"):
    """atoms 객체에서 charge 정보를 추출.
    
    Extended xyz에서는:
      - atoms.arrays['charges'] → per-atom charges
      - atoms.info['total_charge'] → system total charge
    
    Returns:
        atomic_charges: np.ndarray [n_atoms]
        total_charge: float
    """
    n_atoms = len(atoms)
    
    # --- Atomic charges ---
    atomic_charges = None
    
    # (1) arrays에서 찾기 (구버전 ASE: Properties=...charges:R:1)
    if charge_key in atoms.arrays:
        atomic_charges = np.array(atoms.arrays[charge_key], dtype=float)
    # (1b) ASE 3.27+: 'charges' 는 get_charges() 로 접근
    elif charge_key == 'charges':
        try:
            q = atoms.get_charges()
            if q is not None and len(q) == n_atoms:
                atomic_charges = np.array(q, dtype=float)
        except Exception:
            pass
    # (2) info에서 찾기
    elif charge_key in atoms.info:
        val = atoms.info[charge_key]
        if isinstance(val, (list, np.ndarray)):
            atomic_charges = np.array(val, dtype=float)
        elif isinstance(val, str):
            atomic_charges = np.array(
                [float(x) for x in val.split()], dtype=float
            )
    
    # (3) 다른 일반적인 키 이름도 시도
    if atomic_charges is None:
        for alt_key in ['charge', 'npa_charges', 'mulliken_charge',
                        'hirshfeld_charges', 'formal_charges',
                        'initial_charges']:
            if alt_key in atoms.arrays:
                atomic_charges = np.array(
                    atoms.arrays[alt_key], dtype=float
                )
                break
            elif alt_key in atoms.info:
                val = atoms.info[alt_key]
                if isinstance(val, (list, np.ndarray)):
                    atomic_charges = np.array(val, dtype=float)
                    break
    
    if atomic_charges is None:
        atomic_charges = np.zeros(n_atoms)
    
    # --- Total charge ---
    if total_charge_key in atoms.info:
        total_charge = float(atoms.info[total_charge_key])
    else:
        total_charge = float(np.sum(atomic_charges))
    
    return atomic_charges, total_charge


def get_graphset_charge(
    data, cutoff, uniq_element, enr_avg_per_element,
    enr_var, regress_forces=True, max_neigh=None,
    show_progress=False, desc="Converting",
    charge_key="charges",
    total_charge_key="total_charge",
):
    """
    기존 get_graphset을 확장하여 charge 정보를 포함하는 그래프 데이터셋 생성.

    Extended xyz (qm9star_preprocessor.py 출력) 또는 일반 ASE 파일 모두 지원.

    Args:
        data: ASE atoms 리스트
        cutoff: 거리 cutoff (Å)
        uniq_element: {atomic_number: species_index} 딕셔너리
        enr_avg_per_element: 원소별 평균 에너지
        enr_var: 에너지 분산
        regress_forces: force regression 여부
        max_neigh: 최대 이웃 수
        show_progress: tqdm 진행바 표시 여부
        desc: 진행바 설명
        charge_key: atomic charge 키 이름
        total_charge_key: total charge 키 이름

    Returns:
        graph_list: PyG Data 객체 리스트
    """
    graph_list = []
    iterator = tqdm(data, desc=desc, leave=False) if show_progress else data

    for atoms in iterator:
        crds = atoms.get_positions()
        node_enr_avg = np.array([
            enr_avg_per_element[uniq_element[iz]]
            for iz in atoms.numbers
        ])

        # 에너지 (안전하게 읽기)
        enr = _safe_get_energy(atoms) - node_enr_avg.sum()

        # 힘 (안전하게 읽기)
        if regress_forces or regress_forces == 'direct':
            frc = _safe_get_forces(atoms)
            volume = _safe_get_volume(atoms)
        else:
            frc = np.zeros((len(atoms), 3))
            volume = np.zeros(1)

        # Cell 처리
        cell = np.array(atoms.get_cell())
        if np.all(cell == 0.0):
            cell = np.diag([30., 30., 30.])
            atoms.set_cell(cell)

        # Stress
        stress = _safe_get_stress(atoms)
        if np.all(stress == 0):
            volume = np.zeros(1)

        # 이웃 리스트
        iatoms, jatoms, Sij = neighbour_list(
            quantities='ijS', atoms=atoms, cutoff=cutoff
        )
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        # 최대 이웃 수 제한
        if max_neigh is not None:
            Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)
            nonmax_idx = []
            for i in range(len(atoms)):
                idx_i = (iatoms == i).nonzero()[0]
                idx_sorted = np.argsort(dist[idx_i])[:max_neigh]
                nonmax_idx.append(idx_i[idx_sorted])
            nonmax_idx = np.concatenate(nonmax_idx)
            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            num_edges = iatoms.shape[0]
            Sij = Sij[nonmax_idx]

        # Charge 정보 추출
        atomic_charges, total_charge = _extract_charges(
            atoms, charge_key, total_charge_key
        )

        # PyG Data 객체 생성
        graph = Data(
            positions=torch.tensor(crds, dtype=torch.float32),
            species=torch.tensor(species, dtype=torch.long),
            forces=torch.tensor(frc, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),
            num_nodes=num_nodes,
            num_edges=num_edges,
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(
                np.array(cell), dtype=torch.float32
            ).view(1, 3, 3),
            edge_index=torch.tensor(
                np.array([iatoms, jatoms]), dtype=torch.long
            ),
            stress=torch.tensor(stress, dtype=torch.float32),
            volume=torch.tensor(volume),
            # Charge 관련 추가 필드
            atomic_charges=torch.tensor(
                atomic_charges, dtype=torch.float32
            ),
            total_charge=torch.tensor(
                total_charge, dtype=torch.float32
            ),
        )
        graph_list.append(graph)

    return graph_list


def get_dataloader_charge(
    fname, ntrain, nvalid, nbatch, cutoff, random_seed,
    element=None, regress_forces=True, max_neigh=None,
    charge_key="charges", total_charge_key="total_charge",
    rank=0, world_size=1,
):
    """
    Charge 정보를 포함하는 DataLoader 생성.

    입력 파일로 qm9star_preprocessor.py로 변환된 extended xyz 또는
    일반 ASE 파일을 사용할 수 있습니다.

    Args:
        fname: 구조 파일 경로 (.xyz, .traj 등)
        ntrain: 학습 데이터 개수
        nvalid: 검증 데이터 개수
        nbatch: 배치 크기
        cutoff: 거리 cutoff
        random_seed: 랜덤 시드
        element: 원소 목록 ("auto" 또는 리스트)
        regress_forces: force regression 여부
        max_neigh: 최대 이웃 수
        charge_key: atomic charge 키 이름
        total_charge_key: total charge 키 이름
        rank: DDP rank
        world_size: DDP world_size

    Returns:
        train_loader, valid_loader, uniq_element, enr_avg_per_element
    """
    # 데이터 읽기
    traj = read(fname, index=':')

    # 원소 정보 자동 추출 또는 수동 지정
    if element is None or element == "auto":
        element = sorted(
            list(set(int(atom.number) for atoms in traj for atom in atoms))
        )
    elif isinstance(element, str):
        element = [int(e) for e in element.split()]

    # 원소별 평균 에너지 계산
    enr_avg_per_element, uniq_element, enr_var = get_enr_avg_per_element(traj, element)

    # 랜덤 셔플 후 train/valid 분할
    np.random.seed(random_seed)
    idx = np.random.permutation(len(traj))
    train_data = [traj[i] for i in idx[:ntrain]]
    valid_data = [traj[i] for i in idx[ntrain:ntrain + nvalid]]

    # 그래프 데이터셋 생성 (charge 포함)
    if rank == 0:
        print(f"\n\033[32mConverting training data ({ntrain}) "
              f"with charge info...\033[0m")
    train_graphs = get_graphset_charge(
        train_data, cutoff, uniq_element,
        enr_avg_per_element, enr_var,
        regress_forces=regress_forces,
        max_neigh=max_neigh,
        show_progress=(rank == 0),
        desc="Train",
        charge_key=charge_key,
        total_charge_key=total_charge_key,
    )

    if rank == 0:
        print(f"\033[32mConverting validation data ({nvalid}) "
              f"with charge info...\033[0m")
    valid_graphs = get_graphset_charge(
        valid_data, cutoff, uniq_element,
        enr_avg_per_element, enr_var,
        regress_forces=regress_forces,
        max_neigh=max_neigh,
        show_progress=(rank == 0),
        desc="Valid",
        charge_key=charge_key,
        total_charge_key=total_charge_key,
    )

    # DataLoader 생성
    train_loader = DataLoader(train_graphs, batch_size=nbatch, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=nbatch, shuffle=False)

    return train_loader, valid_loader, uniq_element, enr_avg_per_element


def get_dataloader_charge_to_predict(
    fname, ndata, nbatch, cutoff, model_ckpt,
    regress_forces=True, max_neigh=None,
    charge_key="charges", total_charge_key="total_charge",
):
    """
    평가용 DataLoader 생성 (charge 정보 포함).

    Args:
        fname: 구조 파일 경로
        ndata: 데이터 개수 (파일 경로 또는 정수)
        nbatch: 배치 크기
        cutoff: 거리 cutoff
        model_ckpt: 학습된 모델의 체크포인트
        regress_forces: force regression 여부
        max_neigh: 최대 이웃 수
        charge_key: atomic charge 키 이름
        total_charge_key: total charge 키 이름

    Returns:
        data_loader, uniq_element, enr_avg_per_element
    """
    # 데이터 읽기
    if isinstance(ndata, str):
        traj = read(ndata, index=':')
    else:
        traj = read(fname, index=f':{ndata}')

    # 체크포인트에서 원소 정보 복원
    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']
    enr_var = 1.0  # 예측 시에는 분산을 정규화에 사용하지 않음

    # 그래프 데이터셋 생성
    graphs = get_graphset_charge(
        traj, cutoff, uniq_element,
        enr_avg_per_element, enr_var,
        regress_forces=regress_forces,
        max_neigh=max_neigh,
        show_progress=True,
        desc="Predict",
        charge_key=charge_key,
        total_charge_key=total_charge_key,
    )

    data_loader = DataLoader(graphs, batch_size=nbatch, shuffle=False)
    return data_loader, uniq_element, enr_avg_per_element
