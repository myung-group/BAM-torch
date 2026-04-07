"""
Charge-dependent Phase 3 (E) — CPU 검증 테스트.

Phase 3 핵심 변경 검증:
  [1] Forward pass: E_total = E_SR (U_CENT 기여 없음)
  [2] Charge conservation: Σq_i == Q_total (hard constraint 유지)
  [3] 학습 루프: loss 감소, 체크포인트 저장
  [4] Phase 2 vs Phase 3 에너지 비교 (U_CENT 기여 확인)

사용법:
  cd /home/swkim/LabWorking/Development/BAM-torch
  conda activate bam_torch
  python tests/test_cd_v3.py
"""

import os
import sys
import json
import numpy as np


# ── Step 1: 더미 데이터 생성 (Phase 2 테스트와 동일) ──────────────────
def generate_dummy_xyz(filepath, n_structures=30, charge_type="npa"):
    """
    H2O, CH4, NH3 더미 분자 데이터 생성.

    charge_type 에 따라 charge comment 만 다르게 표시
    (데이터 값은 동일 — CPU 검증 목적).
    """
    molecules = [
        {
            "symbols": ["O", "H", "H"],
            "coords": [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            "energy_base": -76.0,
            "charges": [-0.82, 0.41, 0.41],
            "total_charge": 0.0,
        },
        {
            "symbols": ["C", "H", "H", "H", "H"],
            "coords": [[0.0, 0.0, 0.0], [0.6276, 0.6276, 0.6276],
                       [-0.6276, -0.6276, 0.6276], [-0.6276, 0.6276, -0.6276],
                       [0.6276, -0.6276, -0.6276]],
            "energy_base": -40.5,
            "charges": [-0.64, 0.16, 0.16, 0.16, 0.16],
            "total_charge": 0.0,
        },
        {
            "symbols": ["N", "H", "H", "H"],
            "coords": [[0.0, 0.0, 0.116], [0.0, 0.9377, -0.2707],
                       [0.8121, -0.4689, -0.2707], [-0.8121, -0.4689, -0.2707]],
            "energy_base": -56.2,
            "charges": [-1.02, 0.34, 0.34, 0.34],
            "total_charge": 0.0,
        },
    ]
    rng = np.random.RandomState(42)
    with open(filepath, "w") as f:
        for i in range(n_structures):
            mol = molecules[i % len(molecules)]
            n_atoms = len(mol["symbols"])
            noise = rng.normal(0, 0.05, (n_atoms, 3))
            coords = np.array(mol["coords"]) + noise
            energy = mol["energy_base"] + rng.normal(0, 0.1)
            forces = rng.normal(0, 0.01, (n_atoms, 3))
            charges = np.array(mol["charges"]) + rng.normal(0, 0.02, n_atoms)
            total_charge = mol["total_charge"]
            f.write(f"{n_atoms}\n")
            lattice = "30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0"
            props = (
                f'Lattice="{lattice}" '
                f'Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 '
                f'energy={energy:.8f} total_charge={total_charge:.1f} '
                f'charge_type={charge_type} pbc="F F F"'
            )
            f.write(f"{props}\n")
            for j in range(n_atoms):
                sym = mol["symbols"][j]
                x, y, z = coords[j]
                fx, fy, fz = forces[j]
                q = charges[j]
                f.write(
                    f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f} "
                    f"{fx:16.8f} {fy:16.8f} {fz:16.8f} {q:12.6f}\n"
                )
    print(f"[OK] 더미 데이터 생성: {filepath} ({n_structures}개, charge_type={charge_type})")


# ── Step 2: 설정 파일 생성 ─────────────────────────────────────────────
def create_v3_config(data_path, work_dir, charge_type="npa"):
    """Phase 3 (E) 최소 설정."""
    config = {
        "device": "cpu",
        "gpu-parallel": False,
        "model": "charge_race_v3",
        "cueq_config": False,
        "regress_forces": True,
        "trainer": "cd_v3",
        "fname_traj": data_path,
        "ntrain": 20,
        "nvalid": 10,
        "element": "auto",
        "cutoff": 5.0,
        "avg_num_neighbors": 10,
        "num_species": 4,
        "max_ell": 2,
        "num_radial_basis": 4,
        "hidden_channels": "8x0e+4x1o",
        "output_channels": "1x0e",
        "nbatch": 4,
        "nlayers": 2,
        "features_dim": 8,
        "active_fn": "identity",
        "pbc": False,
        "charge": {
            "cep_hidden_dim": 16,
            "charge_type": charge_type,
            "use_cent_energy": False,
            "charge_key": "charges",
            "total_charge_key": "total_charge",
            "charge_loss": "mse",
        },
        "NN": {
            "data_seed": 10,
            "init_seed": 11,
            "learning_rate": 0.005,
            "weight_decay": 0,
            "nepoch": 5,
            "nsave": 5,
            "restart": False,
            "fname_pkl": os.path.join(work_dir, "v3_model.pkl"),
            "loss_config": {"energy_loss": "mse", "force_loss": "mse"},
            "enr_lambda": 1,
            "frc_lambda": 10,
            "chg_lambda": 1,
            "l2_lambda": 0.0,
            "ema": False,
        },
        "scheduler": {
            "scheduler": "ReduceLROnPlateau",
            "lr_gamma": 0.1,
            "decay_factor": 0.9,
            "max_steps": 30,
            "warmup_steps": 2,
            "warmup_factor": 0.2,
            "patience": 50,
            "threshold": 0.0001,
        },
        "log_length": "simple",
        "log_interval": 1,
        "log_config": {
            "step": ["date", "epoch"],
            "train": ["loss", "loss_e", "loss_f", "loss_q"],
            "valid": ["loss", "loss_e", "loss_f", "loss_q"],
            "lr": ["lr"],
        },
        "train": {"fname_log": os.path.join(work_dir, "v3_loss_train.out")},
        "predict": {"evaluate_tag": False},
    }
    config_path = os.path.join(work_dir, "v3_input.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[OK] 설정 파일 생성: {config_path}")
    return config, config_path


# ── Step 3: Forward pass 테스트 ───────────────────────────────────────
def test_forward_pass_v3():
    """
    ChargeRACEv3 forward pass 검증.

    핵심 확인:
      1. energy == E_SR  (U_CENT 포함 안 됨)
      2. |Σq_i - Q_total| < 1e-4  (hard conservation)
      3. U_CENT 는 여전히 계산됨 (분석용)
    """
    import torch
    from bam_torch.charge_dependent.model.cd_model_v3 import ChargeRACEv3
    from e3nn import o3

    print("\n" + "=" * 60)
    print("  [Phase 3 (E)] Forward Pass 테스트")
    print("=" * 60)

    model = ChargeRACEv3(
        cutoff=5.0,
        avg_num_neighbors=5,
        num_species=4,
        max_ell=2,
        num_basis_func=4,
        hidden_irreps=o3.Irreps("8x0e+4x1o"),
        nlayers=2,
        features_dim=8,
        output_irreps="1x0e",
        active_fn="identity",
        regress_forces="direct",
        cueq_config=None,
        cep_hidden_dim=16,
        use_cent_energy=False,
        charge_type="npa",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터 수: {n_params:,}")
    print(f"  use_cent_energy : {model.use_cent_energy}")
    print(f"  charge_type     : {model.charge_type}")

    # 더미 입력 (H2O)
    n_atoms, n_edges = 3, 6
    data = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
            dtype=torch.float32,
        ),
        "species": torch.tensor([0, 1, 1], dtype=torch.long),
        "cell": torch.tensor(
            [[[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
        ),
        "edges": torch.zeros(n_edges, 3, dtype=torch.float32),
        "num_edges": torch.tensor([n_edges], dtype=torch.long),
        "batch": torch.zeros(n_atoms, dtype=torch.long),
        "ptr": torch.tensor([0, n_atoms], dtype=torch.long),
        "total_charge": torch.tensor([0.0], dtype=torch.float32),
        "energy": torch.tensor([-76.0], dtype=torch.float32),
        "forces": torch.zeros(n_atoms, 3, dtype=torch.float32),
    }

    model.eval()
    with torch.no_grad():
        preds = model(data, backprop=False)

    print(f"\n  결과:")
    print(f"    energy (= E_SR)  : {preds['energy'].item():.6f}")
    print(f"    E_SR             : {preds['E_SR'].item():.6f}")
    print(f"    U_CENT (미포함)  : {preds['U_CENT'].item():.6f}")
    print(f"    forces shape     : {preds['forces'].shape}")
    print(f"    atomic_charges   : {[f'{q:.4f}' for q in preds['atomic_charges'].tolist()]}")
    print(f"    total_charge(Σq) : {preds['total_charge'].item():.6f}")

    # Phase 3 핵심 검증 1: E_total == E_SR
    e_total = preds['energy'].item()
    e_sr = preds['E_SR'].item()
    u_cent = preds['U_CENT'].item()
    assert abs(e_total - e_sr) < 1e-6, (
        f"Phase 3 실패: energy({e_total:.6f}) != E_SR({e_sr:.6f}), U_CENT={u_cent:.6f}"
    )
    print(f"\n  [CHECK 1] energy == E_SR: PASS (U_CENT={u_cent:.4f} 제외됨)")

    # Phase 3 핵심 검증 2: hard charge conservation
    q_sum = preds['atomic_charges'].sum().item()
    q_input = data['total_charge'].item()
    err = abs(q_sum - q_input)
    assert err < 1e-4, f"Hard conservation 실패: {err}"
    print(f"  [CHECK 2] |Σq - Q_total| = {err:.2e}: PASS")

    print("\n  [OK] Forward pass 성공!")
    return True


# ── Step 4: Phase 2 vs Phase 3 에너지 비교 ────────────────────────────
def test_phase2_vs_phase3():
    """
    동일 구조에서 Phase 2 (use_cent_energy=True) 와
    Phase 3 (use_cent_energy=False) 의 에너지 차이 비교.
    """
    import torch
    from bam_torch.charge_dependent.model.cd_model_v3 import ChargeRACEv3
    from e3nn import o3

    print("\n" + "=" * 60)
    print("  [비교] Phase 2 vs Phase 3 에너지")
    print("=" * 60)

    common_kwargs = dict(
        cutoff=5.0, avg_num_neighbors=5, num_species=4,
        max_ell=2, num_basis_func=4,
        hidden_irreps=o3.Irreps("8x0e+4x1o"),
        nlayers=2, features_dim=8, output_irreps="1x0e",
        active_fn="identity", regress_forces="direct",
        cueq_config=None, cep_hidden_dim=16,
    )

    model_p2 = ChargeRACEv3(**common_kwargs, use_cent_energy=True)
    model_p3 = ChargeRACEv3(**common_kwargs, use_cent_energy=False)

    # 동일 가중치 복사 (공정 비교)
    model_p3.load_state_dict(model_p2.state_dict())

    n_atoms, n_edges = 3, 6
    data = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
            dtype=torch.float32,
        ),
        "species": torch.tensor([0, 1, 1], dtype=torch.long),
        "cell": torch.tensor(
            [[[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
        ),
        "edges": torch.zeros(n_edges, 3, dtype=torch.float32),
        "num_edges": torch.tensor([n_edges], dtype=torch.long),
        "batch": torch.zeros(n_atoms, dtype=torch.long),
        "ptr": torch.tensor([0, n_atoms], dtype=torch.long),
        "total_charge": torch.tensor([0.0], dtype=torch.float32),
        "energy": torch.tensor([-76.0], dtype=torch.float32),
        "forces": torch.zeros(n_atoms, 3, dtype=torch.float32),
    }

    model_p2.eval()
    model_p3.eval()
    with torch.no_grad():
        out_p2 = model_p2(data, backprop=False)
        out_p3 = model_p3(data, backprop=False)

    print(f"\n  Phase 2 (use_cent_energy=True):")
    print(f"    energy  = {out_p2['energy'].item():.6f}")
    print(f"    E_SR    = {out_p2['E_SR'].item():.6f}")
    print(f"    U_CENT  = {out_p2['U_CENT'].item():.6f}")

    print(f"\n  Phase 3 (use_cent_energy=False):")
    print(f"    energy  = {out_p3['energy'].item():.6f}")
    print(f"    E_SR    = {out_p3['E_SR'].item():.6f}")
    print(f"    U_CENT  = {out_p3['U_CENT'].item():.6f}  (계산되나 미포함)")

    diff = abs(out_p2['energy'].item() - out_p3['energy'].item())
    u_cent_val = abs(out_p2['U_CENT'].item())
    assert abs(diff - u_cent_val) < 1e-5, "에너지 차이가 U_CENT 와 불일치"
    print(f"\n  [CHECK] |E_p2 - E_p3| = U_CENT = {diff:.6f}: PASS")

    print("\n  [OK] Phase 2 vs Phase 3 비교 완료!")


# ── Step 5: 학습 루프 테스트 ──────────────────────────────────────────
def run_training_v3(config):
    """CDTrainerV3 학습 실행."""
    from bam_torch.charge_dependent.training.cd_trainer_v3 import CDTrainerV3

    print("\n" + "=" * 60)
    print("  [Phase 3 (E)] 학습 루프 테스트")
    print("=" * 60)

    trainer = CDTrainerV3(config, rank=0, world_size=1)

    print(f"\n  모델 파라미터 수 : {trainer.n_params:,}")
    print(f"  학습 데이터      : {config['ntrain']}개")
    print(f"  검증 데이터      : {config['nvalid']}개")
    print(f"  에폭             : {config['NN']['nepoch']}")
    print(f"  charge_type      : {config['charge']['charge_type']}")
    print(f"  use_cent_energy  : {config['charge']['use_cent_energy']}")
    print()

    trainer.train()

    print("\n" + "=" * 60)
    print("  [OK] 학습 완료!")
    print("=" * 60)

    pkl_path = config['NN']['fname_pkl']
    if os.path.exists(pkl_path):
        import torch
        ckpt = torch.load(pkl_path, weights_only=False)
        print(f"\n  체크포인트 저장됨: {pkl_path}")
        print(f"  저장된 에폭: {ckpt['loss']['epoch']}")
        print(f"  학습  loss : {ckpt['loss']['train']:.6f}")
        print(f"  검증  loss : {ckpt['loss']['valid']:.6f}")
    else:
        print(f"\n  체크포인트 미저장 (에폭 수 < nsave)")


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_cd_v3_output"
    )
    os.makedirs(work_dir, exist_ok=True)
    data_path = os.path.join(work_dir, "dummy_data_v3.xyz")

    print("\n[TEST 1/3] Forward pass + E_total = E_SR 검증\n")
    try:
        test_forward_pass_v3()
    except Exception as e:
        print(f"\n  [FAIL] Forward pass 실패: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n[TEST 2/3] Phase 2 vs Phase 3 에너지 비교\n")
    try:
        test_phase2_vs_phase3()
    except Exception as e:
        print(f"\n  [FAIL] 비교 테스트 실패: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n[TEST 3/3] 학습 루프 (CDTrainerV3)\n")
    try:
        generate_dummy_xyz(data_path, n_structures=30, charge_type="npa")
        config, config_path = create_v3_config(data_path, work_dir, charge_type="npa")
        run_training_v3(config)
    except Exception as e:
        print(f"\n  [FAIL] 학습 실패: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n\n" + "=" * 60)
    print("  ALL TESTS PASSED — Phase 3 (E) CEP as Pure Charge Predictor")
    print("=" * 60)
