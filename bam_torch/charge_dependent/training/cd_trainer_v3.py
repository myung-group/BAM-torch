"""
Charge-dependent trainer Phase 3 (E) for BAM-torch.

Phase 3 (E) 설계 — CEP as Pure Charge Predictor:
  - E_total = E_SR only (U_CENT 에너지 기여 제거)
  - charge_type 파라미터로 NPA / Mulliken / Hirshfeld 전환 가능
  - Hard charge conservation (CEP Lagrange 해석해) 유지
  - 물리적 mismatch 해결: QEq 에너지 가정 없이 전하만 학습

Phase 2 (CDTrainer) 와의 차이:
  - use_cent_energy: False (기본값, config 로 override 가능)
  - charge_type: 'npa' / 'mulliken' / 'hirshfeld' 선택
  - ChargeRACEv3 모델 사용

Phase 3.5 변경 (Issue #5):
  - parse_model_config() / parse_charge_config() 중앙 유틸로 파라미터 매핑
"""

from bam_torch.charge_dependent.training.cd_trainer import CDTrainer
from bam_torch.charge_dependent.model import MODEL_REGISTRY
from bam_torch.utils.model_config import (
    parse_model_config,
    parse_cueq_config,
    parse_charge_config,
)


class CDTrainerV3(CDTrainer):
    """
    Phase 3 (E) Charge-dependent 모델 학습 Trainer.

    CDTrainer 를 상속하며 set_model() 만 오버라이드:
      - ChargeRACEv3 모델 사용
      - use_cent_energy=False (E_total = E_SR)
      - charge_type config 지원

    Loss 구성 (Phase 2 와 동일):
      total_loss = enr_lambda × loss_E
                 + frc_lambda × loss_F
                 + chg_lambda × loss_Q  (pred vs target charges)

    Config 예시 (charge 섹션):
      "charge": {
          "cep_hidden_dim": 64,
          "charge_type": "mulliken",   # "npa" / "mulliken" / "hirshfeld"
          "use_cent_energy": false,    # true 이면 Phase 2 동작
          "charge_key": "charges",
          "total_charge_key": "total_charge",
          "charge_loss": "mse"
      }
    """

    def __init__(self, json_data, rank=0, world_size=1):
        super().__init__(json_data, rank, world_size)

    def set_model(self):
        """ChargeRACEv3 (Phase 3, CEP as pure charge predictor) 모델 구성.

        Uses centralized parse_model_config() for config → constructor
        parameter mapping (Phase 3.5, Issue #5 fix).
        """
        # Common model parameters (centralized mapping)
        model_kwargs = parse_model_config(self.json_data)

        # CuEquivariance
        cueq_config = parse_cueq_config(self.json_data)
        model_kwargs['cueq_config'] = cueq_config
        if cueq_config is not None:
            self.msg += '\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
        else:
            self.msg += '\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'

        # Phase 3 charge-dependent parameters
        cd_kwargs = parse_charge_config(self.json_data)
        model_kwargs.update(cd_kwargs)

        # Instantiate model via registry
        model_name = self.json_data["model"].lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"Unknown charge-dependent model: {self.json_data['model']}"
            )

        model = model_cls(**model_kwargs)

        energy_mode = (
            "E_SR + U_CENT" if cd_kwargs['use_cent_energy'] else "E_SR only"
        )
        self.msg += (
            f'\n\033[33m -- Phase 3 (E): CEP as pure charge predictor\033[0m\n'
            f'\033[33m -- charge_type : {cd_kwargs["charge_type"]}\033[0m\n'
            f'\033[33m -- energy mode : {energy_mode}\033[0m\n'
        )
        return model
