"""
Charge-dependent trainer Phase 3 (E) for BAM-torch.

Phase 3 (E) design — CEP as Pure Charge Predictor:
  - E_total = E_SR only (U_CENT energy contribution removed)
  - charge_type parameter for NPA / Mulliken / Hirshfeld switching
  - Hard charge conservation (CEP Lagrange analytical solution) maintained
  - Resolves physical mismatch: learns charges without QEq energy assumptions

Differences from Phase 2 (CDTrainer):
  - use_cent_energy: False (default, overridable via config)
  - charge_type: 'npa' / 'mulliken' / 'hirshfeld' selection
  - Uses ChargeRACEv3 model

Phase 3.5 changes (Issue #5):
  - parse_model_config() / parse_charge_config() centralized utils for parameter mapping
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
    Phase 3 (E) Charge-dependent model Trainer.

    Inherits CDTrainer and overrides only set_model():
      - Uses ChargeRACEv3 model
      - use_cent_energy=False (E_total = E_SR)
      - charge_type config support

    Loss composition (same as Phase 2):
      total_loss = enr_lambda * loss_E
                 + frc_lambda * loss_F
                 + chg_lambda * loss_Q  (pred vs target charges)

    Config example (charge section):
      "charge": {
          "cep_hidden_dim": 64,
          "charge_type": "mulliken",   // "npa" / "mulliken" / "hirshfeld"
          "use_cent_energy": false,    // true reverts to Phase 2 behavior
          "charge_key": "charges",
          "total_charge_key": "total_charge",
          "charge_loss": "mse"
      }
    """

    def __init__(self, json_data, rank=0, world_size=1):
        super().__init__(json_data, rank, world_size)

    def set_model(self):
        """Configure ChargeRACEv3 (Phase 3, CEP as pure charge predictor) model.

        Uses centralized parse_model_config() for config -> constructor
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
