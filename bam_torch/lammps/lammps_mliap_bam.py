"""LAMMPS ML-IAP (unified, python) adapter for BAM-torch RACE models.

Runs the model inside LAMMPS through ``pair_style mliap unified`` with the
eager PyTorch stack, so kernel accelerations that cannot be serialized to
TorchScript (OpenEquivariance, cuEquivariance, torch.compile) work in LAMMPS
- unlike the libtorch ``pair_style bam`` path, which is limited to e3nn.

Design (modeled on MACE's lammps_mliap_mace.py):
  - LAMMPS provides per-pair vectors rij = r_j - r_i (full directed list with
    i local), per-atom element indices ``elems`` (0-based into element_types),
    and ghost atoms.  RACE expects Rij = r_receiver - r_sender, so the model
    is fed sender=pair_j, receiver=pair_i and model_vectors = -rij.
  - Per-pair forces: fij = dE/d(rij) = -grad(E, model_vectors); LAMMPS applies
    f[i] += fij, f[j] -= fij and derives the virial, so NPT works.
  - Multi-layer message passing needs ghost features refreshed between layers.
    LAMMPS_MP wraps data.forward_exchange / reverse_exchange, which exist only
    in the KOKKOS ML-IAP coupling -> run with ``-k on g 1 -sf kk``.  A no-ghost
    system (isolated molecule) also works on the plain coupling.
  - Element mapping comes from the checkpoint's ``uniq_element`` {Z: species}
    by default; pass ``elements=[...]`` (symbols in species order) to override
    for checkpoints whose stored table does not reflect the trained encoding
    (e.g. omol/opoly-style data encodes species as Z-1 while storing an
    identity table - use elements=chemical_symbols[1:num_species+1]).

Export/load (the object is rebuilt from the pkl at unpickling time via
__reduce__, so OEQ JIT modules never need to be pickled):

    from bam_torch.lammps.lammps_mliap_bam import rebuild_bam_mliap
    import torch
    obj = torch.save(rebuild_bam_mliap("model.pkl", backend="oeq"),
                     "bam_mliap_oeq.pt")

LAMMPS input:

    pair_style mliap unified /path/bam_mliap_oeq.pt 0
    pair_coeff * * H C O        # element names for each LAMMPS atom type

Runtime requirements: the embedded python must import this module and
bam_torch (put both on PYTHONPATH along with <lammps>/python); the KOKKOS
coupling needs ``cupy``; torch>=2.6 needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
(the .pt stores a rebuild callable); torch<2.6 with OEQ needs a
``torch.library.register_autocast`` no-op shim (OEQ imports it at module
scope).
"""
from typing import Dict, List, Optional

import torch
from ase.data import chemical_symbols

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:
    class MLIAPUnified:                       # allows import outside LAMMPS
        def __init__(self, *a, **k):
            pass


class LAMMPS_MP(torch.autograd.Function):
    """Ghost-atom feature exchange (KOKKOS coupling only)."""

    @staticmethod
    def forward(ctx, feats, data):
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


class BAMEdgeModel(torch.nn.Module):
    """RACE forward re-expressed over LAMMPS pair vectors with ghost exchange.

    Mirrors RACE.forward exactly (embedding -> per-layer interaction/product/
    readout -> act_fn -> sum over layers) but takes edge vectors as the
    differentiable input and refreshes ghost features between layers.
    """

    def __init__(self, race, enr_avg_species):
        super().__init__()
        self.race = race
        for p in self.race.parameters():
            p.requires_grad = False
        n = race.num_species
        enr = torch.zeros(n, dtype=torch.float32)
        for s, v in (enr_avg_species or {}).items():
            if 0 <= int(s) < n:
                enr[int(s)] = float(v)
        self.register_buffer("enr_avg", enr)
        self.register_buffer("r_max", torch.tensor(float(race.cutoff)))
        self.nlayers = race.nlayers

    def forward(self, vectors, species, edge_index, nlocal, lammps_data):
        race = self.race
        Rij = vectors / race.cutoff
        node_attrs = torch.nn.functional.one_hot(
            species, num_classes=race.num_species
        ).to(next(race.parameters()).dtype)
        node_feats = race.node_embedding(node_attrs)
        lengths = torch.norm(Rij, dim=1)
        edge_attrs = race.spherical_harmonics(Rij)
        edge_feats = race.radial_embedding(lengths.unsqueeze(1), node_attrs,
                                           edge_index, species)
        x_node_feats = race.linear_x(node_feats)

        outputs = []
        n_l = len(race.interactions)
        for k in range(n_l):
            node_feats, sc = race.interactions[k](
                node_attrs=node_attrs, node_feats=node_feats,
                edge_attrs=edge_attrs, edge_feats=edge_feats,
                edge_index=edge_index)
            node_feats = race.products[k](
                x_node_feats=x_node_feats, node_feats=node_feats, sc=sc)
            outputs.append(race.readouts[k](node_feats)[:, 0])
            if k < n_l - 1 and lammps_data is not None:
                node_feats = LAMMPS_MP.apply(node_feats.contiguous(),
                                             lammps_data)

        node_energy = race.act_fn(torch.stack(outputs, dim=-1)).sum(dim=-1)
        return node_energy[:nlocal] + self.enr_avg[species[:nlocal]]


class LAMMPS_MLIAP_BAM(MLIAPUnified):
    """BAM-RACE integration for LAMMPS via the ML-IAP unified interface."""

    def __init__(self, race, enr_avg_species, elements: Optional[List[str]] = None):
        super().__init__()
        self.model = BAMEdgeModel(race, enr_avg_species)
        if elements is None:
            elements = [chemical_symbols[z]
                        for z in range(1, race.num_species + 1)]
        self.element_types: List[str] = list(elements)
        self.ndescriptors = 1
        self.nparams = 1
        self.rcutfac = 0.5 * float(race.cutoff)     # MACE convention
        self.device = "cpu"
        self.initialized = False
        self._factory_args = None

    def _initialize(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()
        if using_kokkos:
            self.device = torch.as_tensor(data.elems).device
        else:
            self.device = torch.device("cpu")
        self.has_exchange = hasattr(data, "forward_exchange")
        self.model = self.model.to(self.device)
        self.initialized = True

    def compute_forces(self, data):
        natoms = int(data.nlocal)
        npairs = int(data.npairs)
        if not self.initialized:
            self._initialize(data)
        if natoms == 0 or npairs <= 1:
            return

        species = torch.as_tensor(data.elems, dtype=torch.int64,
                                  device=self.device)
        rij = torch.as_tensor(data.rij).to(torch.float32).to(self.device)
        vectors = (-rij).detach().requires_grad_(True)      # RACE: recv - send
        edge_index = torch.stack([
            torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
            torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
        ], dim=0)                                            # [senders, receivers]

        if (not self.has_exchange) and self.model.nlayers > 1 \
                and int(edge_index.max()) >= natoms:
            raise RuntimeError(
                "Ghost atoms participate in pairs but this ML-IAP coupling "
                "has no forward_exchange(); multi-layer models need the "
                "KOKKOS coupling - run with '-k on g 1 -sf kk'.")

        node_energy_local = self.model(
            vectors, species, edge_index, natoms,
            data if self.has_exchange else None)
        total_e = node_energy_local.sum()
        (grad,) = torch.autograd.grad([total_e], [vectors])
        if self.device.type != "cpu":
            torch.cuda.synchronize()

        try:                                   # kokkos coupling: writable view
            eatoms = torch.as_tensor(data.eatoms)
            eatoms.copy_(node_energy_local.detach().to(eatoms.dtype))
        except (AttributeError, TypeError):    # plain coupling: setter only
            data.eatoms = node_energy_local.detach().cpu().double().numpy()
        data.energy = float(node_energy_local.sum().item())
        pf = (-grad).to(torch.float64)         # fij = dE/d(rij)
        if hasattr(data, "update_pair_forces_gpu") and self.device.type != "cpu":
            data.update_pair_forces_gpu(pf)
        else:
            data.update_pair_forces(pf.cpu().numpy())

    def __reduce__(self):
        args = getattr(self, "_factory_args", None)
        if args:
            return (rebuild_bam_mliap, tuple(args))
        return super().__reduce__()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass


def rebuild_bam_mliap(pkl_path: str, backend: str = "e3nn",
                      elements: Optional[List[str]] = None):
    """Build a LAMMPS_MLIAP_BAM from a training checkpoint.

    Used both for export and for unpickling (via __reduce__), so backend
    modules that cannot be pickled (OEQ JIT) are rebuilt at load time.

    Args:
        pkl_path: RACE training checkpoint (model.pkl).
        backend:  'e3nn' (default) or 'oeq' (requires openequivariance).
        elements: chemical symbols in *species order*, overriding the mapping
            derived from the checkpoint's uniq_element.  Required for
            checkpoints whose stored table does not match the trained
            encoding (omol/opoly-style 0-based z-table: pass
            ase.data.chemical_symbols[1:num_species + 1]).
    """
    from e3nn import o3
    from bam_torch.model.models import RACE

    ck = torch.load(pkl_path, map_location="cpu", weights_only=False)
    cfg = ck["input.json"]
    kw = {}
    if backend == "oeq":
        from bam_torch.model.wrapper_ops import OEQConfig, OEQ_AVAILABLE
        if not OEQ_AVAILABLE:
            raise RuntimeError("backend='oeq' requested but openequivariance "
                               "is not importable")
        kw["oeq_config"] = OEQConfig(enabled=True, optimize_all=True)
    race = RACE(cutoff=cfg["cutoff"], avg_num_neighbors=cfg["avg_num_neighbors"],
                num_species=cfg["num_species"], max_ell=cfg["max_ell"],
                num_basis_func=cfg["num_radial_basis"],
                hidden_irreps=o3.Irreps(cfg["hidden_channels"]),
                nlayers=cfg["nlayers"], features_dim=cfg["features_dim"],
                output_irreps=o3.Irreps(cfg.get("output_channels", "1x0e")),
                active_fn=cfg.get("active_fn", "identity"),
                regress_forces="false",
                interaction_block=cfg.get("interaction_block", "slow"), **kw)
    sd = {k.replace("module.", ""): v for k, v in ck["params"].items()}
    race.load_state_dict(sd, strict=False)   # e3nn-only buffers differ per backend
    ema_state = ck.get("ema_state")
    if ema_state and ema_state.get("shadow_params"):
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(race.parameters(),
                                       decay=ema_state.get("decay", 0.999))
        ema.load_state_dict(ema_state)
        ema.copy_to(race.parameters())

    # This adapter does not implement the e_corr (scale_shift) energy
    # correction that pair_bam applies in lammps_bam.py.  A checkpoint
    # trained with use_scale_shift=True would silently lose a constant
    # per-atom energy offset here, so fail loudly instead.
    _vss = ck.get("valid_scale_shift")
    if _vss:
        _vals = list(_vss.values()) if isinstance(_vss, dict) else list(_vss)
        _ec = float(torch.tensor([float(v) for v in _vals]).flatten().mean())
        if abs(_ec) > 1e-12:
            raise RuntimeError(
                "checkpoint has e_corr=%.6g but this ML-IAP adapter ignores "
                "it; implement e_corr (see bam_torch/lammps/lammps_bam.py) "
                "before using this checkpoint." % _ec)

    race.criterion = None
    race.eval()

    # species-indexed reference energies + element list from the checkpoint
    uniq = ck.get("uniq_element") or {}
    enr_raw = ck.get("enr_avg_per_element") or {}
    if elements is not None:
        # explicit override: species s <-> elements[s]; reference energies are
        # remapped assuming enr_raw is keyed by atomic number
        sym_to_z = {chemical_symbols[z]: z for z in range(1, 119)}
        enr = {s: float(enr_raw.get(sym_to_z.get(sym, -1), 0.0))
               for s, sym in enumerate(elements)}
    else:
        elements = [None] * cfg["num_species"]
        enr = {}
        for z, s in uniq.items():
            z, s = int(z), int(s)
            if 1 <= z < len(chemical_symbols) and 0 <= s < len(elements):
                elements[s] = chemical_symbols[z]
                enr[s] = float(enr_raw.get(z, 0.0))
        elements = [e or "X" for e in elements]

    obj = LAMMPS_MLIAP_BAM(race, enr, elements=elements)
    obj._factory_args = (pkl_path, backend, list(elements))
    return obj
