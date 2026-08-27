"""LAMMPS ML-IAP (unified, python) adapter for BAM-torch RACE models.

Modeled on MACE's lammps_mliap_mace.py. Key design points:
  - LAMMPS gives per-pair vectors rij = r_j - r_i (full directed list, i local),
    per-atom element indices `elems` (0-based into element_types), and ghost
    atoms. We feed RACE with sender=pair_j, receiver=pair_i and
    model_vectors = -rij  (RACE convention Rij = r_receiver - r_sender).
  - Per-pair forces: fij = dE/d(rij) = -grad(E, model_vectors); LAMMPS applies
    f[i] += fij, f[j] -= fij via update_pair_forces.
  - Multi-layer ghost refresh: LAMMPS_MP autograd fn wrapping
    data.forward_exchange / reverse_exchange (KOKKOS coupling only).
  - element_types = chemical symbols for Z=1..num_species in order, so LAMMPS's
    elems index == trained 0-based species (Z-1) automatically.
  - eatoms/energy set directly (MACE pattern); per-atom reference energies
    (enr_avg, species-indexed) included so energies are DFT-comparable.
"""
from typing import List

import os
import torch
from ase.data import chemical_symbols


def node_dtype_ref(race):
    return next(race.parameters()).dtype

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:
    class MLIAPUnified:                       # allows import outside LAMMPS
        def __init__(self, *a, **k):
            pass


class LAMMPS_MP(torch.autograd.Function):
    """Ghost-atom feature exchange (copied from MACE mace/tools/utils.py)."""

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

    Mirrors bam_torch RACE.forward exactly (embedding -> per-layer
    interaction/product/readout -> identity act -> sum over layers), but takes
    edge vectors as the differentiable input and refreshes ghost features
    between layers.
    """

    def __init__(self, race, enr_avg_species, bond_cutoff=3.2, bond_cut_mat=None,
                 bond_topology=None, prior_config=None, n_beads_total=0,
                 e_corr=0.0):
        super().__init__()
        self.race = race
        self.use_bond_flag = bool(getattr(race, "use_bond_flag", False))
        self.register_buffer("bond_cutoff", torch.tensor(float(bond_cutoff)))
        # Per-type-pair bond cutoffs (num_species x num_species). A single global
        # scalar cutoff misclassifies systems where bonded and inter-molecular
        # contact distances overlap (for octanol the non-bonded (A4,OH) pair gets
        # as close as 2.63 A while the longest bond is 2.84 A). Type pairs that are
        # not bonded in the topology are left at 0 so they can never be flagged.
        if bond_cut_mat is not None:
            m = torch.tensor(bond_cut_mat, dtype=torch.float32)
            self.register_buffer("bond_cut_mat", m)
            self.use_bond_mat = True
        else:
            n = int(race.num_species)
            self.register_buffer("bond_cut_mat",
                                 torch.full((n, n), float(bond_cutoff), dtype=torch.float32))
            self.use_bond_mat = False

        # Topology-based bond lookup. When LAMMPS supplies atom tags (patched
        # build) this is used instead of distances: a distance criterion is
        # fundamentally ambiguous wherever the bonded and non-bonded distance
        # distributions overlap. Bonded pairs are pre-expanded into a set of
        # global (tag_i, tag_j) keys, which also covers mixtures (e.g. an
        # octanol 5-bead + water 1-bead interface) and branched topologies.
        # Two schema forms are accepted:
        #   1) {"n_beads_per_mol": 5, "bonds": [[0,1],...]}   (uniform system)
        #   ② {"segments": [{"n_beads_per_mol":5,"count":100,"bonds":[...]},
        #                   {"n_beads_per_mol":1,"count":900,"bonds":[]}]}  (mixture)
        segs = None
        if bond_topology:
            if bond_topology.get("segments"):
                segs = bond_topology["segments"]
            elif bond_topology.get("n_beads_per_mol"):
                segs = [{"n_beads_per_mol": int(bond_topology["n_beads_per_mol"]),
                         "count": None, "bonds": bond_topology.get("bonds", [])}]
        N = int(n_beads_total or 0)
        self.n_beads_total = N
        keys = []
        mol_of = torch.full((max(N, 1),), -1, dtype=torch.int64)
        if segs and N > 0:
            off, mol_id = 0, 0
            for sg in segs:
                npm = int(sg["n_beads_per_mol"])
                cnt = sg.get("count")
                cnt = int(cnt) if cnt else max(0, (N - off) // npm)
                for m_ in range(cnt):
                    base = off + m_ * npm
                    mol_of[base:base + npm] = mol_id
                    for a, b in sg.get("bonds", []):
                        i_, j_ = base + int(a), base + int(b)
                        lo, hi = (i_, j_) if i_ < j_ else (j_, i_)
                        keys.append(lo * N + hi)
                    mol_id += 1
                off += cnt * npm
        self.register_buffer("mol_of_tag", mol_of)
        if segs and N > 0:
            self.register_buffer("bond_keys",
                                 torch.tensor(sorted(set(keys)), dtype=torch.int64)
                                 if keys else torch.zeros(0, dtype=torch.int64))
            self.has_topology = True
        else:
            self.register_buffer("bond_keys", torch.zeros(0, dtype=torch.int64))
            self.has_topology = False

        # Repulsive prior for delta learning. The harmonic bond term is handled by
        # LAMMPS `bond_style harmonic` instead: it acts on the data-file topology,
        # so it still applies to bonds stretched beyond the pair cutoff, which a
        # pair-list based term cannot do. Intra-molecular exclusion uses atom tags.
        self.has_repulsive = False
        nsp = int(race.num_species)
        sig = torch.full((nsp, nsp), 3.0, dtype=torch.float32)
        if prior_config and prior_config.get("type") == "harmonic_repulsive":
            rep = prior_config.get("repulsive") or {}
            sd = rep.get("sigma_matrix") or {}
            if sd:
                vals = [float(v) for v in sd.values()]
                sig.fill_(sum(vals) / len(vals))          # same default as the reference prior
                for k_, v in sd.items():
                    a, b = (int(x) for x in k_.split("-"))
                    if a < nsp and b < nsp:
                        sig[a, b] = float(v); sig[b, a] = float(v)
            self.rep_eps = float(rep.get("epsilon", 0.001))
            self.rep_rc = float(rep.get("cutoff", 10.0))
            self.rep_maxf = float(rep.get("max_force") or 1.0)
            self.has_repulsive = True
        else:
            self.rep_eps = 0.0; self.rep_rc = 0.0; self.rep_maxf = 1.0
        self.register_buffer("sigma_mat", sig)

        for p in self.race.parameters():
            p.requires_grad = False
        n = race.num_species
        enr = torch.zeros(n, dtype=torch.float32)
        for s, v in enr_avg_species.items():
            if 0 <= int(s) < n:
                enr[int(s)] = float(v)
        self.register_buffer("enr_avg", enr)
        # e_corr (= valid_scale_shift_origin): LAMMPS_BAM on the pair_bam path
        # divides it by the atom count and adds it to every local atom. It was
        # missing here, which shifted the total energy for any model with
        # e_corr != 0; match the pair_bam behaviour.
        self.register_buffer("e_corr", torch.tensor(float(e_corr), dtype=torch.float32))
        self.register_buffer("r_max", torch.tensor(float(race.cutoff)))
        self.nlayers = race.nlayers

    def pair_keys(self, tag_send, tag_recv):
        lo = torch.minimum(tag_send, tag_recv)
        hi = torch.maximum(tag_send, tag_recv)
        return lo * self.n_beads_total + hi

    def topo_edge_bond(self, tag_send, tag_recv):
        """Exact bonded/non-bonded decision from global bead IDs (same rule as
        pair_bam). Being a set lookup, it also covers mixtures and branched
        topologies unchanged.
        """
        return torch.isin(self.pair_keys(tag_send, tag_recv), self.bond_keys).long()

    def same_molecule(self, tag_send, tag_recv):
        """Exclusion test for the repulsive prior (all intra-molecular pairs).
        Correct for mixtures as well."""
        if not self.has_topology:
            return torch.zeros_like(tag_send, dtype=torch.bool)
        return self.mol_of_tag[tag_send] == self.mol_of_tag[tag_recv]

    def repulsive_pair_forces(self, vectors, species, edge_index, tag_send, tag_recv,
                              double_counted: bool):
        """Pair forces of the purely repulsive V = 4*eps*(sigma/r)^12*fc(r) term,
        applied to inter-molecular pairs only.

        Identical to RepulsiveLJPriorTorch (cosine cutoff plus max_force clamp);
        the exclusion of all intra-molecular pairs is decided from atom tags.
        LAMMPS applies f[i] += fij and f[j] -= fij, so with a full neighbour list
        each direction contributes half.
        """
        same_mol = self.same_molecule(tag_send, tag_recv)
        r = torch.norm(vectors, dim=1)
        active = (~same_mol) & (r > 1e-10) & (r < self.rep_rc)
        f = torch.zeros_like(vectors)
        if not bool(active.any()):
            return f
        rv = vectors[active]; rr = r[active]
        sig = self.sigma_mat[species[edge_index[0]][active],
                             species[edge_index[1]][active]]
        sr12 = (sig / rr).pow(12)
        pi = 3.141592653589793
        x = pi * rr / self.rep_rc
        fc = 0.5 * (1.0 + torch.cos(x))
        dfc = -0.5 * pi / self.rep_rc * torch.sin(x)
        dV_dr = 4.0 * self.rep_eps * (-12.0 * sr12 / rr * fc + sr12 * dfc)
        dV_dr = torch.clamp(dV_dr, -self.rep_maxf, self.rep_maxf)
        scale = 0.5 if double_counted else 1.0
        f[active] = (-scale * dV_dr).unsqueeze(1) * (rv / rr.unsqueeze(1))
        return f

    def forward(self, vectors, species, edge_index, nlocal, lammps_data,
                edge_bond_topo=None):
        race = self.race
        Rij = vectors / race.cutoff
        node_attrs = torch.nn.functional.one_hot(
            species, num_classes=race.num_species).to(node_dtype_ref(race))
        node_feats = race.node_embedding(node_attrs)
        lengths = torch.norm(Rij, dim=1)
        edge_attrs = race.spherical_harmonics(Rij)
        edge_feats = race.radial_embedding(lengths.unsqueeze(1), node_attrs,
                                           edge_index, species)
        edge_bond = None
        if self.use_bond_flag:
            if edge_bond_topo is not None:
                edge_bond = edge_bond_topo            # topology based (exact)
            else:
                dist_A = lengths * race.cutoff        # distance fallback (approximate)
                thr = self.bond_cut_mat[species[edge_index[0]], species[edge_index[1]]]
                edge_bond = (dist_A < thr).long()
            edge_feats = torch.cat([edge_feats, edge_bond.unsqueeze(-1).float()], dim=-1)
        x_node_feats = race.linear_x(node_feats)

        outputs = []
        n_l = len(race.interactions)
        for k in range(n_l):
            node_feats, sc = race.interactions[k](
                node_attrs=node_attrs, node_feats=node_feats,
                edge_attrs=edge_attrs, edge_feats=edge_feats,
                edge_index=edge_index, edge_bond=edge_bond)
            node_feats = race.products[k](
                x_node_feats=x_node_feats, node_feats=node_feats, sc=sc)
            outputs.append(race.readouts[k](node_feats)[:, 0])
            if k < n_l - 1 and lammps_data is not None:
                node_feats = LAMMPS_MP.apply(node_feats.contiguous(), lammps_data)

        node_energy = race.act_fn(torch.stack(outputs, dim=-1)).sum(dim=-1)
        node_energy_local = node_energy[:nlocal] + self.enr_avg[species[:nlocal]]
        if float(self.e_corr) != 0.0 and nlocal > 0:
            node_energy_local = node_energy_local + self.e_corr / float(nlocal)
        return node_energy_local


class LAMMPS_MLIAP_BAM(MLIAPUnified):
    """BAM-RACE integration for LAMMPS via the ML-IAP unified interface."""

    def __init__(self, race, enr_avg_species, bond_cutoff=3.2, bond_cut_mat=None,
                 bond_topology=None, prior_config=None, n_beads_total=0,
                 e_corr=0.0, **kwargs):
        super().__init__()
        self._topo_logged = False
        self._double_counted = None
        self._tag_s = None
        self._tag_r = None
        self.model = BAMEdgeModel(race, enr_avg_species, bond_cutoff=bond_cutoff,
                                  bond_cut_mat=bond_cut_mat,
                                  bond_topology=bond_topology,
                                  prior_config=prior_config,
                                  n_beads_total=n_beads_total, e_corr=e_corr)
        self.element_types: List[str] = [
            chemical_symbols[z] for z in range(1, race.num_species + 1)
        ]
        self.ndescriptors = 1
        self.nparams = 1
        self.rcutfac = 0.5 * float(race.cutoff)     # MACE convention
        self.device = "cpu"
        self.initialized = False

    def _initialize(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()
        if using_kokkos:
            self.device = torch.as_tensor(data.elems).device
        else:
            self.device = torch.device("cpu")
        self.has_exchange = hasattr(data, "forward_exchange")
        # Multi-layer models need ghost feature exchange; compute_forces() raises
        # if ghosts actually take part in pairs without it.
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
                "Ghost atoms participate in pairs but this ML-IAP coupling has "
                "no forward_exchange (KOKKOS build required for periodic/parallel).")
        # A patched LAMMPS build exposes per-pair global atom IDs (tags). Use the
        # topology rule when they are present, otherwise fall back to the
        # per-type-pair distance cutoffs.
        edge_bond_topo = None
        if self.model.use_bond_flag and self.model.has_topology:
            it = getattr(data, "itags", None)
            jt = getattr(data, "jtags", None)
            if it is not None and jt is not None:
                tag_r = torch.as_tensor(it, dtype=torch.int64).to(self.device) - 1
                tag_s = torch.as_tensor(jt, dtype=torch.int64).to(self.device) - 1
                edge_bond_topo = self.model.topo_edge_bond(tag_s, tag_r)
                self._tag_s, self._tag_r = tag_s, tag_r
                if not self._topo_logged:
                    print("  [mliap] bond flag from atom tags (exact mode)")
                    self._topo_logged = True
            elif not self._topo_logged:
                print("  [mliap] no atom tags; falling back to per-type-pair "
                      "distance cutoffs (approximate)")
                self._topo_logged = True

        node_energy_local = self.model(vectors, species, edge_index,
                                       natoms, data if self.has_exchange else None,
                                       edge_bond_topo)
        total_e = node_energy_local.sum()
        (grad,) = torch.autograd.grad([total_e], [vectors])
        fij = grad          # dE/d(rij) = -dE/d(vectors); force on i = +dE/drij... sign: see module docstring
        if self.device.type != "cpu":
            torch.cuda.synchronize()

        try:                                   # kokkos coupling: writable view
            eatoms = torch.as_tensor(data.eatoms)
            eatoms.copy_(node_energy_local.detach().to(eatoms.dtype))
        except (AttributeError, TypeError):    # plain coupling: setter only
            data.eatoms = node_energy_local.detach().cpu().double().numpy()
        data.energy = float(node_energy_local.sum().item())
        pf = -fij

        # ---- delta learning: F_total = F_prior + F_delta ----
        # Only the repulsive term is added here; the bond term is handled by
        # LAMMPS bond_style harmonic.
        if self.model.has_repulsive and getattr(self, "_tag_s", None) is not None:
            if self._double_counted is None:
                key = torch.minimum(self._tag_s, self._tag_r) * (int(self._tag_r.max()) + 1) \
                      + torch.maximum(self._tag_s, self._tag_r)
                nuniq = int(torch.unique(key).numel())
                self._double_counted = (nuniq * 2 <= key.numel() + 1)
                print("  [mliap] neighbour list is %s -> prior weight %.1f"
                      % ("full" if self._double_counted else "half",
                         0.5 if self._double_counted else 1.0))
            pf = pf + self.model.repulsive_pair_forces(
                vectors.detach(), species, edge_index,
                self._tag_s, self._tag_r, self._double_counted)

        pf = pf.to(torch.float64)
        if hasattr(data, "update_pair_forces_gpu") and self.device.type != "cpu":
            data.update_pair_forces_gpu(pf)
        else:
            data.update_pair_forces(pf.cpu().numpy())

    def __reduce__(self):
        # Preferred: fully self-contained payload (weights embedded).
        payload = getattr(self, "_payload", None)
        if payload is not None:
            return (rebuild_bam_mliap_selfcontained, (payload,))
        # Legacy path: re-read the model.pkl at load time (kept for .pt files
        # exported before the self-contained format).
        args = getattr(self, "_factory_args", None)
        if args:
            return (rebuild_bam_mliap, tuple(args))
        return super().__reduce__()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass


def derive_bond_cutoff_matrix(cfg, verbose=True):
    """Measure per-type-pair bond cutoffs from the training NPZ.

    For every type pair the largest intra-molecular bonded distance and the
    smallest non-bonded distance of the same type pair are measured, and the
    midpoint is used as the threshold. Type pairs that are not bonded in the
    topology get 0. If the two ranges overlap a warning is printed and the
    conservative choice (the largest bonded distance) is used.
    """
    import numpy as np
    cg = cfg.get("cg_config", {}) or {}
    topo = cg.get("bond_topology") or {}
    npz_path = cg.get("fname_npz")
    bonds = topo.get("bonds")
    npm = topo.get("n_beads_per_mol")
    nsp = int(cfg.get("num_cg_types", cfg.get("num_species", 1)))
    if not (bonds and npm and npz_path and os.path.exists(npz_path)):
        if verbose:
            print("  [bond-mat] no topology/NPZ; using the global scalar cutoff")
        return None
    z = np.load(npz_path, allow_pickle=True)
    types = np.array(z["types"]).astype(int)
    box = np.diag(np.array(z["cells"][0], dtype=float))
    nmol = len(types) // npm
    mol = np.repeat(np.arange(nmol), npm)
    locid = np.tile(np.arange(npm), nmol)
    bondset = {tuple(sorted(b)) for b in bonds}
    frames = z["positions"][-50:].astype(float) % box
    mx = np.zeros((nsp, nsp)); mn = np.full((nsp, nsp), np.inf)
    iu = np.triu_indices(len(types), 1)
    ti, tj = types[iu[0]], types[iu[1]]
    li, lj = locid[iu[0]], locid[iu[1]]
    same = mol[iu[0]] == mol[iu[1]]
    isbond = same & np.array([tuple(sorted((a, b))) in bondset for a, b in zip(li, lj)])
    for F in frames:
        d = F[None, :, :] - F[:, None, :]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=-1)[iu]
        for a in range(nsp):
            for b in range(a, nsp):
                m = ((ti == a) & (tj == b)) | ((ti == b) & (tj == a))
                if (m & isbond).any():
                    v = r[m & isbond].max(); mx[a, b] = mx[b, a] = max(mx[a, b], v)
                if (m & ~isbond).any():
                    v = r[m & ~isbond].min(); mn[a, b] = mn[b, a] = min(mn[a, b], v)
    cut = np.zeros((nsp, nsp))
    for a in range(nsp):
        for b in range(nsp):
            if mx[a, b] <= 0:
                continue                      # type pair with no bond -> 0
            if mn[a, b] > mx[a, b]:
                cut[a, b] = 0.5 * (mx[a, b] + mn[a, b])
            else:
                cut[a, b] = mx[a, b]
                if verbose:
                    print("  [bond-mat] type pair (%d,%d): bonded and non-bonded "
                          "distances overlap"
                          " (bond max %.3f >= nonbond min %.3f)" % (a, b, mx[a, b], mn[a, b]))
    if verbose:
        print("  [bond-mat] per-type-pair bond cutoffs (A):")
        for a in range(nsp):
            for b in range(a, nsp):
                if cut[a, b] > 0:
                    print("      (%d,%d) %.3f   [bond max %.3f | nonbond min %.3f]"
                          % (a, b, cut[a, b], mx[a, b], mn[a, b]))
    return cut.tolist()


def rebuild_bam_mliap_selfcontained(payload):
    """Self-contained factory: rebuilds the model purely from the embedded
    payload (config + state_dict + wrapper options).

    The architecture is still reconstructed at load time because OEQ modules are
    not picklable, but the *weights* and every derived setting travel inside the
    .pt file. Unlike rebuild_bam_mliap() this needs no model.pkl, no dataset NPZ
    and no absolute paths, so the deployed file can be moved or shared freely.
    """
    import torch as _t
    from e3nn import o3
    from bam_torch.model.models import RACE

    cfg = payload["cfg"]
    kw = {}
    if payload.get("backend") == "oeq":
        from bam_torch.model.wrapper_ops import OEQConfig
        kw["oeq_config"] = OEQConfig(enabled=True, optimize_all=True)
    race = RACE(cutoff=cfg["cutoff"], avg_num_neighbors=cfg["avg_num_neighbors"],
                num_species=cfg.get("num_cg_types", cfg.get("num_species", 1)),
                use_bond_flag=bool(cfg.get("use_bond_flag", False)),
                max_ell=cfg["max_ell"], num_basis_func=cfg["num_radial_basis"],
                hidden_irreps=o3.Irreps(cfg["hidden_channels"]),
                nlayers=cfg["nlayers"], features_dim=cfg["features_dim"],
                output_irreps=o3.Irreps(cfg.get("output_channels", "1x0e")),
                active_fn=cfg.get("active_fn", "identity"),
                regress_forces="false",
                interaction_block=payload.get("interaction_block", "slow"), **kw)
    sd = payload.get("state_dict")
    if sd is not None:
        race.load_state_dict(sd, strict=False)
    race.criterion = None
    race.eval()

    obj = LAMMPS_MLIAP_BAM(race, payload["enr_avg"],
                           bond_cutoff=payload.get("bond_cutoff", 3.2),
                           bond_cut_mat=payload.get("bond_cut_mat"),
                           bond_topology=payload.get("bond_topology"),
                           prior_config=payload.get("prior_config"),
                           n_beads_total=payload.get("n_beads_total", 0),
                           e_corr=payload.get("e_corr", 0.0))
    obj._payload = payload
    return obj


def rebuild_bam_mliap(pkl_path, backend="e3nn", bond_cutoff=3.2,
                      interaction_block="slow", load_weights=True,
                      bond_cut_mat="auto"):
    """Factory used both for export and for unpickling (__reduce__), so OEQ
    JIT modules never need to be pickled - the model is rebuilt at load."""
    import torch as _t
    from e3nn import o3
    from bam_torch.model.models import RACE
    ck = _t.load(pkl_path, map_location="cpu", weights_only=False)
    cfg = ck["input.json"]
    kw = {}
    if backend == "oeq":
        from bam_torch.model.wrapper_ops import OEQConfig
        kw["oeq_config"] = OEQConfig(enabled=True, optimize_all=True)
    _nsp = cfg.get("num_cg_types", cfg.get("num_species", 1))
    _ubf = bool(cfg.get("use_bond_flag", False))
    race = RACE(cutoff=cfg["cutoff"], avg_num_neighbors=cfg["avg_num_neighbors"],
                num_species=_nsp, use_bond_flag=_ubf, max_ell=cfg["max_ell"],
                num_basis_func=cfg["num_radial_basis"],
                hidden_irreps=o3.Irreps(cfg["hidden_channels"]),
                nlayers=cfg["nlayers"], features_dim=cfg["features_dim"],
                output_irreps=o3.Irreps(cfg.get("output_channels", "1x0e")),
                active_fn=cfg.get("active_fn", "identity"),
                regress_forces="false", interaction_block=interaction_block, **kw)
    if load_weights:
        sd = {k.replace("module.", ""): v for k, v in ck["params"].items()}
        race.load_state_dict(sd, strict=False)
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(race.parameters(), decay=ck["ema_state"].get("decay", 0.999))
        ema.load_state_dict(ck["ema_state"]); ema.copy_to(race.parameters())
    else:
        # Benchmarking only: same architecture, random weights. The trajectory is
        # meaningless but the ns/day figure is valid.
        print("  [speed-only] weights not loaded (random initialisation)")
    print("  conv_tp[0] =", type(race.interactions[0].conv_tp).__module__ + "." +
          type(race.interactions[0].conv_tp).__name__)
    race.criterion = None
    race.eval()
    enr = {z - 1: float(v) for z, v in ck.get("enr_avg_per_element", {}).items() if z >= 1}
    if bond_cut_mat == "auto":
        bond_cut_mat = derive_bond_cutoff_matrix(cfg)
    bond_topo = (cfg.get("cg_config", {}) or {}).get("bond_topology")
    # e_corr is NaN for force-only models (same guard as create_lammps_cg)
    _ec = ck.get("valid_scale_shift_origin", None)
    e_corr = 0.0
    if _ec is not None:
        e_corr = float(_ec.item() if hasattr(_ec, "item") else _ec)
        if e_corr != e_corr:
            print("  [mliap] e_corr is NaN (force-only model) -> 0.0")
            e_corr = 0.0
    if e_corr != 0.0:
        print("  [mliap] e_corr = %.6g (distributed per atom)" % e_corr)

    prior_cfg = (cfg.get("cg_config", {}) or {}).get("prior")
    n_beads_total = 0
    _npz = (cfg.get("cg_config", {}) or {}).get("fname_npz")
    if _npz and os.path.exists(_npz):
        import numpy as _np
        n_beads_total = int(len(_np.load(_npz, allow_pickle=True)["types"]))
    if prior_cfg:
        print("  [mliap] delta prior detected: %s (repulsive term here, bond term "
              "via LAMMPS bond_style)"
              % prior_cfg.get("type"))
    obj = LAMMPS_MLIAP_BAM(race, enr, bond_cutoff=bond_cutoff, bond_cut_mat=bond_cut_mat,
                           bond_topology=bond_topo, prior_config=prior_cfg,
                           n_beads_total=n_beads_total, e_corr=e_corr)
    # Self-contained payload: everything needed to rebuild at load time, so the
    # exported .pt no longer depends on model.pkl, the dataset NPZ or absolute
    # paths. __reduce__ prefers this over _factory_args.
    obj._payload = {
        "cfg": cfg,
        "state_dict": {k: v.detach().cpu() for k, v in race.state_dict().items()},
        "backend": backend,
        "interaction_block": interaction_block,
        "bond_cutoff": bond_cutoff,
        "bond_cut_mat": bond_cut_mat,
        "bond_topology": bond_topo,
        "prior_config": prior_cfg,
        "n_beads_total": n_beads_total,
        "enr_avg": enr,
        "e_corr": e_corr,
        "source_pkl": pkl_path,   # provenance only, not read at load time
    }
    return obj
