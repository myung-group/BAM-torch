"""Mean-force matching (MFM) dataset pipeline for coarse-grained models.

MFM labels are *mean* forces: each AA snapshot is propagated with the CG bead
centres held fixed by GROMACS pull constraints, and the constraint forces are
averaged over the second half of that run. Compared with plain force matching
this removes the instantaneous-force noise floor, at the cost of one constrained
MD run per snapshot.

Three stages, all driven by the same mapping file:

  1. ``pull``      snapshot .gro            -> index file + pull mdp
  2. ``snapshot``  constrained .gro/.trr    -> per-snapshot npz (positions, forces)
  3. ``assemble``  many per-snapshot npz    -> single training npz

Mapping file (JSON)::

    {
      "name": "octanol",
      "n_atoms_per_mol": 27,
      "beads": [[0, 1, 9, 10, 11, 12, 13], ...],
      "bead_types": [0, 1, 2, 3, 4],          # optional, defaults to 0..n_beads-1
      "masses": {"C": 12.011, "O": 15.999},   # optional, anything else is 1.008
      "molar_mass": 130.23                    # optional, only used for a density report
    }

``beads`` lists atom indices *within one molecule*, in the AA ordering of the
trajectory. Every atom that carries force should appear exactly once.
"""
import argparse
import glob
import json
import os

import numpy as np

# MDAnalysis reports forces in kJ/mol/A; models are trained in eV/A.
KJ_PER_MOL_PER_A_TO_EV_PER_A = 0.01036427230133
DEFAULT_MASSES = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974}
DEFAULT_MASS = 1.008


class CGMapping:
    """Atom -> bead mapping for one molecular species."""

    def __init__(self, cfg):
        self.name = cfg.get("name", "cg")
        self.n_atoms_per_mol = int(cfg["n_atoms_per_mol"])
        self.beads = [list(map(int, b)) for b in cfg["beads"]]
        self.n_beads = len(self.beads)
        self.bead_types = list(map(int, cfg.get("bead_types", range(self.n_beads))))
        if len(self.bead_types) != self.n_beads:
            raise ValueError("bead_types has %d entries but there are %d beads"
                             % (len(self.bead_types), self.n_beads))
        self.masses = dict(DEFAULT_MASSES)
        self.masses.update(cfg.get("masses", {}))
        self.molar_mass = cfg.get("molar_mass")
        covered = sorted(i for b in self.beads for i in b)
        if len(covered) != len(set(covered)):
            raise ValueError("an atom index appears in more than one bead")
        if covered and covered[-1] >= self.n_atoms_per_mol:
            raise ValueError("bead atom index %d exceeds n_atoms_per_mol=%d"
                             % (covered[-1], self.n_atoms_per_mol))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def atom_mass(self, name):
        """Mass from an atom name, matching the longest known element prefix."""
        nm = name.strip()
        for sym in sorted(self.masses, key=len, reverse=True):
            if nm.upper().startswith(sym.upper()):
                return self.masses[sym]
        return DEFAULT_MASS

    def mass_vector(self, names):
        return np.array([self.atom_mass(n) for n in names], dtype=np.float64)


def bead_centres(pos_per_mol, mass_per_mol, mapping, box):
    """Mass-weighted bead centres, (n_mol, n_beads, 3).

    Molecules are unwrapped relative to their first atom before averaging, then
    the centres are wrapped back into the box.
    """
    ref = pos_per_mol[:, 0:1, :]
    dr = pos_per_mol - ref
    dr -= box * np.round(dr / box)
    unwrapped = ref + dr
    out = np.zeros((pos_per_mol.shape[0], mapping.n_beads, 3), dtype=np.float64)
    for b, atoms in enumerate(mapping.beads):
        w = mass_per_mol[:, atoms][:, :, None]
        c = (unwrapped[:, atoms, :] * w).sum(1) / mass_per_mol[:, atoms].sum(1, keepdims=True)
        out[:, b, :] = c - box * np.floor(c / box)
    return out


def _read_gro(path):
    """Minimal .gro reader: returns (names, positions in A, box in A)."""
    lines = open(path).read().splitlines()
    nat = int(lines[1])
    names, pos = [], np.zeros((nat, 3))
    for i, ln in enumerate(lines[2:2 + nat]):
        names.append(ln[10:15].strip())
        pos[i] = [float(ln[20:28]), float(ln[28:36]), float(ln[36:44])]
    box = np.array([float(x) for x in lines[2 + nat].split()[:3]])
    return names, pos, box


def write_pull_inputs(gro, ndx_out, mdp_out, mapping, n_mol_use=0, offset=0.05):
    """Write a GROMACS index file with one group per bead plus a pull mdp that
    constrains every bead centre in x, y and z.

    The pull origin is placed `offset` nm behind the centre along each axis so
    the coordinate value is `offset` rather than 0: a zero distance is rejected
    by GROMACS, and using the box origin gives distances beyond half the box.
    Units here are nm, as in the .gro file.
    """
    names, pos, box = _read_gro(gro)
    nat = len(names)
    mass = np.tile(mapping.mass_vector(names[:mapping.n_atoms_per_mol]),
                   nat // mapping.n_atoms_per_mol)
    n_mol = nat // mapping.n_atoms_per_mol
    use = min(n_mol_use, n_mol) if n_mol_use > 0 else n_mol

    targets = []
    with open(ndx_out, "w") as f:
        f.write("[ System ]\n")
        for i in range(nat):
            f.write(str(i + 1) + ("\n" if (i + 1) % 15 == 0 else " "))
        f.write("\n")
        b = 0
        for m in range(use):
            for atoms in mapping.beads:
                idx = [m * mapping.n_atoms_per_mol + a for a in atoms]
                targets.append((pos[idx] * mass[idx, None]).sum(0) / mass[idx].sum())
                f.write("[ bead_%d ]\n" % b + " ".join(str(i + 1) for i in idx) + "\n")
                b += 1
    targets = np.array(targets)
    n_bead = len(targets)

    with open(mdp_out, "w") as f:
        f.write("pull = yes\npull-ngroups = %d\npull-ncoords = %d\n"
                "pull-nstfout = 100\npull-nstxout = 0\n" % (n_bead, n_bead * 3))
        for b in range(n_bead):
            f.write("pull-group%d-name = bead_%d\n" % (b + 1, b))
        c = 1
        for b in range(n_bead):
            for axis in range(3):
                vec = [0, 0, 0]
                vec[axis] = 1
                org = targets[b].copy()
                org[axis] -= offset
                f.write("pull-coord%d-type = constraint\n"
                        "pull-coord%d-geometry = direction\n" % (c, c))
                f.write("pull-coord%d-groups = 0 %d\n"
                        "pull-coord%d-origin = %.5f %.5f %.5f\n"
                        % (c, b + 1, c, org[0], org[1], org[2]))
                f.write("pull-coord%d-vec = %d %d %d\npull-coord%d-init = %.4f\n"
                        % (c, vec[0], vec[1], vec[2], c, offset))
                c += 1
    np.save(ndx_out + ".targets.npy", targets)
    print("wrote %d bead groups (%d mol x %d), box %s"
          % (n_bead, use, mapping.n_beads, np.round(box, 3).tolist()))
    return targets


def mean_force_snapshot(con_gro, con_trr, out_npz, mapping, average_from=0.5):
    """Average bead forces over the tail of a constrained run and save one npz.

    `average_from` is the fraction of the trajectory to skip before averaging;
    the default keeps the second half, letting the constrained run relax first.
    """
    import warnings

    import MDAnalysis as mda
    warnings.filterwarnings("ignore")

    na, nb = mapping.n_atoms_per_mol, mapping.n_beads
    u = mda.Universe(con_gro, con_trr)
    n_mol = len(u.atoms) // na
    masses = np.tile(mapping.mass_vector([a.name for a in u.atoms[:na]]), (n_mol, 1))

    n_frames = len(u.trajectory)
    start = int(n_frames * average_from)
    acc = np.zeros((n_mol, nb, 3))
    count = 0
    for i, ts in enumerate(u.trajectory):
        if i < start:
            continue
        if ts.forces is None:
            raise RuntimeError("%s carries no forces; set nstfout in the mdp" % con_trr)
        frc = ts.forces.reshape(n_mol, na, 3) * KJ_PER_MOL_PER_A_TO_EV_PER_A
        for b, atoms in enumerate(mapping.beads):
            acc[:, b, :] += frc[:, atoms, :].sum(1)
        count += 1
    if count == 0:
        raise RuntimeError("no frames left after skipping %.0f%%" % (100 * average_from))
    mean_force = acc / count

    ug = mda.Universe(con_gro)
    box = ug.dimensions[:3].astype(np.float64)
    pos = ug.atoms.positions.reshape(n_mol, na, 3).astype(np.float64)
    com = bead_centres(pos, masses, mapping, box)

    types = np.tile(np.array(mapping.bead_types, dtype=np.int32), n_mol)
    np.savez(out_npz,
             positions=com.reshape(-1, 3).astype(np.float32),
             forces=mean_force.reshape(-1, 3).astype(np.float32),
             types=types,
             cell=np.diag(box).astype(np.float32))
    mag = np.linalg.norm(mean_force.reshape(-1, 3), axis=1)
    print("%s: averaged %d/%d frames  |F| mean %.4f max %.4f  box %s"
          % (out_npz, count, n_frames, mag.mean(), mag.max(), np.round(box, 3).tolist()))
    return out_npz


def assemble(pattern, out_npz, mapping, expected=0, source=""):
    """Stack per-snapshot npz files into one training dataset."""
    files = sorted(glob.glob(pattern),
                   key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))) or 0))
    if not files:
        raise SystemExit("no files matched %s" % pattern)
    P, F, C = [], [], []
    for f in files:
        d = np.load(f)
        P.append(d["positions"]); F.append(d["forces"]); C.append(d["cell"])
    P = np.stack(P).astype(np.float32)
    F = np.stack(F).astype(np.float32)
    C = np.stack(C).astype(np.float32)
    n_frames, n_bead = P.shape[0], P.shape[1]
    n_mol = n_bead // mapping.n_beads
    types = np.tile(np.array(mapping.bead_types, dtype=np.int32), n_mol)
    meta = {"system": mapping.name, "n_beads_per_mol": mapping.n_beads,
            "method": "MFM_constrained_MD", "unit_position": "Angstrom",
            "unit_force": "eV/Angstrom", "n_frames": int(n_frames), "n_mol": int(n_mol),
            "bead_types": mapping.bead_types, "source": source,
            "force_def": "tail average of bead-mapped atomic forces (mean force)"}
    np.savez(out_npz, positions=P, forces=F, cells=C, types=types,
             energies=np.zeros(n_frames, np.float64), metadata=meta)

    mag = np.linalg.norm(F.reshape(-1, 3), axis=1)
    L = np.diag(C.mean(0))
    print("assembled %d snapshots -> %s" % (n_frames, out_npz))
    print("  beads %d (%d mol x %d) | |F| mean %.4f min %.4f max %.4f eV/A"
          % (n_bead, n_mol, mapping.n_beads, mag.mean(), mag.min(), mag.max()))
    print("  mean box %s A" % np.round(L, 3).tolist())
    if mapping.molar_mass:
        dens = (n_mol * mapping.molar_mass) / 6.02214076e23 / (np.prod(L) * 1e-24)
        print("  density %.4f g/mL" % dens)
    if expected and n_frames != expected:
        print("  WARNING: expected %d snapshots, got %d -- some runs failed"
              % (expected, n_frames))
    return out_npz


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mapping", required=True, help="CG mapping JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="write index + pull mdp for one snapshot")
    p.add_argument("--gro", required=True)
    p.add_argument("--ndx", required=True)
    p.add_argument("--mdp", required=True)
    p.add_argument("--n-mol", type=int, default=0, help="0 = all molecules")
    p.add_argument("--offset", type=float, default=0.05, help="pull origin offset (nm)")

    s = sub.add_parser("snapshot", help="mean force from one constrained run")
    s.add_argument("--gro", required=True)
    s.add_argument("--trr", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--average-from", type=float, default=0.5)

    a = sub.add_parser("assemble", help="stack per-snapshot npz into a dataset")
    a.add_argument("--pattern", default="mf_snap*.npz")
    a.add_argument("--out", required=True)
    a.add_argument("--expected", type=int, default=0)
    a.add_argument("--source", default="")

    args = ap.parse_args()
    mapping = CGMapping.load(args.mapping)
    if args.cmd == "pull":
        write_pull_inputs(args.gro, args.ndx, args.mdp, mapping, args.n_mol, args.offset)
    elif args.cmd == "snapshot":
        mean_force_snapshot(args.gro, args.trr, args.out, mapping, args.average_from)
    else:
        assemble(args.pattern, args.out, mapping, args.expected, args.source)


if __name__ == "__main__":
    main()
