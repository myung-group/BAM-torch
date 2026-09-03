# 3BPA LAMMPS MD example (`pair_style bam`)

Run MD on the 3BPA molecule with the RACE model trained in
`../example-3BPA_300K`, using LAMMPS instead of the ASE calculator.
This folder is used by the "LAMMPS md simulation" section of the
BAM-torch tutorial notebook.

## Files

- `race.in`               - LAMMPS input: `pair_style bam`, NVT at 300 K, 0.5 fs timestep
- `convert_to_lammps.py`  - converts the trained checkpoint (model.pkl) directly to the
                            TorchScript model consumed by `pair_style bam` (merges
                            make_pt_oeq.py --backend e3nn and create_lammps.py in memory;
                            the two-step path via model.pt breaks with e3nn >= 0.6)
- `3bpa_300K.data` - first frame of `../dataset/test_300K.xyz` in LAMMPS data format
                     (type order 1=C, 2=H, 3=N, 4=O - must match `pair_coeff`)
- `make_data.py`   - regenerates `3bpa_300K.data` from the dataset with ASE

## Steps

1. **Build LAMMPS with the ML-BAM package** - see `bam_torch/lammps/README.md`
   (or the tutorial notebook's Step 1, which downloads a prebuilt Colab binary).

2. **Convert the trained checkpoint** (from `../example-3BPA_300K`):

       python convert_to_lammps.py --pkl ../example-3BPA_300K/model.pkl

3. **Run MD**:

       lmp -in race.in

4. **Post-process** the trajectory back into an ASE `.traj`
   (merges `dump.lammpstrj` with the energies in `log.lammps`):

       python /path/to/BAM-torch/bam_torch/lammps/lammpsout_to_traj.py
