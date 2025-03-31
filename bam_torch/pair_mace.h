/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mace,PairMACE);
// clang-format on
#else

#ifndef LMP_PAIR_MACE_H
#define LMP_PAIR_MACE_H

#include "pair.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace LAMMPS_NS {

class PairMACE : public Pair {

 public:

  PairMACE(class LAMMPS *);
  ~PairMACE() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void allocate();

 protected:

  bool domain_decomposition = true;
  torch::Device device = torch::kCPU;
  torch::jit::script::Module model;
  torch::ScalarType torch_float_dtype;
  double r_max;
  double r_max_squared;
  int64_t num_interactions;
  std::vector<int64_t> mace_atomic_numbers;
  std::vector<int64_t> lammps_atomic_numbers;
  int mace_type(int lammps_type);
  const std::array<std::string,118> periodic_table =
    { "H", "He",
     "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
     "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
     "K",  "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
     "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                       "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                       "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

};
}    // namespace LAMMPS_NS

#endif
#endif
