"""
MARTINI-style coarse-grained mapping registry for biomolecular systems.

Defines atom-name-to-bead mappings for:
- Lipids (DLIPC, DPPC, POPC, DOPC, DOPE, CHL1)
- DNA nucleotides (DA, DT, DG, DC)
- RNA nucleotides (RA, RU, RG, RC)
- Protein amino acids (20 standard residues)
- Water/ions (SOL, WAT, etc.)

Each mapping entry uses CHARMM36 atom naming conventions (compatible with CHARMM-GUI output).
For other force fields (AMBER, OPLS), use the `custom_mappings` parameter in ResidueBasedCGMapper.

Reference:
    MARTINI 3: Souza et al., Nature Methods 18, 382-388 (2021)
    MARTINI 2: Marrink et al., J. Phys. Chem. B 111, 7812-7824 (2007)
"""

from typing import Dict, List, Optional


# =============================================================================
# Atomic masses for common elements (g/mol)
# =============================================================================
ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'P': 30.974, 'S': 32.065, 'Na': 22.990, 'Cl': 35.453,
    'K': 39.098, 'Ca': 40.078, 'Mg': 24.305, 'Zn': 65.380,
    'Fe': 55.845, 'Se': 78.971,
}


def get_element_from_name(atom_name: str) -> str:
    """Extract element symbol from atom name (CHARMM convention)."""
    name = atom_name.strip()
    if not name:
        return 'C'
    first = name[0]
    if first in ATOMIC_MASSES:
        return first
    if len(name) >= 2 and name[:2] in ATOMIC_MASSES:
        return name[:2]
    return first


def get_mass_from_name(atom_name: str) -> float:
    """Get atomic mass from atom name."""
    element = get_element_from_name(atom_name)
    return ATOMIC_MASSES.get(element, 12.011)


# =============================================================================
# MARTINI MAPPING REGISTRY
# =============================================================================

MARTINI_MAPPINGS: Dict[str, dict] = {}


# -----------------------------------------------------------------------------
# LIPIDS
# -----------------------------------------------------------------------------

MARTINI_MAPPINGS['DLIPC'] = {
    'description': 'Dilinoleoyl-phosphatidylcholine (MARTINI)',
    'category': 'lipid',
    'bead_order': ['NC3', 'PO4', 'GL1', 'GL2', 'C1A', 'C2A', 'D3A', 'C4A',
                   'C1B', 'C2B', 'D3B', 'C4B'],
    'atom_names': {
        'NC3': ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C',
                'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B'],
        'PO4': ['C11', 'H11A', 'H11B', 'P', 'O13', 'O14', 'O12', 'O11'],
        'GL1': ['C1', 'HA', 'HB', 'C2', 'HS', 'O21', 'C21', 'O22'],
        'GL2': ['C3', 'HX', 'HY', 'O31', 'C31', 'O32'],
        'C1A': ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
        'C2A': ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R'],
        'D3A': ['C210', 'H10R', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'C213', 'H13R'],
        'C4A': ['C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S', 'C216', 'H16R', 'H16S',
                'C217', 'H17R', 'H17S', 'C218', 'H18R', 'H18S', 'H18T'],
        'C1B': ['C32', 'H2X', 'H2Y', 'C33', 'H3X', 'H3Y', 'C34', 'H4X', 'H4Y', 'C35', 'H5X', 'H5Y'],
        'C2B': ['C36', 'H6X', 'H6Y', 'C37', 'H7X', 'H7Y', 'C38', 'H8X', 'H8Y', 'C39', 'H9X'],
        'D3B': ['C310', 'H10X', 'C311', 'H11X', 'H11Y', 'C312', 'H12X', 'C313', 'H13X'],
        'C4B': ['C314', 'H14X', 'H14Y', 'C315', 'H15X', 'H15Y', 'C316', 'H16X', 'H16Y',
                'C317', 'H17X', 'H17Y', 'C318', 'H18X', 'H18Y', 'H18Z'],
    },
}

MARTINI_MAPPINGS['DPPC'] = {
    'description': 'Dipalmitoylphosphatidylcholine (MARTINI)',
    'category': 'lipid',
    'bead_order': ['NC3', 'PO4', 'GL1', 'GL2', 'C1A', 'C2A', 'C3A', 'C4A',
                   'C1B', 'C2B', 'C3B', 'C4B'],
    'atom_names': {
        'NC3': ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C',
                'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B'],
        'PO4': ['C11', 'H11A', 'H11B', 'P', 'O13', 'O14', 'O12', 'O11'],
        'GL1': ['C1', 'HA', 'HB', 'C2', 'HS', 'O21', 'C21', 'O22'],
        'GL2': ['C3', 'HX', 'HY', 'O31', 'C31', 'O32'],
        'C1A': ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
        'C2A': ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R', 'H9S'],
        'C3A': ['C210', 'H10R', 'H10S', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S'],
        'C4A': ['C213', 'H13R', 'H13S', 'C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S', 'C216', 'H16R', 'H16S', 'H16T'],
        'C1B': ['C32', 'H2X', 'H2Y', 'C33', 'H3X', 'H3Y', 'C34', 'H4X', 'H4Y', 'C35', 'H5X', 'H5Y'],
        'C2B': ['C36', 'H6X', 'H6Y', 'C37', 'H7X', 'H7Y', 'C38', 'H8X', 'H8Y', 'C39', 'H9X', 'H9Y'],
        'C3B': ['C310', 'H10X', 'H10Y', 'C311', 'H11X', 'H11Y', 'C312', 'H12X', 'H12Y'],
        'C4B': ['C313', 'H13X', 'H13Y', 'C314', 'H14X', 'H14Y', 'C315', 'H15X', 'H15Y', 'C316', 'H16X', 'H16Y', 'H16Z'],
    },
}

MARTINI_MAPPINGS['POPC'] = {
    'description': '1-Palmitoyl-2-oleoylphosphatidylcholine (MARTINI)',
    'category': 'lipid',
    'bead_order': ['NC3', 'PO4', 'GL1', 'GL2', 'C1A', 'D2A', 'C3A', 'C4A',
                   'C1B', 'C2B', 'C3B', 'C4B'],
    'atom_names': {
        'NC3': ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C',
                'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B'],
        'PO4': ['C11', 'H11A', 'H11B', 'P', 'O13', 'O14', 'O12', 'O11'],
        'GL1': ['C1', 'HA', 'HB', 'C2', 'HS', 'O21', 'C21', 'O22'],
        'GL2': ['C3', 'HX', 'HY', 'O31', 'C31', 'O32'],
        'C1A': ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
        'D2A': ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R'],
        'C3A': ['C210', 'H10R', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S'],
        'C4A': ['C213', 'H13R', 'H13S', 'C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S',
                'C216', 'H16R', 'H16S', 'C217', 'H17R', 'H17S', 'C218', 'H18R', 'H18S', 'H18T'],
        'C1B': ['C32', 'H2X', 'H2Y', 'C33', 'H3X', 'H3Y', 'C34', 'H4X', 'H4Y', 'C35', 'H5X', 'H5Y'],
        'C2B': ['C36', 'H6X', 'H6Y', 'C37', 'H7X', 'H7Y', 'C38', 'H8X', 'H8Y', 'C39', 'H9X', 'H9Y'],
        'C3B': ['C310', 'H10X', 'H10Y', 'C311', 'H11X', 'H11Y', 'C312', 'H12X', 'H12Y'],
        'C4B': ['C313', 'H13X', 'H13Y', 'C314', 'H14X', 'H14Y', 'C315', 'H15X', 'H15Y', 'C316', 'H16X', 'H16Y', 'H16Z'],
    },
}

MARTINI_MAPPINGS['DOPC'] = {
    'description': 'Dioleoylphosphatidylcholine (MARTINI)',
    'category': 'lipid',
    'bead_order': ['NC3', 'PO4', 'GL1', 'GL2', 'C1A', 'D2A', 'C3A', 'C4A',
                   'C1B', 'D2B', 'C3B', 'C4B'],
    'atom_names': {
        'NC3': ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C',
                'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B'],
        'PO4': ['C11', 'H11A', 'H11B', 'P', 'O13', 'O14', 'O12', 'O11'],
        'GL1': ['C1', 'HA', 'HB', 'C2', 'HS', 'O21', 'C21', 'O22'],
        'GL2': ['C3', 'HX', 'HY', 'O31', 'C31', 'O32'],
        'C1A': ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
        'D2A': ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R'],
        'C3A': ['C210', 'H10R', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S'],
        'C4A': ['C213', 'H13R', 'H13S', 'C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S',
                'C216', 'H16R', 'H16S', 'C217', 'H17R', 'H17S', 'C218', 'H18R', 'H18S', 'H18T'],
        'C1B': ['C32', 'H2X', 'H2Y', 'C33', 'H3X', 'H3Y', 'C34', 'H4X', 'H4Y', 'C35', 'H5X', 'H5Y'],
        'D2B': ['C36', 'H6X', 'H6Y', 'C37', 'H7X', 'H7Y', 'C38', 'H8X', 'H8Y', 'C39', 'H9X'],
        'C3B': ['C310', 'H10X', 'C311', 'H11X', 'H11Y', 'C312', 'H12X', 'H12Y'],
        'C4B': ['C313', 'H13X', 'H13Y', 'C314', 'H14X', 'H14Y', 'C315', 'H15X', 'H15Y', 'C316', 'H16X', 'H16Y', 'H16Z'],
    },
}

MARTINI_MAPPINGS['DSPC'] = {
    'description': 'Distearoylphosphatidylcholine (MARTINI)',
    'category': 'lipid',
    'bead_order': ['NC3', 'PO4', 'GL1', 'GL2', 'C1A', 'C2A', 'C3A', 'C4A',
                   'C1B', 'C2B', 'C3B', 'C4B'],
    'atom_names': {
        'NC3': ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C',
                'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B'],
        'PO4': ['C11', 'H11A', 'H11B', 'P', 'O13', 'O14', 'O11', 'O12'],
        'GL1': ['C1', 'HA', 'HB', 'C2', 'HS', 'O21', 'C21', 'O22'],
        'GL2': ['C3', 'HX', 'HY', 'O31', 'C31', 'O32'],
        'C1A': ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
        'C2A': ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R', 'H9S'],
        'C3A': ['C210', 'H10R', 'H10S', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S', 'C213', 'H13R', 'H13S'],
        'C4A': ['C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S', 'C216', 'H16R', 'H16S',
                'C217', 'H17R', 'H17S', 'C218', 'H18R', 'H18S', 'H18T'],
        'C1B': ['C32', 'H2X', 'H2Y', 'C33', 'H3X', 'H3Y', 'C34', 'H4X', 'H4Y', 'C35', 'H5X', 'H5Y'],
        'C2B': ['C36', 'H6X', 'H6Y', 'C37', 'H7X', 'H7Y', 'C38', 'H8X', 'H8Y', 'C39', 'H9X', 'H9Y'],
        'C3B': ['C310', 'H10X', 'H10Y', 'C311', 'H11X', 'H11Y', 'C312', 'H12X', 'H12Y', 'C313', 'H13X', 'H13Y'],
        'C4B': ['C314', 'H14X', 'H14Y', 'C315', 'H15X', 'H15Y', 'C316', 'H16X', 'H16Y',
                'C317', 'H17X', 'H17Y', 'C318', 'H18X', 'H18Y', 'H18Z'],
    },
}

MARTINI_MAPPINGS['DOPE'] = {
    'description': 'Dioleoylphosphatidylethanolamine (MARTINI)',
    'category': 'lipid',
    'bead_order': ['NH3', 'PO4', 'GL1', 'GL2', 'C1A', 'D2A', 'C3A', 'C4A',
                   'C1B', 'D2B', 'C3B', 'C4B'],
    'atom_names': {
        'NH3': ['N', 'HN1', 'HN2', 'HN3', 'C12', 'H12A', 'H12B'],
        'PO4': ['C11', 'H11A', 'H11B', 'P', 'O13', 'O14', 'O11', 'O12'],
        'GL1': ['C1', 'HA', 'HB', 'C2', 'HS', 'O21', 'C21', 'O22'],
        'GL2': ['C3', 'HX', 'HY', 'O31', 'C31', 'O32'],
        'C1A': ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
        'D2A': ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R'],
        'C3A': ['C210', 'H10R', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S'],
        'C4A': ['C213', 'H13R', 'H13S', 'C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S',
                'C216', 'H16R', 'H16S', 'C217', 'H17R', 'H17S', 'C218', 'H18R', 'H18S', 'H18T'],
        'C1B': ['C32', 'H2X', 'H2Y', 'C33', 'H3X', 'H3Y', 'C34', 'H4X', 'H4Y', 'C35', 'H5X', 'H5Y'],
        'D2B': ['C36', 'H6X', 'H6Y', 'C37', 'H7X', 'H7Y', 'C38', 'H8X', 'H8Y', 'C39', 'H9X'],
        'C3B': ['C310', 'H10X', 'C311', 'H11X', 'H11Y', 'C312', 'H12X', 'H12Y'],
        'C4B': ['C313', 'H13X', 'H13Y', 'C314', 'H14X', 'H14Y', 'C315', 'H15X', 'H15Y',
                'C316', 'H16X', 'H16Y', 'C317', 'H17X', 'H17Y', 'C318', 'H18X', 'H18Y', 'H18Z'],
    },
}

MARTINI_MAPPINGS['CHL1'] = {
    'description': 'Cholesterol (MARTINI)',
    'category': 'lipid',
    'bead_order': ['ROH', 'R1', 'R2', 'R3', 'R4', 'R5', 'C1', 'C2'],
    'atom_names': {
        'ROH': ['C3', 'H3', 'O3', "H3'", 'C4', 'H4A', 'H4B', 'C2', 'H2A', 'H2B', 'C1', 'H1A', 'H1B'],
        'R1':  ['C5', 'C6', 'H6', 'C10', 'C19', 'H19A', 'H19B', 'H19C'],
        'R2':  ['C7', 'H7A', 'H7B', 'C8', 'H8', 'C9', 'H9'],
        'R3':  ['C11', 'H11A', 'H11B', 'C12', 'H12A', 'H12B', 'C14', 'H14'],
        'R4':  ['C13', 'C18', 'H18A', 'H18B', 'H18C', 'C15', 'H15A', 'H15B'],
        'R5':  ['C16', 'H16A', 'H16B', 'C17', 'H17', 'C20', 'H20', 'C21', 'H21A', 'H21B', 'H21C'],
        'C1':  ['C22', 'H22A', 'H22B', 'C23', 'H23A', 'H23B', 'C24', 'H24A', 'H24B'],
        'C2':  ['C25', 'H25', 'C26', 'H26A', 'H26B', 'H26C', 'C27', 'H27A', 'H27B', 'H27C'],
    },
}

# -----------------------------------------------------------------------------
# DNA NUCLEOTIDES (MARTINI 2 style, CHARMM36 atom names)
# -----------------------------------------------------------------------------

MARTINI_MAPPINGS['DA'] = {
    'description': 'Deoxyadenosine (DNA adenine, MARTINI)',
    'category': 'dna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'", "H2''", "O3'"],
        'SC1': ['N9', 'C4', 'C8', 'H8'],
        'SC2': ['N3', 'C2', 'H2', 'N1'],
        'SC3': ['C6', 'N6', 'H61', 'H62'],
        'SC4': ['C5', 'N7'],
    },
}

MARTINI_MAPPINGS['DT'] = {
    'description': 'Deoxythymidine (DNA thymine, MARTINI)',
    'category': 'dna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'", "H2''", "O3'"],
        'SC1': ['N1', 'C6', 'H6', 'C2', 'O2'],
        'SC2': ['N3', 'H3', 'C4', 'O4'],
        'SC3': ['C5', 'C7', 'H71', 'H72', 'H73'],
    },
}

MARTINI_MAPPINGS['DG'] = {
    'description': 'Deoxyguanosine (DNA guanine, MARTINI)',
    'category': 'dna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'", "H2''", "O3'"],
        'SC1': ['N9', 'C4', 'C8', 'H8'],
        'SC2': ['N3', 'C2', 'N2', 'H21', 'H22'],
        'SC3': ['N1', 'H1', 'C6', 'O6'],
        'SC4': ['C5', 'N7'],
    },
}

MARTINI_MAPPINGS['DC'] = {
    'description': 'Deoxycytidine (DNA cytosine, MARTINI)',
    'category': 'dna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'", "H2''", "O3'"],
        'SC1': ['N1', 'C6', 'H6', 'C2', 'O2'],
        'SC2': ['N3', 'C4', 'N4', 'H41', 'H42'],
        'SC3': ['C5', 'H5'],
    },
}

# -----------------------------------------------------------------------------
# RNA NUCLEOTIDES (MARTINI 2 style, CHARMM36 atom names)
# RNA has 2'-OH on sugar -> extra atom in BB3
# -----------------------------------------------------------------------------

MARTINI_MAPPINGS['RA'] = {
    'description': 'Adenosine (RNA adenine, MARTINI)',
    'category': 'rna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'1", "O2'", "HO'2", "O3'"],
        'SC1': ['N9', 'C4', 'C8', 'H8'],
        'SC2': ['N3', 'C2', 'H2', 'N1'],
        'SC3': ['C6', 'N6', 'H61', 'H62'],
        'SC4': ['C5', 'N7'],
    },
}

MARTINI_MAPPINGS['RU'] = {
    'description': 'Uridine (RNA uracil, MARTINI)',
    'category': 'rna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'1", "O2'", "HO'2", "O3'"],
        'SC1': ['N1', 'C6', 'H6', 'C2', 'O2', 'N3', 'H3'],
        'SC2': ['C4', 'O4', 'C5', 'H5'],
    },
}

MARTINI_MAPPINGS['RG'] = {
    'description': 'Guanosine (RNA guanine, MARTINI)',
    'category': 'rna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3', 'SC4'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'1", "O2'", "HO'2", "O3'"],
        'SC1': ['N9', 'C4', 'C8', 'H8'],
        'SC2': ['N3', 'C2', 'N2', 'H21', 'H22'],
        'SC3': ['N1', 'H1', 'C6', 'O6'],
        'SC4': ['C5', 'N7'],
    },
}

MARTINI_MAPPINGS['RC'] = {
    'description': 'Cytidine (RNA cytosine, MARTINI)',
    'category': 'rna',
    'bead_order': ['BB1', 'BB2', 'BB3', 'SC1', 'SC2', 'SC3'],
    'atom_names': {
        'BB1': ['P', 'O1P', 'O2P'],
        'BB2': ["O5'", "C5'", "H5'", "H5''"],
        'BB3': ["C4'", "H4'", "O4'", "C1'", "H1'", "C3'", "H3'", "C2'", "H2'1", "O2'", "HO'2", "O3'"],
        'SC1': ['N1', 'C6', 'H6', 'C2', 'O2'],
        'SC2': ['N3', 'C4', 'N4', 'H41', 'H42'],
        'SC3': ['C5', 'H5'],
    },
}

# -----------------------------------------------------------------------------
# PROTEIN AMINO ACIDS (MARTINI 3 style, CHARMM36 atom names)
# BB = backbone bead, SC1-SC4 = sidechain beads
# -----------------------------------------------------------------------------

MARTINI_MAPPINGS['GLY'] = {
    'description': 'Glycine',
    'category': 'protein',
    'bead_order': ['BB'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA1', 'HA2', 'C', 'O'],
    },
}

MARTINI_MAPPINGS['ALA'] = {
    'description': 'Alanine',
    'category': 'protein',
    'bead_order': ['BB'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O', 'CB', 'HB1', 'HB2', 'HB3'],
    },
}

MARTINI_MAPPINGS['VAL'] = {
    'description': 'Valine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23'],
    },
}

MARTINI_MAPPINGS['LEU'] = {
    'description': 'Leucine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13',
                'CD2', 'HD21', 'HD22', 'HD23'],
    },
}

MARTINI_MAPPINGS['ILE'] = {
    'description': 'Isoleucine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB', 'CG2', 'HG21', 'HG22', 'HG23', 'CG1', 'HG11', 'HG12',
                'CD', 'HD1', 'HD2', 'HD3'],
    },
}

MARTINI_MAPPINGS['PRO'] = {
    'description': 'Proline',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2', 'CD', 'HD1', 'HD2'],
    },
}

MARTINI_MAPPINGS['PHE'] = {
    'description': 'Phenylalanine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1', 'SC2', 'SC3'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG'],
        'SC2': ['CD1', 'HD1', 'CE1', 'HE1'],
        'SC3': ['CD2', 'HD2', 'CE2', 'HE2', 'CZ', 'HZ'],
    },
}

MARTINI_MAPPINGS['TRP'] = {
    'description': 'Tryptophan',
    'category': 'protein',
    'bead_order': ['BB', 'SC1', 'SC2', 'SC3', 'SC4'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'CD1', 'HD1'],
        'SC2': ['NE1', 'HE1', 'CE2', 'CD2'],
        'SC3': ['CZ2', 'HZ2', 'CH2', 'HH2'],
        'SC4': ['CZ3', 'HZ3', 'CE3', 'HE3'],
    },
}

MARTINI_MAPPINGS['SER'] = {
    'description': 'Serine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'OG', 'HG1'],
    },
}

MARTINI_MAPPINGS['THR'] = {
    'description': 'Threonine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB', 'OG1', 'HG1', 'CG2', 'HG21', 'HG22', 'HG23'],
    },
}

MARTINI_MAPPINGS['CYS'] = {
    'description': 'Cysteine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'SG', 'HG1'],
    },
}

MARTINI_MAPPINGS['MET'] = {
    'description': 'Methionine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2', 'SD', 'CE', 'HE1', 'HE2', 'HE3'],
    },
}

MARTINI_MAPPINGS['ASP'] = {
    'description': 'Aspartate',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'OD1', 'OD2'],
    },
}

MARTINI_MAPPINGS['GLU'] = {
    'description': 'Glutamate',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2', 'CD', 'OE1', 'OE2'],
    },
}

MARTINI_MAPPINGS['ASN'] = {
    'description': 'Asparagine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'OD1', 'ND2', 'HD21', 'HD22'],
    },
}

MARTINI_MAPPINGS['GLN'] = {
    'description': 'Glutamine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
    },
}

MARTINI_MAPPINGS['LYS'] = {
    'description': 'Lysine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1', 'SC2'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2'],
        'SC2': ['CD', 'HD1', 'HD2', 'CE', 'HE1', 'HE2', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
    },
}

MARTINI_MAPPINGS['ARG'] = {
    'description': 'Arginine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1', 'SC2'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG', 'HG1', 'HG2'],
        'SC2': ['CD', 'HD1', 'HD2', 'NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22'],
    },
}

MARTINI_MAPPINGS['HIS'] = {
    'description': 'Histidine (HSD/HSE/HSP)',
    'category': 'protein',
    'bead_order': ['BB', 'SC1', 'SC2', 'SC3'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG'],
        'SC2': ['ND1', 'HD1', 'CE1', 'HE1'],
        'SC3': ['CD2', 'HD2', 'NE2', 'HE2'],
    },
}
# HSD/HSE/HSP variants share the same heavy atom mapping
MARTINI_MAPPINGS['HSD'] = dict(MARTINI_MAPPINGS['HIS'], description='Histidine (delta-protonated)')
MARTINI_MAPPINGS['HSE'] = dict(MARTINI_MAPPINGS['HIS'], description='Histidine (epsilon-protonated)')
MARTINI_MAPPINGS['HSP'] = dict(MARTINI_MAPPINGS['HIS'], description='Histidine (doubly protonated)')

MARTINI_MAPPINGS['TYR'] = {
    'description': 'Tyrosine',
    'category': 'protein',
    'bead_order': ['BB', 'SC1', 'SC2', 'SC3'],
    'atom_names': {
        'BB': ['N', 'HN', 'CA', 'HA', 'C', 'O'],
        'SC1': ['CB', 'HB1', 'HB2', 'CG'],
        'SC2': ['CD1', 'HD1', 'CE1', 'HE1'],
        'SC3': ['CD2', 'HD2', 'CE2', 'HE2', 'CZ', 'OH', 'HH'],
    },
}

# -----------------------------------------------------------------------------
# ALKANES (small molecules)
# -----------------------------------------------------------------------------

MARTINI_MAPPINGS['OCT'] = {
    'description': 'Octane C8H18 (MARTINI, OPLS-AA naming)',
    'category': 'alkane',
    'bead_order': ['C1A', 'C1B'],
    'atom_names': {
        'C1A': ['C00', 'C01', 'C02', 'C03',
                'H08', 'H09', 'H0A', 'H0B', 'H0C', 'H0D', 'H0E', 'H0F', 'H0G'],
        'C1B': ['C04', 'C05', 'C06', 'C07',
                'H0H', 'H0I', 'H0J', 'H0K', 'H0M', 'H0N', 'H0O', 'H0P', 'H0Q'],
    },
}

MARTINI_MAPPINGS['OCT4'] = {
    'description': 'Octane C8H18 4-bead (2+2+2+2, OPLS-AA naming)',
    'category': 'alkane',
    'bead_order': ['E1', 'M1', 'M2', 'E2'],
    'atom_names': {
        'E1': ['C00', 'C01', 'H08', 'H09', 'H0A', 'H0B', 'H0C'],
        'M1': ['C02', 'C03', 'H0D', 'H0E', 'H0F', 'H0G'],
        'M2': ['C04', 'C05', 'H0H', 'H0I', 'H0J', 'H0K'],
        'E2': ['C06', 'C07', 'H0M', 'H0N', 'H0O', 'H0P', 'H0Q'],
    },
}

MARTINI_MAPPINGS['OCT8'] = {
    'description': 'Octane C8H18 8-bead (1 carbon + Hs per bead, OPLS-AA naming)',
    'category': 'alkane',
    'bead_order': ['CH3a', 'CH2b', 'CH2c', 'CH2d', 'CH2e', 'CH2f', 'CH2g', 'CH3h'],
    'atom_names': {
        'CH3a': ['C00', 'H08', 'H09', 'H0A'],
        'CH2b': ['C01', 'H0B', 'H0C'],
        'CH2c': ['C02', 'H0D', 'H0E'],
        'CH2d': ['C03', 'H0F', 'H0G'],
        'CH2e': ['C04', 'H0H', 'H0I'],
        'CH2f': ['C05', 'H0J', 'H0K'],
        'CH2g': ['C06', 'H0M', 'H0N'],
        'CH3h': ['C07', 'H0O', 'H0P', 'H0Q'],
    },
}

MARTINI_MAPPINGS['OCN'] = {
    'description': 'Octanol C8H17OH (MARTINI, OPLS-AA naming)',
    'category': 'alkane',
    'bead_order': ['C1A', 'C1B', 'P1'],
    'atom_names': {
        'C1A': ['C00', 'C01', 'C02', 'C03',
                'H09', 'H0A', 'H0B', 'H0C', 'H0D', 'H0E', 'H0F', 'H0G', 'H0H'],
        'C1B': ['C04', 'C05', 'C06',
                'H0I', 'H0J', 'H0K', 'H0M', 'H0N', 'H0O'],
        'P1':  ['C07', 'O08',
                'H0P', 'H0Q', 'H0R'],
    },
}

# -----------------------------------------------------------------------------
# WATER / IONS (special handling)
# -----------------------------------------------------------------------------

MARTINI_MAPPINGS['SOL'] = {
    'description': 'MARTINI water (4 molecules -> 1 W bead)',
    'category': 'water',
    'type': 'water_cluster',
    'waters_per_bead': 4,
    'bead_order': ['W'],
    'atom_names': {},
}

# Alternative water residue names (all map to the same SOL definition)
for _water_name in ['WAT', 'HOH', 'TIP3', 'TIP4', 'SPC', 'SPCE', 'TIP3P']:
    MARTINI_MAPPINGS[_water_name] = dict(MARTINI_MAPPINGS['SOL'],
                                          description=f'Water ({_water_name} -> MARTINI W bead)')

# Ions (typically excluded from CG mapping)
for _ion_name in ['NA', 'CL', 'K', 'CA', 'MG', 'ZN']:
    MARTINI_MAPPINGS[_ion_name] = {
        'description': f'{_ion_name} ion',
        'category': 'ion',
        'type': 'ion',
        'bead_order': [],
        'atom_names': {},
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_residue_mapping(resname: str) -> Optional[dict]:
    """Get MARTINI mapping for a residue name.

    Args:
        resname: Residue name (e.g., 'DLIPC', 'ALA', 'DA', 'SOL')

    Returns:
        Mapping dict or None if not found
    """
    return MARTINI_MAPPINGS.get(resname.strip())


def get_available_residues() -> Dict[str, str]:
    """Get dict of all available residue names and their descriptions."""
    return {name: m['description'] for name, m in MARTINI_MAPPINGS.items()
            if m.get('category') != 'ion'}


def get_residues_by_category(category: str) -> Dict[str, str]:
    """Get residue names for a specific category (lipid, dna, rna, protein, water)."""
    return {name: m['description'] for name, m in MARTINI_MAPPINGS.items()
            if m.get('category') == category}


def print_martini_registry():
    """Print all available MARTINI mappings organized by category."""
    categories = ['lipid', 'dna', 'rna', 'protein', 'alkane', 'water', 'ion']
    category_labels = {
        'lipid': 'Lipids',
        'dna': 'DNA Nucleotides',
        'rna': 'RNA Nucleotides',
        'protein': 'Protein Amino Acids',
        'alkane': 'Alkanes',
        'water': 'Water',
        'ion': 'Ions (excluded from CG mapping)',
    }

    print("=" * 70)
    print("MARTINI Coarse-Grained Mapping Registry")
    print("=" * 70)

    for cat in categories:
        residues = get_residues_by_category(cat)
        if not residues:
            continue
        print(f"\n  {category_labels.get(cat, cat)}:")
        for name, desc in sorted(residues.items()):
            mapping = MARTINI_MAPPINGS[name]
            n_beads = len(mapping.get('bead_order', []))
            bead_names = ', '.join(mapping.get('bead_order', []))
            if mapping.get('type') == 'water_cluster':
                print(f"    {name:8s} - {desc}")
                print(f"             {mapping['waters_per_bead']} molecules -> 1 W bead")
            elif mapping.get('type') == 'ion':
                print(f"    {name:8s} - {desc}")
            else:
                print(f"    {name:8s} - {desc} ({n_beads} beads: {bead_names})")

    print("\n" + "=" * 70)

MARTINI_MAPPINGS['OCN5'] = {
    'description': 'Octanol C8H17OH 5-bead (2C groups + OH)',
    'category': 'alkane',
    'bead_order': ['A1', 'A2', 'A3', 'A4', 'OH'],
    'atom_names': {
        'A1': ['C00', 'C01', 'H09', 'H0A', 'H0B', 'H0C', 'H0D'],
        'A2': ['C02', 'C03', 'H0E', 'H0F', 'H0G', 'H0H'],
        'A3': ['C04', 'C05', 'H0I', 'H0J', 'H0K', 'H0M'],
        'A4': ['C06', 'C07', 'H0N', 'H0O', 'H0P', 'H0Q'],
        'OH': ['O08', 'H0R'],
    },
}

MARTINI_MAPPINGS['OCN9'] = {
    'description': 'Octanol C8H17OH 9-bead (1 heavy atom per bead)',
    'category': 'alkane',
    'bead_order': ['CH3a', 'CH2b', 'CH2c', 'CH2d', 'CH2e', 'CH2f', 'CH2g', 'CH2h', 'OHi'],
    'atom_names': {
        'CH3a': ['C00', 'H09', 'H0A', 'H0B'],
        'CH2b': ['C01', 'H0C', 'H0D'],
        'CH2c': ['C02', 'H0E', 'H0F'],
        'CH2d': ['C03', 'H0G', 'H0H'],
        'CH2e': ['C04', 'H0I', 'H0J'],
        'CH2f': ['C05', 'H0K', 'H0M'],
        'CH2g': ['C06', 'H0N', 'H0O'],
        'CH2h': ['C07', 'H0P', 'H0Q'],
        'OHi':  ['O08', 'H0R'],
    },
}
