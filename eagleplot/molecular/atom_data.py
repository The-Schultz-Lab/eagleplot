"""Atomic property data for molecular visualization.

Provides element colors (CPK scheme), symbols, van der Waals radii,
and helper functions for element lookups by atomic number.
"""

BOHR_TO_ANGSTROM = 0.529177210903

# CPK-style colors indexed by atomic number (index 0 = unknown/placeholder)
_atom_colors_raw = [
    "#000000",  # 0  unknown
    "#FFFFFF",  # 1  H
    "cyan",  # 2  He
    "violet",  # 3  Li
    "#B22222",  # 4  Be
    "beige",  # 5  B
    "#444444",  # 6  C
    "blue",  # 7  N
    "red",  # 8  O
    "green",  # 9  F
    "cyan",  # 10 Ne
    "#FF1493",  # 11 Na
    "#A52A2A",  # 12 Mg
    "#800080",  # 13 Al
    "#D2691E",  # 14 Si
    "#228B22",  # 15 P
    "yellow",  # 16 S
    "green",  # 17 Cl
    "cyan",  # 18 Ar
    "#0000CD",  # 19 K
    "#8B4513",  # 20 Ca
    "#8FBC8F",  # 21 Sc
    "#B8860B",  # 22 Ti
    "#4682B4",  # 23 V
    "#B22222",  # 24 Cr
    "#2E8B57",  # 25 Mn
    "#FFD700",  # 26 Fe
    "#DAA520",  # 27 Co
    "#A52A2A",  # 28 Ni
    "#4169E1",  # 29 Cu
    "#708090",  # 30 Zn
    "#C0C0C0",  # 31 Ga
    "#808000",  # 32 Ge
    "#00FF00",  # 33 As
    "#00CED1",  # 34 Se
    "#0000FF",  # 35 Br
    "cyan",  # 36 Kr
    "#8A2BE2",  # 37 Rb
    "#8B4513",  # 38 Sr
    "#9400D3",  # 39 Y
    "#FF4500",  # 40 Zr
    "#4682B4",  # 41 Nb
    "#B22222",  # 42 Mo
    "#FFD700",  # 43 Tc
    "#A52A2A",  # 44 Ru
    "#228B22",  # 45 Rh
    "#8B4513",  # 46 Pd
    "#00CED1",  # 47 Ag
    "#696969",  # 48 Cd
    "#C0C0C0",  # 49 In
    "#808000",  # 50 Sn
    "#228B22",  # 51 Sb
    "#D2691E",  # 52 Te
    "#00FF00",  # 53 I
    "cyan",  # 54 Xe
    "#8A2BE2",  # 55 Cs
    "#8B4513",  # 56 Ba
    "#FF4500",  # 57 La
]

# Pad to 119 entries (elements up to Og, index 118)
atom_colors: list[str] = _atom_colors_raw + ["#C0C0C0"] * (119 - len(_atom_colors_raw))

# Element symbols indexed by atomic number (index 0 = unknown placeholder)
atom_symbols: list[str] = [
    "unk",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

# Van der Waals radii in Angstroms indexed by atomic number (index 0 = fallback)
_vdw_radii_raw = [
    1.70,  # 0  unknown/fallback
    1.20,  # 1  H
    1.40,  # 2  He
    1.82,  # 3  Li
    1.53,  # 4  Be
    1.92,  # 5  B
    1.70,  # 6  C
    1.55,  # 7  N
    1.52,  # 8  O
    1.47,  # 9  F
    1.54,  # 10 Ne
    2.27,  # 11 Na
    1.73,  # 12 Mg
    1.84,  # 13 Al
    2.10,  # 14 Si
    1.80,  # 15 P
    1.80,  # 16 S
    1.75,  # 17 Cl
    1.88,  # 18 Ar
    2.75,  # 19 K
    2.31,  # 20 Ca
    2.11,  # 21 Sc
    2.00,  # 22 Ti
    2.00,  # 23 V
    2.00,  # 24 Cr
    2.00,  # 25 Mn
    2.00,  # 26 Fe
    2.00,  # 27 Co
    1.63,  # 28 Ni
    1.40,  # 29 Cu
    1.39,  # 30 Zn
    1.87,  # 31 Ga
    2.11,  # 32 Ge
    1.85,  # 33 As
    1.90,  # 34 Se
    1.85,  # 35 Br
    2.02,  # 36 Kr
]

# Pad to 119 entries with 2.00 Å as a reasonable default
vdw_radii: list[float] = _vdw_radii_raw + [2.00] * (119 - len(_vdw_radii_raw))

# Reverse lookup: element symbol → atomic number
symbol_to_number: dict[str, int] = {sym: i for i, sym in enumerate(atom_symbols)}


def get_atom_color(atomic_number: int) -> str:
    """Return the CPK color string for an element.

    Args:
        atomic_number: Atomic number (1-based; 0 returns the unknown color).

    Returns:
        Hex color string or named color. Falls back to ``"#C0C0C0"`` (silver)
        for out-of-range values.
    """
    if 0 <= atomic_number < len(atom_colors):
        return atom_colors[atomic_number]
    return "#C0C0C0"


def get_atom_symbol(atomic_number: int) -> str:
    """Return the element symbol for an atomic number.

    Args:
        atomic_number: Atomic number (1-based).

    Returns:
        Element symbol string (e.g. ``"C"``, ``"O"``).  Returns ``"?"``
        for out-of-range values.
    """
    if 0 <= atomic_number < len(atom_symbols):
        return atom_symbols[atomic_number]
    return "?"


def get_vdw_radius(atomic_number: int) -> float:
    """Return the van der Waals radius (Å) for an element.

    Args:
        atomic_number: Atomic number (1-based).

    Returns:
        Van der Waals radius in Angstroms.  Returns ``1.70`` for
        out-of-range values.
    """
    if 0 <= atomic_number < len(vdw_radii):
        return vdw_radii[atomic_number]
    return 1.70
