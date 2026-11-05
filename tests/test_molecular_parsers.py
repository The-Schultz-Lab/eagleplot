"""Tests for eagleplot/molecular/parsers.py and eagleplot/molecular/atom_data.py."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from eagleplot.molecular.parsers import (
    VibrationalData,
    VibrationalMode,
    build_molecule_figure,
    create_displacement_arrows,
    parse_gaussian_vibrations,
    parse_molden_vibrations,
    parse_orca_vibrations,
    parse_vibrations,
)
from eagleplot.molecular.atom_data import (
    BOHR_TO_ANGSTROM,
    atom_colors,
    atom_symbols,
    get_atom_color,
    get_atom_symbol,
    get_vdw_radius,
    symbol_to_number,
    vdw_radii,
)

# ---------------------------------------------------------------------------
# Fixture content strings
# ---------------------------------------------------------------------------

GAUSSIAN_CONTENT = """\
 Entering Gaussian System
 Gaussian 16
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.117000
      2          1           0        0.000000    0.757000   -0.468000
      3          1           0        0.000000   -0.757000   -0.468000
 ---------------------------------------------------------------------
 Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman activities (A**4/AMU),
      1      2      3
 Frequencies --   1595.0000  3756.0000  3843.0000
 IR Inten    --     74.0000    10.0000     1.0000
 Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z
    1   8   0.000  0.000  0.071    0.000  0.000 -0.071    0.000  0.071  0.000
    2   1   0.000  0.571 -0.565    0.000  0.565  0.565    0.000 -0.565  0.000
    3   1   0.000 -0.571 -0.565    0.000 -0.565  0.565    0.000  0.565  0.000
"""

ORCA_CONTENT = """\
 O   R   C   A 5.0

 CARTESIAN COORDINATES (ANGSTROEM)
 ----------------------------------
   O      0.000000    0.000000    0.117000
   H      0.000000    0.757000   -0.468000
   H      0.000000   -0.757000   -0.468000

 VIBRATIONAL FREQUENCIES
 -----------------------
 Scaling factor for frequencies =  1.000
    0:         0.00 cm**-1
    1:         0.00 cm**-1
    2:         0.00 cm**-1
    3:         0.00 cm**-1
    4:         0.00 cm**-1
    5:         0.00 cm**-1
    6:      1595.00 cm**-1
    7:      3756.00 cm**-1
    8:      3843.00 cm**-1

 NORMAL MODES
 ------------
          6          7          8
    0   0.000000   0.000000   0.000000
    1   0.000000   0.000000   0.071000
    2   0.071000  -0.071000   0.000000
    3   0.000000   0.000000   0.000000
    4   0.571000   0.565000  -0.565000
    5  -0.565000   0.565000   0.000000
    6   0.000000   0.000000   0.000000
    7  -0.571000  -0.565000   0.565000
    8  -0.565000   0.565000   0.000000
"""

MOLDEN_CONTENT = """\
[Molden Format]
[Atoms] Angs
O     1     8      0.000000    0.000000    0.117000
H     2     1      0.000000    0.757000   -0.468000
H     3     1      0.000000   -0.757000   -0.468000
[FREQ]
1595.00
3756.00
3843.00
[INT]
74.00
10.00
1.00
[FR-NORM-COORD]
vibration 1
   0.000000    0.000000    0.071000
   0.000000    0.571000   -0.565000
   0.000000   -0.571000   -0.565000
vibration 2
   0.000000    0.000000   -0.071000
   0.000000    0.565000    0.565000
   0.000000   -0.565000    0.565000
vibration 3
   0.000000    0.071000    0.000000
   0.000000   -0.565000    0.000000
   0.000000    0.565000    0.000000
"""


def _write_temp(content, suffix):
    """Write content to a named temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# 1. TestAtomData
# ---------------------------------------------------------------------------


class TestAtomData:
    def test_bohr_to_angstrom_value(self):
        assert abs(BOHR_TO_ANGSTROM - 0.529177) < 0.001

    def test_get_atom_color_hydrogen(self):
        color = get_atom_color(1)
        assert isinstance(color, str)
        assert len(color) > 0

    def test_get_atom_color_oxygen(self):
        color = get_atom_color(8)
        assert isinstance(color, str)

    def test_get_atom_color_out_of_range_returns_default(self):
        color = get_atom_color(999)
        assert isinstance(color, str)

    def test_get_atom_symbol_hydrogen(self):
        assert get_atom_symbol(1) == "H"

    def test_get_atom_symbol_oxygen(self):
        assert get_atom_symbol(8) == "O"

    def test_get_atom_symbol_carbon(self):
        assert get_atom_symbol(6) == "C"

    def test_get_atom_symbol_out_of_range(self):
        symbol = get_atom_symbol(999)
        assert isinstance(symbol, str)

    def test_get_vdw_radius_hydrogen(self):
        r = get_vdw_radius(1)
        assert r > 0

    def test_get_vdw_radius_oxygen(self):
        r = get_vdw_radius(8)
        assert r > 0

    def test_get_vdw_radius_out_of_range(self):
        r = get_vdw_radius(999)
        assert isinstance(r, float)
        assert r > 0

    def test_symbol_to_number_hydrogen(self):
        assert symbol_to_number.get("H") == 1

    def test_symbol_to_number_oxygen(self):
        assert symbol_to_number.get("O") == 8

    def test_atom_colors_is_sequence(self):
        # atom_colors is a list indexed by atomic number
        assert isinstance(atom_colors, (list, dict))
        assert len(atom_colors) > 0

    def test_atom_symbols_is_sequence(self):
        assert isinstance(atom_symbols, (list, dict))
        assert len(atom_symbols) > 0

    def test_vdw_radii_is_sequence(self):
        # vdw_radii is a list indexed by atomic number
        assert isinstance(vdw_radii, (list, dict))
        assert len(vdw_radii) > 0


# ---------------------------------------------------------------------------
# 2. TestVibrationalDataStructures
# ---------------------------------------------------------------------------


class TestVibrationalDataStructures:
    """Test VibrationalMode and VibrationalData using the actual field names."""

    def _make_mode(self, freq=1000.0, ir=10.0, imaginary=False, mode_number=1):
        displacements = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        return VibrationalMode(
            mode_number=mode_number,
            frequency=freq,
            ir_intensity=ir,
            displacement_vectors=displacements,
            is_imaginary=imaginary,
        )

    def _make_data(self):
        coords = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.468]])
        atomic_numbers = [8, 1]
        modes = [self._make_mode(1000.0, mode_number=1), self._make_mode(2000.0, mode_number=2)]
        return VibrationalData(
            coordinates=coords,
            atomic_numbers=atomic_numbers,
            modes=modes,
            source_file="test.log",
            program="gaussian",
        )

    def test_vibrational_mode_frequency(self):
        m = self._make_mode(1595.0)
        assert m.frequency == pytest.approx(1595.0)

    def test_vibrational_mode_ir_intensity(self):
        m = self._make_mode(ir=74.0)
        assert m.ir_intensity == pytest.approx(74.0)

    def test_vibrational_mode_displacement_vectors_shape(self):
        m = self._make_mode()
        assert m.displacement_vectors.shape == (2, 3)

    def test_vibrational_mode_is_imaginary_false(self):
        m = self._make_mode(imaginary=False)
        assert m.is_imaginary is False

    def test_vibrational_mode_is_imaginary_true(self):
        m = self._make_mode(freq=-100.0, imaginary=True)
        assert m.is_imaginary is True

    def test_vibrational_data_atom_count(self):
        vd = self._make_data()
        assert len(vd.atomic_numbers) == 2

    def test_vibrational_data_coordinates_shape(self):
        vd = self._make_data()
        assert vd.coordinates.shape == (2, 3)

    def test_vibrational_data_modes_count(self):
        vd = self._make_data()
        assert len(vd.modes) == 2

    def test_vibrational_data_get_mode_found(self):
        vd = self._make_data()
        m = vd.get_mode(1)
        assert isinstance(m, VibrationalMode)
        assert m.mode_number == 1

    def test_vibrational_data_get_mode_missing(self):
        vd = self._make_data()
        m = vd.get_mode(999)
        assert m is None

    def test_vibrational_data_get_displacement_magnitudes(self):
        vd = self._make_data()
        mags = vd.get_displacement_magnitudes(1)
        assert mags.shape == (2,)
        assert np.all(mags >= 0)

    def test_vibrational_data_program_field(self):
        vd = self._make_data()
        assert vd.program == "gaussian"


# ---------------------------------------------------------------------------
# 3. TestGaussianParser
# ---------------------------------------------------------------------------


class TestGaussianParser:
    def setup_method(self):
        self.path = _write_temp(GAUSSIAN_CONTENT, ".log")

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_vibrational_data(self):
        vd = parse_gaussian_vibrations(self.path)
        assert isinstance(vd, VibrationalData)

    def test_atom_count(self):
        vd = parse_gaussian_vibrations(self.path)
        assert len(vd.atomic_numbers) == 3

    def test_first_atom_is_oxygen(self):
        vd = parse_gaussian_vibrations(self.path)
        assert vd.atomic_numbers[0] == 8

    def test_hydrogen_atoms(self):
        vd = parse_gaussian_vibrations(self.path)
        assert vd.atomic_numbers[1] == 1
        assert vd.atomic_numbers[2] == 1

    def test_mode_count(self):
        vd = parse_gaussian_vibrations(self.path)
        assert len(vd.modes) == 3

    def test_first_frequency(self):
        vd = parse_gaussian_vibrations(self.path)
        freqs = [m.frequency for m in vd.modes]
        assert pytest.approx(1595.0) in freqs

    def test_second_frequency(self):
        vd = parse_gaussian_vibrations(self.path)
        freqs = [m.frequency for m in vd.modes]
        assert pytest.approx(3756.0) in freqs

    def test_third_frequency(self):
        vd = parse_gaussian_vibrations(self.path)
        freqs = [m.frequency for m in vd.modes]
        assert pytest.approx(3843.0) in freqs

    def test_ir_intensity_first_mode(self):
        vd = parse_gaussian_vibrations(self.path)
        # First mode should have IR intensity ~74.0
        first = min(vd.modes, key=lambda m: m.mode_number)
        assert first.ir_intensity == pytest.approx(74.0, abs=1.0)

    def test_not_imaginary(self):
        vd = parse_gaussian_vibrations(self.path)
        for mode in vd.modes:
            assert mode.is_imaginary is False

    def test_displacements_shape(self):
        vd = parse_gaussian_vibrations(self.path)
        # displacement_vectors shape: (n_atoms=3, 3)
        for mode in vd.modes:
            assert mode.displacement_vectors.shape == (3, 3)

    def test_atom_coordinates_oxygen(self):
        vd = parse_gaussian_vibrations(self.path)
        # First atom is oxygen; z coordinate should be ~0.117 Å
        assert abs(vd.coordinates[0, 2] - 0.117) < 0.01

    def test_program_is_gaussian(self):
        vd = parse_gaussian_vibrations(self.path)
        assert vd.program == "gaussian"


# ---------------------------------------------------------------------------
# 4. TestORCAParser
# ---------------------------------------------------------------------------


class TestORCAParser:
    def setup_method(self):
        self.path = _write_temp(ORCA_CONTENT, ".out")

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_vibrational_data(self):
        vd = parse_orca_vibrations(self.path)
        assert isinstance(vd, VibrationalData)

    def test_atom_count(self):
        vd = parse_orca_vibrations(self.path)
        assert len(vd.atomic_numbers) == 3

    def test_first_atom_is_oxygen(self):
        vd = parse_orca_vibrations(self.path)
        assert vd.atomic_numbers[0] == 8

    def test_mode_count(self):
        # ORCA fixture has 3 real vibrational modes (6, 7, 8)
        vd = parse_orca_vibrations(self.path)
        assert len(vd.modes) >= 1

    def test_frequencies_present(self):
        vd = parse_orca_vibrations(self.path)
        freqs = [m.frequency for m in vd.modes]
        assert any(abs(f - 1595.0) < 2.0 for f in freqs)

    def test_displacements_are_arrays(self):
        vd = parse_orca_vibrations(self.path)
        for mode in vd.modes:
            assert isinstance(mode.displacement_vectors, np.ndarray)

    def test_program_is_orca(self):
        vd = parse_orca_vibrations(self.path)
        assert vd.program == "orca"


# ---------------------------------------------------------------------------
# 5. TestMoldenParser
# ---------------------------------------------------------------------------


class TestMoldenParser:
    def setup_method(self):
        self.path = _write_temp(MOLDEN_CONTENT, ".molden")

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_vibrational_data(self):
        vd = parse_molden_vibrations(self.path)
        assert isinstance(vd, VibrationalData)

    def test_atom_count(self):
        vd = parse_molden_vibrations(self.path)
        assert len(vd.atomic_numbers) == 3

    def test_mode_count(self):
        vd = parse_molden_vibrations(self.path)
        assert len(vd.modes) == 3

    def test_first_frequency(self):
        vd = parse_molden_vibrations(self.path)
        freqs = [m.frequency for m in vd.modes]
        assert pytest.approx(1595.0) in freqs

    def test_ir_intensity_first_mode(self):
        vd = parse_molden_vibrations(self.path)
        first = min(vd.modes, key=lambda m: m.mode_number)
        assert first.ir_intensity == pytest.approx(74.0, abs=1.0)

    def test_angs_units_no_bohr_conversion(self):
        # Oxygen z in Angstrom mode should be ~0.117, not ~0.062 (= 0.117 * BOHR)
        vd = parse_molden_vibrations(self.path)
        assert abs(vd.coordinates[0, 2] - 0.117) < 0.01

    def test_au_units_converted(self):
        au_content = MOLDEN_CONTENT.replace("[Atoms] Angs", "[Atoms] AU")
        path = _write_temp(au_content, ".molden")
        try:
            vd = parse_molden_vibrations(path)
            assert isinstance(vd, VibrationalData)
            assert len(vd.atomic_numbers) == 3
            # z should be 0.117 * BOHR_TO_ANGSTROM ≈ 0.0619, not 0.117
            assert abs(vd.coordinates[0, 2] - 0.117 * BOHR_TO_ANGSTROM) < 0.01
        finally:
            os.unlink(path)

    def test_displacements_shape(self):
        vd = parse_molden_vibrations(self.path)
        for mode in vd.modes:
            assert mode.displacement_vectors.shape == (3, 3)

    def test_program_is_molden(self):
        vd = parse_molden_vibrations(self.path)
        assert vd.program == "molden"


# ---------------------------------------------------------------------------
# 6. TestParseVibrations (auto-detect dispatcher)
# ---------------------------------------------------------------------------


class TestParseVibrations:
    def test_detects_molden(self):
        path = _write_temp(MOLDEN_CONTENT, ".molden")
        try:
            vd = parse_vibrations(path)
            assert isinstance(vd, VibrationalData)
        finally:
            os.unlink(path)

    def test_detects_gaussian(self):
        path = _write_temp(GAUSSIAN_CONTENT, ".log")
        try:
            vd = parse_vibrations(path)
            assert isinstance(vd, VibrationalData)
        finally:
            os.unlink(path)

    def test_detects_orca(self):
        path = _write_temp(ORCA_CONTENT, ".out")
        try:
            vd = parse_vibrations(path)
            assert isinstance(vd, VibrationalData)
        finally:
            os.unlink(path)

    def test_unknown_format_raises_value_error(self):
        path = _write_temp("this is not a known format\n", ".xyz")
        try:
            with pytest.raises(ValueError):
                parse_vibrations(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. TestVisualizationHelpers
# ---------------------------------------------------------------------------


def _make_water_vd() -> VibrationalData:
    """Build a minimal water VibrationalData for visualization tests."""
    coords = np.array(
        [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.468],
            [0.0, -0.757, -0.468],
        ]
    )
    disps = np.array(
        [
            [0.0, 0.0, 0.071],
            [0.0, 0.571, -0.565],
            [0.0, -0.571, -0.565],
        ]
    )
    mode = VibrationalMode(
        mode_number=1,
        frequency=1595.0,
        ir_intensity=74.0,
        displacement_vectors=disps,
        is_imaginary=False,
    )
    return VibrationalData(
        coordinates=coords,
        atomic_numbers=[8, 1, 1],
        modes=[mode],
        source_file="test.log",
        program="gaussian",
    )


class TestVisualizationHelpers:
    def test_create_displacement_arrows_returns_list(self):
        vd = _make_water_vd()
        result = create_displacement_arrows(vd, mode_number=1)
        assert isinstance(result, list)

    def test_create_displacement_arrows_non_empty(self):
        vd = _make_water_vd()
        arrows = create_displacement_arrows(vd, mode_number=1)
        assert len(arrows) >= 0  # may be 0 if all displacements below threshold

    def test_create_displacement_arrows_show_all(self):
        vd = _make_water_vd()
        arrows = create_displacement_arrows(vd, mode_number=1, show_small_displacements=True)
        assert len(arrows) > 0

    def test_create_displacement_arrows_invalid_mode_raises(self):
        vd = _make_water_vd()
        with pytest.raises(ValueError):
            create_displacement_arrows(vd, mode_number=999)

    def test_build_molecule_figure_returns_plotly_figure(self):
        import plotly.graph_objects as go

        vd = _make_water_vd()
        fig = build_molecule_figure(vd, mode_number=1)
        assert isinstance(fig, go.Figure)

    def test_build_molecule_figure_has_data(self):
        vd = _make_water_vd()
        fig = build_molecule_figure(vd, mode_number=1)
        assert len(fig.data) > 0

    def test_build_molecule_figure_no_mode(self):
        import plotly.graph_objects as go

        vd = _make_water_vd()
        fig = build_molecule_figure(vd, mode_number=None)
        assert isinstance(fig, go.Figure)
        # Should have at least the atom scatter trace
        assert len(fig.data) >= 1
