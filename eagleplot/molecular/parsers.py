"""Molecular vibration parsers and visualization helpers.

Provides functionality for parsing vibrational frequency data from quantum
chemistry output files (Gaussian, ORCA, Molden) and building Plotly figures
for molecular visualization.

Classes:
    VibrationalMode: Data structure for a single vibrational mode.
    VibrationalData: Container for a complete vibrational analysis.

Functions:
    parse_gaussian_vibrations: Parse Gaussian .log/.out files.
    parse_orca_vibrations: Parse ORCA .out files.
    parse_molden_vibrations: Parse Molden .molden files.
    parse_vibrations: Auto-detect format and parse.
    create_displacement_arrows: Generate Plotly Cone arrow traces.
    build_molecule_figure: Build a self-contained Plotly 3-D figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import plotly.graph_objects as go

from .atom_data import (
    BOHR_TO_ANGSTROM,
    get_atom_color,
    get_atom_symbol,
    get_vdw_radius,
    symbol_to_number,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VibrationalMode:
    """Single vibrational mode data.

    Attributes:
        mode_number: Mode index (1-based, following QM convention).
        frequency: Frequency in cm⁻¹ (negative for imaginary modes).
        ir_intensity: IR intensity in km/mol, or ``None`` if not available.
        displacement_vectors: Array of shape ``(n_atoms, 3)`` with Cartesian
            displacements.
        is_imaginary: ``True`` when the frequency is negative.
    """

    mode_number: int
    frequency: float
    ir_intensity: Optional[float]
    displacement_vectors: np.ndarray  # shape: (n_atoms, 3)
    is_imaginary: bool


@dataclass
class VibrationalData:
    """Complete vibrational analysis data.

    Attributes:
        coordinates: Atomic coordinates of shape ``(n_atoms, 3)`` in Angstroms.
        atomic_numbers: List of atomic numbers (length ``n_atoms``).
        modes: List of :class:`VibrationalMode` objects.
        source_file: Original filename for reference.
        program: Source program identifier — ``"gaussian"``, ``"orca"``, or
            ``"molden"``.
    """

    coordinates: np.ndarray  # shape: (n_atoms, 3)
    atomic_numbers: list[int]
    modes: list[VibrationalMode]
    source_file: str
    program: str

    def get_mode(self, mode_number: int) -> Optional[VibrationalMode]:
        """Retrieve a mode by its 1-based mode number.

        Args:
            mode_number: Mode number to retrieve.

        Returns:
            :class:`VibrationalMode` if found, ``None`` otherwise.
        """
        for mode in self.modes:
            if mode.mode_number == mode_number:
                return mode
        return None

    def get_displacement_magnitudes(self, mode_number: int) -> np.ndarray:
        """Calculate per-atom displacement magnitude for a given mode.

        Args:
            mode_number: Mode number to analyse (1-based).

        Returns:
            1-D array of displacement magnitudes (one value per atom).
            An empty array is returned when the mode is not found.
        """
        mode = self.get_mode(mode_number)
        if mode is None:
            return np.array([])
        return np.linalg.norm(mode.displacement_vectors, axis=1)


# ---------------------------------------------------------------------------
# Gaussian parser
# ---------------------------------------------------------------------------


def parse_gaussian_vibrations(filepath: str) -> VibrationalData:
    """Parse vibrational data from a Gaussian .log or .out file.

    Strategy:

    1. Find the last ``Standard orientation`` section for equilibrium
       coordinates.
    2. Locate the ``Harmonic frequencies`` section.
    3. Parse frequency blocks (printed in groups of up to 5 modes).
    4. Extract frequencies, IR intensities, and displacement vectors.

    Args:
        filepath: Path to the Gaussian output file.

    Returns:
        :class:`VibrationalData` populated from the file.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If no vibrational data or coordinates are found.
    """
    with open(filepath) as f:
        content = f.read()

    # 1. Extract coordinates from the last "Standard orientation" section.
    coord_pattern = r"Standard orientation:.*?-{50,}.*?-{50,}\s*(.*?)\s*-{50,}"
    coord_matches = list(re.finditer(coord_pattern, content, re.DOTALL))
    if not coord_matches:
        raise ValueError("No coordinates found in Gaussian file")

    last_coords_block = coord_matches[-1].group(1)
    coords: list[list[float]] = []
    atomic_numbers: list[int] = []

    coord_line_pattern = r"\s*\d+\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    for match in re.finditer(coord_line_pattern, last_coords_block):
        atomic_numbers.append(int(match.group(1)))
        coords.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])

    coords_arr = np.array(coords)
    n_atoms = len(coords_arr)

    # 2. Find the vibrational section.
    vib_pattern = r"Harmonic frequencies \(cm\*\*-1\), IR intensities.*?(?=\n\s*-{3,}|\Z)"
    vib_match = re.search(vib_pattern, content, re.DOTALL)
    if not vib_match:
        raise ValueError("No vibrational data found in Gaussian file")

    vib_section = vib_match.group(0)

    # 3. Parse frequency blocks.
    modes: list[VibrationalMode] = []

    mode_header_pattern = (
        r"^\s+(\d+)\s{5,}(\d+)(?:\s{5,}(\d+))?(?:\s{5,}(\d+))?(?:\s{5,}(\d+))?\s*$"
    )

    mode_blocks = []
    for match in re.finditer(mode_header_pattern, vib_section, re.MULTILINE):
        mode_nums = [int(x) for x in match.groups() if x is not None]
        next_match = re.search(mode_header_pattern, vib_section[match.end() :], re.MULTILINE)
        end_pos = match.end() + next_match.start() if next_match else len(vib_section)
        block_content = vib_section[match.end() : end_pos]
        mode_blocks.append((mode_nums, block_content))

    for mode_nums, block in mode_blocks:
        freq_match = re.search(
            r"Frequencies --\s+([-\d.]+)\s+([-\d.]+)"
            r"\s*(?:([-\d.]+))?\s*(?:([-\d.]+))?\s*(?:([-\d.]+))?",
            block,
        )
        if not freq_match:
            continue

        frequencies = [float(x) for x in freq_match.groups() if x is not None]

        ir_match = re.search(
            r"IR Inten\s+--\s+([-\d.]+)\s+([-\d.]+)"
            r"\s*(?:([-\d.]+))?\s*(?:([-\d.]+))?\s*(?:([-\d.]+))?",
            block,
        )
        ir_intensities: list[Optional[float]] = (
            [float(x) for x in ir_match.groups() if x is not None]
            if ir_match
            else [None] * len(frequencies)
        )

        disp_start = block.find("Atom  AN")
        if disp_start == -1:
            continue

        disp_section = block[disp_start:]
        disp_lines = disp_section.split("\n")[1:]  # skip header

        n_modes_in_block = len(frequencies)
        displacements = [np.zeros((n_atoms, 3)) for _ in range(n_modes_in_block)]

        atom_idx = 0
        line_idx = 0
        while atom_idx < n_atoms and line_idx < len(disp_lines):
            line = disp_lines[line_idx]
            if not line.strip():
                line_idx += 1
                continue
            parts = line.split()
            if len(parts) < 5:
                line_idx += 1
                continue
            displacement_values = parts[2:]
            for mode_idx in range(n_modes_in_block):
                start_idx = mode_idx * 3
                if start_idx + 2 < len(displacement_values):
                    displacements[mode_idx][atom_idx, 0] = float(displacement_values[start_idx])
                    displacements[mode_idx][atom_idx, 1] = float(displacement_values[start_idx + 1])
                    displacements[mode_idx][atom_idx, 2] = float(displacement_values[start_idx + 2])
            atom_idx += 1
            line_idx += 1

        for mode_num, freq, ir_int, disp in zip(
            mode_nums[: len(frequencies)],
            frequencies,
            ir_intensities,
            displacements,
        ):
            modes.append(
                VibrationalMode(
                    mode_number=mode_num,
                    frequency=freq,
                    ir_intensity=ir_int,
                    displacement_vectors=disp,
                    is_imaginary=(freq < 0),
                )
            )

    return VibrationalData(
        coordinates=coords_arr,
        atomic_numbers=atomic_numbers,
        modes=modes,
        source_file=filepath,
        program="gaussian",
    )


# ---------------------------------------------------------------------------
# ORCA parser
# ---------------------------------------------------------------------------


def parse_orca_vibrations(filepath: str) -> VibrationalData:
    """Parse vibrational data from an ORCA .out file.

    Strategy:

    1. Find ``CARTESIAN COORDINATES (ANGSTROEM)`` for the geometry.
    2. Locate the ``VIBRATIONAL FREQUENCIES`` section.
    3. Skip the first 6 translations/rotations (near-zero frequencies).
    4. Extract displacement vectors from the ``NORMAL MODES`` section
       (ORCA prints modes in columns, 6 per block).

    Args:
        filepath: Path to the ORCA output file.

    Returns:
        :class:`VibrationalData` populated from the file.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If required sections are absent or malformed.
    """
    with open(filepath) as f:
        content = f.read()

    # 1. Coordinates.
    coord_pattern = r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*-+\s*(.*?)(?:\n\s*\n|\Z)"
    coord_match = re.search(coord_pattern, content, re.DOTALL)
    if not coord_match:
        raise ValueError("No coordinates found in ORCA file")

    coord_section = coord_match.group(1)
    coords: list[list[float]] = []
    atomic_numbers: list[int] = []

    coord_line_pattern = r"\s*([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    for match in re.finditer(coord_line_pattern, coord_section):
        sym = match.group(1)
        x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
        atomic_num = symbol_to_number.get(sym)
        if atomic_num is None:
            raise ValueError(f"Unknown element symbol: {sym}")
        atomic_numbers.append(atomic_num)
        coords.append([x, y, z])

    coords_arr = np.array(coords)
    n_atoms = len(coords_arr)

    # 2. Vibrational frequencies.
    freq_pattern = r"VIBRATIONAL FREQUENCIES\s*-+.*?Scaling factor.*?\n(.*?)(?:\n\s*-+|\Z)"
    freq_match = re.search(freq_pattern, content, re.DOTALL)
    if not freq_match:
        raise ValueError("No vibrational frequencies found in ORCA file")

    freq_section = freq_match.group(1)
    frequencies: list[tuple[int, float]] = []
    freq_line_pattern = r"\s*(\d+):\s+([-\d.]+)\s+cm"
    for match in re.finditer(freq_line_pattern, freq_section):
        mode_idx = int(match.group(1))
        freq = float(match.group(2))
        frequencies.append((mode_idx, freq))

    # Keep only non-zero modes (skip translations/rotations).
    vibrational_freqs = [(idx, freq) for idx, freq in frequencies if abs(freq) > 1.0]

    # 3. Normal modes.
    modes_pattern = r"NORMAL MODES\s*-+(.*?)(?:\n-{3,}|\Z)"
    modes_match = re.search(modes_pattern, content, re.DOTALL)
    if not modes_match:
        raise ValueError("No normal modes found in ORCA file")

    modes_section = modes_match.group(1)

    mode_header_pattern = (
        r"^\s+(\d+)(?:\s+(\d+))?(?:\s+(\d+))?(?:\s+(\d+))?(?:\s+(\d+))?(?:\s+(\d+))?\s*$"
    )

    mode_blocks: list[tuple[list[int], list[list[float]]]] = []
    lines = modes_section.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        header_match = re.match(mode_header_pattern, line)
        if header_match:
            mode_nums = [int(x) for x in header_match.groups() if x is not None]
            displacement_data: list[list[float]] = []
            i += 1
            for _ in range(n_atoms * 3):
                if i >= len(lines):
                    break
                data_line = lines[i]
                parts = data_line.split()
                if len(parts) >= 2:
                    values = [float(x) for x in parts[1:]]
                    displacement_data.append(values)
                i += 1
            if len(displacement_data) == n_atoms * 3:
                mode_blocks.append((mode_nums, displacement_data))
        else:
            i += 1

    # 4. Build VibrationalMode objects.
    all_mode_displacements: dict[int, list[float]] = {}
    for mode_nums, disp_data in mode_blocks:
        for mode_idx, mode_num in enumerate(mode_nums):
            if mode_num not in all_mode_displacements:
                all_mode_displacements[mode_num] = []
            for row in disp_data:
                if mode_idx < len(row):
                    all_mode_displacements[mode_num].append(row[mode_idx])

    modes: list[VibrationalMode] = []
    for mode_idx, freq in vibrational_freqs:
        if mode_idx in all_mode_displacements:
            disp_vector = all_mode_displacements[mode_idx]
            if len(disp_vector) == n_atoms * 3:
                disp_array = np.array(disp_vector).reshape((n_atoms, 3))
                mode_number = len(modes) + 1
                modes.append(
                    VibrationalMode(
                        mode_number=mode_number,
                        frequency=freq,
                        ir_intensity=None,
                        displacement_vectors=disp_array,
                        is_imaginary=(freq < 0),
                    )
                )

    return VibrationalData(
        coordinates=coords_arr,
        atomic_numbers=atomic_numbers,
        modes=modes,
        source_file=filepath,
        program="orca",
    )


# ---------------------------------------------------------------------------
# Molden parser
# ---------------------------------------------------------------------------


def parse_molden_vibrations(filepath: str) -> VibrationalData:
    """Parse vibrational data from a Molden format file.

    Strategy:

    1. Parse ``[Atoms]`` section for coordinates and atomic numbers.
       Coordinates may be in Angstroms (``Angs``) or Bohr (``AU``).
    2. Parse ``[FREQ]`` section for frequencies.
    3. Parse ``[INT]`` section for IR intensities (optional).
    4. Parse ``[FR-NORM-COORD]`` section for displacement vectors.

    Args:
        filepath: Path to the Molden file.

    Returns:
        :class:`VibrationalData` populated from the file.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If required sections are missing or malformed.
    """
    with open(filepath) as f:
        content = f.read()

    # 1. [Atoms] section.
    atoms_pattern = r"\[Atoms\]\s+(Angs|AU)\s*\n(.*?)(?:\n\[|\Z)"
    atoms_match = re.search(atoms_pattern, content, re.DOTALL | re.IGNORECASE)
    if not atoms_match:
        raise ValueError("No [Atoms] section found in Molden file")

    unit = atoms_match.group(1).upper()
    atoms_section = atoms_match.group(2)

    coords: list[list[float]] = []
    atomic_numbers: list[int] = []

    atom_line_pattern = r"\s*([A-Z][a-z]?)\s+\d+\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    for match in re.finditer(atom_line_pattern, atoms_section):
        atomic_num = int(match.group(2))
        x, y, z = float(match.group(3)), float(match.group(4)), float(match.group(5))
        if unit == "AU":
            x *= BOHR_TO_ANGSTROM
            y *= BOHR_TO_ANGSTROM
            z *= BOHR_TO_ANGSTROM
        atomic_numbers.append(atomic_num)
        coords.append([x, y, z])

    coords_arr = np.array(coords)
    n_atoms = len(coords_arr)

    # 2. [FREQ] section.
    freq_pattern = r"\[FREQ\]\s*\n(.*?)(?:\n\[|\Z)"
    freq_match = re.search(freq_pattern, content, re.DOTALL)
    if not freq_match:
        raise ValueError("No [FREQ] section found in Molden file")

    frequencies: list[float] = []
    for line in freq_match.group(1).strip().split("\n"):
        line = line.strip()
        if line:
            frequencies.append(float(line))

    # 3. [INT] section (optional).
    int_pattern = r"\[INT\]\s*\n(.*?)(?:\n\[|\Z)"
    int_match = re.search(int_pattern, content, re.DOTALL)

    ir_intensities: list[Optional[float]] = []
    if int_match:
        for line in int_match.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                ir_intensities.append(float(line))

    if not ir_intensities:
        ir_intensities = [None] * len(frequencies)
    elif len(ir_intensities) < len(frequencies):
        ir_intensities.extend([None] * (len(frequencies) - len(ir_intensities)))

    # 4. [FR-NORM-COORD] section.
    norm_coord_pattern = r"\[FR-NORM-COORD\]\s*\n(.*?)(?:\n\[|\Z)"
    norm_coord_match = re.search(norm_coord_pattern, content, re.DOTALL)
    if not norm_coord_match:
        raise ValueError("No [FR-NORM-COORD] section found in Molden file")

    norm_coord_section = norm_coord_match.group(1)
    modes: list[VibrationalMode] = []

    vibration_blocks = re.split(r"vibration\s+(\d+)", norm_coord_section)
    # vibration_blocks = ['', '1', '\n...', '2', '\n...', ...]
    for i in range(1, len(vibration_blocks), 2):
        if i + 1 >= len(vibration_blocks):
            break
        mode_number = int(vibration_blocks[i])
        block_content = vibration_blocks[i + 1].strip()

        displacement_vectors: list[list[float]] = []
        for line in block_content.split("\n"):
            line = line.strip()
            if line:
                values = list(map(float, line.split()))
                if len(values) == 3:
                    displacement_vectors.append(values)

        if len(displacement_vectors) == n_atoms:
            disp_array = np.array(displacement_vectors)
            freq_idx = mode_number - 1
            if freq_idx < len(frequencies):
                freq = frequencies[freq_idx]
                ir_int = ir_intensities[freq_idx] if freq_idx < len(ir_intensities) else None
                modes.append(
                    VibrationalMode(
                        mode_number=mode_number,
                        frequency=freq,
                        ir_intensity=ir_int,
                        displacement_vectors=disp_array,
                        is_imaginary=(freq < 0),
                    )
                )

    return VibrationalData(
        coordinates=coords_arr,
        atomic_numbers=atomic_numbers,
        modes=modes,
        source_file=filepath,
        program="molden",
    )


# ---------------------------------------------------------------------------
# Auto-detect dispatcher
# ---------------------------------------------------------------------------


def parse_vibrations(filepath: str) -> VibrationalData:
    """Auto-detect the file format and parse vibrational data.

    Detection strategy:

    * ``.molden`` extension **or** ``[Molden Format]`` in the first 1 kB →
      :func:`parse_molden_vibrations`.
    * ``Gaussian`` in the first 500 bytes → :func:`parse_gaussian_vibrations`.
    * ``O   R   C   A`` in the first 500 bytes → :func:`parse_orca_vibrations`.

    Args:
        filepath: Path to the vibration file.

    Returns:
        :class:`VibrationalData` object.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the file format cannot be detected.
    """
    import os

    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath) as f:
        first_kb = f.read(1024)

    if ext == ".molden" or "[Molden Format]" in first_kb:
        return parse_molden_vibrations(filepath)
    elif "Gaussian" in first_kb[:500]:
        return parse_gaussian_vibrations(filepath)
    elif "O   R   C   A" in first_kb[:500]:
        return parse_orca_vibrations(filepath)
    else:
        raise ValueError(f"Cannot detect vibration file format: {filepath}")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def create_displacement_arrows(
    vib_data: VibrationalData,
    mode_number: int,
    amplitude: float = 1.0,
    arrow_scale: float = 1.0,
    color: str = "red",
    show_small_displacements: bool = False,
    displacement_threshold: float = 0.01,
) -> list[go.Cone]:
    """Create 3-D arrow (Cone) traces for vibrational displacements.

    Arrows are auto-scaled so that the largest displacement is ~15 % of the
    molecular extent, keeping them visible but not overwhelming.

    Args:
        vib_data: :class:`VibrationalData` object.
        mode_number: Which mode to visualise (1-based).
        amplitude: Displacement amplitude multiplier.
        arrow_scale: Visual scale factor applied to cone ``sizeref``.
        color: Arrow color (any Plotly-accepted string).
        show_small_displacements: When ``False``, hide arrows whose magnitude
            is below *displacement_threshold*.
        displacement_threshold: Minimum displacement magnitude to display.

    Returns:
        List containing a single :class:`plotly.graph_objects.Cone` trace
        (empty list if no atoms pass the threshold).

    Raises:
        ValueError: If the requested mode is not found in *vib_data*.
    """
    mode = vib_data.get_mode(mode_number)
    if mode is None:
        raise ValueError(f"Mode {mode_number} not found")

    coords = vib_data.coordinates

    # Molecular size as mean coordinate range (replaces removed np.ptp).
    coord_ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
    molecular_size = float(np.mean(coord_ranges))

    displacements = mode.displacement_vectors * amplitude
    magnitudes = np.linalg.norm(displacements, axis=1)

    max_magnitude = float(np.max(magnitudes)) if len(magnitudes) > 0 else 1.0
    if max_magnitude > 0:
        auto_scale = (0.15 * molecular_size) / max_magnitude
        displacements = displacements * auto_scale

    if not show_small_displacements:
        mask = magnitudes > displacement_threshold
        coords_filtered = coords[mask]
        displacements_filtered = displacements[mask]
    else:
        coords_filtered = coords
        displacements_filtered = displacements

    if len(coords_filtered) == 0:
        return []

    trace = go.Cone(
        x=coords_filtered[:, 0],
        y=coords_filtered[:, 1],
        z=coords_filtered[:, 2],
        u=displacements_filtered[:, 0],
        v=displacements_filtered[:, 1],
        w=displacements_filtered[:, 2],
        colorscale=[[0, color], [1, color]],
        sizemode="scaled",
        sizeref=arrow_scale * 0.3,
        showscale=False,
        name=f"Mode {mode_number} ({mode.frequency:.1f} cm\u207b\u00b9)",
        hovertemplate=(
            f"<b>Mode {mode_number}</b><br>"
            f"Frequency: {mode.frequency:.2f} cm\u207b\u00b9<br>"
            "Displacement: %{u:.3f}, %{v:.3f}, %{w:.3f}<br>"
            "<extra></extra>"
        ),
    )
    return [trace]


def build_molecule_figure(
    vib_data: VibrationalData,
    mode_number: Optional[int] = None,
    arrow_color: str = "red",
    arrow_scale: float = 1.0,
    amplitude: float = 1.0,
) -> go.Figure:
    """Build a Plotly figure showing atoms as a 3-D scatter plot.

    Atoms are represented as markers coloured and sized by element type
    (CPK colours, sizes proportional to van der Waals radii). No RDKit
    dependency is required.

    If *mode_number* is provided, displacement arrow cones are added on
    top of the atom scatter trace.

    Args:
        vib_data: :class:`VibrationalData` to visualise.
        mode_number: If given, add displacement arrows for this mode
            (1-based).
        arrow_color: Color for the displacement arrows.
        arrow_scale: Visual scale factor for arrow cones.
        amplitude: Displacement amplitude multiplier for arrows.

    Returns:
        :class:`plotly.graph_objects.Figure` ready for display.
    """
    coords = vib_data.coordinates
    atomic_nums = vib_data.atomic_numbers

    # Build per-atom visual properties.
    colors = [get_atom_color(z) for z in atomic_nums]
    # Scale vdW radius to a reasonable marker pixel size (multiply by ~15).
    sizes = [get_vdw_radius(z) * 15 for z in atomic_nums]
    labels = [get_atom_symbol(z) for z in atomic_nums]

    atom_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        marker=dict(
            color=colors,
            size=sizes,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=labels,
        textposition="top center",
        name="Atoms",
        hovertemplate=(
            "<b>%{text}</b><br>x: %{x:.3f} Å<br>y: %{y:.3f} Å<br>z: %{z:.3f} Å<br><extra></extra>"
        ),
    )

    fig = go.Figure(data=[atom_trace])

    # Optionally overlay displacement arrows.
    if mode_number is not None:
        arrow_traces = create_displacement_arrows(
            vib_data=vib_data,
            mode_number=mode_number,
            amplitude=amplitude,
            arrow_scale=arrow_scale,
            color=arrow_color,
        )
        for trace in arrow_traces:
            fig.add_trace(trace)

    # Clean 3-D layout with hidden axes.
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig
