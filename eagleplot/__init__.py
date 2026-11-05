"""Modules Package.

This package contains the core functionality for the EaglePlot toolkit.

Available Modules
-----------------
io : Data input/output operations
math : Mathematical and statistical functions
plotting : Visualization and plotting tools
gui : Interactive GUI components
"""

# Single source of truth for the version. `pyproject.toml` reads this via
# `[tool.setuptools.dynamic]`, and cli.py / Home.py / docs/conf.py import it, so
# the number cannot drift between the package, the CLI, and the GUI.
# See audit A-26.
__version__ = "3.0.0"

__all__ = ["io", "math", "plotting", "gui", "__version__"]
