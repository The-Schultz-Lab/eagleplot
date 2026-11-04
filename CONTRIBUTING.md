# Contributing to Plottle

Thanks for helping improve Plottle. Bug reports, feature ideas, documentation fixes and
code are all welcome.

## Reporting bugs and requesting features

Open an [issue](https://github.com/The-Schultz-Lab/plottle/issues/new/choose) and pick the
bug or feature template. For a bug, the most useful things to include are:

- what you did (the page, button or command, and the data file type),
- what happened, including the full error message or traceback,
- what you expected, and
- your Plottle version (`plottle --version`), Python version and operating system.

**Security problems** should not go in a public issue. See [SECURITY.md](SECURITY.md).

## Setting up a development environment

Plottle supports Python 3.9 and later (except 3.9.7).

```bash
git clone https://github.com/The-Schultz-Lab/plottle.git
cd plottle
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,formats,nist]"
pre-commit install                 # optional: runs ruff on every commit
```

The `dev` extra pins `ruff` and `mypy` to the exact versions CI uses, so a local run
matches CI. Its lint tools need Python 3.10 or later. On 3.9, install `.[test,formats,nist]`
to run the tests and run the linters from a newer interpreter.

## Before opening a pull request

Run the same checks CI runs:

```bash
ruff check .
ruff format --check .        # or `ruff format .` to fix
MPLBACKEND=Agg pytest tests/
```

`MPLBACKEND=Agg` stops matplotlib from trying to open windows.

- **Add or update tests** for any behaviour you change. GUI pages can be tested with
  `streamlit.testing.v1.AppTest`; see `tests/test_gui_pages.py` and
  `tests/test_gui_safety.py` for examples.
- **Keep Python 3.9 working.** Avoid `match`, `X | Y` type unions evaluated at runtime,
  and standard-library modules newer than 3.9 (check the "New in version" note).
- **Follow the conventions of the module you are editing**, and give public functions a
  NumPy-style docstring with Parameters and Returns sections.
- **Update `CHANGELOG.md`** under `## [Unreleased]` for anything a user would notice.

## Pull requests

- Branch from `main`, keep each PR to one topic, and describe *why* as well as *what*.
- CI runs lint, type checks and the test suite on Linux, macOS and Windows across
  Python 3.9 to 3.12. A PR needs all checks green before it is merged.
- Formatting-only commits can be listed in `.git-blame-ignore-revs` so they don't clutter
  `git blame`.

## Code of conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
