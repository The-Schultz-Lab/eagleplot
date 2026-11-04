"""Repository-level checks: script permissions, line endings, setup-script guards.

Audit items A-08 (executable bits), A-36 (setup scripts) and A-42 (.gitattributes).
A-08 was once recorded as fixed in a commit message while the files stayed 100644, so
the mode is checked from git's index rather than trusted.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SHELL_SCRIPTS = ["setup.command", "launch.command", "setup.sh", "launch.sh"]
SETUP_SCRIPTS = ["setup.command", "setup.sh", "setup.bat"]


def _git_modes():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-s", *SHELL_SCRIPTS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.split("\t")[1]: line.split()[0] for line in out.splitlines()}


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_shell_script_is_executable_in_git(script):
    # Finder refuses to run a .command file without the executable bit, and
    # `./setup.sh` fails with "Permission denied". The bit must be in git's index,
    # since a fresh clone takes the mode from there.
    modes = _git_modes()
    assert script in modes, f"{script} is not tracked"
    assert modes[script] == "100755", f"{script} is {modes[script]} in git, not 100755"


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_shell_script_has_bash_shebang(script):
    assert (REPO / script).read_text(encoding="utf-8").startswith("#!/bin/bash\n")


@pytest.mark.parametrize("script", SETUP_SCRIPTS)
def test_setup_script_checks_python_version(script):
    # A too-old Python otherwise fails later with an obscure pip resolver error.
    text = (REPO / script).read_text(encoding="utf-8")
    assert "sys.version_info" in text
    assert "(3, 9)" in text and "(3, 9, 7)" in text


def test_gitattributes_pins_script_line_endings_and_binaries():
    rules = {
        tuple(line.split())
        for line in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for expected in [
        ("*.command", "text", "eol=lf"),
        ("*.sh", "text", "eol=lf"),
        ("*.bat", "text", "eol=crlf"),
        ("*.png", "binary"),
        ("*.npy", "binary"),
        ("*.ico", "binary"),
    ]:
        assert expected in rules, f"missing .gitattributes rule: {' '.join(expected)}"
