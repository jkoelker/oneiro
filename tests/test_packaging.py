"""Built-package checks for the local backport and its licensing data."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_backport_is_in_built_wheel(tmp_path: Path) -> None:
    """An offline wheel must ship every local Krea module and its licensing data."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(
        ROOT / "src",
        project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    (project / "dist").mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=project,
        env={**os.environ, "PIP_NO_INDEX": "1", "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = list((project / "dist").glob("*.whl"))
    assert len(wheels) == 1
    prefix = "oneiro/pipelines/backports/krea2/"
    backport = ROOT / "src" / prefix
    with zipfile.ZipFile(wheels[0]) as wheel:
        for source in [*backport.glob("*.py"), backport / "LICENSE", backport / "PROVENANCE.md"]:
            assert wheel.read(prefix + source.name) == source.read_bytes()
