"""Contracts, registry and config services must not drag in torch or FastAPI."""

import subprocess
import sys

CHECK = """
import sys
import opendpd, opendpd.schemas, opendpd.core.registry, opendpd.services.config, opendpd.services.workspace
import opendpd.services.recipes, opendpd.commands
heavy = [m for m in ("torch", "fastapi", "uvicorn", "matplotlib", "pandas") if m in sys.modules]
assert not heavy, heavy
assert opendpd.__version__
print("light")
"""


def test_light_imports_do_not_load_heavy_modules():
    result = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "light"


def test_public_api_still_resolves_lazily():
    import opendpd

    assert callable(opendpd.train_pa) and callable(opendpd.create_dataset)
    assert "train_pa" in dir(opendpd)


def test_cli_help_is_light():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; from opendpd.cli import studio_main; sys.exit(studio_main(['--help']))"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "run" in result.stdout and "validate" in result.stdout
