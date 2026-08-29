from pathlib import Path
import runpy
import sys


def run_legacy(name, argv):
    root = Path(__file__).resolve().parents[2] / "src" / "pipelines" / "implementations"
    old = sys.argv
    sys.argv = [str(root / name)] + list(argv)
    try:
        return runpy.run_path(str(root / name), run_name="__main__")
    finally:
        sys.argv = old


def run_unconditional(argv):
    return run_legacy("legacy_2d_flow.py", argv)


def run_conditional(argv):
    return run_legacy("legacy_conditional_flow.py", argv)


def run_ot(argv):
    return run_legacy("legacy_ot_flow.py", argv)
