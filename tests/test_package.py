"""The package exports its Inspect tasks lazily: `import sparc` must not need the inspect extra."""

import subprocess
import sys

import pytest


def test_import_does_not_load_inspect():
    code = "import sys, sparc, sparc.validation; print('inspect_ai' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_tasks_exported_from_package():
    pytest.importorskip("inspect_ai", reason="needs the inspect extra: pip install -e '.[inspect]'")
    import sparc
    from sparc import tasks

    assert sparc.__getattr__("sparc") is tasks.sparc
    assert sparc.sparc_hard is tasks.sparc_hard


def test_unknown_attribute():
    import sparc

    with pytest.raises(AttributeError):
        sparc.no_such_task
