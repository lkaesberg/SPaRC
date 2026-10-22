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


def test_prompts_state_the_rules_as_the_puzzles_use_them():
    """Default prompts use the corrected rule wording; the *_paper copies keep the original."""
    import inspect

    from sparc import prompts

    old_phrases = ("(star or square)", "exact same shape and color", "Cancel out regular polyshapes",
                   "cancel out regular polyshapes", "Rotation of polyshapes is generally allowed")
    for name in ("single_shot", "single_shot_visual", "gym_step", "gym_step_traceback", "gym_visual",
                 "gym_visual_traceback"):
        assert f"{name}_paper" in prompts.PROMPT_REGISTRY
        module = inspect.getsource(inspect.getmodule(prompts.PROMPT_REGISTRY[name]))
        assert not any(p in module for p in old_phrases), name
        paper = inspect.getsource(inspect.getmodule(prompts.PROMPT_REGISTRY[f"{name}_paper"]))
        assert any(p in paper for p in old_phrases), name
