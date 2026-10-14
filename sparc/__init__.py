"""SPaRC: a spatial pathfinding reasoning benchmark.

The Inspect tasks `sparc` and `sparc_hard` live in `sparc.tasks` and need the optional Inspect extra
(pip install "sparc-puzzle[inspect]"). They load on first access, so `import sparc` stays cheap and
works without Inspect installed.
"""

__all__ = ["sparc", "sparc_hard"]


def __getattr__(name: str):
    if name in __all__:
        from . import tasks

        return getattr(tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
