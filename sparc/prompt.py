"""
Backward-compatible prompt module.
This module re-exports the prompt generation functions for backward compatibility.
The actual prompts are now in the sparc/prompts/ directory.
"""

from typing import Dict

from sparc.prompts import (
    get_prompt,
    auto_detect_prompt,
    AVAILABLE_PROMPTS,
    PROMPT_REGISTRY,
)
from sparc.prompts.single_shot import get_prompt as _get_single_shot
from sparc.prompts.single_shot_visual import get_prompt as _get_single_shot_visual
from sparc.prompts.gym_step import get_prompt as _get_gym_step
from sparc.prompts.gym_step_traceback import get_prompt as _get_gym_step_traceback
from sparc.prompts.gym_visual import get_prompt as _get_gym_visual
from sparc.prompts.gym_visual_traceback import get_prompt as _get_gym_visual_traceback


# Backward-compatible functions that return just the content string
def generate_prompt(puzzle_data: Dict) -> str:
    """Generate single-shot prompt (backward compatible)."""
    prompt_dict = _get_single_shot(puzzle_data)
    return prompt_dict["user"]


def generate_prompt_visual(puzzle_data: Dict) -> str:
    """Generate single-shot visual prompt (backward compatible)."""
    prompt_dict = _get_single_shot_visual(puzzle_data)
    return prompt_dict["user"]


def generate_prompt_step_by_step(puzzle_data: Dict) -> str:
    """Generate gym step-by-step prompt without traceback (backward compatible)."""
    prompt_dict = _get_gym_step(puzzle_data)
    return prompt_dict["system"]


def generate_prompt_step_by_step_traceback(puzzle_data: Dict) -> str:
    """Generate gym step-by-step prompt with traceback (backward compatible)."""
    prompt_dict = _get_gym_step_traceback(puzzle_data)
    return prompt_dict["system"]


def generate_prompt_step_by_step_visual(puzzle_data: Dict) -> str:
    """Generate visual gym prompt without traceback (backward compatible)."""
    prompt_dict = _get_gym_visual(puzzle_data)
    return prompt_dict["system"]


def generate_prompt_step_by_step_visual_traceback(puzzle_data: Dict) -> str:
    """Generate visual gym prompt with traceback (backward compatible)."""
    prompt_dict = _get_gym_visual_traceback(puzzle_data)
    return prompt_dict["system"]


# Export everything
__all__ = [
    "generate_prompt",
    "generate_prompt_visual",
    "generate_prompt_step_by_step",
    "generate_prompt_step_by_step_traceback",
    "generate_prompt_step_by_step_visual",
    "generate_prompt_step_by_step_visual_traceback",
    "get_prompt",
    "auto_detect_prompt",
    "AVAILABLE_PROMPTS",
    "PROMPT_REGISTRY",
]
