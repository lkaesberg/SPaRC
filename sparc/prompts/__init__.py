"""
Prompt modules for different SPaRC solving modes.

Available prompts:
- single_shot: One-shot puzzle solving (default for non-gym mode)
- gym_step: Step-by-step gym mode without traceback
- gym_step_traceback: Step-by-step gym mode with traceback enabled
- gym_visual: Visual gym mode without traceback
- gym_visual_traceback: Visual gym mode with traceback enabled
"""

from .single_shot import get_prompt as get_single_shot_prompt
from .single_shot_visual import get_prompt as get_single_shot_visual_prompt
from .gym_step import get_prompt as get_gym_step_prompt
from .gym_step_traceback import get_prompt as get_gym_step_traceback_prompt
from .gym_visual import get_prompt as get_gym_visual_prompt
from .gym_visual_traceback import get_prompt as get_gym_visual_traceback_prompt

# Map prompt names to their functions
PROMPT_REGISTRY = {
    "single_shot": get_single_shot_prompt,
    "single_shot_visual": get_single_shot_visual_prompt,
    "gym_step": get_gym_step_prompt,
    "gym_step_traceback": get_gym_step_traceback_prompt,
    "gym_visual": get_gym_visual_prompt,
    "gym_visual_traceback": get_gym_visual_traceback_prompt,
}

# All available prompt names
AVAILABLE_PROMPTS = list(PROMPT_REGISTRY.keys())


def get_prompt(prompt_name: str, puzzle_data: dict) -> dict:
    """
    Get the prompt dict (system + user messages) for the given prompt type.
    
    Args:
        prompt_name: Name of the prompt type (see AVAILABLE_PROMPTS)
        puzzle_data: The puzzle data dictionary
        
    Returns:
        Dict with 'system' and 'user' message content
    """
    if prompt_name not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt type: {prompt_name}. Available: {AVAILABLE_PROMPTS}")
    return PROMPT_REGISTRY[prompt_name](puzzle_data)


def auto_detect_prompt(gym: bool, visual: bool, traceback: bool) -> str:
    """
    Auto-detect the appropriate prompt based on mode flags.
    
    Args:
        gym: Whether gym mode is enabled
        visual: Whether visual mode is enabled
        traceback: Whether traceback is enabled
        
    Returns:
        The prompt name to use
    """
    if not gym:
        if visual:
            return "single_shot_visual"
        return "single_shot"
    elif visual:
        if traceback:
            return "gym_visual_traceback"
        else:
            return "gym_visual"
    else:
        if traceback:
            return "gym_step_traceback"
        else:
            return "gym_step"
