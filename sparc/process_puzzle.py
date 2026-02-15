import time
import asyncio
import heapq
import random
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from rich.console import Console

import gymnasium as gym
import SPaRC_Gym
import numpy as np
import json
import re
import traceback

from sparc.prompt import generate_prompt, generate_prompt_step_by_step, generate_prompt_step_by_step_traceback, generate_prompt_step_by_step_visual, generate_prompt_step_by_step_visual_traceback, get_prompt, AVAILABLE_PROMPTS
from sparc.validation import extract_solution_path, validate_solution, analyze_path

from sparc_visualization.plot import get_puzzle_image

console = Console()


def _get_content_and_reasoning(message) -> Tuple[Optional[str], Optional[str]]:
    """Extract content and optional reasoning from a chat completion message.
    Supports APIs (e.g. gpt-oss) that return reasoning in reasoning_content or reasoning."""
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    return (content, reasoning if reasoning else None)

async def process_puzzle(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, prompt_name: str = "single_shot", extra_body: Optional[Dict] = None) -> Dict:
    """Process a single puzzle asynchronously with retry logic for connection errors"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            # Get the prompt dict with system and user messages
            prompt_dict = get_prompt(prompt_name, puzzle_data)
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt_dict["system"],
                    },
                    {"role": "user", "content": prompt_dict["user"]},
                ],
                temperature=temperature,
                **(extra_body or {}),
            )
            
            msg = response.choices[0].message
            content, reasoning = _get_content_and_reasoning(msg)
            if content is None:
                raise ValueError("API returned empty response (content is None)")
            message = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            extracted_path = extract_solution_path(content, puzzle_data)
            solved = validate_solution(extracted_path, puzzle_data)
            analysis = analyze_path(extracted_path, puzzle_data)
            
            processing_time = time.time() - start_time
            
            return {
                'puzzle_id': puzzle_id,
                'puzzle_data': puzzle_data,
                'extracted_path': extracted_path,
                'solved': solved,
                'analysis': analysis,
                'processing_time': processing_time,
                'message': message,
                'error': None
            }
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                console.print(f"[yellow]⚠️  Connection error on puzzle {puzzle_id} (attempt {attempt + 1}/{max_retries + 1}): {str(e)}[/]")
                console.print(f"[yellow]🔄 Retrying in {wait_time} seconds...[/]")
                await asyncio.sleep(wait_time)
                continue
            else:
                console.print(f"[red]❌ ERROR on puzzle {puzzle_id} after {max_retries} retries: {str(e)}[/]")
                traceback.print_exc()
                # Instead of exiting, we re-raise the exception so it can be handled by the batch processor
                raise e


async def process_puzzle_visual(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, plot_type: str = "path_cell_annotated", prompt_name: str = "single_shot_visual", extra_body: Optional[Dict] = None) -> Dict:
    """Process a single puzzle asynchronously using visual representation"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            # Generate visual representation
            b64_image = get_puzzle_image(puzzle_data, plot_type=plot_type, base_64_image=True)
            
            # Get the prompt dict with system and user messages
            prompt_dict = get_prompt(prompt_name, puzzle_data)
            
            # Create message with image
            messages = [
                {
                    "role": "system",
                    "content": prompt_dict["system"],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_dict["user"]
                        }
                    ]
                }
            ]
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **(extra_body or {}),
            )
            
            msg = response.choices[0].message
            content, reasoning = _get_content_and_reasoning(msg)
            if content is None:
                raise ValueError("API returned empty response (content is None)")
            message = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            extracted_path = extract_solution_path(content, puzzle_data)
            solved = validate_solution(extracted_path, puzzle_data)
            analysis = analyze_path(extracted_path, puzzle_data)
            
            processing_time = time.time() - start_time
            
            return {
                'puzzle_id': puzzle_id,
                'puzzle_data': puzzle_data,
                'extracted_path': extracted_path,
                'solved': solved,
                'analysis': analysis,
                'processing_time': processing_time,
                'message': message,
                'visual_mode': True,
                'plot_type': plot_type,
                'error': None
            }
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                console.print(f"[yellow]⚠️  Connection error on puzzle {puzzle_id} (attempt {attempt + 1}/{max_retries + 1}): {str(e)}[/]")
                console.print(f"[yellow]🔄 Retrying in {wait_time} seconds...[/]")
                await asyncio.sleep(wait_time)
                continue
            else:
                console.print(f"[red]❌ ERROR on puzzle {puzzle_id} after {max_retries} retries: {str(e)}[/]")
                traceback.print_exc()
                raise e


async def process_puzzle_step_by_step(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, gym_traceback: bool = False, prompt_name: str = None, extra_body: Optional[Dict] = None) -> Dict:
    """(step-by-step) Process a single puzzle asynchronously with retry logic for connection errors"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            max_steps = 100
            
            env = gym.make("SPaRC-Gym", render_mode=None, traceback=gym_traceback, observation='SPaRC', max_steps=max_steps)
            options = {'puzzle_id': puzzle_id}
            obs, info = env.reset(options=options)
            
            reward = 0
            all_messages = []
            all_actions = []
            extracted_path = []
            
            # Record starting position (gym now returns (x, y))
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append((loc[0], loc[1]))
            
            # Use provided prompt_name or auto-select based on traceback
            if prompt_name is None:
                prompt_name = "gym_step_traceback" if gym_traceback else "gym_step"
            
            # Get the prompt dict with system message
            prompt_dict = get_prompt(prompt_name, puzzle_data)
            system_message = {"role": "system", "content": prompt_dict["system"]}
            
            terminated = False
            truncated = False
            step = 0
            
            for step in range(max_steps + 1):
                # Build clean observation for LLM (only inference-relevant info)
                agent_loc = info.get('agent_location', 'unknown')
                legal_actions = info.get('legal_actions', [])
                grid_size = (info.get('grid_x_size', 0), info.get('grid_y_size', 0))
                
                action_names = {0: 'RIGHT', 1: 'UP', 2: 'LEFT', 3: 'DOWN'}
                legal_str = ', '.join([f"{a}={action_names.get(a, '?')}" for a in legal_actions])
                
                # Convert observation grid to clean string representation
                # obs is a JSON string of a 2D array
                obs_grid = json.loads(obs)
                obs_str = '\n'.join([str(row) for row in obs_grid])
                
                user_payload = f"""Step: {step + 1}
Current Position: {agent_loc}
Legal Actions: [{legal_str}]

Grid State:
{obs_str}

You MAY think step-by-step, but you MUST end your response with:
Final: <digit>
Where <digit> is exactly one of 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN."""
                
                messages = [
                    system_message,
                    {"role": "user", "content": user_payload}
                ]
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **(extra_body or {}),
                )
                
                msg = response.choices[0].message
                content, step_reasoning = _get_content_and_reasoning(msg)
                if content is None:
                    raise ValueError("API returned empty response (content is None)")
                reply = content.strip()
                # Store full output: reasoning in <think>...</think>, then content (for gpt-oss etc.)
                step_message = f"<think>{step_reasoning}</think>\n{reply}" if step_reasoning else reply
                all_messages.append(step_message)
                reply = reply.split("</think>")[-1].strip()  # Remove any preceding think tags
                
                # Parse action - search for "Final: <digit>" anywhere in response
                m = re.search(r'Final:\s*([0-3])', reply, re.IGNORECASE)
                
                if not m:
                    # Fallback: look for a standalone digit 0-3 at the end
                    m_fallback = re.search(r'\b([0-3])\s*$', reply)
                    if m_fallback:
                        action = int(m_fallback.group(1))
                    else:
                        raise ValueError(f"Invalid model output, no 'Final: <0-3>' found in response: {reply}")
                else:
                    action = int(m.group(1))
                
                all_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record new position after the step (gym now returns (x, y))
                if 'agent_location' in info:
                    loc = info['agent_location']
                    extracted_path.append((loc[0], loc[1]))
                
                if terminated or truncated:
                    break
            
            processing_time = time.time() - start_time
            solved = reward == 1
            
            return {
                'puzzle_id': puzzle_id,
                'puzzle_data': puzzle_data,
                'solved': solved,
                'processing_time': processing_time,
                'message': all_messages,
                'actions': all_actions,
                'extracted_path': make_json_safe(extracted_path),
                'observation': make_json_safe(obs),
                'info': make_json_safe(info),
                'reward': make_json_safe(reward),
                'reached_end': terminated,
                'no_legal_actions': truncated,
                'steps_taken': step + 1,
                'error': None
            }
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                console.print(f"[yellow]⚠️  Connection error on puzzle {puzzle_id} (attempt {attempt + 1}/{max_retries + 1}): {str(e)}[/]")
                console.print(f"[yellow]🔄 Retrying in {wait_time} seconds...[/]")
                await asyncio.sleep(wait_time)
                continue
            else:
                console.print(f"[red]❌ ERROR on puzzle {puzzle_id} after {max_retries} retries: {str(e)}[/]")
                traceback.print_exc()
                # Instead of exiting, we re-raise the exception so it can be handled by the batch processor
                raise e

async def process_puzzle_step_by_step_visual(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, gym_traceback: bool = False, plot_type: str = "path_cell_annotated", prompt_name: str = None, extra_body: Optional[Dict] = None) -> Dict:
    """(step-by-step visual) Process a single puzzle using visual representations at each step"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            max_steps = 100
            
            env = gym.make("SPaRC-Gym", render_mode=None, traceback=gym_traceback, observation='SPaRC', max_steps=max_steps)
            options = {'puzzle_id': puzzle_id}
            obs, info = env.reset(options=options)
            
            reward = 0
            all_messages = []
            all_actions = []
            extracted_path = []
            
            # Record starting position (gym now returns (x, y))
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append((loc[0], loc[1]))
            
            # Use provided prompt_name or auto-select based on traceback
            if prompt_name is None:
                prompt_name = "gym_visual_traceback" if gym_traceback else "gym_visual"
            
            # Get the prompt dict with system message
            prompt_dict = get_prompt(prompt_name, puzzle_data)
            system_message = {"role": "system", "content": prompt_dict["system"]}
            
            terminated = False
            truncated = False
            step = 0
            
            for step in range(max_steps + 1):
                # Generate current state image with path traced so far
                current_path = [{"x": p[0], "y": p[1]} for p in extracted_path]
                b64_image = get_puzzle_image(puzzle_data, plot_type=plot_type, base_64_image=True, path=current_path if current_path else None, save_to_disk=True)
                
                # Create simple observation text for visual mode (gym now returns (x, y))
                agent_loc = info.get('agent_location', 'unknown')

                legal_actions = info.get('legal_actions', [])
                action_names = {0: 'RIGHT', 1: 'UP', 2: 'LEFT', 3: 'DOWN'}
                legal_str = ', '.join([f"{a}={action_names.get(a, '?')}" for a in legal_actions])
                obs_text = f"""Step {step + 1} | Position: {agent_loc} | Legal moves: [{legal_str}]

You MAY think step-by-step, but you MUST end your response with:
Final: <digit>
Where <digit> is exactly one of 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN."""
                
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": obs_text
                            }
                        ]
                    }
                ]
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **(extra_body or {}),
                )
                
                msg = response.choices[0].message
                content, step_reasoning = _get_content_and_reasoning(msg)
                if content is None:
                    raise ValueError("API returned empty response (content is None)")
                reply = content.strip()
                step_message = f"<think>{step_reasoning}</think>\n{reply}" if step_reasoning else reply
                all_messages.append(step_message)
                reply = reply.split("</think>")[-1].strip()  # Remove any preceding think tags

                
                # Parse action - search for "Final: <digit>" anywhere in response
                m = re.search(r'Final:\s*([0-3])', reply, re.IGNORECASE)
                
                if not m:
                    # Fallback: look for a standalone digit 0-3 at the end
                    m_fallback = re.search(r'\b([0-3])\s*$', reply)
                    if m_fallback:
                        action = int(m_fallback.group(1))
                    else:
                        raise ValueError(f"Invalid model output, no 'Final: <0-3>' found in response: {reply}")
                else:
                    action = int(m.group(1))
                
                all_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record new position after the step (gym now returns (x, y))
                if 'agent_location' in info:
                    loc = info['agent_location']
                    extracted_path.append((loc[0], loc[1]))
                
                if terminated or truncated:
                    break
            
            processing_time = time.time() - start_time
            solved = reward == 1
            
            return {
                'puzzle_id': puzzle_id,
                'puzzle_data': puzzle_data,
                'solved': solved,
                'processing_time': processing_time,
                'message': all_messages,
                'actions': all_actions,
                'extracted_path': make_json_safe(extracted_path),
                'observation': make_json_safe(obs),
                'info': make_json_safe(info),
                'reward': make_json_safe(reward),
                'reached_end': terminated,
                'no_legal_actions': truncated,
                'steps_taken': step + 1,
                'visual_mode': True,
                'plot_type': plot_type,
                'error': None
            }
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                console.print(f"[yellow]⚠️  Connection error on puzzle {puzzle_id} (attempt {attempt + 1}/{max_retries + 1}): {str(e)}[/]")
                console.print(f"[yellow]🔄 Retrying in {wait_time} seconds...[/]")
                await asyncio.sleep(wait_time)
                continue
            else:
                console.print(f"[red]❌ ERROR on puzzle {puzzle_id} after {max_retries} retries: {str(e)}[/]")
                traceback.print_exc()
                raise e


async def process_puzzle_step_by_step_random(puzzle_data: Dict, puzzle_index: int, gym_traceback: bool = False) -> Dict:
    """(step-by-step random ablation) Process a single puzzle using random action selection.
    
    This is a baseline ablation that replaces the LLM with uniform random
    selection from legal actions at each step. No API calls are made.
    """
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    
    try:
        max_steps = 100
        
        env = gym.make("SPaRC-Gym", render_mode=None, traceback=gym_traceback, observation='SPaRC', max_steps=max_steps)
        options = {'puzzle_id': puzzle_id}
        obs, info = env.reset(options=options)
        
        reward = 0
        all_actions = []
        extracted_path = []
        
        # Record starting position
        if 'agent_location' in info:
            loc = info['agent_location']
            extracted_path.append((loc[0], loc[1]))
        
        terminated = False
        truncated = False
        step = 0
        
        for step in range(max_steps + 1):
            legal_actions = info.get('legal_actions', [])
            
            if not legal_actions:
                break
            
            # Random choice from legal actions instead of LLM
            action = random.choice(legal_actions)
            
            all_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record new position after the step
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append((loc[0], loc[1]))
            
            if terminated or truncated:
                break
        
        processing_time = time.time() - start_time
        solved = reward == 1
        
        return {
            'puzzle_id': puzzle_id,
            'puzzle_data': puzzle_data,
            'solved': solved,
            'processing_time': processing_time,
            'message': [],
            'actions': all_actions,
            'extracted_path': make_json_safe(extracted_path),
            'observation': make_json_safe(obs),
            'info': make_json_safe(info),
            'reward': make_json_safe(reward),
            'reached_end': terminated,
            'no_legal_actions': truncated,
            'steps_taken': step + 1,
            'error': None
        }
        
    except Exception as e:
        console.print(f"[red]❌ ERROR on puzzle {puzzle_id}: {str(e)}[/]")
        traceback.print_exc()
        raise e


def _astar_find_path(puzzle_data: Dict) -> Optional[List[Tuple[int, int]]]:
    """Find shortest path from S to E using A* on the puzzle grid.
    
    The grid uses the convention:
    - Path cells have at least one even coordinate.
    - Rule cells (both coordinates odd) are impassable.
    - Gaps ('G') are impassable.
    
    Returns:
        List of (x, y) tuples from start to end, or None if no path exists.
    """
    puzzle_array = puzzle_data.get("puzzle_array", [])
    if not puzzle_array:
        return None

    height = len(puzzle_array)
    width = len(puzzle_array[0]) if height > 0 else 0

    # Find start and end positions
    start = None
    end = None
    for y in range(height):
        for x in range(width):
            cell = puzzle_array[y][x]
            if cell == "S":
                start = (x, y)
            elif cell == "E":
                end = (x, y)

    if not start or not end:
        return None

    # Build set of traversable cells
    valid_cells = set()
    for y in range(height):
        for x in range(width):
            cell = puzzle_array[y][x]
            # Gaps are impassable
            if cell == "G":
                continue
            # Rule cells (both coordinates odd) are impassable
            if x % 2 == 1 and y % 2 == 1:
                continue
            valid_cells.add((x, y))

    # A* search — Manhattan distance heuristic
    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    # Directions matching gym actions: 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    open_set: list = [(heuristic(start), 0, start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], int] = {start: 0}

    while open_set:
        _f, g, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in valid_cells:
                continue
            new_g = g + 1
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_g + heuristic(neighbor), new_g, neighbor))

    return None  # No path found


def _path_to_actions(path: List[Tuple[int, int]]) -> List[int]:
    """Convert a sequence of (x, y) positions into gym action indices.
    
    Action mapping: 0=RIGHT (+x), 1=UP (-y), 2=LEFT (-x), 3=DOWN (+y)
    """
    delta_to_action = {
        (1, 0): 0,   # RIGHT
        (0, -1): 1,  # UP
        (-1, 0): 2,  # LEFT
        (0, 1): 3,   # DOWN
    }
    actions = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        action = delta_to_action.get((dx, dy))
        if action is None:
            raise ValueError(f"Non-adjacent step in path: {path[i-1]} -> {path[i]}")
        actions.append(action)
    return actions


async def process_puzzle_step_by_step_astar(puzzle_data: Dict, puzzle_index: int, gym_traceback: bool = False) -> Dict:
    """(step-by-step A* ablation) Process a single puzzle using A* pathfinding.
    
    Pre-computes the shortest valid path from S to E on the puzzle grid
    (avoiding gaps and rule cells) and executes it through the gym.
    No API calls are made.
    """
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")

    try:
        max_steps = 100

        # Pre-compute A* path on the puzzle grid
        astar_path = _astar_find_path(puzzle_data)

        env = gym.make("SPaRC-Gym", render_mode=None, traceback=gym_traceback, observation='SPaRC', max_steps=max_steps)
        options = {'puzzle_id': puzzle_id}
        obs, info = env.reset(options=options)

        reward = 0
        all_actions = []
        extracted_path = []

        # Record starting position
        if 'agent_location' in info:
            loc = info['agent_location']
            extracted_path.append((loc[0], loc[1]))

        terminated = False
        truncated = False
        step = 0

        if astar_path is not None:
            planned_actions = _path_to_actions(astar_path)
        else:
            planned_actions = []

        for step in range(max_steps + 1):
            legal_actions = info.get('legal_actions', [])

            if not legal_actions:
                break

            if step < len(planned_actions) and planned_actions[step] in legal_actions:
                # Follow the A* plan
                action = planned_actions[step]
            else:
                # Fallback: random choice if A* plan is exhausted or action rejected
                action = random.choice(legal_actions)

            all_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)

            # Record new position after the step
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append((loc[0], loc[1]))

            if terminated or truncated:
                break

        processing_time = time.time() - start_time
        solved = reward == 1

        return {
            'puzzle_id': puzzle_id,
            'puzzle_data': puzzle_data,
            'solved': solved,
            'processing_time': processing_time,
            'message': [],
            'actions': all_actions,
            'extracted_path': make_json_safe(extracted_path),
            'observation': make_json_safe(obs),
            'info': make_json_safe(info),
            'reward': make_json_safe(reward),
            'reached_end': terminated,
            'no_legal_actions': truncated,
            'steps_taken': step + 1,
            'error': None
        }

    except Exception as e:
        console.print(f"[red]❌ ERROR on puzzle {puzzle_id}: {str(e)}[/]")
        traceback.print_exc()
        raise e


def make_json_safe(obj, seen=None):
    if seen is None:
        seen = set()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            if isinstance(k, np.generic):
                k = k.item()
            elif not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            safe[k] = make_json_safe(v, seen)
        return safe
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v, seen) for v in obj]
    return str(obj)


def _make_key_safe(key):
    """Convert a dictionary key to a JSON-safe type."""
    if isinstance(key, np.integer):
        return int(key)
    if isinstance(key, np.floating):
        return float(key)
    if isinstance(key, np.bool_):
        return bool(key)
    if isinstance(key, (int, float, str, bool)) or key is None:
        return key
    return str(key)
