import time
import asyncio
from typing import Dict, Optional
from openai import AsyncOpenAI
from rich.console import Console

import gymnasium as gym
import SPaRC_Gym
import numpy as np
import json
import re
import traceback

from sparc.prompt import generate_prompt, generate_prompt_step_by_step, generate_prompt_step_by_step_traceback, generate_prompt_step_by_step_visual, generate_prompt_step_by_step_visual_traceback
from sparc.validation import extract_solution_path, validate_solution, analyze_path

from sparc_visualization.plot import get_puzzle_image
from sparc_visualization.prompt import generate_prompt as generate_visual_prompt

console = Console()

async def process_puzzle(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int) -> Dict:
    """Process a single puzzle asynchronously with retry logic for connection errors"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at solving puzzles games.",
                    },
                    {"role": "user", "content": generate_prompt(puzzle_data)},
                ],
                temperature=temperature,
            )
            
            message = response.choices[0].message.content
            if message is None:
                raise ValueError("API returned empty response (content is None)")
            extracted_path = extract_solution_path(message, puzzle_data)
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


async def process_puzzle_visual(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, plot_type: str = "path_cell_annotated", prompt_type: str = "prompt_engineering") -> Dict:
    """Process a single puzzle asynchronously using visual representation"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            # Generate visual representation
            b64_image = get_puzzle_image(puzzle_data, plot_type=plot_type, base_64_image=True)
            text_prompt = generate_visual_prompt(puzzle_data, plot_type, prompt_type)
            
            # Create message with image
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at solving visual puzzles.",
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
                            "text": text_prompt
                        }
                    ]
                }
            ]
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            
            message = response.choices[0].message.content
            if message is None:
                raise ValueError("API returned empty response (content is None)")
            extracted_path = extract_solution_path(message, puzzle_data)
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
                'prompt_type': prompt_type,
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


async def process_puzzle_step_by_step(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, gym_traceback: bool = False) -> Dict:
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
            
            # Record starting position (gym returns (y, x), we store as (x, y))
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append((loc[1], loc[0]))
            
            # Use traceback prompt if traceback is enabled
            if gym_traceback:
                prompt_content = generate_prompt_step_by_step_traceback(puzzle_data)
            else:
                prompt_content = generate_prompt_step_by_step(puzzle_data)
            system_message = {"role": "system", "content": prompt_content}
            
            terminated = False
            truncated = False
            step = 0
            
            for step in range(max_steps + 1):
                user_payload = json.dumps(make_json_safe({'obs': obs, 'info': info, 'reward': reward}))
                messages = [
                    system_message,
                    {"role": "user", "content": user_payload}
                ]
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("API returned empty response (content is None)")
                reply = content.strip()
                all_messages.append(reply)
                
                # Parse action from last line
                last_line = reply.splitlines()[-1].strip() if reply else ""
                m = re.match(r"^(?:Final:\s*)?([0-3])$", last_line)
                
                if not m:
                    raise ValueError(f"Invalid model output, no 'Final: <0-3>' found. Last line: {last_line}")
                
                action = int(m.group(1))
                all_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record new position after the step
                if 'agent_location' in info:
                    loc = info['agent_location']
                    extracted_path.append((loc[1], loc[0]))
                
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

async def process_puzzle_step_by_step_visual(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int, gym_traceback: bool = False, plot_type: str = "path_cell_annotated", prompt_type: str = "prompt_engineering") -> Dict:
    """(step-by-step visual) Process a single puzzle using visual representations at each step"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            max_steps = 100
            
            env = gym.make("SPaRC-Gym", render_mode="human", traceback=gym_traceback, observation='SPaRC', max_steps=max_steps)
            options = {'puzzle_id': puzzle_id}
            obs, info = env.reset(options=options)
            
            reward = 0
            all_messages = []
            all_actions = []
            extracted_path = []
            
            # Record starting position (gym returns (y, x), we store as (x, y))
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append((loc[1], loc[0]))
            
            # Generate visual prompt for step-by-step mode
            if gym_traceback:
                visual_prompt = generate_prompt_step_by_step_visual_traceback(puzzle_data)
            else:
                visual_prompt = generate_prompt_step_by_step_visual(puzzle_data)
            
            system_message = {"role": "system", "content": visual_prompt}
            
            terminated = False
            truncated = False
            step = 0
            
            for step in range(max_steps + 1):
                # Generate current state image with path traced so far
                # Note: agent_location from gym is (y, x), so we swap to (x, y) for the image
                current_path = [{"x": p[0], "y": p[1]} for p in extracted_path]
                b64_image = get_puzzle_image(puzzle_data, plot_type=plot_type, base_64_image=True, path=current_path if current_path else None)
                
                # Create simple observation text for visual mode
                agent_loc = info.get('agent_location', 'unknown')
                # Convert from (y, x) to (x, y) for display
                agent_loc = (agent_loc[1], agent_loc[0])

                legal_actions = info.get('legal_actions', [])
                action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
                legal_str = ', '.join([f"{a}={action_names.get(a, '?')}" for a in legal_actions])
                obs_text = f"Step {step + 1} | Position: {agent_loc} | Legal moves: [{legal_str}]"
                
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
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("API returned empty response (content is None)")
                reply = content.strip()
                all_messages.append(reply)
                
                # Parse action from last line
                last_line = reply.splitlines()[-1].strip() if reply else ""
                m = re.match(r"^(?:Final:\s*)?([0-3])$", last_line)
                
                if not m:
                    raise ValueError(f"Invalid model output, no 'Final: <0-3>' found. Last line: {last_line}")
                
                action = int(m.group(1))
                all_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record new position after the step
                if 'agent_location' in info:
                    loc = info['agent_location']
                    extracted_path.append((loc[1], loc[0]))
                
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
                'prompt_type': prompt_type,
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


def make_json_safe(obj, seen=None):
    """Convert numpy arrays and other non-JSON-serializable objects to JSON-safe types."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        # Convert both keys and values to JSON-safe types
        return {_make_key_safe(k): make_json_safe(v, seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v, seen) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
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
