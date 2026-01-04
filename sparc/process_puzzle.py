import time
import asyncio
from typing import Dict
from openai import AsyncOpenAI
from rich.console import Console

import gymnasium as gym
import SPaRC_Gym
import numpy as np
import json
import re

from sparc.prompt import generate_prompt, generate_prompt_step_by_step
from sparc.validation import extract_solution_path, validate_solution, analyze_path

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
                # Instead of exiting, we re-raise the exception so it can be handled by the batch processor
                raise e


async def process_puzzle_step_by_step(client: AsyncOpenAI, puzzle_data: Dict, model: str, temperature: float, puzzle_index: int) -> Dict:
    """(step-by-step) Process a single puzzle asynchronously with retry logic for connection errors"""
    start_time = time.time()
    puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            max_steps = 100
            keep_turns = 4
            
            env = gym.make("SPaRC-Gym", render_mode=None, traceback=False, observation='SPaRC', max_steps=max_steps)
            options = {'puzzle_id': puzzle_id}
            obs, info = env.reset(options=options)
            
            reward = 0
            all_messages = []
            all_actions = []
            extracted_path = []
            
            # Record starting position
            if 'agent_location' in info:
                loc = info['agent_location']
                extracted_path.append(tuple(loc) if isinstance(loc, list) else loc)
            
            messages = [{"role": "system", "content": generate_prompt_step_by_step()}]
            
            terminated = False
            truncated = False
            step = 0
            
            for step in range(max_steps + 1):
                user_payload = json.dumps(make_json_safe({'obs': obs, 'info': info, 'reward': reward}))
                messages.append({"role": "user", "content": user_payload})
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                
                reply = response.choices[0].message.content.strip()
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
                    extracted_path.append(tuple(loc) if isinstance(loc, list) else loc)
                
                # Append the action taken (not the raw reply) to maintain clean history
                messages.append({"role": "assistant", "content": f"Final: {action}"})
                
                # Keep only system message + last N turns
                system = messages[0]
                tail = messages[-(keep_turns * 2):]
                messages = [system] + tail
                
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
                'extracted_path': extracted_path,
                'observation': obs,
                'info': info,
                'reward': reward,
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
                # Instead of exiting, we re-raise the exception so it can be handled by the batch processor
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
