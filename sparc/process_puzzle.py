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

from sparc.prompt import generate_prompt
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
            env = gym.make("SPaRC-Gym", render_mode=None, traceback=False, observation = 'SPaRC', max_steps=100)
            options = {'puzzle_id': puzzle_id}
            obs, info = env.reset(options=options)
            Keep_Turns = 4
            reward = 0
            all_messages = []
            steps = 0
            messages=[{ "role": "system", "content": f"You are an expert at solving puzzles games. {generate_prompt(puzzle_data, step_by_step=True)}"}]
            for step in range(max_steps+1):
                steps += 1
                user_payload = json.dumps(make_json_safe({'obs':obs,'info':info,'reward':reward}))
                messages.append({"role":"user","content":user_payload})
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                
                reply = response.choices[0].message.content.strip()
                all_messages.append(reply)
                
                # Try to extract action, with retries if format is invalid
                action = None
                action_retries = 3
                for action_attempt in range(action_retries):
                    last_line = reply.splitlines()[-1].strip() if reply else ""
                    m = re.match(r"^(?:Final:\s*)?([0-3])$", last_line)
                    if m:
                        action = int(m.group(1))
                        break
                    else:
                        # Try to find any digit 0-3 in the last line as fallback
                        fallback = re.search(r"[0-3]", last_line)
                        if fallback:
                            action = int(fallback.group(0))
                            break
                    
                    if action_attempt < action_retries - 1:
                        # Retry: ask for a valid action format
                        console.print(f"[yellow]Invalid action format. Retrying...[/]")
                        messages.append({"role": "assistant", "content": reply})
                        messages.append({"role": "user", "content": "Invalid format. Please respond with only: Final: <digit> where <digit> is 0 (right), 1 (up), 2 (left), or 3 (down)."})
                        response = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                        )
                        reply = response.choices[0].message.content.strip()
                        all_messages.append(reply)
                
                if action is None:
                    raise ValueError(f"Failed to get valid action after {action_retries} attempts. Last response: {reply[:200]}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                messages.append({"role":"assistant","content": f"Final: {action}"})
                system = messages[0]
                tail = messages[-(Keep_Turns*2):]
                messages = [system] + tail
                
                if terminated or truncated:
                    break
                
            processing_time = time.time() - start_time
            if reward == 1:
                solved = True
            else:
                solved = False
                
            return {
                'puzzle_id': puzzle_id,
                'puzzle_data': puzzle_data,
                'solved': solved,
                'processing_time': processing_time,
                'message': all_messages,
                'observation': obs,
                'info': info,
                'reward': reward,
                'reached_end': terminated,  # Reached end location (doesn't mean solved)
                'no_legal_actions': truncated,  # Ran out of legal moves before reaching end
                'steps_taken': steps,
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
