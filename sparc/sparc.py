import argparse
import asyncio
import json
import os
import signal
import time
from typing import Dict, List, Set
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich import box
from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path
from sparc.tables import create_statistics_table, create_detailed_results_table
from datasets import load_dataset
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
import aiohttp

console = Console()

# Global variable for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    shutdown_requested = True
    console.print("\n[yellow]⚠️  Graceful shutdown requested. Will finish current batch and save results...[/]")

def format_puzzle_info(puzzle_data: Dict) -> str:
    """Format puzzle information for display"""
    grid_size = puzzle_data.get("grid_size", {"width": 0, "height": 0})
    puzzle_id = puzzle_data.get("id", "unknown")
    puzzle_difficulty = puzzle_data.get("difficulty_level", "unknown")
    return f"Puzzle {puzzle_id} | Size: {grid_size['width']}x{grid_size['height']} | Difficulty: {puzzle_difficulty}"


def save_results(results: List[Dict], filename: str) -> None:
    """Save results to a JSON file"""
    try:
        # Convert results to a serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'puzzle_id': result['puzzle_id'],
                'solved': result['solved'],
                'analysis': result['analysis'],
                'processing_time': result['processing_time'],
                'extracted_path': result['extracted_path'],
                'error': result.get('error'),
                'puzzle_data': result['puzzle_data']
            }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump({
                'results': serializable_results,
                'total_processed': len(results),
                'timestamp': time.time()
            }, f, indent=2)
        
        console.print(f"[green]💾 Results saved to {filename}[/]")
    except Exception as e:
        console.print(f"[red]❌ Failed to save results: {str(e)}[/]")


def load_results(filename: str) -> tuple[List[Dict], Set[str]]:
    """Load results from a JSON file and return results and processed puzzle IDs"""
    if not os.path.exists(filename):
        return [], set()
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        processed_ids = {result['puzzle_id'] for result in results}
        
        console.print(f"[green]📂 Loaded {len(results)} previous results from {filename}[/]")
        return results, processed_ids
    
    except Exception as e:
        console.print(f"[red]❌ Failed to load results: {str(e)}[/]")
        return [], set()


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
                temperature=1,
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
                exit(1)



async def process_batch(client: AsyncOpenAI, batch_puzzles: List[tuple], model: str, temperature: float, verbose: bool) -> List[Dict]:
    """Process a batch of puzzles concurrently"""
    tasks = []
    for puzzle_data, puzzle_index in batch_puzzles:
        task = process_puzzle(client, puzzle_data, model, temperature, puzzle_index)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            puzzle_data, puzzle_index = batch_puzzles[i]
            puzzle_id = puzzle_data.get("id", f"idx_{puzzle_index}")
            if verbose:
                console.print(f"[red]❌ ERROR on puzzle {puzzle_id}: {str(result)}[/]")
            processed_results.append({
                'puzzle_id': puzzle_id,
                'puzzle_data': puzzle_data,
                'extracted_path': None,
                'solved': False,
                'analysis': {'fully_valid_path': False},
                'processing_time': 0.0,
                'error': str(result)
            })
        else:
            processed_results.append(result)
            
            if verbose and result:
                puzzle_id = result['puzzle_id']
                solved = result['solved']
                status_style = "green" if solved else "red"
                status = "✅ SOLVED" if solved else "❌ FAILED"
                puzzle_info = format_puzzle_info(result['puzzle_data'])
                path_len = len(result['extracted_path']) if result['extracted_path'] else 0
                
                console.print(f"[{status_style}]{status}[/] {puzzle_info} | Path: {path_len} steps | Time: {result['processing_time']:.2f}s")
                
                if solved and result['extracted_path']:
                    path_preview = result['extracted_path'][:3] + ["..."] + result['extracted_path'][-3:] if len(result['extracted_path']) > 6 else result['extracted_path']
                    console.print(f"   [dim]Path: {path_preview}[/]")
                
                if not result['analysis']['fully_valid_path']:
                    issues = []
                    if not result['analysis']['starts_at_start_ends_at_exit']:
                        issues.append("start/end")
                    if not result['analysis']['connected_line']:
                        issues.append("disconnected")
                    if not result['analysis']['non_intersecting_line']:
                        issues.append("intersecting")
                    if not result['analysis']['no_rule_crossing']:
                        issues.append("rule violations")
                    console.print(f"   [red]Issues: {', '.join(issues)}[/]")
                console.print()
    
    return processed_results


async def process_dataset_async(dataset, client: AsyncOpenAI, model: str, temperature: float, batch_size: int, verbose: bool, results_file: str, skip_processed: Set[str]) -> List[Dict]:
    """Process the entire dataset in batches with graceful shutdown support"""
    global shutdown_requested
    
    total_puzzles = len(dataset)
    all_results = []
    
    # Load existing results if any
    existing_results, _ = load_results(results_file)
    all_results.extend(existing_results)
    
    # Count remaining puzzles to process
    remaining_puzzles = [i for i in range(total_puzzles) if dataset[i].get("id", f"idx_{i}") not in skip_processed]
    total_remaining = len(remaining_puzzles)
    
    if total_remaining == 0:
        console.print("[green]✅ All puzzles already processed![/]")
        return all_results
    
    console.print(f"[cyan]🔄 Resuming processing: {len(existing_results)} already done, {total_remaining} remaining[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=not verbose
    ) as progress:
        
        task = progress.add_task("[cyan]Processing puzzles...", total=total_remaining)
        
        # Process remaining puzzles in batches
        for batch_start in range(0, total_remaining, batch_size):
            if shutdown_requested:
                console.print("[yellow]🛑 Shutdown requested, stopping after current batch...[/]")
                break
                
            batch_end = min(batch_start + batch_size, total_remaining)
            batch_indices = remaining_puzzles[batch_start:batch_end]
            batch_puzzles = [(dataset[i], i) for i in batch_indices]
            
            current_batch_ids = [puzzle_data.get("id", f"idx_{i}") for puzzle_data, i in batch_puzzles]
            progress.update(task, description=f"[cyan]Processing batch: {', '.join(current_batch_ids[:3])}{'...' if len(current_batch_ids) > 3 else ''}")
            
            batch_results = await process_batch(client, batch_puzzles, model, temperature, verbose)
            all_results.extend(batch_results)
            
            # Save intermediate results after each batch
            save_results(all_results, results_file)
            
            progress.update(task, advance=len(batch_puzzles))
            
            if shutdown_requested:
                break
    
    return all_results


def main() -> None:
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="SPaRC: A Spatial Pathfinding and Reasoning Challenge for grounding language models in spatial cognition")
    parser.add_argument(
        "--api-key", 
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base-url", 
        default="https://api.openai.com/v1",
        help="API base URL (default: https://api.openai.com/v1)"
    )
    parser.add_argument(
        "--model", 
        default="gpt-4",
        help="Model name to use (default: gpt-4)"
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        default=1.0,
        help="Temperature for model generation"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=5,
        help="Number of puzzles to process concurrently (default: 5)"
    )
    parser.add_argument(
        "--results-file", 
        default="sparc_results.json",
        help="File to save/load intermediate results (default: sparc_results.json)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Ignore existing results file and start fresh"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed output for each puzzle"
    )
    
    args = parser.parse_args()

    # Header
    console.print(Panel.fit("🧩 SPaRC: Spatial Pathfinding and Reasoning Challenge", style="bold blue"))
    
    with console.status("[bold green]Loading SPaRC dataset..."):
        dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")
    
    total_puzzles = len(dataset)
    
    # Load existing results unless overwrite is requested
    skip_processed = set()
    if not args.overwrite:
        _, skip_processed = load_results(args.results_file)
    elif os.path.exists(args.results_file):
        console.print(f"[yellow]🗑️  Overwrite requested, ignoring existing {args.results_file}[/]")
    
    # Configuration info
    config_table = Table(box=box.SIMPLE)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Dataset Size", str(total_puzzles))
    config_table.add_row("Already Processed", str(len(skip_processed)))
    config_table.add_row("Remaining", str(total_puzzles - len(skip_processed)))
    config_table.add_row("Model", args.model)
    config_table.add_row("Temperature", str(args.temperature))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Results File", args.results_file)
    config_table.add_row("Base URL", args.base_url)
    
    console.print(Panel(config_table, title="Configuration", style="blue"))
    console.print("[dim]💡 Press Ctrl+C to gracefully stop after current batch[/]")

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    
    try:
        # Run async processing
        results = asyncio.run(process_dataset_async(
            dataset, client, args.model, args.temperature, args.batch_size, args.verbose, args.results_file, skip_processed
        ))
        
        if shutdown_requested:
            console.print("[yellow]🛑 Processing stopped by user request[/]")
        else:
            console.print("[green]✅ Processing completed successfully![/]")
        
    except KeyboardInterrupt:
        console.print("[yellow]🛑 Interrupted during processing[/]")
        return
    
    # Display final results
    console.print("\n")
    console.print(create_statistics_table(results))
    console.print("\n")
    console.print(create_detailed_results_table(results))
    
    console.print(f"\n[green]📁 Final results saved to {args.results_file}[/]")


if __name__ == "__main__":
    main()
