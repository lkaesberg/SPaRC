import argparse
import asyncio
import json
import os
import signal
import time
import csv
import datetime
from typing import Dict, List, Set, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
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
    """Save results to a JSONL file (one JSON object per line).
    Each line contains the original puzzle data with an added
    "result" key that stores the solver output, analysis and timing.
    """
    try:
        with open(filename, 'w') as f:
            for result in results:
                # The original puzzle data
                puzzle_obj = dict(result['puzzle_data'])  # shallow copy
                # Attach result meta-data
                puzzle_obj['result'] = {
                    'puzzle_id': result['puzzle_id'],
                    'solved': result['solved'],
                    'analysis': result['analysis'],
                    'processing_time': result['processing_time'],
                    'extracted_path': result['extracted_path'],
                    'message': result.get('message'),
                    'error': result.get('error'),
                }
                f.write(json.dumps(puzzle_obj) + "\n")

        console.print(f"[green]💾 Results saved to {filename}[/]")
    except Exception as e:
        console.print(f"[red]❌ Failed to save results: {str(e)}[/]")


def load_results(filename: str) -> tuple[List[Dict], Set[str]]:
    """Load results from a JSONL or legacy JSON file.
    Returns a tuple (results_list, processed_ids_set).
    """
    if not os.path.exists(filename):
        return [], set()

    # Determine by extension
    is_jsonl = filename.endswith('.jsonl')

    try:
        if is_jsonl:
            results = []
            with open(filename, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    puzzle_data = {k: v for k, v in obj.items() if k != 'result'}
                    result_meta = obj.get('result', {})
                    combined = {
                        'puzzle_id': result_meta.get('puzzle_id', puzzle_data.get('id', 'unknown')),
                        'puzzle_data': puzzle_data,
                        'extracted_path': result_meta.get('extracted_path'),
                        'solved': result_meta.get('solved', False),
                        'analysis': result_meta.get('analysis', {}),
                        'processing_time': result_meta.get('processing_time', 0.0),
                        'message': result_meta.get('message'),
                        'error': result_meta.get('error'),
                    }
                    results.append(combined)
            processed_ids = {r['puzzle_id'] for r in results}
        else:
            # Legacy JSON format
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


async def process_dataset_async(dataset, client: AsyncOpenAI, model: str, temperature: float, batch_size: int, verbose: bool, results_file: str, skip_processed: Set[str], max_new: Optional[int] = None) -> List[Dict]:
    """Process the dataset in batches with graceful shutdown support
    Only up to `max_new` unseen puzzles will be processed if specified.
    """
    global shutdown_requested
    
    total_puzzles = len(dataset)
    all_results = []
    
    # Load existing results if any
    existing_results, _ = load_results(results_file)
    all_results.extend(existing_results)
    
    # Count remaining puzzles to process
    remaining_puzzles = [i for i in range(total_puzzles) if dataset[i].get("id", f"idx_{i}") not in skip_processed]
    total_remaining = len(remaining_puzzles)
    
    # Apply limit if requested
    if max_new is not None:
        remaining_puzzles = remaining_puzzles[:max_new]
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
        TextColumn("ETA: {task.fields[eta]}", justify="right"),
        console=console,
        transient=not verbose
    ) as progress:
        
        task = progress.add_task("[cyan]Processing puzzles...", total=total_remaining, eta="--:--:--")
        
        start_time_overall = time.time()
        processed_so_far = 0
        avg_time_est: Optional[float] = None  # seconds per puzzle based on completed batches
        
        # Process remaining puzzles in batches
        for batch_start in range(0, total_remaining, batch_size):
            if shutdown_requested:
                console.print("[yellow]🛑 Shutdown requested, stopping after current batch...[/]")
                break
                
            batch_end = min(batch_start + batch_size, total_remaining)
            batch_indices = remaining_puzzles[batch_start:batch_end]
            batch_puzzles = [(dataset[i], i) for i in batch_indices]
            
            current_batch_ids = [puzzle_data.get("id", f"idx_{i}") for puzzle_data, i in batch_puzzles]
            progress.update(task, description="[cyan]Processing batch...")
            
            # Launch batch processing as background task so we can refresh ETA countdown
            batch_task = asyncio.create_task(process_batch(client, batch_puzzles, model, temperature, verbose))

            # Periodically refresh ETA while the batch is running
            while not batch_task.done():
                elapsed = time.time() - start_time_overall
                if avg_time_est is not None:
                    eta_seconds = max(0.0, avg_time_est * total_remaining - elapsed)
                else:
                    eta_seconds = float('inf')
                progress.update(task, eta=_format_duration(eta_seconds))
                await asyncio.sleep(1)

            batch_results = await batch_task
            all_results.extend(batch_results)
            
            # Save intermediate results after each batch
            save_results(all_results, results_file)
            
            # Also persist updated tables after each batch
            save_tables_csv(all_results, results_file.rsplit('.', 1)[0])
            
            # Update processed count and ETA
            processed_so_far += len(batch_puzzles)
            elapsed = time.time() - start_time_overall
            # Update average estimate now that a batch finished
            avg_time_est = elapsed / processed_so_far
            eta_seconds = max(0.0, avg_time_est * total_remaining - elapsed)
            eta_str = _format_duration(eta_seconds)
            
            progress.update(task, advance=len(batch_puzzles), eta=eta_str)
            
            if shutdown_requested:
                break
    
    return all_results


def save_tables_csv(results: List[Dict], filename_base: str) -> None:
    """Save summary statistics and detailed per-puzzle results as CSV.
    Two files are produced:
        <prefix>_stats.csv   – aggregated metrics
        <prefix>_details.csv – per-puzzle info
    """
    try:
        # 1) Summary statistics -------------------------------------------------
        total = len(results)
        solved_count = sum(1 for r in results if r['solved'])
        success_rate = (solved_count / total * 100) if total else 0.0

        valid_paths = sum(1 for r in results if r['analysis']['fully_valid_path'])
        connected_paths = sum(1 for r in results if r['analysis']['connected_line'])
        start_end_correct = sum(1 for r in results if r['analysis']['starts_at_start_ends_at_exit'])
        non_intersecting = sum(1 for r in results if r['analysis']['non_intersecting_line'])
        no_rule_crossing = sum(1 for r in results if r['analysis']['no_rule_crossing'])

        difficulty_counts = {lvl: 0 for lvl in range(1, 6)}
        difficulty_solved = {lvl: 0 for lvl in range(1, 6)}
        for r in results:
            lvl = r['puzzle_data'].get('difficulty_level')
            if lvl in difficulty_counts:
                difficulty_counts[lvl] += 1
                if r['solved']:
                    difficulty_solved[lvl] += 1

        path_lengths = [len(r['extracted_path']) for r in results if r['extracted_path']]
        avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0

        processing_times = [r['processing_time'] for r in results]
        total_time = sum(processing_times)
        avg_time = total_time / total if total else 0

        stats_rows = [
            ("Total Puzzles Processed", total, "100.0%"),
            ("Correctly Solved", solved_count, f"{success_rate:.1f}%"),
            ("Failed", total - solved_count, f"{100 - success_rate:.1f}%"),
            ("", "", ""),
            ("Fully Valid Paths", valid_paths, f"{valid_paths/total*100:.1f}%" if total else "0.0%"),
            ("Connected Paths", connected_paths, f"{connected_paths/total*100:.1f}%" if total else "0.0%"),
            ("Correct Start/End", start_end_correct, f"{start_end_correct/total*100:.1f}%" if total else "0.0%"),
            ("Non-Intersecting", non_intersecting, f"{non_intersecting/total*100:.1f}%" if total else "0.0%"),
            ("No Rule Violations", no_rule_crossing, f"{no_rule_crossing/total*100:.1f}%" if total else "0.0%"),
            ("", "", ""),
        ]

        # Difficulty rows
        for lvl in range(1, 6):
            solved_lvl = difficulty_solved[lvl]
            total_lvl = difficulty_counts[lvl]
            pct = solved_lvl / total_lvl * 100 if total_lvl else 0.0
            stats_rows.append((f"Difficulty {lvl} Solved", f"{solved_lvl}/{total_lvl}", f"{pct:.1f}%"))

        stats_rows.extend([
            ("", "", ""),
            ("Avg Path Length", f"{avg_path_length:.1f} steps", ""),
            ("Min Path Length", f"{min(path_lengths) if path_lengths else 0} steps", ""),
            ("Max Path Length", f"{max(path_lengths) if path_lengths else 0} steps", ""),
            ("", "", ""),
            ("Total Time", f"{total_time:.1f} seconds", ""),
            ("Avg Time per Puzzle", f"{avg_time:.2f} seconds", ""),
            ("Puzzles per Minute", f"{(total/total_time*60):.1f}" if total_time else "0.0", ""),
        ])

        with open(f"{filename_base}_stats.csv", "w", newline="") as f_stats:
            writer = csv.writer(f_stats)
            writer.writerow(["Metric", "Value", "Percentage"])
            writer.writerows(stats_rows)

        # 2) Detailed per-puzzle results --------------------------------------
        with open(f"{filename_base}_details.csv", "w", newline="") as f_det:
            writer = csv.writer(f_det)
            writer.writerow(["Puzzle ID", "Difficulty", "Solved", "Path Length", "Time (s)", "Issues"])
            for r in results:
                puzzle_id = r['puzzle_id']
                difficulty = r['puzzle_data'].get('difficulty_level', 'N/A')
                solved_status = "PASS" if r['solved'] else "FAIL"
                path_len = len(r['extracted_path']) if r['extracted_path'] else 0
                time_taken = f"{r['processing_time']:.2f}"

                issues = []
                analysis = r.get('analysis', {})
                if analysis and not analysis.get('fully_valid_path', True):
                    if not analysis.get('starts_at_start_ends_at_exit', True):
                        issues.append("start/end")
                    if not analysis.get('connected_line', True):
                        issues.append("disconnected")
                    if not analysis.get('non_intersecting_line', True):
                        issues.append("intersecting")
                    if not analysis.get('no_rule_crossing', True):
                        issues.append("rules")
                issues_str = ", ".join(issues) if issues else "None"

                writer.writerow([puzzle_id, difficulty, solved_status, path_len, time_taken, issues_str])

        console.print(f"[green]📑 CSV tables saved with prefix {filename_base}_*.csv[/]")
    except Exception as e:
        console.print(f"[red]❌ Failed to save CSV tables: {e}[/]")


def _format_duration(seconds: float) -> str:
    """Return HH:MM:SS string for a duration in seconds."""
    if seconds <= 0 or seconds == float('inf'):
        return "--:--:--"
    return str(datetime.timedelta(seconds=int(seconds)))


def _compute_eta(elapsed: float, processed: int, total: int) -> float:
    """Return seconds remaining based on average time per finished sample."""
    if processed == 0:
        return float('inf')
    avg = elapsed / processed
    expected_total = avg * total
    remaining = max(0.0, expected_total - elapsed)
    return remaining


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
        default=None,
        help="File to save/load intermediate results (default: <model>.jsonl)"
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
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Process at most this many new puzzles before terminating"
    )
    
    args = parser.parse_args()

    # Determine results file: use provided name or model name, ensure .jsonl extension
    base_name = args.results_file if args.results_file else args.model.replace('/', '_')
    results_file = base_name if base_name.endswith('.jsonl') else f"{base_name}.jsonl"

    # Header
    console.print(Panel.fit("🧩 SPaRC: Spatial Pathfinding and Reasoning Challenge", style="bold blue"))
    
    with console.status("[bold green]Loading SPaRC dataset..."):
        dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")
    
    total_puzzles = len(dataset)
    
    # Load existing results unless overwrite is requested
    skip_processed = set()
    if not args.overwrite:
        _, skip_processed = load_results(results_file)
    elif os.path.exists(results_file):
        console.print(f"[yellow]🗑️  Overwrite requested, ignoring existing {results_file}[/]")
    
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
    config_table.add_row("Results File", results_file)
    config_table.add_row("Max New", str(args.max_new) if args.max_new else "All")
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
            dataset, client, args.model, args.temperature, args.batch_size, args.verbose, results_file, skip_processed, args.max_new
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
    
    console.print(f"\n[green]📁 Final results saved to {results_file}[/]")

    # Save tables as CSV
    save_tables_csv(results, results_file.rsplit('.', 1)[0])


if __name__ == "__main__":
    main()
