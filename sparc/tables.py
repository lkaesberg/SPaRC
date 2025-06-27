from typing import Dict, List
from rich.table import Table
from rich import box

def create_statistics_table(results: List[Dict]) -> Table:
    """Create a rich table with comprehensive statistics"""
    total = len(results)
    if total == 0:
        return Table()
    
    # Basic statistics
    solved_count = sum(1 for r in results if r['solved'])
    success_rate = (solved_count / total) * 100
    
    # Path analysis statistics
    valid_paths = sum(1 for r in results if r['analysis']['fully_valid_path'])
    connected_paths = sum(1 for r in results if r['analysis']['connected_line'])
    start_end_correct = sum(1 for r in results if r['analysis']['starts_at_start_ends_at_exit'])
    non_intersecting = sum(1 for r in results if r['analysis']['non_intersecting_line'])
    no_rule_crossing = sum(1 for r in results if r['analysis']['no_rule_crossing'])
    
    # Difficulty distribution statistics
    difficulty_counts = {level: 0 for level in range(1, 6)}
    difficulty_solved_counts = {level: 0 for level in range(1, 6)}
    for r in results:
        level = r['puzzle_data'].get('difficulty_level')
        if level in difficulty_counts:
            difficulty_counts[level] += 1
            if r['solved']:
                difficulty_solved_counts[level] += 1
    
    # Path length statistics
    path_lengths = [len(r['extracted_path']) for r in results if r['extracted_path']]
    avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0
    
    # Time statistics
    processing_times = [r['processing_time'] for r in results]
    total_time = sum(processing_times)
    avg_time = total_time / total if total > 0 else 0
    
    # Create main statistics table
    stats_table = Table(title="📊 SPaRC Dataset Processing Results", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", width=30)
    stats_table.add_column("Value", style="magenta", width=15)
    stats_table.add_column("Percentage", style="green", width=15)
    
    # Overall results
    stats_table.add_row("Total Puzzles Processed", str(total), "100.0%")
    stats_table.add_row("Correctly Solved", str(solved_count), f"{success_rate:.1f}%")
    stats_table.add_row("Failed", str(total - solved_count), f"{100 - success_rate:.1f}%")
    stats_table.add_row("", "", "")  # Separator
    
    # Path analysis
    stats_table.add_row("Fully Valid Paths", str(valid_paths), f"{(valid_paths/total)*100:.1f}%")
    stats_table.add_row("Connected Paths", str(connected_paths), f"{(connected_paths/total)*100:.1f}%")
    stats_table.add_row("Correct Start/End", str(start_end_correct), f"{(start_end_correct/total)*100:.1f}%")
    stats_table.add_row("Non-Intersecting", str(non_intersecting), f"{(non_intersecting/total)*100:.1f}%")
    stats_table.add_row("No Rule Violations", str(no_rule_crossing), f"{(no_rule_crossing/total)*100:.1f}%")
    stats_table.add_row("", "", "")  # Separator
    
    # Difficulty distribution
    for level in range(1, 6):
        total_level = difficulty_counts[level]
        solved_level = difficulty_solved_counts[level]
        solved_pct = (solved_level / total_level * 100) if total_level > 0 else 0.0
        value_col = f"{solved_level}/{total_level}" if total_level > 0 else "0/0"
        stats_table.add_row(f"Difficulty {level} Solved", value_col, f"{solved_pct:.1f}%")
    stats_table.add_row("", "", "")  # Separator
    
    # Path length statistics
    stats_table.add_row("Avg Path Length", f"{avg_path_length:.1f} steps", "")
    if path_lengths:
        stats_table.add_row("Min Path Length", f"{min(path_lengths)} steps", "")
        stats_table.add_row("Max Path Length", f"{max(path_lengths)} steps", "")
    stats_table.add_row("", "", "")  # Separator
    
    # Time statistics
    stats_table.add_row("Total Time", f"{total_time:.1f} seconds", "")
    stats_table.add_row("Avg Time per Puzzle", f"{avg_time:.2f} seconds", "")
    stats_table.add_row("Puzzles per Minute", f"{(total / total_time * 60):.1f}", "")
    
    return stats_table


def create_detailed_results_table(results: List[Dict], show_limit: int = 20) -> Table:
    """Create a detailed table showing individual puzzle results"""
    table = Table(title=f"🔍 Detailed Results (showing first {min(show_limit, len(results))} puzzles)", box=box.SIMPLE)
    table.add_column("Puzzle ID", style="cyan", width=12)
    table.add_column("Difficulty", style="yellow", width=10)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Path Length", style="blue", width=10)
    table.add_column("Time (s)", style="green", width=8)
    table.add_column("Issues", style="red", width=30)
    
    for i, result in enumerate(results[:show_limit]):
        puzzle_id = result['puzzle_id']
        difficulty = result['puzzle_data'].get('difficulty_level', 'N/A')
        status = "✅ PASS" if result['solved'] else "❌ FAIL"
        path_len = len(result['extracted_path']) if result['extracted_path'] else 0
        time_taken = f"{result['processing_time']:.2f}"
        
        # Collect issues
        issues = []
        if 'analysis' in result and not result['analysis']['fully_valid_path']:
            if not result['analysis']['starts_at_start_ends_at_exit']:
                issues.append("start/end")
            if not result['analysis']['connected_line']:
                issues.append("disconnected")
            if not result['analysis']['non_intersecting_line']:
                issues.append("intersecting")
            if not result['analysis']['no_rule_crossing']:
                issues.append("rules")
        
        issues_str = ", ".join(issues) if issues else "None"
        
        table.add_row(
            puzzle_id,
            str(difficulty),
            status,
            str(path_len),
            time_taken,
            issues_str
        )
    
    return table