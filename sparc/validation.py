import re
from typing import List, Optional, Dict


def extract_solution_path(
    solution_text: str, puzzle_data: Dict = None
) -> Optional[List[Dict[str, int]]]:
    """Extract solution path from LLM's response

    Args:
        solution_text: Text response from the LLM
        puzzle_data: Optional puzzle data dict to extract end point from

    Returns:
        List of coordinate dicts or None if no path found
    """
    # First, check if "Solution" appears in the text
    solution_marker = "####"
    if solution_marker in solution_text:
        # Only process text after "####"
        solution_part = solution_text.split(solution_marker)[-1]
    else:
        # If no solution marker, use the full text
        solution_part = solution_text

    # Look for coordinate patterns like (0,0) -> (0,1) or similar
    # Pattern for (x,y) or (x, y) coordinates
    coord_pattern = r"\((\d+),\s*(\d+)\)"
    coords = re.findall(coord_pattern, solution_part)

    if coords:
        # Extract end point from puzzle data if provided
        end_point = None
        if puzzle_data:
            puzzle_array = puzzle_data.get("puzzle_array", [])
            for y, row in enumerate(puzzle_array):
                for x, cell in enumerate(row):
                    if cell == "E":
                        end_point = {"x": x, "y": y}
                        break
                if end_point:
                    break

        # Convert string coordinates to integer dicts
        path = []
        for x, y in coords:
            point = {"x": int(x), "y": int(y)}
            path.append(point)

            # Stop extracting if we've reached the end point
            if (
                end_point
                and point["x"] == end_point["x"]
                and point["y"] == end_point["y"]
            ):
                break

        return path

    # If no coordinates found, return None
    return None


def validate_solution(extracted_path: Optional[List[Dict[str, int]]], puzzle_data: Dict) -> bool:
    """Validate if the solution is valid by comparing with known solutions

    Args:
        extracted_path: Optional list of coordinate dicts (can be None if no path extracted)
        puzzle_data: Dictionary containing puzzle data including solutions

    Returns:
        Boolean indicating if the solution is valid. Returns False if no valid path is provided.
    """
    # If no path was extracted, validation fails immediately.
    if not extracted_path:
        return False

    extracted_path = [(p["x"], p["y"]) for p in extracted_path]
    if len(extracted_path) < 2:
        return False

    # Check against all valid solutions in the database
    all_solutions = puzzle_data.get("solutions", [])
    if not all_solutions:
        return False

    # For each solution in the database, check if our path matches
    for solution in all_solutions:
        solution_path = [(p["x"], p["y"]) for p in solution["path"]]

        # Check if the paths match exactly
        if extracted_path == solution_path:
            return True

    return False


def analyze_path(solution_path: Optional[List[Dict[str, int]]], puzzle: Dict) -> Dict:
    """Analyze the solution path for detailed validation metrics

    Args:
        solution_path: Optional list of coordinate dicts representing the solution path (can be None)
        puzzle: Puzzle dictionary with puzzle array and metadata

    Returns:
        Dictionary with detailed path analysis results
    """
    # Early return with all metrics False if no path was provided
    if not solution_path:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    solution_path = [(p["x"], p["y"]) for p in solution_path]

    # Get puzzle information
    puzzle_array = puzzle.get("puzzle_array", [])
    if not puzzle_array:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    # Find start and end points
    start_point = None
    end_point = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_point = (x, y)
            elif cell == "E":
                end_point = (x, y)

    if not start_point or not end_point:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    # Check if path starts at start and ends at exit
    starts_at_start = len(solution_path) > 0 and solution_path[0] == start_point
    ends_at_exit = len(solution_path) > 0 and solution_path[-1] == end_point
    starts_at_start_ends_at_exit = starts_at_start and ends_at_exit

    # Check if path is connected (no gaps)
    connected_line = True
    for i in range(1, len(solution_path)):
        prev_x, prev_y = solution_path[i - 1]
        curr_x, curr_y = solution_path[i]
        # Check if adjacent (Manhattan distance of 1)
        if abs(prev_x - curr_x) + abs(prev_y - curr_y) != 1:
            connected_line = False
            break

    # Check if path doesn't intersect with itself
    non_intersecting_line = len(set(solution_path)) == len(solution_path)

    # Check if there's a connected path from start to exit
    start_to_exit_connected = starts_at_start_ends_at_exit and connected_line

    # Identify rule cells as cells where both coordinates are odd
    rule_cells = set()
    for y in range(len(puzzle_array)):
        for x in range(len(puzzle_array[0]) if len(puzzle_array) > 0 else 0):
            if x % 2 == 1 and y % 2 == 1:
                rule_cells.add((x, y))

    # Check if path crosses rule cells
    no_rule_crossing = not any((x, y) in rule_cells for x, y in solution_path[1:-1])

    # Check if path is fully valid
    fully_valid_path = (
        starts_at_start_ends_at_exit
        and connected_line
        and non_intersecting_line
        and no_rule_crossing
    )

    return {
        "starts_at_start_ends_at_exit": starts_at_start_ends_at_exit,
        "connected_line": connected_line,
        "non_intersecting_line": non_intersecting_line,
        "start_to_exit_connected": start_to_exit_connected,
        "no_rule_crossing": no_rule_crossing,
        "fully_valid_path": fully_valid_path,
    }
