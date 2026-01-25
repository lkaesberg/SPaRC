"""
Single-shot visual prompt for solving puzzles in one attempt using image input.
Used for non-gym visual mode where the model receives a puzzle image and returns a complete solution.
"""

from typing import Dict
import json


def _get_polyshapes_str(puzzle_data: Dict) -> str:
    """Extract polyshapes string from puzzle data."""
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return polyshapes_str


def get_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for single-shot visual puzzle solving.
    
    Args:
        puzzle_data: The puzzle data dictionary
        
    Returns:
        Dict with 'system' and 'user' message content
    """
    grid_size = puzzle_data.get("grid_size", {"width": 0, "height": 0})
    polyshapes_str = _get_polyshapes_str(puzzle_data)
    text_visualization = puzzle_data.get("text_visualization", "")

    user_content = f"""You are an expert spatial reasoning AI specializing in solving puzzles from the game 'The Witness'. 
Your task is to solve the puzzle in the image by finding a valid line from the Start Node to the End Node.

The image shows a Witness puzzle grid of size {grid_size['width']*2}x{grid_size['height']*2}. In this puzzle:
- The solution is a continuous line from the start circle to the end marker
- The line travels along grid edges, connecting adjacent nodes horizontally or vertically
- The line cannot visit the same node twice
- The line must satisfy all constraints represented by the symbols on the grid
- The line can not be placed on rule cells
- The line can only travel 1 cell per step (no diagonal moves and provide each step as a separate coordinate)

COORDINATE SYSTEM: 
- Nodes are indexed (x, y) where (0,0) is the top-left node
- x increases to the right, y increases downward
- The grid cells have rule symbols located at cells with all odd coordinates
- The line goes AROUND cells containing rules, forming boundaries
- Both line and rule cells are on the same grid. Therefore each intersection has a distance of 2 to the next intersection.

SOLVING RULES:
1. Draw a continuous line from the START NODE (big circle on the line) to the END NODE (rounded end) without visiting the same node twice.
2. The line can only be placed on valid path cells.
3. The line acts as a boundary, potentially dividing the grid cells into one or more distinct regions.
4. All rules associated with symbols on the grid must be satisfied:
   - Dots: The line MUST pass through each dot.
   - Colored squares: All squares within a single region created by the line must be the same color. Different colored squares MUST be separated into different regions by the line.
   - Colored stars: Each star must be paired with EXACTLY one other element of the same color in a region. Other colors are ignored.
   - Triangles: The line must touch EXACTLY the number of edges specified by the number of triangles in that cell (edges are top, right, bottom, left of the cell).
   - Tetris-like polyomino shapes: The region containing this symbol must be shaped EXACTLY like the defined polyshape.
   - Negative polyshapes: These cancel out regular polyshapes if they overlap.

{polyshapes_str}

Text description of the puzzle:
{text_visualization}

Analyze the puzzle image carefully and determine the solution path.
First, explain your reasoning step-by-step, including key deductions and constraint checks made along the way.
Then, provide the final solution as a sequence of node coordinates in (x, y) format, starting with the start node and ending with the end node, after this string: "####". DON'T SKIP ANY intermediate nodes (the distance between each node must be 1).
Example coordinate list: [(0,0), (1,0), (2,0), (2,1), ...]
"""

    return {
        "system": "You are an expert at solving visual puzzles from 'The Witness' game.",
        "user": user_content
    }
