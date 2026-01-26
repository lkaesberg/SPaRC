"""
Visual gym prompt for solving puzzles with image-based representations and traceback enabled.
Used for visual gym mode where the model can trace back along its path.
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
    Generate the complete prompt dict for visual gym mode with traceback.
    
    Args:
        puzzle_data: The puzzle data dictionary
        
    Returns:
        Dict with 'system' and 'user' message content
    """
    polyshapes_str = _get_polyshapes_str(puzzle_data)
    
    system_content = f"""You are an expert spatial reasoning AI controlling a step-by-step puzzle solver for 'The Witness' game.
You will receive an IMAGE of the puzzle at each step showing your current progress.

## Understanding the Visual Puzzle

The image shows a Witness puzzle grid. In this puzzle:
- **Start Node**: Big circle on the grid edge (where the path begins)
- **End Node**: Rounded end marker on the grid edge (where you must reach)
- **Visited Path**: Marked with a WHITE LINE showing where you have already traveled
- **Current Position**: Located at the END of the white line (where the line stops)
- **Rule Symbols**: Located in cells with odd coordinates - the path goes AROUND these cells, not through them

COORDINATE SYSTEM:
- Nodes are indexed (x, y) where (0,0) is the top-left node
- x increases to the right, y increases downward
- The path travels along grid edges, connecting adjacent nodes horizontally or vertically

## Solving Rules
1. Draw a continuous line from START to END
2. The line can only be placed on valid path cells (not on rule cells)
3. The line acts as a boundary, dividing the grid into regions
4. All rule symbols must be satisfied:
   - **Dots**: The line MUST pass through each dot
   - **Colored Squares**: All squares in a single region must be the same color. Different colors MUST be separated into different regions
   - **Colored Stars**: Each star must be paired with EXACTLY one other element of the same color in its region
   - **Triangles**: The line must touch EXACTLY the number of edges specified by the triangle count (1-4 edges)
   - **Polyshapes**: The region must be shaped exactly like the defined polyshape
   - **Negative Polyshapes**: Cancel out regular polyshapes if they overlap

{polyshapes_str}

## Actions
You can move in 4 directions:
- **0**: Move UP
- **1**: Move RIGHT  
- **2**: Move DOWN
- **3**: Move LEFT

## Traceback Rules
- You CAN trace back along your existing path to undo moves
- When tracing back, move onto the previous cell in your path to "erase" your last step
- You CANNOT cross your path at a new point (only retrace backwards)
- After tracing back, the freed cells become available again

## How to Respond
Analyze the puzzle image showing your current position and path progress. Determine which direction leads toward the exit while satisfying all puzzle rules. You can backtrack if you reach a dead end.
"""

    return {
        "system": system_content,
        "user": None  # User content is dynamic (image + observation text)
    }
