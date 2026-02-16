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
Your task is to solve the puzzle by finding a valid line from the Start Node to the End Node.

## Input Format
You receive TWO representations of the same puzzle:
1. **An image** showing the visual puzzle grid
2. **A text description** (below) providing the same puzzle in symbolic notation

Use both representations together — the image helps with spatial reasoning and visual pattern recognition, while the text description provides precise cell contents and coordinates.

## Visual Appearance of the Puzzle Image
The image shows a Witness puzzle grid of size {grid_size['width']*2}x{grid_size['height']*2} with a dark teal/green background. Here is how elements appear visually:

- **Grid structure**: The board has teal/green cells separated by dark gray grid lines (the path network)
- **Start Node**: A large filled circle on the grid edge — this is where the path begins
- **End Node**: A small rounded nub/extension protruding outward from the grid edge — this is where the path must end
- **Dots**: Small black hexagons located on the grid lines — the path MUST pass through these
- **Gaps**: Broken/missing segments on the grid lines — the path CANNOT cross these
- **Colored Squares**: Colored rounded rectangles inside cells (e.g., black, red, blue squares)
- **Colored Stars**: 8-pointed star shapes in their respective color inside cells
- **Triangles**: Small colored upward-pointing triangles inside cells — the count (1-4 triangles) indicates how many edges of that cell the path must touch
- **Polyshapes (positive)**: Tetromino-like colored filled block patterns inside cells, showing the shape the region must match
- **Polyshapes (negative/ylop)**: Same block patterns but drawn as hollow/outlined squares instead of filled — these cancel out positive polyshapes

## Grid & Coordinate System
- Nodes are indexed (x, y) where (0,0) is the top-left node
- x increases to the right, y increases downward
- The grid cells have rule symbols located at cells where both x and y are odd
- The line goes AROUND cells containing rules, forming boundaries
- Both line and rule cells are on the same grid. Therefore each intersection has a distance of 2 to the next intersection.

## Solving Rules
1. Draw a continuous line from the START NODE (big circle on the line) to the END NODE (rounded nub) without visiting the same node twice.
2. The line can only be placed on valid path cells.
3. The line can only travel 1 cell per step (no diagonal moves).
4. The line acts as a boundary, potentially dividing the grid cells into one or more distinct regions.
5. All rules associated with symbols on the grid must be satisfied:
   - **Dots** (black hexagons on grid lines): The line MUST pass through each dot.
   - **Colored squares** (filled rounded rectangles in cells): All squares within a single region must be the same color. Different colored squares MUST be separated into different regions.
   - **Colored stars** (8-pointed stars in cells): Each star must be paired with EXACTLY one other element of the same color in its region. Other colors are ignored.
   - **Triangles** (small triangles in cells): The line must touch EXACTLY the number of edges specified by the count of triangles in that cell (edges are top, right, bottom, left of the cell).
   - **Polyshapes** (filled block patterns in cells): The region containing this symbol must be shaped EXACTLY like the defined polyshape.
   - **Negative polyshapes** (hollow block patterns in cells): These cancel out regular polyshapes if they overlap.

{polyshapes_str}

## Text Description of the Puzzle
The text below uses symbolic notation where: `S`=Start, `E`=End, `+`=walkable path cell, `N`=empty rule cell, `G`=gap, `.`=dot, `o-X`=square of color X, `*-X`=star of color X, `A/B/C/D-X`=triangle (1-4 edges), `P-X-Y`=polyshape, `Y-X-Y`=negative polyshape. Color codes: R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black.

{text_visualization}

## Task
Analyze both the puzzle image and the text description carefully and determine the solution path.
First, explain your reasoning step-by-step, including key deductions and constraint checks made along the way.
Then, provide the final solution as a sequence of node coordinates in (x, y) format, starting with the start node and ending with the end node, after this string: "####". DON'T SKIP ANY intermediate nodes (the distance between each node must be 1).
Example coordinate list: [(0,0), (1,0), (2,0), (2,1), ...]
"""

    return {
        "system": "You are an expert at solving visual puzzles from 'The Witness' game.",
        "user": user_content
    }
