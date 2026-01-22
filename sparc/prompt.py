from typing import Dict
import json


def generate_prompt(puzzle_data: Dict) -> str:
    grid_size = puzzle_data.get("grid_size", {"width": 0, "height": 0})
    puzzle_array = puzzle_data.get("puzzle_array", [])
    grid_str = "\n".join(map(str, puzzle_array))
    start_pos = None
    end_pos = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_pos = f"({x}, {y})"
            elif cell == "E":
                end_pos = f"({x}, {y})"

    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"

    return f"""
    ## Objective
    You are a specialized AI proficient in spatial reasoning and solving puzzles from the game 'The Witness'. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    ## Core Concepts & Grid Basics
    *   **Grid Dimensions:** The puzzle grid has {grid_size['width']} columns and {grid_size['height']} rows.
    *   **Coordinate System:** Nodes are identified by `(x, y)` coordinates. `(0,0)` is the top-left node. `x` increases to the right, `y` increases downwards.
    *   **Path:** The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    *   **No Revisits:** The path **CANNOT** visit the same node more than once.
    *   **Valid Path Cells:** The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    *   **Rule Cells:** Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.
    *   **Regions:** The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.

    ## Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y

    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black

    ## Detailed Solving Rules
    The drawn path must satisfy **ALL** applicable constraints:

    1.  **Path Constraints:**
        *   Path **MUST** start at `S` and end at `E`.
        *   Path connects adjacent nodes (horizontal/vertical moves only).
        *   Nodes **CANNOT** be revisited.
        *   Path **MUST** pass through all Dot (`.`) cells.
        *   Path **CANNOT** pass through any Gap (`G`) cells.

    2.  **Region-Based Rules** (Apply to areas enclosed by the path):
        *   **Squares (`o-X`):** All squares within a single region **MUST** be the same color. Squares of different colors **MUST** be separated into different regions by the path.
        *   **Stars (`*-X`):** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other element (star or square) *of the same color*. Other colors within the region are irrelevant to this specific star's rule.
        *   **Polyshapes (`P-X-Y`):** The region containing this symbol **MUST** be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries. If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is NOT allowed. They must fit within the provided space in their given orientation.
        *   **Negative Polyshapes (`Y-X-Y`):** These "subtract" shape requirements, typically within the same region as corresponding positive polyshapes. A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region. If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one. Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

    3.  **Path-Based Rules (Edge Touching):**
        *   **Triangles (`A-X`, `B-X`, `C-X`, `D-X`):** The path **MUST** touch a specific number of edges of the cell containing the triangle symbol.
            *   `A-X` (1): Path touches **EXACTLY 1** edge of the triangle's cell.
            *   `B-X` (2): Path touches **EXACTLY 2** edges of the triangle's cell.
            *   `C-X` (3): Path touches **EXACTLY 3** edges of the triangle's cell.
            *   `D-X` (4): Path touches **EXACTLY 4** edges (fully surrounds) the triangle's cell.

    ## EXAMPLE PUZZLE GRID:

    ["+",".","+","+","+","E","+"]
    ["+","C-R","+","o-K","+","o-K","+"]
    ["S","+","+","+","+","+","+"]
    ["+","P-G-112","+","*-G","+","P-B-624","+"]
    ["+","+","+","+","+","+","+"]
    ["+","*-G","+","*-G","+","o-K","+"]
    ["+","+","+",".","+","+","+"]

    EXAMPLE POLYSHAPE DEFINITIONS:
    Shape 112:
    [0,1,0,0]
    [0,1,0,0]
    [0,1,0,0]
    [0,0,0,0]

    Shape 624:
    [0,1,0,0]
    [0,1,1,0]
    [0,1,0,0]
    [0,0,0,0]

    EXAMPLE SOLUTION:

    We start at (0,2) and draw a line to (0,0).
    We then draw a line to (2,0) to reach the dot at (1,0) and surround the 3 count triangle.
    We then draw a line to (2,2) here we go down to touch the third side of the triangle cell and therefore validate the 3 count triangle.
    We continue down to (2,6) to validate the polyshape 112 and also the green star with the green polyshape
    After this we draw a line to (4,6) to start validating the polyshape 624 by surrounding it.
    Therefore we have to draw a line to (6,4) over (4,4) which creates a region for the stone at (5,5) which validates the stone.
    We continue up to (6,2) for the polyshape 624 and then go to (4,2) and after this to (4,0) to finaly validate the polyshape 624.
    This also validates the two green stars at (3,3) and (3,5) with each other and the black stone at (3,1) because its the only stone in its region.
    This line also creates a region for the black stone at (5,1) because its the only stone in its region.
    Now we can draw a line to (5,0) to reach the end node.

    #### (0,2),(0,1),(0,0),(1,0),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(3,6),(4,6),(4,5),(4,4),(5,4),(6,4),(6,3),(6,2),(5,2),(4,2),(4,1),(4,0),(5,0)

    ## Puzzle Input Data
    *   **Start Node:** {start_pos}
    *   **End Node:** {end_pos}
    *   **Grid Layout:**
        ```
        {grid_str}
        ```
    *   **Polyshape Definitions (if applicable):**
        *   Shapes are defined by 2D arrays where '1' indicates an occupied cell and '0' indicates an empty cell.
        ```
        {polyshapes_str}
        ```

    ## Task & Output Format
    1.  **Solve the Puzzle:** Determine the valid path from the Start Node to the End Node that satisfies all rules.
    2.  **Explain Reasoning:** Provide a step-by-step explanation of your thought process. Detail key deductions, how constraints were applied, and any backtracking or choices made.
    3.  **Provide Solution Path:** After the reasoning, output the exact marker string `####` followed immediately by the solution path as a list of node coordinates `(x, y)`. Include all intermediate nodes from start to end.

    **Example Solution Path Format:**
    ####
    (0,0), (1,0), (2,0), (2,1), ...
    """

def generate_prompt_step_by_step(puzzle_data: Dict) -> str:
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can not traceback your path. you can not visit a cell twice.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.


    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y
        
    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black
    
        
    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN NOT be revisited. You cannot visit a cell twice.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
        
        Polyshapes(poly): The region containing this symbol MUST be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries.
        If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is NOT allowed. They must fit within the provided space in their given orientation.
        
        Negative Polyshapes(ylop): These subtract shape requirements, typically within the same region as corresponding positive polyshapes.
        A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region.
        If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one.
        Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. 
    {polyshapes_str}

    At each turn you'll receive current Information as JSON.
    Observation: The current state of the grid, including the path and any rule cells.
    The Observation is a json string representation:
    Example observation: [["+","+","+","+","+","G","V"],["+","N",".","*-K","+","N","V"],["V","V","V","V","V","V","V"],["L","N","+","*-K","+","N","+"],["+",".","+","+","+","+","+"]]

    Info: Additional information about the puzzle, including:
    solution_count: The number of valid solutions for the current puzzle.
    difficulty: The difficulty level of the puzzle, ranging from 1 (easy) to 5 (hard).
    grid_x_size: The width of the grid.
    grid_y_size: The height of the grid.
    legal_actions: A list of legal actions you can take, represented as integers (0=right, 1=up, 2=left, 3=down).
    current_step: The current step number in the episode.
    agent_location: The current location of the agent in the grid.
    Rewards: A dictionary containing the normal reward and the outcome reward at the current step.
    rule_status: A dictionary indicating the status of each rule type (e.g., whether all squares are correctly grouped by color).

    Reward: The current reward.

    You MAY think step‐by‐step (feel free to "<think>…"), but you MUST end with:
    Final: <digit>
    where <digit> is exactly one of 0=right, 1=up, 2=left, 3=down.
    No other output beyond your reasoning and that Final line.
    """

def generate_prompt_step_by_step_traceback(puzzle_data: Dict) -> str:
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can traceback your path, but you MUST do so in the same way you came, without crossing over your own path. 
    When tracing back, you can only move to the last cell you occupied, and then continue from there. Also when you trace back, the nodes you no longer use in your path are free to be used again.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.


    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y
        
    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black


    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN be revisited. But only if you trace back to the last cell you occupied (and from there again and again ...).
        Otherwise you CANNOT cross your own path.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
        
        Polyshapes(poly): The region containing this symbol MUST be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries.
        If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is generally allowed unless context implies otherwise.
        
        Negative Polyshapes(ylop): These subtract shape requirements, typically within the same region as corresponding positive polyshapes.
        A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region.
        If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one.
        Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. 
    {polyshapes_str}

    At each turn you'll receive current Information as JSON.
    Observation: The current state of the grid, including the path and any rule cells.
    The Observation is a json string representation:
    Example observation: [["+","+","+","+","+","G","V"],["+","N",".","*-K","+","N","V"],["V","V","V","V","V","V","V"],["L","N","+","*-K","+","N","+"],["+",".","+","+","+","+","+"]]

    Info: Additional information about the puzzle, including:
    solution_count: The number of valid solutions for the current puzzle.
    difficulty: The difficulty level of the puzzle, ranging from 1 (easy) to 5 (hard).
    grid_x_size: The width of the grid.
    grid_y_size: The height of the grid.
    legal_actions: A list of legal actions you can take, represented as integers (0=right, 1=up, 2=left, 3=down).
    current_step: The current step number in the episode.
    agent_location: The current location of the agent in the grid.
    Rewards: A dictionary containing the normal reward and the outcome reward at the current step.
    rule_status: A dictionary indicating the status of each rule type (e.g., whether all squares are correctly grouped by color).

    Reward: The current reward.

    You MAY think step‐by‐step (feel free to "<think>…"), but you MUST end with:
    Final: <digit>
    where <digit> is exactly one of 0=right, 1=up, 2=left, 3=down.
    No other output beyond your reasoning and that Final line.
    """


def generate_prompt_step_by_step_visual(puzzle_data: Dict) -> str:
    """Generate a visual step-by-step prompt for gym mode without traceback."""
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    
    return f"""You are an autonomous agent controlling a path-finding puzzle solver inspired by "The Witness" game.
You will receive an IMAGE of the puzzle at each step showing your current progress.

## Understanding the Visual Puzzle

Looking at the puzzle image:
- **Start Point**: A small circular connector on the edge of the grid (where the path begins)
- **End Point**: A small circular connector on the edge of the grid (where you must reach)
- **Green Cells**: Valid path cells you can move through
- **Gray Border/Lines**: The grid structure
- **Colored Squares**: Rule symbols that must be satisfied
- **Colored Stars**: Must be paired with exactly one other element of the same color in the same region
- **Triangles**: The path must touch a specific number of edges of that cell (number of triangles shown)
- **Black Hexagon/Diamond**: Current position marker

## Your Goal
Navigate from the start to the end, drawing a continuous path that:
1. Does NOT cross itself (no revisiting cells)
2. Separates different colored squares into different regions
3. Pairs each star with exactly one other same-colored element per region
4. Touches the correct number of edges for triangle cells

## Actions
You can move in 4 directions:
- **0**: Move UP
- **1**: Move RIGHT  
- **2**: Move DOWN
- **3**: Move LEFT

{polyshapes_str}

## How to Respond
Look at the current puzzle image showing your path progress. Analyze which direction leads toward the exit while following the puzzle rules.

You MAY think step-by-step, but you MUST end your response with:
Final: <digit>

Where <digit> is exactly one of 0, 1, 2, or 3.
"""


def generate_prompt_step_by_step_visual_traceback(puzzle_data: Dict) -> str:
    """Generate a visual step-by-step prompt for gym mode with traceback enabled."""
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    
    return f"""You are an autonomous agent controlling a path-finding puzzle solver inspired by "The Witness" game.
You will receive an IMAGE of the puzzle at each step showing your current progress.

## Understanding the Visual Puzzle

Looking at the puzzle image:
- **Start Point**: A small circular connector on the edge of the grid (where the path begins)
- **End Point**: A small circular connector on the edge of the grid (where you must reach)
- **Green Cells**: Valid path cells you can move through
- **Gray Border/Lines**: The grid structure
- **Colored Squares**: Rule symbols that must be satisfied
- **Colored Stars**: Must be paired with exactly one other element of the same color in the same region
- **Triangles**: The path must touch a specific number of edges of that cell (number of triangles shown)
- **Black Hexagon/Diamond**: Current position marker
- **Path Line**: Shows where you have already traveled

## Your Goal
Navigate from the start to the end, drawing a continuous path that:
1. You CAN trace back along your existing path to undo moves
2. You CANNOT cross your path at a new point (only retrace backwards)
3. Separates different colored squares into different regions
4. Pairs each star with exactly one other same-colored element per region
5. Touches the correct number of edges for triangle cells

## Actions
You can move in 4 directions:
- **0**: Move UP
- **1**: Move RIGHT  
- **2**: Move DOWN
- **3**: Move LEFT

{polyshapes_str}

## How to Respond
Look at the current puzzle image showing your path progress. Analyze which direction leads toward the exit while following the puzzle rules. You can backtrack if needed.

You MAY think step-by-step, but you MUST end your response with:
Final: <digit>

Where <digit> is exactly one of 0, 1, 2, or 3.
"""
