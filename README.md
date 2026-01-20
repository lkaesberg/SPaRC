<p align="center">
  <img src="https://raw.githubusercontent.com/lkaesberg/SPaRC/main/logo.png" alt="SPaRC Logo" width="128"/>
</p>

# SPaRC: Spatial Pathfinding and Reasoning Challenge

**A comprehensive toolkit for spatial reasoning puzzle solving and model evaluation**

<div align="center">
  <a href="https://sparc.gipplab.org/" style="margin-right:1em; text-decoration:none; font-size:1.1em;">
    <strong>🌐 Visit the Website</strong>
  </a>
  |
  <a href="https://huggingface.co/datasets/lkaesberg/SPaRC" style="margin-left:1em; text-decoration:none; font-size:1.1em;">
    <strong>🤗 Dataset on Hugging Face</strong>
  </a>
  |
  <a href="https://pypi.org/project/sparc-puzzle/" style="margin-left:1em; text-decoration:none; font-size:1.1em;">
    <strong>📦 PyPI Package</strong>
  </a>
</div>

## Overview

SPaRC provides a comprehensive framework for evaluating language models on spatial reasoning tasks inspired by "The Witness" puzzle game. This package includes tools for dataset processing, solution validation, and model evaluation with beautiful terminal output.

## Installation

Install the package from PyPI:

```bash
pip install sparc-puzzle
```

Or install from source:

```bash
git clone https://github.com/lkaesberg/SPaRC.git
cd SPaRC
pip install -e .
```

## Quick Start

### 1. Testing a Model on the Dataset

Run the complete benchmark on your model:

```bash
sparc --api-key "your-openai-api-key" --model "gpt-4" --batch-size 5
```

**Key Features:**
- 🔄 **Resume Support**: Automatically saves progress and resumes from where you left off
- ⚡ **Batching**: Process multiple puzzles concurrently for faster evaluation
- 🎨 **Rich Output**: Beautiful terminal interface with progress tracking
- 🛑 **Graceful Shutdown**: Press Ctrl+C to stop after current batch

**Example with different endpoints:**

```bash
# OpenAI API
sparc --api-key "sk-..." --model "gpt-4"

# Custom endpoint (e.g., local model)
sparc --api-key "your-key" --base-url "http://localhost:8080/v1" --model "llama-3.3-70b"

# Resume interrupted session
sparc --api-key "your-key" --model "gpt-4"  # Automatically resumes

# Fresh start (ignore previous results)
sparc --api-key "your-key" --model "gpt-4" --overwrite
```

### 2. Using the Validation API

Use SPaRC's validation functions in your own code:

```python
from sparc.validation import extract_solution_path, validate_solution, analyze_path
from sparc.prompt import generate_prompt
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")
puzzle = dataset[0]

# Generate prompt for your model
puzzle_prompt = [
                  {
                    "role": "system",
                    "content": "You are an expert at solving puzzles games.",
                  },
                  {
                    "role": "user", 
                    "content": generate_prompt(puzzle)
                  }
                ]


# Your model generates a response
model_response = "... model response with path coordinates ..."

# Extract the path from model response
extracted_path = extract_solution_path(model_response, puzzle)
# Returns: [{"x": 0, "y": 2}, {"x": 0, "y": 1}, ...]

# Validate against ground truth
is_correct = validate_solution(extracted_path, puzzle)
# Returns: True/False

# Get detailed analysis
analysis = analyze_path(extracted_path, puzzle)
# Returns: {
#   "starts_at_start_ends_at_exit": True,
#   "connected_line": True,
#   "non_intersecting_line": True,
#   "no_rule_crossing": True,
#   "fully_valid_path": True
# }
```

## CLI Reference

### Basic Usage

```bash
sparc --api-key "your-key" [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--api-key` | **Required** | OpenAI API key or your model's API key |
| `--base-url` | `https://api.openai.com/v1` | API endpoint URL |
| `--model` | `gpt-4` | Model name to evaluate |
| `--temperature` | `1.0` | Generation temperature |
| `--batch-size` | `5` | Number of concurrent requests |
| `--results-file` | `<model>.jsonl` | File to save results |
| `--overwrite` | `False` | Ignore existing results and start over |
| `--verbose` | `False` | Show detailed output for each puzzle |
| `--max-new` | `None` | Process at most this many new puzzles |
| `--gym` | `False` | Use step-by-step gym mode instead of single-shot |
| `--gym-traceback` | `False` | Enable traceback visualization in gym mode |
| `--run-name` | `None` | Suffix for output filename (e.g., `experiment1`) |

### Examples

```bash
# Basic evaluation
sparc --api-key "sk-..." --model "gpt-4"

# High throughput with larger batches
sparc --api-key "sk-..." --model "gpt-3.5-turbo" --batch-size 20

# Conservative approach with lower temperature
sparc --api-key "sk-..." --model "gpt-4" --temperature 0.1

# Verbose output to see each puzzle result
sparc --api-key "sk-..." --model "gpt-4" --verbose

# Custom results file
sparc --api-key "sk-..." --model "claude-3" --results-file "claude_results.json"

# Process only 10 new puzzles
sparc --api-key "sk-..." --model "gpt-4" --max-new 10

# Step-by-step gym mode (agent receives feedback after each move)
sparc --api-key "sk-..." --model "gpt-4" --gym

# Gym mode with traceback (shows path history in observations)
sparc --api-key "sk-..." --model "gpt-4" --gym --gym-traceback

# Named experiment run
sparc --api-key "sk-..." --model "gpt-4" --run-name "experiment1"
```

### Core Functions

#### `extract_solution_path(solution_text: str, puzzle_data: Dict) -> List[Dict[str, int]]`
Extracts coordinate path from model response text.

#### `validate_solution(extracted_path: List[Dict[str, int]], puzzle_data: Dict) -> bool`
Validates if the extracted path matches any ground truth solution.

#### `analyze_path(solution_path: List[Dict[str, int]], puzzle: Dict) -> Dict`
Provides detailed analysis of path validity and rule compliance.

#### `generate_prompt(puzzle_data: Dict) -> str`
Generates the formatted prompt for a puzzle.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use SPaRC in your research, please cite:

```bibtex
@inproceedings{kaesberg-etal-2025-sparc,
    title = "{SP}a{RC}: A Spatial Pathfinding Reasoning Challenge",
    author = "Kaesberg, Lars Benedikt and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.526/",
    doi = "10.18653/v1/2025.emnlp-main.526",
    pages = "10370--10401"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- 🌐 **Website**: [sparc.gipplab.org](https://sparc.gipplab.org/)
- 📚 **Dataset**: [Hugging Face](https://huggingface.co/datasets/lkaesberg/SPaRC)
- 🐛 **Issues**: [GitHub Issues](https://github.com/lkaesberg/SPaRC/issues)
- 📖 **Documentation**: [GitHub Repository](https://github.com/lkaesberg/SPaRC)
