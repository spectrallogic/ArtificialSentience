# arc_challenge/explore_task.py
"""Just load ONE ARC task and see what it looks like"""
import json
import os


def find_arc_data():
    """Find where the ConceptARC data is"""
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level

    possible_paths = [
        os.path.join(project_root, "arc_data", "corpus"),
        "arc_data\\corpus",
        "arc_data/corpus",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            # Normalize path for Windows
            path = os.path.normpath(path)
            print(f"✓ Found data at: {path}")
            return path

    print("❌ Could not find ConceptARC data!")
    return None


def load_task(filepath):
    """Load a single ARC task from JSON"""
    with open(filepath, 'r') as f:
        task = json.load(f)
    return task


def print_grid(grid, name="Grid"):
    """Print a grid in a readable way"""
    print(f"\n{name}:")
    print(f"Shape: {len(grid)} rows x {len(grid[0])} cols")
    for row in grid:
        print(' '.join(str(cell) for cell in row))


def explore_one_task():
    """Look at one task to understand the format"""

    # First, find the data
    data_path = find_arc_data()
    if data_path is None:
        return

    # List available concept folders
    concepts = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"\n✓ Found {len(concepts)} concept groups:")
    for concept in sorted(concepts)[:5]:  # Show first 5
        print(f"  - {concept}")
    if len(concepts) > 5:
        print(f"  ... and {len(concepts) - 5} more")

    # Pick one easy task to start - use proper path joining
    center_path = os.path.join(data_path, "Center")

    # List files in Center to see what's available
    if os.path.exists(center_path):
        center_files = [f for f in os.listdir(center_path) if f.endswith('.json')]
        print(f"\n✓ Found {len(center_files)} tasks in Center concept")
        task_file = os.path.join(center_path, center_files[0])  # Use first file
        print(f"✓ Loading: {center_files[0]}")
    else:
        print(f"\n❌ Could not find Center folder")
        return

    if not os.path.exists(task_file):
        print(f"\n❌ Could not find task file: {task_file}")
        return

    task = load_task(task_file)

    print("\n" + "=" * 60)
    print("EXPLORING ONE ARC TASK: 'Center' Concept")
    print("=" * 60)

    # Look at training examples
    print(f"\nNumber of training examples: {len(task['train'])}")

    for i, example in enumerate(task['train']):
        print(f"\n--- Training Example {i + 1} ---")
        print_grid(example['input'], "Input")
        print_grid(example['output'], "Output")

    # Look at test examples
    print(f"\n\nNumber of test examples: {len(task['test'])}")
    for i, example in enumerate(task['test']):
        print(f"\n--- Test Example {i + 1} ---")
        print_grid(example['input'], "Test Input")
        print_grid(example['output'], "Expected Output")

    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("- Grids are lists of lists (2D arrays)")
    print("- Values are 0-9 (representing colors: 0=black, 1=blue, 2=red, etc.)")
    print("- We need to learn the pattern from 'train' examples")
    print("- Then apply it to 'test' inputs")
    print("\nCan you see the pattern? What's the transformation rule?")
    print("=" * 60)


if __name__ == "__main__":
    explore_one_task()