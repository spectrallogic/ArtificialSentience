# arc_challenge/test_one_task_simple.py
"""
Train on ONE task only - with PROPER metrics that don't count background.

CRITICAL FIX: Don't let the model cheat by matching background pixels!
Only measure accuracy on pixels that actually matter.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
import numpy as np
from baby_asi import ASISeed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_task(task_path):
    """Load a single ARC task"""
    with open(task_path, 'r') as f:
        return json.load(f)


def grid_to_vector(grid, max_dim):
    """Convert grid (list of lists) to flat vector with padding"""
    arr = np.array(grid, dtype=np.float32)
    flat = arr.flatten() / 9.0  # Normalize colors 0-9 to 0-1

    # Pad to max_dim
    if len(flat) < max_dim:
        padded = np.zeros(max_dim, dtype=np.float32)
        padded[:len(flat)] = flat
        return padded
    return flat


def vector_to_grid(vec, height, width):
    """Convert flat vector back to grid"""
    arr = (vec * 9.0).round().clip(0, 9).astype(int)
    # Only take the actual grid size
    arr = arr[:height * width]
    return arr.reshape(height, width)


def compute_focused_accuracy(pred_grid, target_grid, input_grid):
    """
    Compute accuracy that DOESN'T count background matching.

    Only measure pixels that:
    1. Changed from input to output (the actual transformation)
    2. Are non-zero in either target or prediction
    """
    # Find pixels that matter (not background in target OR changed from input)
    changed_pixels = (input_grid != target_grid)
    non_background = (target_grid != 0) | (pred_grid != 0)
    important_pixels = changed_pixels | non_background

    if important_pixels.sum() == 0:
        # If no important pixels, fall back to all pixels
        important_pixels = np.ones_like(target_grid, dtype=bool)

    # Only compute accuracy on important pixels
    correct_important = (pred_grid[important_pixels] == target_grid[important_pixels]).sum()
    total_important = important_pixels.sum()

    # Also compute total accuracy for comparison
    correct_all = (pred_grid == target_grid).sum()
    total_all = pred_grid.size

    return {
        'focused_accuracy': 100.0 * correct_important / total_important,
        'total_accuracy': 100.0 * correct_all / total_all,
        'important_pixels': total_important,
        'total_pixels': total_all,
        'correct_important': correct_important
    }


def weighted_loss(pred, target, input_vec, height, width):
    """
    Loss that focuses on pixels that actually change.
    Background pixels get less weight.
    """
    # Reshape to grids for analysis
    actual_size = height * width

    pred_grid = (pred[:actual_size] * 9.0).round().clip(0, 9)
    target_grid = (target[:actual_size] * 9.0).round().clip(0, 9)
    input_grid = (input_vec[:actual_size] * 9.0).round().clip(0, 9)

    # Create weight mask
    weights = torch.ones_like(pred)

    # High weight for changed pixels
    changed_mask = (input_grid != target_grid).float()
    weights[:actual_size] = 1.0 + 4.0 * changed_mask  # 5x weight for changed pixels

    # Compute weighted MSE
    loss = (weights * (pred - target) ** 2).mean()

    return loss


def visualize_example(input_grid, target_grid, pred_grid, accuracy_info, title="Example"):
    """Visualize with focus on what matters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: The grids
    axes[0, 0].imshow(input_grid, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    axes[0, 0].set_title('Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_grid, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    axes[0, 1].set_title('Target (Correct Answer)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_grid, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    axes[0, 2].set_title(f'AI Prediction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Analysis
    # What changed from input?
    changed_from_input = (input_grid != target_grid).astype(float)
    axes[1, 0].imshow(changed_from_input, cmap='Oranges', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].set_title('What Should Change\n(orange = transformation)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # What the AI got right/wrong (focused on important pixels)
    important_pixels = (input_grid != target_grid) | (target_grid != 0) | (pred_grid != 0)
    correctness = np.zeros_like(target_grid, dtype=float)
    correctness[important_pixels] = (pred_grid[important_pixels] == target_grid[important_pixels]).astype(float)
    correctness[~important_pixels] = 0.5  # Gray for unimportant background

    axes[1, 1].imshow(correctness, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].set_title(f'Correctness Map\n(green=right, red=wrong, gray=background)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Error map
    error = np.abs(target_grid.astype(float) - pred_grid.astype(float))
    axes[1, 2].imshow(error, cmap='Reds', vmin=0, vmax=9, interpolation='nearest')
    axes[1, 2].set_title('Error Magnitude', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # Add accuracy text
    focused_acc = accuracy_info['focused_accuracy']
    total_acc = accuracy_info['total_accuracy']
    important_px = accuracy_info['important_pixels']

    fig.text(0.5, 0.02,
             f"Focused Accuracy (important pixels): {focused_acc:.1f}% ({accuracy_info['correct_important']}/{important_px} pixels)\n"
             f"Total Accuracy (all pixels): {total_acc:.1f}%",
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


def test_one_task():
    """Test with ONE task only - with PROPER metrics"""

    print("=" * 70)
    print("BABY ASI: SINGLE TASK MASTERY (WITH REAL METRICS)")
    print("=" * 70)
    print("CRITICAL FIX: Only count pixels that actually matter!")
    print("Background pixels don't count toward accuracy.")
    print("=" * 70)

    # Load ONE task
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_path = os.path.join(project_root, "arc_data", "corpus", "Center", "Center1.json")

    if not os.path.exists(task_path):
        print(f"❌ Task not found at: {task_path}")
        return

    task = load_task(task_path)
    print(f"\n✓ Loaded task: Center_Center1")
    print(f"  Training examples: {len(task['train'])}")
    print(f"  Test examples: {len(task['test'])}")

    # Find maximum dimension across ALL examples
    max_dim = 0
    all_examples = task['train'] + task['test']

    for example in all_examples:
        for grid in [example['input'], example['output']]:
            height = len(grid)
            width = len(grid[0]) if grid else 0
            max_dim = max(max_dim, height * width)

    print(f"  Max grid dimension: {max_dim} pixels")

    # Prepare training data
    train_data = []

    for example in task['train']:
        train_data.append({
            'input': example['input'],
            'output': example['output'],
            'height': len(example['output']),
            'width': len(example['output'][0])
        })

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧠 Creating model (device: {device})...")

    model = ASISeed(
        input_dim=max_dim,
        model_dim=128,
        num_clusters=8,
        core_rank=2,
        build_ema=False,
        use_heads=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # Training with weighted loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 150  # Train longer
    print(f"\n🎓 Training for {num_epochs} epochs (with weighted loss)...")

    growth_count = 0
    stuck_count = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_stuck = 0

        for i, example in enumerate(train_data):
            # Convert to tensors
            x_in = torch.from_numpy(grid_to_vector(example['input'], max_dim)).float().to(device)
            y_target = torch.from_numpy(grid_to_vector(example['output'], max_dim)).float().to(device)

            # Forward with recursive refinement
            model.train()
            y_pred, info = model.recursive_solve(
                x_in, y_target,
                max_iterations=15,  # More iterations
                tau=1.0,
                eps=0.1,
                verbose=False
            )

            # Compute WEIGHTED loss (focus on changed pixels)
            loss = weighted_loss(y_pred, y_target, x_in, example['height'], example['width'])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Check for stuck and trigger growth
            if info['is_stuck']:
                epoch_stuck += 1
                stuck_count += 1

                from collections import Counter
                cluster_counts = Counter(info['clusters_used'])
                most_used_cluster = cluster_counts.most_common(1)[0][0]

                old_rank = model.layer.U_res[most_used_cluster].shape[1] if model.layer.U_res[
                                                                                most_used_cluster].numel() > 0 else 0
                model.grow_cluster(most_used_cluster, grow_rank=2)
                new_rank = model.layer.U_res[most_used_cluster].shape[1]
                growth_count += 1

                if epoch % 15 == 0:
                    print(f"    🌱 Epoch {epoch + 1}, Ex {i + 1}: Cluster {most_used_cluster} {old_rank}→{new_rank}")

        avg_loss = epoch_loss / len(train_data)

        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.5f}, Stuck={epoch_stuck}/{len(train_data)}, Growth={growth_count}")

    print(f"\n✓ Training complete!")
    print(f"  Total growth events: {growth_count}")
    print(f"  Total stuck events: {stuck_count}")

    final_ranks = [model.layer.U_res[i].shape[1] for i in range(model.num_clusters)]
    print(f"  Final ranks: {final_ranks}")

    # Testing with proper metrics
    print(f"\n🧪 Testing on {len(task['test'])} examples (with REAL metrics)...")
    model.eval()

    test_results_focused = []
    test_results_total = []

    for test_idx, test_example in enumerate(task['test']):
        input_grid = test_example['input']
        output_grid = test_example['output']

        height = len(output_grid)
        width = len(output_grid[0])

        print(f"\n  Test {test_idx + 1}: Grid size {height}x{width}")

        # Convert to tensors
        x_in = torch.from_numpy(grid_to_vector(input_grid, max_dim)).float().to(device)
        y_target = torch.from_numpy(grid_to_vector(output_grid, max_dim)).float().to(device)

        # Predict
        with torch.no_grad():
            y_pred, info = model.recursive_solve(
                x_in, y_target,
                max_iterations=15,
                tau=0.1,
                eps=0.0,
                verbose=False
            )

        # Convert back to grids
        pred_grid = vector_to_grid(y_pred.cpu().numpy(), height, width)
        target_grid = np.array(output_grid)
        input_grid_np = np.array(input_grid)

        # Compute FOCUSED accuracy (ignore background)
        accuracy_info = compute_focused_accuracy(pred_grid, target_grid, input_grid_np)

        test_results_focused.append(accuracy_info['focused_accuracy'])
        test_results_total.append(accuracy_info['total_accuracy'])

        print(f"    Focused accuracy (what matters): {accuracy_info['focused_accuracy']:.1f}%")
        print(f"    Total accuracy (with background): {accuracy_info['total_accuracy']:.1f}%")
        print(f"    Important pixels: {accuracy_info['important_pixels']}/{accuracy_info['total_pixels']}")

        # Visualize
        fig = visualize_example(
            input_grid_np,
            target_grid,
            pred_grid,
            accuracy_info,
            title=f"Test Example {test_idx + 1}"
        )

        filename = f'test_result_{test_idx + 1}_focused.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved to: {filename}")
        plt.close()

    avg_focused = sum(test_results_focused) / len(test_results_focused)
    avg_total = sum(test_results_total) / len(test_results_total)

    print(f"\n{'=' * 70}")
    print(f"REAL RESULTS (FOCUSED ON WHAT MATTERS)")
    print(f"{'=' * 70}")
    print(f"  Focused accuracy: {avg_focused:.1f}% (ignoring background)")
    print(f"  Total accuracy: {avg_total:.1f}% (including background)")
    print(f"  Random baseline: ~10%")

    if avg_focused > 80:
        print(f"  ✓✓✓ EXCELLENT! Model truly learned the pattern!")
    elif avg_focused > 50:
        print(f"  ✓✓ GOOD! Model learned something real!")
    elif avg_focused > 30:
        print(f"  ✓ OKAY! Some learning, but needs work")
    else:
        print(f"  ✗ POOR! Model is mostly guessing on important pixels")

    print(f"\n💡 The difference between focused and total accuracy shows")
    print(f"   how much the model was 'cheating' with background pixels.")
    print(f"   Difference: {avg_total - avg_focused:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_one_task()

'''
======================================================================
BABY ASI: SINGLE TASK MASTERY (WITH REAL METRICS)
======================================================================
CRITICAL FIX: Only count pixels that actually matter!
Background pixels don't count toward accuracy.
======================================================================

✓ Loaded task: Center_Center1
  Training examples: 2
  Test examples: 3
  Max grid dimension: 400 pixels

🧠 Creating model (device: cpu)...
✓ Model created: 319,634 parameters

🎓 Training for 150 epochs (with weighted loss)...
    🌱 Epoch 1, Ex 1: Cluster 4 0→2
    🌱 Epoch 1, Ex 2: Cluster 0 0→2
  Epoch 1/150: Loss=0.01342, Stuck=2/2, Growth=2
  Epoch 15/150: Loss=0.00416, Stuck=1/2, Growth=21
  Epoch 30/150: Loss=0.00055, Stuck=0/2, Growth=21
  Epoch 45/150: Loss=0.00001, Stuck=0/2, Growth=21
  Epoch 60/150: Loss=0.00000, Stuck=0/2, Growth=21
  Epoch 75/150: Loss=0.00000, Stuck=0/2, Growth=21
  Epoch 90/150: Loss=0.00000, Stuck=0/2, Growth=21
  Epoch 105/150: Loss=0.00000, Stuck=0/2, Growth=21
  Epoch 120/150: Loss=0.00000, Stuck=0/2, Growth=21
  Epoch 135/150: Loss=0.00000, Stuck=0/2, Growth=21
  Epoch 150/150: Loss=0.00000, Stuck=0/2, Growth=21

✓ Training complete!
  Total growth events: 21
  Total stuck events: 21
  Final ranks: [6, 4, 4, 2, 10, 4, 8, 4]

🧪 Testing on 3 examples (with REAL metrics)...

  Test 1: Grid size 11x11
    Focused accuracy (what matters): 9.3%
    Total accuracy (with background): 67.8%
    Important pixels: 43/121
    ✓ Saved to: test_result_1_focused.png

  Test 2: Grid size 15x15
    Focused accuracy (what matters): 0.0%
    Total accuracy (with background): 70.2%
    Important pixels: 67/225
    ✓ Saved to: test_result_2_focused.png

  Test 3: Grid size 20x20
    Focused accuracy (what matters): 3.5%
    Total accuracy (with background): 79.5%
    Important pixels: 85/400
    ✓ Saved to: test_result_3_focused.png

======================================================================
REAL RESULTS (FOCUSED ON WHAT MATTERS)
======================================================================
  Focused accuracy: 4.3% (ignoring background)
  Total accuracy: 72.5% (including background)
  Random baseline: ~10%
  ✗ POOR! Model is mostly guessing on important pixels

💡 The difference between focused and total accuracy shows
   how much the model was 'cheating' with background pixels.
   Difference: 68.2%
======================================================================
'''





