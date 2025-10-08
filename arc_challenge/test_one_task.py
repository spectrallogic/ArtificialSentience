# arc_challenge/test_one_task.py
"""
Test if our ASISeed can learn ONE ARC task
Just a proof of concept - can we learn the pattern at all?
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
import numpy as np
from baby_asi import ASISeed


def load_task(task_path):
    """Load a single ARC task"""
    with open(task_path, 'r') as f:
        return json.load(f)


def grid_to_vector(grid):
    """Convert grid (list of lists) to flat vector"""
    arr = np.array(grid, dtype=np.float32)
    return arr.flatten() / 9.0  # Normalize colors 0-9 to 0-1


def vector_to_grid(vec, height, width):
    """Convert flat vector back to grid"""
    arr = (vec * 9.0).round().clip(0, 9).astype(int)
    return arr.reshape(height, width)


def test_one_task():
    """Test if we can learn ONE task"""

    print("=" * 60)
    print("TESTING BABY ASI ON ONE ARC TASK")
    print("=" * 60)

    # Load the Center task we just explored
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_path = os.path.join(project_root, "arc_data", "corpus", "Center", "Center1.json")

    task = load_task(task_path)
    print(f"\n✓ Loaded task with {len(task['train'])} training examples")

    # Calculate max dimension from ALL examples (train + test)
    all_grids = []
    for example in task['train']:
        all_grids.append(example['input'])
        all_grids.append(example['output'])
    for example in task['test']:
        all_grids.append(example['input'])
        all_grids.append(example['output'])

    max_dim = max(len(g) * len(g[0]) for g in all_grids)
    print(f"Max grid dimension: {max_dim} pixels")

    # Prepare training data
    train_inputs = []
    train_outputs = []

    for example in task['train']:
        inp = grid_to_vector(example['input'])
        out = grid_to_vector(example['output'])
        train_inputs.append(inp)
        train_outputs.append(out)

    # Create model
    print(f"\n🧠 Creating ASISeed model...")
    model = ASISeed(
        input_dim=max_dim,
        model_dim=128,
        num_clusters=8,  # Small for this test
        core_rank=2,
        build_ema=False,
        use_heads=True  # We need prediction!
    ).to('cpu')

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Train on the examples
    print(f"\n🎓 Training on {len(train_inputs)} examples...")
    print("(This will take a moment...)")

    epochs = 500
    for epoch in range(epochs):
        total_loss = 0

        for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
            # Pad to max_dim if needed
            x = np.zeros(max_dim, dtype=np.float32)
            y = np.zeros(max_dim, dtype=np.float32)
            x[:len(inp)] = inp
            y[:len(out)] = out

            x_t = torch.from_numpy(x).to('cpu')
            y_t = torch.from_numpy(y).to('cpu')

            # Forward pass
            z = model.encoder(x_t)
            k = model.router(z)
            h = model.layer(z, active_cluster=k)

            # Reconstruct
            x_recon = model.decoder(h)

            # Loss
            loss = F.mse_loss(x_recon, y_t)  # Learn to produce OUTPUT not input!

            # Update
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update router
            with torch.no_grad():
                model.router.update_centroid(k, z)
            model.update_buffers(k, z)

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(train_inputs)
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

    print(f"\n✓ Training complete!")

    # Test on the test examples
    print(f"\n🧪 Testing on {len(task['test'])} test examples...")

    test_accuracies = []

    for test_idx, test_example in enumerate(task['test']):
        print(f"\n--- Test Example {test_idx + 1} ---")

        test_input = grid_to_vector(test_example['input'])
        test_output = grid_to_vector(test_example['output'])

        # Pad
        x = np.zeros(max_dim, dtype=np.float32)
        x[:len(test_input)] = test_input
        x_t = torch.from_numpy(x).to('cpu')

        # Predict
        with torch.no_grad():
            z = model.encoder(x_t)
            k = model.router(z)
            h = model.layer(z, active_cluster=k)
            pred = model.decoder(h)

        # Convert back to grid
        h, w = len(test_example['input']), len(test_example['input'][0])
        pred_vec = pred.cpu().numpy()[:h * w]
        pred_grid = vector_to_grid(pred_vec, h, w)
        true_grid = np.array(test_example['output'])

        # Check accuracy
        accuracy = (pred_grid == true_grid).mean() * 100
        test_accuracies.append(accuracy)
        print(f"Pixel accuracy: {accuracy:.1f}%")
        print(f"Used cluster: {k}")

        # Show a sample (only for small grids)
        if h <= 11 and w <= 11:
            print("\nPredicted (first few rows):")
            for row in pred_grid[:5]:
                print(' '.join(str(c) for c in row))
            print("\nExpected (first few rows):")
            for row in true_grid[:5]:
                print(' '.join(str(c) for c in row))

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    avg_accuracy = sum(test_accuracies) / len(test_accuracies)
    print(f"\nAverage test accuracy: {avg_accuracy:.1f}%")
    print(f"Random baseline: ~10%")

    if avg_accuracy > 50:
        print("✓✓✓ EXCELLENT! Model learned the pattern!")
    elif avg_accuracy > 20:
        print("✓✓ GOOD! Model learned something!")
    elif avg_accuracy > 10:
        print("✓ OKAY! Better than random!")
    else:
        print("⚠ Hmm, basically random. Need improvements!")

    print("\nWhat did we learn?")
    print("✓ Model can process grid data")
    print("✓ Model can learn input→output mappings")
    print(f"? Generalization: {avg_accuracy:.1f}% accuracy")
    print("\nNext steps:")
    print("- Add deep supervision (iteratively improve answer)")
    print("- Add recursive reasoning (like TRM paper)")
    print("- Try more tasks!")
    print("=" * 60)


if __name__ == "__main__":
    test_one_task()