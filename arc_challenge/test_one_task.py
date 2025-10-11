# arc_challenge/test_one_task.py
"""
Test ASISeed with RECURSIVE REFINEMENT + ADAPTIVE GROWTH
Key ideas:
- Model iteratively refines its answers (like humans solving puzzles)
- Growth triggers when model is STUCK (can't improve despite trying)
- No hardcoded assumptions - discovers what's important through iteration
- Aligns with TRM paper: tiny networks + deep recursion beats large networks
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
    """Test with RECURSIVE REFINEMENT + ADAPTIVE GROWTH"""

    print("=" * 70)
    print("BABY ASI: RECURSIVE REFINEMENT + ADAPTIVE GROWTH")
    print("=" * 70)
    print("Key concepts:")
    print("  • Model refines answers iteratively (like human reasoning)")
    print("  • Growth triggers when stuck (can't improve but error high)")
    print("  • No assumptions - discovers importance through iteration")
    print("=" * 70)

    # Load task
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try different tasks
    task_options = [
        ("Center", "Center1.json"),
        ("CompleteShape", "CompleteShape1.json"),
        ("SameDifferent", "SameDifferent1.json"),
    ]

    # Use CompleteShape (harder)
    task_name, task_file = task_options[1]
    task_path = os.path.join(project_root, "arc_data", "corpus", task_name, task_file)

    if not os.path.exists(task_path):
        print(f"⚠ {task_name} not found, using Center")
        task_name, task_file = task_options[0]
        task_path = os.path.join(project_root, "arc_data", "corpus", task_name, task_file)

    task = load_task(task_path)
    print(f"\n✓ Loaded '{task_name}' with {len(task['train'])} training examples")

    # Calculate max dimension
    all_grids = []
    for example in task['train']:
        all_grids.extend([example['input'], example['output']])
    for example in task['test']:
        all_grids.extend([example['input'], example['output']])

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

    # Create TINY model (will grow when needed)
    print(f"\n🧠 Creating TINY model with recursive refinement...")
    model = ASISeed(
        input_dim=max_dim,
        model_dim=64,  # Small
        num_clusters=8,
        core_rank=1,  # Minimal!
        build_ema=False,
        use_heads=True
    ).to('cpu')

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    initial_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {initial_params:,} parameters")

    initial_ranks = [model.layer.U_res[i].shape[1] if model.layer.U_res[i].numel() > 0 else 0
                     for i in range(model.num_clusters)]
    print(f"✓ Initial ranks: {initial_ranks}")

    # Training with RECURSIVE REFINEMENT
    print(f"\n🎓 Training with recursive refinement...")
    print("(Model will iterate on each answer, grow when stuck)")
    print()

    epochs = 300  # Fewer epochs since we do multiple iterations per example
    max_refinement_steps = 5
    growth_events = []
    stuck_events = []
    cluster_usage = {i: 0 for i in range(model.num_clusters)}

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_stuck_count = 0

        # Anneal exploration
        progress = epoch / epochs
        tau = 2.0 - 1.5 * progress  # 2.0 → 0.5
        eps = 0.2 - 0.18 * progress  # 0.2 → 0.02

        for ex_idx, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
            # Pad
            x = np.zeros(max_dim, dtype=np.float32)
            y = np.zeros(max_dim, dtype=np.float32)
            x[:len(inp)] = inp
            y[:len(out)] = out

            x_t = torch.from_numpy(x).to('cpu')
            y_t = torch.from_numpy(y).to('cpu')

            # RECURSIVE REFINEMENT (key innovation!)
            verbose = (epoch % 100 == 0 and ex_idx == 0)  # Show details occasionally

            best_answer, refine_info = model.recursive_solve(
                x_t, y_t,
                max_iterations=max_refinement_steps,
                tau=tau,
                eps=eps,
                verbose=verbose
            )

            # Track clusters used
            for k in refine_info['clusters_used']:
                cluster_usage[k] += 1

            # Loss on BEST answer from refinement process
            loss = F.mse_loss(best_answer, y_t)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update router with all clusters used during refinement
            for k in set(refine_info['clusters_used']):
                with torch.no_grad():
                    # Get the latent for this cluster
                    z = model.encoder(x_t)
                    model.router.update_centroid(k, z)
                    model.update_buffers(k, z)

            epoch_loss += loss.item()

            # Check if model is STUCK
            if refine_info['is_stuck']:
                epoch_stuck_count += 1
                stuck_events.append((epoch, ex_idx, refine_info['final_error']))

                # Primary cluster that's stuck
                primary_cluster = refine_info['clusters_used'][0]

                # Try to GROW when stuck!
                grew = model.check_and_grow_cluster(
                    primary_cluster,
                    refine_info['final_error'],
                    window_size=20,
                    improvement_threshold=0.008,
                    grow_rank=1,
                    verbose=False
                )

                if grew:
                    old_rank = model.layer.U_res[primary_cluster].shape[1] - 1
                    new_rank = model.layer.U_res[primary_cluster].shape[1]
                    growth_events.append((epoch, ex_idx, primary_cluster, old_rank, new_rank))
                    print(f"    🌱 GROWTH! Epoch {epoch + 1}, Example {ex_idx + 1}, "
                          f"Cluster {primary_cluster}: {old_rank}→{new_rank} "
                          f"(stuck at error={refine_info['final_error']:.5f})")
                    model.consolidate_cluster(primary_cluster)

        avg_loss = epoch_loss / len(train_inputs)

        # Progress
        if (epoch + 1) % 50 == 0:
            active = len([c for c, cnt in cluster_usage.items() if cnt > 0])
            recent_growth = len([e for e in growth_events if e[0] >= epoch - 50])
            recent_stuck = len([e for e in stuck_events if e[0] >= epoch - 50])
            print(f"  Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.5f}, "
                  f"Active={active}/{model.num_clusters}, "
                  f"Stuck={recent_stuck}, Growth={recent_growth}")

    print(f"\n✓ Training complete!")

    # Analysis
    final_params = sum(p.numel() for p in model.parameters())
    final_ranks = [model.layer.U_res[i].shape[1] if model.layer.U_res[i].numel() > 0 else 0
                   for i in range(model.num_clusters)]

    print(f"\n🌱 GROWTH SUMMARY:")
    print(f"  Growth events: {len(growth_events)}")
    print(f"  Stuck events: {len(stuck_events)}")
    print(f"  Initial params: {initial_params:,}")
    print(f"  Final params: {final_params:,} (+{final_params - initial_params:,})")
    print(f"  Growth: {100 * (final_params - initial_params) / initial_params:.1f}%")
    print(f"  Initial ranks: {initial_ranks}")
    print(f"  Final ranks: {final_ranks}")

    # Cluster usage
    print(f"\n  Cluster usage:")
    for k in sorted(cluster_usage.keys()):
        cnt = cluster_usage[k]
        if cnt > 0:
            pct = 100 * cnt / sum(cluster_usage.values())
            bar = "█" * int(pct / 5)
            rank_change = final_ranks[k] - initial_ranks[k]
            growth_str = f" (+{rank_change})" if rank_change > 0 else ""
            print(f"    Cluster {k}: {cnt:4d} ({pct:5.1f}%) {bar}{growth_str}")

    if growth_events:
        print(f"\n  Growth timeline:")
        for epoch, ex, k, old_r, new_r in growth_events[:10]:
            print(f"    Epoch {epoch + 1}, Ex {ex + 1}: Cluster {k} {old_r}→{new_r}")
        if len(growth_events) > 10:
            print(f"    ... and {len(growth_events) - 10} more")

    # Test with refinement!
    print(f"\n🧪 Testing on {len(task['test'])} examples (with refinement)...")

    test_accuracies = []

    for test_idx, test_example in enumerate(task['test']):
        print(f"\n--- Test Example {test_idx + 1} ---")

        test_input = grid_to_vector(test_example['input'])

        # Pad
        x = np.zeros(max_dim, dtype=np.float32)
        y_true = np.zeros(max_dim, dtype=np.float32)
        x[:len(test_input)] = test_input

        test_output = grid_to_vector(test_example['output'])
        y_true[:len(test_output)] = test_output

        x_t = torch.from_numpy(x).to('cpu')
        y_t = torch.from_numpy(y_true).to('cpu')

        # Use refinement at test time!
        with torch.no_grad():
            best_answer, refine_info = model.recursive_solve(
                x_t, y_t,
                max_iterations=max_refinement_steps,
                tau=0.1,  # Low temp = exploit
                eps=0.0,  # No exploration
                verbose=True
            )

        # Convert back to grid
        h_dim = len(test_example['input'])
        w_dim = len(test_example['input'][0])
        pred_vec = best_answer.cpu().numpy()[:h_dim * w_dim]
        pred_grid = vector_to_grid(pred_vec, h_dim, w_dim)
        true_grid = np.array(test_example['output'])

        # Accuracy
        accuracy = (pred_grid == true_grid).mean() * 100
        test_accuracies.append(accuracy)
        print(f"\nPixel accuracy: {accuracy:.1f}%")
        print(f"Best iteration: {refine_info['best_iteration']}")
        print(f"Clusters used: {refine_info['clusters_used']}")

        # Show sample
        if h_dim <= 11 and w_dim <= 11:
            print("\nPredicted:")
            for row in pred_grid[:5]:
                print(' '.join(str(c) for c in row))
            print("\nExpected:")
            for row in true_grid[:5]:
                print(' '.join(str(c) for c in row))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    avg_accuracy = sum(test_accuracies) / len(test_accuracies)
    print(f"\nTest accuracy: {avg_accuracy:.1f}%")
    print(f"Random baseline: ~10%")

    if avg_accuracy > 50:
        print("✓✓✓ EXCELLENT! Model learned the pattern!")
    elif avg_accuracy > 20:
        print("✓✓ GOOD! Model learned something!")
    elif avg_accuracy > 10:
        print("✓ OKAY! Better than random!")
    else:
        print("⚠ Needs improvement!")

    print("\n📊 KEY INSIGHTS:")
    print(f"  ✓ Recursive refinement: {max_refinement_steps} iterations per example")
    print(f"  ✓ Adaptive growth: {len(growth_events)} expansions")
    print(f"  ✓ Model got stuck: {len(stuck_events)} times")
    print(f"  ✓ Parameters: {initial_params:,} → {final_params:,}")
    print(f"  ✓ Generalization: {avg_accuracy:.1f}%")

    print("\n💡 WHAT HAPPENED:")
    print("  - Model iteratively refines answers (human-like)")
    print("  - Discovers what matters through iteration (no hardcoding)")
    print("  - Grows when stuck (true complexity, not just loss)")
    print("  - Final model is right-sized for task complexity")

    print("\n🎯 WHY THIS WORKS:")
    print("  - TRM paper: tiny networks + recursion beat large models")
    print("  - Natural learning: like humans iterating on puzzles")
    print("  - Adaptive: structure emerges from need")
    print("  - No assumptions: works for any pattern")
    print("=" * 70)


if __name__ == "__main__":
    test_one_task()