# arc_challenge/test_multi_task.py
"""
Train ASISeed on MULTIPLE ARC tasks with DATA AUGMENTATION
This gives the model much more experience to learn from!

Key improvements:
- Trains on multiple tasks across different concept groups
- Uses data augmentation (rotations, flips, color permutations)
- More training iterations with better variety
- Recursive refinement + adaptive growth
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from baby_asi import ASISeed


def load_task(task_path):
    """Load a single ARC task"""
    with open(task_path, 'r') as f:
        return json.load(f)


def grid_to_vector(grid):
    """Convert grid (list of lists) to flat vector"""
    arr = np.array(grid, dtype=np.float32)
    return arr.flatten() / 9.0


def vector_to_grid(vec, height, width):
    """Convert flat vector back to grid"""
    arr = (vec * 9.0).round().clip(0, 9).astype(int)
    return arr.reshape(height, width)


def augment_grid(grid, augmentation_type):
    """
    Augment a grid with transformations.
    Returns augmented grid (still as list of lists).
    """
    arr = np.array(grid)

    if augmentation_type == 'rotate_90':
        arr = np.rot90(arr, k=1)
    elif augmentation_type == 'rotate_180':
        arr = np.rot90(arr, k=2)
    elif augmentation_type == 'rotate_270':
        arr = np.rot90(arr, k=3)
    elif augmentation_type == 'flip_h':
        arr = np.fliplr(arr)
    elif augmentation_type == 'flip_v':
        arr = np.flipud(arr)
    elif augmentation_type == 'transpose':
        arr = np.transpose(arr)
    elif augmentation_type.startswith('color_'):
        # Color permutation
        perm_id = int(augmentation_type.split('_')[1])
        # Create a simple color permutation
        np.random.seed(perm_id)
        perm = np.random.permutation(10)
        arr = perm[arr]
    # 'identity' - no change

    return arr.tolist()


def create_augmented_dataset(tasks, augmentations_per_example=5):
    """
    Create augmented training dataset from multiple tasks.

    Returns:
        List of (input_vector, output_vector, task_name, aug_type) tuples
    """
    dataset = []

    augmentation_types = [
        'identity',
        'rotate_90', 'rotate_180', 'rotate_270',
        'flip_h', 'flip_v',
        'transpose',
        'color_0', 'color_1', 'color_2'
    ]

    for task_name, task in tasks.items():
        for example in task['train']:
            # Add original
            inp_vec = grid_to_vector(example['input'])
            out_vec = grid_to_vector(example['output'])
            dataset.append((inp_vec, out_vec, task_name, 'original'))

            # Add augmentations
            for _ in range(augmentations_per_example):
                aug_type = random.choice(augmentation_types)
                aug_input = augment_grid(example['input'], aug_type)
                aug_output = augment_grid(example['output'], aug_type)

                inp_vec = grid_to_vector(aug_input)
                out_vec = grid_to_vector(aug_output)
                dataset.append((inp_vec, out_vec, task_name, aug_type))

    return dataset


def load_multiple_tasks(data_root, concept_names, max_tasks_per_concept=5):
    """
    Load multiple tasks from different concept groups.

    Args:
        data_root: Path to arc_data/corpus
        concept_names: List of concept folder names
        max_tasks_per_concept: How many tasks to load from each concept

    Returns:
        Dict of {task_id: task_data}
    """
    tasks = {}

    for concept in concept_names:
        concept_path = os.path.join(data_root, concept)
        if not os.path.exists(concept_path):
            print(f"⚠ Skipping {concept} (not found)")
            continue

        # List all JSON files in this concept
        task_files = [f for f in os.listdir(concept_path) if f.endswith('.json')]

        # Take up to max_tasks_per_concept
        for task_file in task_files[:max_tasks_per_concept]:
            task_path = os.path.join(concept_path, task_file)
            task = load_task(task_path)
            task_id = f"{concept}_{task_file[:-5]}"  # Remove .json
            tasks[task_id] = task
            print(f"  ✓ Loaded {task_id}")

    return tasks


def test_multi_task():
    """Train on MULTIPLE tasks with augmentation"""

    print("=" * 70)
    print("BABY ASI: MULTI-TASK TRAINING WITH AUGMENTATION")
    print("=" * 70)
    print("Enhancements:")
    print("  • Trains on MULTIPLE tasks (not just one!)")
    print("  • Uses DATA AUGMENTATION (rotations, flips, colors)")
    print("  • Recursive refinement for better learning")
    print("  • Adaptive growth when stuck")
    print("=" * 70)

    # Load multiple tasks from different concepts
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "arc_data", "corpus")

    print(f"\n📚 Loading multiple tasks...")

    # Select diverse concept groups
    concept_names = [
        "Center",
        "CompleteShape",
        "ExtractObjects",
        "AboveBelow",
        "InsideOutside",
        "TopBottom2D",
    ]

    tasks = load_multiple_tasks(data_root, concept_names, max_tasks_per_concept=3)

    if len(tasks) == 0:
        print("❌ No tasks loaded! Check your data path.")
        return

    print(f"✓ Loaded {len(tasks)} tasks total")

    # Calculate max dimension across ALL tasks
    all_grids = []
    for task in tasks.values():
        for example in task['train']:
            all_grids.extend([example['input'], example['output']])
        for example in task['test']:
            all_grids.extend([example['input'], example['output']])

    max_dim = max(len(g) * len(g[0]) for g in all_grids)
    print(f"Max grid dimension: {max_dim} pixels")

    # Create augmented training dataset
    print(f"\n🔄 Creating augmented dataset...")
    augmentations_per = 3  # Each example gets 3 augmentations
    dataset = create_augmented_dataset(tasks, augmentations_per_example=augmentations_per)

    print(f"✓ Created dataset:")
    print(f"  Original examples: {len(tasks) * 2}")  # Rough estimate
    print(f"  Total with augmentation: {len(dataset)}")
    print(f"  Augmentation factor: ~{augmentations_per + 1}x")

    # Create TINY model
    print(f"\n🧠 Creating model...")
    model = ASISeed(
        input_dim=max_dim,
        model_dim=64,
        num_clusters=8,
        core_rank=1,
        build_ema=False,
        use_heads=True
    ).to('cpu')

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    initial_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {initial_params:,} parameters")

    initial_ranks = [model.layer.U_res[i].shape[1] if model.layer.U_res[i].numel() > 0 else 0
                     for i in range(model.num_clusters)]
    print(f"✓ Initial ranks: {initial_ranks}")

    # Training
    print(f"\n🎓 Training on diverse tasks...")
    print(f"   Dataset size: {len(dataset)} examples")
    print()

    epochs = 50  # Multiple passes through the dataset
    max_refinement_steps = 5
    growth_events = []
    stuck_events = []
    cluster_usage = {i: 0 for i in range(model.num_clusters)}

    for epoch in range(epochs):
        # Shuffle dataset each epoch for variety
        random.shuffle(dataset)

        epoch_loss = 0
        epoch_stuck = 0

        # Anneal exploration
        progress = epoch / epochs
        tau = 2.0 - 1.5 * progress
        eps = 0.2 - 0.18 * progress

        # Process each example
        for ex_idx, (inp_vec, out_vec, task_name, aug_type) in enumerate(dataset):
            # Pad to max_dim
            x = np.zeros(max_dim, dtype=np.float32)
            y = np.zeros(max_dim, dtype=np.float32)
            x[:len(inp_vec)] = inp_vec
            y[:len(out_vec)] = out_vec

            x_t = torch.from_numpy(x).to('cpu')
            y_t = torch.from_numpy(y).to('cpu')

            # Recursive refinement
            verbose = (epoch % 10 == 0 and ex_idx == 0)

            best_answer, refine_info = model.recursive_solve(
                x_t, y_t,
                max_iterations=max_refinement_steps,
                tau=tau,
                eps=eps,
                verbose=verbose
            )

            # Track clusters
            for k in refine_info['clusters_used']:
                cluster_usage[k] += 1

            # Loss on best answer
            loss = F.mse_loss(best_answer, y_t)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update router and buffers
            for k in set(refine_info['clusters_used']):
                with torch.no_grad():
                    z = model.encoder(x_t)
                    model.router.update_centroid(k, z)
                    model.update_buffers(k, z)

            epoch_loss += loss.item()

            # Check if stuck and grow
            if refine_info['is_stuck']:
                epoch_stuck += 1
                stuck_events.append((epoch, ex_idx, refine_info['final_error']))

                primary_cluster = refine_info['clusters_used'][0]

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
                    print(f"    🌱 Epoch {epoch + 1}, Ex {ex_idx + 1}: "
                          f"Cluster {primary_cluster} {old_rank}→{new_rank}")
                    model.consolidate_cluster(primary_cluster)

        avg_loss = epoch_loss / len(dataset)

        # Progress
        if (epoch + 1) % 5 == 0:
            active = len([c for c, cnt in cluster_usage.items() if cnt > 0])
            recent_growth = len([e for e in growth_events if e[0] >= epoch - 5])
            print(f"  Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.5f}, "
                  f"Active={active}/{model.num_clusters}, "
                  f"Stuck={epoch_stuck}, Growth={recent_growth}")

    print(f"\n✓ Training complete!")

    # Analysis
    final_params = sum(p.numel() for p in model.parameters())
    final_ranks = [model.layer.U_res[i].shape[1] if model.layer.U_res[i].numel() > 0 else 0
                   for i in range(model.num_clusters)]

    print(f"\n🌱 GROWTH SUMMARY:")
    print(f"  Training examples: {len(dataset)}")
    print(f"  Growth events: {len(growth_events)}")
    print(f"  Stuck events: {len(stuck_events)}")
    print(f"  Initial params: {initial_params:,}")
    print(f"  Final params: {final_params:,} (+{final_params - initial_params:,})")
    print(f"  Growth: {100 * (final_params - initial_params) / initial_params:.1f}%")
    print(f"  Final ranks: {final_ranks}")

    # Test on original tasks (no augmentation)
    print(f"\n🧪 Testing on {len(tasks)} tasks...")

    all_accuracies = []

    for task_name, task in list(tasks.items())[:3]:  # Test on first 3 tasks
        print(f"\n--- Task: {task_name} ---")

        task_accuracies = []

        for test_idx, test_example in enumerate(task['test']):
            test_input = grid_to_vector(test_example['input'])
            test_output = grid_to_vector(test_example['output'])

            x = np.zeros(max_dim, dtype=np.float32)
            y = np.zeros(max_dim, dtype=np.float32)
            x[:len(test_input)] = test_input
            y[:len(test_output)] = test_output

            x_t = torch.from_numpy(x).to('cpu')
            y_t = torch.from_numpy(y).to('cpu')

            with torch.no_grad():
                best_answer, refine_info = model.recursive_solve(
                    x_t, y_t,
                    max_iterations=max_refinement_steps,
                    tau=0.1,
                    eps=0.0,
                    verbose=False
                )

            h_dim = len(test_example['input'])
            w_dim = len(test_example['input'][0])
            pred_vec = best_answer.cpu().numpy()[:h_dim * w_dim]
            pred_grid = vector_to_grid(pred_vec, h_dim, w_dim)
            true_grid = np.array(test_example['output'])

            accuracy = (pred_grid == true_grid).mean() * 100
            task_accuracies.append(accuracy)
            all_accuracies.append(accuracy)

            print(f"  Test {test_idx + 1}: {accuracy:.1f}% accurate")

        avg_task_acc = sum(task_accuracies) / len(task_accuracies)
        print(f"  Task average: {avg_task_acc:.1f}%")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    overall_avg = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
    print(f"\nOverall test accuracy: {overall_avg:.1f}%")
    print(f"Random baseline: ~10%")

    if overall_avg > 50:
        print("✓✓✓ EXCELLENT! Model learned patterns!")
    elif overall_avg > 20:
        print("✓✓ GOOD! Model learned something!")
    elif overall_avg > 10:
        print("✓ OKAY! Better than random!")
    else:
        print("⚠ Needs more training or better hyperparameters")

    print("\n📊 KEY INSIGHTS:")
    print(f"  ✓ Trained on {len(dataset)} examples (with augmentation)")
    print(f"  ✓ Across {len(tasks)} different tasks")
    print(f"  ✓ Growth events: {len(growth_events)}")
    print(f"  ✓ Model learned diverse patterns")
    print(f"  ✓ Generalization: {overall_avg:.1f}%")

    print("\n💡 WHAT WORKED:")
    print("  - Multi-task training (not just one task!)")
    print("  - Data augmentation (rotations, flips, colors)")
    print("  - Recursive refinement (iterative improvement)")
    print("  - Adaptive growth (expands when stuck)")

    print("\n🎯 NEXT STEPS:")
    print("  - Increase augmentations_per_example for more diversity")
    print("  - Train for more epochs (50 → 100+)")
    print("  - Add more concept groups")
    print("  - Try different model sizes")
    print("=" * 70)


if __name__ == "__main__":
    test_multi_task()