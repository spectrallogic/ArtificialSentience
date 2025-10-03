# creativity_transfer_test.py
# Test: Does "forgetting" create better abstractions that enable creative transfer?
# Hypothesis: A model that abstracts (forgets details) can solve hybrid tasks better than one that memorizes

import torch
import torch.nn.functional as F
from asi_model import ASISeed, CuriosityStream, to_device


def create_task_stream(task_id, input_dim=32, seed=None):
    """Create a stream with distinct patterns for each task."""
    if seed is None:
        seed = 100 + task_id * 10
    return CuriosityStream(input_dim=input_dim, num_sources=3 + task_id, seed=seed)


def create_hybrid_stream(task_A_stream, task_B_stream, blend_ratio=0.5):
    """Create a hybrid stream that requires knowledge from BOTH tasks."""

    class HybridStream:
        def __init__(self, stream_a, stream_b, ratio):
            self.stream_a = stream_a
            self.stream_b = stream_b
            self.ratio = ratio

        def next(self):
            x_a = self.stream_a.next()
            x_b = self.stream_b.next()
            # Blend both tasks
            return self.ratio * x_a + (1 - self.ratio) * x_b

    return HybridStream(task_A_stream, task_B_stream, blend_ratio)


def evaluate_on_task(model, stream, num_samples=50):
    """Measure how well model performs on a task."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_samples):
            x = to_device(stream.next()).to(next(model.parameters()).device)
            x_hat, k, z, h = model(x)
            loss = F.mse_loss(x_hat, x)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_on_task(model, stream, steps=200, lr=1e-3, consolidate=True):
    """Train model on one task with optional consolidation."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device

    for step in range(steps):
        x = to_device(stream.next()).to(device)
        x_hat, k, z, h = model(x)
        loss = F.mse_loss(x_hat, x)

        # Update router and buffers
        with torch.no_grad():
            model.router.update_centroid(k, z)
        model.update_buffers(k, z)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Consolidate if requested
    if consolidate:
        active_clusters = [k for k in range(model.num_clusters) if len(model.buffers[k]) > 0]
        for k in active_clusters:
            model.consolidate_cluster(k)


def train_model_with_forgetting(device, input_dim=32):
    """Train a model that consolidates (abstracts/forgets details)."""
    model = ASISeed(input_dim=input_dim, model_dim=64, num_clusters=3,
                    core_rank=2, build_ema=False, use_heads=False).to(device)

    task_A = create_task_stream(0)
    task_B = create_task_stream(1)

    print("  Training with CONSOLIDATION (abstracts/forgets)...")
    train_on_task(model, task_A, steps=200, consolidate=True)
    train_on_task(model, task_B, steps=200, consolidate=True)

    return model, task_A, task_B


def train_model_without_forgetting(device, input_dim=32):
    """Train a model WITHOUT consolidation (tries to remember everything)."""
    model = ASISeed(input_dim=input_dim, model_dim=64, num_clusters=3,
                    core_rank=2, build_ema=False, use_heads=False).to(device)

    task_A = create_task_stream(0)
    task_B = create_task_stream(1)

    print("  Training WITHOUT consolidation (memorizes details)...")
    train_on_task(model, task_A, steps=200, consolidate=False)
    train_on_task(model, task_B, steps=200, consolidate=False)

    return model, task_A, task_B


def main():
    print("=" * 70)
    print("CREATIVITY & TRANSFER TEST")
    print("Hypothesis: Forgetting details → Better abstractions → Better transfer")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ==========================
    # Train TWO models
    # ==========================
    print("\n" + "=" * 70)
    print("TRAINING MODEL 1: With Consolidation (Your Approach)")
    print("=" * 70)
    model_forgets, task_A1, task_B1 = train_model_with_forgetting(device)

    print("\n" + "=" * 70)
    print("TRAINING MODEL 2: Without Consolidation (Memorization)")
    print("=" * 70)
    model_memorizes, task_A2, task_B2 = train_model_without_forgetting(device)

    # ==========================
    # Test on ORIGINAL tasks
    # ==========================
    print("\n" + "=" * 70)
    print("PHASE 1: Performance on Original Tasks")
    print("=" * 70)

    print("\nModel 1 (Consolidation):")
    perf_1_A = evaluate_on_task(model_forgets, task_A1)
    perf_1_B = evaluate_on_task(model_forgets, task_B1)
    print(f"  Task A: {perf_1_A:.4f}")
    print(f"  Task B: {perf_1_B:.4f}")
    print(f"  Average: {(perf_1_A + perf_1_B) / 2:.4f}")

    print("\nModel 2 (No Consolidation):")
    perf_2_A = evaluate_on_task(model_memorizes, task_A2)
    perf_2_B = evaluate_on_task(model_memorizes, task_B2)
    print(f"  Task A: {perf_2_A:.4f}")
    print(f"  Task B: {perf_2_B:.4f}")
    print(f"  Average: {(perf_2_A + perf_2_B) / 2:.4f}")

    # ==========================
    # Test on HYBRID tasks (CREATIVITY TEST)
    # ==========================
    print("\n" + "=" * 70)
    print("PHASE 2: Transfer to NOVEL Hybrid Tasks")
    print("(These blend Task A + Task B in ways the model never saw)")
    print("=" * 70)

    # Create multiple hybrid ratios
    hybrid_ratios = [0.25, 0.5, 0.75]

    print("\nModel 1 (Consolidation) on Hybrids:")
    hybrid_scores_1 = []
    for ratio in hybrid_ratios:
        hybrid = create_hybrid_stream(task_A1, task_B1, ratio)
        score = evaluate_on_task(model_forgets, hybrid, num_samples=30)
        hybrid_scores_1.append(score)
        print(f"  {int(ratio * 100)}% A + {int((1 - ratio) * 100)}% B: {score:.4f}")
    avg_hybrid_1 = sum(hybrid_scores_1) / len(hybrid_scores_1)
    print(f"  Average Hybrid: {avg_hybrid_1:.4f}")

    print("\nModel 2 (No Consolidation) on Hybrids:")
    hybrid_scores_2 = []
    for ratio in hybrid_ratios:
        hybrid = create_hybrid_stream(task_A2, task_B2, ratio)
        score = evaluate_on_task(model_memorizes, hybrid, num_samples=30)
        hybrid_scores_2.append(score)
        print(f"  {int(ratio * 100)}% A + {int((1 - ratio) * 100)}% B: {score:.4f}")
    avg_hybrid_2 = sum(hybrid_scores_2) / len(hybrid_scores_2)
    print(f"  Average Hybrid: {avg_hybrid_2:.4f}")

    # ==========================
    # Test on NOVEL task
    # ==========================
    print("\n" + "=" * 70)
    print("PHASE 3: Zero-Shot Transfer to Completely Novel Task")
    print("(Task C - never seen before)")
    print("=" * 70)

    task_C_stream = create_task_stream(2)  # Brand new task

    perf_1_C = evaluate_on_task(model_forgets, task_C_stream)
    perf_2_C = evaluate_on_task(model_memorizes, task_C_stream)

    print(f"\nModel 1 (Consolidation) on Task C: {perf_1_C:.4f}")
    print(f"Model 2 (No Consolidation) on Task C: {perf_2_C:.4f}")

    # ==========================
    # ANALYSIS
    # ==========================
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    # Calculate metrics
    original_task_gap = ((perf_1_A + perf_1_B) - (perf_2_A + perf_2_B)) / 2
    hybrid_gap = avg_hybrid_1 - avg_hybrid_2
    novel_gap = perf_1_C - perf_2_C

    print("\n📊 Performance Comparison:")
    print(f"  Original Tasks (A+B avg):")
    print(f"    Consolidation:     {(perf_1_A + perf_1_B) / 2:.4f}")
    print(f"    No Consolidation:  {(perf_2_A + perf_2_B) / 2:.4f}")
    print(
        f"    Gap: {original_task_gap:+.4f} {'(consolidation worse)' if original_task_gap > 0 else '(consolidation better!)'}")

    print(f"\n  Hybrid Tasks (creativity):")
    print(f"    Consolidation:     {avg_hybrid_1:.4f}")
    print(f"    No Consolidation:  {avg_hybrid_2:.4f}")
    print(f"    Gap: {hybrid_gap:+.4f} {'(consolidation worse)' if hybrid_gap > 0 else '(consolidation better!)'}")

    print(f"\n  Novel Task C (transfer):")
    print(f"    Consolidation:     {perf_1_C:.4f}")
    print(f"    No Consolidation:  {perf_2_C:.4f}")
    print(f"    Gap: {novel_gap:+.4f} {'(consolidation worse)' if novel_gap > 0 else '(consolidation better!)'}")

    # ==========================
    # VERDICT
    # ==========================
    print("\n" + "=" * 70)
    print("VERDICT: Is 'Forgetting' Actually Creative?")
    print("=" * 70)

    creativity_wins = 0
    total_tests = 3

    # Test 1: Hybrids
    if hybrid_gap < 0:  # Lower is better
        creativity_wins += 1
        print("\n✓ TEST 1 PASS: Consolidation handles hybrid tasks BETTER")
        print("  → Abstraction enables creative blending!")
    else:
        print("\n✗ TEST 1 FAIL: Consolidation handles hybrid tasks WORSE")
        print("  → Forgetting details hurt, didn't help")

    # Test 2: Novel transfer
    if novel_gap < 0:
        creativity_wins += 1
        print("\n✓ TEST 2 PASS: Consolidation transfers to novel task BETTER")
        print("  → Abstract knowledge generalizes!")
    else:
        print("\n✗ TEST 2 FAIL: Consolidation transfers to novel task WORSE")
        print("  → Memorization actually transferred better")

    # Test 3: Not TOO much worse on originals
    if original_task_gap < 0.005:  # Within 0.005 tolerance
        creativity_wins += 1
        print("\n✓ TEST 3 PASS: Consolidation maintains original performance")
        print("  → Abstraction didn't sacrifice too much")
    else:
        print("\n✗ TEST 3 FAIL: Consolidation hurt original tasks significantly")
        print("  → Too much forgetting, not enough retention")

    # Final verdict
    print("\n" + "=" * 70)
    if creativity_wins >= 2:
        print("🎉 HYPOTHESIS SUPPORTED!")
        print("   Your 'forgetting for creativity' approach shows promise!")
        print("   Consolidation creates abstractions that:")
        if hybrid_gap < 0:
            print("   ✓ Enable creative blending of concepts")
        if novel_gap < 0:
            print("   ✓ Transfer better to novel situations")
        print("\n   This IS potentially unique and valuable!")
        print("   Next steps: Scale this up and publish the findings.")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED")
        print("   Consolidation didn't show clear creativity benefits.")
        print("   The 'forgetting' hurt more than it helped.")
        print("\n   Consider:")
        print("   - Tuning consolidation hyperparameters")
        print("   - Adding selective forgetting (keep important, forget noise)")
        print("   - Or pivoting to traditional anti-forgetting methods")

    print("=" * 70)


if __name__ == "__main__":
    main()

'''
======================================================================
CREATIVITY & TRANSFER TEST
Hypothesis: Forgetting details → Better abstractions → Better transfer
======================================================================

Device: cpu

======================================================================
TRAINING MODEL 1: With Consolidation (Your Approach)
======================================================================
  Training with CONSOLIDATION (abstracts/forgets)...

======================================================================
TRAINING MODEL 2: Without Consolidation (Memorization)
======================================================================
  Training WITHOUT consolidation (memorizes details)...

======================================================================
PHASE 1: Performance on Original Tasks
======================================================================

Model 1 (Consolidation):
  Task A: 0.0157
  Task B: 0.0128
  Average: 0.0142

Model 2 (No Consolidation):
  Task A: 0.0147
  Task B: 0.0123
  Average: 0.0135

======================================================================
PHASE 2: Transfer to NOVEL Hybrid Tasks
(These blend Task A + Task B in ways the model never saw)
======================================================================

Model 1 (Consolidation) on Hybrids:
  25% A + 75% B: 0.0093
  50% A + 50% B: 0.0078
  75% A + 25% B: 0.0092
  Average Hybrid: 0.0088

Model 2 (No Consolidation) on Hybrids:
  25% A + 75% B: 0.0092
  50% A + 50% B: 0.0071
  75% A + 25% B: 0.0093
  Average Hybrid: 0.0085

======================================================================
PHASE 3: Zero-Shot Transfer to Completely Novel Task
(Task C - never seen before)
======================================================================

Model 1 (Consolidation) on Task C: 0.0307
Model 2 (No Consolidation) on Task C: 0.0298

======================================================================
RESULTS ANALYSIS
======================================================================

📊 Performance Comparison:
  Original Tasks (A+B avg):
    Consolidation:     0.0142
    No Consolidation:  0.0135
    Gap: +0.0007 (consolidation worse)

  Hybrid Tasks (creativity):
    Consolidation:     0.0088
    No Consolidation:  0.0085
    Gap: +0.0002 (consolidation worse)

  Novel Task C (transfer):
    Consolidation:     0.0307
    No Consolidation:  0.0298
    Gap: +0.0009 (consolidation worse)

======================================================================
VERDICT: Is 'Forgetting' Actually Creative?
======================================================================

✗ TEST 1 FAIL: Consolidation handles hybrid tasks WORSE
  → Forgetting details hurt, didn't help

✗ TEST 2 FAIL: Consolidation transfers to novel task WORSE
  → Memorization actually transferred better

✓ TEST 3 PASS: Consolidation maintains original performance
  → Abstraction didn't sacrifice too much

======================================================================
❌ HYPOTHESIS NOT SUPPORTED
   Consolidation didn't show clear creativity benefits.
   The 'forgetting' hurt more than it helped.

   Consider:
   - Tuning consolidation hyperparameters
   - Adding selective forgetting (keep important, forget noise)
   - Or pivoting to traditional anti-forgetting methods
======================================================================

'''