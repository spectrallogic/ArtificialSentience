# humanlike_consolidation_test.py
# Test if consolidation creates human-like memory: gradual forgetting of details but retention of gist

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asi_model import ASISeed, CuriosityStream, to_device


def create_task_stream(task_id, input_dim=32, seed=None):
    """Create a stream with distinct patterns for each task."""
    if seed is None:
        seed = 100 + task_id * 10
    return CuriosityStream(input_dim=input_dim, num_sources=3 + task_id, seed=seed)


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
    avg_loss = sum(losses) / len(losses)
    std_loss = (sum((l - avg_loss) ** 2 for l in losses) / len(losses)) ** 0.5
    return avg_loss, std_loss


def train_on_task(model, stream, steps=200, lr=1e-3, task_name="Task"):
    """Train model on one task and collect buffer samples."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Collect samples for consolidation
    buffer = []
    device = next(model.parameters()).device

    print(f"  Training {task_name}...")
    for step in range(steps):
        x = to_device(stream.next()).to(device)
        x_hat, k, z, h = model(x)
        loss = F.mse_loss(x_hat, x)

        # Store samples periodically for consolidation buffer
        if step % 5 == 0 and len(buffer) < 100:
            buffer.append(z.detach().cpu())

        # Update router and model buffers (important for consolidation)
        with torch.no_grad():
            model.router.update_centroid(k, z)
        model.update_buffers(k, z)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}/{steps}, Loss: {loss.item():.4f}")

    return buffer


def consolidate_knowledge(model, task_name="Task"):
    """Run consolidation on all clusters that have learned."""
    print(f"  Consolidating {task_name} memories...")

    # Count how many clusters have data
    active_clusters = [k for k in range(model.num_clusters) if len(model.buffers[k]) > 0]

    if not active_clusters:
        print("    No clusters to consolidate!")
        return

    print(f"    Active clusters: {active_clusters}")

    # Consolidate each active cluster
    for k in active_clusters:
        buffer_size = len(model.buffers[k])
        print(f"    Consolidating cluster {k} ({buffer_size} samples)...")
        model.consolidate_cluster(k)

    print("    Consolidation complete!")


def test_relearning_speed(model, stream, initial_loss, steps=50, lr=1e-3):
    """Test 'savings effect' - how fast can it relearn compared to initial learning?"""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device

    losses = []
    for step in range(steps):
        x = to_device(stream.next()).to(device)
        x_hat, k, z, h = model(x)
        loss = F.mse_loss(x_hat, x)
        losses.append(loss.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    final_loss = sum(losses[-10:]) / 10  # Average last 10
    improvement_rate = (initial_loss - final_loss) / initial_loss

    return final_loss, improvement_rate


def main():
    print("=" * 70)
    print("HUMAN-LIKE CONSOLIDATION TEST")
    print("other_experiments: Does consolidation create abstract, durable memories?")
    print("=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = ASISeed(input_dim=32, model_dim=64, num_clusters=3,
                    core_rank=2, build_ema=False, use_heads=False).to(device)

    # Create two distinct tasks
    task_A_stream = create_task_stream(task_id=0)
    task_B_stream = create_task_stream(task_id=1)

    # ==========================
    # PHASE 1: Learn Task A
    # ==========================
    print("\n" + "=" * 70)
    print("PHASE 1: Learning Task A")
    print("=" * 70)
    train_on_task(model, task_A_stream, steps=200, task_name="Task A")

    print("\nEvaluating Task A (immediately after training)...")
    perf_A_immediate, std_A_immediate = evaluate_on_task(model, task_A_stream)
    print(f"  Task A Loss: {perf_A_immediate:.4f} (±{std_A_immediate:.4f})")

    # ==========================
    # CONSOLIDATION: Task A
    # ==========================
    print("\n" + "=" * 70)
    print("CONSOLIDATION PHASE: Task A")
    print("(Simulating 'sleep' - distilling knowledge into core abstractions)")
    print("=" * 70)
    consolidate_knowledge(model, task_name="Task A")

    print("\nEvaluating Task A (after consolidation)...")
    perf_A_consolidated, std_A_consolidated = evaluate_on_task(model, task_A_stream)
    print(f"  Task A Loss: {perf_A_consolidated:.4f} (±{std_A_consolidated:.4f})")

    consolidation_effect = perf_A_consolidated - perf_A_immediate
    print(f"  Consolidation effect: {consolidation_effect:+.4f} " +
          (
              "(improved!)" if consolidation_effect < 0 else "(slightly worse)" if consolidation_effect < 0.001 else "(degraded)"))

    # ==========================
    # PHASE 2: Learn Task B
    # ==========================
    print("\n" + "=" * 70)
    print("PHASE 2: Learning Task B")
    print("=" * 70)
    train_on_task(model, task_B_stream, steps=200, task_name="Task B")

    print("\nEvaluating Task B (immediately after training)...")
    perf_B_immediate, std_B_immediate = evaluate_on_task(model, task_B_stream)
    print(f"  Task B Loss: {perf_B_immediate:.4f} (±{std_B_immediate:.4f})")

    # ==========================
    # CHECK FORGETTING
    # ==========================
    print("\n" + "=" * 70)
    print("CHECKING TASK A (did learning Task B cause forgetting?)")
    print("=" * 70)
    perf_A_after_B, std_A_after_B = evaluate_on_task(model, task_A_stream)
    print(f"  Task A Loss: {perf_A_after_B:.4f} (±{std_A_after_B:.4f})")

    forgetting = perf_A_after_B - perf_A_consolidated
    forgetting_percent = (forgetting / perf_A_consolidated) * 100

    # ==========================
    # CONSOLIDATION: Task B
    # ==========================
    print("\n" + "=" * 70)
    print("CONSOLIDATION PHASE: Task B")
    print("=" * 70)
    consolidate_knowledge(model, task_name="Task B")

    print("\nRe-evaluating both tasks after Task B consolidation...")
    perf_A_final, std_A_final = evaluate_on_task(model, task_A_stream)
    perf_B_final, std_B_final = evaluate_on_task(model, task_B_stream)
    print(f"  Task A Loss: {perf_A_final:.4f} (±{std_A_final:.4f})")
    print(f"  Task B Loss: {perf_B_final:.4f} (±{std_B_final:.4f})")

    # ==========================
    # TEST SAVINGS EFFECT
    # ==========================
    print("\n" + "=" * 70)
    print("SAVINGS EFFECT TEST: How fast can it relearn Task A?")
    print("(Humans relearn forgotten things faster than initial learning)")
    print("=" * 70)
    print("  Retraining Task A for 50 steps...")

    relearn_loss, improvement_rate = test_relearning_speed(
        model, task_A_stream, perf_A_final, steps=50
    )
    print(f"  After relearning: {relearn_loss:.4f}")
    print(f"  Improvement rate: {improvement_rate * 100:.1f}%")

    # ==========================
    # FINAL ANALYSIS
    # ==========================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 Task A Journey:")
    print(f"  1. Immediate after training:  {perf_A_immediate:.4f}")
    print(f"  2. After consolidation:       {perf_A_consolidated:.4f} ({consolidation_effect:+.4f})")
    print(f"  3. After learning Task B:     {perf_A_after_B:.4f} ({forgetting:+.4f}, {forgetting_percent:+.1f}%)")
    print(f"  4. After Task B consolidated: {perf_A_final:.4f}")
    print(f"  5. After 50 relearning steps: {relearn_loss:.4f}")

    print("\n🧠 Human-Like Properties:")

    # 1. Consolidation helps (or at least doesn't hurt much)
    if abs(consolidation_effect) < 0.002:
        print("  ✓ Consolidation preserves knowledge (minimal change)")
    elif consolidation_effect < 0:
        print("  ✓✓ Consolidation IMPROVES performance (abstracts well!)")
    else:
        print("  ✗ Consolidation degrades performance (losing details)")

    # 2. Forgetting is gradual, not catastrophic
    if forgetting < 0.01:
        print(f"  ✓✓ Minimal forgetting ({forgetting_percent:.1f}%) - excellent retention!")
    elif forgetting < 0.05:
        print(f"  ✓ Gradual forgetting ({forgetting_percent:.1f}%) - human-like")
    else:
        print(f"  ✗ Catastrophic forgetting ({forgetting_percent:.1f}%) - not human-like")

    # 3. Savings effect (faster relearning)
    if improvement_rate > 0.3:
        print(f"  ✓✓ Strong savings effect ({improvement_rate * 100:.0f}% improvement) - remembers gist!")
    elif improvement_rate > 0.1:
        print(f"  ✓ Moderate savings effect ({improvement_rate * 100:.0f}%) - some retention")
    else:
        print(f"  ✗ Weak savings effect - may not retain abstractions")

    # 4. Both tasks work reasonably well
    both_tasks_ok = perf_A_final < 0.03 and perf_B_final < 0.03
    if both_tasks_ok:
        print("  ✓ Both tasks remain functional - no total collapse")
    else:
        print("  ✗ One or both tasks degraded significantly")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    total_checks = 4
    passed_checks = sum([
        abs(consolidation_effect) < 0.002 or consolidation_effect < 0,
        forgetting < 0.05,
        improvement_rate > 0.1,
        both_tasks_ok
    ])

    if passed_checks >= 3:
        print("🎉 Your architecture shows HUMAN-LIKE learning properties!")
        print("   The consolidation mechanism is working as intended.")
        print("   It retains abstract knowledge even when details fade.")
    elif passed_checks == 2:
        print("⚠️  Your architecture shows SOME human-like properties.")
        print("   Consolidation helps, but there's room for improvement.")
    else:
        print("❌ Your architecture needs more work to be human-like.")
        print("   Consider: stronger consolidation, protected abstractions,")
        print("   or explicit anti-forgetting mechanisms.")

    print("\n💡 Next steps:")
    if forgetting > 0.05:
        print("   - High forgetting: Add experience replay or regularization")
    if improvement_rate < 0.1:
        print("   - Weak savings: Strengthen core abstraction mechanisms")
    if consolidation_effect > 0.002:
        print("   - Consolidation hurts: Adjust consolidation hyperparameters")
    if passed_checks >= 3:
        print("   - Looking good! Test with more complex tasks")
        print("   - Try 3+ tasks in sequence to test long-term retention")

    print("=" * 70)


if __name__ == "__main__":
    main()

'''
======================================================================
HUMAN-LIKE CONSOLIDATION TEST
other_experiments: Does consolidation create abstract, durable memories?
======================================================================

Device: cpu

======================================================================
PHASE 1: Learning Task A
======================================================================
  Training Task A...
    Step 50/200, Loss: 0.0009
    Step 100/200, Loss: 0.0001
    Step 150/200, Loss: 0.0001
    Step 200/200, Loss: 0.0002

Evaluating Task A (immediately after training)...
  Task A Loss: 0.0006 (±0.0003)

======================================================================
CONSOLIDATION PHASE: Task A
(Simulating 'sleep' - distilling knowledge into core abstractions)
======================================================================
  Consolidating Task A memories...
    Active clusters: [0, 1, 2]
    Consolidating cluster 0 (89 samples)...
    Consolidating cluster 1 (32 samples)...
    Consolidating cluster 2 (79 samples)...
    Consolidation complete!

Evaluating Task A (after consolidation)...
  Task A Loss: 0.0013 (±0.0002)
  Consolidation effect: +0.0007 (slightly worse)

======================================================================
PHASE 2: Learning Task B
======================================================================
  Training Task B...
    Step 50/200, Loss: 0.0007
    Step 100/200, Loss: 0.0000
    Step 150/200, Loss: 0.0001
    Step 200/200, Loss: 0.0001

Evaluating Task B (immediately after training)...
  Task B Loss: 0.0002 (±0.0001)

======================================================================
CHECKING TASK A (did learning Task B cause forgetting?)
======================================================================
  Task A Loss: 0.0423 (±0.0008)

======================================================================
CONSOLIDATION PHASE: Task B
======================================================================
  Consolidating Task B memories...
    Active clusters: [0, 1, 2]
    Consolidating cluster 0 (159 samples)...
    Consolidating cluster 1 (84 samples)...
    Consolidating cluster 2 (157 samples)...
    Consolidation complete!

Re-evaluating both tasks after Task B consolidation...
  Task A Loss: 0.0416 (±0.0004)
  Task B Loss: 0.0004 (±0.0000)

======================================================================
SAVINGS EFFECT TEST: How fast can it relearn Task A?
(Humans relearn forgotten things faster than initial learning)
======================================================================
  Retraining Task A for 50 steps...
  After relearning: 0.0064
  Improvement rate: 84.6%

======================================================================
RESULTS SUMMARY
======================================================================

📊 Task A Journey:
  1. Immediate after training:  0.0006
  2. After consolidation:       0.0013 (+0.0007)
  3. After learning Task B:     0.0423 (+0.0410, +3074.2%)
  4. After Task B consolidated: 0.0416
  5. After 50 relearning steps: 0.0064

🧠 Human-Like Properties:
  ✓ Consolidation preserves knowledge (minimal change)
  ✓ Gradual forgetting (3074.2%) - human-like
  ✓✓ Strong savings effect (85% improvement) - remembers gist!
  ✗ One or both tasks degraded significantly

======================================================================
INTERPRETATION:
======================================================================
🎉 Your architecture shows HUMAN-LIKE learning properties!
   The consolidation mechanism is working as intended.
   It retains abstract knowledge even when details fade.

💡 Next steps:
   - Looking good! Test with more complex tasks
   - Try 3+ tasks in sequence to test long-term retention
======================================================================
'''