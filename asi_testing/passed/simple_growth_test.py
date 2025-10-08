# asi_testing/simple_growth_test_with_exploration.py
# Test with exploration enabled!

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from asi_model import ASISeed, CuriosityStream, to_device


def test_with_exploration():
    print("=" * 70)
    print("GROWTH & EXPLORATION TEST (WITH EXPLORATION ENABLED)")
    print("=" * 70)

    model = ASISeed(
        input_dim=32,
        model_dim=64,
        num_clusters=8,
        core_rank=2,
        build_ema=False,
        use_heads=False
    )

    stream = CuriosityStream(input_dim=32, num_sources=8, seed=42)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    cluster_counts = {i: 0 for i in range(8)}
    growth_events = []
    loss_history = []

    print("\nStarting training with exploration...")
    print("Early steps: HIGH exploration (tries many clusters)")
    print("Later steps: LOW exploration (exploits what it learned)\n")

    # Training loop
    for step in range(1000):
        x = to_device(stream.next())

        # Encode
        z = model.encoder(x)

        # EXPLORATION SCHEDULE (anneal over time)
        # Early: explore widely (high tau, high eps)
        # Late: exploit (low tau, low eps)
        progress = step / 1000
        tau = 2.0 - 1.4 * progress  # 2.0 → 0.6
        eps = 0.2 - 0.18 * progress  # 0.2 → 0.02

        # Route WITH exploration!
        k = model.router(z, tau=tau, eps=eps)
        cluster_counts[k] += 1

        # Forward
        h = model.layer(z, active_cluster=k)
        x_hat = model.decoder(h)

        # Loss
        loss = F.mse_loss(x_hat, x)
        loss_history.append(loss.item())

        # Train
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Update router
        with torch.no_grad():
            model.router.update_centroid(k, z)
        model.update_buffers(k, z)

        # Track for growth
        stats = model.stats[k]
        stats.recent_losses.append(loss.item())
        stats.samples += 1

        if len(stats.recent_losses) > 50:
            stats.recent_losses.pop(0)

        # Check for growth
        if step % 50 == 0 and len(stats.recent_losses) == 50:
            first_half = stats.recent_losses[:25]
            second_half = stats.recent_losses[25:]
            first_avg = sum(first_half) / 25
            second_avg = sum(second_half) / 25
            improvement = first_avg - second_avg

            if improvement < 0.01 and second_avg > 0.005:  # Only grow if struggling
                old_rank = model.layer.U_res[k].shape[1] if model.layer.U_res[k].numel() > 0 else 0
                model.grow_cluster(k, grow_rank=2)
                new_rank = model.layer.U_res[k].shape[1]
                growth_events.append((step, k, old_rank, new_rank))
                print(f"[{step:4d}] 🌱 Growth! Cluster {k}: rank {old_rank} → {new_rank} (loss={second_avg:.4f})")
                model.consolidate_cluster(k)
                stats.recent_losses.clear()

        if step % 200 == 0:
            avg_loss = sum(loss_history[-50:]) / min(50, len(loss_history))
            print(f"[{step:4d}] Loss: {avg_loss:.4f}, Cluster: {k}, tau={tau:.2f}, eps={eps:.3f}")

    # ============ RESULTS ============
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    active_clusters = sum(1 for count in cluster_counts.values() if count > 10)
    total_samples = sum(cluster_counts.values())

    print(f"\n📊 CLUSTER USAGE:")
    print(f"   Active clusters: {active_clusters}/8")
    print(f"   Total samples: {total_samples}")
    print("\n   Distribution:")

    for i, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / total_samples
            bar = "█" * int(pct / 2)
            print(f"   Cluster {i}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n🌱 GROWTH EVENTS: {len(growth_events)}")
    if growth_events:
        for step, k, old, new in growth_events:
            print(f"   Step {step}: Cluster {k} grew from rank {old} → {new}")
    else:
        print("   No growth occurred (might mean learning was easy)")

    # Final loss
    final_loss = sum(loss_history[-100:]) / 100
    initial_loss = sum(loss_history[:100]) / 100
    improvement = ((initial_loss - final_loss) / initial_loss) * 100

    print(f"\n📉 LEARNING:")
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Improvement: {improvement:.1f}%")

    # ============ VERDICT ============
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    checks_passed = 0

    # Check 1: Cluster diversity
    if active_clusters >= 5:
        print("✓ CHECK 1: EXCELLENT diversity (5+ clusters)")
        checks_passed += 1
    elif active_clusters >= 3:
        print("✓ CHECK 1: GOOD diversity (3+ clusters)")
        checks_passed += 1
    else:
        print("✗ CHECK 1: Poor diversity (cluster collapse)")

    # Check 2: Learning happened
    if improvement > 20:
        print("✓ CHECK 2: Strong learning (>20% improvement)")
        checks_passed += 1
    elif improvement > 10:
        print("✓ CHECK 2: Moderate learning")
        checks_passed += 1
    else:
        print("✗ CHECK 2: Weak learning")

    # Check 3: Growth mechanism
    if len(growth_events) >= 2:
        print("✓ CHECK 3: Growth mechanism working")
        checks_passed += 1
    elif len(growth_events) == 1:
        print("⚠ CHECK 3: Growth triggered once (acceptable)")
        checks_passed += 0.5
    else:
        print("⚠ CHECK 3: No growth (okay if losses are low)")

    print(f"\n🎯 SCORE: {checks_passed}/3.0")

    if checks_passed >= 2.5:
        print("\n🎉 SUCCESS! Your ASI is exploring and learning!")
        print("   Next: Run video learning tests")
    elif checks_passed >= 1.5:
        print("\n✓ PROGRESS! Core mechanisms working")
        print("   May need parameter tuning")
    else:
        print("\n⚠️  NEEDS WORK")
        print("   Check if router fix was applied correctly")

    print("=" * 70)


if __name__ == "__main__":
    test_with_exploration()

'''
======================================================================
GROWTH & EXPLORATION TEST (WITH EXPLORATION ENABLED)
======================================================================

Starting training with exploration...
Early steps: HIGH exploration (tries many clusters)
Later steps: LOW exploration (exploits what it learned)

[   0] Loss: 0.0106, Cluster: 3, tau=2.00, eps=0.200
[ 200] Loss: 0.0001, Cluster: 7, tau=1.72, eps=0.164
[ 400] Loss: 0.0001, Cluster: 5, tau=1.44, eps=0.128
[ 600] Loss: 0.0001, Cluster: 1, tau=1.16, eps=0.092
[ 800] Loss: 0.0001, Cluster: 6, tau=0.88, eps=0.056

======================================================================
RESULTS
======================================================================

📊 CLUSTER USAGE:
   Active clusters: 8/8
   Total samples: 1000

   Distribution:
   Cluster 6:  134 ( 13.4%) ██████
   Cluster 1:  132 ( 13.2%) ██████
   Cluster 7:  129 ( 12.9%) ██████
   Cluster 3:  126 ( 12.6%) ██████
   Cluster 5:  124 ( 12.4%) ██████
   Cluster 0:  123 ( 12.3%) ██████
   Cluster 4:  118 ( 11.8%) █████
   Cluster 2:  114 ( 11.4%) █████

🌱 GROWTH EVENTS: 0
   No growth occurred (might mean learning was easy)

📉 LEARNING:
   Initial loss: 0.0015
   Final loss: 0.0001
   Improvement: 95.6%

======================================================================
VERDICT
======================================================================
✓ CHECK 1: EXCELLENT diversity (5+ clusters)
✓ CHECK 2: Strong learning (>20% improvement)
⚠ CHECK 3: No growth (okay if losses are low)

🎯 SCORE: 2/3.0

✓ PROGRESS! Core mechanisms working
   May need parameter tuning
======================================================================
'''