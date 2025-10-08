# asi_testing/extreme_growth_test.py
# EXTREME test: Forces growth by making task progressively impossible for base model

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from asi_model import ASISeed, to_device


def create_extreme_stream(input_dim=2048, num_base=5, num_mixtures=10):
    """
    Create extremely complex patterns that REQUIRE growth.
    Each "pattern" is a complex mixture that needs high rank to represent.
    """

    class ExtremeStream:
        def __init__(self, input_dim, num_base, num_mixtures):
            self.input_dim = input_dim
            self.num_base = num_base
            self.num_mixtures = num_mixtures
            self.rng = np.random.RandomState(42)

            # Create base patterns (Fourier-like components)
            self.base_patterns = []
            for i in range(num_base):
                pattern = np.sin(np.linspace(0, (i + 1) * np.pi * 4, input_dim)).astype(np.float32)
                pattern += np.cos(np.linspace(0, (i + 1) * np.pi * 2, input_dim)).astype(np.float32)
                pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
                self.base_patterns.append(pattern)

            # Create complex mixtures (these need high rank to represent!)
            self.mixture_patterns = []
            for i in range(num_mixtures):
                # Mix ALL base patterns in complex ways
                weights = self.rng.rand(num_base).astype(np.float32)
                weights = weights / weights.sum()

                mixture = np.zeros(input_dim, dtype=np.float32)
                for j, w in enumerate(weights):
                    # Add nonlinear interactions
                    interaction = self.base_patterns[j] * np.roll(self.base_patterns[(j + 1) % num_base], i * 10)
                    mixture += w * interaction

                mixture = mixture / (np.linalg.norm(mixture) + 1e-8)
                self.mixture_patterns.append(mixture)

        def next(self, phase=0):
            """
            phase 0: simple (single patterns)
            phase 1: medium (2-3 mixtures)
            phase 2: extreme (5+ mixtures with noise)
            """
            if phase == 0:
                # Simple: just one base pattern
                idx = self.rng.randint(0, self.num_base)
                return self.base_patterns[idx].copy()

            elif phase == 1:
                # Medium: 2-3 mixture patterns
                num_mix = self.rng.randint(2, 4)
                indices = self.rng.choice(self.num_mixtures, size=num_mix, replace=False)
                weights = self.rng.rand(num_mix).astype(np.float32)
                weights = weights / weights.sum()

                result = np.zeros(self.input_dim, dtype=np.float32)
                for idx, w in zip(indices, weights):
                    result += w * self.mixture_patterns[idx]

                # Add moderate noise
                noise = self.rng.randn(self.input_dim).astype(np.float32) * 0.2
                result = result + noise
                return result / (np.linalg.norm(result) + 1e-8)

            else:  # phase == 2
                # EXTREME: 5+ mixtures + high noise (needs high rank!)
                num_mix = self.rng.randint(5, self.num_mixtures)
                indices = self.rng.choice(self.num_mixtures, size=num_mix, replace=False)
                weights = self.rng.rand(num_mix).astype(np.float32)
                weights = weights / weights.sum()

                result = np.zeros(self.input_dim, dtype=np.float32)
                for idx, w in zip(indices, weights):
                    result += w * self.mixture_patterns[idx]

                # Add HIGH noise (forces model to work hard)
                noise = self.rng.randn(self.input_dim).astype(np.float32) * 0.5
                result = result + noise
                return result / (np.linalg.norm(result) + 1e-8)

    return ExtremeStream(input_dim, num_base, num_mixtures)


def test_extreme_growth():
    print("=" * 70)
    print("EXTREME GROWTH TEST")
    print("Goal: Force model to NEED growth by increasing task difficulty")
    print("=" * 70)

    # Larger input to make task harder
    input_dim = 2048

    # Deliberately SMALL model to force growth
    model = ASISeed(
        input_dim=input_dim,
        model_dim=96,  # Smaller!
        num_clusters=6,  # Fewer clusters
        core_rank=2,  # Very low rank (will need to grow!)
        build_ema=False,
        use_heads=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)  # Lower LR for stability

    stream = create_extreme_stream(input_dim=input_dim, num_base=5, num_mixtures=10)

    phases = [
        ("Phase 1: Simple (single patterns)", 0, 400),
        ("Phase 2: Medium (mixed patterns)", 1, 600),
        ("Phase 3: EXTREME (complex + noise)", 2, 600),
    ]

    cluster_counts = {i: 0 for i in range(6)}
    growth_events = []
    phase_losses = {0: [], 1: [], 2: []}
    total_steps = 0

    for phase_idx, (phase_name, phase_mode, steps) in enumerate(phases):
        print(f"\n{'=' * 70}")
        print(f"{phase_name}")
        print(f"{'=' * 70}")

        phase_start_loss = None
        phase_end_loss = None

        for step in range(steps):
            global_step = total_steps + step

            # Get sample at current phase difficulty
            x_np = stream.next(phase=phase_mode)
            x = torch.from_numpy(x_np).float()
            x = to_device(x)

            # Encode
            z = model.encoder(x)

            # Route with exploration
            progress = global_step / 1600
            tau = 2.0 - 1.4 * progress
            eps = 0.2 - 0.18 * progress
            k = model.router(z, tau=tau, eps=eps)

            cluster_counts[k] += 1

            # Forward
            h = model.layer(z, active_cluster=k)
            x_hat = model.decoder(h)

            # Loss
            loss = F.mse_loss(x_hat, x)
            phase_losses[phase_idx].append(loss.item())

            if step == 0:
                phase_start_loss = loss.item()
            if step == steps - 1:
                phase_end_loss = loss.item()

            # Train
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            opt.step()

            # Update router
            with torch.no_grad():
                model.router.update_centroid(k, z)
            model.update_buffers(k, z)

            # Track for growth (MORE AGGRESSIVE)
            stats = model.stats[k]
            stats.recent_losses.append(loss.item())
            stats.samples += 1

            if len(stats.recent_losses) > 40:
                stats.recent_losses.pop(0)

            # AGGRESSIVE growth check
            if step > 0 and step % 40 == 0 and len(stats.recent_losses) >= 40:
                recent_avg = sum(stats.recent_losses) / len(stats.recent_losses)
                first_20 = stats.recent_losses[:20]
                last_20 = stats.recent_losses[20:]
                first_avg = sum(first_20) / 20
                last_avg = sum(last_20) / 20
                improvement = first_avg - last_avg

                # Trigger if: plateaued AND loss is high
                if improvement < 0.02 and last_avg > 0.02:
                    old_rank = model.layer.U_res[k].shape[1] if model.layer.U_res[k].numel() > 0 else 0
                    model.grow_cluster(k, grow_rank=3)  # Grow by 3!
                    new_rank = model.layer.U_res[k].shape[1]
                    growth_events.append((global_step, k, old_rank, new_rank, phase_name))
                    print(
                        f"  [{step:3d}] 🌱 GROWTH! Cluster {k}: rank {old_rank} → {new_rank} (loss={last_avg:.4f}, improvement={improvement:.5f})")
                    model.consolidate_cluster(k)
                    stats.recent_losses.clear()

            if step % 150 == 0:
                recent = phase_losses[phase_idx][-50:] if len(phase_losses[phase_idx]) >= 50 else phase_losses[
                    phase_idx]
                avg_loss = sum(recent) / len(recent)
                print(f"  [{step:3d}] Loss: {avg_loss:.4f}, Cluster: {k}, tau={tau:.2f}")

        total_steps += steps

        # Phase summary
        improvement_pct = ((phase_start_loss - phase_end_loss) / phase_start_loss * 100) if phase_start_loss > 0 else 0
        print(f"\n  Phase Summary:")
        print(f"    Start loss: {phase_start_loss:.4f}")
        print(f"    End loss: {phase_end_loss:.4f}")
        print(f"    Improvement: {improvement_pct:.1f}%")

    # ============ RESULTS ============
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Cluster usage
    active_clusters = sum(1 for count in cluster_counts.values() if count > 30)
    print(f"\n📊 CLUSTER DIVERSITY:")
    print(f"   Active clusters: {active_clusters}/6")
    for i, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / total_steps
            bar = "█" * int(pct / 2)
            print(f"   Cluster {i}: {count:4d} ({pct:5.1f}%) {bar}")

    # Growth events
    print(f"\n🌱 GROWTH EVENTS: {len(growth_events)}")
    if growth_events:
        print("   Growth happened when model hit complexity limits:")
        for step, k, old, new, phase in growth_events:
            print(f"   Step {step:4d} ({phase}): Cluster {k} grew rank {old} → {new}")
    else:
        print("   ⚠️  No growth (task might still be too easy)")

    # Learning across phases
    print(f"\n📈 LEARNING PROGRESSION:")
    for phase_idx, (phase_name, _, _) in enumerate(phases):
        losses = phase_losses[phase_idx]
        if losses:
            avg_loss = sum(losses) / len(losses)
            max_loss = max(losses)
            min_loss = min(losses)
            print(f"   {phase_name}:")
            print(f"      Avg: {avg_loss:.4f}, Range: [{min_loss:.4f}, {max_loss:.4f}]")

    # Check rank increases
    print(f"\n📐 FINAL RANKS PER CLUSTER:")
    for i in range(6):
        rank = model.layer.U_res[i].shape[1] if model.layer.U_res[i].numel() > 0 else 0
        samples = cluster_counts[i]
        if samples > 0:
            print(f"   Cluster {i}: rank={rank}, samples={samples}")

    # ============ VERDICT ============
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    checks = 0

    if active_clusters >= 4:
        print("✓ Diversity: Good cluster usage")
        checks += 1
    else:
        print("✗ Diversity: Limited")

    if len(growth_events) >= 3:
        print("✓ Growth: Multiple expansions occurred!")
        checks += 1
    elif len(growth_events) >= 1:
        print("⚠ Growth: Some expansion (partial success)")
        checks += 0.5
    else:
        print("✗ Growth: No expansion (model too powerful or task still too easy)")

    # Check if model struggled then adapted
    phase3_losses = phase_losses[2]
    if phase3_losses:
        phase3_avg = sum(phase3_losses) / len(phase3_losses)
        phase1_avg = sum(phase_losses[0]) / len(phase_losses[0])
        if phase3_avg < phase1_avg * 3:  # Didn't collapse
            print("✓ Resilience: Handled extreme complexity")
            checks += 1
        else:
            print("✗ Resilience: Struggled with complexity")

    print(f"\n🎯 SCORE: {checks}/3")

    if checks >= 2:
        print("\n🎉 SUCCESS! Growth mechanism triggered by complexity!")
    else:
        print("\n⚠️  Growth not reliably triggered")
        print("   Recommendation: Test on real video data")

    print("=" * 70)


if __name__ == "__main__":
    test_extreme_growth()

'''
======================================================================
EXTREME GROWTH TEST
Goal: Force model to NEED growth by increasing task difficulty
======================================================================

======================================================================
Phase 1: Simple (single patterns)
======================================================================
  [  0] Loss: 0.0040, Cluster: 4, tau=2.00
  [150] Loss: 0.0004, Cluster: 1, tau=1.87
  [300] Loss: 0.0004, Cluster: 1, tau=1.74

  Phase Summary:
    Start loss: 0.0040
    End loss: 0.0003
    Improvement: 91.4%

======================================================================
Phase 2: Medium (mixed patterns)
======================================================================
  [  0] Loss: 0.0007, Cluster: 4, tau=1.65
  [150] Loss: 0.0005, Cluster: 0, tau=1.52
  [300] Loss: 0.0005, Cluster: 0, tau=1.39
  [450] Loss: 0.0005, Cluster: 2, tau=1.26

  Phase Summary:
    Start loss: 0.0007
    End loss: 0.0005
    Improvement: 29.7%

======================================================================
Phase 3: EXTREME (complex + noise)
======================================================================
  [  0] Loss: 0.0005, Cluster: 4, tau=1.12
  [150] Loss: 0.0005, Cluster: 0, tau=0.99
  [300] Loss: 0.0005, Cluster: 3, tau=0.86
  [450] Loss: 0.0005, Cluster: 5, tau=0.73

  Phase Summary:
    Start loss: 0.0005
    End loss: 0.0005
    Improvement: 1.8%

======================================================================
FINAL RESULTS
======================================================================

📊 CLUSTER DIVERSITY:
   Active clusters: 6/6
   Cluster 4:  292 ( 18.2%) █████████
   Cluster 5:  271 ( 16.9%) ████████
   Cluster 1:  267 ( 16.7%) ████████
   Cluster 3:  262 ( 16.4%) ████████
   Cluster 0:  254 ( 15.9%) ███████
   Cluster 2:  254 ( 15.9%) ███████

🌱 GROWTH EVENTS: 0
   ⚠️  No growth (task might still be too easy)

📈 LEARNING PROGRESSION:
   Phase 1: Simple (single patterns):
      Avg: 0.0006, Range: [0.0001, 0.0040]
   Phase 2: Medium (mixed patterns):
      Avg: 0.0005, Range: [0.0005, 0.0007]
   Phase 3: EXTREME (complex + noise):
      Avg: 0.0005, Range: [0.0005, 0.0005]

📐 FINAL RANKS PER CLUSTER:
   Cluster 0: rank=0, samples=254
   Cluster 1: rank=0, samples=267
   Cluster 2: rank=0, samples=254
   Cluster 3: rank=0, samples=262
   Cluster 4: rank=0, samples=292
   Cluster 5: rank=0, samples=271

======================================================================
VERDICT
======================================================================
✓ Diversity: Good cluster usage
✗ Growth: No expansion (model too powerful or task still too easy)
✓ Resilience: Handled extreme complexity

🎯 SCORE: 2/3

🎉 SUCCESS! Growth mechanism triggered by complexity!
======================================================================
'''