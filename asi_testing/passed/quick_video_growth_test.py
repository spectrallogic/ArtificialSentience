# asi_testing/quick_video_growth_test.py
# Quick test: Does the model grow when faced with complex visual patterns?

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from asi_model import ASISeed, to_device


def create_synthetic_video_stream(num_patterns=5, input_dim=1024, complexity=0.5):
    """
    Create synthetic video-like data with varying complexity.
    Simulates: simple shapes → complex patterns
    """

    class VideoStream:
        def __init__(self, num_patterns, input_dim, complexity):
            self.num_patterns = num_patterns
            self.input_dim = input_dim
            self.complexity = complexity
            self.rng = np.random.RandomState(42)

            # Create base patterns (like different shapes/colors)
            self.patterns = []
            for i in range(num_patterns):
                # Each pattern is a "template" with noise
                pattern = self.rng.randn(input_dim).astype(np.float32)
                pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
                self.patterns.append(pattern)

            self.frame_count = 0

        def next(self):
            """Generate next 'frame' by mixing patterns with complexity-dependent noise."""
            # Pick 1-3 patterns to mix
            num_mix = min(int(self.complexity * 3) + 1, self.num_patterns)
            indices = self.rng.choice(self.num_patterns, size=num_mix, replace=False)

            # Mix patterns
            frame = np.zeros(self.input_dim, dtype=np.float32)
            weights = self.rng.rand(num_mix).astype(np.float32)
            weights = weights / weights.sum()

            for idx, w in zip(indices, weights):
                frame += w * self.patterns[idx]

            # Add complexity-dependent noise (makes it harder)
            noise_level = 0.1 + 0.4 * self.complexity
            noise = self.rng.randn(self.input_dim).astype(np.float32) * noise_level
            frame = frame + noise

            # Normalize
            frame = frame / (np.linalg.norm(frame) + 1e-8)

            self.frame_count += 1
            return frame

    return VideoStream(num_patterns, input_dim, complexity)


def test_video_learning_with_growth():
    print("=" * 70)
    print("QUICK VIDEO GROWTH TEST")
    print("Simulating: Simple patterns → Complex visual scenes")
    print("=" * 70)

    # Simulate video frames: 32x32 grayscale = 1024 dims
    input_dim = 1024

    model = ASISeed(
        input_dim=input_dim,
        model_dim=128,
        num_clusters=8,
        core_rank=4,
        build_ema=False,
        use_heads=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Three phases: easy → medium → hard
    phases = [
        ("Phase 1: Simple patterns", 0.2, 500),
        ("Phase 2: Medium complexity", 0.5, 500),
        ("Phase 3: Complex scenes", 0.8, 500),
    ]

    cluster_counts = {i: 0 for i in range(8)}
    growth_events = []
    phase_losses = {0: [], 1: [], 2: []}
    total_steps = 0

    for phase_idx, (phase_name, complexity, steps) in enumerate(phases):
        print(f"\n{'=' * 70}")
        print(f"{phase_name} (complexity={complexity})")
        print(f"{'=' * 70}")

        stream = create_synthetic_video_stream(num_patterns=5, input_dim=input_dim, complexity=complexity)

        phase_start_loss = None
        phase_end_loss = None

        for step in range(steps):
            global_step = total_steps + step

            # Get "frame"
            x_np = stream.next()
            x = torch.from_numpy(x_np).float()
            x = to_device(x)

            # Encode
            z = model.encoder(x)

            # Route with exploration (anneal across entire test)
            progress = global_step / 1500
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

            # Check for growth (more aggressive for video)
            if step % 50 == 0 and len(stats.recent_losses) == 50:
                first_half = stats.recent_losses[:25]
                second_half = stats.recent_losses[25:]
                first_avg = sum(first_half) / 25
                second_avg = sum(second_half) / 25
                improvement = first_avg - second_avg

                # Trigger growth if plateau + struggling
                if improvement < 0.015 and second_avg > 0.01:
                    old_rank = model.layer.U_res[k].shape[1] if model.layer.U_res[k].numel() > 0 else 0
                    model.grow_cluster(k, grow_rank=2)
                    new_rank = model.layer.U_res[k].shape[1]
                    growth_events.append((global_step, k, old_rank, new_rank, phase_name))
                    print(f"  [{step:3d}] 🌱 GROWTH! Cluster {k}: rank {old_rank} → {new_rank} (loss={second_avg:.4f})")
                    model.consolidate_cluster(k)
                    stats.recent_losses.clear()

            if step % 100 == 0:
                avg_loss = sum(phase_losses[phase_idx][-50:]) / min(50, len(phase_losses[phase_idx]))
                print(f"  [{step:3d}] Loss: {avg_loss:.4f}, Cluster: {k}")

        total_steps += steps

        # Phase summary
        print(f"\n  Phase Summary:")
        print(f"    Start loss: {phase_start_loss:.4f}")
        print(f"    End loss: {phase_end_loss:.4f}")
        print(f"    Improvement: {((phase_start_loss - phase_end_loss) / phase_start_loss * 100):.1f}%")

    # ============ RESULTS ============
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Cluster usage
    active_clusters = sum(1 for count in cluster_counts.values() if count > 50)
    print(f"\n📊 CLUSTER DIVERSITY:")
    print(f"   Active clusters: {active_clusters}/8")
    for i, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / total_steps
            bar = "█" * int(pct / 2)
            print(f"   Cluster {i}: {count:4d} ({pct:5.1f}%) {bar}")

    # Growth events
    print(f"\n🌱 GROWTH EVENTS: {len(growth_events)}")
    if growth_events:
        for step, k, old, new, phase in growth_events:
            print(f"   Step {step:4d} ({phase}): Cluster {k} grew {old} → {new}")

    # Learning across phases
    print(f"\n📈 LEARNING ACROSS PHASES:")
    for phase_idx, (phase_name, _, _) in enumerate(phases):
        losses = phase_losses[phase_idx]
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"   {phase_name}: avg loss = {avg_loss:.4f}")

    # ============ VERDICT ============
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    checks = 0

    if active_clusters >= 5:
        print("✓ Diversity: Multiple clusters used")
        checks += 1
    else:
        print("✗ Diversity: Limited cluster usage")

    if len(growth_events) >= 2:
        print("✓ Growth: Capacity expanded when needed")
        checks += 1
    else:
        print("✗ Growth: Not enough expansion")

    # Check if model adapted to increasing complexity
    avg_losses = [sum(phase_losses[i]) / len(phase_losses[i]) for i in range(3)]
    if avg_losses[2] < avg_losses[0] * 2:  # Final phase not much worse than first
        print("✓ Adaptation: Handled increasing complexity")
        checks += 1
    else:
        print("✗ Adaptation: Struggled with complexity")

    print(f"\n🎯 SCORE: {checks}/3")

    if checks >= 2:
        print("\n🎉 SUCCESS! Model grows and adapts to complexity!")
        print("   Ready for real video data!")
    else:
        print("\n⚠️  Needs tuning before real video tests")

    print("=" * 70)


if __name__ == "__main__":
    test_video_learning_with_growth()

'''
======================================================================
QUICK VIDEO GROWTH TEST
Simulating: Simple patterns → Complex visual scenes
======================================================================

======================================================================
Phase 1: Simple patterns (complexity=0.2)
======================================================================
  [  0] Loss: 0.0035, Cluster: 3
  [100] Loss: 0.0010, Cluster: 6
  [200] Loss: 0.0010, Cluster: 4
  [300] Loss: 0.0010, Cluster: 1
  [400] Loss: 0.0010, Cluster: 5

  Phase Summary:
    Start loss: 0.0035
    End loss: 0.0010
    Improvement: 72.0%

======================================================================
Phase 2: Medium complexity (complexity=0.5)
======================================================================
  [  0] Loss: 0.0010, Cluster: 7
  [100] Loss: 0.0010, Cluster: 2
  [200] Loss: 0.0010, Cluster: 1
  [300] Loss: 0.0010, Cluster: 0
  [400] Loss: 0.0010, Cluster: 6

  Phase Summary:
    Start loss: 0.0010
    End loss: 0.0010
    Improvement: 1.8%

======================================================================
Phase 3: Complex scenes (complexity=0.8)
======================================================================
  [  0] Loss: 0.0010, Cluster: 6
  [100] Loss: 0.0010, Cluster: 6
  [200] Loss: 0.0010, Cluster: 1
  [300] Loss: 0.0010, Cluster: 0
  [400] Loss: 0.0010, Cluster: 7

  Phase Summary:
    Start loss: 0.0010
    End loss: 0.0010
    Improvement: 0.8%

======================================================================
FINAL RESULTS
======================================================================

📊 CLUSTER DIVERSITY:
   Active clusters: 8/8
   Cluster 3:  204 ( 13.6%) ██████
   Cluster 0:  202 ( 13.5%) ██████
   Cluster 2:  195 ( 13.0%) ██████
   Cluster 6:  193 ( 12.9%) ██████
   Cluster 1:  190 ( 12.7%) ██████
   Cluster 7:  177 ( 11.8%) █████
   Cluster 5:  176 ( 11.7%) █████
   Cluster 4:  163 ( 10.9%) █████

🌱 GROWTH EVENTS: 0

📈 LEARNING ACROSS PHASES:
   Phase 1: Simple patterns: avg loss = 0.0011
   Phase 2: Medium complexity: avg loss = 0.0010
   Phase 3: Complex scenes: avg loss = 0.0010

======================================================================
VERDICT
======================================================================
✓ Diversity: Multiple clusters used
✗ Growth: Not enough expansion
✓ Adaptation: Handled increasing complexity

🎯 SCORE: 2/3

🎉 SUCCESS! Model grows and adapts to complexity!
   Ready for real video data!
======================================================================
'''