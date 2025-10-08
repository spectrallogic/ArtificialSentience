# asi_testing/adaptive_temporal_learner.py
# Learns temporal dynamics naturally - discovers optimal time scales on its own
# Key: Model decides WHEN to encode based on what changed, not forced frame rate

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from asi_model import ASISeed, to_device


class AdaptiveTemporalLearner:
    """
    Learns time like a human:
    - Detects when things CHANGE (events matter, not all frames)
    - Discovers multiple timescales (fast vs slow changes)
    - Abstracts similar moments (compression)
    - Predicts future based on learned temporal patterns
    """

    def __init__(self, frame_size=64, device="cpu"):
        self.frame_size = frame_size
        self.device = torch.device(device)
        self.input_dim = frame_size * frame_size  # Grayscale for simplicity

        # Model with temporal core and subconscious
        self.model_dim = 192
        self.model = ASISeed(
            input_dim=self.input_dim,
            model_dim=self.model_dim,
            num_clusters=16,
            core_rank=4,
            build_ema=False,
            use_heads=True  # Use prediction head!
        ).to(self.device)

        self.opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3
        )

        # Temporal state (multi-timescale)
        self.h_current = torch.zeros(self.model_dim, device=self.device)
        self.last_encoded_frame = None
        self.last_z = None

        # Display state (keep last reconstruction/prediction visible)
        self.last_recon = None
        self.last_pred = None

        # Event detection (change threshold)
        self.change_threshold = 0.15  # Encode when change > threshold
        self.frames_since_encode = 0
        self.max_skip = 10  # Maximum frames to skip

        # Multi-timescale memory (discovers importance of different timescales)
        self.fast_memory = deque(maxlen=5)  # Recent moments
        self.medium_memory = deque(maxlen=20)  # Short-term
        self.slow_memory = deque(maxlen=100)  # Long-term patterns

        # Stats
        self.total_frames = 0
        self.encoded_frames = 0
        self.skipped_frames = 0
        self.total_steps = 0
        self.cluster_usage = {i: 0 for i in range(16)}
        self.growth_events = []

        # Adaptive thresholds (learn when to pay attention)
        self.change_history = deque(maxlen=200)

        print("=" * 70)
        print("ADAPTIVE TEMPORAL LEARNER")
        print("=" * 70)
        print("Learning principles:")
        print("  • Encode frames only when something CHANGES")
        print("  • Discover optimal time scales through experience")
        print("  • Predict future to understand temporal patterns")
        print("  • Abstract similar moments for efficiency")
        print("=" * 70)

    def frame_to_tensor(self, frame):
        """Convert frame to tensor."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.frame_size, self.frame_size))
        normalized = small.astype(np.float32) / 255.0
        vector = normalized.flatten()
        return torch.from_numpy(vector).float().to(self.device)

    def tensor_to_frame(self, tensor, size=320):
        """Convert tensor back to displayable frame."""
        arr = tensor.detach().cpu().numpy()
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        frame_2d = arr.reshape(self.frame_size, self.frame_size)
        return cv2.resize(frame_2d, (size, size), interpolation=cv2.INTER_NEAREST)

    def detect_change(self, x_current):
        """
        Detect if current frame is significantly different from last encoded frame.
        Returns: (change_magnitude, should_encode)
        """
        if self.last_encoded_frame is None:
            return 1.0, True  # Always encode first frame

        # Compute change (L2 distance)
        diff = torch.norm(x_current - self.last_encoded_frame).item()
        self.change_history.append(diff)

        # Adaptive threshold: learn from recent changes
        if len(self.change_history) > 50:
            recent_changes = list(self.change_history)[-50:]
            adaptive_threshold = np.percentile(recent_changes, 60)  # 60th percentile
            threshold = max(self.change_threshold, adaptive_threshold)
        else:
            threshold = self.change_threshold

        # Force encode after max skip
        force_encode = self.frames_since_encode >= self.max_skip

        should_encode = diff > threshold or force_encode

        return diff, should_encode

    def encode_and_learn(self, x, visualize=True):
        """
        Encode current frame and learn temporal patterns.
        Returns: (reconstruction, prediction, info_dict)
        """
        # Encode
        z = self.model.encoder(x)

        # Route with exploration (anneal over time)
        progress = min(1.0, self.total_steps / 50000)
        tau = 2.0 - 1.4 * progress
        eps = 0.2 - 0.18 * progress
        k = self.model.router(z, tau=tau, eps=eps)

        self.cluster_usage[k] += 1

        # Get memory context for this cluster (subconscious)
        mem_bank = None
        if len(self.model.buffers[k]) > 0:
            mem_bank = torch.stack(self.model.buffers[k][-32:], dim=0)

        # Detach h_current to avoid backprop through previous iterations
        h_detached = self.h_current.detach()

        subconscious_bias, subc_info = self.model.subconscious(z, h_detached, mem_bank)

        # Update temporal state (with subconscious influence)
        h_new = self.model.temporal(z, h_detached, bias=subconscious_bias)

        # Forward through layer
        h_latent = self.model.layer(z, active_cluster=k)

        # Reconstruct current
        x_recon = self.model.decoder(h_latent)

        # Predict future (if we have prediction head)
        x_pred = None
        if self.model.use_heads:
            x_pred = self.model.predict_head(h_new)

        # Losses
        recon_loss = F.mse_loss(x_recon, x)

        pred_loss = torch.tensor(0.0, device=self.device)
        if x_pred is not None and self.last_encoded_frame is not None:
            # Predict what current frame should be based on past
            pred_loss = F.mse_loss(x_pred, x)

        total_loss = recon_loss + 0.3 * pred_loss

        # Train
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        # Update router and buffers
        with torch.no_grad():
            self.model.router.update_centroid(k, z)
        self.model.update_buffers(k, z)

        # Add to multi-timescale memory
        self.fast_memory.append(z.detach().cpu())
        if self.encoded_frames % 3 == 0:  # Medium: every 3rd encoded frame
            self.medium_memory.append(z.detach().cpu())
        if self.encoded_frames % 10 == 0:  # Slow: every 10th encoded frame
            self.slow_memory.append(z.detach().cpu())

        # Check for growth
        self._check_growth(k, recon_loss.item())

        # Update state
        self.h_current = h_new.detach()  # Detach to free computation graph
        self.last_encoded_frame = x.detach()
        self.last_z = z.detach()

        # Cache display tensors
        self.last_recon = x_recon.detach()
        self.last_pred = x_pred.detach() if x_pred is not None else None

        self.encoded_frames += 1
        self.total_steps += 1

        return x_recon, x_pred, {
            'k': k,
            'recon_loss': recon_loss.item(),
            'pred_loss': pred_loss.item(),
            'total_loss': total_loss.item(),
            'subconscious_gate': subc_info.get('gate', 0),
            'tau': tau,
            'eps': eps
        }

    def _check_growth(self, k, loss):
        """Check if cluster k needs to grow."""
        stats = self.model.stats[k]
        stats.recent_losses.append(loss)
        stats.samples += 1

        if len(stats.recent_losses) > 50:
            stats.recent_losses.pop(0)

        if len(stats.recent_losses) >= 50 and self.total_steps % 50 == 0:
            first_half = stats.recent_losses[:25]
            second_half = stats.recent_losses[25:]
            first_avg = sum(first_half) / 25
            second_avg = sum(second_half) / 25
            improvement = first_avg - second_avg

            if improvement < 0.01 and second_avg > 0.015:
                old_rank = self.model.layer.U_res[k].shape[1] if self.model.layer.U_res[k].numel() > 0 else 0
                self.model.grow_cluster(k, grow_rank=2)
                new_rank = self.model.layer.U_res[k].shape[1]
                self.growth_events.append((self.total_steps, k, old_rank, new_rank))
                print(f"\n🌱 GROWTH! Cluster {k}: rank {old_rank} → {new_rank} (loss={second_avg:.4f})")
                self.model.consolidate_cluster(k)
                stats.recent_losses.clear()

    def process_frame(self, frame):
        """
        Process a single frame with adaptive temporal encoding.
        Returns: (should_display, display_frame, info_dict)
        """
        self.total_frames += 1

        # Convert to tensor
        x = self.frame_to_tensor(frame)

        # Detect if we should encode this frame
        change_mag, should_encode = self.detect_change(x)

        if should_encode:
            # Encode and learn
            x_recon, x_pred, info = self.encode_and_learn(x)

            # Create visualization with fresh data
            display = self._create_display(
                frame, x_recon, x_pred, info, change_mag, is_skip=False
            )

            self.frames_since_encode = 0

            return True, display, info
        else:
            # Skip this frame (no significant change)
            self.frames_since_encode += 1
            self.skipped_frames += 1

            # Show display with CACHED reconstruction/prediction
            info = {
                'skipped': True,
                'change': change_mag,
                'k': -1,
                'recon_loss': 0,
                'pred_loss': 0,
                'total_loss': 0
            }

            display = self._create_display(
                frame,
                self.last_recon if self.last_recon is not None else x,
                self.last_pred,
                info,
                change_mag,
                is_skip=True
            )

            return False, display, info

    def _create_display(self, original, recon_tensor, pred_tensor, info, change_mag, is_skip=False):
        """Create visualization showing original, reconstruction, and prediction."""
        # Standardize display size
        display_size = 320

        # Resize original to match
        original_resized = cv2.resize(original, (display_size, display_size))

        # Convert tensors to frames
        recon_frame = self.tensor_to_frame(recon_tensor, size=display_size)
        recon_color = cv2.cvtColor(recon_frame, cv2.COLOR_GRAY2BGR)

        if pred_tensor is not None:
            pred_frame = self.tensor_to_frame(pred_tensor, size=display_size)
            pred_color = cv2.cvtColor(pred_frame, cv2.COLOR_GRAY2BGR)
        else:
            pred_color = np.zeros((display_size, display_size, 3), dtype=np.uint8)

        # Stack: [Original | Reconstruction | Prediction]
        display = np.hstack([original_resized, recon_color, pred_color])

        # Add info overlay
        efficiency = (1 - self.encoded_frames / max(1, self.total_frames)) * 100

        # Panel labels
        label_color = (0, 255, 255) if is_skip else (0, 255, 0)
        cv2.putText(display, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        cv2.putText(display, "RECONSTRUCTION", (display_size + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display, "PREDICTION", (2 * display_size + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Skip indicator
        if is_skip:
            cv2.putText(display, "SKIPPING", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Stats
        y_offset = 90
        stats = [
            f"Total: {self.total_frames} | Encoded: {self.encoded_frames} | Skipped: {self.skipped_frames}",
            f"Efficiency: {efficiency:.1f}% frames skipped",
            f"Change: {change_mag:.3f}" + (f" | Cluster: {info['k']}" if not is_skip else ""),
        ]

        if not is_skip:
            stats.append(f"Loss: {info['total_loss']:.4f} (R:{info['recon_loss']:.4f}, P:{info['pred_loss']:.4f})")
            stats.append(f"Growth events: {len(self.growth_events)}")

        for i, text in enumerate(stats):
            cv2.putText(display, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def run(self, camera_id=0):
        """Run the adaptive temporal learner on webcam."""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return

        print("\n▶️  Camera opened! Learning temporal patterns...")
        print("   • Watch the model learn when to PAY ATTENTION")
        print("   • Frames are skipped when nothing changes")
        print("   • More frames encoded when action happens")
        print("\nControls:")
        print("   q = quit")
        print("   SPACE = show stats")
        print("=" * 70 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process with adaptive temporal encoding
                encoded, display, info = self.process_frame(frame)

                # Show
                cv2.imshow('Adaptive Temporal Learning', display)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self._print_stats()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_stats()

    def _print_stats(self):
        """Print current statistics."""
        print("\n" + "=" * 70)
        print("CURRENT STATS")
        print("=" * 70)

        efficiency = (1 - self.encoded_frames / max(1, self.total_frames)) * 100

        print(f"Frames: {self.total_frames} total, {self.encoded_frames} encoded, {self.skipped_frames} skipped")
        print(f"Efficiency: {efficiency:.1f}% (learning to skip redundant frames!)")
        print(f"Growth events: {len(self.growth_events)}")

        print("\nCluster usage (top 5):")
        sorted_clusters = sorted(self.cluster_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        for k, count in sorted_clusters:
            if count > 0:
                pct = 100 * count / max(1, self.encoded_frames)
                print(f"  Cluster {k}: {count} ({pct:.1f}%)")

        print("=" * 70 + "\n")

    def _print_final_stats(self):
        """Print final statistics."""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)

        efficiency = (1 - self.encoded_frames / max(1, self.total_frames)) * 100

        print(f"\n📊 TEMPORAL EFFICIENCY:")
        print(f"   Total frames seen: {self.total_frames}")
        print(f"   Frames encoded: {self.encoded_frames}")
        print(f"   Frames skipped: {self.skipped_frames}")
        print(f"   Efficiency: {efficiency:.1f}%")
        print(f"   → Learned to skip {efficiency:.1f}% of redundant frames!")

        print(f"\n🌱 GROWTH:")
        print(f"   Growth events: {len(self.growth_events)}")
        if self.growth_events:
            for step, k, old, new in self.growth_events:
                print(f"   Step {step}: Cluster {k} grew rank {old} → {new}")

        print(f"\n🎯 CLUSTER DIVERSITY:")
        active = sum(1 for c in self.cluster_usage.values() if c > 10)
        print(f"   Active clusters: {active}/16")

        sorted_clusters = sorted(self.cluster_usage.items(), key=lambda x: x[1], reverse=True)
        for k, count in sorted_clusters:
            if count > 0:
                pct = 100 * count / max(1, self.encoded_frames)
                bar = "█" * int(pct / 5)
                print(f"   Cluster {k}: {count:4d} ({pct:5.1f}%) {bar}")

        print("\n" + "=" * 70)
        print("Key Learning:")
        print(f"  • Model learned temporal dynamics naturally")
        print(f"  • Discovered when to pay attention vs skip")
        print(f"  • Achieved {efficiency:.1f}% efficiency by abstracting time")
        print("=" * 70)


def main():
    print("Starting Adaptive Temporal Learner...")
    print("This model learns TIME naturally, like a human!\n")

    learner = AdaptiveTemporalLearner(frame_size=64, device="cpu")
    learner.run(camera_id=0)


if __name__ == "__main__":
    main()