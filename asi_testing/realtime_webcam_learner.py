# realtime_webcam_learner.py
# Continuous temporal prediction: AI always predicts the next frame
# LEFT: What actually happens | RIGHT: What AI thinks will happen next

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
import sys
import os

# Import your ASI model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from asi_model import ASISeed, to_device


class TemporalPredictor:
    """
    Always-predicting temporal learner.

    Core idea:
    1. AI sees current frame
    2. AI predicts what the NEXT frame will be
    3. Compare prediction to actual next frame
    4. Learn from prediction error
    5. Repeat continuously

    This forces the AI to discover temporal patterns:
    - Motion continuity (hand moving left continues left)
    - Lighting changes
    - Face movements
    - Background stability
    """

    def __init__(self,
                 frame_size=32,
                 temporal_window=6,  # Remember last 6 frames for better motion learning
                 creativity=0.22,  # Slightly more randomness for variety
                 device="cpu"):

        self.frame_size = frame_size
        self.temporal_window = temporal_window
        self.creativity = creativity
        self.device = torch.device(device)

        # Input: current frame + motion context
        self.input_dim = frame_size * frame_size

        # IMPORTANT: Define num_clusters FIRST before using it anywhere
        self.num_clusters = 7

        print(f"📹 Initializing Enhanced Temporal Predictor")
        print(f"   Frame size: {frame_size}x{frame_size} ({self.input_dim} pixels)")
        print(f"   Temporal window: {temporal_window} frames (better motion memory)")
        print(f"   Creativity level: {creativity} (0=deterministic, 1=chaotic)")
        print(f"   Device: {device}")

        # ASI model for temporal prediction (bigger for 32x32 = 1024 dims)
        self.model = ASISeed(
            input_dim=self.input_dim,
            model_dim=512,  # Increased for higher resolution
            num_clusters=self.num_clusters,
            core_rank=16,  # More capacity for detailed predictions
            build_ema=False,  # We want raw predictions, not smoothed
            use_heads=True  # Use prediction head
        ).to(self.device)

        # Optimizer (slightly higher LR for bigger model)
        self.opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=2e-3
        )

        # Temporal memory: recent frames + motion deltas
        self.frame_history = deque(maxlen=temporal_window)
        self.vector_history = deque(maxlen=temporal_window)
        self.motion_history = deque(maxlen=temporal_window)  # Track movement between frames

        # Tracking
        self.total_frames = 0
        self.prediction_losses = []
        self.cluster_history = []

        # Continuous learning & growth tracking
        self.cluster_stats = {i: {"samples": 0, "recent_losses": deque(maxlen=50), "growth_count": 0}
                              for i in range(self.num_clusters)}
        self.total_growth_events = 0
        self.last_consolidation = 0
        self.consolidation_interval = 300  # Consolidate every 300 frames (~10 seconds)
        self.growth_check_interval = 50  # Check for growth every 50 frames
        self.plateau_threshold = 0.15  # If loss stays above this, consider growth

        # Current prediction (always available)
        self.current_prediction = None
        self.last_latent = None
        self.last_cluster = 0
        self.motion_momentum = None  # For predicting motion continuity

        # Display size
        self.display_size = 320

        # Performance tracking
        self.prediction_accuracy_history = []

    def frame_to_vector(self, frame):
        """Convert frame to vector"""
        small = cv2.resize(frame, (self.frame_size, self.frame_size))

        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        small = small.astype(np.float32) / 255.0
        vector = small.flatten()

        return torch.from_numpy(vector).float()

    def vector_to_frame(self, vector):
        """Convert vector to displayable frame"""
        vector_np = vector.detach().cpu().numpy()
        vector_np = np.clip(vector_np, 0, 1)

        frame_2d = vector_np.reshape(self.frame_size, self.frame_size)
        frame_2d = (frame_2d * 255).astype(np.uint8)

        # Upscale for display
        frame_display = cv2.resize(frame_2d, (self.display_size, self.display_size),
                                   interpolation=cv2.INTER_NEAREST)

        return frame_display

    def get_temporal_context(self):
        """
        Create temporal context from recent frames.
        Now includes motion information for better pattern learning.
        """
        if len(self.vector_history) < 2:
            return None

        # Compute average motion from recent frames
        recent_motion = []
        for i in range(1, min(4, len(self.vector_history))):
            delta = self.vector_history[-i] - self.vector_history[-i - 1]
            recent_motion.append(delta)

        if recent_motion:
            # Average motion gives us velocity/momentum
            avg_motion = torch.stack(recent_motion).mean(dim=0)
            return avg_motion

        return None

    def check_and_grow(self, cluster, loss):
        """
        Check if this cluster needs to grow (seeing new patterns it can't handle).
        This is the "hungry matrix" in action!
        """
        stats = self.cluster_stats[cluster]
        stats["recent_losses"].append(loss)

        # Only check if we have enough samples
        if len(stats["recent_losses"]) < 50:
            return False

        # Check if loss is plateaued at a high level (struggling with complexity)
        avg_loss = sum(stats["recent_losses"]) / len(stats["recent_losses"])

        # If loss is high and not improving, this cluster needs more capacity
        if avg_loss > self.plateau_threshold:
            # Check if we're actually plateaued (not improving)
            first_half = list(stats["recent_losses"])[:25]
            second_half = list(stats["recent_losses"])[25:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            improvement = first_avg - second_avg

            # If improvement is minimal, we need more capacity
            if improvement < 0.01:
                print(f"\n🌱 GROWTH EVENT: Cluster {cluster} struggling with new patterns!")
                print(f"   Current loss: {avg_loss:.4f}, Improvement: {improvement:.4f}")
                print(f"   Growing capacity to handle complexity...")

                # Grow this cluster
                self.model.grow_cluster(cluster, grow_rank=2)
                stats["growth_count"] += 1
                self.total_growth_events += 1

                # Clear recent losses to give it a fresh start
                stats["recent_losses"].clear()

                print(f"   ✓ Growth complete! Total expansions: {stats['growth_count']}")
                return True

        return False

    def consolidate_learning(self):
        """
        Periodically consolidate learning - like sleep/memory consolidation.
        Distills learned patterns into core abstractions.
        """
        print(f"\n💤 CONSOLIDATION: Solidifying learned patterns into core memory...")

        # Consolidate each active cluster
        active_clusters = [k for k in range(self.model.num_clusters)
                           if self.cluster_stats[k]["samples"] > 20]

        if not active_clusters:
            print("   No clusters to consolidate yet")
            return

        for k in active_clusters:
            if len(self.model.buffers[k]) > 10:
                self.model.consolidate_cluster(k)
                print(f"   ✓ Cluster {k} consolidated ({len(self.model.buffers[k])} samples)")

        print(f"   ✓ Consolidation complete! Memory solidified.\n")

    def detect_novelty(self, cluster, loss):
        """
        Detect if we're seeing something genuinely new vs just noise.
        High loss + unfamiliar cluster = novelty!
        """
        stats = self.cluster_stats[cluster]

        # Is this cluster rarely used?
        total_samples = sum(s["samples"] for s in self.cluster_stats.values())
        cluster_usage = stats["samples"] / max(1, total_samples)

        # Is the loss high?
        is_struggling = loss > 0.2

        # Is this a rare cluster?
        is_rare = cluster_usage < 0.05 and stats["samples"] < 20

        return is_struggling and is_rare

    def compute_motion_delta(self, current_vector, previous_vector):
        """Compute motion delta between two frame vectors."""
        if previous_vector is None:
            return torch.zeros_like(current_vector)
        return current_vector - previous_vector

    def predict_next_frame(self, current_vector):
        """
        Predict what the next frame will look like using temporal context.
        Uses creative sampling + motion momentum for realistic movement.

        Input: current frame vector
        Output: predicted next frame vector (with creative variation + motion)
        """
        with torch.no_grad():
            current_vector = current_vector.to(self.device)

            # Encode current frame to latent space
            z = self.model.encoder(current_vector)
            k = self.model.router(z)

            # Get motion context (velocity/momentum from recent frames)
            motion_context = self.get_temporal_context()
            if motion_context is not None:
                motion_context = motion_context.to(self.device)
                # Store momentum for continuity
                self.motion_momentum = motion_context

            # CREATIVE SAMPLING: Add noise to latent (like diffusion/VAE sampling)
            # This makes it "imagine" rather than just copy
            if self.creativity > 0:
                # Sample from a distribution around the latent
                latent_noise = torch.randn_like(z) * self.creativity
                z_creative = z + latent_noise

                # Keep it within reasonable bounds (prevent explosion)
                z_creative = torch.clamp(z_creative, -10, 10)
            else:
                z_creative = z

            # Generate from the creative latent
            h = self.model.layer(z_creative, active_cluster=k)

            # Decode to frame prediction
            x_next_pred = self.model.decoder(h)

            # Add motion momentum (predict motion continues)
            if self.motion_momentum is not None and len(self.vector_history) >= 2:
                # Apply a fraction of the momentum to the prediction
                # This makes movement patterns continue (hand moving left keeps moving left)
                momentum_strength = 0.4  # How much motion carries forward
                x_next_pred = x_next_pred + momentum_strength * self.motion_momentum.to(self.device)

            # Add pixel-level variation for texture (like image generators do)
            if self.creativity > 0:
                pixel_noise = torch.randn_like(x_next_pred) * (self.creativity * 0.08)
                x_next_pred = x_next_pred + pixel_noise
                x_next_pred = torch.clamp(x_next_pred, 0, 1)

            # Store for learning later
            self.last_latent = z  # Store clean latent, not noisy one
            self.last_cluster = k

            return x_next_pred, k

    def learn_from_prediction_error(self, predicted_vector, actual_vector):
        """
        Learn from the difference between prediction and reality.
        Now also learns from temporal patterns (motion).
        """
        predicted_vector = predicted_vector.to(self.device)
        actual_vector = actual_vector.to(self.device)

        # Get current encoding
        x_hat, k, z, h = self.model(actual_vector)

        # Three losses:
        # 1. Reconstruction loss (can it understand the current frame?)
        recon_loss = F.mse_loss(x_hat, actual_vector)

        # 2. Prediction loss (can it predict the next frame?)
        pred_loss = self.model.prediction_loss(h, actual_vector)

        # 3. Motion consistency loss (does predicted motion match actual motion?)
        motion_loss = torch.tensor(0.0, device=self.device)
        if len(self.vector_history) >= 2:
            # Compute actual motion that just happened
            actual_motion = actual_vector - self.vector_history[-1].to(self.device)
            # Compute predicted motion
            if self.current_prediction is not None:
                pred_motion = predicted_vector - self.vector_history[-1].to(self.device)
                # Loss: how well did we predict the motion?
                motion_loss = F.mse_loss(pred_motion, actual_motion)

        # Total loss (emphasize prediction and motion)
        loss = 0.2 * recon_loss + 0.5 * pred_loss + 0.3 * motion_loss

        # Backprop
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Update router
        with torch.no_grad():
            self.model.router.update_centroid(k, z)
        self.model.update_buffers(k, z)

        # Track
        self.prediction_losses.append(pred_loss.item())
        self.cluster_history.append(k)

        # Track cluster-specific stats for growth detection
        self.cluster_stats[k]["samples"] += 1

        # Calculate prediction accuracy (lower loss = better prediction)
        accuracy = max(0, 1 - pred_loss.item() * 10)  # Rough metric
        self.prediction_accuracy_history.append(accuracy)

        return loss.item(), pred_loss.item(), k

    def create_display(self, webcam_frame, predicted_frame, total_loss, pred_loss, cluster, fps):
        """
        Create side-by-side display showing reality vs prediction.
        """
        # Prepare webcam frame (reality)
        webcam_display = cv2.resize(webcam_frame, (self.display_size, self.display_size))

        # Prepare predicted frame (AI's guess)
        if len(predicted_frame.shape) == 2:
            predicted_display = cv2.cvtColor(predicted_frame, cv2.COLOR_GRAY2BGR)
        else:
            predicted_display = predicted_frame.copy()

        if predicted_display.shape[:2] != (self.display_size, self.display_size):
            predicted_display = cv2.resize(predicted_display, (self.display_size, self.display_size))

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Left panel: REALITY
        cv2.putText(webcam_display, "REALITY (Now)", (10, 30),
                    font, 0.7, (0, 255, 0), 2)

        # Stats on left
        stats_y = 60
        stats = [
            f"Frame: {self.total_frames}",
            f"Loss: {total_loss:.4f}",
            f"Pred: {pred_loss:.4f}",
            f"Cluster: {cluster}",
            f"Growths: {self.total_growth_events}",
            f"FPS: {fps:.1f}",
            f"Memory: {len(self.frame_history)}/{self.temporal_window}",
            f"Creativity: {self.creativity:.2f}",
        ]

        for stat in stats:
            cv2.putText(webcam_display, stat, (10, stats_y),
                        font, 0.4, (255, 255, 255), 1)
            stats_y += 20

        # Right panel: GENERATION (not just prediction)
        gen_label = "AI GENERATION" if self.creativity > 0.05 else "AI PREDICTION"
        cv2.putText(predicted_display, gen_label, (10, 30),
                    font, 0.6, (255, 255, 0), 2)

        # Prediction quality indicator
        if len(self.prediction_accuracy_history) > 0:
            accuracy = self.prediction_accuracy_history[-1]
            acc_text = f"Accuracy: {accuracy * 100:.1f}%"
            color = (0, 255, 0) if accuracy > 0.7 else (0, 165, 255) if accuracy > 0.4 else (0, 0, 255)
            cv2.putText(predicted_display, acc_text, (10, 60),
                        font, 0.5, color, 1)

        # Creativity level indicator (visual bar)
        creativity_text = f"Creativity: {self.creativity:.2f}"
        cv2.putText(predicted_display, creativity_text, (10, 85),
                    font, 0.4, (255, 255, 255), 1)

        # Draw creativity bar
        bar_x, bar_y = 10, 95
        bar_width = int(self.creativity * 200)  # Max 200px wide
        bar_height = 8
        # Background bar (gray)
        cv2.rectangle(predicted_display, (bar_x, bar_y), (bar_x + 200, bar_y + bar_height),
                      (50, 50, 50), -1)
        # Creativity bar (gradient from green to red)
        bar_color = (0, int(255 * (1 - self.creativity)), int(255 * self.creativity))
        cv2.rectangle(predicted_display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      bar_color, -1)

        # Motion momentum indicator (shows if predicting motion continuity)
        if self.motion_momentum is not None:
            momentum_mag = torch.abs(self.motion_momentum).mean().item()
            if momentum_mag > 0.03:
                momentum_text = f"Momentum: {momentum_mag:.3f}"
                cv2.putText(predicted_display, momentum_text, (10, 115),
                            font, 0.4, (0, 255, 255), 1)

        # Growth indicator (show total growth events)
        if self.total_growth_events > 0:
            growth_text = f"Adaptations: {self.total_growth_events}"
            cv2.putText(predicted_display, growth_text, (10, 135),
                        font, 0.4, (0, 255, 0), 1)

        # Visual indicator: how good is the prediction?
        if len(self.prediction_losses) > 0:
            recent_pred_loss = np.mean(self.prediction_losses[-30:])
            if recent_pred_loss < 0.05:
                status = "EXCELLENT"
                status_color = (0, 255, 0)
            elif recent_pred_loss < 0.1:
                status = "GOOD"
                status_color = (0, 255, 255)
            elif recent_pred_loss < 0.2:
                status = "LEARNING..."
                status_color = (0, 165, 255)
            else:
                status = "GUESSING"
                status_color = (0, 0, 255)

            cv2.putText(predicted_display, status, (10, self.display_size - 20),
                        font, 0.6, status_color, 2)

        # Add border to prediction panel
        border_color = (0, 255, 0) if pred_loss < 0.1 else (0, 165, 255) if pred_loss < 0.2 else (0, 0, 255)
        cv2.rectangle(predicted_display, (0, 0),
                      (self.display_size - 1, self.display_size - 1),
                      border_color, 2)

        # Combine side-by-side
        combined = np.hstack([webcam_display, predicted_display])

        return combined

    def run(self):
        """
        Main loop: continuous temporal prediction.

        Flow:
        1. Read frame from webcam
        2. Predict what NEXT frame will be
        3. Show prediction on right side
        4. Read actual next frame
        5. Learn from prediction error
        6. Repeat
        """
        print("\n" + "=" * 70)
        print("🤖 CONTINUOUS TEMPORAL PREDICTION")
        print("=" * 70)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Show stats")
        print("  'r' - Reset model (start fresh)")
        print("  '+' - Increase creativity (more random/imaginative)")
        print("  '-' - Decrease creativity (more deterministic)")
        print("  '0' - Reset creativity to default (0.15)")
        print("\nDisplay:")
        print("  LEFT:  Reality (what's actually happening)")
        print("  RIGHT: AI's prediction (what it thinks happens next)")
        print("\nThe AI is ALWAYS learning and adapting:")
        print("  → Remembers last 6 frames")
        print("  → Tracks motion patterns (velocity/momentum)")
        print("  → Predicts motion continuity")
        print("  → Generates with creative variation")
        print("  → Learns from prediction errors")
        print("  → GROWS when seeing new patterns it can't handle")
        print("  → Consolidates learning periodically (like sleep)")
        print("\nWatch for:")
        print("  🌱 GROWTH events (model expands capacity)")
        print("  💤 CONSOLIDATION (memory solidification)")
        print("  ✨ NEW PATTERN alerts (novelty detection)")
        print("\nWatch it learn patterns like:")
        print("  • Motion continuity (hand moving left keeps moving left)")
        print("  • Velocity prediction (fast motion continues fast)")
        print("  • Lighting changes")
        print("  • Your movement habits")
        print("  • NEW scenes/objects (adapts in real-time!)")
        print("=" * 70)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        print("\n✓ Webcam opened successfully!")
        print("🎬 Starting continuous prediction...\n")

        # Timing
        last_fps_time = time.time()
        fps_counter = 0
        current_fps = 0

        # Initialize with first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab initial frame")
            return

        current_vector = self.frame_to_vector(frame)
        self.frame_history.append(frame.copy())
        self.vector_history.append(current_vector)

        # Default predicted frame (black at start)
        predicted_frame = np.zeros((self.display_size, self.display_size), dtype=np.uint8)

        total_loss = 0.0
        pred_loss = 0.0
        cluster = 0

        try:
            while True:
                # FPS calculation
                fps_counter += 1
                if time.time() - last_fps_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    last_fps_time = time.time()

                # STEP 1: Predict next frame based on current state
                if len(self.vector_history) > 0:
                    predicted_vector, cluster = self.predict_next_frame(current_vector)
                    predicted_frame = self.vector_to_frame(predicted_vector)

                # STEP 2: Read ACTUAL next frame from webcam
                ret, next_frame = cap.read()
                if not ret:
                    break

                next_vector = self.frame_to_vector(next_frame)

                # STEP 2.5: Track motion for temporal learning
                if len(self.vector_history) > 0:
                    motion = self.compute_motion_delta(next_vector, self.vector_history[-1])
                    self.motion_history.append(motion)

                # STEP 3: Display BEFORE learning (so you see the prediction vs reality)
                combined = self.create_display(next_frame, predicted_frame,
                                               total_loss, pred_loss, cluster, current_fps)
                cv2.imshow('Temporal Prediction - Reality vs AI', combined)

                # STEP 4: Learn from prediction error
                if self.current_prediction is not None:
                    total_loss, pred_loss, cluster = self.learn_from_prediction_error(
                        predicted_vector, next_vector
                    )

                    # CONTINUOUS LEARNING: Check if we need to grow
                    if self.total_frames % self.growth_check_interval == 0:
                        grew = self.check_and_grow(cluster, pred_loss)
                        if grew:
                            # Visual feedback of growth
                            print(f"   🧠 Model adapted! Now has more capacity for cluster {cluster}")

                    # CONSOLIDATION: Periodically solidify learning
                    if self.total_frames - self.last_consolidation >= self.consolidation_interval:
                        self.consolidate_learning()
                        self.last_consolidation = self.total_frames

                    # NOVELTY DETECTION: Alert when seeing something new
                    if self.detect_novelty(cluster, pred_loss):
                        print(f"   ✨ NEW PATTERN detected in cluster {cluster}! Learning...")

                # STEP 5: Update state for next iteration
                current_vector = next_vector
                self.frame_history.append(next_frame.copy())
                self.vector_history.append(next_vector)
                self.total_frames += 1
                self.current_prediction = predicted_vector

                # Show learning progress periodically
                if self.total_frames % 150 == 0:
                    avg_pred_loss = np.mean(self.prediction_losses[-150:]) if self.prediction_losses else 0
                    print(f"[Frame {self.total_frames}] Avg prediction loss: {avg_pred_loss:.4f}")

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('s'):
                    self.print_stats()
                elif key == ord('r'):
                    print("\n🔄 Resetting model...")
                    self.model = ASISeed(
                        input_dim=self.input_dim,
                        model_dim=512,  # Match initialization
                        num_clusters=self.num_clusters,
                        core_rank=16,  # Match initialization
                        build_ema=False,
                        use_heads=True
                    ).to(self.device)
                    self.opt = torch.optim.Adam(
                        [p for p in self.model.parameters() if p.requires_grad],
                        lr=2e-3
                    )
                    self.prediction_losses = []
                    self.cluster_history = []
                    # Reset growth stats
                    self.cluster_stats = {i: {"samples": 0, "recent_losses": deque(maxlen=50), "growth_count": 0}
                                          for i in range(self.num_clusters)}
                    self.total_growth_events = 0
                    self.last_consolidation = 0
                    print("✓ Model reset complete")
                elif key == ord('+') or key == ord('='):
                    self.creativity = min(1.0, self.creativity + 0.05)
                    print(f"\n🎨 Creativity increased to {self.creativity:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.creativity = max(0.0, self.creativity - 0.05)
                    print(f"\n🎨 Creativity decreased to {self.creativity:.2f}")
                elif key == ord('0'):
                    self.creativity = 0.22
                    print(f"\n🎨 Creativity reset to {self.creativity:.2f}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()

    def print_stats(self):
        """Print current statistics"""
        print("\n" + "=" * 60)
        print("📊 CURRENT STATS")
        print("=" * 60)
        print(f"Total frames: {self.total_frames}")
        print(f"Total growth events: {self.total_growth_events}")

        if self.prediction_losses:
            recent_losses = self.prediction_losses[-100:]
            print(f"\nPrediction performance (last 100 frames):")
            print(f"  Average loss: {np.mean(recent_losses):.4f}")
            print(f"  Best loss: {np.min(recent_losses):.4f}")
            print(f"  Worst loss: {np.max(recent_losses):.4f}")

        if self.prediction_accuracy_history:
            recent_acc = self.prediction_accuracy_history[-100:]
            print(f"\nPrediction accuracy:")
            print(f"  Average: {np.mean(recent_acc) * 100:.1f}%")

        if self.cluster_history:
            print(f"\nCluster usage (last 100 frames):")
            from collections import Counter
            cluster_counts = Counter(self.cluster_history[-100:])
            for k, count in sorted(cluster_counts.items()):
                pct = (count / min(100, len(self.cluster_history))) * 100
                growths = self.cluster_stats[k]["growth_count"]
                growth_str = f" (grew {growths}x)" if growths > 0 else ""
                print(f"  Cluster {k}: {count} ({pct:.1f}%){growth_str}")

        print("=" * 60)

    def print_final_stats(self):
        """Print final statistics"""
        print("\n" + "=" * 70)
        print("📈 FINAL CONTINUOUS LEARNING STATS")
        print("=" * 70)
        print(f"Total frames predicted: {self.total_frames}")
        print(f"Runtime: ~{self.total_frames / 30:.1f} seconds")
        print(f"Total growth events: {self.total_growth_events}")
        print(f"Consolidations performed: {self.total_frames // self.consolidation_interval}")

        if self.prediction_losses and len(self.prediction_losses) > 100:
            print(f"\nPrediction loss progression:")
            early = np.mean(self.prediction_losses[:100])
            late = np.mean(self.prediction_losses[-100:])
            improvement = early - late
            print(f"  Initial (first 100):  {early:.4f}")
            print(f"  Final (last 100):     {late:.4f}")
            print(f"  Improvement:          {improvement:.4f} ({improvement / early * 100:.1f}%)")

            print(f"\n💡 Interpretation:")
            if improvement > 0.05:
                print("  ✓✓ STRONG continuous learning!")
                print("  ✓  AI continuously adapted to new patterns")
            elif improvement > 0.02:
                print("  ✓ MODERATE continuous learning")
                print("  → AI discovered and adapted to some patterns")
            elif improvement > 0:
                print("  ⚠ WEAK learning detected")
                print("  → Try more varied movements for better adaptation")
            else:
                print("  ⚠ No improvement (or got worse)")
                print("  → Scene might be too static or chaotic")

        if self.cluster_history:
            print(f"\n🎯 Cluster discovery & growth:")
            unique_clusters = len(set(self.cluster_history))
            print(f"  Clusters used: {unique_clusters}/7")

            from collections import Counter
            cluster_counts = Counter(self.cluster_history)
            print(f"  Distribution & adaptations:")
            for k, count in sorted(cluster_counts.items()):
                pct = (count / len(self.cluster_history)) * 100
                stats = self.cluster_stats[k]
                growths = stats["growth_count"]
                samples = stats["samples"]
                growth_str = f", grew {growths}x" if growths > 0 else ""
                print(f"    Cluster {k}: {count} frames ({pct:.1f}%), {samples} samples{growth_str}")

            if unique_clusters >= 5:
                print("  ✓ Good diversity - multiple temporal patterns found!")
            elif unique_clusters >= 3:
                print("  ⚠ Moderate diversity")
            else:
                print("  ⚠ Low diversity - mostly one pattern")

            if self.total_growth_events > 0:
                print(f"\n  🌱 Growth occurred {self.total_growth_events} times!")
                print(f"     The model ADAPTED to handle new complexity!")
                print(f"     This is the 'hungry matrix' learning in action!")

        print("\n🚀 This demonstrated:")
        print("  • Continuous temporal prediction (no batch processing)")
        print("  • Motion memory (6-frame working memory)")
        print("  • Velocity/momentum prediction (physics-like)")
        print("  • Self-supervised learning (learns from prediction errors)")
        print("  • DYNAMIC GROWTH (model expands when seeing new patterns)")
        print("  • CONSOLIDATION (memory solidification like sleep)")
        print("  • Pattern discovery (clusters = different motion types)")
        print("  • Creative generation (samples variations like video generators)")
        print("  • Real-time adaptation (always learning, always improving)")
        print("=" * 70)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    CONTINUOUS LEARNING TEMPORAL GENERATOR - "HUNGRY MATRIX"   ║
    ║                                                               ║
    ║  LEFT:  What actually happens (reality)                      ║
    ║  RIGHT: What AI generates/imagines (creative prediction)     ║
    ║                                                               ║
    ║  The AI NEVER STOPS LEARNING:                                ║
    ║    • Learns from every frame (real-time adaptation)          ║
    ║    • GROWS when it sees new patterns (dynamic capacity)      ║
    ║    • Consolidates learning like sleep (memory formation)     ║
    ║    • Tracks motion with 6-frame working memory               ║
    ║    • Predicts with creative variation (like video gen)       ║
    ║                                                               ║
    ║  Watch it adapt to YOU in real-time!                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    print("Configuration:")
    print("  • 32×32 pixel frames (1024 dimensions)")
    print("  • 6-frame temporal window (motion memory)")
    print("  • Motion momentum tracking (velocity prediction)")
    print("  • Learning: reconstruction + prediction + motion")
    print("  • 7 clusters for different motion patterns")
    print("  • 512 hidden dims, rank 16 (high capacity)")
    print("  • Creativity: 0.22 (adjustable with +/- keys)")
    print()
    print("🧠 CONTINUOUS LEARNING FEATURES:")
    print("  • Auto-growth: Expands capacity when seeing new patterns")
    print("  • Growth check: Every 50 frames")
    print("  • Consolidation: Every 300 frames (~10 seconds)")
    print("  • Novelty detection: Alerts when learning something new")
    print()
    print("Creativity levels:")
    print("  0.00 = Deterministic (ghost trail)")
    print("  0.15 = Subtle variation")
    print("  0.22 = Balanced (default - imagination + accuracy)")
    print("  0.35 = High creativity (multiple possibilities)")
    print("  0.50+ = Abstract/dreamy (artistic)")
    print()
    print("Watch the border color on the right panel:")
    print("  🟢 Green = Excellent prediction")
    print("  🟡 Yellow = Good prediction")
    print("  🔵 Blue = Still learning")
    print("  🔴 Red = Poor prediction (guessing)")
    print()

    input("Press ENTER to start continuous prediction...")

    predictor = TemporalPredictor(
        frame_size=16,  # 32x32 resolution
        temporal_window=6,
        creativity=0.12,  # Enhanced creativity
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    predictor.run()


if __name__ == "__main__":
    main()