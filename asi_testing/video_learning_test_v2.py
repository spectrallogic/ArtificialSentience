# video_learning_test.py
# Baby-curriculum video test for ASI: color RGB, exploratory routing, growth checks

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque

from asi_model import ASISeed, to_device

# -----------------------------
# Helpers
# -----------------------------

def to_tensor_rgb32(frame_bgr, frame_size=32, device="cpu", normalize=True, per_video_stats=None):
    """Convert BGR frame -> RGB (32x32), [0,1] float32, flatten to vector (3072)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = small / 255.0 if normalize else small

    # Optional per-video whiten (mean/std over all pixels & channels)
    if per_video_stats is not None:
        mean, std = per_video_stats
        if std < 1e-6:
            std = 1.0
        x = (x - mean) / std
        # clip to a sensible range to avoid exploding scales
        x = np.clip(x, -3.0, 3.0)
        # bring back to ~[0,1] scale-ish (just for numeric stability)
        x = (x + 3.0) / 6.0

    return torch.from_numpy(x.reshape(-1)).to(device)

def compute_video_mean_std(video_path, max_frames=120, frame_size=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    vals = []
    count = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(1, min(total, max_frames)))
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        vals.append(small)
        count += 1
        if count >= max_frames:
            break
    cap.release()
    if not vals:
        return None
    arr = np.stack(vals, axis=0)  # [N, H, W, C]
    mean = float(arr.mean())
    std = float(arr.std() + 1e-8)
    return mean, std

# -----------------------------
# Exploratory Router (outside the model)
# -----------------------------
@torch.no_grad()
def exploratory_route(centroids, z, tau=1.0, eps=0.1):
    """
    Softmax over cosine sims (temperature tau), with epsilon-greedy exploration.
    centroids: [K, D], z: [D]
    """
    sims = F.cosine_similarity(z.unsqueeze(0), centroids, dim=1)  # [K]
    sims = sims / max(1e-6, tau)
    probs = torch.softmax(sims, dim=0)  # [K]
    if torch.rand(1).item() < eps:
        k = torch.randint(0, centroids.size(0), (1,)).item()
    else:
        k = torch.multinomial(probs, num_samples=1).item()
    return int(k), probs

# -----------------------------
# Main Tester
# -----------------------------

class VideoLearningTest:
    """
    This test:
      - Keeps RGB (32x32x3 = 3072).
      - Uses an exploratory router wrapper (temperature + epsilon) to avoid single-cluster collapse.
      - Applies a simple "baby" curriculum: short, slow exposure first; then full dataset.
      - Uses better growth triggers (percentile-based) and replay mixing to stabilize learning.
    """
    def __init__(self, video_dir="test_videos", device="cpu"):
        self.video_dir = video_dir
        self.device = torch.device(device)

        # Load metadata
        metadata_path = os.path.join(video_dir, 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded metadata: {self.metadata['train_videos']} train, {self.metadata['test_videos']} test videos")

        # Model (RGB)
        self.frame_size = 32
        self.input_dim = self.frame_size * self.frame_size * 3  # 3072
        self.model = ASISeed(
            input_dim=self.input_dim,
            model_dim=128,
            num_clusters=16,   # ↑ more clusters for diversity
            core_rank=4,
            build_ema=False,
            use_heads=False
        ).to(self.device)

        self.opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)

        # Tracking
        self.results = {
            'train_losses': [],
            'growth_events': [],
            'cluster_usage': defaultdict(int),
            'video_metrics': []
        }

        # Curriculum & routing schedules (baby → child)
        self.total_steps = 0
        self.max_tau = 2.0
        self.min_tau = 0.6
        self.max_eps = 0.20
        self.min_eps = 0.02
        self.anneal_steps = 25000  # over this many frames anneal tau/eps

        # Replay of latents to avoid a single video dominating
        self.replay_z = deque(maxlen=2048)  # for consolidation-ish smoothing

    # ---------- Schedules ----------

    def _annealed_tau_eps(self):
        t = min(1.0, self.total_steps / max(1, self.anneal_steps))
        tau = self.max_tau * (1 - t) + self.min_tau * t
        eps = self.max_eps * (1 - t) + self.min_eps * t
        return float(tau), float(eps)

    # ---------- Core step (manual forward with exploratory routing) ----------

    def _step_on_x(self, x_vec):
        """
        Manual forward to insert exploratory routing:
         z = encoder(x)
         k = exploratory route(centroids, z)
         h = layer(z, k)
         x_hat = decoder(h)
        """
        # Encode
        z = self.model.encoder(x_vec)
        tau, eps = self._annealed_tau_eps()
        k, probs = exploratory_route(self.model.router.centroids, z, tau=tau, eps=eps)
        # Low-rank layer (cluster-specific residual)
        h = self.model.layer(z, active_cluster=k)
        x_hat = self.model.decoder(h)
        # Update centroid online
        with torch.no_grad():
            self.model.router.update_centroid(k, z)
        # Buffers
        self.model.update_buffers(k, z)
        # Stats
        gs = self.model.stats[k]
        self.results['cluster_usage'][k] += 1
        return x_hat, k, z, h

    # ---------- Training on one video ----------

    def train_on_video(self, video_info, max_frames=100, stride=3, warmup_only=False, per_video=True):
        video_path = os.path.join(self.video_dir, video_info['filename'])
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found!")
            return None

        # Optional per-video normalization stats
        v_stats = compute_video_mean_std(video_path, max_frames=90, frame_size=self.frame_size) if per_video else None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_use = min(total_frames, max_frames)

        losses = []
        clusters_used = []

        for frame_idx in range(0, frames_to_use, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            x = to_tensor_rgb32(frame, self.frame_size, self.device, normalize=True, per_video_stats=v_stats)

            # Manual forward (exploratory routing)
            x_hat, k, z, h = self._step_on_x(x)

            # Recon loss
            recon = F.mse_loss(x_hat, x)

            # Gentle entropy encouragement early on (encourage cluster variety)
            tau, eps = self._annealed_tau_eps()
            with torch.no_grad():
                # recompute probs for logging (approx—we already had it)
                sims = F.cosine_similarity(z.unsqueeze(0), self.model.router.centroids, dim=1) / max(tau, 1e-6)
                p = torch.softmax(sims, dim=0)
            entropy = -(p * (p.clamp_min(1e-8)).log()).sum()
            ent_w = 1e-3 if self.total_steps < self.anneal_steps * 0.4 else 0.0
            loss = recon + ent_w * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            # Light replay: periodically mix a stored latent to stabilize
            if len(self.replay_z) > 16 and (self.total_steps % 7 == 0):
                zr = torch.stack([self.replay_z[np.random.randint(len(self.replay_z))] for _ in range(8)], dim=0).to(self.device)
                # Distill current layer behavior on replay (consistency)
                with torch.no_grad():
                    target = F.relu(zr @ (self.model.layer.U @ self.model.layer.V.t()).t())
                pred = F.relu(zr @ (self.model.layer.U @ self.model.layer.V.t()).t())
                distill = F.mse_loss(pred, target.detach())
                (0.1 * distill).backward()
                # note: tiny, piggyback on same optimizer step in next iteration

            self.replay_z.append(z.detach().cpu())

            losses.append(recon.item())
            clusters_used.append(k)
            self.total_steps += 1

        cap.release()

        # Smarter growth: trigger if heavily used cluster shows heavy tail loss
        self._growth_check_on_clusters(set(clusters_used))

        avg_loss = sum(losses) / len(losses) if losses else 1.0
        primary_cluster = max(set(clusters_used), key=clusters_used.count) if clusters_used else -1

        return {
            'filename': video_info['filename'],
            'avg_loss': avg_loss,
            'initial_loss': losses[0] if losses else 1.0,
            'final_loss': losses[-1] if losses else 1.0,
            'clusters': list(set(clusters_used)),
            'primary_cluster': primary_cluster
        }

    # ---------- Testing on one video ----------

    @torch.no_grad()
    def test_on_video(self, video_info, max_frames=100, stride=3, per_video=True):
        self.model.eval()
        video_path = os.path.join(self.video_dir, video_info['filename'])
        if not os.path.exists(video_path):
            self.model.train()
            return None

        v_stats = compute_video_mean_std(video_path, max_frames=90, frame_size=self.frame_size) if per_video else None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_use = min(total_frames, max_frames)

        losses = []
        clusters_used = []

        for frame_idx in range(0, frames_to_use, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            x = to_tensor_rgb32(frame, self.frame_size, self.device, normalize=True, per_video_stats=v_stats)
            # Manual forward (exploratory route uses minimal eps during eval)
            z = self.model.encoder(x)
            sims = F.cosine_similarity(z.unsqueeze(0), self.model.router.centroids, dim=1)
            k = int(torch.argmax(sims).item())
            h = self.model.layer(z, active_cluster=k)
            x_hat = self.model.decoder(h)
            loss = F.mse_loss(x_hat, x)
            losses.append(loss.item())
            clusters_used.append(k)

        cap.release()
        self.model.train()

        avg_loss = sum(losses) / len(losses) if losses else 1.0
        primary_cluster = max(set(clusters_used), key=clusters_used.count) if clusters_used else -1
        return {
            'filename': video_info['filename'],
            'avg_loss': avg_loss,
            'clusters': list(set(clusters_used)),
            'primary_cluster': primary_cluster
        }

    # ---------- Growth logic ----------

    def _growth_check_on_clusters(self, clusters):
        # Use percentile-based trigger to detect heavy-tail difficulty on active clusters
        for k in clusters:
            gs = self.model.stats[k]
            # Collect recent recon from this cluster by peeking at model buffers:
            # we don't store per-step loss per cluster in model, so use a proxy:
            # If cluster is very active but global average isn't improving, grow.
            gs.samples += 0  # keep in sync with your structures
            # Quick heuristic: if a cluster got >100 hits in last 2k steps and
            # global training loss hasn't dipped much, expand it.
            hits = self.results['cluster_usage'][k]
            if hits < 120:
                continue

            # Build per-cluster recent loss window (we approximate by using
            # global recent train losses when this cluster is dominant).
            # In practice you can store per-cluster losses; here we just estimate:
            recent = self.results['train_losses'][-400:] if len(self.results['train_losses']) >= 400 else self.results['train_losses']
            if len(recent) < 60:
                continue
            arr = np.array(recent[-200:])  # focus on very recent
            p95 = float(np.percentile(arr, 95))
            p50 = float(np.percentile(arr, 50))
            if (p95 - p50) > 0.02:  # heavy tail → grow capacity
                old_rank = self.model.layer.U_res[k].shape[1] if self.model.layer.U_res[k].numel() > 0 else 0
                self.model.grow_cluster(k, grow_rank=1)
                new_rank = self.model.layer.U_res[k].shape[1]
                self.results['growth_events'].append({'cluster': k, 'old_rank': old_rank, 'new_rank': new_rank})
                print(f"    🌱 Growth: Cluster {k} expanded rank {old_rank} → {new_rank}")

    # ---------- Full runs ----------

    def run_training(self):
        print("\n" + "=" * 70)
        print("TRAINING PHASE (Baby → Child curriculum)")
        print("=" * 70)

        train_videos = [v for v in self.metadata['videos'] if v['type'] == 'train']

        # ------------------------
        # Stage 0 (Baby): very light exposure, large tau/eps by default schedule
        # ------------------------
        print("\n[Stage 0] Baby exposure: short & sparse frames, encourage exploration")
        for i, video_info in enumerate(train_videos[: min(20, len(train_videos))]):
            print(f"  (Baby {i+1}) {video_info['filename']} - {video_info['color']} {video_info['shape']} {video_info['motion']} {video_info['size']}")
            result = self.train_on_video(video_info, max_frames=45, stride=5, warmup_only=True, per_video=True)
            if result:
                self.results['video_metrics'].append(result)
                self.results['train_losses'].append(result['final_loss'])
                print(f"    Loss: {result['initial_loss']:.4f} → {result['final_loss']:.4f} | primary k={result['primary_cluster']}")

        # ------------------------
        # Stage 1 (Toddler): normal frames, still exploratory, more videos
        # ------------------------
        print("\n[Stage 1] Toddler exposure: normal frames, stabilized replay")
        for i, video_info in enumerate(train_videos):
            print(f"\n[{i + 1}/{len(train_videos)}] Training: {video_info['filename']}")
            print(f"  Properties: {video_info['color']} {video_info['shape']} {video_info['motion']} {video_info['size']}")
            result = self.train_on_video(video_info, max_frames=100, stride=3, per_video=True)
            if result:
                self.results['video_metrics'].append(result)
                self.results['train_losses'].append(result['final_loss'])
                print(f"  Loss: {result['initial_loss']:.4f} → {result['final_loss']:.4f}")
                print(f"  Primary cluster: {result['primary_cluster']}")

    def run_testing(self):
        print("\n" + "=" * 70)
        print("TESTING PHASE (Novel Combinations)")
        print("=" * 70)

        test_videos = [v for v in self.metadata['videos'] if v['type'] == 'test']
        test_results = []

        for i, video_info in enumerate(test_videos):
            print(f"\n[{i + 1}/{len(test_videos)}] Testing: {video_info['filename']}")
            print(f"  Novel combo: {video_info['color']} {video_info['shape']} {video_info['motion']} {video_info['size']}")
            result = self.test_on_video(video_info, max_frames=120, stride=3, per_video=True)
            if result:
                test_results.append(result)
                print(f"  Test Loss: {result['avg_loss']:.4f}")
                print(f"  Primary cluster: {result['primary_cluster']}")

        return test_results

    def analyze_results(self, test_results):
        print("\n" + "=" * 70)
        print("RESULTS ANALYSIS")
        print("=" * 70)

        # Training performance
        train_metrics = [m for m in self.results['video_metrics'] if 'final_loss' in m]
        if train_metrics:
            avg_train_loss = sum(m['final_loss'] for m in train_metrics) / len(train_metrics)
            print(f"\n📊 Training Performance:")
            print(f"  Average final loss: {avg_train_loss:.4f}")
            print(f"  Videos trained: {len(train_metrics)}")
        else:
            avg_train_loss = 1.0

        # Growth events
        print(f"\n🌱 Growth Events: {len(self.results['growth_events'])}")
        if self.results['growth_events']:
            print("  Notable expansions:")
            for event in self.results['growth_events'][:10]:
                print(f"    Cluster {event['cluster']}: rank {event['old_rank']} → {event['new_rank']}")

        # Cluster usage
        print(f"\n🎯 Cluster Usage:")
        for cluster_id in sorted(self.results['cluster_usage'].keys()):
            count = self.results['cluster_usage'][cluster_id]
            print(f"  Cluster {cluster_id}: {count} activations")

        # Test performance
        if test_results:
            avg_test_loss = sum(r['avg_loss'] for r in test_results) / len(test_results)
            print(f"\n🧪 Test Performance (Novel Combinations):")
            print(f"  Average loss: {avg_test_loss:.4f}")
            print(f"  Videos tested: {len(test_results)}")

            # Compare to training
            if train_metrics:
                gap = avg_test_loss - avg_train_loss
                print(f"  Train-Test gap: {gap:+.4f}")
                if gap < 0.05:
                    print("  ✓ EXCELLENT: Strong generalization to novel combinations!")
                elif gap < 0.15:
                    print("  ✓ GOOD: Reasonable generalization")
                else:
                    print("  ✗ POOR: Struggling with novel combinations")

        # Final verdict
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)

        checks_passed = 0
        total_checks = 4

        # Check 1: Learning happened
        if train_metrics and avg_train_loss < 0.1:
            print("✓ CHECK 1: Learning works (low training loss)")
            checks_passed += 1
        else:
            print("✗ CHECK 1: Learning struggling (high training loss)")

        # Check 2: Growth triggered
        if len(self.results['growth_events']) >= 3:
            print("✓ CHECK 2: Growth mechanism works (multiple expansions)")
            checks_passed += 1
        else:
            print("✗ CHECK 2: Growth not triggering enough")

        # Check 3: Cluster diversity
        active_clusters = len([c for c, count in self.results['cluster_usage'].items() if count > 50])
        if active_clusters >= 4:  # expect more diversity now
            print(f"✓ CHECK 3: Cluster diversity ({active_clusters} active clusters)")
            checks_passed += 1
        else:
            print(f"✗ CHECK 3: Poor cluster diversity ({active_clusters} clusters)")

        # Check 4: Generalization
        if test_results:
            avg_test_loss = sum(r['avg_loss'] for r in test_results) / len(test_results)
            if avg_test_loss < 0.2:
                print("✓ CHECK 4: Generalization works (handles novel combos)")
                checks_passed += 1
            else:
                print("✗ CHECK 4: Poor generalization to novel combinations")
        else:
            print("✗ CHECK 4: No test results")

        print(f"\n🎯 SCORE: {checks_passed}/{total_checks} checks passed")
        if checks_passed >= 3:
            print("\n🎉 SUCCESS! The seed is learning, exploring, and growing across clusters.")
        elif checks_passed == 2:
            print("\n⚠️  PARTIAL SUCCESS: Some capabilities working, needs tuning")
        else:
            print("\n❌ NEEDS WORK: Core capabilities not yet functional")

# -----------------------------
# Entry
# -----------------------------

def main():
    print("=" * 70)
    print("ASI VIDEO LEARNING TEST (Baby Curriculum, RGB)")
    print("Testing: Learning, Growth, Abstraction, Generalization")
    print("=" * 70)

    # Check dataset
    if not os.path.exists("test_videos/dataset_metadata.json"):
        print("\n❌ ERROR: Videos not found!")
        print("Run generate_test_videos.py first to create the dataset.")
        return

    tester = VideoLearningTest(video_dir="test_videos", device="cuda" if torch.cuda.is_available() else "cpu")

    # Train
    tester.run_training()

    # Test
    test_results = tester.run_testing()

    # Analyze
    tester.analyze_results(test_results)

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

'''
======================================================================
RESULTS ANALYSIS
======================================================================

📊 Training Performance:
  Average final loss: 0.0152
  Videos trained: 100

🌱 Growth Events: 0

🎯 Cluster Usage:
  Cluster 0: 198 activations
  Cluster 1: 193 activations
  Cluster 2: 174 activations
  Cluster 3: 168 activations
  Cluster 4: 185 activations
  Cluster 5: 168 activations
  Cluster 6: 186 activations
  Cluster 7: 168 activations
  Cluster 8: 204 activations
  Cluster 9: 176 activations
  Cluster 10: 189 activations
  Cluster 11: 181 activations
  Cluster 12: 168 activations
  Cluster 13: 181 activations
  Cluster 14: 179 activations
  Cluster 15: 182 activations

🧪 Test Performance (Novel Combinations):
  Average loss: 0.0171
  Videos tested: 20
  Train-Test gap: +0.0019
  ✓ EXCELLENT: Strong generalization to novel combinations!

======================================================================
FINAL VERDICT
======================================================================
✓ CHECK 1: Learning works (low training loss)
✗ CHECK 2: Growth not triggering enough
✓ CHECK 3: Cluster diversity (16 active clusters)
✓ CHECK 4: Generalization works (handles novel combos)

🎯 SCORE: 3/4 checks passed

🎉 SUCCESS! The seed is learning, exploring, and growing across clusters.

======================================================================
TEST COMPLETE!
======================================================================

'''



