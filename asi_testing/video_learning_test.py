# video_learning_test.py
# Comprehensive test of ASI learning from 100 generated videos

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from asi_model import ASISeed, to_device
import matplotlib.pyplot as plt
from collections import defaultdict


class VideoLearningTest:
    def __init__(self, video_dir="test_videos", device="cpu"):
        self.video_dir = video_dir
        self.device = torch.device(device)

        # Load metadata
        metadata_path = os.path.join(video_dir, 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded metadata: {self.metadata['train_videos']} train, {self.metadata['test_videos']} test videos")

        # Initialize model
        # Video frames: 64x64x3 = 12,288 dims → Too large!
        # Use smaller encoding
        self.frame_size = 32
        self.input_dim = self.frame_size * self.frame_size * 3  # 32x32x3 = 3072 (RGB)

        self.model = ASISeed(
            input_dim=self.input_dim,
            model_dim=128,
            num_clusters=10,  # More clusters for diverse videos
            core_rank=4,
            build_ema=False,
            use_heads=False
        ).to(self.device)

        self.opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3
        )

        # Tracking
        self.results = {
            'train_losses': [],
            'growth_events': [],
            'cluster_usage': defaultdict(int),
            'video_metrics': []
        }

    def load_video_frame(self, video_path, frame_idx=0):
        """Load a single frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert to grayscale and downscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.frame_size, self.frame_size))

        # Normalize to [0, 1]
        normalized = small.astype(np.float32) / 255.0

        # Flatten to vector
        return torch.from_numpy(normalized.flatten())

    def train_on_video(self, video_info, max_frames=100, show_progress=False):
        """Train on a single video (RGB)"""
        video_path = os.path.join(self.video_dir, video_info['filename'])

        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found!")
            return None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_use = min(total_frames, max_frames)

        losses = []
        clusters_used = []

        for frame_idx in range(0, frames_to_use, 3):  # Every 3rd frame for speed
            # Load frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # --- RGB, downscale, normalize ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (self.frame_size, self.frame_size))
            normalized = small.astype(np.float32) / 255.0
            x = torch.from_numpy(normalized.flatten()).to(self.device)

            # Forward pass
            x_hat, k, z, h = self.model(x)
            loss = F.mse_loss(x_hat, x)

            # Backward pass
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Update router/buffers
            k_idx = int(k.item()) if isinstance(k, torch.Tensor) else int(k)
            with torch.no_grad():
                self.model.router.update_centroid(k_idx, z)
            self.model.update_buffers(k_idx, z)

            losses.append(loss.item())
            clusters_used.append(k_idx)
            self.results['cluster_usage'][k_idx] += 1

        cap.release()

        # Check for growth
        for k_idx in set(clusters_used):
            stats = self.model.stats[k_idx]
            if len(stats.recent_losses) >= 20:
                recent_avg = sum(stats.recent_losses) / len(stats.recent_losses)
                if recent_avg > 0.02 and stats.samples > 30:
                    old_rank = self.model.layer.U_res[k_idx].shape[1] if self.model.layer.U_res[
                                                                             k_idx].numel() > 0 else 0
                    self.model.grow_cluster(k_idx, grow_rank=1)
                    new_rank = self.model.layer.U_res[k_idx].shape[1]

                    event = {
                        'video': video_info['filename'],
                        'cluster': k_idx,
                        'old_rank': old_rank,
                        'new_rank': new_rank
                    }
                    self.results['growth_events'].append(event)

                    if show_progress:
                        print(f"    🌱 Growth: Cluster {k_idx} expanded to rank {new_rank}")

        avg_loss = sum(losses) / len(losses) if losses else 1.0

        return {
            'filename': video_info['filename'],
            'avg_loss': avg_loss,
            'initial_loss': losses[0] if losses else 1.0,
            'final_loss': losses[-1] if losses else 1.0,
            'clusters': list(set(clusters_used)),
            'primary_cluster': max(set(clusters_used), key=clusters_used.count) if clusters_used else -1
        }

    def test_on_video(self, video_info, max_frames=100):
        """Test on a single video (RGB, no training)"""
        video_path = os.path.join(self.video_dir, video_info['filename'])

        if not os.path.exists(video_path):
            return None

        self.model.eval()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_use = min(total_frames, max_frames)

        losses = []
        clusters_used = []

        with torch.no_grad():
            for frame_idx in range(0, frames_to_use, 3):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # --- RGB, downscale, normalize ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = cv2.resize(rgb, (self.frame_size, self.frame_size))
                normalized = small.astype(np.float32) / 255.0
                x = torch.from_numpy(normalized.flatten()).to(self.device)

                x_hat, k, z, h = self.model(x)
                loss = F.mse_loss(x_hat, x)

                losses.append(loss.item())
                k_idx = int(k.item()) if isinstance(k, torch.Tensor) else int(k)
                clusters_used.append(k_idx)

        cap.release()
        self.model.train()

        avg_loss = sum(losses) / len(losses) if losses else 1.0

        return {
            'filename': video_info['filename'],
            'avg_loss': avg_loss,
            'clusters': list(set(clusters_used)),
            'primary_cluster': max(set(clusters_used), key=clusters_used.count) if clusters_used else -1
        }

    def run_training(self):
        """Train on all training videos"""
        print("\n" + "=" * 70)
        print("TRAINING PHASE")
        print("=" * 70)

        train_videos = [v for v in self.metadata['videos'] if v['type'] == 'train']

        for i, video_info in enumerate(train_videos):
            print(f"\n[{i + 1}/{len(train_videos)}] Training on: {video_info['filename']}")
            print(
                f"  Properties: {video_info['color']} {video_info['shape']} {video_info['motion']} {video_info['size']}")

            result = self.train_on_video(video_info, max_frames=100, show_progress=True)

            if result:
                self.results['video_metrics'].append(result)
                print(f"  Loss: {result['initial_loss']:.4f} → {result['final_loss']:.4f}")
                print(f"  Primary cluster: {result['primary_cluster']}")

    def run_testing(self):
        """Test on novel combinations"""
        print("\n" + "=" * 70)
        print("TESTING PHASE (Novel Combinations)")
        print("=" * 70)

        test_videos = [v for v in self.metadata['videos'] if v['type'] == 'test']
        test_results = []

        for i, video_info in enumerate(test_videos):
            print(f"\n[{i + 1}/{len(test_videos)}] Testing: {video_info['filename']}")
            print(
                f"  Novel combo: {video_info['color']} {video_info['shape']} {video_info['motion']} {video_info['size']}")

            result = self.test_on_video(video_info, max_frames=100)

            if result:
                test_results.append(result)
                print(f"  Test Loss: {result['avg_loss']:.4f}")
                print(f"  Primary cluster: {result['primary_cluster']}")

        return test_results

    def analyze_results(self, test_results):
        """Analyze and report results"""
        print("\n" + "=" * 70)
        print("RESULTS ANALYSIS")
        print("=" * 70)

        # Training performance
        train_metrics = [m for m in self.results['video_metrics']]
        if train_metrics:
            avg_train_loss = sum(m['final_loss'] for m in train_metrics) / len(train_metrics)
            print(f"\n📊 Training Performance:")
            print(f"  Average final loss: {avg_train_loss:.4f}")
            print(f"  Videos trained: {len(train_metrics)}")

        # Growth events
        print(f"\n🌱 Growth Events: {len(self.results['growth_events'])}")
        if self.results['growth_events']:
            print("  Notable expansions:")
            for event in self.results['growth_events'][:5]:
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
        active_clusters = len([c for c, count in self.results['cluster_usage'].items() if count > 10])
        if active_clusters >= 3:
            print(f"✓ CHECK 3: Cluster diversity ({active_clusters} active clusters)")
            checks_passed += 1
        else:
            print(f"✗ CHECK 3: Poor cluster diversity ({active_clusters} clusters)")

        # Check 4: Generalization
        if test_results and avg_test_loss < 0.2:
            print("✓ CHECK 4: Generalization works (handles novel combos)")
            checks_passed += 1
        else:
            print("✗ CHECK 4: Poor generalization to novel combinations")

        print(f"\n🎯 SCORE: {checks_passed}/{total_checks} checks passed")

        if checks_passed >= 3:
            print("\n🎉 SUCCESS! Your ASI model is learning, growing, and abstracting!")
        elif checks_passed == 2:
            print("\n⚠️  PARTIAL SUCCESS: Some capabilities working, needs tuning")
        else:
            print("\n❌ NEEDS WORK: Core capabilities not yet functional")


def main():
    print("=" * 70)
    print("ASI VIDEO LEARNING TEST")
    print("Testing: Learning, Growth, Abstraction, Generalization")
    print("=" * 70)

    # Check if videos exist
    if not os.path.exists("test_videos/dataset_metadata.json"):
        print("\n❌ ERROR: Videos not found!")
        print("Run generate_test_videos.py first to create the dataset.")
        return

    # Initialize tester
    tester = VideoLearningTest(video_dir="test_videos", device="cpu")

    # Run training
    tester.run_training()

    # Run testing
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
  Average final loss: 0.0377
  Videos trained: 80

🌱 Growth Events: 0

🎯 Cluster Usage:
  Cluster 8: 1 activations
  Cluster 9: 2719 activations

🧪 Test Performance (Novel Combinations):
  Average loss: 0.0598
  Videos tested: 20
  Train-Test gap: +0.0221
  ✓ EXCELLENT: Strong generalization to novel combinations!

======================================================================
FINAL VERDICT
======================================================================
✓ CHECK 1: Learning works (low training loss)
✗ CHECK 2: Growth not triggering enough
✗ CHECK 3: Poor cluster diversity (1 clusters)
✓ CHECK 4: Generalization works (handles novel combos)

🎯 SCORE: 2/4 checks passed

⚠️  PARTIAL SUCCESS: Some capabilities working, needs tuning

======================================================================
TEST COMPLETE!
======================================================================
'''