# realtime_text_learner.py
# Watch the AI learn from your text in real-time!
# See its "thoughts" (reconstructions) and watch it grow

import torch
import torch.nn.functional as F
from asi_model import ASISeed, to_device
import time


class SimpleTextEncoder:
    """Convert text to simple vectors for learning"""

    def __init__(self, vocab_size=256, embed_dim=32):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Simple embedding: one-hot → learned embedding
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        # Initialize properly
        torch.nn.init.normal_(self.embed.weight, mean=0, std=0.1)

    def text_to_vector(self, text):
        """Convert text snippet to a single vector (average of char embeddings)"""
        if not text or not text.strip():
            return torch.zeros(self.embed_dim)

        # Convert to byte indices (simple, works with any text)
        chars = text[:50]  # Max 50 chars
        indices = []
        for c in chars:
            idx = ord(c) % self.vocab_size  # Use modulo to ensure in range
            indices.append(idx)

        if not indices:
            return torch.zeros(self.embed_dim)

        try:
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            embeddings = self.embed(indices_tensor)
            # Average pooling
            return embeddings.mean(dim=0).detach()
        except Exception as e:
            print(f"Encoding error: {e}")
            return torch.zeros(self.embed_dim)


class RealtimeTextLearner:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # Text encoder
        self.encoder = SimpleTextEncoder(vocab_size=256, embed_dim=32)

        # ASI Model (tiny for fast learning)
        self.model = ASISeed(
            input_dim=32,
            model_dim=48,
            num_clusters=5,  # More clusters for different topics
            core_rank=2,
            build_ema=False,
            use_heads=False
        ).to(self.device)

        # Optimizer
        self.opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.encoder.embed.parameters()),
            lr=3e-3
        )

        # Stats tracking
        self.total_steps = 0
        self.cluster_stats = {i: {"samples": 0, "recent_loss": []} for i in range(5)}
        self.growth_events = []

    def learn_from_text(self, text, num_steps=10):
        """Learn from a text snippet"""
        if not text.strip():
            return None

        print(f"\n{'=' * 70}")
        print(f"📝 YOU TYPED: '{text}'")
        print(f"{'=' * 70}")

        # Convert text to vector (no batch dimension needed)
        x = self.encoder.text_to_vector(text).to(self.device)

        # Training loop
        losses = []
        for step in range(num_steps):
            # Forward pass (ASISeed expects unbatched input)
            x_hat, k, z, h = self.model(x)

            # Loss
            loss = F.mse_loss(x_hat, x)
            losses.append(loss.item())

            # Backward
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Update router (z is already the right shape)
            with torch.no_grad():
                self.model.router.update_centroid(k, z)
            self.model.update_buffers(k, z)

            # Track stats
            self.cluster_stats[k]["samples"] += 1
            self.cluster_stats[k]["recent_loss"].append(loss.item())
            if len(self.cluster_stats[k]["recent_loss"]) > 20:
                self.cluster_stats[k]["recent_loss"].pop(0)

        self.total_steps += num_steps

        # Show results
        final_loss = losses[-1]
        improvement = losses[0] - losses[-1]

        print(f"\n🧠 AI'S INTERNAL STATE:")
        print(f"   Active Cluster: {k}")
        print(f"   Initial Loss: {losses[0]:.4f}")
        print(f"   Final Loss: {final_loss:.4f}")
        print(f"   Improvement: {improvement:.4f} ({'learned!' if improvement > 0.001 else 'already knew this'})")

        # Show reconstruction quality (proxy for "what AI thinks")
        reconstruction_quality = 1 - min(final_loss, 1.0)
        bars = int(reconstruction_quality * 20)
        print(f"\n🎯 UNDERSTANDING: [{'█' * bars}{'░' * (20 - bars)}] {reconstruction_quality * 100:.1f}%")

        # Show cluster usage
        print(f"\n📊 CLUSTER ACTIVITY:")
        for i in range(self.model.num_clusters):
            stats = self.cluster_stats[i]
            if stats["samples"] > 0:
                avg_loss = sum(stats["recent_loss"]) / len(stats["recent_loss"]) if stats["recent_loss"] else 0
                active = "←" if i == k else ""
                print(f"   Cluster {i}: {stats['samples']:4d} samples, loss={avg_loss:.4f} {active}")

        # Check if growth should happen (simplified)
        if len(self.cluster_stats[k]["recent_loss"]) >= 20:
            recent_avg = sum(self.cluster_stats[k]["recent_loss"]) / len(self.cluster_stats[k]["recent_loss"])
            if recent_avg > 0.01 and self.cluster_stats[k]["samples"] > 50:  # Struggling
                old_rank = self.model.layer.U_res[k].shape[1] if self.model.layer.U_res[k].numel() > 0 else 0
                self.model.grow_cluster(k, grow_rank=1)
                new_rank = self.model.layer.U_res[k].shape[1]
                event = f"Step {self.total_steps}: Cluster {k} grew from rank {old_rank} → {new_rank}"
                self.growth_events.append(event)
                print(f"\n🌱 GROWTH EVENT: {event}")
                self.model.consolidate_cluster(k)
                print(f"   Consolidated knowledge into core abstractions")

        return {
            "cluster": k,
            "loss": final_loss,
            "improvement": improvement
        }

    def show_summary(self):
        """Show overall learning summary"""
        print(f"\n{'=' * 70}")
        print(f"📈 LEARNING SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total training steps: {self.total_steps}")
        print(f"Growth events: {len(self.growth_events)}")
        if self.growth_events:
            print("\n🌱 Growth history:")
            for event in self.growth_events:
                print(f"   {event}")

        print("\n🧠 Final cluster state:")
        for i in range(self.model.num_clusters):
            stats = self.cluster_stats[i]
            rank = self.model.layer.U_res[i].shape[1] if self.model.layer.U_res[i].numel() > 0 else 0
            print(f"   Cluster {i}: {stats['samples']} samples, rank={rank}")


def main():
    print("=" * 70)
    print("🤖 REALTIME TEXT LEARNER")
    print("=" * 70)
    print("\nType anything and watch the AI learn!")
    print("Commands:")
    print("  - Type text to teach the AI")
    print("  - Type 'summary' to see learning stats")
    print("  - Type 'quit' to exit")
    print("\nTips:")
    print("  - Try different topics to see clusters emerge")
    print("  - Repeat phrases to see it memorize")
    print("  - Watch for growth events when it struggles")
    print("=" * 70)

    learner = RealtimeTextLearner(device="cpu")

    # Example texts to try
    print("\n💡 Example topics to try:")
    print("   'I love pizza and pasta'")
    print("   'My cat is very fluffy'")
    print("   'Python programming is fun'")
    print("   'The weather is sunny today'")

    while True:
        try:
            text = input("\n> ").strip()

            if not text:
                continue
            elif text.lower() == 'quit':
                break
            elif text.lower() == 'summary':
                learner.show_summary()
            else:
                learner.learn_from_text(text, num_steps=15)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    learner.show_summary()
    print("\n👋 Thanks for teaching me!")


if __name__ == "__main__":
    main()

'''
======================================================================
🤖 REALTIME TEXT LEARNER
======================================================================

Type anything and watch the AI learn!
Commands:
  - Type text to teach the AI
  - Type 'summary' to see learning stats
  - Type 'quit' to exit

Tips:
  - Try different topics to see clusters emerge
  - Repeat phrases to see it memorize
  - Watch for growth events when it struggles
======================================================================

💡 Example topics to try:
   'I love pizza and pasta'
   'My cat is very fluffy'
   'Python programming is fun'
   'The weather is sunny today'

> I love pizza

======================================================================
📝 YOU TYPED: 'I love pizza'
======================================================================

🧠 AI'S INTERNAL STATE:
   Active Cluster: 4
   Initial Loss: 0.0058
   Final Loss: 0.0014
   Improvement: 0.0044 (learned!)

🎯 UNDERSTANDING: [███████████████████░] 99.9%

📊 CLUSTER ACTIVITY:
   Cluster 4:   15 samples, loss=0.0035 ←

> My cat is fluffy

======================================================================
📝 YOU TYPED: 'My cat is fluffy'
======================================================================

🧠 AI'S INTERNAL STATE:
   Active Cluster: 4
   Initial Loss: 0.0021
   Final Loss: 0.0003
   Improvement: 0.0018 (learned!)

🎯 UNDERSTANDING: [███████████████████░] 100.0%

📊 CLUSTER ACTIVITY:
   Cluster 4:   30 samples, loss=0.0013 ←

> I love pizza

======================================================================
📝 YOU TYPED: 'I love pizza'
======================================================================

🧠 AI'S INTERNAL STATE:
   Active Cluster: 4
   Initial Loss: 0.0008
   Final Loss: 0.0003
   Improvement: 0.0005 (already knew this)

🎯 UNDERSTANDING: [███████████████████░] 100.0%

📊 CLUSTER ACTIVITY:
   Cluster 4:   45 samples, loss=0.0006 ←

> My cat is fluffy

======================================================================
📝 YOU TYPED: 'My cat is fluffy'
======================================================================

🧠 AI'S INTERNAL STATE:
   Active Cluster: 4
   Initial Loss: 0.0007
   Final Loss: 0.0002
   Improvement: 0.0004 (already knew this)

🎯 UNDERSTANDING: [███████████████████░] 100.0%

📊 CLUSTER ACTIVITY:
   Cluster 4:   60 samples, loss=0.0004 ←

> 

'''