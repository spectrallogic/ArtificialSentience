# simple_forgetting_test.py
# Dead simple test: Does the model forget Task A after learning Task B?

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baby_asi import ASISeed, CuriosityStream, to_device


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
    return sum(losses) / len(losses)


def train_on_task(model, stream, steps=200, lr=1e-3):
    """Train model on one task."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for step in range(steps):
        x = to_device(stream.next()).to(next(model.parameters()).device)
        x_hat, k, z, h = model(x)
        loss = F.mse_loss(x_hat, x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{steps}, Loss: {loss.item():.4f}")


def main():
    print("=" * 60)
    print("SIMPLE FORGETTING TEST")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASISeed(input_dim=32, model_dim=64, num_clusters=3,
                    core_rank=2, build_ema=False, use_heads=False).to(device)

    # Create two distinct tasks
    task_A_stream = create_task_stream(task_id=0)
    task_B_stream = create_task_stream(task_id=1)

    print("\n[1] Training on TASK A...")
    train_on_task(model, task_A_stream, steps=200)

    print("\n[2] Evaluating on TASK A (after training)...")
    perf_A_after_A = evaluate_on_task(model, task_A_stream)
    print(f"    Task A Loss: {perf_A_after_A:.4f}")

    print("\n[3] Training on TASK B...")
    train_on_task(model, task_B_stream, steps=200)

    print("\n[4] Evaluating on TASK B (after training)...")
    perf_B_after_B = evaluate_on_task(model, task_B_stream)
    print(f"    Task B Loss: {perf_B_after_B:.4f}")

    print("\n[5] RE-EVALUATING on TASK A (to check forgetting)...")
    perf_A_after_B = evaluate_on_task(model, task_A_stream)
    print(f"    Task A Loss: {perf_A_after_B:.4f}")

    # Calculate forgetting
    forgetting = perf_A_after_B - perf_A_after_A
    forgetting_percent = (forgetting / perf_A_after_A) * 100

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Task A performance BEFORE Task B: {perf_A_after_A:.4f}")
    print(f"Task A performance AFTER Task B:  {perf_A_after_B:.4f}")
    print(f"Forgetting (higher = worse):      {forgetting:.4f} ({forgetting_percent:+.1f}%)")

    if forgetting < 0:
        print("✓ GOOD: No forgetting! (Task A actually improved)")
    elif forgetting < 0.01:
        print("✓ GOOD: Minimal forgetting (< 0.01)")
    elif forgetting < 0.05:
        print("⚠ OK: Some forgetting, but manageable")
    else:
        print("✗ BAD: Significant catastrophic forgetting!")

    print("=" * 60)


if __name__ == "__main__":
    main()

'''
============================================================
SIMPLE FORGETTING TEST
============================================================

[1] Training on TASK A...
  Step 50/200, Loss: 0.0005
  Step 100/200, Loss: 0.0001
  Step 150/200, Loss: 0.0001
  Step 200/200, Loss: 0.0001

[2] Evaluating on TASK A (after training)...
    Task A Loss: 0.0006

[3] Training on TASK B...
  Step 50/200, Loss: 0.0012
  Step 100/200, Loss: 0.0000
  Step 150/200, Loss: 0.0001
  Step 200/200, Loss: 0.0001

[4] Evaluating on TASK B (after training)...
    Task B Loss: 0.0002

[5] RE-EVALUATING on TASK A (to check forgetting)...
    Task A Loss: 0.0377

============================================================
RESULTS:
============================================================
Task A performance BEFORE Task B: 0.0006
Task A performance AFTER Task B:  0.0377
Forgetting (higher = worse):      0.0371 (+6124.9%)
⚠ OK: Some forgetting, but manageable
============================================================
'''