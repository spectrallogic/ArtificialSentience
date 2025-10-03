# asi_testing/webcam_learning_viz.py
# Live, always-learning webcam demo:
# Left  = webcam
# Right = "mirrored thoughts" (last predicted next frame; fallback: reconstruction)
# Online training with exploratory routing + next-frame prediction.

import os
import sys
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Make sure we can import ASISeed from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asi_model import ASISeed  # uses optional predict_head when use_heads=True :contentReference[oaicite:1]{index=1}

# -----------------------------
# Config
# -----------------------------
FRAME_SIZE = 32                 # learning resolution (32x32 RGB)
INPUT_DIM  = FRAME_SIZE * FRAME_SIZE * 3
MODEL_DIM  = 128
NUM_CLUSTERS = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exploration schedule (keeps clusters diverse at start, then stabilizes)
ANNEAL_STEPS = 50_000           # steps to anneal tau/eps
TAU_MAX, TAU_MIN = 2.0, 0.6     # temperature for softmax over centroid sims
EPS_MAX, EPS_MIN = 0.20, 0.02   # epsilon for random routing

# Training weights
PRED_W = 0.30                   # weight for next-frame prediction loss (teacher = true next frame)
ENT_W_EARLY = 1e-3              # entropy bonus early
ENT_EARLY_FRAC = 0.4            # apply entropy bonus for first 40% of ANNEAL_STEPS

# Growth (optional lightweight trigger)
GROW_USAGE_MIN = 250            # cluster must be used at least this many times before growth check
GROW_P95_GAP = 0.02             # 95th - 50th percentile loss gap to trigger growth

# Display
WIN_W = 640                     # window width (side-by-side panels)
FONT  = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------
# Utils
# -----------------------------
def to_tensor_rgb32(frame_bgr, device, per_stream_stats=None):
    """BGR -> RGB, resize 32x32, [0..1], flatten -> torch[3072]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if per_stream_stats is not None:
        mean, std = per_stream_stats
        std = std if std > 1e-6 else 1.0
        small = (small - mean) / std
        small = np.clip(small, -3.0, 3.0)
        small = (small + 3.0) / 6.0
    return torch.from_numpy(small.reshape(-1)).to(device)

def vec_to_img(x_vec):
    """torch[3072] -> uint8 image (32x32x3), scaled [0..255]."""
    x = x_vec.detach().float().cpu().numpy().reshape(FRAME_SIZE, FRAME_SIZE, 3)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

@torch.no_grad()
def exploratory_route(centroids, z, tau=1.0, eps=0.1):
    """Softmax over cosine sims (temperature tau) + epsilon-greedy exploration."""
    sims = F.cosine_similarity(z.unsqueeze(0), centroids, dim=1) / max(1e-6, tau)
    probs = torch.softmax(sims, dim=0)
    if torch.rand(1).item() < eps:
        k = torch.randint(0, centroids.size(0), (1,)).item()
    else:
        k = torch.multinomial(probs, 1).item()
    return int(k), probs

def anneal_tau_eps(step):
    t = min(1.0, step / max(1, ANNEAL_STEPS))
    tau = TAU_MAX * (1 - t) + TAU_MIN * t
    eps = EPS_MAX * (1 - t) + EPS_MIN * t
    return float(tau), float(eps), t

# -----------------------------
# Main
# -----------------------------
def main():
    # Model: enable heads to use next-frame prediction:contentReference[oaicite:2]{index=2}
    model = ASISeed(input_dim=INPUT_DIM, model_dim=MODEL_DIM,
                    num_clusters=NUM_CLUSTERS, core_rank=4,
                    build_ema=False, use_heads=True).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam (device 0).")
        return

    # Buffers
    last_pred_x = None          # last predicted future frame (as vector)
    last_h = None               # last latent state for prediction teacher
    replay_z = deque(maxlen=2048)
    cluster_hits = defaultdict(int)
    recent_losses = deque(maxlen=500)
    recent_losses_global = deque(maxlen=4000)
    total_steps = 0
    enable_growth = True
    show_prediction_view = True

    # Per-stream normalization (estimate once from first frames)
    mean_est, std_est = None, None
    pre_samples = []
    for _ in range(24):
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE)).astype(np.float32) / 255.0
        pre_samples.append(small)
    if pre_samples:
        arr = np.stack(pre_samples, axis=0)
        mean_est = float(arr.mean())
        std_est  = float(arr.std() + 1e-8)
    stream_stats = (mean_est, std_est) if pre_samples else None

    t_prev = time.time()
    fps_ema = None

    print("▶ Live: q=quit, p=toggle prediction view, g=toggle growth, s=snapshot")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Learning vector
        x_t = to_tensor_rgb32(frame, DEVICE, per_stream_stats=stream_stats)

        # Encode current, route with exploration
        z_t = model.encoder(x_t)
        tau, eps, frac = anneal_tau_eps(total_steps)
        k_t, p = exploratory_route(model.router.centroids, z_t, tau=tau, eps=eps)
        h_t = model.layer(z_t, active_cluster=k_t)
        x_hat_t = model.decoder(h_t)

        # Teacher for prediction: we want the *previous prediction* to match current truth (x_t)
        pred_loss = torch.tensor(0.0, device=DEVICE)
        if last_h is not None and model.use_heads:
            x_pred_t_from_prev = model.predict_head(last_h)     # prediction made at t-1 for frame t:contentReference[oaicite:3]{index=3}
            pred_loss = F.mse_loss(x_pred_t_from_prev, x_t)

        # Reconstruction loss on current frame
        recon_loss = F.mse_loss(x_hat_t, x_t)

        # Gentle entropy encouragement early
        with torch.no_grad():
            sims = F.cosine_similarity(z_t.unsqueeze(0), model.router.centroids, dim=1) / max(tau, 1e-6)
            probs = torch.softmax(sims, dim=0)
        entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum()
        ent_w = ENT_W_EARLY if frac < ENT_EARLY_FRAC else 0.0

        # Total loss
        loss = recon_loss + PRED_W * pred_loss + ent_w * entropy

        # Train
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Update centroid & buffers
        with torch.no_grad():
            model.router.update_centroid(k_t, z_t)
        model.update_buffers(k_t, z_t)

        # Prepare *future* prediction (what we think x_{t+1} will be)
        next_pred = None
        if model.use_heads:
            next_pred = model.predict_head(h_t).detach()
            last_pred_x = next_pred.clone()
        last_h = h_t.detach()

        # Growth check (very light: only when heavily used + heavy-tail losses)
        cluster_hits[k_t] += 1
        recent_losses.append(float(recon_loss.detach().cpu()))
        recent_losses_global.append(float(recon_loss.detach().cpu()))
        if enable_growth and cluster_hits[k_t] >= GROW_USAGE_MIN and len(recent_losses) >= 120:
            arr = np.array(recent_losses)
            p95, p50 = np.percentile(arr, 95), np.percentile(arr, 50)
            if (p95 - p50) > GROW_P95_GAP:
                old_rank = model.layer.U_res[k_t].shape[1] if model.layer.U_res[k_t].numel() > 0 else 0
                model.grow_cluster(k_t, grow_rank=1)  # keeps EMA shapes in sync internally:contentReference[oaicite:4]{index=4}
                new_rank = model.layer.U_res[k_t].shape[1]
                print(f"🌱 Growth: Cluster {k_t} rank {old_rank} → {new_rank}")

        # Update replay and occasionally self-distill core on replay (tiny)
        replay_z.append(z_t.detach().cpu())
        if len(replay_z) > 32 and (total_steps % 9 == 0):
            zr = torch.stack([replay_z[np.random.randint(len(replay_z))] for _ in range(16)], dim=0).to(DEVICE)
            with torch.no_grad():
                target = F.relu(zr @ (model.layer.U @ model.layer.V.t()).t())
            pred = F.relu(zr @ (model.layer.U @ model.layer.V.t()).t())
            (0.05 * F.mse_loss(pred, target)).backward()
            # Note: relies on next opt.step(); kept very small to avoid hitching

        # ---------- Build the display ----------
        # Left panel: webcam (resized)
        disp_h = 480
        left = cv2.resize(frame, (WIN_W // 2, disp_h), interpolation=cv2.INTER_AREA)

        # Right panel: "mirrored thoughts"
        # Use last predicted frame (what we thought the future would be).
        if show_prediction_view and last_pred_x is not None:
            right_img_small = vec_to_img(last_pred_x)  # predicted next frame (from previous step)
            label = "mirrored thoughts: last prediction"
        else:
            right_img_small = vec_to_img(x_hat_t)      # fallback: current reconstruction
            label = "mirrored thoughts: reconstruction"

        right = cv2.resize(right_img_small, (WIN_W // 2, disp_h), interpolation=cv2.INTER_NEAREST)

        # HUD
        hud = right.copy()
        cv2.putText(hud, label, (12, 28), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"k={k_t} loss={loss.item():.4f} recon={recon_loss.item():.4f} pred={pred_loss.item():.4f}",
                    (12, 56), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"tau={tau:.2f} eps={eps:.2f}", (12, 84), FONT, 0.6, (200, 255, 200), 2, cv2.LINE_AA)

        # FPS
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        fps = 1.0 / dt
        fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)
        t_prev = t_now
        cv2.putText(hud, f"FPS~{fps_ema:.1f}", (12, 112), FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Side-by-side
        panel = np.concatenate([left, hud], axis=1)
        cv2.imshow("ArtificialSentience | Live Baby Mind (q=quit, p=pred view, g=growth, s=snap)", panel)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_prediction_view = not show_prediction_view
        elif key == ord('g'):
            enable_growth = not enable_growth
            print(f"Growth toggled: {enable_growth}")
        elif key == ord('s'):
            snap = int(time.time())
            cv2.imwrite(f"baby_snap_{snap}.png", panel)
            print(f"Saved baby_snap_{snap}.png")

        total_steps += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
