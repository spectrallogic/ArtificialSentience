# asi_testing/webcam_learning_viz_hq.py
# Live, higher-resolution (96x96 RGB) always-learning webcam demo.
# Left  : webcam feed
# Right : 2x2 "mind mirror" -> Reconstruction, Next-frame Prediction, Recon-Error, Pred-Error
#
# Notes:
# - Uses exploratory routing (temperature + epsilon) to keep clusters diverse.
# - Trains on every frame with reconstruction + next-frame prediction loss.
# - Growth trigger is light; toggle with 'g'.
# - If it stutters on CPU, reduce HQ_SIZE to 64 and/or MODEL_DIM to 96.

import os
import sys
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import ASISeed from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baby_asi import ASISeed  # needs use_heads=True for predict_head

# -----------------------------
# Config (bump these if you want even higher res)
# -----------------------------
HQ_SIZE    = 96                       # learning resolution (96x96 RGB). Try 128 if your GPU is strong.
INPUT_DIM  = HQ_SIZE * HQ_SIZE * 3
MODEL_DIM  = 128
NUM_CLUSTERS = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exploration schedule
ANNEAL_STEPS = 80_000                 # steps to anneal tau/eps
TAU_MAX, TAU_MIN = 2.0, 0.6           # temperature for softmax over centroid sims
EPS_MAX, EPS_MIN = 0.20, 0.02         # epsilon for random routing

# Loss weights
PRED_W = 0.35                         # next-frame prediction loss weight
ENT_W_EARLY = 1e-3                    # entropy bonus early
ENT_EARLY_FRAC = 0.4                  # first 40% of anneal

# Growth (optional)
GROW_USAGE_MIN = 350
GROW_P95_GAP   = 0.02

# Display params
WIN_W = 1200
FONT  = cv2.FONT_HERSHEY_SIMPLEX

# NEW: HUD sizing
LABEL_FONT_SCALE = 0.45
LABEL_THICKNESS  = 1
HUD_FONT_SCALE   = 0.48
HUD_THICKNESS    = 1


# -----------------------------
# Utils
# -----------------------------
def to_tensor_rgb(frame_bgr, device, size=HQ_SIZE, per_stream_stats=None):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if per_stream_stats is not None:
        mean, std = per_stream_stats
        std = std if std > 1e-6 else 1.0
        small = (small - mean) / std
        small = np.clip(small, -3.0, 3.0)
        small = (small + 3.0) / 6.0
    return torch.from_numpy(small.reshape(-1)).to(device)

def vec_to_img(x_vec, size=HQ_SIZE):
    x = x_vec.detach().float().cpu().numpy().reshape(size, size, 3)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

@torch.no_grad()
def exploratory_route(centroids, z, tau=1.0, eps=0.1):
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

def colored_heatmap(diff_img_gray):
    # diff_img_gray expected in [0..1], np.float32, shape [H,W]
    d = np.clip(diff_img_gray * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)

# -----------------------------
# Main
# -----------------------------
def main():
    # Model with prediction head
    model = ASISeed(input_dim=INPUT_DIM, model_dim=MODEL_DIM,
                    num_clusters=NUM_CLUSTERS, core_rank=4,
                    build_ema=False, use_heads=True).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam (device 0).")
        return

    # Per-stream normalization estimate
    pre = []
    for _ in range(24):
        ok, f = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (HQ_SIZE, HQ_SIZE)).astype(np.float32) / 255.0
        pre.append(small)
    stream_stats = None
    if pre:
        arr = np.stack(pre, axis=0)
        stream_stats = (float(arr.mean()), float(arr.std() + 1e-8))

    # Buffers & stats
    last_pred_x = None
    last_h = None
    replay_z = deque(maxlen=2048)
    cluster_hits = defaultdict(int)
    recent_losses = deque(maxlen=600)
    total_steps = 0
    enable_growth = True
    prefer_prediction_view = True

    fps_ema = None
    t_prev = time.time()

    print("▶ Live HQ: q=quit, p=toggle prediction priority, g=toggle growth, s=snapshot")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Prepare x_t
        x_t = to_tensor_rgb(frame, DEVICE, size=HQ_SIZE, per_stream_stats=stream_stats)

        # Encode, route, decode
        z_t = model.encoder(x_t)
        tau, eps, frac = anneal_tau_eps(total_steps)
        k_t, probs = exploratory_route(model.router.centroids, z_t, tau=tau, eps=eps)
        h_t = model.layer(z_t, active_cluster=k_t)
        x_hat_t = model.decoder(h_t)

        # Next-frame prediction loss (teach prev pred to match current truth)
        pred_loss = torch.tensor(0.0, device=DEVICE)
        if last_h is not None and model.use_heads:
            x_pred_from_prev = model.predict_head(last_h)
            pred_loss = F.mse_loss(x_pred_from_prev, x_t)

        recon_loss = F.mse_loss(x_hat_t, x_t)

        # Entropy encouragement early
        with torch.no_grad():
            sims = F.cosine_similarity(z_t.unsqueeze(0), model.router.centroids, dim=1) / max(tau, 1e-6)
            p = torch.softmax(sims, dim=0)
        entropy = -(p * (p.clamp_min(1e-8)).log()).sum()
        ent_w = ENT_W_EARLY if frac < ENT_EARLY_FRAC else 0.0

        loss = recon_loss + PRED_W * pred_loss + ent_w * entropy

        # Train
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Online updates
        with torch.no_grad():
            model.router.update_centroid(k_t, z_t)
        model.update_buffers(k_t, z_t)

        # Prepare next prediction
        if model.use_heads:
            last_pred_x = model.predict_head(h_t).detach()
        last_h = h_t.detach()

        # Growth (optional, lightweight)
        cluster_hits[k_t] += 1
        recent_losses.append(float(recon_loss.detach().cpu()))
        if enable_growth and cluster_hits[k_t] >= GROW_USAGE_MIN and len(recent_losses) >= 180:
            arr = np.array(recent_losses, dtype=np.float32)
            p95, p50 = np.percentile(arr, 95), np.percentile(arr, 50)
            if (p95 - p50) > GROW_P95_GAP:
                old_rank = model.layer.U_res[k_t].shape[1] if model.layer.U_res[k_t].numel() > 0 else 0
                model.grow_cluster(k_t, grow_rank=1)
                new_rank = model.layer.U_res[k_t].shape[1]
                print(f"🌱 Growth: Cluster {k_t} rank {old_rank} → {new_rank}")

        # Display panels
        # Left: webcam resized to match window height
        disp_h = 700
        left = cv2.resize(frame, (WIN_W // 2, disp_h), interpolation=cv2.INTER_AREA)

        # Build right 2x2 grid
        recon_img   = vec_to_img(x_hat_t, size=HQ_SIZE)
        pred_img    = vec_to_img(last_pred_x if last_pred_x is not None else x_hat_t, size=HQ_SIZE)

        # Error maps (absolute difference in [0..1], average across channels)
        x_np    = vec_to_img(x_t, size=HQ_SIZE)        # NOTE: vec_to_img clamps [0..1], color conversions match
        x_f     = cv2.cvtColor(x_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        recon_f = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pred_f  = cv2.cvtColor(pred_img,  cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        recon_err = np.mean(np.abs(recon_f - x_f), axis=2).astype(np.float32)   # [H,W]
        pred_err  = np.mean(np.abs(pred_f  - x_f), axis=2).astype(np.float32)   # [H,W]

        recon_hm = colored_heatmap(recon_err / max(1e-6, recon_err.max()))
        pred_hm  = colored_heatmap(pred_err  / max(1e-6, pred_err.max()))

        # Labels
        def stamp(img, text, y):
            cv2.putText(img, text, (8, y), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        top_left     = recon_img.copy()
        top_right    = pred_img.copy()
        bottom_left  = recon_hm
        bottom_right = pred_hm

        stamp(top_left,    "reconstruction", 24)
        stamp(top_right,   "next-frame prediction", 24)
        stamp(bottom_left, "recon error (hot=bad)", 24)
        stamp(bottom_right,"pred error (hot=bad)", 24)

        # Assemble 2x2 grid → then resize to fit half window
        tile_hq = np.vstack([
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right])
        ])

        right = cv2.resize(tile_hq, (WIN_W // 2, disp_h), interpolation=cv2.INTER_NEAREST)

        # HUD overlay on right
        hud = right.copy()
        cv2.putText(hud, f"k={k_t} loss={loss.item():.4f}  recon={recon_loss.item():.4f}  pred={pred_loss.item():.4f}",
                    (12, 48), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"tau={tau:.2f} eps={eps:.2f}  clusters={NUM_CLUSTERS}",
                    (12, 78), FONT, 0.6, (200, 255, 200), 2, cv2.LINE_AA)

        # FPS
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        fps = 1.0 / dt
        fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)
        t_prev = t_now
        cv2.putText(hud, f"FPS~{fps_ema:.1f}", (12, 108), FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Side-by-side panels
        panel = np.concatenate([left, hud], axis=1)
        title = "ArtificialSentience | Live HQ Baby Mind (q quit, p pred priority, g growth, s snapshot)"
        cv2.imshow(title, panel)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            prefer_prediction_view = not prefer_prediction_view  # reserved (currently both are shown)
        elif key == ord('g'):
            enable_growth = not enable_growth
            print(f"Growth toggled: {enable_growth}")
        elif key == ord('s'):
            snap = int(time.time())
            cv2.imwrite(f"baby_hq_snap_{snap}.png", panel)
            print(f"Saved baby_hq_snap_{snap}.png")

        total_steps += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
