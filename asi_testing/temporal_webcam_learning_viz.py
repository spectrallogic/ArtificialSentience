# asi_testing/temporal_webcam_learning_viz.py
# Webcam temporal learning with subconscious bias + EMA teacher (+1 step distillation)
# - Student learns every frame via teacher distillation on +1
# - Long-horizon (+30) rendered closed-loop with episodic anchoring (viz only)
# - No in-place ops in the autograd graph (fixes the earlier runtime error)

import os, sys, time, math
from collections import defaultdict, deque
import cv2, numpy as np, torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asi_model import ASISeed, ema_update_

# -------------------- Config --------------------
HQ_SIZE          = 96
INPUT_DIM        = HQ_SIZE * HQ_SIZE * 3
MODEL_DIM        = 128
NUM_CLUSTERS     = 16
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Optimizer (decoder a bit slower to reduce saturation)
LR_MAIN          = 1e-3
LR_DECODER       = 5e-4
MAX_GRAD_NORM    = 1.0

# Exploration / routing
ANNEAL_STEPS     = 80_000
TAU_MAX, TAU_MIN = 2.0, 0.6
EPS_MAX, EPS_MIN = 0.20, 0.02

# Distillation horizon (teacher)
TEACHER_POOL     = 2      # pooled MSE on pixels for teacher +1
P1_PIX_T_W       = 0.35   # student x̂_{t+1} vs teacher x̂_{t+1}
P1_LAT_T_W       = 0.30   # student h_{t+1} vs teacher h_{t+1}
SUBC_ALIGN_W     = 0.25   # align student subconscious bias to teacher Δh

# Base losses
RECON_W          = 1.0
ENT_W_EARLY      = 1e-3
ENT_EARLY_FRAC   = 0.4
VAR_W            = 0.15   # variance preserver on student +1 pixels

# Long-horizon visualization (no training grads from these)
FAR_H            = 30
KNN_STEPS        = 8
KNN_BLEND        = 0.12
KNN_MIN_SIMS     = 0.25
REANCHOR_EVERY   = 16
REANCHOR_GAMMA   = 0.04

# UI
WIN_W               = 1200
FONT                = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE    = 0.38
LABEL_THICKNESS     = 1
HUD_FONT_SCALE      = 0.40
HUD_THICKNESS       = 1

# -------------------- Utils --------------------
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

def colored_heatmap(diff_img_gray):
    p95 = float(np.percentile(diff_img_gray, 95)) if diff_img_gray.size > 0 else 1.0
    denom = max(1e-6, p95)
    norm = np.clip(diff_img_gray / denom, 0.0, 1.0)
    d = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)

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

def pooled_mse(x_pred, x_true, size=HQ_SIZE, pool=2):
    p = x_pred.view(1, size, size, 3).permute(0,3,1,2)
    t = x_true.view(1, size, size, 3).permute(0,3,1,2)
    if pool > 1:
        p = F.avg_pool2d(p, kernel_size=pool, stride=pool)
        t = F.avg_pool2d(t, kernel_size=pool, stride=pool)
    return F.mse_loss(p, t)

def cos_loss(a, b, eps=1e-8):
    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).clamp(-1+eps, 1-eps).mean()

def variance_preserver(x_pred, min_std=0.06):
    # no in-place ops here; return a scalar
    x = x_pred.view(1, -1)
    std = x.std(dim=1, unbiased=False)
    short = torch.clamp(min_std - std, min=0.0)
    return (short * short).mean()

# -------------------- Episodic memory for viz anchoring --------------------
class EpisodicMem:
    def __init__(self, cap=320):
        self.buf = deque(maxlen=cap)  # (z,h,x,k)
    def add(self, z, h, x, k):
        self.buf.append((z.detach().cpu(), h.detach().cpu(), x.detach().cpu(), int(k)))
    def knn_h(self, h_query, k_query, min_sim=KNN_MIN_SIMS):
        if not self.buf: return None
        hq = F.normalize(h_query.detach().cpu(), dim=0)
        best_sim, best_h = -1.0, None
        for (z, h, x, kc) in reversed(self.buf):
            if kc != k_query: continue
            sim = float(F.cosine_similarity(hq.unsqueeze(0), F.normalize(h, dim=0).unsqueeze(0), dim=1))
            if sim > best_sim:
                best_sim, best_h = sim, h
        if best_h is not None and best_sim >= min_sim:
            return best_h.to(h_query.device)
        return None

# -------------------- Teacher builder --------------------
def build_ema_from(model: ASISeed) -> ASISeed:
    ema = ASISeed(input_dim=model._hparams["input_dim"],
                  model_dim=model._hparams["model_dim"],
                  num_clusters=model._hparams["num_clusters"],
                  core_rank=model._hparams["core_rank"],
                  build_ema=False,
                  use_heads=model._hparams["use_heads"]).to(next(model.parameters()).device)
    ema.load_state_dict(model.state_dict(), strict=False)
    for p in ema.parameters():
        p.requires_grad = False
    return ema

# -------------------- Closed-loop viz rollout (no grads) --------------------
@torch.no_grad()
def rollout_closed_loop(model: ASISeed, h0, z0, k_route, mem: EpisodicMem, steps=30):
    h = h0.detach().clone()
    frames = []
    for i in range(1, steps + 1):
        # subconscious bias each step
        s_bias, _ = model.subconscious_bias(z0, h, k_route)
        h = model.temporal.step(z0, h, bias=s_bias).detach()
        if (i % KNN_STEPS) == 0:
            h_anchor = mem.knn_h(h, k_route)
            if h_anchor is not None:
                h = (1.0 - KNN_BLEND) * h + KNN_BLEND * h_anchor
        if (i % REANCHOR_EVERY) == 0:
            h = (1.0 - REANCHOR_GAMMA) * h + REANCHOR_GAMMA * h0
        x = model.decoder(h).detach()
        frames.append(x)
    return frames

# -------------------- Main --------------------
def main():
    model = ASISeed(input_dim=INPUT_DIM, model_dim=MODEL_DIM,
                    num_clusters=NUM_CLUSTERS, core_rank=4,
                    build_ema=False, use_heads=True).to(DEVICE)

    # Two-group optimizer (use id() to avoid tensor-equality pitfalls)
    dec_params = list(model.decoder.parameters())
    dec_ids    = {id(p) for p in dec_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in dec_ids]
    opt = torch.optim.Adam([
        {"params": other_params, "lr": LR_MAIN},
        {"params": dec_params,   "lr": LR_DECODER}
    ])

    # EMA teacher for instant +1 supervision
    teacher = build_ema_from(model)
    EMA_BETA = 0.998

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam (device 0).")
        return

    # Stream normalization warmup
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

    mem = EpisodicMem(cap=320)

    total_steps = 0
    fps_ema, t_prev = None, time.time()

    far_frames, far_targets = [], []
    far_idx = 0

    print("▶ Webcam Subconscious Demo (teacher +1 distillation) — [q quit, s snapshot]")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # -------- Perception & routing --------
        x_t = to_tensor_rgb(frame, DEVICE, size=HQ_SIZE, per_stream_stats=stream_stats)
        z_t = model.encoder(x_t)
        tau, eps, frac = anneal_tau_eps(total_steps)
        k_t, _ = exploratory_route(model.router.centroids, z_t, tau=tau, eps=eps)

        # Low-rank slice and reconstruction
        h_t = model.layer(z_t, active_cluster=k_t)
        x_hat_t = model.decoder(h_t)

        # Subconscious bias (student) & +1 prediction (student)
        s_now, _info = model.subconscious_bias(z_t, h_t, k_t)
        h_pred1_S = model.temporal.step(z_t, h_t, bias=s_now)           # keep graph
        x_pred1_S = model.decoder(h_pred1_S)                             # keep graph

        # Teacher +1 (no grad) for immediate supervision
        with torch.no_grad():
            s_T, _ = teacher.subconscious_bias(z_t.detach(), h_t.detach(), k_t)
            h_pred1_T = teacher.temporal.step(z_t.detach(), h_t.detach(), bias=s_T)
            x_pred1_T = teacher.decoder(h_pred1_T)

        # -------- Losses (ALL are out-of-place ops; no in-place on graph tensors) --------
        recon_loss   = F.mse_loss(x_hat_t, x_t)
        p1_pix_T_loss= pooled_mse(x_pred1_S, x_pred1_T, size=HQ_SIZE, pool=TEACHER_POOL)
        p1_lat_T_loss= cos_loss(h_pred1_S, h_pred1_T)

        # Align subconscious direction to teacher delta-h
        delta_T = (h_pred1_T - h_t).detach()
        subc_align = cos_loss(s_now, delta_T)

        # Encourage non-flat predictions
        var_loss = variance_preserver(x_pred1_S)

        # Router entropy (early)
        with torch.no_grad():
            sims = F.cosine_similarity(z_t.unsqueeze(0), model.router.centroids, dim=1) / max(1e-6, tau)
            p = torch.softmax(sims, dim=0)
        entropy = -(p * (p.clamp_min(1e-8)).log()).sum()
        ent_w = ENT_W_EARLY if frac < ENT_EARLY_FRAC else 0.0

        loss = (
            RECON_W      * recon_loss +
            P1_PIX_T_W   * p1_pix_T_loss +
            P1_LAT_T_W   * p1_lat_T_loss +
            SUBC_ALIGN_W * subc_align +
            VAR_W        * var_loss +
            ent_w        * entropy
        )

        # -------- Train --------
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
        opt.step()

        # Online updates
        with torch.no_grad():
            model.router.update_centroid(k_t, z_t)
            ema_update_(teacher, model, beta=EMA_BETA)

        # Episodic memory update (for viz anchoring only)
        mem.add(z_t, h_t, x_t, k_t)

        # -------- Long-horizon visualization (no grad) --------
        need_new_clip = (len(far_frames) == 0) or (far_idx >= len(far_frames))
        if need_new_clip:
            far_frames.clear(); far_targets.clear(); far_idx = 0
            with torch.no_grad():
                xs = rollout_closed_loop(model, h0=h_t, z0=z_t, k_route=k_t, mem=mem, steps=FAR_H)
            for i in range(FAR_H):
                far_frames.append(vec_to_img(xs[i], size=HQ_SIZE))
                far_targets.append(total_steps + (i + 1))
        far_img = far_frames[far_idx]
        far_target_step = far_targets[far_idx]
        far_idx += 1

        # Error map (switch to +30 once GT arrives)
        if far_target_step == total_steps:
            x_np  = vec_to_img(x_t, size=HQ_SIZE)
            x_f   = cv2.cvtColor(x_np,    cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            far_f = cv2.cvtColor(far_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            err   = np.mean(np.abs(far_f - x_f), axis=2).astype(np.float32)
            err_hm = colored_heatmap(err / max(1e-6, err.max()))
            err_label = f"pred +{FAR_H}f error (hot=bad)"
        else:
            x_np    = vec_to_img(x_t, size=HQ_SIZE)
            recon_np= vec_to_img(x_hat_t.detach(), size=HQ_SIZE)
            x_f     = cv2.cvtColor(x_np,        cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            recon_f = cv2.cvtColor(recon_np,    cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            err     = np.mean(np.abs(recon_f - x_f), axis=2).astype(np.float32)
            err_hm  = colored_heatmap(err / max(1e-6, err.max()))
            err_label = "recon error (waiting for +30f GT)"

        # -------- Display --------
        disp_h = 700
        left = cv2.resize(frame, (WIN_W // 2, disp_h), interpolation=cv2.INTER_AREA)

        recon_img   = vec_to_img(x_hat_t.detach(), size=HQ_SIZE)
        pred1_img   = vec_to_img(x_pred1_S.detach(), size=HQ_SIZE)
        pred_far_img= far_img

        def stamp(img, text, y):
            cv2.putText(img, text, (8, y), FONT, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS, cv2.LINE_AA)

        tl = recon_img.copy();    stamp(tl, "reconstruction", 22)
        tr = pred1_img.copy();    stamp(tr, "+1 prediction (student)", 22)
        bl = pred_far_img.copy(); stamp(bl, f"+{FAR_H} future (closed-loop+episodic)", 22)
        br = err_hm;              stamp(br, err_label, 22)

        tile  = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
        right = cv2.resize(tile, (WIN_W // 2, disp_h), interpolation=cv2.INTER_NEAREST)

        # HUD
        hud = right.copy()
        hud_y = 44
        cv2.putText(
            hud,
            f"k={k_t}  loss={loss.item():.4f}  recon={recon_loss.item():.4f}  "
            f"p1_pixT={p1_pix_T_loss.item():.4f}  p1_latT={p1_lat_T_loss.item():.4f}",
            (12, hud_y), FONT, HUD_FONT_SCALE, (0, 255, 255), HUD_THICKNESS, cv2.LINE_AA
        )
        cv2.putText(
            hud,
            f"subcAlign={subc_align.item():.4f}  var={var_loss.item():.4f}",
            (12, hud_y + 24), FONT, HUD_FONT_SCALE, (200, 255, 200), HUD_THICKNESS, cv2.LINE_AA
        )
        # FPS
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        fps = 1.0 / dt
        fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)
        t_prev = t_now
        cv2.putText(hud, f"FPS~{fps_ema:.1f}", (12, hud_y + 48), FONT, HUD_FONT_SCALE, (0, 200, 255), HUD_THICKNESS, cv2.LINE_AA)

        panel = np.concatenate([left, hud], axis=1)
        title = "ArtificialSentience | Webcam Subconscious Demo (teacher +1 distill)  [q quit, s snap]"
        cv2.imshow(title, panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            snap = int(time.time())
            cv2.imwrite(f"baby_temporal_snap_{snap}.png", panel)
            print(f"Saved baby_temporal_snap_{snap}.png")

        total_steps += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
