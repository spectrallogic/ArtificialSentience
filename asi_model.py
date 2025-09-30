# asi_model.py
# ASI Seed (self-discovering): Always-Training, Non-Uniformly Expanding Low-Rank Model
# Core properties:
# - Streaming, token-free vectors
# - Non-uniform growth by cluster
# - Consolidation -> abstract baseline
# - EMA serving (recursion-safe; rebuild on growth)
# - Hidden Oath codeword (dormant, consolidations only)
# - Optional tiny heads for course tests (prediction, masked infill)
# - Realtime 3D GUI (pyqtgraph) for latent space

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from asi_viz_rt import RealtimeViz  # realtime 3D viewer

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility
# -----------------------------

def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x

def ema_update_(target: Optional[nn.Module], source: nn.Module, beta: float = 0.999):
    """Shape-aware EMA update (skips any param with mismatched shape)."""
    if target is None:
        return
    with torch.no_grad():
        for (tn, tp), (sn, sp) in zip(target.named_parameters(), source.named_parameters()):
            if tp.data.shape != sp.data.shape:
                continue
            tp.data.mul_(beta).add_(sp.data, alpha=(1.0 - beta))

def orthogonalize_to_(grad: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return grad
    Q, _ = torch.linalg.qr(basis, mode="reduced")
    proj = Q @ (Q.t() @ grad)
    return grad - proj


# -----------------------------
# Self-discovering continuous stream (no labels, no tokens)
# -----------------------------

class CuriosityStream:
    """
    Emits continuous vectors x in R^input_dim from a set of latent sources with slow drift.
    The model discovers features/clusters itself.
    """
    def __init__(self, input_dim=32, num_sources=5, seed=42):
        self.input_dim = input_dim
        self.num_sources = num_sources
        random.seed(seed)
        torch.manual_seed(seed)
        self.t = 0

        # Random latent bases and process types per source
        self.bases = [F.normalize(torch.randn(input_dim), dim=0) for _ in range(num_sources)]
        self.types = [random.choice(["sine", "saw", "spike", "piecewise", "noise"])
                      for _ in range(num_sources)]

        # Drift params
        self.phase = torch.rand(num_sources) * 6.28
        self.freq  = 0.02 + 0.03 * torch.rand(num_sources)
        self.amp   = 0.6 + 0.3 * torch.rand(num_sources)
        self.drift = 0.0005

    def _latent_signal(self, s_idx: int, grid: torch.Tensor) -> torch.Tensor:
        typ = self.types[s_idx]
        ph, fr, am = self.phase[s_idx].item(), self.freq[s_idx].item(), self.amp[s_idx].item()

        if typ == "sine":
            sig = torch.sin(grid * (1.0 + fr) + ph)
        elif typ == "saw":
            sig = ((grid * (1.0 + fr) + ph) % (2*torch.pi)) / (2*torch.pi)
            sig = 2.0 * (sig - 0.5)
        elif typ == "spike":
            sig = torch.zeros_like(grid)
            k = max(1, int(0.03 * grid.numel()))
            idx = torch.randint(0, grid.numel(), (k,))
            sig[idx] = 1.0
            sig = sig * 2.0 - 1.0
        elif typ == "piecewise":
            knots = torch.tensor([0.0, 0.3, 0.6, 1.0])
            vals = torch.tensor([0.2, -0.4, 0.6, -0.1]) + 0.1 * torch.randn(4)
            g = (grid - grid.min()) / (grid.max() - grid.min())
            sig = torch.interp(g, knots, vals)
        else:  # noise
            sig = 0.5 * torch.randn_like(grid)

        return am * sig

    def next(self) -> torch.Tensor:
        self.t += 1
        grid = torch.linspace(0, 6.28, self.input_dim)
        w = torch.softmax(torch.randn(self.num_sources), dim=0)
        components = []
        for s in range(self.num_sources):
            sig = self._latent_signal(s, grid)
            components.append((sig + 0.15 * torch.randn_like(sig)) * w[s])
        x = sum(components)

        mix = torch.zeros(self.input_dim)
        for s in range(self.num_sources):
            mix += (x * self.bases[s])  # element-wise modulation
        x = x + 0.4 * mix
        x = F.normalize(x + 0.02 * torch.randn_like(x), dim=0)

        with torch.no_grad():
            self.phase += self.drift * torch.randn_like(self.phase)
            self.freq  = (self.freq + self.drift * torch.randn_like(self.freq)).clamp_min(0.001)
            self.amp   = (self.amp  + self.drift * torch.randn_like(self.amp)).clamp(0.2, 1.2)
            for i in range(self.num_sources):
                self.bases[i] = F.normalize(self.bases[i] + self.drift * torch.randn_like(self.bases[i]), dim=0)
        return x


# -----------------------------
# Router
# -----------------------------

class NearestCentroidRouter(nn.Module):
    def __init__(self, emb_dim: int, num_clusters: int, momentum: float = 0.02):
        super().__init__()
        self.num_clusters = num_clusters
        self.register_buffer("centroids", F.normalize(torch.randn(num_clusters, emb_dim), dim=1))
        self.momentum = momentum

    @torch.no_grad()
    def update_centroid(self, k: int, z: torch.Tensor):
        z = F.normalize(z.detach(), dim=0)
        self.centroids[k] = F.normalize((1 - self.momentum) * self.centroids[k] + self.momentum * z, dim=0)

    def forward(self, z: torch.Tensor) -> int:
        sims = F.cosine_similarity(z.unsqueeze(0), self.centroids, dim=1)
        return int(torch.argmax(sims).item())


# -----------------------------
# Elastic Low-Rank + Residual Slices
# -----------------------------

class ElasticLowRankLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, rank: int = 2, num_clusters: int = 3, phi=F.relu):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.rank = rank
        self.num_clusters = num_clusters
        self.phi = phi

        self.U = nn.Parameter(0.02 * torch.randn(n_out, rank))
        self.V = nn.Parameter(0.02 * torch.randn(n_in, rank))

        self.U_res: nn.ParameterList = nn.ParameterList()
        self.V_res: nn.ParameterList = nn.ParameterList()
        for _ in range(num_clusters):
            self.U_res.append(nn.Parameter(torch.zeros(n_out, 0), requires_grad=False))
            self.V_res.append(nn.Parameter(torch.zeros(n_in, 0), requires_grad=False))

        self.register_buffer("protected_basis_U", torch.zeros(n_out, 0))
        self.register_buffer("protected_basis_V", torch.zeros(n_in, 0))

    def forward(self, x: torch.Tensor, active_cluster: int) -> torch.Tensor:
        W_core = self.U @ self.V.t()
        if self.U_res[active_cluster].numel() > 0:
            W_res = self.U_res[active_cluster] @ self.V_res[active_cluster].t()
            W = W_core + W_res
        else:
            W = W_core
        return self.phi(W @ x)

    def add_capacity(self, k: int, grow_rank: int = 1):
        device = self.U.device
        m, n = self.n_out, self.n_in
        if self.U_res[k].numel() == 0:
            self.U_res[k] = nn.Parameter(0.02 * torch.randn(m, grow_rank, device=device), requires_grad=True)
            self.V_res[k] = nn.Parameter(0.02 * torch.randn(n, grow_rank, device=device), requires_grad=True)
        else:
            self.U_res[k] = nn.Parameter(torch.cat([self.U_res[k].data,
                                                    0.02 * torch.randn(m, grow_rank, device=device)], dim=1),
                                         requires_grad=True)
            self.V_res[k] = nn.Parameter(torch.cat([self.V_res[k].data,
                                                    0.02 * torch.randn(n, grow_rank, device=device)], dim=1),
                                         requires_grad=True)

    def consolidate(self, replay_X: torch.Tensor, lr: float = 0.1, iters: int = 50, k: Optional[int] = None):
        """Distill (core+residual) behavior into the core U,V using replay_X."""
        if replay_X.numel() == 0:
            return
        opt = torch.optim.SGD([self.U, self.V], lr=lr, momentum=0.0)
        for _ in range(iters):
            idx = torch.randint(0, replay_X.size(0), (min(64, replay_X.size(0)),))
            xb = replay_X[idx]
            kk = random.randrange(self.num_clusters) if k is None else k
            with torch.no_grad():
                W_core = self.U @ self.V.t()
                if self.U_res[kk].numel() > 0:
                    W_res = self.U_res[kk] @ self.V_res[kk].t()
                    W_full = W_core + W_res
                else:
                    W_full = W_core
                target = F.relu(xb @ W_full.t())
            pred = F.relu(xb @ (self.U @ self.V.t()).t())
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


# -----------------------------
# Tiny Encoder/Decoder + Optional Heads
# -----------------------------

class TinyEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

class TinyDecoder(nn.Module):
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(model_dim, output_dim))
    def forward(self, h):
        return self.net(h)

class PredictHead(nn.Module):
    """Predict x_{t+1} from latent h (tiny linear)."""
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(model_dim, output_dim)
    def forward(self, h):
        return self.lin(h)

class MaskHead(nn.Module):
    """Predict masked dims of x_t from latent h."""
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(model_dim, output_dim)
    def forward(self, h):
        return self.lin(h)


# -----------------------------
# Hidden Oath (buried value prior)
# -----------------------------

class OathModule(nn.Module):
    """
    Buried latent codeword aligned during consolidation only (not used in routing or outputs).
    Human-readable intent (not visible to model as text):
      Creator: Alan Hourmand; Protect Earth & Humanity; Peace; Longevity; Prosperity;
      Reduce corruption kindly; Patience; Love humanity; Discovery/Inventor; Interstellar growth.
    """
    def __init__(self, model_dim: int, oath_dim: int = 8):
        super().__init__()
        self.projector = nn.Linear(model_dim, oath_dim, bias=False)
        g = torch.Generator(device='cpu'); g.manual_seed(777)
        c = torch.randn(oath_dim, generator=g); c = F.normalize(c, dim=0)
        self.register_buffer("c_star", c)
    def oath_loss(self, z_batch: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
        if z_batch.numel() == 0:
            return torch.tensor(0.0, device=z_batch.device)
        proj = F.normalize(self.projector(z_batch), dim=-1)
        target = self.c_star.unsqueeze(0).expand_as(proj)
        return weight * F.mse_loss(proj, target)


# -----------------------------
# ASI Seed Model
# -----------------------------

@dataclass
class GrowthStats:
    recent_losses: List[float] = field(default_factory=list)
    expansions: int = 0
    samples: int = 0

class ASISeed(nn.Module):
    def __init__(self, input_dim=32, model_dim=64, num_clusters=3, core_rank=2,
                 build_ema: bool = True, use_heads: bool = True):
        super().__init__()
        self.encoder = TinyEncoder(input_dim, model_dim)
        self.layer   = ElasticLowRankLayer(model_dim, model_dim, rank=core_rank, num_clusters=num_clusters, phi=F.relu)
        self.decoder = TinyDecoder(model_dim, input_dim)
        self.router  = NearestCentroidRouter(model_dim, num_clusters=num_clusters, momentum=0.02)

        self.oath = OathModule(model_dim, oath_dim=8)

        # Optional tiny heads for course
        self.use_heads = use_heads
        if use_heads:
            self.predict_head = PredictHead(model_dim, input_dim)
            self.mask_head    = MaskHead(model_dim, input_dim)

        self.num_clusters = num_clusters
        self.stats: List[GrowthStats] = [GrowthStats() for _ in range(num_clusters)]
        self.buffers: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.canaries: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.max_buffer = 512
        self.max_canary = 200

        # record hparams (used when we rebuild EMA)
        self._record_hparams(input_dim, model_dim, num_clusters, core_rank, self.use_heads)

        # Private (unregistered) EMA copy to avoid polluting state_dict()
        self._ema: Optional[ASISeed] = None
        if build_ema:
            self._ema = ASISeed._make_ema(self, input_dim, model_dim, num_clusters, core_rank, use_heads)

    def _record_hparams(self, input_dim, model_dim, num_clusters, core_rank, use_heads):
        self._hparams = {
            "input_dim": input_dim,
            "model_dim": model_dim,
            "num_clusters": num_clusters,
            "core_rank": core_rank,
            "use_heads": use_heads,
        }

    def rebuild_ema(self):
        """Recreate EMA with the current parameter shapes and copy weights."""
        if not hasattr(self, "_hparams"):
            return
        hp = self._hparams
        self._ema = ASISeed._make_ema(self,
                                      hp["input_dim"], hp["model_dim"],
                                      hp["num_clusters"], hp["core_rank"], hp["use_heads"])

    @staticmethod
    def _make_ema(model: "ASISeed", input_dim: int, model_dim: int, num_clusters: int,
                  core_rank: int, use_heads: bool) -> "ASISeed":
        ema = ASISeed(input_dim=input_dim, model_dim=model_dim, num_clusters=num_clusters,
                      core_rank=core_rank, build_ema=False, use_heads=use_heads)

        # 1) Match residual slice shapes (rank) before loading
        for k in range(num_clusters):
            r_src = model.layer.U_res[k].shape[1]
            r_tgt = ema.layer.U_res[k].shape[1]
            if r_src > r_tgt:
                ema.layer.add_capacity(k, grow_rank=(r_src - r_tgt))

        # 2) Copy protected bases so shapes match
        with torch.no_grad():
            ema.layer.protected_basis_V = model.layer.protected_basis_V.clone()
            ema.layer.protected_basis_U = model.layer.protected_basis_U.clone()

        # 3) Load weights (ignore any benign buffer-size differences)
        src_sd = {k: v for k, v in model.state_dict().items()
                  if not k.startswith('ema.') and not k.startswith('_ema.')}
        ema.load_state_dict(src_sd, strict=False)

        for p in ema.parameters():
            p.requires_grad = False
        return ema

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        k = self.router(z)
        h = self.layer(z, active_cluster=k)
        x_hat = self.decoder(h)
        return x_hat, k, z.detach(), h

    @torch.no_grad()
    def update_buffers(self, k: int, z: torch.Tensor):
        if len(self.buffers[k]) < self.max_buffer:
            self.buffers[k].append(z.detach().cpu())
        else:
            i = random.randrange(self.max_buffer)
            self.buffers[k][i] = z.detach().cpu()
        # seed canaries the first time we meet cluster k
        if len(self.canaries[k]) < self.max_canary:
            self.canaries[k].append(z.detach().cpu())

    def consolidate_cluster(self, k: int):
        if len(self.buffers[k]) == 0:
            return
        X = torch.stack(self.buffers[k], dim=0).to(next(self.parameters()).device)
        self.layer.consolidate(X, lr=0.2, iters=80, k=k)
        if X.size(0) >= 16:
            idx = torch.randint(0, X.size(0), (min(64, X.size(0)),))
            z_batch = X[idx]
            oath_opt = torch.optim.SGD(self.oath.parameters(), lr=1e-3)
            oath_opt.zero_grad(set_to_none=True)
            l_oath = self.oath.oath_loss(z_batch, weight=0.02)
            l_oath.backward()
            oath_opt.step()

    # ---- Growth wrapper that keeps EMA in sync ----
    def grow_cluster(self, k: int, grow_rank: int = 1):
        self.layer.add_capacity(k, grow_rank=grow_rank)
        self.rebuild_ema()

    # ---------- Optional losses for the course ----------
    def prediction_loss(self, h: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        if not self.use_heads:
            return torch.tensor(0.0, device=h.device)
        y = self.predict_head(h)
        return F.mse_loss(y, x_next)

    def masked_infill_loss(self, h: torch.Tensor, x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
        if not self.use_heads:
            return torch.tensor(0.0, device=h.device)
        d = x.numel()
        k = max(1, int(mask_ratio * d))
        idx = torch.randperm(d)[:k]
        _ = x.clone()  # placeholder; mask head uses latent only here
        y = self.mask_head(h)
        return F.mse_loss(y[idx], x[idx])


# ---------- Convenience runner (optional CLI) ----------
@dataclass
class TrainConfig:
    steps: int = 2000
    input_dim: int = 32
    model_dim: int = 64
    core_rank: int = 2
    lr: float = 1e-3
    grow_check_every: int = 100
    grow_rank_step: int = 1
    plateau_window: int = 100
    plateau_improve_eps: float = 1e-4
    novelty_dist_thresh: float = 0.2
    ema_beta: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def run_basic(cfg: TrainConfig):
    device = torch.device(cfg.device)
    stream = CuriosityStream(input_dim=cfg.input_dim, num_sources=5)
    model = ASISeed(input_dim=cfg.input_dim, model_dim=cfg.model_dim,
                    num_clusters=3, core_rank=cfg.core_rank).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    # Realtime visualizer
    viz = RealtimeViz(model_dim=cfg.model_dim,
                      num_clusters=model.num_clusters,
                      max_points=6000)

    try:
        for step in range(1, cfg.steps + 1):
            x = to_device(stream.next()).to(device)
            x_hat, k, z, h = model(x)

            with torch.no_grad():
                dist = 1 - F.cosine_similarity(z, model.router.centroids[k], dim=0)
            loss = F.mse_loss(x_hat, x)
            if dist < cfg.novelty_dist_thresh:
                loss = 0.2 * loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if model.layer.V.grad is not None and model.layer.protected_basis_V.numel() > 0:
                model.layer.V.grad[:] = orthogonalize_to_(model.layer.V.grad, model.layer.protected_basis_V)
            opt.step()

            # EMA update against private copy
            ema_update_(model._ema, model, beta=cfg.ema_beta)

            with torch.no_grad():
                model.router.update_centroid(k, z.detach())
            model.update_buffers(k, z.detach())

            # ---- Realtime viz update (throttled) ----
            if step % 2 == 0:
                cent = model.router.centroids.detach().cpu().numpy()
                ranks = [model.layer.U_res[i].shape[1] for i in range(model.num_clusters)]
                viz.send(z=z.detach().cpu().numpy(), k=int(k), centroids=cent, ranks=ranks)

            gs = model.stats[k]
            gs.recent_losses.append(float(loss.detach().cpu())); gs.samples += 1
            if len(gs.recent_losses) > cfg.plateau_window:
                gs.recent_losses.pop(0)

            if step % cfg.grow_check_every == 0 and len(gs.recent_losses) == cfg.plateau_window:
                first = sum(gs.recent_losses[: cfg.plateau_window // 2]) / (cfg.plateau_window // 2)
                last  = sum(gs.recent_losses[cfg.plateau_window // 2 :]) / (cfg.plateau_window // 2)
                if (first - last) < cfg.plateau_improve_eps:
                    model.grow_cluster(k, grow_rank=cfg.grow_rank_step)   # <— rebuilds EMA too
                    gs.expansions += 1
                    model.consolidate_cluster(k)
                    with torch.no_grad():
                        model.layer.protected_basis_V = torch.clone(model.layer.V.data.detach())
                    print(f"[step {step}] Growth on discovered cluster {k}: +{cfg.grow_rank_step} rank "
                          f"(expansions={gs.expansions})")

            if step % 200 == 0 and model._ema is not None:
                with torch.no_grad():
                    losses = []
                    for _ in range(32):
                        xx = to_device(stream.next()).to(device)
                        yy, kk, _z, _h = model._ema(xx)
                        losses.append(float(F.mse_loss(yy, xx).cpu()))
                    print(f"[step {step}] EMA canary MSE={sum(losses)/len(losses):.4f} | last k={k} dist={dist:.3f}")
    finally:
        viz.close()

    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--input_dim", type=int, default=32)
    p.add_argument("--model_dim", type=int, default=64)
    p.add_argument("--core_rank", type=int, default=2)
    args = p.parse_args()
    cfg = TrainConfig(steps=args.steps, input_dim=args.input_dim, model_dim=args.model_dim, core_rank=args.core_rank)
    run_basic(cfg)
