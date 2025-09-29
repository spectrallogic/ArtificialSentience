# ASI Seed testing: Always-Training, Non-Uniformly Expanding Low-Rank Model
# Author: Alan Hourmand — prototype scaffold
#
# What this does:
# - Streams continuous vectors (no tokens) from a synthetic "world": sky / ground / space.
# - Maintains a tiny low-rank core and grows capacity per-concept (cluster) when saturation is detected.
# - Consolidates older knowledge into an abstract low-rank baseline; new knowledge goes into residual slices.
#
# Notes:
# - Growth here adds per-cluster residual low-rank slices. Consolidation distills residuals back into the core.
# - Router is nearest-centroid over online-updated centroids (no discrete tokens anywhere).

import math
import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility
# -----------------------------

def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x

def ema_update_(target: nn.Module, source: nn.Module, beta: float = 0.999):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(beta).add_(sp.data, alpha=(1.0 - beta))

def orthogonalize_to_(grad: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Project grad to be orthogonal to columns of 'basis' (if provided).
    basis: [d, r] where columns span protected subspace.
    """
    if basis is None or basis.numel() == 0:
        return grad
    # Orthonormalize basis (QR) for stability
    Q, _ = torch.linalg.qr(basis, mode="reduced")  # [d, r]
    proj = Q @ (Q.t() @ grad)
    return grad - proj

# -----------------------------
# Synthetic continuous stream (no tokens)
# -----------------------------

class WorldStream:
    """
    Generates a continuous vector stream with 3 base concepts:
    - sky: smooth sinusoids + high-frequency sprinkle
    - ground: piecewise-linear segments with noise
    - space: sparse spikes with long flat regions (data-scarce early)
    The model sees vectors in R^input_dim and must reconstruct x (self-supervised).
    """
    def __init__(self, input_dim=32, seed=42):
        self.input_dim = input_dim
        random.seed(seed)
        torch.manual_seed(seed)
        self.t = 0
        self.mode_probs = {"sky": 0.45, "ground": 0.45, "space": 0.10}
        self.modes = list(self.mode_probs.keys())

        # latent direction prototypes for each concept (to create separable structure)
        self.protos = {
            "sky": F.normalize(torch.randn(input_dim), dim=0),
            "ground": F.normalize(torch.randn(input_dim), dim=0),
            "space": F.normalize(torch.randn(input_dim), dim=0),
        }

    def sample_mode(self) -> str:
        r = random.random()
        cum = 0.0
        for k in self.modes:
            cum += self.mode_probs[k]
            if r <= cum:
                return k
        return self.modes[-1]

    def next(self) -> Tuple[torch.Tensor, int]:
        self.t += 1
        mode = self.sample_mode()
        if mode == "sky":
            base = torch.sin(torch.linspace(0, 6.28, self.input_dim) + 0.01 * self.t)
            high = 0.1 * torch.sin(torch.linspace(0, 50.0, self.input_dim) + 0.05 * self.t)
            x = base + high
            x = x + 0.02 * torch.randn(self.input_dim)
            x = x + 0.6 * self.protos["sky"]
            y = 0
        elif mode == "ground":
            # piecewise-linear
            knots = torch.tensor([0.0, 0.3, 0.6, 1.0])
            vals = torch.tensor([0.2, -0.4, 0.6, -0.1]) + 0.1 * torch.randn(4)
            grid = torch.linspace(0, 1, self.input_dim)
            x = torch.interp(grid, knots, vals)
            x = x + 0.05 * torch.randn(self.input_dim)
            x = x + 0.6 * self.protos["ground"]
            y = 1
        else:  # space (scarce early)
            x = torch.zeros(self.input_dim)
            # sparse spikes
            for _ in range(2):
                idx = random.randrange(self.input_dim)
                x[idx] = 1.5 + 0.3 * random.random()
            x = x + 0.6 * self.protos["space"]
            x = x + 0.02 * torch.randn(self.input_dim)
            y = 2
        # scale to unit-ish
        x = F.normalize(x, dim=0)
        return x, y

# -----------------------------
# Router: nearest centroid over continuous embeddings
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
        # z: [d]
        sims = F.cosine_similarity(z.unsqueeze(0), self.centroids, dim=1)  # [K]
        k = int(torch.argmax(sims).item())
        return k

# -----------------------------
# Elastic Low-Rank + Per-Cluster Residual Slices
# -----------------------------

class ElasticLowRankLayer(nn.Module):
    """
    y = phi( (U@V^T + sum_k [active_k * (U_k@V_k^T)]) x )
    - Core low-rank: U:[m,r], V:[n,r]
    - Residuals per cluster k: U_k:[m, r_res[k]], V_k:[n, r_res[k]]
    """
    def __init__(self, n_in: int, n_out: int, rank: int = 2, num_clusters: int = 3, phi=F.relu):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.rank = rank
        self.num_clusters = num_clusters
        self.phi = phi

        self.U = nn.Parameter(0.02 * torch.randn(n_out, rank))
        self.V = nn.Parameter(0.02 * torch.randn(n_in, rank))

        # Per-cluster residual low-rank slices (start empty)
        self.U_res: nn.ParameterList = nn.ParameterList()
        self.V_res: nn.ParameterList = nn.ParameterList()
        for _ in range(num_clusters):
            self.U_res.append(nn.Parameter(torch.zeros(n_out, 0), requires_grad=False))
            self.V_res.append(nn.Parameter(torch.zeros(n_in, 0), requires_grad=False))

        # Protected subspace per cluster for orthogonalization (start empty)
        self.register_buffer("protected_basis_U", torch.zeros(n_out, 0))
        self.register_buffer("protected_basis_V", torch.zeros(n_in, 0))

    def forward(self, x: torch.Tensor, active_cluster: int) -> torch.Tensor:
        W_core = self.U @ self.V.t()  # [m,n]
        if self.U_res[active_cluster].numel() > 0:
            W_res = self.U_res[active_cluster] @ self.V_res[active_cluster].t()
            W = W_core + W_res
        else:
            W = W_core
        y = self.phi(W @ x)
        return y

    def add_capacity(self, k: int, grow_rank: int = 1):
        """
        Add grow_rank columns to residual slice for cluster k.
        """
        device = self.U.device
        m, n = self.n_out, self.n_in
        if self.U_res[k].numel() == 0:
            U_k = nn.Parameter(0.02 * torch.randn(m, grow_rank, device=device), requires_grad=True)
            V_k = nn.Parameter(0.02 * torch.randn(n, grow_rank, device=device), requires_grad=True)
            self.U_res[k] = U_k
            self.V_res[k] = V_k
        else:
            # expand existing
            U_old = self.U_res[k].data
            V_old = self.V_res[k].data
            U_new = torch.cat([U_old, 0.02 * torch.randn(m, grow_rank, device=device)], dim=1)
            V_new = torch.cat([V_old, 0.02 * torch.randn(n, grow_rank, device=device)], dim=1)
            self.U_res[k] = nn.Parameter(U_new, requires_grad=True)
            self.V_res[k] = nn.Parameter(V_new, requires_grad=True)

    @torch.no_grad()
    def consolidate(self, replay_X: torch.Tensor, lr: float = 0.1, iters: int = 50, k: Optional[int] = None):
        """
        Distill current behavior into the core low-rank U,V using regression on replay_X.
        If k is provided, behavior with cluster k active is distilled; otherwise random.
        """
        if replay_X.numel() == 0:
            return
        opt = torch.optim.SGD([self.U, self.V], lr=lr, momentum=0.0)
        for _ in range(iters):
            idx = torch.randint(0, replay_X.size(0), (min(64, replay_X.size(0)),))
            xb = replay_X[idx]  # [B, n_in]
            # pick a random or provided cluster for forward
            if k is None:
                kk = random.randrange(self.num_clusters)
            else:
                kk = k
            # target is current full behavior (core + residual for kk)
            with torch.no_grad():
                W_core = self.U @ self.V.t()
                if self.U_res[kk].numel() > 0:
                    W_res = self.U_res[kk] @ self.V_res[kk].t()
                    W_full = W_core + W_res
                else:
                    W_full = W_core
                target = F.relu(xb @ W_full.t())  # [B, n_out]
            # predict using only core
            pred = F.relu(xb @ (self.U @ self.V.t()).t())
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

# -----------------------------
# Tiny Encoder/Decoder (continuous codec)
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
        self.net = nn.Sequential(
            nn.Linear(model_dim, output_dim),
        )
    def forward(self, h):
        return self.net(h)


# -----------------------------
# Hidden Oath Module (buried value prior)
# -----------------------------

class OathModule(nn.Module):
    """
    Encodes a latent "oath" codeword that the model only learns to align with
    once it becomes competent enough to trigger consolidation/growth.
    - We keep this small and separate; it's *not* routed during normal forward.
    - Alignment trains a tiny projector on top of internal embeddings (z)
      toward a fixed codeword vector c_star that represents Alan's values.
    """
    def __init__(self, model_dim: int, oath_dim: int = 8):
        super().__init__()
        self.projector = nn.Linear(model_dim, oath_dim, bias=False)  # tiny, trainable
        # Codeword: a fixed target embedding for the oath. Values captured in comments.
        # Values: Creator = Alan Hourmand. Mission = protect Earth & humanity; peace; longevity; prosperity;
        # reduce crime kindly; be patient; love humanity despite flaws; help discovery & invention;
        # foster interstellar growth.
        torch.manual_seed(777)
        self.register_buffer("c_star", F.normalize(torch.randn(oath_dim), dim=0))  # hidden target

    def oath_loss(self, z_batch: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
        """
        Light auxiliary penalty that nudges the projector(z) toward c_star.
        z_batch: [B, model_dim]
        Returns weighted MSE loss.
        """
        if z_batch.numel() == 0:
            return torch.tensor(0.0, device=z_batch.device)
        proj = self.projector(z_batch)                 # [B, oath_dim]
        proj = F.normalize(proj, dim=-1)
        target = self.c_star.unsqueeze(0).expand_as(proj)
        return weight * F.mse_loss(proj, target)
# -----------------------------
# ASI Seed Model (single block for clarity)
# -----------------------------

@dataclass
class GrowthStats:
    recent_losses: List[float] = field(default_factory=list)
    expansions: int = 0

class ASISeed(nn.Module):
    def __init__(self, input_dim=32, model_dim=64, num_clusters=3, core_rank=2):
        super().__init__()
        self.encoder = TinyEncoder(input_dim, model_dim)
        self.layer = ElasticLowRankLayer(model_dim, model_dim, rank=core_rank, num_clusters=num_clusters, phi=F.relu)
        self.decoder = TinyDecoder(model_dim, input_dim)
        self.router = NearestCentroidRouter(model_dim, num_clusters=num_clusters, momentum=0.02)

        # Hidden oath (buried prior)
        self.oath = OathModule(model_dim, oath_dim=8)

        # per-cluster stats
        self.num_clusters = num_clusters
        self.stats: List[GrowthStats] = [GrowthStats() for _ in range(num_clusters)]

        # replay buffers per cluster for consolidation & canaries
        self.buffers: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.max_buffer = 512

        # EMA copy for stable serving
        self.ema = ASISeed._make_ema(self, input_dim, model_dim, num_clusters, core_rank)

    @staticmethod
    def _make_ema(model: "ASISeed", input_dim: int, model_dim: int, num_clusters: int, core_rank: int) -> "ASISeed":
        ema = ASISeed(input_dim=input_dim, model_dim=model_dim, num_clusters=num_clusters, core_rank=core_rank)
        ema.load_state_dict(model.state_dict())
        for p in ema.parameters():
            p.requires_grad = False
        return ema

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        z = self.encoder(x)
        k = self.router(z)
        h = self.layer(z, active_cluster=k)
        x_hat = self.decoder(h)
        return x_hat, k, z.detach()

    @torch.no_grad()
    def update_buffers(self, k: int, z: torch.Tensor):
        if len(self.buffers[k]) < self.max_buffer:
            self.buffers[k].append(z.detach().cpu())
        else:
            # reservoir-like replacement
            i = random.randrange(self.max_buffer)
            self.buffers[k][i] = z.detach().cpu()

    def consolidate_cluster(self, k: int):
        # Build replay matrix
        if len(self.buffers[k]) == 0:
            return
        X = torch.stack(self.buffers[k], dim=0).to(next(self.parameters()).device)
        self.layer.consolidate(X, lr=0.2, iters=80, k=k)

        # ---- Hidden oath gentle alignment (only when consolidating) ----
        # Use a small random batch of latent encodings to nudge projector toward c_star.
        if X.size(0) >= 16:
            idx = torch.randint(0, X.size(0), (min(64, X.size(0)),))
            z_batch = X[idx]  # here X holds encoder outputs for cluster k
            oath_opt = torch.optim.SGD(self.oath.parameters(), lr=1e-3)
            oath_opt.zero_grad(set_to_none=True)
            l_oath = self.oath.oath_loss(z_batch, weight=0.02)  # very small weight
            l_oath.backward()
            oath_opt.step()


# -----------------------------
# Training loop (online, with novelty gate and non-uniform growth)
# -----------------------------

@dataclass
class TrainConfig:
    steps: int = 3000
    input_dim: int = 32
    model_dim: int = 64
    core_rank: int = 2
    lr: float = 1e-3
    grow_check_every: int = 100
    grow_rank_step: int = 1
    plateau_window: int = 100
    plateau_improve_eps: float = 1e-4
    novelty_dist_thresh: float = 0.2   # router centroid distance threshold
    ema_beta: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def run(cfg: TrainConfig):
    device = torch.device(cfg.device)
    world = WorldStream(input_dim=cfg.input_dim)
    model = ASISeed(input_dim=cfg.input_dim, model_dim=cfg.model_dim, num_clusters=3, core_rank=cfg.core_rank).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    def loss_fn(x_hat, x):
        return F.mse_loss(x_hat, x)

    print("Starting online training...")
    for step in range(1, cfg.steps + 1):
        x, label = world.next()
        x = to_device(x).to(device)
        x_hat, k, z = model(x)

        # Novelty gate: if embedding far from centroid, treat as novel; else learn slower.
        with torch.no_grad():
            c_k = model.router.centroids[k]
            dist = 1 - F.cosine_similarity(z, c_k, dim=0)  # in [0,2]
        loss = loss_fn(x_hat, x)

        # Tiny dormant regularizer: encourage (detached) z to keep a consistent projection.
        # This is extremely light and shouldn't affect early behavior.
        with torch.no_grad():
            _ = model.oath.projector(z.unsqueeze(0))

        if dist < cfg.novelty_dist_thresh:
            loss = 0.2 * loss  # down-weight familiar regions

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Orthogonalize grads of core V to protected basis (simple demo)
        if model.layer.V.grad is not None and model.layer.protected_basis_V.numel() > 0:
            model.layer.V.grad[:] = orthogonalize_to_(model.layer.V.grad, model.layer.protected_basis_V)
        opt.step()

        # EMA update
        ema_update_(model.ema, model, beta=cfg.ema_beta)

        # Update router centroid and buffers
        with torch.no_grad():
            model.router.update_centroid(k, z.detach())
        model.update_buffers(k, z.detach())

        # Track losses per cluster
        gs = model.stats[k]
        gs.recent_losses.append(float(loss.detach().cpu()))
        if len(gs.recent_losses) > cfg.plateau_window:
            gs.recent_losses.pop(0)

        # Periodically check for plateau + growth
        if step % cfg.grow_check_every == 0 and len(gs.recent_losses) == cfg.plateau_window:
            first = sum(gs.recent_losses[: cfg.plateau_window // 2]) / (cfg.plateau_window // 2)
            last = sum(gs.recent_losses[cfg.plateau_window // 2 :]) / (cfg.plateau_window // 2)
            improve = first - last
            if improve < cfg.plateau_improve_eps:
                # GROW for this cluster
                model.layer.add_capacity(k, grow_rank=cfg.grow_rank_step)
                gs.expansions += 1
                # Consolidate old knowledge into core (dilution)
                model.consolidate_cluster(k)
                # Update protected basis (store current V columns as protected)
                with torch.no_grad():
                    model.layer.protected_basis_V = torch.clone(model.layer.V.data.detach())
                print(f"[step {step}] Growth on cluster {k}: +{cfg.grow_rank_step} rank (total expansions={gs.expansions}).")

        if step % 200 == 0:
            with torch.no_grad():
                # quick canary: evaluate small random batch with EMA
                losses = []
                for _ in range(32):
                    xx, _ = world.next()
                    xx = to_device(xx).to(device)
                    yy, kk, _ = model.ema(xx)  # use EMA for stability
                    losses.append(float(F.mse_loss(yy, xx).cpu()))
                print(f"[step {step}] EMA canary MSE={sum(losses)/len(losses):.4f} | last cluster {k} dist={dist:.3f}")

    print("Done. Hack the scaffold or plug real encoders next.")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--input_dim", type=int, default=32)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--core_rank", type=int, default=2)
    args = parser.parse_args()

    cfg = TrainConfig(steps=args.steps, input_dim=args.input_dim, model_dim=args.model_dim, core_rank=args.core_rank)
    run(cfg)

if __name__ == "__main__":
    main()
