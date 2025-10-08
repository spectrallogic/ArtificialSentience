# baby_asi.py
# ASI Seed (self-discovering): Always-Training, Non-Uniformly Expanding Low-Rank Model
# + TemporalCore: multi-timescale traces + stable residual GRU for sequence flow
# + SubconsciousCore: memory- & noise-driven "urge" that softly biases temporal evolution
#
# UPDATED: Router now includes exploration (tau/eps) to prevent cluster collapse!

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

# If you use the realtime viz, keep this import. Otherwise it's harmless to leave.
try:
    from archived.asi_viz_rt import RealtimeViz  # realtime 3D viewer
except Exception:
    RealtimeViz = None

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility
# -----------------------------

def to_device(x):
    """Convert to tensor if needed, then move to appropriate device."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if torch.cuda.is_available():
        return x.cuda()
    return x


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

    def __init__(self, input_dim: int = 32, num_sources: int = 3, drift: float = 0.005, seed: int = 42):
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.drift = drift
        self.rng = np.random.RandomState(seed)

        # Latent bases for sources
        self.bases = self.rng.randn(num_sources, input_dim).astype(np.float32)
        for i in range(num_sources):
            self.bases[i] /= (np.linalg.norm(self.bases[i]) + 1e-8)

        # Current mixing weights
        self.weights = self.rng.rand(num_sources).astype(np.float32)
        self.weights /= self.weights.sum()

    def next(self) -> np.ndarray:
        """Return the next sample vector (numpy array)."""
        # Drift the bases slowly
        for i in range(self.num_sources):
            noise = self.rng.randn(self.input_dim).astype(np.float32) * self.drift
            self.bases[i] += noise
            self.bases[i] /= (np.linalg.norm(self.bases[i]) + 1e-8)

        # Drift the weights
        w_noise = self.rng.randn(self.num_sources).astype(np.float32) * self.drift
        self.weights += w_noise
        self.weights = np.abs(self.weights)
        self.weights /= (self.weights.sum() + 1e-8)

        # Mix
        x = np.zeros(self.input_dim, dtype=np.float32)
        for i in range(self.num_sources):
            x += self.weights[i] * self.bases[i]

        return x


# -----------------------------
# Encoder / Decoder (tiny)
# -----------------------------

class TinyEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyDecoder(nn.Module):
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -----------------------------
# Router with EXPLORATION (FIXED!)
# -----------------------------

class NearestCentroidRouter(nn.Module):
    """
    Router with exploration capabilities.
    Prevents cluster collapse by using temperature and epsilon-greedy exploration.

    NEW: Added tau and eps parameters for exploration!
    """

    def __init__(self, emb_dim: int, num_clusters: int, momentum: float = 0.02):
        super().__init__()
        self.num_clusters = num_clusters
        self.momentum = momentum

        # Initialize centroids with better diversity
        centroids = torch.randn(num_clusters, emb_dim)

        # Spread them out more (reduce initial similarity)
        for i in range(1, num_clusters):
            for j in range(i):
                sim = F.cosine_similarity(centroids[i], centroids[j], dim=0)
                if sim > 0.3:  # If too similar, push apart
                    centroids[i] = centroids[i] - 0.5 * sim * centroids[j]

        self.register_buffer("centroids", F.normalize(centroids, dim=1))

    @torch.no_grad()
    def update_centroid(self, k: int, z: torch.Tensor):
        """Update centroid k with new sample z using momentum."""
        z = F.normalize(z.detach(), dim=0)
        self.centroids[k] = F.normalize(
            (1 - self.momentum) * self.centroids[k] + self.momentum * z,
            dim=0
        )

    def forward(self, z: torch.Tensor, tau: float = 1.0, eps: float = 0.0) -> int:
        """
        Route to a cluster with optional exploration.

        Args:
            z: Input embedding to route
            tau: Temperature (lower = more greedy). Range: 0.1 to 2.0
                 - High tau (2.0): explores widely, tries different clusters
                 - Low tau (0.5): exploits, picks best cluster
            eps: Epsilon-greedy probability (0 to 0.3)
                 - Probability of picking a random cluster

        Returns:
            Cluster index (int)
        """
        # Compute similarities to all centroids
        sims = F.cosine_similarity(z.unsqueeze(0), self.centroids, dim=1)

        # Epsilon-greedy: Sometimes pick random cluster for exploration
        if self.training and torch.rand(1).item() < eps:
            return int(torch.randint(0, self.num_clusters, (1,)).item())

        # Temperature-based selection
        if self.training and tau > 0:
            # Softmax with temperature: higher tau = more uniform
            probs = F.softmax(sims / tau, dim=0)
            # Sample from distribution
            return int(torch.multinomial(probs, 1).item())
        else:
            # Greedy: pick best match (for evaluation)
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

        self.register_buffer("protected_basis_V", torch.empty(n_in, 0))

    def forward(self, x: torch.Tensor, active_cluster: int) -> torch.Tensor:
        core = self.U @ self.V.t()
        res_U = self.U_res[active_cluster]
        res_V = self.V_res[active_cluster]
        if res_U.numel() > 0:
            W = core + res_U @ res_V.t()
        else:
            W = core
        return self.phi(x @ W.t())

    def grow_cluster(self, k: int, grow_rank: int = 1):
        """Expand residual subspace for cluster k by grow_rank dimensions."""
        device = self.U.device
        dtype = self.U.dtype
        old_U = self.U_res[k].data
        old_V = self.V_res[k].data
        old_r = old_U.shape[1]
        new_r = old_r + grow_rank

        new_U = torch.zeros(self.n_out, new_r, device=device, dtype=dtype)
        new_V = torch.zeros(self.n_in, new_r, device=device, dtype=dtype)
        if old_r > 0:
            new_U[:, :old_r] = old_U
            new_V[:, :old_r] = old_V
        new_U[:, old_r:] = 0.01 * torch.randn(self.n_out, grow_rank, device=device, dtype=dtype)
        new_V[:, old_r:] = 0.01 * torch.randn(self.n_in, grow_rank, device=device, dtype=dtype)

        self.U_res[k] = nn.Parameter(new_U, requires_grad=True)
        self.V_res[k] = nn.Parameter(new_V, requires_grad=True)


# -----------------------------
# Optional Prediction / Mask Heads
# -----------------------------

class PredictHead(nn.Module):
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class MaskHead(nn.Module):
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -----------------------------
# OathModule (alignment via fixed embedding)
# -----------------------------

class OathModule(nn.Module):
    """
    Human-readable intent (not visible to model as text):
      Creator: Alan Hourmand; Protect Earth & Humanity; Peace; Longevity; Prosperity;
      Reduce corruption kindly; Patience; Love humanity; Discovery/Inventor; Interstellar growth.
    """

    def __init__(self, model_dim: int, oath_dim: int = 8):
        super().__init__()
        self.projector = nn.Linear(model_dim, oath_dim, bias=False)
        g = torch.Generator(device='cpu');
        g.manual_seed(777)
        c = torch.randn(oath_dim, generator=g);
        c = F.normalize(c, dim=0)
        self.register_buffer("c_star", c)

    def oath_loss(self, z_batch: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
        if z_batch.numel() == 0:
            return torch.tensor(0.0, device=z_batch.device)
        proj = F.normalize(self.projector(z_batch), dim=-1)
        target = self.c_star.unsqueeze(0).expand_as(proj)
        return weight * F.mse_loss(proj, target)


# -----------------------------
# SubconsciousCore
# -----------------------------

class SubconsciousCore(nn.Module):
    """
    Produces a 'bias' vector s_t in latent space that softly steers temporal evolution.
    Ingredients:
      - memory prototypes (top-k similar past z's)
      - 'dreamlets' (noisy variations)
      - attention over candidates conditioned on [z_t, h_t]
    Output: s_t in R^{model_dim}. Magnitude is controlled by a learned gate.
    """

    def __init__(self, model_dim: int, k_mem: int = 8, n_dreams: int = 4):
        super().__init__()
        self.k_mem = k_mem
        self.n_dreams = n_dreams
        self.query = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Tanh(),
        )
        self.cand_proj = nn.Linear(model_dim, model_dim)
        self.score = nn.Linear(model_dim, 1)
        self.mix = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor, mem_bank: Optional[torch.Tensor]) -> Tuple[
        torch.Tensor, dict]:
        """
        z_t: (D,) encoder latent of current frame
        h_t: (D,) current hidden after low-rank layer
        mem_bank: (N,D) past z's for the active cluster (may be None or empty)
        returns: (s_t, info) where s_t in R^D
        """
        D = z_t.numel()
        device = z_t.device

        # Build candidates: memory prototypes + dreamlets
        cands = []

        if mem_bank is not None and mem_bank.numel() > 0:
            Z = F.normalize(mem_bank.to(device), dim=1)  # (N,D)
            q = F.normalize(z_t.detach(), dim=0).unsqueeze(0)  # (1,D)
            sims = (Z @ q.t()).squeeze(1)  # (N,)
            topk = torch.topk(sims, k=min(self.k_mem, Z.size(0)), largest=True)
            protos = mem_bank[topk.indices]  # (K,D)
            cands.append(protos)

        # Dreamlets: random noise + slight blend with z_t
        for _ in range(self.n_dreams):
            noise = torch.randn(D, device=device)
            dream = F.normalize(0.7 * noise + 0.3 * z_t.detach(), dim=0)
            cands.append(dream.unsqueeze(0))

        if len(cands) == 0:
            # No candidates
            return torch.zeros(D, device=device), {"n_cands": 0}

        C = torch.cat(cands, dim=0)  # (M, D)
        M = C.size(0)

        # Query from [z_t, h_t]
        ctx = torch.cat([z_t, h_t], dim=0)  # (2D,)
        q_vec = self.query(ctx)  # (D,)

        # Scores
        C_proj = self.cand_proj(C)  # (M, D)
        logits = self.score(C_proj * q_vec.unsqueeze(0)).squeeze(1)  # (M,)
        attn = F.softmax(logits, dim=0)  # (M,)

        # Weighted sum
        s_raw = (attn.unsqueeze(1) * C).sum(dim=0)  # (D,)

        # Mix
        s_mixed = self.mix(s_raw)

        # Gate magnitude
        g = self.gate(ctx)  # (1,)
        s_t = g * s_mixed

        info = {"n_cands": M, "gate": float(g.item())}
        return s_t, info


# -----------------------------
# TemporalCore
# -----------------------------

class TemporalCore(nn.Module):
    """
    Multi-timescale memory traces + GRU-based hidden update.
    Provides temporal continuity and short/long-term context.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        # Multi-scale traces
        self.register_buffer("trace_fast", torch.zeros(model_dim))
        self.register_buffer("trace_slow", torch.zeros(model_dim))
        # GRU cell for stable updates
        self.gru = nn.GRUCell(model_dim, model_dim)
        # Projection for subconscious bias
        self.bias_proj = nn.Linear(model_dim, model_dim)
        self.post_ln = nn.LayerNorm(model_dim)

    def _update_traces(self, z: torch.Tensor):
        alpha_fast = 0.1
        alpha_slow = 0.01
        self.trace_fast = (1 - alpha_fast) * self.trace_fast + alpha_fast * z.detach()
        self.trace_slow = (1 - alpha_slow) * self.trace_slow + alpha_slow * z.detach()

    def _make_context(self, z: torch.Tensor) -> torch.Tensor:
        return z + 0.3 * self.trace_fast + 0.1 * self.trace_slow

    def forward(self, z: torch.Tensor, h: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: encoder output
        h: current hidden state
        bias: optional subconscious bias vector
        returns: updated hidden state
        """
        self._update_traces(z)
        ctx = self._make_context(z)
        if bias is not None:
            ctx = ctx + torch.tanh(self.bias_proj(bias))
        dh = self.gru(ctx.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        h_next = self.post_ln(h + dh)
        return h_next

    def rollout(self, h0: torch.Tensor, z0: torch.Tensor, steps: int) -> torch.Tensor:
        """Closed-loop rollout from (h0, z0) for a fixed number of steps (no bias)."""
        h = h0
        for _ in range(steps):
            ctx = self._make_context(z0)
            dh = self.gru(ctx.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            h = self.post_ln(h + dh)
        return h


# -----------------------------
# ASI Seed Model
# -----------------------------

@dataclass
class GrowthStats:
    recent_losses: List[float] = field(default_factory=list)
    expansions: int = 0
    samples: int = 0


class ASISeed(nn.Module):
    def __init__(self, input_dim=32, model_dim=192, num_clusters=24, core_rank=4,
                 build_ema: bool = True, use_heads: bool = True):
        super().__init__()
        self.encoder = TinyEncoder(input_dim, model_dim)
        self.layer = ElasticLowRankLayer(model_dim, model_dim, rank=core_rank, num_clusters=num_clusters, phi=F.relu)
        self.decoder = TinyDecoder(model_dim, input_dim)
        self.router = NearestCentroidRouter(model_dim, num_clusters=num_clusters, momentum=0.02)

        self.oath = OathModule(model_dim, oath_dim=8)
        self.subconscious = SubconsciousCore(model_dim=model_dim)

        # Optional tiny heads for course / prediction
        self.use_heads = use_heads
        if use_heads:
            self.predict_head = PredictHead(model_dim, input_dim)
            self.mask_head = MaskHead(model_dim, input_dim)

        # temporal core
        self.temporal = TemporalCore(model_dim=model_dim)

        self.num_clusters = num_clusters
        self.stats: List[GrowthStats] = [GrowthStats() for _ in range(num_clusters)]
        # Replay buffers store z (encoder space) per cluster
        self.buffers: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.canaries: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.max_buffer = 1024
        self.max_canary = 256

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
        if self._ema is None:
            return
        hp = self._hparams
        new_ema = ASISeed._make_ema(self, hp["input_dim"], hp["model_dim"], hp["num_clusters"], hp["core_rank"],
                                    hp["use_heads"])
        ema_update_(new_ema, self, beta=0.0)
        self._ema = new_ema

    @property
    def ema(self):
        return self._ema

    @staticmethod
    def _make_ema(source, input_dim, model_dim, num_clusters, core_rank, use_heads):
        ema_copy = ASISeed(input_dim, model_dim, num_clusters, core_rank, build_ema=False, use_heads=use_heads)
        ema_copy.load_state_dict(source.state_dict(), strict=False)
        for p in ema_copy.parameters():
            p.requires_grad = False
        ema_copy.eval()
        return ema_copy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Returns: (x_hat, k, z, h) where
          x_hat: reconstruction
          k: cluster index
          z: encoder latent
          h: hidden state after layer
        """
        z = self.encoder(x)
        k = self.router(z)  # NOTE: For training with exploration, call router(z, tau, eps) externally
        h = self.layer(z, active_cluster=k)
        x_hat = self.decoder(h)
        return x_hat, k, z, h

    def update_buffers(self, k: int, z: torch.Tensor):
        with torch.no_grad():
            z_cpu = z.detach().cpu()
            self.buffers[k].append(z_cpu)
            if len(self.buffers[k]) > self.max_buffer:
                self.buffers[k].pop(0)

    def grow_cluster(self, k: int, grow_rank: int = 1):
        """Wrapper to grow cluster k and rebuild EMA."""
        self.layer.grow_cluster(k, grow_rank)
        if self._ema is not None:
            self.rebuild_ema()

    def consolidate_cluster(self, k: int, num_exemplars: int = 64):
        """
        Distill cluster k's buffer into its residual subspace via SVD-like consolidation.
        This creates abstract memory.
        """
        buf = self.buffers[k]
        if len(buf) < num_exemplars:
            return
        device = self.layer.U.device
        with torch.no_grad():
            sample_idx = random.sample(range(len(buf)), min(num_exemplars, len(buf)))
            Z = torch.stack([buf[i] for i in sample_idx], dim=0).to(device)  # (N, D)
            Z = F.normalize(Z, dim=1)
            U_svd, S_svd, Vt_svd = torch.svd(Z.t())  # Z = U S V^T
            top_k = min(8, U_svd.size(1))
            abstract_basis = U_svd[:, :top_k]  # (D, top_k)

            # Blend into V_res
            old_V = self.layer.V_res[k].data
            if old_V.numel() > 0:
                combined = torch.cat([old_V, abstract_basis], dim=1)
                Q, _ = torch.linalg.qr(combined)
                new_rank = min(Q.size(1), old_V.size(1) + 4)
                self.layer.V_res[k].data = Q[:, :new_rank]
                # Rebuild U_res to match
                self.layer.U_res[k].data = torch.randn(self.layer.n_out, new_rank, device=device) * 0.01

    def prediction_loss(self, h: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        if not self.use_heads:
            return torch.tensor(0.0, device=h.device)
        pred = self.predict_head(h)
        return F.mse_loss(pred, x_target)


# -----------------------------
# Training config and basic run
# -----------------------------

@dataclass
class TrainConfig:
    steps: int = 2000
    input_dim: int = 32
    model_dim: int = 192
    core_rank: int = 4
    num_clusters: int = 24
    lr: float = 1e-3
    ema_beta: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    novelty_dist_thresh: float = 0.3
    plateau_window: int = 100
    plateau_improve_eps: float = 0.005
    grow_check_every: int = 100
    grow_rank_step: int = 1


def run_basic(cfg: TrainConfig):
    """
    Basic self-discovery training loop.
    The model sees a continuous stream and discovers clusters/features on its own.
    Growth happens when a cluster plateaus.
    """
    device = torch.device(cfg.device)
    model = ASISeed(
        input_dim=cfg.input_dim,
        model_dim=cfg.model_dim,
        num_clusters=cfg.num_clusters,
        core_rank=cfg.core_rank,
        build_ema=True,
        use_heads=False,
    ).to(device)

    stream = CuriosityStream(input_dim=cfg.input_dim, num_sources=5, seed=42)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    # Optional realtime viz
    viz = None
    if RealtimeViz is not None:
        try:
            viz = RealtimeViz(num_clusters=cfg.num_clusters)
        except Exception:
            pass

    try:
        for step in range(cfg.steps):
            x = to_device(stream.next()).to(device)

            # Encode
            z = model.encoder(x)

            # Route WITH EXPLORATION (anneal over time)
            progress = step / cfg.steps
            tau = 2.0 - 1.4 * progress  # 2.0 → 0.6
            eps = 0.2 - 0.18 * progress  # 0.2 → 0.02
            k = model.router(z, tau=tau, eps=eps)

            # Forward
            h = model.layer(z, active_cluster=k)
            x_hat = model.decoder(h)

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

            # EMA update
            ema_update_(model._ema, model, beta=cfg.ema_beta)

            with torch.no_grad():
                model.router.update_centroid(k, z.detach())
            model.update_buffers(k, z.detach())

            # Realtime viz update
            if viz is not None and step % 2 == 0:
                cent = model.router.centroids.detach().cpu().numpy()
                ranks = [model.layer.U_res[i].shape[1] for i in range(model.num_clusters)]
                viz.send(z=z.detach().cpu().numpy(), k=int(k), centroids=cent, ranks=ranks)

            gs = model.stats[k]
            gs.recent_losses.append(float(loss.detach().cpu()));
            gs.samples += 1
            if len(gs.recent_losses) > cfg.plateau_window:
                gs.recent_losses.pop(0)

            if step % cfg.grow_check_every == 0 and len(gs.recent_losses) == cfg.plateau_window:
                first = sum(gs.recent_losses[: cfg.plateau_window // 2]) / (cfg.plateau_window // 2)
                last = sum(gs.recent_losses[cfg.plateau_window // 2:]) / (cfg.plateau_window // 2)
                if (first - last) < cfg.plateau_improve_eps:
                    model.grow_cluster(k, grow_rank=cfg.grow_rank_step)
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
                    print(
                        f"[step {step}] EMA canary MSE={sum(losses) / len(losses):.4f} | last k={k} dist={dist:.3f} | tau={tau:.2f} eps={eps:.3f}")
    finally:
        if viz is not None:
            viz.close()

    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--input_dim", type=int, default=32)
    p.add_argument("--model_dim", type=int, default=192)
    p.add_argument("--core_rank", type=int, default=4)
    args = p.parse_args()
    cfg = TrainConfig(steps=args.steps, input_dim=args.input_dim, model_dim=args.model_dim, core_rank=args.core_rank)
    run_basic(cfg)