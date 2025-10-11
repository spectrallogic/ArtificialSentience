# baby_asi.py
# ASI Seed (self-discovering): Always-Training, Non-Uniformly Expanding Low-Rank Model
# + TemporalCore: multi-timescale traces + stable residual GRU for sequence flow
# + SubconsciousCore: memory- & noise-driven "urge" that softly biases temporal evolution
# + Easy-to-use growth mechanism: Just call check_and_grow_cluster()!
# + RECURSIVE REFINEMENT: Iteratively improve answers like humans solving puzzles!
#
# UPDATED: Router now includes exploration (tau/eps) to prevent cluster collapse!
# UPDATED: Simplified growth checking for external use
# UPDATED: Added recursive refinement methods for deep supervision

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
# Router with EXPLORATION
# -----------------------------

class NearestCentroidRouter(nn.Module):
    """
    Router with exploration capabilities.
    Prevents cluster collapse by using temperature and epsilon-greedy exploration.
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
            eps: Epsilon-greedy probability (0 to 0.3)

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
            probs = F.softmax(sims / tau, dim=0)
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
        g = torch.Generator(device='cpu')
        g.manual_seed(777)
        c = torch.randn(oath_dim, generator=g)
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
            return torch.zeros(D, device=device), {"n_cands": 0}

        C = torch.cat(cands, dim=0)  # (M, D)
        M = C.size(0)

        ctx = torch.cat([z_t, h_t], dim=0)  # (2D,)
        q_vec = self.query(ctx)  # (D,)

        C_proj = self.cand_proj(C)  # (M, D)
        logits = self.score(C_proj * q_vec.unsqueeze(0)).squeeze(1)  # (M,)
        attn = F.softmax(logits, dim=0)  # (M,)

        s_raw = (attn.unsqueeze(1) * C).sum(dim=0)  # (D,)
        s_mixed = self.mix(s_raw)

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
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.register_buffer("trace_fast", torch.zeros(model_dim))
        self.register_buffer("trace_slow", torch.zeros(model_dim))
        self.gru = nn.GRUCell(model_dim, model_dim)
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
        self._update_traces(z)
        ctx = self._make_context(z)
        if bias is not None:
            ctx = ctx + torch.tanh(self.bias_proj(bias))
        dh = self.gru(ctx.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        h_next = self.post_ln(h + dh)
        return h_next

    def rollout(self, h0: torch.Tensor, z0: torch.Tensor, steps: int) -> torch.Tensor:
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

        self.use_heads = use_heads
        if use_heads:
            self.predict_head = PredictHead(model_dim, input_dim)
            self.mask_head = MaskHead(model_dim, input_dim)

        self.temporal = TemporalCore(model_dim=model_dim)

        self.num_clusters = num_clusters
        self.stats: List[GrowthStats] = [GrowthStats() for _ in range(num_clusters)]
        self.buffers: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.canaries: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.max_buffer = 1024
        self.max_canary = 256

        self._record_hparams(input_dim, model_dim, num_clusters, core_rank, self.use_heads)

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
        z = self.encoder(x)
        k = self.router(z)
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
        self.layer.grow_cluster(k, grow_rank)
        if self._ema is not None:
            self.rebuild_ema()

    def consolidate_cluster(self, k: int, num_exemplars: int = 64):
        buf = self.buffers[k]
        if len(buf) < num_exemplars:
            return
        device = self.layer.U.device
        with torch.no_grad():
            sample_idx = random.sample(range(len(buf)), min(num_exemplars, len(buf)))
            Z = torch.stack([buf[i] for i in sample_idx], dim=0).to(device)
            Z = F.normalize(Z, dim=1)
            U_svd, S_svd, Vt_svd = torch.svd(Z.t())
            top_k = min(8, U_svd.size(1))
            abstract_basis = U_svd[:, :top_k]

            old_V = self.layer.V_res[k].data
            if old_V.numel() > 0:
                combined = torch.cat([old_V, abstract_basis], dim=1)
                Q, _ = torch.linalg.qr(combined)
                new_rank = min(Q.size(1), old_V.size(1) + 4)
                self.layer.V_res[k].data = Q[:, :new_rank]
                self.layer.U_res[k].data = torch.randn(self.layer.n_out, new_rank, device=device) * 0.01

    def prediction_loss(self, h: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        if not self.use_heads:
            return torch.tensor(0.0, device=h.device)
        pred = self.predict_head(h)
        return F.mse_loss(pred, x_target)

    def check_and_grow_cluster(self, k: int, loss: float,
                               window_size: int = 50,
                               improvement_threshold: float = 0.01,
                               grow_rank: int = 2,
                               verbose: bool = True) -> bool:
        """
        Check if cluster k has plateaued and grow it if needed.
        """
        stats = self.stats[k]

        stats.recent_losses.append(loss)
        stats.samples += 1

        if len(stats.recent_losses) > window_size:
            stats.recent_losses.pop(0)

        if len(stats.recent_losses) < window_size:
            return False

        first_half = stats.recent_losses[:window_size // 2]
        second_half = stats.recent_losses[window_size // 2:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        improvement = first_avg - second_avg

        if improvement < improvement_threshold and second_avg > 0.015:
            old_rank = self.layer.U_res[k].shape[1] if self.layer.U_res[k].numel() > 0 else 0
            self.grow_cluster(k, grow_rank=grow_rank)
            new_rank = self.layer.U_res[k].shape[1]
            stats.expansions += 1
            stats.recent_losses.clear()

            if verbose:
                print(f"🌱 GROWTH! Cluster {k}: rank {old_rank} → {new_rank} "
                      f"(loss={second_avg:.4f}, improvement={improvement:.4f})")

            return True

        return False

    # ========================================
    # RECURSIVE REFINEMENT METHODS
    # ========================================

    def refine_answer(self, x_input: torch.Tensor, x_current: torch.Tensor,
                      tau: float = 0.5, eps: float = 0.0) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Try to improve current answer by re-processing it with the input.
        This is the key to recursive refinement!

        Args:
            x_input: Original input (the question)
            x_current: Current answer attempt
            tau: Temperature for routing
            eps: Epsilon for exploration

        Returns:
            (refined_output, cluster_used, latent_state)
        """
        # Combine current answer with input to guide refinement
        combined = x_input + 0.3 * x_current

        # Encode the combined state
        z = self.encoder(combined)

        # Route (can explore or exploit based on tau/eps)
        k = self.router(z, tau=tau, eps=eps)

        # Process through cluster
        h = self.layer(z, active_cluster=k)

        # Generate refined output
        x_refined = self.decoder(h)

        return x_refined, k, h

    def evaluate_improvement(self, x_old: torch.Tensor, x_new: torch.Tensor,
                             x_target: torch.Tensor) -> Tuple[float, float, bool]:
        """
        Evaluate if refinement improved the answer.

        Returns:
            (old_error, new_error, did_improve)
        """
        old_error = F.mse_loss(x_old, x_target).item()
        new_error = F.mse_loss(x_new, x_target).item()

        improvement = old_error - new_error
        did_improve = improvement > 0.0001

        return old_error, new_error, did_improve

    def recursive_solve(self, x_input: torch.Tensor, x_target: torch.Tensor,
                        max_iterations: int = 5, tau: float = 1.0, eps: float = 0.1,
                        verbose: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Recursively refine the answer, like humans iterating on a puzzle.

        Inspired by TRM (Tiny Recursive Model) paper - deep supervision
        through iterative refinement.

        Args:
            x_input: The question/input grid
            x_target: The desired output
            max_iterations: How many refinement attempts
            tau: Temperature (high = explore, low = exploit)
            eps: Epsilon-greedy exploration
            verbose: Print iteration details

        Returns:
            (best_answer, info_dict)
        """
        # Initial attempt (forward pass)
        z = self.encoder(x_input)
        k = self.router(z, tau=tau, eps=eps)
        h = self.layer(z, active_cluster=k)
        x_current = self.decoder(h)

        # Track refinement process
        # Store DETACHED versions for tracking, but keep live version for backprop
        attempts_detached = [x_current.detach()]
        attempts_live = [x_current]  # Keep non-detached for backprop!
        errors = [F.mse_loss(x_current, x_target).item()]
        clusters_used = [k]
        improvements = []

        if verbose:
            print(f"  Initial: error={errors[0]:.5f}, cluster={k}")

        # Iteratively refine
        for iteration in range(max_iterations):
            # Try to improve current answer
            x_refined, k_used, h_refined = self.refine_answer(
                x_input, x_current.detach(),  # Detach input to refinement
                tau=tau * 0.8,  # Gradually reduce exploration
                eps=eps * 0.8
            )

            # Evaluate improvement
            old_err, new_err, improved = self.evaluate_improvement(
                x_current, x_refined, x_target
            )

            attempts_detached.append(x_refined.detach())
            attempts_live.append(x_refined)  # Keep non-detached!
            errors.append(new_err)
            clusters_used.append(k_used)
            improvements.append(new_err - old_err)

            if verbose:
                status = "✓" if improved else "✗"
                print(f"  Iter {iteration + 1}: error={new_err:.5f}, "
                      f"cluster={k_used}, improvement={improvements[-1]:+.5f} {status}")

            # Update current answer
            x_current = x_refined

            # Early stopping if converged
            if new_err < 0.0001:
                if verbose:
                    print(f"  → Converged at iteration {iteration + 1}")
                break

        # Determine if model is "stuck"
        is_stuck = (
                len(improvements) >= 3 and
                sum(improvements[-3:]) > -0.001 and
                errors[-1] > 0.01
        )

        info = {
            'attempts': attempts_detached,  # Use detached for info
            'errors': errors,
            'clusters_used': clusters_used,
            'improvements': improvements,
            'iterations': len(attempts_detached) - 1,
            'final_error': errors[-1],
            'is_stuck': is_stuck,
            'best_iteration': int(np.argmin(errors))
        }

        # Return LIVE (non-detached) best attempt for backprop!
        best_idx = info['best_iteration']
        best_answer = attempts_live[best_idx]  # Use live version!

        return best_answer, info


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
    """Basic self-discovery training loop."""
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

    viz = None
    if RealtimeViz is not None:
        try:
            viz = RealtimeViz(num_clusters=cfg.num_clusters)
        except Exception:
            pass

    try:
        for step in range(cfg.steps):
            x = to_device(stream.next()).to(device)

            z = model.encoder(x)

            progress = step / cfg.steps
            tau = 2.0 - 1.4 * progress
            eps = 0.2 - 0.18 * progress
            k = model.router(z, tau=tau, eps=eps)

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

            ema_update_(model._ema, model, beta=cfg.ema_beta)

            with torch.no_grad():
                model.router.update_centroid(k, z.detach())
            model.update_buffers(k, z.detach())

            if viz is not None and step % 2 == 0:
                cent = model.router.centroids.detach().cpu().numpy()
                ranks = [model.layer.U_res[i].shape[1] for i in range(model.num_clusters)]
                viz.send(z=z.detach().cpu().numpy(), k=int(k), centroids=cent, ranks=ranks)

            if step % cfg.grow_check_every == 0:
                if model.check_and_grow_cluster(
                        k, loss.item(),
                        window_size=cfg.plateau_window,
                        improvement_threshold=cfg.plateau_improve_eps,
                        grow_rank=cfg.grow_rank_step
                ):
                    model.consolidate_cluster(k)
                    with torch.no_grad():
                        model.layer.protected_basis_V = torch.clone(model.layer.V.data.detach())

            if step % 200 == 0 and model._ema is not None:
                with torch.no_grad():
                    losses = []
                    for _ in range(32):
                        xx = to_device(stream.next()).to(device)
                        yy, kk, _z, _h = model._ema(xx)
                        losses.append(float(F.mse_loss(yy, xx).cpu()))
                    print(f"[step {step}] EMA canary MSE={sum(losses) / len(losses):.4f} | "
                          f"last k={k} dist={dist:.3f} | tau={tau:.2f} eps={eps:.3f}")
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
    cfg = TrainConfig(steps=args.steps, input_dim=args.input_dim,
                      model_dim=args.model_dim, core_rank=args.core_rank)
    run_basic(cfg)