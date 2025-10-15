# baby_asi.py — Next‑Gen Baby ASI (Goals • Emotions • Foreground • Temporal)
# -----------------------------------------------------------------------------
# What’s new (high level):
# • Goals/Desires: lightweight GoalSystem with intrinsic drives (curiosity,
#   homeostasis/consistency, mastery), producing a goal bias + loss signal.
# • Emotions: secondary "affect" stream (valence, arousal, tension) that softly
#   steers routing/temporal dynamics (separate from main token/latent flow).
# • Parallel streams: perception (z→h), temporal flow (h_t), and affect run in
#   parallel and cross‑influence each other via gates.
# • Frequencies of consciousness: multi‑band oscillatory gates that modulate
#   attention/aggregation across fast/medium/slow traces.
# • Foreground/Background: a learned ForegroundHead + loss weighting utilities
#   so ARC‑style tasks measure/learn the transformation (not background).
# • Self‑Perception: the model “looks back” at its own output (re‑encode) and
#   optimizes a SelfConsistency loss so it can ‘see/hear’ itself.
# • Hungry Matrix 2.0: generalizes growth checks (plateaus) and consolidates
#   abstract bases; still one‑call: check_and_grow_cluster(k, loss, ...).
# • Temporal Consciousness (optional): integrates the separate module if found.
#
# Backwards‑compatible entry points kept:
#   - ASISeed.forward(x)  -> (x_hat, k, z, h)
#   - ASISeed.check_and_grow_cluster(...)
#   - ASISeed.recursive_solve(...)
# -----------------------------------------------------------------------------

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional realtime viz (safe to miss)
try:
    from archived.asi_viz_rt import RealtimeViz  # noqa: F401
except Exception:
    RealtimeViz = None

# Optional temporal consciousness module (ships as a separate file)
try:
    from temporal_consciousness import TemporalConsciousness  # noqa: F401
except Exception:
    TemporalConsciousness = None


# -----------------------------
# Utility
# -----------------------------

def to_device(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if torch.cuda.is_available():
        return x.cuda()
    return x


def ema_update_(target: Optional[nn.Module], source: nn.Module, beta: float = 0.999):
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
# Tiny Encoder / Decoder
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
# Router with exploration
# -----------------------------

class NearestCentroidRouter(nn.Module):
    def __init__(self, emb_dim: int, num_clusters: int, momentum: float = 0.02):
        super().__init__()
        self.num_clusters = num_clusters
        self.momentum = momentum
        centroids = torch.randn(num_clusters, emb_dim)
        for i in range(1, num_clusters):
            for j in range(i):
                sim = F.cosine_similarity(centroids[i], centroids[j], dim=0)
                if sim > 0.3:
                    centroids[i] = centroids[i] - 0.5 * sim * centroids[j]
        self.register_buffer("centroids", F.normalize(centroids, dim=1))

    @torch.no_grad()
    def update_centroid(self, k: int, z: torch.Tensor):
        z = F.normalize(z.detach(), dim=0)
        self.centroids[k] = F.normalize((1 - self.momentum) * self.centroids[k] + self.momentum * z, dim=0)

    def forward(self, z: torch.Tensor, tau: float = 1.0, eps: float = 0.0) -> int:
        sims = F.cosine_similarity(z.unsqueeze(0), self.centroids, dim=1)
        if self.training and torch.rand(1).item() < eps:
            return int(torch.randint(0, self.num_clusters, (1,)).item())
        if self.training and tau > 0:
            probs = F.softmax(sims / tau, dim=0)
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(sims).item())


# -----------------------------
# Elastic Low‑Rank + Residual per‑cluster
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
# Optional Heads (prediction, foreground mask, emotion)
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
    """
    Foreground estimator in [0,1].
    If unknown, it still produces a soft mask that our losses can use as
    importance weighting (emergent foreground learning).
    """
    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class EmotionHead(nn.Module):
    """Outputs (valence, arousal, tension). Range roughly [-1,1] via Tanh."""
    def __init__(self, model_dim: int, emo_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.Tanh(),
            nn.Linear(model_dim // 2, emo_dim),
            nn.Tanh(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -----------------------------
# Oath (alignment via fixed direction)
# -----------------------------

class OathModule(nn.Module):
    def __init__(self, model_dim: int, oath_dim: int = 8):
        super().__init__()
        self.projector = nn.Linear(model_dim, oath_dim, bias=False)
        g = torch.Generator(device='cpu')
        g.manual_seed(777)
        c = torch.randn(oath_dim, generator=g)
        c = F.normalize(c, dim=0)
        self.register_buffer("c_star", c)

    def oath_loss(self, z_batch: torch.Tensor, weight: float = 0.02) -> torch.Tensor:
        if z_batch.numel() == 0:
            return torch.tensor(0.0, device=z_batch.device)
        proj = F.normalize(self.projector(z_batch), dim=-1)
        target = self.c_star.unsqueeze(0).expand_as(proj)
        return weight * F.mse_loss(proj, target)


# -----------------------------
# Subconscious (memory prototypes + dreamlets)
# -----------------------------

class SubconsciousCore(nn.Module):
    def __init__(self, model_dim: int, k_mem: int = 8, n_dreams: int = 4):
        super().__init__()
        self.k_mem = k_mem
        self.n_dreams = n_dreams
        self.query = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Tanh())
        self.cand_proj = nn.Linear(model_dim, model_dim)
        self.score = nn.Linear(model_dim, 1)
        self.mix = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Tanh(), nn.Linear(model_dim, 1), nn.Sigmoid())

    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor, mem_bank: Optional[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        D = z_t.numel(); device = z_t.device
        cands = []
        if mem_bank is not None and mem_bank.numel() > 0:
            Z = F.normalize(mem_bank.to(device), dim=1)
            q = F.normalize(z_t.detach(), dim=0).unsqueeze(0)
            sims = (Z @ q.t()).squeeze(1)
            topk = torch.topk(sims, k=min(self.k_mem, Z.size(0)), largest=True)
            protos = mem_bank[topk.indices]
            cands.append(protos)
        for _ in range(self.n_dreams):
            noise = torch.randn(D, device=device)
            dream = F.normalize(0.7 * noise + 0.3 * z_t.detach(), dim=0)
            cands.append(dream.unsqueeze(0))
        if len(cands) == 0:
            return torch.zeros(D, device=device), {"n_cands": 0}
        C = torch.cat(cands, dim=0)
        ctx = torch.cat([z_t, h_t], dim=0)
        q_vec = self.query(ctx)
        logits = self.score(self.cand_proj(C) * q_vec.unsqueeze(0)).squeeze(1)
        attn = F.softmax(logits, dim=0)
        s_raw = (attn.unsqueeze(1) * C).sum(dim=0)
        s_mixed = self.mix(s_raw)
        g = self.gate(ctx)
        s_t = g * s_mixed
        return s_t, {"n_cands": C.size(0), "gate": float(g.item())}


# -----------------------------
# Temporal Core with multi‑band gates (frequencies of consciousness)
# -----------------------------

class TemporalCore(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.register_buffer("trace_fast", torch.zeros(model_dim))
        self.register_buffer("trace_slow", torch.zeros(model_dim))
        self.gru = nn.GRUCell(model_dim, model_dim)
        self.bias_proj = nn.Linear(model_dim, model_dim)
        self.post_ln = nn.LayerNorm(model_dim)
        # Multi‑band oscillatory gates (fast/medium/slow)
        self.band_freqs = torch.tensor([3.0, 1.0, 0.3])  # arbitrary units/steps
        self.register_buffer("t", torch.zeros(1))
        self.band_mix = nn.Linear(model_dim * 3, model_dim)

    def _update_traces(self, z: torch.Tensor):
        alpha_fast = 0.10
        alpha_slow = 0.01
        self.trace_fast = (1 - alpha_fast) * self.trace_fast + alpha_fast * z.detach()
        self.trace_slow = (1 - alpha_slow) * self.trace_slow + alpha_slow * z.detach()

    def _multi_band(self, z: torch.Tensor) -> torch.Tensor:
        # Simple sin gates over time index t
        self.t += 1
        phase = self.t.item()
        s_fast = torch.sin(torch.tensor(phase) * self.band_freqs[0]).to(z.device)
        s_med  = torch.sin(torch.tensor(phase) * self.band_freqs[1]).to(z.device)
        s_slow = torch.sin(torch.tensor(phase) * self.band_freqs[2]).to(z.device)
        # Broadcast to vector dims via scaling
        mix = torch.cat([z * s_fast, z * s_med, z * s_slow], dim=0)
        return self.band_mix(mix)

    def _make_context(self, z: torch.Tensor) -> torch.Tensor:
        base = z + 0.3 * self.trace_fast + 0.1 * self.trace_slow
        return base + 0.1 * torch.tanh(self._multi_band(base))

    def forward(self, z: torch.Tensor, h: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._update_traces(z)
        ctx = self._make_context(z)
        if bias is not None:
            ctx = ctx + torch.tanh(self.bias_proj(bias))
        dh = self.gru(ctx.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        h_next = self.post_ln(h + dh)
        return h_next


# -----------------------------
# Goal System (intrinsic drives)
# -----------------------------

class GoalSystem(nn.Module):
    """
    Minimal intrinsic drives that create tension/desire and a goal bias:
      • Curiosity: prefer prediction errors slightly above comfort band
      • Consistency/Homeostasis: avoid wildly fluctuating representations
      • Mastery: reduce error on recurring patterns (buffers)
    Produces: goal_bias (R^D) and a scalar goal_loss for training.
    """
    def __init__(self, model_dim: int, comfort_low=0.01, comfort_high=0.08):
        super().__init__()
        self.model_dim = model_dim
        self.target_band = (comfort_low, comfort_high)
        self.err_proj = nn.Linear(3, model_dim)
        self.homeo_ln = nn.LayerNorm(model_dim)

    def forward(self, pred_err: float, rep_var: float, mastery_err: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Curiosity: penalize errors too low (boredom) or too high (overwhelm)
        lo, hi = self.target_band
        band_loss = (max(0.0, lo - pred_err) + max(0.0, pred_err - hi))
        # Consistency: large rep variance implies agitation
        consistency_loss = rep_var
        # Mastery: want mastery_err low
        mastery_loss = mastery_err
        # Aggregate and project to a bias vector
        triple = torch.tensor([pred_err, rep_var, mastery_err], dtype=torch.float32)
        goal_bias = torch.tanh(self.homeo_ln(self.err_proj(triple)))
        goal_loss = 0.1 * band_loss + 0.05 * consistency_loss + 0.1 * mastery_loss
        return goal_bias, torch.tensor(goal_loss, dtype=torch.float32)


# -----------------------------
# ASI Seed
# -----------------------------

@dataclass
class GrowthStats:
    recent_losses: List[float] = field(default_factory=list)
    expansions: int = 0
    samples: int = 0


class ASISeed(nn.Module):
    def __init__(self, input_dim=32, model_dim=192, num_clusters=24, core_rank=4,
                 build_ema: bool = True, use_heads: bool = True,
                 use_temporal_consciousness: bool = False):
        super().__init__()
        self.encoder = TinyEncoder(input_dim, model_dim)
        self.layer = ElasticLowRankLayer(model_dim, model_dim, rank=core_rank, num_clusters=num_clusters, phi=F.relu)
        self.decoder = TinyDecoder(model_dim, input_dim)
        self.router = NearestCentroidRouter(model_dim, num_clusters=num_clusters, momentum=0.02)
        self.oath = OathModule(model_dim, oath_dim=8)
        self.subconscious = SubconsciousCore(model_dim=model_dim)
        self.goal_system = GoalSystem(model_dim=model_dim)
        self.emotion_head = EmotionHead(model_dim=model_dim)
        self.temporal = TemporalCore(model_dim=model_dim)
        self.temporal_consciousness = None
        if use_temporal_consciousness and TemporalConsciousness is not None:
            self.temporal_consciousness = TemporalConsciousness(model_dim, window_size=7)
        self.use_heads = use_heads
        if use_heads:
            self.predict_head = PredictHead(model_dim, input_dim)
            self.mask_head = MaskHead(model_dim, input_dim)
        self.num_clusters = num_clusters
        self.stats: List[GrowthStats] = [GrowthStats() for _ in range(num_clusters)]
        self.buffers: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        self.max_buffer = 1024
        self._hparams = {
            "input_dim": input_dim, "model_dim": model_dim, "num_clusters": num_clusters,
            "core_rank": core_rank, "use_heads": use_heads,
        }
        self._ema: Optional[ASISeed] = None
        if build_ema:
            self._ema = ASISeed._make_ema(self, input_dim, model_dim, num_clusters, core_rank, use_heads)

    # --- EMA helpers ---
    @staticmethod
    def _make_ema(source, input_dim, model_dim, num_clusters, core_rank, use_heads):
        ema_copy = ASISeed(input_dim, model_dim, num_clusters, core_rank, build_ema=False, use_heads=use_heads)
        ema_copy.load_state_dict(source.state_dict(), strict=False)
        for p in ema_copy.parameters():
            p.requires_grad = False
        ema_copy.eval(); return ema_copy

    def rebuild_ema(self):
        if self._ema is None: return
        hp = self._hparams
        new_ema = ASISeed._make_ema(self, hp["input_dim"], hp["model_dim"], hp["num_clusters"], hp["core_rank"], hp["use_heads"])
        ema_update_(new_ema, self, beta=0.0)
        self._ema = new_ema

    @property
    def ema(self):
        return self._ema

    # --- Core forward ---
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        k = self.router(z)
        h = self.layer(z, active_cluster=k)
        x_hat = self.decoder(h)
        return x_hat, k, z, h

    # --- Memory buffers ---
    def update_buffers(self, k: int, z: torch.Tensor):
        with torch.no_grad():
            z_cpu = z.detach().cpu()
            self.buffers[k].append(z_cpu)
            if len(self.buffers[k]) > self.max_buffer:
                self.buffers[k].pop(0)

    # --- Growth & consolidation ---
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

    # --- Heads & losses ---
    def prediction_loss(self, h: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        if not self.use_heads:
            return torch.tensor(0.0, device=h.device)
        pred = self.predict_head(h)
        return F.mse_loss(pred, x_target)

    def foreground_mask(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_heads:
            return None
        return self.mask_head(h)

    @staticmethod
    def weighted_mse(pred, target, mask_soft=None, fg_weight: float = 5.0):
        if mask_soft is None:
            return F.mse_loss(pred, target)
        # Emphasize foreground; keep background but softer
        w = 1.0 + (fg_weight - 1.0) * mask_soft
        return (w * (pred - target) ** 2).mean()

    def self_consistency_loss(self, x_out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        # Re‑encode own output and compare latent alignment to input latent
        z_out = self.encoder(x_out.detach())
        z_in = self.encoder(x_in.detach())
        return 0.05 * (1 - F.cosine_similarity(z_out, z_in, dim=0))

    # --- Goals & emotions ---
    def affect_and_goals(self, h: torch.Tensor, pred_err: float, rep_var: float, mastery_err: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Emotions from current latent
        emotion_vec = self.emotion_head(h)
        # Goal bias + scalar loss
        goal_bias, goal_loss = self.goal_system(pred_err=pred_err, rep_var=rep_var, mastery_err=mastery_err)
        # Mix (affect gates goals)
        affect_gate = torch.sigmoid(emotion_vec.mean())
        mixed_bias = affect_gate * goal_bias + (1 - affect_gate) * 0.0
        return emotion_vec, mixed_bias, goal_loss

    # --- Growth trigger ---
    def check_and_grow_cluster(self, k: int, loss: float,
                               window_size: int = 50,
                               improvement_threshold: float = 0.01,
                               grow_rank: int = 2,
                               verbose: bool = True) -> bool:
        stats = self.stats[k]
        stats.recent_losses.append(loss); stats.samples += 1
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
            stats.expansions += 1; stats.recent_losses.clear()
            if verbose:
                print(f"🌱 GROWTH! Cluster {k}: rank {old_rank} → {new_rank} (loss={second_avg:.4f}, Δ={improvement:.4f})")
            return True
        return False

    # --- Recursive refinement (unchanged API, slight improvements inside) ---
    def refine_answer(self, x_input: torch.Tensor, x_current: torch.Tensor,
                      tau: float = 0.5, eps: float = 0.0) -> Tuple[torch.Tensor, int, torch.Tensor]:
        combined = x_input + 0.3 * x_current
        z = self.encoder(combined)
        k = self.router(z, tau=tau, eps=eps)
        h = self.layer(z, active_cluster=k)
        x_refined = self.decoder(h)
        return x_refined, k, h

    def evaluate_improvement(self, x_old: torch.Tensor, x_new: torch.Tensor, x_target: torch.Tensor) -> Tuple[float, float, bool]:
        old_error = F.mse_loss(x_old, x_target).item()
        new_error = F.mse_loss(x_new, x_target).item()
        return old_error, new_error, (old_error - new_error) > 0.0001

    def recursive_solve(self, x_input: torch.Tensor, x_target: torch.Tensor,
                        max_iterations: int = 20, tau: float = 1.0, eps: float = 0.1,
                        verbose: bool = False) -> Tuple[torch.Tensor, Dict]:
        # Initial attempt
        z = self.encoder(x_input)
        k = self.router(z, tau=tau, eps=eps)
        h = self.layer(z, active_cluster=k)
        x_current = self.decoder(h)
        attempts_detached = [x_current.detach()]
        attempts_live = [x_current]
        errors = [F.mse_loss(x_current, x_target).item()]
        clusters_used = [k]
        improvements = []
        if verbose:
            print(f"  Initial: error={errors[0]:.5f}, cluster={k}")
        for iteration in range(max_iterations):
            x_refined, k_used, h_refined = self.refine_answer(x_input, x_current.detach(), tau=tau * 0.9, eps=eps * 0.9)
            old_err, new_err, improved = self.evaluate_improvement(x_current, x_refined, x_target)
            attempts_detached.append(x_refined.detach())
            attempts_live.append(x_refined)
            errors.append(new_err)
            clusters_used.append(k_used)
            improvements.append(new_err - old_err)
            if verbose:
                print(f"  Iter {iteration + 1}: error={new_err:.5f}, cluster={k_used}, d={improvements[-1]:+.5f} {'✓' if improved else '✗'}")
            x_current = x_refined
            if new_err < 0.0001:
                if verbose: print(f"  → Converged at iteration {iteration + 1}")
                break
        is_stuck = (len(improvements) >= 5 and sum(improvements[-5:]) > -0.0005 and errors[-1] > 0.005)
        info = {
            'attempts': attempts_detached,
            'errors': errors,
            'clusters_used': clusters_used,
            'improvements': improvements,
            'iterations': len(attempts_detached) - 1,
            'final_error': errors[-1],
            'is_stuck': is_stuck,
            'best_iteration': int(np.argmin(errors))
        }
        best_idx = info['best_iteration']
        best_answer = attempts_live[best_idx]
        return best_answer, info


# -----------------------------
# Curiosity stream (toy self‑discovery)
# -----------------------------

class CuriosityStream:
    def __init__(self, input_dim: int = 32, num_sources: int = 3, drift: float = 0.005, seed: int = 42):
        self.input_dim = input_dim; self.num_sources = num_sources; self.drift = drift
        self.rng = np.random.RandomState(seed)
        self.bases = self.rng.randn(num_sources, input_dim).astype(np.float32)
        for i in range(num_sources):
            self.bases[i] /= (np.linalg.norm(self.bases[i]) + 1e-8)
        self.weights = self.rng.rand(num_sources).astype(np.float32)
        self.weights /= self.weights.sum()

    def next(self) -> np.ndarray:
        for i in range(self.num_sources):
            noise = self.rng.randn(self.input_dim).astype(np.float32) * self.drift
            self.bases[i] += noise
            self.bases[i] /= (np.linalg.norm(self.bases[i]) + 1e-8)
        w_noise = self.rng.randn(self.num_sources).astype(np.float32) * self.drift
        self.weights += w_noise; self.weights = np.abs(self.weights)
        self.weights /= (self.weights.sum() + 1e-8)
        x = np.zeros(self.input_dim, dtype=np.float32)
        for i in range(self.num_sources):
            x += self.weights[i] * self.bases[i]
        return x


# -----------------------------
# Training demo (self‑discovery + affect/goals/foreground/self‑perception)
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
    device = torch.device(cfg.device)
    model = ASISeed(
        input_dim=cfg.input_dim,
        model_dim=cfg.model_dim,
        num_clusters=cfg.num_clusters,
        core_rank=cfg.core_rank,
        build_ema=True,
        use_heads=True,
        use_temporal_consciousness=False,
    ).to(device)
    stream = CuriosityStream(input_dim=cfg.input_dim, num_sources=5, seed=42)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    try:
        for step in range(cfg.steps):
            x = to_device(stream.next()).to(device)
            z = model.encoder(x)
            progress = step / cfg.steps
            tau = 2.0 - 1.4 * progress
            eps = 0.2 - 0.18 * progress
            k = model.router(z, tau=tau, eps=eps)
            # Parallel streams: latent path + temporal path
            h_latent = model.layer(z, active_cluster=k)
            h_temporal = model.temporal(z, torch.zeros_like(h_latent))
            # Combine with small gate
            mix_gate = torch.sigmoid(torch.dot(h_latent, h_temporal) / (h_latent.norm() * h_temporal.norm() + 1e-6))
            h = mix_gate * h_latent + (1 - mix_gate) * h_temporal
            x_hat = model.decoder(h)
            # Heads
            pred = model.predict_head(h) if model.use_heads else None
            mask_soft = model.mask_head(h) if model.use_heads else None
            # Losses
            recon = ASISeed.weighted_mse(x_hat, x, mask_soft, fg_weight=5.0)
            pred_loss = F.mse_loss(pred, x) if pred is not None else torch.tensor(0.0, device=device)
            self_cons = model.self_consistency_loss(x_hat, x)
            # Dist to centroid → curiosity signal proxy
            with torch.no_grad():
                dist = 1 - F.cosine_similarity(z, model.router.centroids[k], dim=0)
            # Representation variance proxy (over batch=1, use latent magnitude)
            rep_var = float(h.var().detach().item())
            # Mastery proxy: EMA error via EMA model (if present)
            if model.ema is not None:
                with torch.no_grad():
                    y_ema, kk, _z, _h = model.ema(x)
                    mastery_err = float(F.mse_loss(y_ema, x).detach().item())
            else:
                mastery_err = float(recon.detach().item())
            emotion_vec, goal_bias, goal_loss = model.affect_and_goals(h, pred_err=float(recon.detach().item()), rep_var=rep_var, mastery_err=mastery_err)
            # Nudge temporal context with goals/affect next step (teacher‑forcing style)
            h = h + 0.05 * goal_bias
            # Alignment loss (soft)
            oath = model.oath.oath_loss(z.unsqueeze(0))
            # Total
            loss = recon + 0.3 * pred_loss + self_cons + oath + goal_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if model.layer.V.grad is not None and model.layer.protected_basis_V.numel() > 0:
                model.layer.V.grad[:] = orthogonalize_to_(model.layer.V.grad, model.layer.protected_basis_V)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update_(model._ema, model, beta=cfg.ema_beta)
            with torch.no_grad():
                model.router.update_centroid(k, z.detach())
            model.update_buffers(k, z.detach())
            if step % cfg.grow_check_every == 0:
                if model.check_and_grow_cluster(
                        k, loss.item(), window_size=cfg.plateau_window,
                        improvement_threshold=cfg.plateau_improve_eps,
                        grow_rank=cfg.grow_rank_step):
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
                    print(f"[step {step}] EMA MSE={sum(losses)/len(losses):.4f} | k={k} dist={dist:.3f} | tau={tau:.2f} eps={eps:.3f}")
    finally:
        pass
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
