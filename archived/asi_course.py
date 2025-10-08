# asi_course.py
# A streaming "course" to evaluate the ASI Seed with realistic, procedural tasks.
# Levels: Discovery -> Prediction -> Concept Birth/Death -> Compositionality
#         -> Domain Swap -> Tool-like heads -> Stress/Rollback
# Saves plots (matplotlib) and a JSON report.

import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from asi_model import (ASISeed, CuriosityStream, to_device, ema_update_, orthogonalize_to_,
                       TrainConfig as BaseCfg)

plt.rcParams.update({"figure.figsize": (6, 4), "figure.dpi": 120})


@dataclass
class CourseCfg:
    steps: int = 5000
    report_path: str = "asi_course_report.json"
    out_dir: str = "asi_outputs"
    seed: int = 42


class ASICourse:
    def __init__(self, model: ASISeed, cfg: CourseCfg, base_cfg: BaseCfg):
        self.m = model
        self.cfg = cfg
        self.base_cfg = base_cfg
        self.device = torch.device(base_cfg.device)
        self.logs: Dict[str, List[float]] = {
            "loss": [], "ema_loss": [], "expansions": [], "cluster_k": [], "abstractness": []
        }
        self.level_marks: List[Dict[str, Any]] = []

    # --------- Helpers ---------
    def _log_step(self, loss, ema_loss, k):
        self.logs["loss"].append(loss)
        self.logs["ema_loss"].append(ema_loss)
        self.logs["cluster_k"].append(k)
        # abstractness index: ratio of baseline params to total (proxy)
        core = self.m.layer.U.numel() + self.m.layer.V.numel()
        tot = core
        for i in range(self.m.num_clusters):
            tot += self.m.layer.U_res[i].numel() + self.m.layer.V_res[i].numel()
        self.logs["abstractness"].append(core / max(1, tot))

        total_exp = sum(s.expansions for s in self.m.stats)
        self.logs["expansions"].append(total_exp)

    def _mark_level(self, name: str, start_step: int):
        self.level_marks.append({"name": name, "start_step": start_step})

    # --------- Levels ---------
    def level_discovery(self, steps: int, stream: CuriosityStream):
        self._mark_level("discovery", len(self.logs["loss"]))
        opt = torch.optim.Adam([p for p in self.m.parameters() if p.requires_grad], lr=self.base_cfg.lr)
        for _ in range(steps):
            x = to_device(stream.next()).to(self.device)
            x_hat, k, z, h = self.m(x)
            dist = (1 - F.cosine_similarity(z, self.m.router.centroids[k], dim=0)).item()
            loss = F.mse_loss(x_hat, x)
            if dist < self.base_cfg.novelty_dist_thresh:
                loss = 0.2 * loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.m.layer.V.grad is not None and self.m.layer.protected_basis_V.numel() > 0:
                self.m.layer.V.grad[:] = orthogonalize_to_(self.m.layer.V.grad, self.m.layer.protected_basis_V)
            opt.step()
            ema_update_(self.m.ema, self.m, beta=self.base_cfg.ema_beta)
            with torch.no_grad():
                self.m.router.update_centroid(k, z)
            self.m.update_buffers(k, z)

            gs = self.m.stats[k]
            gs.recent_losses.append(float(loss.detach().cpu())); gs.samples += 1
            if len(gs.recent_losses) > self.base_cfg.plateau_window:
                gs.recent_losses.pop(0)

            if len(gs.recent_losses) == self.base_cfg.plateau_window and (len(self.logs["loss"]) % self.base_cfg.grow_check_every == 0):
                first = sum(gs.recent_losses[: self.base_cfg.plateau_window // 2]) / (self.base_cfg.plateau_window // 2)
                last  = sum(gs.recent_losses[self.base_cfg.plateau_window // 2 :]) / (self.base_cfg.plateau_window // 2)
                if (first - last) < self.base_cfg.plateau_improve_eps:
                    # Use growth wrapper that rebuilds EMA
                    self.m.grow_cluster(k, grow_rank=self.base_cfg.grow_rank_step)
                    gs.expansions += 1
                    self.m.consolidate_cluster(k)
                    with torch.no_grad():
                        self.m.layer.protected_basis_V = torch.clone(self.m.layer.V.data.detach())

            # EMA canary
            with torch.no_grad():
                if self.m.ema is not None:
                    y, kk, *_ = self.m.ema(x)
                    ema_l = F.mse_loss(y, x).item()
                else:
                    ema_l = loss.item()
            self._log_step(loss.item(), ema_l, k)

    def level_prediction(self, steps: int, stream: CuriosityStream):
        self._mark_level("prediction", len(self.logs["loss"]))
        opt = torch.optim.Adam([p for p in self.m.parameters() if p.requires_grad], lr=self.base_cfg.lr)
        x_prev = to_device(stream.next()).to(self.device)
        for _ in range(steps):
            x = to_device(stream.next()).to(self.device)
            x_hat, k, z, h = self.m(x_prev)  # predict next from prev latent
            recon = F.mse_loss(x_hat, x_prev)
            pred  = self.m.prediction_loss(h, x)  # small supervised head
            loss = recon + 0.5 * pred

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.m.layer.V.grad is not None and self.m.layer.protected_basis_V.numel() > 0:
                self.m.layer.V.grad[:] = orthogonalize_to_(self.m.layer.V.grad, self.m.layer.protected_basis_V)
            opt.step()
            ema_update_(self.m.ema, self.m, beta=self.base_cfg.ema_beta)
            with torch.no_grad():
                self.m.router.update_centroid(k, z)
            self.m.update_buffers(k, z)

            with torch.no_grad():
                ema_l = F.mse_loss(self.m.ema(x_prev)[0], x_prev).item() if self.m.ema is not None else loss.item()
            self._log_step(loss.item(), ema_l, k)
            x_prev = x

    def level_birth_death(self, steps: int, stream: CuriosityStream):
        self._mark_level("birth_death", len(self.logs["loss"]))
        # Simulate birth/death by changing stream.num_sources mid-level
        original = stream.num_sources
        stream.num_sources = original + 2  # birth
        self.level_discovery(steps // 2, stream)
        stream.num_sources = max(2, original - 1)  # death
        self.level_discovery(steps // 2, stream)
        stream.num_sources = original

    def level_composition(self, steps: int, stream: CuriosityStream):
        self._mark_level("composition", len(self.logs["loss"]))
        # Increase interaction strength by boosting bases drift
        old_drift = stream.drift
        stream.drift = old_drift * 2.0
        self.level_discovery(steps, stream)
        stream.drift = old_drift

    def level_domain_swap(self, steps: int):
        self._mark_level("domain_swap", len(self.logs["loss"]))
        # Alternate two different streams
        s1 = CuriosityStream(input_dim=self.base_cfg.input_dim, num_sources=4, seed=123)
        s2 = CuriosityStream(input_dim=self.base_cfg.input_dim, num_sources=7, seed=456)
        half = steps // 2
        self.level_discovery(half, s1)
        self.level_discovery(steps - half, s2)

    def level_tools(self, steps: int, stream: CuriosityStream):
        self._mark_level("tools", len(self.logs["loss"]))
        opt = torch.optim.Adam([p for p in self.m.parameters() if p.requires_grad], lr=self.base_cfg.lr)
        for _ in range(steps):
            x = to_device(stream.next()).to(self.device)
            x_hat, k, z, h = self.m(x)
            recon = F.mse_loss(x_hat, x)
            mask  = self.m.masked_infill_loss(h, x, mask_ratio=0.2)
            loss  = recon + 0.2 * mask

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.m.layer.V.grad is not None and self.m.layer.protected_basis_V.numel() > 0:
                self.m.layer.V.grad[:] = orthogonalize_to_(self.m.layer.V.grad, self.m.layer.protected_basis_V)
            opt.step()
            ema_update_(self.m.ema, self.m, beta=self.base_cfg.ema_beta)
            with torch.no_grad():
                self.m.router.update_centroid(k, z)
            self.m.update_buffers(k, z)

            with torch.no_grad():
                ema_l = F.mse_loss(self.m.ema(x)[0], x).item() if self.m.ema is not None else loss.item()
            self._log_step(loss.item(), ema_l, k)

    def level_stress(self, steps: int, stream: CuriosityStream):
        self._mark_level("stress", len(self.logs["loss"]))
        # Abrupt stats spike: large noise for a bit, then rollback
        old_amp = stream.amp.clone()
        stream.amp *= 1.5
        self.level_discovery(steps // 2, stream)
        stream.amp = old_amp
        self.level_discovery(steps - steps // 2, stream)

    # --------- Run & Save ---------
    def run_course(self):
        s = CuriosityStream(input_dim=self.base_cfg.input_dim, num_sources=5, seed=777)
        self.level_discovery(1000, s)
        self.level_prediction(800, s)
        self.level_birth_death(800, s)
        self.level_composition(800, s)
        self.level_domain_swap(800)
        self.level_tools(600, s)
        self.level_stress(600, s)

    def save_report(self, path: str):
        report = {
            "levels": self.level_marks,
            "metrics": {
                "mean_loss": float(sum(self.logs["loss"]) / max(1, len(self.logs["loss"]))),
                "mean_ema_loss": float(sum(self.logs["ema_loss"]) / max(1, len(self.logs["ema_loss"]))),
                "final_expansions": int(self.logs["expansions"][-1]) if self.logs["expansions"] else 0,
                "final_abstractness": float(self.logs["abstractness"][-1]) if self.logs["abstractness"] else 1.0
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def save_plots(self, out_dir: str):
        import os
        os.makedirs(out_dir, exist_ok=True)
        # Loss
        plt.figure(); plt.plot(self.logs["loss"]); plt.title("Online Loss"); plt.xlabel("step"); plt.ylabel("MSE")
        for lm in self.level_marks: plt.axvline(lm["start_step"], ls="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_dir}/loss.png"); plt.close()
        # EMA
        plt.figure(); plt.plot(self.logs["ema_loss"]); plt.title("EMA Loss"); plt.xlabel("step"); plt.ylabel("MSE")
        for lm in self.level_marks: plt.axvline(lm["start_step"], ls="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_dir}/ema_loss.png"); plt.close()
        # Expansions
        plt.figure(); plt.plot(self.logs["expansions"]); plt.title("Cumulative Expansions"); plt.xlabel("step")
        for lm in self.level_marks: plt.axvline(lm["start_step"], ls="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_dir}/expansions.png"); plt.close()
        # Abstractness
        plt.figure(); plt.plot(self.logs["abstractness"]); plt.title("Abstractness Index (core/total params)")
        plt.xlabel("step")
        for lm in self.level_marks: plt.axvline(lm["start_step"], ls="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_dir}/abstractness.png"); plt.close()


def main():
    base = BaseCfg()
    cfg = CourseCfg()

    # Reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = ASISeed(input_dim=base.input_dim, model_dim=base.model_dim,
                    num_clusters=3, core_rank=base.core_rank, use_heads=True).to(base.device)
    course = ASICourse(model, cfg, base)
    course.run_course()
    course.save_report(cfg.report_path)
    course.save_plots(cfg.out_dir)
    print(f"Course complete. See {cfg.report_path} and ./{cfg.out_dir}/*.png")

if __name__ == "__main__":
    main()
