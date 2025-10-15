"""
asi_live_stream.py — lightweight realtime training streamer for ArtificialSentience

Fix: compatible with websockets ≥12 — use an async context manager inside the
threaded event loop (no more "no running event loop" on Windows/Python 3.10).

Usage (inside your training loop):

    from asi_live_stream import get_streamer, stream_step, stream_growth

    if step == 0:
        get_streamer().start()  # starts ws://0.0.0.0:8765

    stream_step(step=step, loss=float(recon.item()), ema_loss=float(pred_loss.item()),
                self_cons=float(self_cons.item()), cluster=int(k),
                num_clusters=model.num_clusters, tau=float(tau), eps=float(eps),
                dist=float(dist), emotion=[float(v) for v in model.emotion_head(h).detach().cpu().tolist()],
                fg_mean=float(mask_soft.mean().item()) if mask_soft is not None else None)

    # when a growth event happens
    stream_growth(cluster=int(k), old_rank=old_rank, new_rank=new_rank,
                  loss=float(second_avg), delta=float(improvement))

Run a local test stream with fake data:

    python asi_live_stream.py --demo

Then connect the React dashboard (ws://localhost:8765).
"""
from __future__ import annotations

import asyncio
import json
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

try:
    import websockets
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "\n[asi_live_stream] Missing dependency: websockets\n"
        "Install with:  pip install websockets\n"
        f"Original import error: {e}\n"
    )


# -----------------------------
# WebSocket server (producer → many consumers)
# -----------------------------

@dataclass
class _Config:
    host: str = "0.0.0.0"
    port: int = 8765
    max_queue: int = 4096


class LiveStreamer:
    """Threaded WebSocket broadcaster.

    - `.start()` spins up an event loop in a daemon thread
    - `.send(obj)` enqueues JSON-serializable dicts
    - multiple dashboard clients can connect concurrently
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, max_queue: int = 4096):
        self.cfg = _Config(host=host, port=port, max_queue=max_queue)
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=max_queue)
        self._clients: "set[Any]" = set()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            async for _ in ws:  # ignore incoming messages
                pass
        finally:
            self._clients.discard(ws)

    async def _pump(self):
        loop = asyncio.get_event_loop()
        while True:
            msg = await loop.run_in_executor(None, self._q.get)
            dead = []
            for c in list(self._clients):
                try:
                    await c.send(msg)
                except Exception:
                    dead.append(c)
            for d in dead:
                try:
                    self._clients.discard(d)
                except Exception:
                    pass

    async def _serve_forever(self):
        # websockets ≥12 expects a *running* loop at creation; so create inside coroutine
        async with websockets.serve(self._handler, self.cfg.host, self.cfg.port):
            # Start pump in this loop and keep it alive forever
            asyncio.create_task(self._pump())
            await asyncio.Future()  # run forever

    def start(self):
        if self._started:
            return self

        def _runner():
            # New event loop in this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve_forever())
            finally:  # pragma: no cover
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_runner, name="ASI-LiveStreamer", daemon=True)
        self._thread.start()
        # Give the server a tick to boot
        time.sleep(0.05)
        self._started = True
        return self

    def send(self, obj: Dict[str, Any]):
        try:
            self._q.put_nowait(json.dumps(obj))
        except queue.Full:
            # Drop if UI is too slow; training must not block.
            pass


# Global singleton for convenience
_streamer: Optional[LiveStreamer] = None

def get_streamer(host: str = "0.0.0.0", port: int = 8765) -> LiveStreamer:
    global _streamer
    if _streamer is None:
        _streamer = LiveStreamer(host=host, port=port)
    return _streamer


# -----------------------------
# Helper message APIs
# -----------------------------

def stream_step(*, step: int, loss: float, ema_loss: Optional[float], self_cons: Optional[float],
                cluster: int, num_clusters: int, tau: float, eps: float, dist: float,
                emotion: Optional[Iterable[float]] = None, fg_mean: Optional[float] = None,
                extras: Optional[Dict[str, Any]] = None) -> None:
    """Publish a single training step to the dashboard."""
    payload: Dict[str, Any] = {
        "type": "step",
        "step": int(step),
        "loss": float(loss),
        "ema_loss": None if ema_loss is None else float(ema_loss),
        "self_cons": None if self_cons is None else float(self_cons),
        "cluster": int(cluster),
        "num_clusters": int(num_clusters),
        "tau": float(tau),
        "eps": float(eps),
        "dist": float(dist),
        "emotion": list(emotion) if emotion is not None else None,
        "fg_mean": None if fg_mean is None else float(fg_mean),
    }
    if extras:
        payload.update(extras)
    get_streamer().send(payload)


def stream_growth(*, cluster: int, old_rank: int, new_rank: int, loss: float, delta: float,
                  extras: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "type": "growth",
        "cluster": int(cluster),
        "old_rank": int(old_rank),
        "new_rank": int(new_rank),
        "loss": float(loss),
        "delta": float(delta),
    }
    if extras:
        payload.update(extras)
    get_streamer().send(payload)


# -----------------------------
# Demo mode — emits synthetic data for UI testing
# -----------------------------

def _demo_stream(duration_sec: float = 30.0, hz: float = 10.0, num_clusters: int = 24):
    get_streamer().start()
    step = 0
    loss = 0.05
    ema = 0.048
    selfc = 0.005
    tau = 1.6
    eps = 0.15
    dist = 0.3
    fg = 0.2
    k = 0
    t0 = time.time()
    growth_every = 7

    while time.time() - t0 < duration_sec:
        step += 1
        # Wander values
        loss = max(0.001, loss + random.uniform(-0.002, 0.001))
        ema = max(0.001, ema + random.uniform(-0.002, 0.001))
        selfc = max(0.0, selfc + random.uniform(-0.0003, 0.0002))
        tau = max(0.2, tau + random.uniform(-0.02, 0.02))
        eps = max(0.0, eps + random.uniform(-0.01, 0.01))
        dist = min(1.0, max(0.0, dist + random.uniform(-0.02, 0.02)))
        fg = min(1.0, max(0.0, fg + random.uniform(-0.03, 0.03)))
        k = (k + random.choice([0, 1, 1, 2])) % num_clusters
        val = random.uniform(-1.0, 1.0)
        aro = random.uniform(-1.0, 1.0)
        ten = random.uniform(-1.0, 1.0)

        stream_step(
            step=step, loss=loss, ema_loss=ema, self_cons=selfc,
            cluster=k, num_clusters=num_clusters, tau=tau, eps=eps, dist=dist,
            emotion=[val, aro, ten], fg_mean=fg,
        )

        if step % growth_every == 0:
            old_r = random.randint(0, 3)
            new_r = old_r + 1
            stream_growth(cluster=k, old_rank=old_r, new_rank=new_r, loss=loss, delta=random.uniform(0.0, 0.01))

        time.sleep(1.0 / hz)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run demo data generator")
    ap.add_argument("--duration", type=float, default=60.0, help="demo duration seconds")
    ap.add_argument("--hz", type=float, default=8.0, help="messages per second")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    streamer = get_streamer(host=args.host, port=args.port)
    streamer.start()

    if args.demo:
        print(f"[asi_live_stream] demo running at ws://{args.host}:{args.port} for {args.duration}s …")
        _demo_stream(duration_sec=args.duration, hz=args.hz)
        print("[asi_live_stream] demo finished.")
    else:
        print(f"[asi_live_stream] server running at ws://{args.host}:{args.port} — waiting for training loop to send events…")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("[asi_live_stream] exiting…")
