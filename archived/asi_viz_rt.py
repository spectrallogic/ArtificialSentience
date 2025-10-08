# asi_viz_rt.py
# Realtime 3D visualization for ASI latent space using PyQtGraph (OpenGL).
# Shows: streaming latent points colored by cluster + router centroids.
# Runs in a separate process; the training loop sends updates via a multiprocessing.Queue.

import sys
import numpy as np
from multiprocessing import Process, Queue

# PyQtGraph + Qt
from PyQt5 import QtCore, QtWidgets  # NOTE: QtWidgets.QApplication (not QtGui)
import pyqtgraph as pg
import pyqtgraph.opengl as gl


def _colormap_k(k, K):
    """Distinct color per cluster k in [0..K-1] as RGBA floats 0..1."""
    hue = (k / max(1, K)) % 1.0
    r, g, b, _ = pg.hsvColor(hue, sat=1.0, val=1.0).getRgb()
    return np.array([r/255.0, g/255.0, b/255.0, 1.0], dtype=np.float32)


class _VizWindow(gl.GLViewWidget):
    def __init__(self, model_dim, num_clusters, max_points=4000, point_size=3.0, centroid_size=12.0):
        super().__init__()
        self.setWindowTitle("ASI Realtime Latent Space")
        self.setCameraPosition(distance=80, elevation=18, azimuth=35)
        self.opts['center'] = pg.Vector(0, 0, 0)

        # Grid
        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)
        grid.setDepthValue(10)
        self.addItem(grid)

        self.model_dim = model_dim
        self.num_clusters = num_clusters
        self.max_points = max_points
        self.point_size = point_size
        self.centroid_size = centroid_size

        # Fixed random projection D->3 for speed and stability
        rng = np.random.default_rng(202)
        self.R = rng.normal(size=(model_dim, 3)).astype(np.float32)
        self.R /= np.linalg.norm(self.R, axis=0, keepdims=True).clip(1e-8)

        # Per-cluster scatter
        self._buffers = [np.zeros((0, 3), dtype=np.float32) for _ in range(num_clusters)]
        self._plots = []
        for k in range(num_clusters):
            plt = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32),
                                       size=self.point_size,
                                       color=_colormap_k(k, num_clusters),
                                       pxMode=True)
            self.addItem(plt)
            self._plots.append(plt)

        # Centroid markers
        self._centroid_plots = []
        for k in range(num_clusters):
            plt = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32),
                                       size=self.centroid_size,
                                       color=_colormap_k(k, num_clusters),
                                       pxMode=True)
            self.addItem(plt)
            self._centroid_plots.append(plt)

        self._running_scale = 1.0

    def _project3(self, v):
        # v: (D,) -> (3,)
        return (v @ self.R).astype(np.float32)

    def _autoscale(self, all_pts):
        if all_pts.size == 0:
            return self._running_scale
        norms = np.linalg.norm(all_pts, axis=1)
        med = np.median(norms) if norms.size else 1.0
        target = 20.0
        if med > 1e-6:
            s = target / med
            self._running_scale = 0.95 * self._running_scale + 0.05 * s
        return self._running_scale

    def update_scene(self, z, k, centroids=None, ranks=None):
        """
        z: np.ndarray shape (D,)
        k: int cluster index
        centroids: optional np.ndarray (K, D)
        ranks: ignored here (kept for API compatibility)
        """
        # Append new point into cluster buffer (use numpy roll if over limit)
        p = self._project3(z)
        buf = self._buffers[k]
        if buf.shape[0] >= self.max_points // self.num_clusters:
            buf = np.roll(buf, -1, axis=0)
            buf[-1] = p
        else:
            buf = np.concatenate([buf, p.reshape(1, 3)], axis=0)
        self._buffers[k] = buf

        # Update clusters
        all_pts = []
        for kk in range(self.num_clusters):
            pts = self._buffers[kk]
            self._plots[kk].setData(pos=pts)
            if pts.size:
                all_pts.append(pts)
        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
        else:
            all_pts = np.zeros((0, 3), dtype=np.float32)

        s = self._autoscale(all_pts)
        # Update centroids
        if centroids is not None:
            for kk in range(self.num_clusters):
                c3 = self._project3(centroids[kk]) * s
                self._centroid_plots[kk].setData(pos=c3.reshape(1, 3))

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            QtWidgets.QApplication.instance().quit()
        else:
            super().keyPressEvent(ev)


def _viz_process(q: Queue, model_dim: int, num_clusters: int, max_points: int):
    app = QtWidgets.QApplication(sys.argv)  # Fixed: QtWidgets, not QtGui
    win = _VizWindow(model_dim, num_clusters, max_points=max_points)
    win.show()

    # ~60 FPS tick (keeps UI responsive)
    tick = QtCore.QTimer()
    tick.setInterval(16)
    tick.timeout.connect(lambda: None)
    tick.start()

    def poll_queue():
        # Drain queue each tick; ignore malformed messages
        drained = 0
        while drained < 50:
            try:
                msg = q.get_nowait()
            except Exception:
                break
            if msg is None:
                QtWidgets.QApplication.instance().quit()
                return
            try:
                win.update_scene(**msg)
            except Exception:
                pass
            drained += 1

    poller = QtCore.QTimer()
    poller.setInterval(10)
    poller.timeout.connect(poll_queue)
    poller.start()

    sys.exit(app.exec_())


class RealtimeViz:
    """Public API used by the training loop (non-blocking)."""
    def __init__(self, model_dim: int, num_clusters: int, max_points: int = 4000):
        self.q = Queue(maxsize=2000)
        self.p = Process(target=_viz_process, args=(self.q, model_dim, num_clusters, max_points), daemon=True)
        self.p.start()

    def send(self, z: np.ndarray, k: int, centroids: np.ndarray = None, ranks=None):
        # Be robust if child crashed/closed
        if self.p is not None and not self.p.is_alive():
            return
        try:
            payload = {
                "z": np.asarray(z, dtype=np.float32),
                "k": int(k),
                "centroids": np.asarray(centroids, dtype=np.float32) if centroids is not None else None,
                "ranks": None if ranks is None else list(ranks),
            }
            if self.q.full():
                try:
                    _ = self.q.get_nowait()
                except Exception:
                    pass
            self.q.put_nowait(payload)
        except Exception:
            pass

    def close(self):
        try:
            if self.p is not None and self.p.is_alive():
                try:
                    self.q.put_nowait(None)
                except Exception:
                    pass
                self.p.join(timeout=1.0)
        except Exception:
            pass
