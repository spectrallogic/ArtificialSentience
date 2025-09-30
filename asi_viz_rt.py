# asi_viz_rt.py
# Realtime 3D visualization for ASI latent space using PyQtGraph (OpenGL)
# - Shows streaming z vectors via a fixed random projection to 3D
# - Colors by routed cluster
# - Shows router centroids as larger spheres
# - Runs in a separate process; send updates via a multiprocessing.Queue

import sys
import time
import numpy as np
from multiprocessing import Process, Queue
from collections import deque

# PyQtGraph + Qt
from PyQt5 import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl


def _colormap_k(k, K):
    """Distinct color per cluster k in [0..K-1]."""
    # simple HSV wheel
    hue = (k / max(1, K)) % 1.0
    color = pg.hsvColor(hue, sat=1.0, val=1.0).getRgb()  # (r,g,b, a)
    return (color[0], color[1], color[2], 255)


class _VizWindow(gl.GLViewWidget):
    def __init__(self, model_dim, num_clusters, max_points=4000, point_size=3.0, centroid_size=12.0):
        super().__init__()
        self.setWindowTitle("ASI Realtime Latent Space")
        self.setCameraPosition(distance=80, elevation=18, azimuth=35)
        self.opts['center'] = pg.Vector(0, 0, 0)

        # Axes grid
        g = gl.GLGridItem()
        g.scale(10, 10, 1)
        g.setDepthValue(10)
        self.addItem(g)

        self.model_dim = model_dim
        self.num_clusters = num_clusters
        self.max_points = max_points
        self.point_size = point_size
        self.centroid_size = centroid_size

        # Fixed random projection R: D -> 3 (keeps it fast & stable)
        rng = np.random.default_rng(202)
        self.R = rng.normal(size=(model_dim, 3)).astype(np.float32)
        self.R /= np.linalg.norm(self.R, axis=0, keepdims=True).clip(1e-8)

        # Per-cluster point clouds (scatter)
        self.buffers = [deque(maxlen=max_points // num_clusters) for _ in range(num_clusters)]
        self.plots = []
        for k in range(num_clusters):
            color = _colormap_k(k, num_clusters)
            plt = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32),
                                       size=self.point_size,
                                       color=np.array([c/255.0 for c in color]),
                                       pxMode=True)  # pixel size, not world size
            self.addItem(plt)
            self.plots.append(plt)

        # Centroids (as little spheres made from many points)
        self.centroid_plots = []
        for k in range(num_clusters):
            color = _colormap_k(k, num_clusters)
            plt = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32),
                                       size=self.centroid_size,
                                       color=np.array([c/255.0 for c in color]),
                                       pxMode=True)
            self.addItem(plt)
            self.centroid_plots.append(plt)

        # Text overlay (cluster ranks etc.)
        self.text_items = []
        for k in range(num_clusters):
            ti = gl.GLTextItem(color=pg.mkColor(*_colormap_k(k, num_clusters)))
            ti.setData(pos=pg.Vector(0, 0, 0), text=f"C{k}: r=?")
            self.addItem(ti)
            self.text_items.append(ti)

        # World scale normalizer
        self.running_scale = 1.0

    def _project3(self, v):
        # v: (D,) -> (3,)
        return (v @ self.R).astype(np.float32)

    def _autoscale(self, pts3):
        # normalize overall scale for stable visuals
        if pts3.size == 0:
            return 1.0
        # robust scale ~ median L2
        norms = np.linalg.norm(pts3, axis=1)
        med = np.median(norms) if norms.size else 1.0
        target = 20.0
        if med > 1e-6:
            s = target / med
            # smooth
            self.running_scale = 0.95 * self.running_scale + 0.05 * s
        return self.running_scale

    def update_scene(self, z, k, centroids=None, ranks=None):
        """
        z: np.ndarray shape (D,)
        k: int cluster index
        centroids: optional np.ndarray (K, D)
        ranks: optional list/seq length K for residual ranks
        """
        # project and store point
        p = self._project3(z)
        self.buffers[k].append(p)

        # update point clouds
        all_pts = []
        for kk in range(self.num_clusters):
            pts = np.stack(self.buffers[kk], axis=0) if len(self.buffers[kk]) else np.zeros((0, 3), np.float32)
            self.plots[kk].setData(pos=pts)
            if pts.size:
                all_pts.append(pts)
        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
            s = self._autoscale(all_pts)
        else:
            s = self.running_scale

        # update centroids and labels
        if centroids is not None:
            for kk in range(self.num_clusters):
                c3 = self._project3(centroids[kk])
                c3s = c3 * s
                self.centroid_plots[kk].setData(pos=c3s.reshape(1, 3))
                # place text slightly above centroid
                txt = f"C{kk}"
                if ranks is not None and kk < len(ranks):
                    txt += f"  r={ranks[kk]}"
                self.text_items[kk].setData(pos=pg.Vector(*(c3s + np.array([0, 0, 2], dtype=np.float32))), text=txt)

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            QtGui.QApplication.instance().quit()
        else:
            super().keyPressEvent(ev)


def _viz_process(q: Queue, model_dim: int, num_clusters: int, max_points: int):
    app = QtGui.QApplication(sys.argv)
    win = _VizWindow(model_dim, num_clusters, max_points=max_points)
    win.show()

    timer = QtCore.QTimer()
    timer.setInterval(16)  # ~60 FPS
    timer.timeout.connect(lambda: None)
    timer.start()

    def poll_queue():
        try:
            # drain quickly to keep latest
            for _ in range(50):
                msg = q.get_nowait()
                if msg is None:
                    QtGui.QApplication.instance().quit()
                    return
                win.update_scene(**msg)
        except Exception:
            pass

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
        self.model_dim = model_dim
        self.num_clusters = num_clusters

    def send(self, z: np.ndarray, k: int, centroids: np.ndarray = None, ranks=None):
        if z is None:
            return
        try:
            payload = {
                "z": np.asarray(z, dtype=np.float32),
                "k": int(k),
                "centroids": np.asarray(centroids, dtype=np.float32) if centroids is not None else None,
                "ranks": list(ranks) if ranks is not None else None,
            }
            # drop oldest if full
            if self.q.full():
                try: self.q.get_nowait()
                except Exception: pass
            self.q.put_nowait(payload)
        except Exception:
            pass

    def close(self):
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        if self.p.is_alive():
            self.p.join(timeout=1.0)
