"""Learned LoS-blockage predictor (TVT T5 baseline, in the spirit of RaDaR: radar-based blockage prediction).

Inputs = the same information as the proposed planner: confirmed radar track posteriors of O-RU 0
(mean, covariance, map mode) and the UE position estimate with its velocity, per 0.1 s report, UE
and cell. Features per (report, UE, cell): LoS segment O-RU -> UE (length, elevation); for the
N_NEAR tracks with the smallest predicted time to cross the segment's vertical plane (constant
velocity): along-segment coordinate of the crossing [0..1], cross distance now, normal speed,
time to cross (clipped to [-1, 4] s), segment height at the crossing point minus the track's box
height (paper-1 size rule), lane/sidewalk/free one-hot, sqrt of the position-covariance trace;
missing tracks are zero-padded with an indicator. Output: P(LoS model-B loss >= 10 dB) at each
look-ahead of TAUS (multi-label, BCE). Model: MLP 2 x 128 (torch), trained on the TRAINING seeds
only; early stopping on a held-back 10 % of the training jobs. ``predict_loss`` converts the
probabilities into the predicted-loss tensor of the paper-1 trigger/planner code (200 dB if
p >= threshold else 0, linear in tau between the trained look-aheads).
"""

from __future__ import annotations

import math

import numpy as np

TAUS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
N_NEAR = 4
N_FEAT = 3 + N_NEAR * 9


def features(oru: np.ndarray, ue: np.ndarray, ue_vel: np.ndarray, tracks: dict, r: int) -> np.ndarray:
    """Features [U, C, N_FEAT] at report r. oru [C, 3]; ue, ue_vel [U, 3]; tracks from sim.tvt.tracks.pack_posteriors."""
    U, C = ue.shape[0], oru.shape[0]
    out = np.zeros((U, C, N_FEAT))
    valid = tracks["valid"][r]
    mean = tracks["mean"][r][valid]
    cov = tracks["cov"][r][valid]
    size = tracks["size"][r][valid]
    kind = tracks["kind"][r][valid] if "kind" in tracks else np.zeros(int(valid.sum()), int)
    for u in range(U):
        for c in range(C):
            a, b = oru[c], ue[u]
            seg = b - a
            L = float(np.linalg.norm(seg[:2]))
            out[u, c, 0] = L / 50.0
            out[u, c, 1] = (a[2] - b[2]) / max(L, 1e-6)
            if mean.shape[0] == 0:
                out[u, c, 2] = 1.0
                continue
            dirh = seg[:2] / max(L, 1e-6)
            nrm = np.array([-dirh[1], dirh[0]])
            rel = mean[:, :2] - a[:2]
            s = rel @ dirh / max(L, 1e-6)
            dist = rel @ nrm
            # relative normal speed (the UE moves too)
            vn = mean[:, 3:5] @ nrm - ue_vel[u, :2] @ nrm * np.clip(s, 0, 1)
            ttc = np.where(np.abs(vn) > 1e-3, -dist / np.where(np.abs(vn) > 1e-3, vn, 1.0), np.where(np.abs(dist) < 1.0, 0.0, 4.0))
            ttc = np.clip(ttc, -1.0, 4.0)
            s_cross = np.clip(s + (mean[:, 3:5] @ dirh) * np.clip(ttc, 0, 4) / max(L, 1e-6), -0.5, 1.5)
            z_seg = a[2] + (b[2] - a[2]) * np.clip(s_cross, 0, 1)
            order = np.argsort(np.where(ttc >= 0, ttc, 10.0 - ttc))[:N_NEAR]
            for k, i in enumerate(order):
                f = 3 + 9 * k
                out[u, c, f:f + 9] = [s_cross[i], dist[i] / 10.0, vn[i] / 10.0, ttc[i] / 4.0, (z_seg[i] - size[i, 2]) / 5.0,
                                      float(kind[i] == 0), float(kind[i] == 1), math.sqrt(max(np.trace(cov[i][:2, :2]), 0.0)), 1.0]
    return out


def labels(los_loss_db: np.ndarray, r: int, U: int, C: int) -> np.ndarray:
    """Targets [U, C, len(TAUS)] from the 10 ms LoS loss [T, U, C] (report r at step 10 r)."""
    k = 10 * r + np.round(TAUS / 0.01).astype(int)
    k = np.minimum(k, los_loss_db.shape[0] - 1)
    return (los_loss_db[k] >= 10.0).transpose(1, 2, 0).astype(np.float32)


class MLP:
    """Small torch MLP wrapper (fit / predict_proba)."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 128, seed: int = 0):
        import torch

        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, n_out))
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def fit(self, X, Y, Xv, Yv, epochs: int = 60, lr: float = 1e-3, batch: int = 4096, device: str = "cuda") -> dict:
        import torch

        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        net = self.net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xt = torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32, device=device)
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
        Xvt = torch.as_tensor((Xv - self.mu) / self.sd, dtype=torch.float32, device=device)
        Yvt = torch.as_tensor(Yv, dtype=torch.float32, device=device)
        g = torch.Generator(device="cpu").manual_seed(1)
        best, best_state, hist = math.inf, None, []
        for ep in range(epochs):
            perm = torch.randperm(Xt.shape[0], generator=g).to(device)
            net.train()
            for lo in range(0, Xt.shape[0], batch):
                idx = perm[lo:lo + batch]
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), Yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(Xvt), Yvt))
            hist.append(vl)
            if vl < best - 1e-5:
                best = vl
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        self.net = net
        return {"val_bce": best, "history": hist}

    def predict_proba(self, X, device: str = "cuda") -> np.ndarray:
        import torch

        with torch.no_grad():
            Xt = torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32, device=device)
            return torch.sigmoid(self.net.to(device)(Xt)).cpu().numpy()


def predict_loss(prob: np.ndarray, taus_s: np.ndarray, threshold: float) -> np.ndarray:
    """Predicted-loss tensor [R, U, C, len(taus_s)] (200 dB = blocked) from probabilities [R, U, C, len(TAUS)]."""
    p = np.stack([np.interp(taus_s, TAUS, prob[..., :][i]) for i in np.ndindex(prob.shape[:-1])]).reshape(prob.shape[:-1] + (len(taus_s),))
    return np.where(p >= threshold, 200.0, 0.0)
