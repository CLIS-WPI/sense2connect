"""Blockage-aware map-aided multipath tracker (TVT T4, main method).

State s = [p_x, p_y, v_x, v_y, b, delta_0, delta_1, d_1..d_F]: UE horizontal position and
velocity (constant-velocity model, white acceleration q), UE clock bias b (drawn anew every epoch
in the simulation -> re-initialised with variance sigma_b^2 at every prediction), O-RU time
offsets delta_c (constant, prior sigma_sync) and the offsets d_f of the F map faces along their
normals (constant, prior sigma_map; the tracker's map may be wrong by an unknown offset).
Sources per O-RU (sim/tvt/sources.py): LoS and single/double-bounce virtual-anchor chains of the
map that are geometrically valid at the predicted position. Measurements: multipath components
(sim/tvt/extract.py) z = (tau [ns], u_y, u_z) with their local-Fisher variances plus floors.
Visibility q of every source (sim/tvt/visibility.py or an oracle / constant prior) sets three
hypotheses per detection: visible with the nominal covariance R (weight q (1 - eps) P_D,clear),
visible but corrupted (merged with an unresolved path, T2 Finding 2; q eps P_D,clear, R + R_blk) and
blocked ((1 - q) P_D,blk; a blocked LoS arrives as the diffracted detour, R + R_blk, a blocked reflection
is attenuated but keeps its geometry, R - the signal model of the synthesis, paper-2 variant b).
Development iteration v1 (results/TVT/T4/v1) used two modes (clear R / blocked-or-detour R + R_blk for
every source): with sharp (oracle) visibility the tracker discarded exact blocked reflections and
lost track during blockage (p90 2.5 m), so the hypotheses were separated (this version).
Association: loopy BP (sim/tvt/bp.py) over sources and measurements with Poisson clutter of
density lambda_c per O-RU; update: sequential PDA-EKF over the sources (hypotheses: missed, or
measurement j in the clear / blocked mode), moment-matched; then the walkable-area constraint
(projection onto the sidewalk bands of the digital twin). Initialisation and re-initialisation
(after ``reinit_after`` epochs without any association, a large normalised LoS innovation, a
persistent inconsistency with the global search, or at the edge of the street section) by a grid search over the walkable area on the LoS angles and the LoS TDoA of the measured
components (both O-RUs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

IX, IY, IVX, IVY, IB = 0, 1, 2, 3, 4
ID0 = 5  # delta_0, delta_1 at 5, 6; faces from 7


@dataclass
class TrackerParams:
    q_acc: float = 0.5                 # white acceleration PSD [m^2/s^3]
    sigma_b_ns: float = 30.0           # clock-bias prior per epoch (U(-50, 50) ns has std 28.9)
    sigma_sync_ns: float = 1.0
    sigma_map_m: float = 0.0
    floor_tau_ns: float = 0.3
    floor_u: float = 0.005
    blk_tau_ns: float = 3.0
    blk_u: float = 0.05
    pd_los: float = 0.95
    pd_nlos: float = 0.7
    pd_blk: float = 0.5
    eps_robust: float = 0.2            # share of a VISIBLE source's detections with a corrupted (merged / biased) component
    clutter_per_snapshot: float = 2.0
    clutter_volume: float = 2000.0     # ns x (u_y, u_z) area: 500 ns x 4
    gate: float = 25.0                 # chi-square (3 dof) gate on the blocked-mode innovation
    reinit_after: int = 10
    init_sigma_m: float = 0.5
    init_sigma_v: float = 1.0
    nis_reinit: float = math.inf       # mean LoS NIS over the last reinit_after epochs (off: merged/diffracted LoS components
    #                                    inflate the NIS and a restart from the LoS-only grid search is worse; development smoke test)
    search_margin: float = 300.0       # re-initialise when the grid-search cost at the estimate exceeds the grid minimum by this ...
    search_epochs: int = 10            # ... for this many consecutive epochs
    edge_epochs: int = 3               # re-initialise after this many epochs at the x-edge of the street section (UE left it)


@dataclass
class TrackState:
    x: np.ndarray
    P: np.ndarray
    started: bool = False
    bad: int = 0
    off: int = 0
    edge: int = 0
    nis_hist: list = field(default_factory=list)


def wrap(d: np.ndarray, period: float) -> np.ndarray:
    return (d + 0.5 * period) % period - 0.5 * period


class MultipathTracker:
    """One tracker per UE and run; geometry in metres, delays in ns."""

    def __init__(self, faces: list[dict], oru: np.ndarray, period_ns: float, walk: dict, params: TrackerParams, map_error: np.ndarray | None = None,
                 z_ue: float = 1.5):
        from sim.tvt.sources import source_list, source_planes

        self.faces = faces
        self.F = len(faces)
        self.n = ID0 + 2 + self.F
        self.oru = oru  # [C, 3]
        self.C = oru.shape[0]
        self.period = period_ns
        self.walk = walk
        self.p = params
        self.z = z_ue
        self.sources = source_list(self.F)
        self.map_error = np.zeros(self.F) if map_error is None else map_error
        # the tracker's (possibly wrong) map: faces moved by map_error along their normals
        self.N, self.A, self.V, self.fid = source_planes(faces, self.sources, self.map_error)
        self.st = None

    # ------------------------------------------------------------------ helpers
    def _planes(self, x: np.ndarray):
        """Source planes with the CURRENT face-offset estimate d applied on top of the tracker's map."""
        d = x[ID0 + 2:]
        A = self.A.copy()
        for k in range(2):
            f = self.fid[:, k]
            m = f >= 0
            A[m, k] += (d[f[m]])[:, None] * self.N[m, k]
        return self.N, A, self.V

    def valid_sources(self, x: np.ndarray) -> list[np.ndarray]:
        from sim.tvt.sources import validity

        N, A, V = self._planes(x)
        ue = np.array([x[IX], x[IY], self.z])
        return [np.flatnonzero(validity(self.oru[c], ue, self.faces, self.sources, N, A, V, self.fid)) for c in range(self.C)]

    def predict_meas(self, x: np.ndarray, c: int, idx: np.ndarray):
        """h [S, 3], H [S, 3, n] for sources idx of O-RU c at state x."""
        from sim.tvt.sources import predict

        N, A, V = self._planes(x)
        ue = np.array([x[IX], x[IY], self.z])
        pr = predict(self.oru[c], ue, N[idx], A[idx], V[idx])
        self._pred_sx = np.where(pr["u"][:, 0] >= 0.0, 1, -1)  # half-space of each source's departure (back-to-back panels)
        S = len(idx)
        h = np.stack([pr["tau"] + x[ID0 + c] + x[IB], pr["u"][:, 1], pr["u"][:, 2]], -1)
        H = np.zeros((S, 3, self.n))
        H[:, :, IX:IY + 1] = pr["Jp"]
        H[:, 0, IB] = 1.0
        H[:, 0, ID0 + c] = 1.0
        for k in range(2):
            f = self.fid[idx, k]
            m = f >= 0
            H[np.flatnonzero(m), :, ID0 + 2 + f[m]] += pr["Jd"][m][:, :, k]
        return h, H

    # ------------------------------------------------------------------ init
    def _grid(self) -> np.ndarray:
        if getattr(self, "_g", None) is None:
            w = self.walk
            if "rects" in w:  # general walkable area: union of axis-aligned rectangles (x0, x1, y0, y1)
                pts = []
                for x0, x1, y0, y1 in w["rects"]:
                    X, Y = np.meshgrid(np.arange(x0, x1 + 1e-9, 0.25), np.arange(y0, y1 + 1e-9, 0.25), indexing="ij")
                    pts.append(np.stack([X.ravel(), Y.ravel()], -1))
                g = np.unique(np.round(np.concatenate(pts), 3), axis=0)
                self._g = np.concatenate([g, np.full((g.shape[0], 1), self.z)], -1)
            else:
                xs = np.arange(w["x_min"], w["x_max"] + 1e-9, 0.25)
                ys = np.concatenate([np.arange(lo, hi + 1e-9, 0.25) for lo, hi in w["sidewalks_y"]])
                X, Y = np.meshgrid(xs, ys, indexing="ij")
                self._g = np.stack([X.ravel(), Y.ravel(), np.full(X.size, self.z)], -1)
        return self._g

    def search_cost(self, meas: list[dict], g: np.ndarray) -> np.ndarray | None:
        """LoS-angle + LoS-TDoA cost of candidate positions g [G, 3] (None if an O-RU has no component)."""
        cost = np.zeros(g.shape[0])
        tau_sel = []
        for c in range(self.C):
            m = meas[c]
            if not m["valid"].any():
                return None
            d = g - self.oru[c]
            r = np.linalg.norm(d, axis=-1)
            uy, uz = d[:, 1] / r, d[:, 2] / r
            zy, zz = m["uy"][m["valid"]], m["uz"][m["valid"]]
            ca = ((uy[:, None] - zy[None]) ** 2 + (uz[:, None] - zz[None]) ** 2) / 0.02 ** 2
            if "sx" in m:  # back-to-back panels: a component only explains a LoS in its panel's half-space
                gsx = np.where(d[:, 0] >= 0.0, 1, -1)
                ca = np.where(gsx[:, None] == m["sx"][m["valid"]][None], ca, 1e12)
            j = ca.argmin(1)
            cost += ca[np.arange(g.shape[0]), j]
            tau_sel.append((m["tau_ns"][m["valid"]][j], r / 0.299792458))
        if self.C >= 2:
            dt_meas = wrap(tau_sel[0][0] - tau_sel[1][0], self.period)
            dt_geo = tau_sel[0][1] - tau_sel[1][1]
            cost += (dt_meas - dt_geo) ** 2 / 3.0 ** 2
        return cost

    def global_search(self, meas: list[dict]) -> np.ndarray | None:
        """Position from a grid over the walkable area on LoS angles and LoS TDoA of the components (both O-RUs)."""
        g = self._grid()
        cost = self.search_cost(meas, g)
        if cost is None:
            return None
        k = int(np.argmin(cost))
        self._last_search = (g[k, :2], float(cost[k]))
        return g[k, :2]

    def init(self, p0: np.ndarray) -> None:
        x = np.zeros(self.n)
        x[IX], x[IY] = p0
        P = np.zeros((self.n, self.n))
        P[IX, IX] = P[IY, IY] = self.p.init_sigma_m ** 2
        P[IVX, IVX] = P[IVY, IVY] = self.p.init_sigma_v ** 2
        P[IB, IB] = self.p.sigma_b_ns ** 2
        P[ID0, ID0] = P[ID0 + 1, ID0 + 1] = max(self.p.sigma_sync_ns, 1e-4) ** 2
        for f in range(self.F):
            P[ID0 + 2 + f, ID0 + 2 + f] = max(self.p.sigma_map_m, 1e-4) ** 2
        if self.st is not None and self.st.started:  # keep the learnt offsets on a re-initialisation
            x[ID0:] = self.st.x[ID0:]
            P[ID0:, ID0:] = self.st.P[ID0:, ID0:]
        self.st = TrackState(x=x, P=P, started=True)

    # ------------------------------------------------------------------ filter
    def predict(self, dt: float = 0.1) -> None:
        x, P = self.st.x, self.st.P
        Fm = np.eye(self.n)
        Fm[IX, IVX] = Fm[IY, IVY] = dt
        Q = np.zeros((self.n, self.n))
        q = self.p.q_acc
        for a, b in ((IX, IVX), (IY, IVY)):
            Q[a, a] = q * dt ** 3 / 3
            Q[a, b] = Q[b, a] = q * dt ** 2 / 2
            Q[b, b] = q * dt
        x = Fm @ x
        P = Fm @ P @ Fm.T + Q
        x[IB] = 0.0
        P[IB, :] = 0.0
        P[:, IB] = 0.0
        P[IB, IB] = self.p.sigma_b_ns ** 2
        self.st.x, self.st.P = x, P

    def predicted_samples(self, K: int, rng: np.random.Generator) -> np.ndarray:
        """K samples of the predicted UE position [K, 3]."""
        L = np.linalg.cholesky(self.st.P[:2, :2] + 1e-9 * np.eye(2))
        s = self.st.x[:2][None] + rng.standard_normal((K, 2)) @ L.T
        return np.concatenate([s, np.full((K, 1), self.z)], -1)

    def update(self, meas: list[dict], q: list[np.ndarray], idx: list[np.ndarray], los_mask: list[np.ndarray]) -> dict:
        """One epoch. meas[c]: components (tau_ns, uy, uz, var_*, valid); q[c] visibility of sources idx[c]; los_mask[c] bool per source."""
        from sim.tvt.bp import bp_marginals

        pr = self.p
        lam = pr.clutter_per_snapshot / pr.clutter_volume
        info = {"los_assoc": 0, "assoc": 0.0, "nis_los": []}
        plans = []
        for c in range(self.C):
            m = meas[c]
            jj = np.flatnonzero(m["valid"])
            S_idx = idx[c]
            if len(S_idx) == 0 or len(jj) == 0:
                continue
            h, H = self.predict_meas(self.st.x, c, S_idx)
            z = np.stack([m["tau_ns"][jj], m["uy"][jj], m["uz"][jj]], -1)  # [J, 3]
            R = np.stack([m["var_tau"][jj] + pr.floor_tau_ns ** 2, m["var_uy"][jj] + pr.floor_u ** 2, m["var_uz"][jj] + pr.floor_u ** 2], -1)
            Rb = R + np.array([pr.blk_tau_ns ** 2, pr.blk_u ** 2, pr.blk_u ** 2])
            HPH = np.einsum("sai,ij,sbj->sab", H, self.st.P, H)  # [S, 3, 3]
            nu = z[None, :, :] - h[:, None, :]
            nu[..., 0] = wrap(nu[..., 0], self.period)
            qs = np.clip(q[c], 0.0, 1.0)
            pdc = np.where(los_mask[c], pr.pd_los, pr.pd_nlos)
            # three hypotheses per detected source (sum = its likelihood ratio against clutter):
            # 0 visible, nominal covariance R; 1 visible but its component is corrupted (merged with an unresolved path
            # or otherwise biased; weight eps_robust), inflated R + R_blk; 2 blocked: model-B attenuated, detected with
            # P_D,blk; a blocked LoS arrives as a diffracted detour (inflated R + R_blk), a blocked reflection keeps its
            # geometry (R) - the signal model of the synthesis (paper-2 variant b detours the blocked LoS only).
            lm_c = los_mask[c]
            Rblk = np.where(lm_c[:, None, None], Rb[None], R[None])  # [S, J, 3]
            hyps = ((np.broadcast_to(R[None], Rblk.shape), qs * (1.0 - pr.eps_robust) * pdc), (np.broadcast_to(Rb[None], Rblk.shape), qs * pr.eps_robust * pdc),
                    (Rblk, (1.0 - qs) * pr.pd_blk))
            wmode = np.zeros((len(S_idx), len(jj), 3))
            gate = np.zeros((len(S_idx), len(jj)), bool)
            for mode, (Rm, pw) in enumerate(hyps):
                Sm = HPH[:, None] + Rm[..., :, None] * np.eye(3)  # [S, J, 3, 3]
                Si = np.linalg.inv(Sm)
                d2 = np.einsum("sja,sjab,sjb->sj", nu, Si, nu)
                det = np.linalg.det(Sm)
                lik = np.exp(-0.5 * d2) / np.sqrt((2 * np.pi) ** 3 * np.maximum(det, 1e-300))
                gate |= d2 < pr.gate
                wmode[..., mode] = pw[:, None] * lik / lam
            if "sx" in m:  # back-to-back panels: association only within the measuring panel's half-space
                gate &= self._pred_sx[:, None] == np.asarray(m["sx"])[jj][None, :]
            wmode = np.where(gate[..., None], wmode, 0.0)
            w = wmode.sum(-1)
            w0 = qs * (1 - pdc) + (1 - qs) * (1 - pr.pd_blk)
            beta, _ = bp_marginals(w0, w)
            plans.append((c, S_idx, jj, z, R, Rb, Rblk, beta, wmode, los_mask[c]))
        # sequential PDA updates, strongest associations first
        order = []
        for pi_, (c, S_idx, jj, z, R, Rb, Rblk, beta, wmode, lm) in enumerate(plans):
            for si in range(len(S_idx)):
                order.append((1.0 - beta[si, 0], pi_, si))
        order.sort(reverse=True)
        for _, pi_, si in order:
            c, S_idx, jj, z, R, Rb, Rblk, beta, wmode, lm = plans[pi_]
            if beta[si, 0] > 0.999:
                continue
            h, H = self.predict_meas(self.st.x, c, S_idx[si:si + 1])
            h, H = h[0], H[0]
            x, P = self.st.x, self.st.P
            comps = [(beta[si, 0], x, P)]
            for j in range(len(jj)):
                bj = beta[si, j + 1]
                if bj < 1e-6:
                    continue
                tot = wmode[si, j].sum()
                for mode, Rm in ((0, R[j]), (1, Rb[j]), (2, Rblk[si, j])):
                    wm = bj * (wmode[si, j, mode] / tot if tot > 0 else 1.0 / 3.0)
                    if wm < 1e-6:
                        continue
                    S = H @ P @ H.T + np.diag(Rm)
                    nu = z[j] - h
                    nu[0] = wrap(nu[0], self.period)
                    Si = np.linalg.inv(S)
                    if float(nu @ Si @ nu) > pr.gate:  # re-gate at the CURRENT state of the sequential update
                        comps[0] = (comps[0][0] + wm, x, P)
                        continue
                    K = P @ H.T @ Si
                    comps.append((wm, x + K @ nu, P - K @ S @ K.T))
                    if lm[si] and mode == 0:
                        info["nis_los"].append(float(nu @ np.linalg.solve(S, nu)) * wm)
            if lm[si] and beta[si, 0] < 0.5:
                info["los_assoc"] += 1
            info["assoc"] += 1.0 - beta[si, 0]
            wsum = sum(cw for cw, _, _ in comps)
            xm = sum(cw * cx for cw, cx, _ in comps) / wsum
            Pm = sum(cw * (cP + np.outer(cx - xm, cx - xm)) for cw, cx, cP in comps) / wsum
            self.st.x, self.st.P = xm, 0.5 * (Pm + Pm.T)
        self._constrain()
        return info

    def _constrain(self) -> None:
        """Project the position onto the walkable area (sidewalk bands, street extent; or a union of rectangles)."""
        w = self.walk
        x = self.st.x
        if "rects" in w:
            best = None
            for x0, x1, y0, y1 in w["rects"]:
                px, py = float(np.clip(x[IX], x0, x1)), float(np.clip(x[IY], y0, y1))
                d = math.hypot(px - x[IX], py - x[IY])
                if best is None or d < best[0]:
                    best = (d, px, py)
            x[IX], x[IY] = best[1], best[2]
            return
        px = float(np.clip(x[IX], w["x_min"], w["x_max"]))
        best = None
        for lo, hi in w["sidewalks_y"]:
            py = float(np.clip(x[IY], lo, hi))
            d = abs(py - x[IY])
            if best is None or d < best[0]:
                best = (d, py)
        on_crossing = any(abs(x[IX] - xc) <= w["crossing_half_width"] and w["street_y"][0] <= x[IY] <= w["street_y"][1] for xc in w["crossings_x"])
        if not on_crossing:
            x[IY] = best[1]
        x[IX] = px

    def step(self, meas: list[dict], q: list[np.ndarray], idx: list[np.ndarray], los_mask: list[np.ndarray]) -> dict:
        """Update with this epoch's data (the caller has called predict() and computed idx/q at the predicted state)."""
        info = self.update(meas, q, idx, los_mask)
        st = self.st
        if info["assoc"] < 0.5:  # no source associated at all (LoS blockage alone must not restart the track)
            st.bad += 1
        else:
            st.bad = 0
        st.nis_hist.append(np.sum(info["nis_los"]) if info["nis_los"] else np.nan)
        st.nis_hist = st.nis_hist[-self.p.reinit_after:]
        nis = np.nanmean(st.nis_hist) if np.isfinite(st.nis_hist).any() else 0.0
        info["reinit"] = False
        # consistency with the global LoS search (catches a track that follows spurious associations, e.g. after the
        # UE wraps around the periodic street)
        p0 = self.global_search(meas)
        if p0 is not None:
            here = self.search_cost(meas, np.array([[st.x[IX], st.x[IY], self.z]]))
            st.off = st.off + 1 if (here is not None and float(here[0]) - self._last_search[1] > self.p.search_margin) else 0
        if "rects" in self.walk:  # section edges: the outer ends of the rectangles' long axes
            ex = self.walk.get("edges", [])
            at_edge = any(abs(st.x[IX] - xe) < 0.5 and st.x[IVX] * sx > 0.3 for xe, sx in ex.get("x", [])) or \
                any(abs(st.x[IY] - ye) < 0.5 and st.x[IVY] * sy > 0.3 for ye, sy in ex.get("y", []))
        else:
            out_lo = (st.x[IX] - self.walk["x_min"] < 0.5) and st.x[IVX] < -0.3
            out_hi = (self.walk["x_max"] - st.x[IX] < 0.5) and st.x[IVX] > 0.3
            at_edge = out_lo or out_hi  # at the edge of the section and moving out of it
        st.edge = st.edge + 1 if at_edge else 0
        reasons = [n for n, c in (("no_assoc", st.bad >= self.p.reinit_after),
                                  ("nis", len(st.nis_hist) >= self.p.reinit_after and nis > self.p.nis_reinit),
                                  ("search", st.off >= self.p.search_epochs), ("edge", st.edge >= self.p.edge_epochs)) if c]
        info["reinit_reason"] = reasons
        if reasons and p0 is not None:
            self.init(p0)
            info["reinit"] = True
        return info
