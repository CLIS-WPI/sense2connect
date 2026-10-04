"""ADDED AFTER THE SECOND EXTERNAL REVIEW: tracker-error injection into ground-truth tracks.

Two models act on the ground-truth tracks of scripts/review_b5_ablation.py
(every blocker's true position, velocity and size at each 0.1 s report).

REALISTIC (primary), calibrated on the development seeds by
scripts/review2_calibrate.py (results/M5/review2/calibration.json):
- coverage + misses: a blocker can only be tracked inside the 40 m sensing
  range of the radar at oru-0. Each visit of the range copies the
  matched / missed pattern of a measured visit of the same class and
  similar length (acquisition delay, whole-track outages and their
  durations as measured). Outside the range it is never tracked.
- state errors: per track (= matched run) and component, an AR(1) process
  e_k = mu + d_k, d_k = phi d_{k-1} + sqrt(1 - phi^2) sigma w_k, started in
  its stationary distribution, with the measured per-class mean mu, standard
  deviation sigma and lag-1 autocorrelation phi on x (along every lane /
  sidewalk), y (cross), vx, vy, measured on the pinned state the predictor
  sees. Sweeps ("pos" / "vel", sigma s) put the error on the ALONG-line
  component only (x or vx; mu = 0, phi of the class), as for a
  map-constrained track; cross components stay exact.
- false tracks: births per CPI ~ Poisson(measured birth rate per type and
  mount); each birth copies a measured episode (lifetime and states) of the
  same mount: a fragment keeps its measured offset to a target of the
  measured class (attached to a random blocker of that class inside the
  range; any class if none), an "other" track its measured absolute states.
- size rule: the predictor's class-agnostic sizes (sidewalk -> pedestrian
  box; lane or free -> bus box); a track is free with the measured per-class
  share of free-mode CPIs (drawn per matched run).

MEMORYLESS (secondary, limiting case): independent draws every report;
misses Bernoulli per blocker and report; noise white Gaussian on both
horizontal axes ("pos": x and y; "vel": vx and vy).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SENSING_RANGE_M = 40.0
COMP_INDEX = {"x": 0, "y": 1, "vx": 3, "vy": 4}


def _ar1(n: int, sigma: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) sequence of length n, standard deviation sigma, lag-1 correlation phi."""
    if n <= 0 or sigma <= 0:
        return np.zeros(max(n, 0))
    w = rng.normal(0.0, 1.0, n)
    out = np.empty(n)
    out[0] = sigma * w[0]
    a = math_sqrt(1.0 - phi * phi) * sigma
    for k in range(1, n):
        out[k] = phi * out[k - 1] + a * w[k]
    return out


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(x, 0.0)))


def _segments(in_range: np.ndarray, cls: str, cal: dict, rng: np.random.Generator, miss: bool) -> list[tuple[int, int]]:
    """Matched runs [start, end) of one blocker over the reports.

    For every visit of the range (consecutive in-range reports) a measured
    visit pattern of the same class is drawn at random among the 10 measured
    visits whose length is closest to (and preferably at least) this one,
    and its first n CPIs are used; a shorter pattern is continued with its
    last state.
    """
    n = len(in_range)
    if not miss:
        return [(0, n)]
    pats = _visits(cal, cls)
    lens = pats["len"]
    segs = []
    r = 0
    while r < n:
        if not in_range[r]:
            r += 1
            continue
        e = r
        while e < n and in_range[e]:
            e += 1
        L = e - r
        i0 = int(np.searchsorted(lens, L))
        cand = range(i0, min(len(lens), i0 + 10)) if i0 < len(lens) else range(max(0, len(lens) - 10), len(lens))
        pat = pats["bits"][int(cand[int(rng.integers(len(cand)))])]
        bits = (pat + pat[-1] * max(0, L - len(pat)))[:L]
        k = 0
        while k < L:
            if bits[k] == "1":
                j = k
                while j < L and bits[j] == "1":
                    j += 1
                segs.append((r + k, r + j))
                k = j
            else:
                k += 1
        r = e
    return segs


_VISITS: dict = {}


def _visits(cal: dict, cls: str) -> dict:
    key = (id(cal), cls)
    if key not in _VISITS:
        v = sorted(cal["outages"][cls]["visits"], key=len)
        _VISITS[key] = {"bits": v, "len": np.array([len(x) for x in v])}
    return _VISITS[key]


def _pack(cols: list[list[tuple[np.ndarray, np.ndarray]]], n_r: int) -> dict[str, np.ndarray]:
    """Per report a list of (state[5], size[3]) -> padded state [R, K, 5], size [R, K, 3], valid [R, K]."""
    k = max(1, max(len(c) for c in cols))
    state = np.zeros((n_r, k, 5))
    size = np.ones((n_r, k, 3))
    valid = np.zeros((n_r, k), dtype=bool)
    for r, rows in enumerate(cols):
        for i, (s, z) in enumerate(rows):
            state[r, i] = s
            size[r, i] = z
            valid[r, i] = True
    return {"state": state, "size": size, "valid": valid}


def inject_realistic(tr: dict, comp: dict, rng: np.random.Generator, cal: dict, mount: str, diag: dict | None = None) -> dict[str, np.ndarray]:
    """Tracks [R, K] for the predictor from ground truth ``tr`` with the realistic error components ``comp``.

    ``comp``: miss (bool), false (bool), size (bool), noise (None, "cal",
    ("pos", s), ("vel", s) or ("both", s)). ``diag`` collects validation counts.
    """
    st = tr["state"]
    n_r, n_b, _ = st.shape
    radar = tr["oru"][0]
    bus = tr["kinds"]["bus"]
    ped = tr["kinds"]["pedestrian"]
    bus_sz = np.array([float(bus["length_m"]), float(bus["width_m"]), float(bus["height_m"])])
    ped_sz = np.array([float(ped["length_m"]), float(ped["width_m"]), float(ped["height_m"])])
    dist = np.linalg.norm(st[:, :, :3] - radar[None, None, :], axis=-1)
    in_range = dist <= SENSING_RANGE_M
    noise = comp.get("noise")
    cols: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(n_r)]
    for b in range(n_b):
        cls = tr["cls"][b]
        if cls not in cal["errors"]:
            cls = "car"
        segs = _segments(in_range[:, b], cls, cal, rng, bool(comp.get("miss")))
        if diag is not None:
            d = diag.setdefault(cls, {"in_range": 0, "tracked_in_range": 0})
            d["in_range"] += int(in_range[:, b].sum())
            for s0, s1 in segs:
                d["tracked_in_range"] += int(in_range[s0:s1, b].sum())
        e = cal["errors"][cls]
        for s0, s1 in segs:
            n = s1 - s0
            seg = st[s0:s1, b].copy()
            if noise == "cal":
                for c, i in COMP_INDEX.items():
                    seg[:, i] += e["mean"][c] + _ar1(n, e["std"][c], e["phi"][c], rng)
            elif noise is not None:
                kind, s = noise
                if kind in ("pos", "both"):
                    seg[:, 0] += _ar1(n, s, e["phi"]["x"], rng)
                if kind in ("vel", "both"):
                    seg[:, 3] += _ar1(n, s, e["phi"]["vx"], rng)
            if comp.get("size"):
                free = rng.random() < e["free_mode_share"]
                sz = bus_sz if (free or cls != "pedestrian") else ped_sz
            else:
                sz = tr["size"][b]
            for i in range(n):
                cols[s0 + i].append((seg[i], sz))
    if comp.get("false"):
        fm = cal["false_tracks"]["per_mount"][mount]
        n_false = 0
        for r in range(n_r):
            for typ in ("fragment", "other"):
                lib = fm["library"][typ]
                if not lib:
                    continue
                for _ in range(int(rng.poisson(fm["stats"][typ]["birth_rate_per_cpi"]))):
                    ep = lib[int(rng.integers(len(lib)))]
                    if typ == "fragment":
                        cand = [b for b in range(n_b) if in_range[r, b] and tr["cls"][b] == ep["target_cls"]]
                        cand = cand or [b for b in range(n_b) if in_range[r, b]]
                        if not cand:
                            continue
                        b = cand[int(rng.integers(len(cand)))]
                        sz = ped_sz if ep["mode"] == "sidewalk" else bus_sz
                        for i, off in enumerate(ep["offset"]):
                            if r + i >= n_r:
                                break
                            s = st[r + i, b].copy()
                            s[[0, 1, 3, 4]] += np.asarray(off)
                            cols[r + i].append((s, sz))
                            n_false += 1
                    else:
                        sz = ped_sz if ep["mode"][0] == "sidewalk" else bus_sz
                        z = 0.5 * sz[2]
                        for i, sv in enumerate(ep["state"]):
                            if r + i >= n_r:
                                break
                            cols[r + i].append((np.array([sv[0], sv[1], z, sv[2], sv[3]]), sz))
                            n_false += 1
        if diag is not None:
            diag.setdefault("false", {"count": 0, "reports": 0})
            diag["false"]["count"] += n_false
            diag["false"]["reports"] += n_r
    return _pack(cols, n_r)


def inject_memoryless(tr: dict, kind: str, s: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """White Gaussian error, sigma s, on both horizontal position axes ("pos"), velocity axes ("vel") or both ("both")."""
    st = tr["state"].copy()
    n_r, n_b, _ = st.shape
    if kind in ("pos", "both"):
        st[:, :, 0] += rng.normal(0.0, s, (n_r, n_b))
        st[:, :, 1] += rng.normal(0.0, s, (n_r, n_b))
    if kind in ("vel", "both"):
        st[:, :, 3] += rng.normal(0.0, s, (n_r, n_b))
        st[:, :, 4] += rng.normal(0.0, s, (n_r, n_b))
    return {"state": st, "size": np.broadcast_to(tr["size"], (n_r, n_b, 3)).copy(), "valid": np.ones((n_r, n_b), dtype=bool)}
