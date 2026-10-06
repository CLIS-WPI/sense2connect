"""Seed-level statistics for the TVT milestones (ROADMAP_TVT.md research integrity).

The seed is the statistical unit: per seed one number (its runs pooled or averaged), exact
two-sided Wilcoxon signed-rank test over the seeds (all 2^n sign patterns, average ranks for
ties, zero differences dropped as in the paper-1/2 code), bootstrap 95 % CI over seeds.
"""

from __future__ import annotations

import itertools

import numpy as np

BOOT_N = 10_000
BOOT_SEED = 20261006


def wilcoxon_exact(d) -> float:
    """Two-sided exact p-value of the signed-rank statistic for paired differences d (zeros dropped)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w = ranks[d > 0].sum()
    tot = ranks.sum()
    stat = min(w, tot - w)
    cnt = 0
    for signs in itertools.product((0, 1), repeat=n):
        wp = float(np.dot(signs, ranks))
        if min(wp, tot - wp) <= stat + 1e-12:
            cnt += 1
    return min(1.0, cnt / 2 ** n)


def boot_ci(v, n: int = BOOT_N, seed: int = BOOT_SEED) -> list[float]:
    """95 % bootstrap CI of the mean over seeds."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, size=(n, v.size))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def seed_summary(per_seed) -> dict:
    v = np.asarray(per_seed, float)
    return {"mean": float(np.nanmean(v)), "ci95_boot": boot_ci(v), "per_seed": [float(x) for x in v]}


def paired(a, b) -> dict:
    """Seed-paired comparison of a vs b (lists over the same seeds): mean difference a - b, CI, exact Wilcoxon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return {"mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)), "mean_diff": float(np.nanmean(d)), "ci95_boot": boot_ci(d),
            "wilcoxon_p_two_sided": wilcoxon_exact(d), "n_seeds": int(np.isfinite(d).sum()), "n_lower": int((d < 0).sum()), "n_higher": int((d > 0).sum())}
