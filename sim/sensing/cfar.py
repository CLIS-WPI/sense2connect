"""2D cell-averaging CFAR on a range-Doppler power map."""

from __future__ import annotations

import math

import numpy as np


def training_cells(guard: int, train: int) -> int:
    """Number of cells in the full training ring."""
    outer = 2 * (int(guard) + int(train)) + 1
    inner = 2 * int(guard) + 1
    return outer * outer - inner * inner


def alpha_from_pfa(n_cells: int, pfa: float) -> float:
    """CA-CFAR multiplier for a design false-alarm probability per cell."""
    if n_cells < 1:
        raise ValueError("CA-CFAR needs at least one training cell")
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must lie in (0, 1)")
    return float(n_cells) * (float(pfa) ** (-1.0 / float(n_cells)) - 1.0)


def ca_cfar(
    power: np.ndarray,
    *,
    guard: int,
    train: int,
    pfa: float,
    noise_applied: bool,
) -> tuple[np.ndarray, float]:
    """Return a boolean detection mask and the threshold multiplier.

    ``power`` has shape ``[doppler, range]``. Cells without a full training
    window are never detections. CFAR is refused when the map has no noise:
    a noiseless peak detector is not a CFAR result.
    """
    if not noise_applied:
        raise RuntimeError("CFAR requires thermal noise; refusing a noiseless detection")
    if guard < 0 or train < 1:
        raise ValueError("guard must be >= 0 and train must be >= 1")
    image = np.asarray(power, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("CA-CFAR expects a 2D range-Doppler map")
    n_cells = training_cells(guard, train)
    alpha = alpha_from_pfa(n_cells, pfa)
    radius_outer = guard + train
    total = _box_sum(image, radius_outer)
    guard_sum = _box_sum(image, guard)
    train_sum = total - guard_sum
    noise = train_sum / float(n_cells)
    threshold = alpha * noise
    doppler = np.arange(image.shape[0])[:, None]
    lag = np.arange(image.shape[1])[None, :]
    full = (
        (doppler - radius_outer >= 0)
        & (doppler + radius_outer < image.shape[0])
        & (lag - radius_outer >= 0)
        & (lag + radius_outer < image.shape[1])
    )
    mask = full & np.isfinite(image) & (image > threshold)
    return mask, alpha


def _box_sum(image: np.ndarray, radius: int) -> np.ndarray:
    """Sum of each square of half-width ``radius``, clipped to the image."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    cumulative = np.pad(image, ((1, 0), (1, 0)))
    cumulative = np.cumsum(np.cumsum(cumulative, axis=0), axis=1)
    height, width = image.shape
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    y0 = np.clip(y - radius, 0, height)
    y1 = np.clip(y + radius + 1, 0, height)
    x0 = np.clip(x - radius, 0, width)
    x1 = np.clip(x + radius + 1, 0, width)
    return cumulative[y1, x1] - cumulative[y0, x1] - cumulative[y1, x0] + cumulative[y0, x0]


def local_maxima(mask: np.ndarray, power: np.ndarray) -> list[tuple[int, int, float]]:
    """Keep CFAR hits that are at least as strong as their 4-neighbours."""
    hits: list[tuple[int, int, float]] = []
    doppler_i, range_i = np.nonzero(mask)
    for doppler, lag in zip(doppler_i.tolist(), range_i.tolist(), strict=True):
        value = float(power[doppler, lag])
        if not math.isfinite(value):
            continue
        neighbours = []
        if doppler > 0:
            neighbours.append(float(power[doppler - 1, lag]))
        if doppler + 1 < power.shape[0]:
            neighbours.append(float(power[doppler + 1, lag]))
        if lag > 0:
            neighbours.append(float(power[doppler, lag - 1]))
        if lag + 1 < power.shape[1]:
            neighbours.append(float(power[doppler, lag + 1]))
        if neighbours and value < max(neighbours):
            continue
        hits.append((int(doppler), int(lag), value))
    return hits
