"""Sanity checks for the sense2connect environment (Sionna 2.2).

Checks
  1. Environment   : Dr.Jit/Mitsuba variant (CUDA vs. LLVM), PyTorch + GPU
  2. Radar equation: RCSSolver received power and range vs. closed form
  3. Doppler       : RCSSolver Doppler shift vs. -2 v_r / lambda
  4. Blockage      : a sensing target blocks the LoS comm link (PathSolver)
                     -> the coupling sensing <-> blockage the idea relies on
  5. PHY / ISAC    : range-Doppler map via sionna.phy.isac (needs PyTorch)
  6. Benchmark     : wall time per snapshot, to budget the experiments

Usage
  python scripts/sanity_check.py                  # all checks
  python scripts/sanity_check.py --require-gpu    # fail if not on CUDA
  python scripts/sanity_check.py --skip-phy --skip-bench
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.constants import speed_of_light as C

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
failures = []


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------
# 1. Environment
# --------------------------------------------------------------------------
def check_environment(require_gpu):
    import mitsuba as mi
    import drjit as dr
    import sionna.rt as rt

    # Importing sionna.rt selects a variant: cuda_* on GPU, llvm_* on CPU
    variant = mi.variant()
    print(f"sionna-rt {rt.__version__} | mitsuba {mi.__version__} | "
          f"drjit {dr.__version__} | variant {variant}")
    on_cuda = variant.startswith("cuda")
    if require_gpu:
        report("Ray tracing on CUDA/OptiX", on_cuda,
               "(if FAIL: check NVIDIA_DRIVER_CAPABILITIES=all)")
    else:
        print(f"[INFO] Ray tracing backend: {'CUDA' if on_cuda else 'LLVM (CPU)'}")

    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU only"
        print(f"torch {torch.__version__} | {dev}")
        if require_gpu:
            report("PyTorch sees GPU", cuda)
    except (ImportError, OSError):
        print("[INFO] PyTorch not installed: PHY checks will be skipped")


def monostatic_scene(frequency):
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
    scene = load_scene()  # empty scene
    scene.frequency = frequency
    scene.add(Transmitter("tx", position=[0, 0, 0]))
    scene.add(Receiver("rx", position=[0, 0, 0]))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 polarization="V", pattern="iso")
    scene.rx_array = scene.tx_array
    return scene


# --------------------------------------------------------------------------
# 2 + 3. Radar equation and Doppler
# --------------------------------------------------------------------------
def check_radar_equation_and_doppler():
    from sionna.rt.rcs import RCSSolver, ConstantRCSSensingTarget

    scene = monostatic_scene(3.5e9)
    sigma, r, v = 3.0, 100.0, -100.0  # approaching at 100 m/s
    scene.add(ConstantRCSSensingTarget("st", sigma=sigma,
                                       position=(0, 0, r),
                                       velocity=(0, 0, v)))
    paths = RCSSolver(deterministic=True)(scene, max_depth=1, seed=1)
    a, tau = paths.cir(out_type="numpy", normalize_delays=False)
    p_rt = float(np.abs(np.squeeze(a)) ** 2)
    r_rt = float(np.squeeze(tau)) * C / 2

    lam = float(scene.wavelength.numpy()[0])
    p_exp = sigma * lam ** 2 / ((4 * np.pi) ** 3 * r ** 4)
    rel = abs(p_rt - p_exp) / p_exp
    report("Radar equation (power)", rel < 1e-3, f"rel. error {rel:.1e}")
    report("Radar equation (range)", abs(r_rt - r) < 1e-2, f"{r_rt:.3f} m")

    fd_rt = float(np.squeeze(paths.doppler.numpy()))
    fd_exp = -2 * v / lam  # radial velocity v (receding > 0) -> -2 v / lambda
    report("Doppler shift", abs(fd_rt - fd_exp) / abs(fd_exp) < 1e-3,
           f"{fd_rt:.1f} Hz (expected {fd_exp:.1f} Hz)")


# --------------------------------------------------------------------------
# 4. Blockage coupling
# --------------------------------------------------------------------------
def check_blockage():
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    from sionna.rt.rcs import TR38901SensingTarget

    scene = load_scene()
    scene.frequency = 28e9
    scene.add(Transmitter("tx", position=[-50, 0, 1.5]))
    scene.add(Receiver("rx", position=[50, 0, 1.5]))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 polarization="V", pattern="iso")
    scene.rx_array = scene.tx_array
    solver = PathSolver()

    def link_power():
        p = solver(scene, max_depth=1, los=True, specular_reflection=False,
                   diffraction=True, edge_diffraction=True, seed=1)
        a, _ = p.cir(out_type="numpy", normalize_delays=False)
        return float(np.sum(np.abs(a) ** 2))

    p_clear = link_power()
    car = TR38901SensingTarget("car", object_type="vehicle-multi-sp",
                               position=[0, 0, 0.8])
    scene.add(car)
    p_blocked = link_power()
    car.position = [0, 5, 0.8]
    p_moved = link_power()

    loss = ("inf (no power)" if p_blocked == 0
            else f"{10*np.log10(p_clear/p_blocked):.0f} dB")
    report("Sensing target blocks LoS", p_blocked < 1e-6 * p_clear,
           f"loss {loss}")
    report("Link restored when target moves away",
           abs(p_moved - p_clear) / p_clear < 1e-3)
    print("[NOTE] Blockage by a sensing target is binary (absorber, no "
          "diffraction around it). A realistic blockage loss model, e.g., "
          "3GPP TR 38.901 blockage model B, must be applied on top.")


# --------------------------------------------------------------------------
# 5. PHY: range-Doppler map
# --------------------------------------------------------------------------
def check_phy_range_doppler():
    try:
        import torch  # noqa: F401
        from sionna.phy.channel import (cir_to_ofdm_channel,
                                        ofdm_to_delay_doppler_channel,
                                        subcarrier_frequencies)
        from sionna.phy.isac import plot_delay_doppler
    except (ImportError, OSError) as e:
        print(f"[SKIP] PHY check ({e})")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sionna.rt.rcs import RCSSolver, ConstantRCSSensingTarget

    scene = monostatic_scene(3.5e9)
    r, v = 100.0, -20.0
    scene.add(ConstantRCSSensingTarget("st", sigma=3.0, position=(0, 0, r),
                                       velocity=(0, 0, v)))
    paths = RCSSolver(deterministic=True)(scene, max_depth=1, seed=1)

    scs, n_sc, n_t, n_delay = 30e3, 512, 200, 32
    bandwidth = n_sc * scs
    a, tau = paths.cir(sampling_frequency=scs, num_time_steps=n_t,
                       normalize_delays=False, out_type="torch")
    freqs = subcarrier_frequencies(n_sc, scs)
    h_f = cir_to_ofdm_channel(freqs, a[None], tau[None])[0, 0, 0, 0, 0]
    h_dd = ofdm_to_delay_doppler_channel(h_f, l_max=n_delay - 1)
    power = h_dd.abs().square().cpu()

    # Soft check on the range axis: [doppler, delay], delay bins 0..l_max
    l_peak = int(power.sum(dim=0).argmax())
    r_peak = l_peak * C / (2 * bandwidth)
    r_res = C / (2 * bandwidth)
    ok = abs(r_peak - r) <= 1.5 * r_res
    print(f"[{'PASS' if ok else 'WARN'}] Range-Doppler peak at {r_peak:.1f} m "
          f"(target {r} m, resolution {r_res:.1f} m)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _, ax = plt.subplots(figsize=(8, 5))
    plot_delay_doppler(power, fast_time_sample_rate=bandwidth,
                       slow_time_sample_rate=scs,
                       wavelength=float(scene.wavelength.numpy()[0]),
                       domain="range_velocity", ax=ax)
    ax.scatter([r], [-v], color="red", marker="x", s=80, label="Expected")
    ax.legend()
    out = os.path.join(RESULTS_DIR, "sanity_range_doppler.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[INFO] Saved {out} (check visually: peak on the red cross)")


# --------------------------------------------------------------------------
# 6. Benchmark
# --------------------------------------------------------------------------
def benchmark(num_cars, repeats):
    import sionna.rt as rt
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    from sionna.rt.rcs import RCSSolver, TR38901SensingTarget

    scene = load_scene(rt.scene.simple_street_canyon)
    scene.frequency = 28e9
    scene.add(Transmitter("tx", position=[0, 0, 10]))
    scene.add(Receiver("rx", position=[0, 0, 10]))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 polarization="V", pattern="iso")
    scene.rx_array = scene.tx_array
    cars = [TR38901SensingTarget(f"car-{i}", object_type="vehicle-multi-sp",
                                 position=[-40 + 80 * i / max(num_cars - 1, 1),
                                           2.0 * (-1) ** i, 0.8],
                                 velocity=[10.0 * (-1) ** i, 0, 0])
            for i in range(num_cars)]
    scene.add(cars)
    rcs_solver, bg_solver = RCSSolver(), PathSolver()

    def snapshot():
        s = rcs_solver(scene, max_depth=3, samples_per_sp=100_000,
                       buffer_size_per_sp=100_000, seed=1)
        b = bg_solver(scene, max_depth=3, los=False, seed=1)
        a, _ = b.concat(s).cir(out_type="numpy", normalize_delays=False)
        return a

    t0 = time.perf_counter(); snapshot(); t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(repeats):
        snapshot()
    t_avg = (time.perf_counter() - t0) / repeats
    print(f"[INFO] Benchmark ({num_cars} cars, depth 3): first call "
          f"{t_first:.2f} s (incl. JIT), then {t_avg:.3f} s/snapshot "
          f"-> ~{3600/t_avg:,.0f} snapshots/hour")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--skip-phy", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--bench-cars", type=int, default=20)
    ap.add_argument("--bench-repeats", type=int, default=5)
    args = ap.parse_args()

    check_environment(args.require_gpu)
    check_radar_equation_and_doppler()
    check_blockage()
    if not args.skip_phy:
        check_phy_range_doppler()
    if not args.skip_bench:
        benchmark(args.bench_cars, args.bench_repeats)

    print("\n" + ("All checks passed." if not failures
                  else f"{len(failures)} check(s) failed: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
