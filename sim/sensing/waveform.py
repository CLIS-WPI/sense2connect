"""OFDM radar waveform at numerology 3.

One sensing symbol per slot sets the pulse repetition frequency to the
slot rate. A coherent processing interval is a run of those symbols.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.constants import speed_of_light


@dataclass(frozen=True)
class Waveform:
    """Sensing OFDM parameters.

    ``overhead`` is the fraction of OFDM symbols in a slot used for sensing.
    It is recorded for later resource accounting and does not drop symbols
    inside the radar CPI.
    """

    numerology: int
    n_subcarriers: int
    cpi_slots: int
    sensing_symbols_per_slot: int
    symbols_per_slot: int
    carrier_hz: float
    window: bool = True

    @property
    def subcarrier_spacing_hz(self) -> float:
        """Subcarrier spacing [Hz]. Numerology ``mu`` uses ``15 kHz * 2^mu``."""
        return 15e3 * (2 ** int(self.numerology))

    @property
    def slot_s(self) -> float:
        """Slot duration [s]."""
        return 1e-3 / (2 ** int(self.numerology))

    @property
    def prf_hz(self) -> float:
        """Pulse repetition frequency [Hz]. One sensing symbol per slot."""
        symbols = int(self.sensing_symbols_per_slot)
        if symbols != 1:
            raise ValueError("this waveform defines PRF as one sensing symbol per slot")
        return 1.0 / self.slot_s

    @property
    def cpi_s(self) -> float:
        """Coherent processing interval [s]."""
        return int(self.cpi_slots) * self.slot_s

    @property
    def bandwidth_hz(self) -> float:
        """Occupied bandwidth [Hz]."""
        return int(self.n_subcarriers) * self.subcarrier_spacing_hz

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength [m]."""
        return speed_of_light / float(self.carrier_hz)

    @property
    def v_max_mps(self) -> float:
        """Unambiguous approaching speed [m/s], ``lambda * PRF / 4``."""
        return self.wavelength_m * self.prf_hz / 4.0

    @property
    def v_res_mps(self) -> float:
        """Radial-speed resolution [m/s], ``lambda / (2 * CPI)``."""
        return self.wavelength_m / (2.0 * self.cpi_s)

    @property
    def range_resolution_m(self) -> float:
        """Monostatic range resolution [m], ``c / (2 B)``."""
        return speed_of_light / (2.0 * self.bandwidth_hz)

    @property
    def overhead(self) -> float:
        """Fraction of slot symbols spent on sensing."""
        return float(self.sensing_symbols_per_slot) / float(self.symbols_per_slot)


def waveform_from_config(scenario: dict, n_subcarriers: int | None = None) -> Waveform:
    """Build the waveform from ``scenario['sensing_radar']['waveform']``."""
    spec = scenario["sensing_radar"]["waveform"]
    count = int(spec["n_subcarriers"] if n_subcarriers is None else n_subcarriers)
    return Waveform(
        numerology=int(spec["numerology"]),
        n_subcarriers=count,
        cpi_slots=int(spec["cpi_slots"]),
        sensing_symbols_per_slot=int(spec["sensing_symbols_per_slot"]),
        symbols_per_slot=int(spec["symbols_per_slot"]),
        carrier_hz=float(scenario["carrier_hz"]),
        window=bool(spec.get("window", True)),
    )
