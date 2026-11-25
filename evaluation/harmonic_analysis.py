"""Single-sided harmonic spectrum calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicSpectrum:
    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    harmonic_orders: np.ndarray
    fundamental_amplitude: float
    fundamental_frequency_hz: float
    fundamental_valid: bool = True
    cycle_count: float = 1.0
    is_integer_cycle: bool = True
    analysis_warning: str = ""


def harmonic_spectrum(
    waveform: np.ndarray,
    *,
    sampling_frequency_hz: float = 20_000.0,
    fundamental_frequency_hz: float = 50.0,
    max_harmonic: int = 100,
) -> HarmonicSpectrum:
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("waveform must be a one-dimensional array with at least two points.")
    if not np.isfinite(values).all():
        raise ValueError("waveform must contain only finite values.")
    if sampling_frequency_hz <= 0 or fundamental_frequency_hz <= 0 or max_harmonic < 1:
        raise ValueError("Frequencies and max_harmonic must be positive.")
    if sampling_frequency_hz > 0 and sampling_frequency_hz / 2.0 < fundamental_frequency_hz:
        raise ValueError("Sampling frequency is too low to measure the fundamental frequency.")

    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_frequency_hz)
    bin_width = sampling_frequency_hz / values.size
    detrended = values - np.mean(values)
    cycle_count = float(values.size / sampling_frequency_hz * fundamental_frequency_hz)
    is_integer_cycle = bool(abs(cycle_count - round(cycle_count)) <= 0.01)
    window = np.ones(values.size, dtype=np.float64) if is_integer_cycle else np.hanning(values.size)
    if not np.any(window) or float(np.sum(window)) <= np.finfo(np.float64).eps:
        raise ValueError("Unable to construct a valid FFT analysis window.")
    fft = np.fft.rfft(detrended * window)
    amplitude = (
        2.0 * np.abs(fft) / values.size
        if is_integer_cycle
        else 2.0 * np.abs(fft) / float(np.sum(window))
    )
    if values.size % 2 == 0:
        amplitude[-1] *= 0.5

    expected_bin = fundamental_frequency_hz / bin_width
    fundamental_bin = int(np.clip(round(expected_bin), 1, len(amplitude) - 1))
    if is_integer_cycle:
        detected_bin = fundamental_bin
    else:
        search_radius = max(2, int(np.ceil(0.02 * max(expected_bin, 1.0))))
        search_start = max(1, fundamental_bin - search_radius)
        search_stop = min(len(amplitude), fundamental_bin + search_radius + 1)
        detected_bin = int(search_start + np.argmax(amplitude[search_start:search_stop]))
    detected_frequency = float(frequencies[detected_bin])

    peak_to_peak = float(np.ptp(values))
    fundamental_amplitude = float(max(0.0, amplitude[detected_bin]))
    fundamental_valid = bool(
        fundamental_amplitude > np.finfo(np.float64).eps
        and fundamental_amplitude > 1e-9 * max(1.0, peak_to_peak)
    )

    warnings: list[str] = []
    frequency_error = abs(detected_frequency - fundamental_frequency_hz)
    if frequency_error > max(bin_width * 0.75, 0.01):
        warnings.append(
            f"Detected fundamental {detected_frequency:.3f} Hz differs from "
            f"requested {fundamental_frequency_hz:.3f} Hz."
        )
    if not is_integer_cycle:
        warnings.append(
            f"Record spans {cycle_count:.3f} fundamental cycles; incomplete cycles "
            "can add FFT leakage and should be read as an approximate result."
        )
    if not fundamental_valid:
        warnings.append("Fundamental component is too small or invalid; THD is not meaningful.")

    effective_max_harmonic = min(
        max_harmonic,
        int(np.floor((sampling_frequency_hz / 2.0) / detected_frequency)),
    )
    if effective_max_harmonic < 1:
        raise ValueError("Sampling frequency is too low to measure the first harmonic.")
    if max_harmonic >= 2 and effective_max_harmonic < 2:
        raise ValueError(
            "Sampling frequency is too low to measure the requested harmonic band."
        )
    effective_max_harmonic = max(1, effective_max_harmonic)
    orders = np.arange(1, effective_max_harmonic + 1, dtype=np.int64)
    bins = np.rint(orders * detected_frequency / bin_width).astype(np.int64)
    bins = np.clip(bins, 0, len(amplitude) - 1)
    return HarmonicSpectrum(
        frequencies_hz=frequencies[bins],
        amplitudes=amplitude[bins],
        harmonic_orders=orders,
        fundamental_amplitude=fundamental_amplitude,
        fundamental_frequency_hz=detected_frequency,
        fundamental_valid=fundamental_valid,
        cycle_count=cycle_count,
        is_integer_cycle=is_integer_cycle,
        analysis_warning=" ".join(warnings),
    )
