"""s06 — Morlet representation: 40 log frequencies, 7 cycles, zero mean."""

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from ... import plotting as pl


def draw(ctx, subjects):
    first = next(iter(subjects.values()))
    spec = first.spec
    freqs, n_cycles, fs = spec.freqs, spec.n_cycles, spec.sfreq
    sigma_t = n_cycles / (2 * np.pi * freqs)
    sigma_f = freqs / n_cycles
    half = pp.morlet_half_support(freqs, n_cycles, fs)
    ctx.save_table(pd.DataFrame({"frequency_hz": freqs, "sigma_t_ms": 1000 * sigma_t, "sigma_f_hz": sigma_f,
                                 "half_support_samples": half, "wavelet_duration_ms": (2 * half + 1) / fs * 1000}),
                   "morlet_frequencies", "Frecuencias Morlet y su resolución tiempo-frecuencia")

    # 06a — wavelet family and tiling.
    fig, axes = pl.figure(1, 4, width=3.6, height=3.2)
    shown = [0, 13, 27, 39]
    for k, index in enumerate(shown):
        w = first.wavelets[index]
        t = (np.arange(w.size) - (w.size - 1) / 2) / fs * 1000
        axes[0, 0].plot(t, np.real(w) / np.abs(w).max() + 2.2 * k, color=pl.SLOTS[k], lw=0.8,
                        label=f"{freqs[index]:.0f} Hz")
    axes[0, 0].set(title="Wavelets (parte real)", xlabel="Tiempo (ms)", yticks=[])
    axes[0, 0].legend(loc="upper right")
    f_axis = np.linspace(20, 260, 2000)
    for k, f0 in enumerate(freqs):
        axes[0, 1].plot(f_axis, np.exp(-((f_axis - f0) ** 2) / (2 * (f0 / n_cycles) ** 2)),
                        color=pl.KEPT if k % 2 == 0 else pl.SLOTS[2], lw=0.6)
    axes[0, 1].set(title="Respuesta gaussiana de las 40 frecuencias", xlabel="Frecuencia (Hz)", ylabel="Ganancia relativa")
    axes[0, 2].plot(freqs, 1000 * sigma_t, color=pl.KEPT, marker="o", ms=3)
    axes[0, 2].set(title="Resolución temporal σt", xlabel="Frecuencia (Hz)", ylabel="σt (ms)")
    axes[0, 3].plot(freqs, sigma_f, color=pl.KEPT, marker="o", ms=3)
    axes[0, 3].set(title="Resolución espectral σf", xlabel="Frecuencia (Hz)", ylabel="σf (Hz)")
    ctx.save_fig(fig, "s06_wavelet_family", "Familia Morlet de 7 ciclos: σt de 28 a 6 ms y σf de 6 a 29 Hz; cobertura continua de 40–200 Hz")

    # 06b — scalogram excerpt with glove (illustrative channel).
    seconds = min(ctx.eda["excerpt_seconds"] * 2, first.duration_s / 2)
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=2.8, sharex="col")
    for col, (subject, b) in enumerate(subjects.items()):
        center = b.movement.onsets[len(b.movement.onsets) // 2] if len(b.movement.onsets) else None
        window = b.excerpt(seconds, center)
        sl = slice(window.start // b.spec.factor, window.stop // b.spec.factor)
        t = b.t100[sl]
        channel = b.descriptive_channel
        raw_power = b.power100[channel, :, sl].astype(np.float64)
        floor = np.percentile(raw_power[raw_power > 0], 0.5) if (raw_power > 0).any() else 1e-30
        power = np.log10(np.maximum(raw_power, floor))  # FIR ripple can leave tiny negatives
        ax = axes[0, col]
        ax.pcolormesh(t, b.spec.freqs, power, cmap=pl.SEQUENTIAL, shading="nearest",
                      vmin=np.percentile(power, 2), vmax=np.percentile(power, 98))
        ax.set_yscale("log")
        ax.grid(False)
        ax.set(title=f"{subject} · c{b.cmap.kept[channel]} (ilustrativo)", ylabel="Frecuencia (Hz)")
        y100, _ = b.glove100
        for finger, color in enumerate(pl.FINGER_COLORS):
            axes[1, col].plot(t, y100[finger, sl], color=color, lw=1.0, label=pl.FINGERS[finger])
        axes[1, col].set(xlabel="Tiempo original (s)", ylabel="Flexión (u.o.)")
    axes[1, 0].legend(loc="upper left", ncols=2)
    ctx.save_fig(fig, "s06_scalogram", "Escalograma log-potencia (100 Hz) de un canal ilustrativo y flexión de desarrollo (descriptivo)")

    # 06c — zero-mean property and edge effect on an independent excerpt.
    fig, axes = pl.figure(1, 2, width=5.6, height=3.4)
    with_zero, without_zero, reference, interior = first.offset_power
    for label, power, color in (("zero_mean=True", with_zero, pl.KEPT), ("zero_mean=False", without_zero, pl.SLOTS[1])):
        rel = np.abs(power[:, :, interior] - reference[:, :, interior]).max(axis=(0, 2)) / reference[:, :, interior].mean(axis=(0, 2))
        axes[0, 0].plot(freqs, np.maximum(rel, 1e-18), color=color, lw=1.2, label=label)
    axes[0, 0].set(yscale="log", xlabel="Frecuencia (Hz)", ylabel="Cambio relativo de potencia",
                   title="Respuesta a un offset DC de 50 STD")
    axes[0, 0].legend(loc="upper right")
    # Edge effect averaged over many independent 1 s segments of the filtered record.
    profile, centre, starts, seg = first.edge_power
    distance_ms = np.arange(seg // 2) / fs * 1000
    for k, index in enumerate((0, 20, 39)):
        axes[0, 1].plot(distance_ms, profile[index, :seg // 2] / centre[index], color=pl.SLOTS[k], lw=1.1,
                        label=f"{freqs[index]:.0f} Hz (semisoporte {half[index] / fs * 1000:.0f} ms)")
        axes[0, 1].axvline(half[index] / fs * 1000, color=pl.SLOTS[k], lw=0.6)
    axes[0, 1].axhline(1, color=pl.AXIS, lw=0.8)
    axes[0, 1].set(xlabel="Distancia al borde del segmento (ms)", ylabel="Potencia media / interior del segmento",
                   title=f"Efecto de borde (promedio de {len(starts)} segmentos de 1 s)")
    axes[0, 1].legend(loc="lower right")
    ctx.save_fig(fig, "s06_zero_mean_edges", "Con 7 ciclos y f >= 40 Hz, zero_mean apenas cambia la respuesta a DC (la fuga residual viene del truncado a ±5σ y el pasabanda ya elimina el DC); los bordes se atenúan hasta el semisoporte y se enmascaran")


SECTION_NAME = 's06_morlet'
TITLE = 'Transformada Morlet'
JUSTIFIES = '40 frecuencias log 40–200 Hz, 7 ciclos, zero_mean y máscaras de borde'


def offset_power(b, eda):
    fs, freqs, n_cycles = b.fs, b.spec.freqs, b.spec.n_cycles
    half = pp.morlet_half_support(freqs, n_cycles, fs)
    window = b.excerpt(eda["excerpt_seconds"])
    channels = np.linspace(0, b.cmap.n_kept - 1, min(8, b.cmap.n_kept)).astype(int)
    segment = b.cmr[channels, window]
    offset = segment + 50 * segment.std()
    with_zero = pp.morlet_power(offset, fs, freqs, n_cycles, True)
    without_zero = pp.morlet_power(offset, fs, freqs, n_cycles, False)
    reference = pp.morlet_power(segment, fs, freqs, n_cycles, True)
    interior = pp.valid_slice(segment.shape[1], int(half.max()))
    return with_zero, without_zero, reference, interior


def edge_power(b):
    fs, freqs, n_cycles = b.fs, b.spec.freqs, b.spec.n_cycles
    half = pp.morlet_half_support(freqs, n_cycles, fs)
    channels = np.linspace(0, b.cmap.n_kept - 1, min(8, b.cmap.n_kept)).astype(int)
    seg = int(round(fs))
    interior_rec = pp.valid_slice(b.n_samples, b.spec.fir.half_support)
    starts = np.arange(interior_rec.start, interior_rec.stop - seg, seg)[:60]
    profile = np.zeros((freqs.size, seg))
    for start in starts:
        profile += pp.morlet_power(b.filtered[channels[:4], start:start + seg], fs, freqs, n_cycles, True).mean(axis=0)
    profile /= max(len(starts), 1)
    centre = profile[:, int(half.max()):seg - int(half.max())].mean(axis=1, keepdims=True)
    return profile, centre, starts, seg


def prepare_subject(b, config):
    """CLI adapter for the calculations shown in notebook 01."""
    from mne.time_frequency import morlet
    if b.subject == config.data['project']['subjects'][0]:
        b.wavelets = morlet(b.fs, b.spec.freqs, n_cycles=b.spec.n_cycles, zero_mean=True)
        b.offset_power = offset_power(b, config.data['eda'])
        b.edge_power = edge_power(b)
