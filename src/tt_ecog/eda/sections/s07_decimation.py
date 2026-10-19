"""s07 — Decimation of Morlet power to 100 Hz with a real antialias filter."""

import numpy as np
import pandas as pd
from scipy.signal import coherence

from ... import preprocessing as pp
from ...descriptive import welch_psd
from .. import comparators
from ... import plotting as pl
from ._common import empty_panel

TARGET_FREQS = (40, 80, 140, 200)


def power_envelopes(b, seconds: float):
    window = b.excerpt(seconds)
    channels = np.linspace(0, b.cmap.n_kept - 1, min(5, b.cmap.n_kept)).astype(int)
    index = [int(np.argmin(np.abs(b.spec.freqs - f))) for f in TARGET_FREQS]
    power = pp.morlet_power(b.filtered[channels, window], b.fs, b.spec.freqs[index], b.spec.n_cycles, True)
    edge = int(pp.morlet_half_support(b.spec.freqs[index], b.spec.n_cycles, b.fs).max())
    return power[:, :, edge:power.shape[-1] - edge], index


def draw(ctx, subjects):
    seconds = ctx.eda["correlation_excerpt_seconds"]
    nyquist = next(iter(subjects.values())).spec.target_sfreq / 2
    rows = []

    # 07a — spectrum of the power envelope at 1 kHz.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        power, index = b.power_envelopes
        for k, f_index in enumerate(index):
            freq, psd = welch_psd(power[:, k], b.fs, ctx.eda["psd_window_seconds"], ctx.eda["psd_overlap_fraction"])
            psd = np.median(psd, axis=0)
            psd = psd / psd[1:].max()
            ax.plot(freq[1:], psd[1:], color=pl.SLOTS[k], lw=1.0, label=f"{b.spec.freqs[f_index]:.0f} Hz")
            rows.append({"subject": subject, "morlet_frequency_hz": float(b.spec.freqs[f_index]),
                         "envelope_power_fraction_above_target_nyquist": float(psd[freq > nyquist].sum() / psd[1:].sum())})
        f_aa, h_aa = b.spec.antialias.response()
        ax.plot(f_aa[1:], np.abs(h_aa[1:]) ** 2, color=pl.INK, lw=0.8, label="|H|² antialias")
        ax.axvline(nyquist, color=pl.MUTED, lw=0.8)
        ax.set(xscale="log", yscale="log", ylim=(1e-8, 2), xlim=(0.5, 500), title=subject,
               xlabel="Frecuencia de la envolvente (Hz)")
        ax.legend(loc="lower left")
    axes[0, 0].set_ylabel("PSD normalizada de la potencia")
    ctx.save_table(pd.DataFrame(rows), "s07_envelope_bandwidth", "Fracción de energía de la envolvente por encima del Nyquist de 100 Hz")
    ctx.save_fig(fig, "s07_envelope_spectrum", "La envolvente de potencia a 140–200 Hz tiene energía por encima de 50 Hz: hace falta antialias")

    # 07b — naive slicing vs antialiased decimation.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        power, index = b.power_envelopes
        envelope = power[:, -1]
        factor = b.spec.factor
        f_src, p_src = welch_psd(envelope, b.fs, ctx.eda["psd_window_seconds"], 0.5)
        for label, data, color in (("Submuestreo directo [::10]", b.direct_envelope, pl.COMPARATOR),
                                   ("Antialias + decimación", b.antialiased_envelope, pl.KEPT)):
            f, p = welch_psd(data, b.spec.target_sfreq, ctx.eda["psd_window_seconds"], 0.5)
            ax.plot(f[1:], np.median(p, axis=0)[1:], color=color, lw=1.1, label=label)
        keep = (f_src > 0) & (f_src <= nyquist)
        ax.plot(f_src[keep], np.median(p_src, axis=0)[keep], color=pl.MUTED, lw=1.0, label="Referencia a 1 kHz (0–50 Hz)")
        ax.set(xscale="log", yscale="log", title=pl.subject_title(subject, f"{b.spec.freqs[index[-1]]:.0f} Hz"),
               xlabel="Frecuencia (Hz)")
        ax.legend(loc="lower left")
    axes[0, 0].set_ylabel("PSD de la potencia")
    ctx.save_fig(fig, "s07_naive_vs_antialias", "El submuestreo directo pliega energía en 0–50 Hz; la decimación con antialias sigue a la referencia")

    # 07c — information content: coherence between high-gamma power and flexion.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        y100, _ = b.glove100
        valid = b.valid100
        if valid.sum() < 256:
            empty_panel(ax, "Muestras válidas insuficientes")
            continue
        x, y = b.hg100[:, valid], y100[:, valid]
        nper = min(512, x.shape[1] // 2)
        for finger, color in enumerate(pl.FINGER_COLORS):
            f, coh = b.glove_coherence[finger]
            ax.plot(f, np.median(coh, axis=0), color=color, lw=1.0, label=pl.FINGERS[finger])
        ax.axvline(b.signal["glove"]["effective_fs_hz"] / 2, color=pl.MUTED, lw=0.8)
        ax.set(xscale="log", title=pl.subject_title(subject, "mediana de canales"), xlabel="Frecuencia (Hz)")
        ax.legend(loc="upper right")
    axes[0, 0].set_ylabel("Coherencia alta gamma–flexión")
    ctx.save_fig(fig, "s07_coherence", "La relación alta gamma–flexión se concentra en pocos Hz (línea: Nyquist del guante 12.5 Hz); 100 Hz sobra")

    # 07d — antialias response and negative values after decimation.
    spec = next(iter(subjects.values())).spec
    fig, axes = pl.figure(1, 2, width=5.6, height=3.4)
    f, h = spec.antialias.response()
    axes[0, 0].plot(f, 20 * np.log10(np.maximum(np.abs(h), 1e-12)), color=pl.INK, lw=1.0)
    params = spec.antialias.params
    for edge in (params["passband_hz"], params["stopband_hz"]):
        axes[0, 0].axvline(edge, color=pl.MUTED, lw=0.8)
    axes[0, 0].set(xlim=(0, 150), ylim=(-100, 5), xlabel="Frecuencia (Hz)", ylabel="Ganancia (dB)",
                   title=f"Antialias Kaiser ({spec.antialias.n_taps} taps, ±{spec.antialias.half_support / spec.sfreq * 1000:.0f} ms)")
    names = list(subjects)
    fractions = [100 * b.power_report["negative_values"] / b.power_report["total_values"] for b in subjects.values()]
    axes[0, 1].bar(names, fractions, color=pl.KEPT, width=0.5)
    axes[0, 1].set(ylabel="% de valores < 0 tras decimar", title="Rizado FIR (se reporta, no se recorta)")
    axes[0, 1].grid(axis="x", visible=False)
    ctx.save_fig(fig, "s07_antialias", "Respuesta del antialias y fracción de potencia negativa tras decimar")


SECTION_NAME = 's07_decimation'
TITLE = 'Decimación a 100 Hz'
JUSTIFIES = 'Potencia Morlet a 100 Hz con antialias FIR (no submuestreo directo)'


def prepare_subject(b, config):
    """CLI adapter for the calculations shown in notebook 01."""
    b.power_envelopes = power_envelopes(b, config.data['eda']['correlation_excerpt_seconds'])
    envelope = b.power_envelopes[0][:, -1]
    b.direct_envelope = comparators.naive_slice(envelope, b.spec.factor)
    b.antialiased_envelope = pp.decimate(envelope, b.spec.antialias, b.spec.factor)
    x, y = b.hg100[:, b.valid100], b.glove100[0][:, b.valid100]
    b.glove_coherence = [coherence(x, y[finger][None, :], fs=b.spec.target_sfreq,
                                 nperseg=min(512, x.shape[1] // 2), axis=-1) for finger in range(5)]
