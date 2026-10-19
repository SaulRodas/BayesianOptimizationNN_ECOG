"""s05 — Line-noise rejection at 60/120/180 Hz."""

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from ...descriptive import welch_psd
from .. import plotting as pl
from ..context import section

STAGES = [("Cruda (conservados)", pl.MUTED), ("Tras CMR", pl.SLOTS[1]), ("Tras CMR + FIR", pl.KEPT)]


@section("s05_notch", "Rechazo de línea", "Notch FIR en 60/120/180 Hz, su ancho y su estacionariedad")
def run(ctx, subjects):
    lines = ctx.config.data["signal"]["notch_hz"]
    zoom_lines = list(lines) + [240.0]
    zoom_s = ctx.eda["line_zoom_psd_window_seconds"]

    # 05a — high-resolution spectra around each line frequency.
    fig, axes = pl.figure(len(zoom_lines), len(subjects), width=4.2, height=2.3)
    for col, (subject, b) in enumerate(subjects.items()):
        spectra = [welch_psd(data, b.fs, zoom_s, ctx.eda["psd_overlap_fraction"])
                   for data in (b.kept, b.cmr, b.filtered)]
        for row, line in enumerate(zoom_lines):
            ax = axes[row, col]
            for (label, color), (freq, psd) in zip(STAGES, spectra):
                mask = np.abs(freq - line) <= 5
                ax.plot(freq[mask], np.median(psd[:, mask], axis=0), color=color, lw=1.0, label=label)
            ax.set_yscale("log")
            ax.set_title(f"{subject} · {line:g} Hz")
            if col == 0:
                ax.set_ylabel("PSD mediana")
            if row == len(zoom_lines) - 1:
                ax.set_xlabel("Frecuencia (Hz)")
    axes[0, 0].legend(loc="upper left")
    ctx.save_fig(fig, "s05_line_zoom", f"Espectro de alta resolución ({1 / zoom_s:.1f} Hz) alrededor de 60/120/180/240 Hz en cada etapa")

    # 05b — stationarity of line noise over the record.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        nper = int(round(ctx.eda["psd_window_seconds"] * b.fs))
        freq, times, sxx = spectrogram(b.kept, fs=b.fs, window="hann", nperseg=nper, noverlap=nper // 2,
                                       detrend="constant", axis=-1)
        mask = (freq >= 50) & (freq <= 70)
        log_power = np.median(10 * np.log10(np.maximum(sxx[:, mask], 1e-300)), axis=0)
        image = ax.pcolormesh(times, freq[mask], log_power, cmap=pl.SEQUENTIAL, shading="nearest")
        ax.grid(False)
        ax.set(title=pl.subject_title(subject, "crudo, mediana de canales"), xlabel="Tiempo original (s)",
               ylabel="Frecuencia (Hz)")
        fig.colorbar(image, ax=ax, label="dB")
    ctx.save_fig(fig, "s05_line_stationarity", "Espectrograma 50–70 Hz: la línea de 60 Hz es estable en el tiempo (un notch fijo basta)")

    # 05c — line fraction per channel at each stage.
    rows = []
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        freq, raw_psd = b.raw_psd
        stages = [raw_psd[np.array(b.cmap.kept)], b.cmr_psd[1], b.filtered_psd[1]]
        fractions = [b.line_fraction(freq, psd) for psd in stages]
        for (label, color), values in zip(STAGES, fractions):
            ax.scatter(b.cmap.kept, values, s=10, color=color, label=label)
        ax.set(yscale="log", title=subject, xlabel="Canal (0-based)")
        for k, channel in enumerate(b.cmap.kept):
            rows.append({"subject": subject, "channel_0based": channel, "raw": fractions[0][k],
                         "after_cmr": fractions[1][k], "after_cmr_fir": fractions[2][k]})
    axes[0, 0].set_ylabel("Fracción de potencia 40–200 Hz en ±1 Hz de la red")
    axes[0, 0].legend(loc="lower left")
    table = pd.DataFrame(rows)
    ctx.save_table(table, "line_fraction", "Fracción de línea por canal: cruda, tras CMR y tras CMR+FIR")
    ctx.save_fig(fig, "s05_line_fraction", "Fracción de línea por canal en cada etapa (S2 parte de ~0.7)")

    # 05d — notch response around each line and residual at 240 Hz.
    spec = next(iter(subjects.values())).spec
    fig, axes = pl.figure(1, len(zoom_lines), width=3.2, height=3.0, sharey=True)
    for ax, line in zip(axes[0], zoom_lines):
        f = np.linspace(line - 6, line + 6, 1201)
        ax.plot(f, spec.notch.gain_db(f), color=pl.SLOTS[1], lw=1.0, label="notch")
        ax.plot(f, spec.fir.gain_db(f), color=pl.INK, lw=1.0, label="combinado")
        ax.set(title=f"{line:g} Hz · combinado {spec.fir.gain_db([line])[0]:.0f} dB", xlabel="Frecuencia (Hz)",
               ylim=(-110, 5))
    axes[0, 0].set_ylabel("Ganancia (dB)")
    axes[0, 0].legend(loc="lower left")
    ctx.save_fig(fig, "s05_notch_response", "Respuesta del notch (rechazo ±1 Hz, transición 2 Hz) y del kernel combinado; 240 Hz queda fuera por el pasabanda")
