"""s08 — Glove: recover the effective 25 Hz samples and interpolate to 100 Hz."""

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from ...descriptive import welch_psd
from .. import comparators
from .. import plotting as pl
from ..context import section
from ._common import empty_panel


@section("s08_glove", "Guante de datos", "Recuperar las muestras de 25 Hz e interpolar con spline cúbico a 100 Hz")
def run(ctx, subjects):
    # 08a — storage pattern; saved BEFORE recovery so a failed contract still leaves evidence.
    rows = []
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=3.0)
    for col, (subject, b) in enumerate(subjects.items()):
        info = b.glove_inspection
        rows.append({"subject": subject, "identified": info["identified"], "method": info["method"],
                     "best_candidate": info["best_candidate_method"], "phase": info["phase"],
                     "relative_rms_error": info["relative_rms_error"], "tolerance": info["tolerance"],
                     "change_fraction_at_phase": info["change_fraction_at_phase"],
                     "storage_hz": b.glove_record.metadata["sampling_hz_glove_storage"]})
        ax = axes[0, col]
        if info["errors_by_method_phase"]:
            for k, (method, errors) in enumerate(info["errors_by_method_phase"].items()):
                ax.plot(np.arange(len(errors)), np.maximum(errors, 1e-17), color=pl.SLOTS[k], lw=1.0,
                        marker="o", ms=2.5, label=method)
            ax.axvline(info["phase"], color=pl.INK, lw=0.8)
            ax.set(yscale="log", xlabel="Fase (filas de 1 kHz)", ylabel="Error RMS relativo",
                   title=f"{subject} · {info['method']} · fase {info['phase']}")
            ax.legend(loc="lower right")
        else:
            empty_panel(ax, "Guante ya almacenado a 25 Hz")
        ax = axes[1, col]
        lengths = info["run_lengths"]
        if len(lengths):
            ax.hist(np.clip(lengths, 0, 400), bins=np.arange(0, 404, 4), color=pl.KEPT)
            for multiple in range(40, 401, 40):
                ax.axvline(multiple, color=pl.MUTED, lw=0.6)
            ax.set(yscale="log", xlabel="Longitud de racha de valores idénticos (muestras)", ylabel="Conteo",
                   title="Rachas en la señal almacenada")
        else:
            empty_panel(ax, "Sin rachas que analizar")
    ctx.save_table(pd.DataFrame(rows), "glove_storage", "Patrón de almacenamiento del guante (método, fase, error)")
    ctx.save_fig(fig, "s08_glove_storage", "Identificación del patrón 25 Hz → 1 kHz: error de reconstrucción por fase y rachas múltiplos de 40")

    # 08b — stored signal, recovered samples and interpolations (2 s around peak speed).
    fig, axes = pl.figure(5, len(subjects), width=4.6, height=1.7, sharex="col")
    for col, (subject, b) in enumerate(subjects.items()):
        y25, t25 = b.glove25
        y100, _ = b.glove100
        peak = float(t25[np.argmax(b.movement.speed.sum(axis=0))])
        lo, hi = peak - 1.0, peak + 1.0
        rows_1k = slice(max(0, int(lo * b.fs)), min(b.glove_record.values.shape[1], int(hi * b.fs)))
        t1k = np.arange(rows_1k.start, rows_1k.stop) / b.glove_record.metadata["sampling_hz_glove_storage"]
        in25 = (t25 >= lo) & (t25 <= hi)
        in100 = (b.t100 >= lo) & (b.t100 <= hi)
        lin = comparators.linear(t25, y25, b.t100[in100])
        for finger in range(5):
            ax = axes[finger, col]
            if b.glove_record.metadata["sampling_hz_glove_storage"] != b.glove_record.metadata["sampling_hz_glove_effective"]:
                ax.plot(t1k, b.glove_record.values[finger, rows_1k], color=pl.MUTED, lw=0.8, label="Almacenado 1 kHz")
            ax.plot(b.t100[in100], lin[finger], color=pl.COMPARATOR, lw=0.9, label="Lineal (contraste)")
            ax.plot(b.t100[in100], y100[finger, in100], color=pl.KEPT, lw=1.1, label="Cúbico 100 Hz")
            ax.plot(t25[in25], y25[finger, in25], "o", color=pl.INK, ms=3, label="Muestras 25 Hz")
            ax.set_ylabel(pl.FINGERS[finger])
            if finger == 0:
                ax.set_title(subject)
        axes[-1, col].set_xlabel("Tiempo original (s)")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    ctx.save_fig(fig, "s08_glove_interpolation", "Muestras efectivas de 25 Hz y su interpolación cúbica a 100 Hz (2 s alrededor de la máxima velocidad)")

    # 08c — glove spectrum.
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=3.0)
    fs25 = None
    for col, (subject, b) in enumerate(subjects.items()):
        y25, _ = b.glove25
        fs25 = b.signal["glove"]["effective_fs_hz"]
        freq, psd = welch_psd(y25, fs25, min(10.24, y25.shape[1] / fs25 / 2), 0.5)
        for finger, color in enumerate(pl.FINGER_COLORS):
            axes[0, col].plot(freq[1:], psd[finger, 1:], color=color, lw=1.0, label=pl.FINGERS[finger])
            cumulative = np.cumsum(psd[finger, 1:]) / psd[finger, 1:].sum()
            axes[1, col].plot(freq[1:], cumulative, color=color, lw=1.0, label=pl.FINGERS[finger])
        axes[0, col].set(yscale="log", xscale="log", title=subject, ylabel="PSD flexión")
        axes[1, col].set(xscale="log", ylim=(0, 1.02), xlabel="Frecuencia (Hz)", ylabel="Potencia acumulada")
        for ax in axes[:, col]:
            ax.axvline(fs25 / 2, color=pl.MUTED, lw=0.8)
    axes[0, 0].legend(loc="lower left")
    ctx.save_fig(fig, "s08_glove_spectrum", "La flexión concentra su potencia en pocos Hz, muy por debajo del Nyquist de 12.5 Hz")

    # 08d — hold out every other 25 Hz sample and compare interpolators.
    rows = []
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.2, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        y25, t25 = b.glove25
        t_known, y_known = t25[::2], y25[:, ::2]
        target = (t25 > t_known[0]) & (t25 < t_known[-1])
        target[::2] = False
        scale = np.maximum(y25.std(axis=1), 1e-12)
        cubic, _ = pp.interpolate_glove(t_known, y_known, t25[target])
        estimates = {"Retención (ZOH)": comparators.zero_order_hold(t_known, y_known, t25[target]),
                     "Lineal": comparators.linear(t_known, y_known, t25[target]),
                     "Cúbico (pipeline)": cubic}
        width = 0.25
        for k, (name, estimate) in enumerate(estimates.items()):
            rmse = np.sqrt(np.mean((estimate - y25[:, target]) ** 2, axis=1)) / scale
            ax.bar(np.arange(5) + (k - 1) * width, rmse, width=width * 0.9,
                   color=[pl.MUTED, pl.COMPARATOR, pl.KEPT][k], label=name)
            for finger in range(5):
                rows.append({"subject": subject, "finger": pl.FINGERS[finger], "method": name,
                             "rmse_over_std": float(rmse[finger])})
        ax.set_xticks(np.arange(5), pl.FINGERS, rotation=30)
        ax.set(title=subject)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("RMSE / STD del dedo")
    axes[0, 0].legend(loc="upper left")
    ctx.save_table(pd.DataFrame(rows), "s08_interpolation_holdout", "Error de interpolación al ocultar muestras alternas de 25 Hz")
    ctx.save_fig(fig, "s08_interpolation_holdout", "Validación por muestras ocultas: el spline cúbico reconstruye mejor que lineal y retención")
