"""s11 — Blocked cross-validation and inputs for the (still pending) purge."""

import math

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from ... import plotting as pl


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of each row for lags 0..max_lag (FFT based)."""
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size, axis=-1)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=-1)[..., :max_lag + 1]
    return acf / np.maximum(acf[..., :1], 1e-300)


def draw(ctx, subjects):
    validation = ctx.config.data["validation"]
    stage1 = set(validation["stage1_validation_blocks"])

    # 11a — timeline of blocks, glove and onsets.
    onset_rows = []
    fig, axes = pl.figure(len(subjects), 1, width=13, height=2.4)
    for row, (subject, b) in enumerate(subjects.items()):
        ax = axes[row, 0]
        y100, _ = b.glove100
        step = pl.plot_step(y100.shape[1], ctx.eda["plot_max_points"])
        span = np.nanmax(y100) - np.nanmin(y100) or 1.0
        for finger, color in enumerate(pl.FINGER_COLORS):
            ax.plot(b.t100[::step], (y100[finger, ::step] - np.nanmin(y100[finger])) / span + finger, color=color,
                    lw=0.7, label=pl.FINGERS[finger])
            onsets = b.movement.onsets_by_finger[finger]
            ax.plot(onsets, np.full(onsets.size, finger - 0.15), "|", color=color, ms=4)
        for block in b.blocks:
            start, stop = block.start_source / b.fs, block.stop_source / b.fs
            if block.name in stage1:
                ax.axvspan(start, stop, color=pl.SLOTS[3], alpha=0.12, lw=0)
            ax.axvline(start, color=pl.AXIS, lw=0.8)
            ax.text((start + stop) / 2, 1.0, block.name + (" (val. etapa 1)" if block.name in stage1 else ""),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
            for finger in range(5):
                onsets = b.movement.onsets_by_finger[finger]
                onset_rows.append({"subject": subject, "block": block.name, "finger": pl.FINGERS[finger],
                                   "n_onsets": int(((onsets >= start) & (onsets < stop)).sum())})
        ax.set(ylabel=subject, yticks=[], xlabel="Tiempo original (s)" if row == len(subjects) - 1 else None)
    axes[0, 0].legend(loc="upper right", ncols=5, fontsize=7, bbox_to_anchor=(1, 1.35))
    ctx.save_table(pd.DataFrame(onset_rows), "s11_onsets_by_block", "Inicios de movimiento por dedo y bloque")
    ctx.save_fig(fig, "s11_blocks_timeline", "Bloques cronológicos B1–B5 (sombreado: validación de la etapa 1) con flexión e inicios por dedo")

    # 11b — temporal dependence.
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=3.0, sharex=True)
    window_s = validation["train_window_samples"] / next(iter(subjects.values())).spec.target_sfreq
    for col, (subject, b) in enumerate(subjects.items()):
        fs = b.spec.target_sfreq
        valid = b.valid100
        max_lag = int(min(ctx.eda["autocorr_max_lag_seconds"] * fs, valid.sum() // 2))
        lags = np.arange(max_lag + 1) / fs
        y100, _ = b.glove100
        acf_y = b.glove_autocorrelation
        for finger, color in enumerate(pl.FINGER_COLORS):
            axes[0, col].plot(lags, acf_y[finger], color=color, lw=1.0, label=pl.FINGERS[finger])
        acf_x = b.power_autocorrelation
        axes[1, col].fill_between(lags, np.percentile(acf_x, 10, axis=0), np.percentile(acf_x, 90, axis=0),
                                  color=pl.KEPT, alpha=0.2, lw=0, label="p10–p90 canales")
        axes[1, col].plot(lags, np.median(acf_x, axis=0), color=pl.KEPT, lw=1.1, label="Mediana canales")
        for ax in axes[:, col]:
            ax.axhline(0, color=pl.AXIS, lw=0.8)
            ax.axvline(window_s, color=pl.INK, lw=0.8)
        axes[0, col].set(title=subject, ylabel="Autocorrelación flexión")
        axes[1, col].set(xlabel="Retardo (s)", ylabel="Autocorrelación alta gamma")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[1, 0].legend(loc="upper right", fontsize=7)
    ctx.save_fig(fig, "s11_autocorrelation", f"Dependencia temporal de varios segundos (línea: ventana de {window_s:g} s): exige bloques y purga")

    # 11c — measured support of the full chain vs the purge candidate.
    spec = next(iter(subjects.values())).spec
    result = next(iter(subjects.values())).measured_support
    analytic = result["analytic"]
    candidate = validation["purge_candidate_for_tests"]
    fig, axes = pl.figure(1, 1, width=9, height=3.4)
    ax = axes[0, 0]
    ax.plot(result["lags_target"], np.maximum(result["response"], 1e-18), color=pl.KEPT, lw=1.1,
            label="Respuesta al impulso (máx entre frecuencias)")
    ax.axhline(result["tolerance"], color=pl.MUTED, lw=0.8, label=f"Tolerancia {result['tolerance']:g}")
    for value, color, label in ((analytic["total_half_support_samples_target"], pl.INK, "Soporte analítico"),
                                (spec.delay_samples, pl.SLOTS[1], "Retardo 20"),
                                (candidate, pl.SLOTS[7], f"Candidato de purga {candidate}")):
        ax.axvline(value, color=color, lw=1.0, label=label)
        ax.axvline(-value, color=color, lw=1.0)
    ax.set(yscale="log", ylim=(1e-12, 2), xlim=(-max(candidate, 1) * 1.1, max(candidate, 1) * 1.1),
           xlabel="Muestras a 100 Hz desde el impulso", ylabel="Respuesta normalizada",
           title=f"Soporte medido ±{result['measured_half_support_target']} muestras (analítico ±{analytic['total_half_support_samples_target']})")
    ax.legend(loc="upper right", fontsize=7)
    ctx.save_fig(fig, "s11_chain_support", "Soporte temporal medido de FIR→Morlet→antialias frente al candidato de purga (la purga sigue pendiente)")
    ctx.save_json({"analytic": analytic, "measured_half_support_target": result["measured_half_support_target"],
                   "tolerance": result["tolerance"], "purge_candidate_for_tests": candidate,
                   "delay_samples_target": spec.delay_samples,
                   "window_samples_target": validation["train_window_samples"],
                   "status": "informational; purge_samples_at_100hz remains null until V04/V09",
                   "minimum_gap_hint_target": int(math.ceil(2 * result["measured_half_support_target"] + spec.delay_samples))},
                  "support", "Soporte analítico y medido de la cadena; insumo para fijar la purga en el paso 3")


SECTION_NAME = 's11_validation'
TITLE = 'Validación por bloques'
JUSTIFIES = 'CV temporal por bloques B1–B5 con purga; insumos para fijarla en el paso 3'


def prepare_subject(b, config):
    """CLI adapter for the calculations shown in notebook 01."""
    max_lag = int(min(config.data['eda']['autocorr_max_lag_seconds'] * b.spec.target_sfreq, b.valid100.sum() // 2))
    b.glove_autocorrelation = autocorrelation(b.glove100[0][:, b.valid100], max_lag)
    b.power_autocorrelation = autocorrelation(b.hg100[:, b.valid100], max_lag)
    if b.subject == config.data['project']['subjects'][0]:
        b.measured_support = pp.measured_support(b.spec, config.data['verification']['support_tolerance'])
