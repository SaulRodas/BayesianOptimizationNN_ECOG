"""s04 — Analysis band 40–200 Hz."""

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from ...descriptive import welch_psd
from ... import plotting as pl
from ._common import empty_panel


def movement_rest_ratio(b, eda) -> tuple[np.ndarray, np.ndarray, int, int]:
    """log10 PSD(movement windows) - log10 PSD(rest windows) per channel."""
    window = int(round(eda["window_seconds"] * b.fs))
    per_target = b.spec.factor
    moving = b.movement.moving_target
    log_move, log_rest = None, None
    n_move = n_rest = 0
    edge = b.spec.fir.half_support
    freq = None
    for start in range(edge, b.n_samples - window - edge + 1, window):
        fraction = moving[start // per_target:(start + window) // per_target].mean()
        if 0.05 <= fraction <= 0.5:
            continue
        freq, psd = welch_psd(b.cmr[:, start:start + window], b.fs, eda["window_seconds"] / 2, 0.5)
        logp = np.log10(np.maximum(psd, 1e-300))
        if fraction > 0.5:
            log_move = logp if log_move is None else log_move + logp
            n_move += 1
        else:
            log_rest = logp if log_rest is None else log_rest + logp
            n_rest += 1
    if not n_move or not n_rest:
        return freq, None, n_move, n_rest
    return freq, log_move / n_move - log_rest / n_rest, n_move, n_rest


def event_related_power(b, eda) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Median-over-channels and illustrative-channel ERSP (dB) around onsets."""
    diag = eda["diagnostic_morlet"]
    movement = eda["movement"]
    freqs = pp.morlet_frequencies(diag["freq_range_hz"], diag["n_frequencies"])
    fs_t = b.spec.target_sfreq
    lo = int(round(movement["ersp_window_seconds"][0] * fs_t))
    hi = int(round(movement["ersp_window_seconds"][1] * fs_t))
    b0 = int(round(movement["baseline_seconds"][0] * fs_t)) - lo
    b1 = int(round(movement["baseline_seconds"][1] * fs_t)) - lo
    edge = max(b.edge100, int(np.ceil(pp.morlet_half_support(freqs[:1], diag["n_cycles"], b.fs)[0] / b.spec.factor)))
    n_t = len(b.t100)
    centers = [int(round(t * fs_t)) for t in b.movement.onsets]
    centers = [c for c in centers if c + lo >= edge and c + hi < n_t - edge]
    lags = np.arange(lo, hi + 1) / fs_t
    if not centers:
        return freqs, lags, None, None, 0
    ersp = np.empty((b.cmap.n_kept, freqs.size, lags.size))
    for channel in range(b.cmap.n_kept):
        power, _ = pp.morlet_power_decimated(b.cmr[channel:channel + 1], b.spec, freqs, diag["n_cycles"])
        power = power[0].astype(np.float64)
        # ERSP = epoch-averaged power over its baseline. Decimated power can dip slightly
        # below zero sample-wise because of FIR ripple; epoch averages do not.
        mean = np.stack([power[:, c + lo:c + hi + 1] for c in centers]).mean(axis=0)
        baseline = mean[:, b0:b1].mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ersp[channel] = np.where((mean > 0) & (baseline > 0), 10 * np.log10(mean / baseline), np.nan)
    return freqs, lags, np.nanmedian(ersp, axis=0), ersp[b.descriptive_channel], len(centers)


def draw(ctx, subjects):
    band = ctx.config.data["signal"]["bandpass_hz"]
    lines = ctx.config.data["signal"]["notch_hz"]

    # 04a — full-range spectrum after CMR.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        freq, psd = b.cmr_psd
        mask = (freq >= 1) & (freq <= 500)
        ax.fill_between(freq[mask], np.percentile(psd[:, mask], 10, axis=0), np.percentile(psd[:, mask], 90, axis=0),
                        color=pl.KEPT, alpha=0.2, lw=0, label="p10–p90 canales")
        ax.plot(freq[mask], np.median(psd[:, mask], axis=0), color=pl.KEPT, lw=1.2, label="Mediana")
        pl.shade_band(ax, band)
        pl.mark_lines(ax, lines)
        ax.set(xscale="log", yscale="log", title=pl.subject_title(subject, "tras CMR"), xlabel="Frecuencia (Hz)")
        ax.legend(loc="lower left")
    axes[0, 0].set_ylabel("PSD (u.o.²/Hz)")
    ctx.save_fig(fig, "s04_full_spectrum", "Espectro 1–500 Hz tras CMR: caída 1/f y límite de adquisición cerca de 200 Hz")

    # 04b — movement vs rest spectral change (development labels, descriptive).
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        freq, ratio, n_move, n_rest = b.movement_rest_spectrum
        if ratio is None:
            empty_panel(ax, f"Ventanas insuficientes (mov {n_move}, reposo {n_rest})")
            continue
        mask = (freq >= 2) & (freq <= 500)
        ax.plot(freq[mask], 10 * np.median(ratio[:, mask], axis=0), color=pl.KEPT, lw=1.2, label="Mediana canales")
        ax.plot(freq[mask], 10 * np.percentile(ratio[:, mask], 75, axis=0), color=pl.SLOTS[1], lw=1.0,
                label="Percentil 75 canales")
        ax.axhline(0, color=pl.AXIS, lw=0.8)
        pl.shade_band(ax, band)
        ax.set(xscale="log", title=pl.subject_title(subject, f"{n_move} mov / {n_rest} reposo"),
               xlabel="Frecuencia (Hz)")
        ax.legend(loc="upper left")
    axes[0, 0].set_ylabel("Movimiento − reposo (dB)")
    ctx.save_fig(fig, "s04_movement_rest", "Cambio espectral movimiento vs reposo (etiquetas de desarrollo, descriptivo): aumento en alta gamma")

    # 04c — ERSP around movement onsets on a wide diagnostic grid.
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=3.2)
    rows = []
    for col, (subject, b) in enumerate(subjects.items()):
        freqs, lags, median, illustrative, n_epochs = b.event_related_power
        if median is None:
            empty_panel(axes[0, col], "Sin inicios de movimiento utilizables")
            empty_panel(axes[1, col], "Sin inicios de movimiento utilizables")
            continue
        for row, (data, label) in enumerate(((median, "mediana de canales"),
                                             (illustrative, f"canal ilustrativo c{b.cmap.kept[b.descriptive_channel]}"))):
            ax = axes[row, col]
            limit = max(float(np.nanpercentile(np.abs(data), 99)), 0.5)
            image = ax.pcolormesh(lags, freqs, data, cmap=pl.DIVERGING, vmin=-limit, vmax=limit, shading="nearest")
            ax.set_yscale("log")
            for edge in band:
                ax.axhline(edge, color=pl.INK, lw=0.8)
            ax.axvline(0, color=pl.INK, lw=0.8)
            ax.grid(False)
            ax.set(title=f"{subject} · {label} · {n_epochs} inicios", xlabel="Tiempo desde el inicio (s)",
                   ylabel="Frecuencia (Hz)")
            fig.colorbar(image, ax=ax, label="dB vs línea base")
        hg = (freqs >= band[0]) & (freqs <= band[1])
        low = freqs < band[0]
        post = lags >= 0
        rows.append({"subject": subject, "n_onsets": n_epochs,
                     "median_ersp_db_40_200_post": float(np.nanmean(median[hg][:, post])),
                     "median_ersp_db_below_40_post": float(np.nanmean(median[low][:, post]))})
    ctx.save_table(pd.DataFrame(rows), "s04_ersp_summary", "ERSP medio posterior al inicio: 40–200 Hz frente a <40 Hz")
    ctx.save_fig(fig, "s04_ersp", "ERSP alrededor de inicios de movimiento (4–250 Hz): la respuesta se concentra en 40–200 Hz (líneas negras)")

    # 04d — bandpass design.
    spec = next(iter(subjects.values())).spec
    fig, axes = pl.figure(1, 2, width=5.6, height=3.4)
    for design, color in ((spec.bandpass, pl.SLOTS[0]), (spec.notch, pl.SLOTS[1]), (spec.fir, pl.INK)):
        f, h = design.response()
        axes[0, 0].plot(f, 20 * np.log10(np.maximum(np.abs(h), 1e-12)), color=color, lw=1.0,
                        label=f"{design.name} ({design.n_taps} taps)")
    pl.shade_band(axes[0, 0], band)
    axes[0, 0].set(ylim=(-100, 5), xlabel="Frecuencia (Hz)", ylabel="Ganancia (dB)", title="Respuesta en frecuencia")
    axes[0, 0].legend(loc="lower right")
    t_ms = (np.arange(spec.fir.n_taps) - spec.fir.half_support) / spec.sfreq * 1000
    axes[0, 1].plot(t_ms, spec.fir.taps, color=pl.INK, lw=0.8)
    axes[0, 1].set(xlabel="Tiempo (ms)", ylabel="Coeficiente",
                   title=f"Respuesta al impulso combinada (±{spec.fir.half_support / spec.sfreq * 1000:.0f} ms)")
    ctx.save_fig(fig, "s04_filter_design", "Diseño FIR: pasabanda 40–200 Hz, notch y kernel combinado (fase cero)")
    ctx.save_json({"bandpass": spec.bandpass.to_json(), "notch": spec.notch.to_json(),
                   "combined": spec.fir.to_json(), "antialias": spec.antialias.to_json()},
                  "filter_designs", "Diseños FIR efectivos (tipo, taps, soporte, fase, hash)")


SECTION_NAME = 's04_band'
TITLE = 'Banda 40–200 Hz'
JUSTIFIES = 'Banda analítica 40–200 Hz y diseño del pasabanda'


def prepare_subject(b, config):
    """CLI adapter for the calculations shown in notebook 01."""
    b.movement_rest_spectrum = movement_rest_ratio(b, config.data['eda'])
    b.event_related_power = event_related_power(b, config.data['eda'])
