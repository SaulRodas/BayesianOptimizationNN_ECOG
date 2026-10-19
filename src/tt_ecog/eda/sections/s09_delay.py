"""s09 — Fixed 200 ms delay and its direction: ECoG(t) -> flexion(t + 0.2 s)."""

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from ... import plotting as pl
from ._common import empty_panel, pearson_rows


def lagged_correlation(b, max_lag: int, min_lag: int):
    """Pearson |r| between high-gamma power (C) and flexion (5) per lag, via align_delay.

    Positive lag L pairs ECoG(t) with flexion(t + L): ECoG leads.
    """
    y100, _ = b.glove100
    valid = b.valid100
    lags = np.arange(min_lag, max_lag + 1)
    corr = np.full((lags.size, b.cmap.n_kept, 5), np.nan)
    for k, lag in enumerate(lags):
        if lag >= 0:
            x, y, xi, yi = pp.align_delay(b.hg100, y100, int(lag))
        else:
            y, x, yi, xi = pp.align_delay(y100, b.hg100, int(-lag))
        keep = valid[xi] & valid[yi]
        if keep.sum() > 10:
            corr[k] = pearson_rows(x[:, keep], y[:, keep])
    return lags, corr


def draw(ctx, subjects):
    lag_range = ctx.eda["lag_range_seconds"]
    rows, peaks = [], []
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=3.2)
    for col, (subject, b) in enumerate(subjects.items()):
        fs = b.spec.target_sfreq
        lags, corr = b.lagged_correlation
        seconds = lags / fs
        delay_s = b.spec.delay_samples / fs
        if np.isnan(corr).all():
            empty_panel(axes[0, col], "Muestras válidas insuficientes")
            empty_panel(axes[1, col], "Muestras válidas insuficientes")
            continue
        abs_corr = np.abs(corr)
        ax = axes[0, col]
        for finger, color in enumerate(pl.FINGER_COLORS):
            curve = np.nanmedian(abs_corr[:, :, finger], axis=1)
            ax.plot(seconds, curve, color=color, lw=1.1, label=pl.FINGERS[finger])
            for k, lag in enumerate(seconds):
                rows.append({"subject": subject, "finger": pl.FINGERS[finger], "lag_s": float(lag),
                             "median_abs_r": float(curve[k]),
                             "p90_abs_r": float(np.nanpercentile(abs_corr[k, :, finger], 90))})
        ax.axvline(0, color=pl.AXIS, lw=0.8)
        ax.axvline(delay_s, color=pl.INK, lw=1.0)
        ax.text(delay_s, 1.0, " +200 ms", transform=ax.get_xaxis_transform(), fontsize=7, va="bottom")
        ax.set(title=pl.subject_title(subject, "mediana de canales"), xlabel="Retardo (s); >0 = ECoG adelanta",
               ylabel="|r| alta gamma–flexión")
        ax.legend(loc="upper left", fontsize=7)
        best = np.nanmax(abs_corr, axis=2)  # (lags, channels)
        ax = axes[1, col]
        image = ax.pcolormesh(seconds, np.array(b.cmap.kept), best.T, cmap=pl.SEQUENTIAL, shading="nearest")
        ax.axvline(delay_s, color=pl.INK, lw=1.0)
        ax.axvline(0, color=pl.AXIS, lw=0.8)
        ax.grid(False)
        ax.set(xlabel="Retardo (s)", ylabel="Canal (0-based)", title="máx |r| entre dedos por canal")
        fig.colorbar(image, ax=ax, label="|r|")
        for channel in range(b.cmap.n_kept):
            for finger in range(5):
                k = int(np.nanargmax(abs_corr[:, channel, finger]))
                peaks.append({"subject": subject, "channel_0based": b.cmap.kept[channel],
                              "finger": pl.FINGERS[finger], "peak_lag_s": float(seconds[k]),
                              "peak_abs_r": float(abs_corr[k, channel, finger])})
    ctx.save_table(pd.DataFrame(rows), "lag_xcorr", "Correlación |r| alta gamma–flexión por retardo y dedo")
    ctx.save_fig(fig, "s09_lag_xcorr", "Correlación alta gamma–flexión según el retardo (etiquetas de desarrollo, descriptivo)")

    peaks_frame = pd.DataFrame(peaks, columns=["subject", "channel_0based", "finger", "peak_lag_s", "peak_abs_r"])
    ctx.save_table(peaks_frame, "s09_peak_lags", "Retardo del pico de |r| por canal y dedo")
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.2, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        frame = peaks_frame[peaks_frame.subject == subject]
        if frame.empty:
            empty_panel(ax, "Sin datos")
            continue
        strong = frame[frame.peak_abs_r >= frame.peak_abs_r.quantile(0.75)]
        bins = np.linspace(lag_range[0], lag_range[1], 41)
        ax.hist(frame.peak_lag_s, bins=bins, color=pl.MUTED, label="Todos los pares canal-dedo")
        ax.hist(strong.peak_lag_s, bins=bins, color=pl.KEPT, label="Cuartil superior de |r|")
        ax.axvline(b.spec.delay_samples / b.spec.target_sfreq, color=pl.INK, lw=1.0)
        ax.axvline(0, color=pl.AXIS, lw=0.8)
        ax.set(title=pl.subject_title(subject, f"mediana fuerte {strong.peak_lag_s.median():+.2f} s"),
               xlabel="Retardo del pico (s)")
        ax.legend(loc="upper left", fontsize=7)
    axes[0, 0].set_ylabel("Pares canal-dedo")
    ctx.save_fig(fig, "s09_peak_lags", "Distribución del retardo de máxima correlación: picos en retardos positivos (ECoG adelanta)")


SECTION_NAME = 's09_delay'
TITLE = 'Retardo 200 ms'
JUSTIFIES = 'Retardo fijo de 200 ms en la dirección ECoG(t) → flexión(t+0.2 s)'


def prepare_subject(b, config):
    """CLI adapter for the calculations shown in notebook 01."""
    lag_range = config.data['eda']['lag_range_seconds']
    fs = b.spec.target_sfreq
    b.lagged_correlation = lagged_correlation(b, int(round(lag_range[1] * fs)), int(round(lag_range[0] * fs)))
