"""s03 — Common reference: instantaneous median of kept channels, not the mean."""

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from .. import comparators
from .. import plotting as pl
from ..context import section
from ._common import empty_panel

VARIANTS = [("CAR (todos)", pl.SLOTS[1]), ("CMR (todos)", pl.SLOTS[2]),
            ("CAR (conservados)", pl.SLOTS[3]), ("CMR (conservados)", pl.SLOTS[0])]


def _references(b, window: slice) -> list[np.ndarray]:
    raw = b.raw[:, window].astype(np.float64)
    kept = b.kept[:, window]
    return [comparators.common_average(raw), pp.common_median(raw),
            comparators.common_average(kept), pp.common_median(kept)]


@section("s03_reference", "Referencia común",
         "Mediana común instantánea (CMR) de los canales conservados; excluir antes de referenciar")
def run(ctx, subjects):
    context_s = ctx.eda["episode_context_seconds"]
    window_s = ctx.eda["window_seconds"]
    cases = [(subject, b, channel) for subject, b in subjects.items() for channel in b.cmap.excluded]

    # 03a — reference signals during the worst episode of each excluded channel.
    if cases:
        fig, axes = pl.figure(len(cases), 2, width=5.6, height=2.3)
        for row, (subject, b, channel) in enumerate(cases):
            center = (int(np.argmax(b.std_ratio[channel])) + 0.5) * window_s
            window = b.excerpt(2 * context_s + window_s, center)
            t = np.arange(window.start, window.stop) / b.fs
            refs = _references(b, window)
            for col, skip_first in enumerate((False, True)):
                ax = axes[row, col]
                for k, ((label, color), ref) in enumerate(zip(VARIANTS, refs)):
                    if skip_first and k == 0:
                        continue
                    ax.plot(t, ref, color=color, lw=0.8, label=label)
                ax.set_title(f"{subject} · episodio de c{channel}" + (" (sin CAR-todos)" if skip_first else ""))
                ax.set_ylabel("Referencia (u.o.)")
            axes[row, 0].set_xlabel("Tiempo original (s)")
            axes[row, 1].set_xlabel("Tiempo original (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=4)
        ctx.save_fig(fig, "s03_reference_episodes",
                     "Señal de referencia durante los peores episodios: la media arrastra el artefacto a todos los canales; la mediana no")

    # 03b — robustness of each reference over the whole record.
    rows = []
    for subject, b in subjects.items():
        typical = float(np.median(b.channel_summary.loc[~b.channel_summary["excluded"], "typical_local_std"]))
        refs = _references(b, slice(0, b.n_samples))
        for (label, _), ref in zip(VARIANTS, refs):
            rows.append({"subject": subject, "reference": label,
                         "p999_abs_over_typical_std": float(np.percentile(np.abs(ref - np.median(ref)), 99.9)) / typical,
                         "max_abs_over_typical_std": float(np.abs(ref - np.median(ref)).max()) / typical})
    table = pd.DataFrame(rows)
    ctx.save_table(table, "s03_reference_robustness", "Robustez de cada referencia (p99.9 y máximo de |ref| / STD típica)")
    fig, axes = pl.figure(1, 2, width=5.6, height=3.4)
    for ax, metric, label in zip(axes[0], ("p999_abs_over_typical_std", "max_abs_over_typical_std"),
                                 ("p99.9 |ref| / STD típica", "máx |ref| / STD típica")):
        width = 0.2
        names = list(subjects)
        for k, (variant, color) in enumerate(VARIANTS):
            values = [table[(table.subject == s) & (table.reference == variant)][metric].iloc[0] for s in names]
            ax.bar(np.arange(len(names)) + (k - 1.5) * width, values, width=width * 0.9, color=color, label=variant)
        ax.set_xticks(np.arange(len(names)), names)
        ax.set(yscale="log", ylabel=label)
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="upper left")
    ctx.save_fig(fig, "s03_reference_robustness", "La CMR de conservados mantiene la referencia en la escala de un canal típico")

    # 03c — PSD and line fraction before/after CMR (kept channels).
    fig, axes = pl.figure(2, len(subjects), width=4.6, height=3.2)
    line_rows = []
    for col, (subject, b) in enumerate(subjects.items()):
        freq, raw_psd = b.raw_psd
        kept_psd = raw_psd[np.array(b.cmap.kept)]
        _, cmr_psd = b.cmr_psd
        mask = (freq >= 1) & (freq <= 500)
        ax = axes[0, col]
        ax.plot(freq[mask], np.median(kept_psd[:, mask], axis=0), color=pl.MUTED, lw=1.0, label="Antes de CMR")
        ax.plot(freq[mask], np.median(cmr_psd[:, mask], axis=0), color=pl.KEPT, lw=1.2, label="Después de CMR")
        pl.shade_band(ax, ctx.config.data["signal"]["bandpass_hz"])
        ax.set(xscale="log", yscale="log", title=subject, xlabel="Frecuencia (Hz)", ylabel="PSD mediana (u.o.²/Hz)")
        ax.legend(loc="lower left")
        before = b.line_fraction(freq, kept_psd)
        after = b.line_fraction(freq, cmr_psd)
        ax = axes[1, col]
        ax.scatter(before, after, s=12, color=pl.KEPT)
        ax.plot([0, 1], [0, 1], color=pl.MUTED, lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Fracción de línea antes de CMR",
               ylabel="Fracción de línea después de CMR")
        for channel, x0, y0 in zip(b.cmap.kept, before, after):
            line_rows.append({"subject": subject, "channel_0based": channel,
                              "line_fraction_before_cmr": x0, "line_fraction_after_cmr": y0})
    ctx.save_table(pd.DataFrame(line_rows), "s03_line_fraction_cmr", "Fracción de línea por canal antes y después de la CMR")
    ctx.save_fig(fig, "s03_cmr_spectrum", "La CMR elimina el componente común (incluido ruido de línea compartido)")

    # 03d — inter-channel correlation of the band-limited signal before/after CMR.
    seconds = ctx.eda["correlation_excerpt_seconds"]
    fig, axes = pl.figure(2, len(subjects), width=4.0, height=3.6)
    for col, (subject, b) in enumerate(subjects.items()):
        window = b.excerpt(seconds)
        before = pp.apply_fir(b.kept[:, window], b.spec.fir)
        interior = pp.valid_slice(before.shape[1], b.spec.fir.half_support)
        if interior.stop - interior.start < 2:
            empty_panel(axes[0, col], "Extracto demasiado corto")
            empty_panel(axes[1, col], "Extracto demasiado corto")
            continue
        after = b.filtered[:, window]
        for row, (data, label) in enumerate(((before[:, interior], "antes de CMR"), (after[:, interior], "después de CMR"))):
            corr = np.corrcoef(data)
            off = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
            ax = axes[row, col]
            image = ax.imshow(corr, cmap=pl.DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
            ax.grid(False)
            ax.set(title=f"{subject} · {label} · |r| medio {off.mean():.2f}", xlabel="Canal efectivo",
                   ylabel="Canal efectivo")
    fig.colorbar(image, ax=axes[:, -1], label="r de Pearson (40–200 Hz)")
    ctx.save_fig(fig, "s03_channel_correlation", "Correlación entre canales en 40–200 Hz: la CMR reduce la correlación compartida")
