"""s02 — Atypical channels: evidence for the fixed exclusion list."""

from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from ... import preprocessing as pp
from ...descriptive import grouped_intervals
from .. import plotting as pl
from ..context import section

EXCLUDED_STYLES = [pl.EXCLUDED, pl.SLOTS[6], pl.SLOTS[1]]


@section("s02_channels", "Canales atípicos",
         "Exclusión fija de S1-54, S2-20 y S2-37 antes de la CMR (evidencia solo de ECoG)")
def run(ctx, subjects):
    levels = ctx.eda["std_ratio_levels"]
    summary = pd.concat([b.channel_summary for b in subjects.values()], ignore_index=True)
    ctx.save_table(summary, "channel_summary", "Resumen por canal: STD local, cociente pico/típico, fracción de ventanas y línea")

    # 02a — per-channel metrics, excluded channels in red.
    metrics = [("typical_local_std", "STD local típica (u.o.)", True),
               ("peak_over_typical", "Pico / típica de STD local", True),
               (f"fraction_windows_above_{levels[0]}x", f"Fracción de ventanas > {levels[0]}×", False),
               ("line_fraction_raw", "Fracción de línea (40–200 Hz)", False)]
    fig, axes = pl.figure(len(metrics), len(subjects), width=4.6, height=2.3, sharex="col")
    for col, (subject, b) in enumerate(subjects.items()):
        frame = b.channel_summary
        for row, (column, label, log) in enumerate(metrics):
            ax = axes[row, col]
            kept = ~frame["excluded"]
            ax.scatter(frame.loc[kept, "channel_0based"], frame.loc[kept, column], s=10, color=pl.KEPT,
                       label="Conservado")
            ax.scatter(frame.loc[~kept, "channel_0based"], frame.loc[~kept, column], s=22, color=pl.EXCLUDED,
                       label="Excluido")
            for _, item in frame[~kept].iterrows():
                ax.annotate(f"c{item['channel_0based']}", (item["channel_0based"], item[column]),
                            xytext=(4, 2), textcoords="offset points", fontsize=7, color=pl.INK_SECONDARY)
            if log:
                pl.adaptive_log(ax, frame[column])
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(subject)
            if row == len(metrics) - 1:
                ax.set_xlabel("Canal (0-based)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2)
    ctx.save_fig(fig, "s02_channel_metrics", "Métricas por canal; los excluidos destacan en STD pico/típica y fracción de ventanas")

    # 02b — ranking of the peak/typical ratio.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=4.2)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        top = b.channel_summary.nlargest(15, "peak_over_typical").iloc[::-1]
        colors = [pl.EXCLUDED if e else pl.KEPT for e in top["excluded"]]
        ax.barh([f"c{c}" for c in top["channel_0based"]], top["peak_over_typical"], color=colors, height=0.7)
        ax.set_xscale("log")
        for level in levels:
            ax.axvline(level, color=pl.MUTED, lw=0.7)
            ax.text(level, 1.0, f"{level:g}×", transform=ax.get_xaxis_transform(), fontsize=7,
                    color=pl.INK_SECONDARY, ha="center", va="bottom")
        ax.set(title=pl.subject_title(subject, "top 15"), xlabel="STD local pico / típica (log)")
        ax.grid(axis="y", visible=False)
    ctx.save_fig(fig, "s02_ranking", "Ranking del cociente pico/típico: los tres canales excluidos se separan del resto (líneas = referencias descriptivas, no umbrales)")

    # 02c — time x channel heatmap of local variability.
    fig, axes = pl.figure(1, len(subjects), width=4.8, height=4.2)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        image = ax.imshow(np.log10(np.maximum(b.std_ratio, 1e-3)), aspect="auto", interpolation="nearest",
                          cmap=pl.SEQUENTIAL, vmin=0, vmax=np.log10(max(levels[-1], 10)) + 1,
                          extent=[0, b.duration_s, b.raw.shape[0] - 0.5, -0.5])
        for channel in b.cmap.excluded:
            ax.annotate(f"c{channel}", xy=(b.duration_s, channel), xytext=(4, 0), textcoords="offset points",
                        color=pl.EXCLUDED, fontsize=8, va="center", annotation_clip=False)
        ax.grid(False)
        ax.set(title=subject, xlabel="Tiempo original (s)", ylabel="Canal (0-based)")
    fig.colorbar(image, ax=axes[0, -1], label="log10(STD local / típica del canal)")
    ctx.save_fig(fig, "s02_variability_map", "Mapa tiempo×canal de variabilidad local: episodios de los canales excluidos")

    # 02d — one evidence figure per excluded channel (names referenced from the YAML).
    episodes = []
    window_s = ctx.eda["window_seconds"]
    context_s = ctx.eda["episode_context_seconds"]
    for subject, b in subjects.items():
        common = pp.common_median(b.kept)
        for entry in ctx.config.channel_exclusions(subject):
            channel = entry["index"]
            figure_path = PurePosixPath(entry["evidence_figure"])
            if figure_path.parent.as_posix() != "fig" or figure_path.suffix != ".png":
                raise ValueError(f"evidence_figure must be fig/<name>.png, got {figure_path}")
            ratio = b.std_ratio[channel]
            for level in levels:
                for start, stop in grouped_intervals(ratio > level, 0):
                    episodes.append({"subject": subject, "channel_0based": channel, "level_x": level,
                                     "start_s": start * window_s, "end_s": min(stop * window_s, b.duration_s),
                                     "peak_ratio": float(ratio[start:stop].max())})
            worst = int(np.argmax(ratio))
            center = (worst + 0.5) * window_s
            window = b.excerpt(2 * context_s + window_s, center)
            t = np.arange(window.start, window.stop) / b.fs
            kept = np.array(b.cmap.kept)
            neighbours = kept[np.argsort(np.abs(kept - channel))[:2]]
            fig, axes = pl.figure(3, 1, width=10, height=2.4)
            ax = axes[0, 0]
            times = (np.arange(ratio.size) + 0.5) * window_s
            kept_ratio = b.std_ratio[kept]
            ax.fill_between(times, np.percentile(kept_ratio, 5, axis=0), np.percentile(kept_ratio, 95, axis=0),
                            color=pl.KEPT, alpha=0.2, lw=0, label="Conservados p5–p95")
            ax.plot(times, ratio, color=pl.EXCLUDED, lw=1.0, label=f"c{channel}")
            pl.adaptive_log(ax, np.r_[ratio, kept_ratio.ravel()])
            ax.axvspan(window.start / b.fs, window.stop / b.fs, color=pl.MUTED, alpha=0.15, lw=0)
            ax.set(title=pl.wrap(f"{subject} · canal {channel} (0-based) — {entry['reason']}", 130),
                   ylabel="STD local / típica", xlabel="Tiempo original (s)")
            ax.title.set_fontsize(8.5)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax = axes[1, 0]
            ax.plot(t, b.raw[channel, window], color=pl.EXCLUDED, lw=0.7, label=f"c{channel}")
            ax.set(ylabel="u.o.", title=f"c{channel} crudo · zoom ±{context_s:g} s alrededor de la ventana más extrema")
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax = axes[2, 0]
            for k, other in enumerate(neighbours):
                ax.plot(t, b.raw[other, window], color=pl.SLOTS[k + 1], lw=0.7, label=f"c{other}")
            ax.plot(t, common[window], color=pl.INK, lw=0.9, label="Mediana común")
            ax.set(ylabel="u.o.", xlabel="Tiempo original (s)",
                   title="Mismo intervalo en canales conservados (índices contiguos; no implica vecindad física)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ctx.save_fig(fig, figure_path.stem, f"Evidencia de exclusión {subject} canal {channel}")
    ctx.save_table(pd.DataFrame(episodes, columns=["subject", "channel_0based", "level_x", "start_s",
                                                   "end_s", "peak_ratio"]),
                   "s02_episodes", "Episodios de los canales excluidos por nivel de STD local (descriptivo)")

    # 02e — PSD of excluded channels against kept channels.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        freq, psd = b.raw_psd
        mask = (freq >= 1) & (freq <= 500)
        kept_psd = psd[np.array(b.cmap.kept)][:, mask]
        ax.fill_between(freq[mask], np.percentile(kept_psd, 5, axis=0), np.percentile(kept_psd, 95, axis=0),
                        color=pl.KEPT, alpha=0.2, lw=0, label="Conservados p5–p95")
        ax.plot(freq[mask], np.median(kept_psd, axis=0), color=pl.KEPT, lw=1.2, label="Mediana conservados")
        for k, channel in enumerate(b.cmap.excluded):
            ax.plot(freq[mask], psd[channel, mask], color=EXCLUDED_STYLES[k % 3], lw=1.0, label=f"c{channel}")
        pl.mark_lines(ax, ctx.config.data["signal"]["notch_hz"])
        ax.set(xscale="log", yscale="log", title=subject, xlabel="Frecuencia (Hz)")
        if not b.cmap.excluded:
            ax.text(0.5, 0.92, "Sin exclusiones", transform=ax.transAxes, ha="center", color=pl.INK_SECONDARY)
        ax.legend(loc="lower left")
    axes[0, 0].set_ylabel("PSD (u.o.²/Hz)")
    ctx.save_fig(fig, "s02_excluded_psd", "PSD de los canales excluidos frente a la banda de los conservados")
