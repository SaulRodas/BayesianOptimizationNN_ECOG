"""s01 — Development data: integrity, dimensions, hashes and a raw view."""

import numpy as np
import pandas as pd

from ... import plotting as pl


def draw(ctx, subjects):
    rows = []
    for subject, b in subjects.items():
        ecog, glove = b.ecog_record.metadata, b.glove_record.metadata
        rows.append({"subject": subject, "n_channels": b.raw.shape[0], "n_kept": b.cmap.n_kept,
                     "excluded_0based": " ".join(map(str, b.cmap.excluded)) or "-",
                     "n_samples": b.n_samples, "duration_s": b.duration_s, "ecog_dtype": ecog["dtype"],
                     "adapter": ecog["adapter"], "sha256": ecog["sha256"],
                     "glove_shape": "x".join(map(str, glove["shape"])),
                     "glove_storage_hz": glove["sampling_hz_glove_storage"]})
    ctx.save_table(pd.DataFrame(rows), "s01_integrity", "Integridad: formas, duración, dtype y SHA-256 por sujeto")

    max_points = ctx.eda["plot_max_points"]
    fig, axes = pl.figure(1, len(subjects), width=4.8, height=4.2)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        step = pl.plot_step(b.n_samples, max_points)
        view = b.raw[:, ::step].astype(np.float64)
        summary = b.channel_summary
        scale = np.maximum(1.4826 * summary["mad"].to_numpy(), 1e-12)[:, None]
        z = (view - summary["median"].to_numpy()[:, None]) / scale
        image = ax.imshow(np.clip(z, -6, 6), aspect="auto", interpolation="nearest", cmap=pl.DIVERGING,
                          vmin=-6, vmax=6, extent=[0, b.duration_s, b.raw.shape[0] - 0.5, -0.5])
        for channel in b.cmap.excluded:
            ax.annotate(f"c{channel}", xy=(b.duration_s, channel), xytext=(4, 0), textcoords="offset points",
                        color=pl.EXCLUDED, fontsize=8, va="center", annotation_clip=False)
        ax.grid(False)
        ax.set(title=pl.subject_title(subject, f"cruda, cada {step} muestras"),
               xlabel="Tiempo original (s)", ylabel="Canal (0-based)")
    fig.colorbar(image, ax=axes[0, -1], label="z robusto por canal (recortado ±6)")
    ctx.save_fig(fig, "s01_raw_heatmap", "ECoG cruda de desarrollo (z robusto por canal); canales excluidos señalados en rojo")

    seconds = ctx.eda["excerpt_seconds"]
    fig, axes = pl.figure(1, len(subjects), width=4.8, height=5.0)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        window = b.excerpt(seconds)
        t = np.arange(window.start, window.stop) / b.fs
        kept = np.array(b.cmap.kept)
        shown = list(kept[np.linspace(0, kept.size - 1, 8).astype(int)]) + list(b.cmap.excluded)
        shown = sorted(set(shown))
        spacing = 8 * float(np.median(b.channel_summary["typical_local_std"]))
        for k, channel in enumerate(shown):
            trace = b.raw[channel, window].astype(np.float64)
            trace = trace - np.median(trace)
            excluded = channel in b.cmap.excluded
            ax.plot(t, np.clip(trace, -3 * spacing, 3 * spacing) + k * spacing, lw=0.6,
                    color=pl.EXCLUDED if excluded else pl.INK_SECONDARY)
        ax.set_yticks([k * spacing for k in range(len(shown))], [f"c{c}" for c in shown])
        for label, channel in zip(ax.get_yticklabels(), shown):
            if channel in b.cmap.excluded:
                label.set_color(pl.EXCLUDED)
        ax.set(title=pl.subject_title(subject, f"{seconds:g} s centrales"), xlabel="Tiempo original (s)")
    ctx.save_fig(fig, "s01_raw_traces", "Trazas crudas (8 canales conservados y los excluidos en rojo); amplitud recortada a ±3 separaciones")


SECTION_NAME = 's01_overview'
TITLE = 'Datos de desarrollo'
JUSTIFIES = 'Contexto: integridad, dimensiones y hashes de S1/S2/S3'
