"""s10 — Scaling: RobustScaler(10, 90) for features and MinMax for targets, per fold."""

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from ... import preprocessing as pp
from ... import plotting as pl
from ._common import empty_panel

GAUSSIAN_IQR_10_90 = 2.5631031310892007  # (q90 - q10) / sigma for a normal distribution


def draw(ctx, subjects):
    # 10a — heavy tails of Morlet power for the illustrative channel.
    shown = (0, 20, 39)
    fig, axes = pl.figure(len(shown), len(subjects), width=4.2, height=2.4)
    for col, (subject, b) in enumerate(subjects.items()):
        valid = b.valid100
        channel = b.descriptive_channel
        for row, index in enumerate(shown):
            values = b.power100[channel, index, valid].astype(np.float64)
            values = values[values > 0]
            ax = axes[row, col]
            ax.hist(np.log10(values), bins=80, color=pl.KEPT)
            mean, std = values.mean(), values.std()
            q10, q50, q90 = np.percentile(values, [10, 50, 90])
            for value, color, label in ((q50, pl.INK, "mediana"), (q10, pl.SLOTS[2], "q10/q90"), (q90, pl.SLOTS[2], None),
                                        (mean, pl.COMPARATOR, "media±std")):
                ax.axvline(np.log10(value), color=color, lw=1.0, label=label)
            if mean + std > 0:
                ax.axvline(np.log10(mean + std), color=pl.COMPARATOR, lw=1.0)
            ax.set(title=f"{subject} · c{b.cmap.kept[channel]} · {b.spec.freqs[index]:.0f} Hz",
                   xlabel="log10 potencia" if row == len(shown) - 1 else None)
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)
    ctx.save_fig(fig, "s10_power_tails", "Distribución de la potencia Morlet: cola derecha pesada; media y STD quedan dominadas por extremos")

    # 10b — std vs robust scale (sklearn RobustScaler semantics) per channel x frequency.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.8)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        ratio = b.scale_ratio
        image = ax.imshow(ratio.T, aspect="auto", origin="lower", cmap=pl.SEQUENTIAL,
                          norm=LogNorm(vmin=max(ratio.min(), 0.5), vmax=max(ratio.max(), 2)), interpolation="nearest",
                          extent=[-0.5, b.cmap.n_kept - 0.5, 0, b.spec.freqs.size])
        ticks = [0, 13, 26, 39]
        ax.set_yticks([t + 0.5 for t in ticks], [f"{b.spec.freqs[t]:.0f}" for t in ticks])
        ax.grid(False)
        ax.set(title=pl.subject_title(subject, f"mediana {np.median(ratio):.1f}"), xlabel="Canal efectivo",
               ylabel="Frecuencia (Hz)")
        pl.plain_colorbar_ticks(fig.colorbar(image, ax=ax, label="STD / escala robusta (1 = gaussiana)"))
    ctx.save_fig(fig, "s10_std_vs_robust", "STD frente a la escala de RobustScaler(10,90, unit_variance): >1 indica colas pesadas")

    # 10c — drift between chronological blocks.
    block_rows = []
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.4, sharey=True)
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        # float32 keeps this at ~400 MB per real subject.
        log_power = np.log10(np.maximum(b.power100, np.float32(1e-30)))
        valid = b.valid100
        overall = np.median(log_power[:, :, valid], axis=2)
        y100, _ = b.glove100
        for k, block in enumerate(b.blocks):
            mask = np.zeros_like(valid)
            mask[block.start_target:block.stop_target] = True
            mask &= valid
            if mask.sum() < 2:
                continue
            drift = np.median(log_power[:, :, mask], axis=2) - overall
            ax.plot(b.spec.freqs, 10 * drift.mean(axis=0), color=pl.SLOTS[k], lw=1.1, label=block.name)
            block_rows.append({"subject": subject, "block": block.name, "start_s": block.start_source / b.fs,
                               "end_s": block.stop_source / b.fs, "valid_samples_100hz": int(mask.sum()),
                               "median_hg_log10": float(np.median(b.hg100[:, mask])),
                               "drift_db_mean": float(10 * drift.mean()),
                               **{f"flexion_mean_{f}": float(y100[i, mask].mean()) for i, f in enumerate(pl.FINGERS)},
                               **{f"flexion_std_{f}": float(y100[i, mask].std()) for i, f in enumerate(pl.FINGERS)}})
        ax.axhline(0, color=pl.AXIS, lw=0.8)
        ax.set(xscale="log", title=subject, xlabel="Frecuencia (Hz)")
        ax.legend(loc="upper right")
    axes[0, 0].set_ylabel("Mediana del bloque − registro (dB)")
    ctx.save_table(pd.DataFrame(block_rows), "block_stats", "Estadísticos por bloque B1–B5 (deriva de potencia y flexión)")
    ctx.save_fig(fig, "s10_block_drift", "Deriva de la potencia entre bloques: los escaladores deben ajustarse con el entrenamiento de cada fold")

    # 10d — MinMax fitted on the other blocks: targets outside [0, 1] in the held-out block.
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=3.2, sharey=True)
    outside_rows = []
    for ax, (subject, b) in zip(axes[0], subjects.items()):
        y100, _ = b.glove100
        valid = b.valid100
        names = [block.name for block in b.blocks]
        fractions = b.minmax_outside
        if np.isnan(fractions).all():
            empty_panel(ax, "Bloques sin muestras válidas")
            continue
        width = 0.16
        for finger, color in enumerate(pl.FINGER_COLORS):
            ax.bar(np.arange(len(names)) + (finger - 2) * width, 100 * fractions[:, finger], width=width * 0.9,
                   color=color, label=pl.FINGERS[finger])
            for k, name in enumerate(names):
                outside_rows.append({"subject": subject, "held_out_block": name, "finger": pl.FINGERS[finger],
                                     "fraction_outside_0_1": float(fractions[k, finger])})
        ax.set_xticks(np.arange(len(names)), names)
        ax.set(title=subject, xlabel="Bloque evaluado")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("% fuera de [0, 1] con MinMax de los otros bloques")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    ctx.save_table(pd.DataFrame(outside_rows), "s10_minmax_outside", "Fracción de flexión fuera de [0,1] con MinMax ajustado sin el bloque")
    ctx.save_fig(fig, "s10_minmax_heldout", "MinMax ajustado con los otros bloques: la validación puede salir de [0,1] (solo transform, nunca fit)")


SECTION_NAME = 's10_scaling'
TITLE = 'Escalado'
JUSTIFIES = 'RobustScaler(10, 90) y MinMax ajustados por fold, sin z-score global'


def scale_ratio(b, config):
    valid = np.flatnonzero(b.valid100)[::4]
    matrix = pp.features_to_matrix(b.power100[:, :, valid]).astype(np.float64)
    scaler = pp.make_feature_scaler(config).fit(matrix)  # descriptive only: whole record
    ratio = (matrix.std(axis=0) / np.maximum(scaler.scale_, 1e-300)).reshape(b.cmap.n_kept, b.spec.freqs.size)
    return ratio


def minmax_outside(b, config):
    y100, _ = b.glove100
    valid = b.valid100
    fractions = np.full((len(b.blocks), 5), np.nan)
    for k, block in enumerate(b.blocks):
        held = np.zeros_like(valid)
        held[block.start_target:block.stop_target] = True
        train, test = valid & ~held, valid & held
        if train.sum() < 2 or test.sum() < 2:
            continue
        scaler = pp.make_target_scaler(config).fit(y100[:, train].T)
        scaled = scaler.transform(y100[:, test].T)
        fractions[k] = ((scaled < 0) | (scaled > 1)).mean(axis=0)
    return fractions


def prepare_subject(b, config):
    """CLI adapter for the calculations shown in notebook 01."""
    b.scale_ratio = scale_ratio(b, config)
    b.minmax_outside = minmax_outside(b, config)
