"""Metrics on reconstructed trajectories (protocol §6.2, §8.3, §11.3), numpy only.

Pearson per finger and its macro mean over the five fingers are the objective;
CCC, MSE and RMSE are secondary. A finger whose series is nonfinite or (almost)
constant has an undefined Pearson: the evaluation is marked invalid with a
diagnostic and the macro value is None. Fingers are never dropped from the mean.
"""

from typing import Any

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")


def _degenerate(x: np.ndarray, relative: float) -> bool:
    return float(x.std()) <= relative * max(float(np.abs(x).max()), np.finfo(np.float64).tiny)


def trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray, *,
                       degenerate_std_relative: float) -> dict[str, Any]:
    """Metrics over the columns where ``valid`` is True, identical for every finger.

    ``y_true`` and ``y_pred`` are (fingers, T) in the units the errors must be
    reported in. Population (ddof=0) moments throughout.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or valid.shape != y_true.shape[1:]:
        raise ValueError(f"Expected (fingers, T) arrays and a (T,) mask; got {y_true.shape}, {y_pred.shape}, "
                         f"{valid.shape}")
    names = FINGERS if y_true.shape[0] == len(FINGERS) else tuple(f"output_{i}" for i in range(y_true.shape[0]))
    diagnostics: list[str] = []
    result: dict[str, Any] = {"fingers": list(names), "n_samples": int(valid.sum()), "pearson": [], "ccc": [],
                              "mse": [], "rmse": []}
    if valid.sum() < 2:
        diagnostics.append(f"only {int(valid.sum())} valid samples")
    for finger, name in enumerate(names):
        t, p = y_true[finger, valid], y_pred[finger, valid]
        if valid.sum() < 2 or not (np.isfinite(t).all() and np.isfinite(p).all()):
            if valid.sum() >= 2:
                diagnostics.append(f"{name}: nonfinite values")
            for key in ("pearson", "ccc", "mse", "rmse"):
                result[key].append(None)
            continue
        error = float(np.mean((p - t) ** 2))
        result["mse"].append(error)
        result["rmse"].append(float(np.sqrt(error)))
        mt, mp = t.mean(), p.mean()
        covariance = float(np.mean((t - mt) * (p - mp)))
        vt, vp = float(t.var()), float(p.var())
        denominator = vt + vp + float(mt - mp) ** 2
        result["ccc"].append(2 * covariance / denominator if denominator > 0 else None)
        if denominator <= 0:
            diagnostics.append(f"{name}: CCC undefined (both series constant and equal)")
        bad = [label for label, series in (("target", t), ("prediction", p))
               if _degenerate(series, degenerate_std_relative)]
        if bad:
            diagnostics.append(f"{name}: Pearson undefined, near-constant {' and '.join(bad)}")
            result["pearson"].append(None)
        else:
            result["pearson"].append(float(covariance / np.sqrt(vt * vp)))
    result["valid"] = all(value is not None for value in result["pearson"]) and valid.sum() >= 2
    result["pearson_macro"] = float(np.mean(result["pearson"])) if result["valid"] else None
    ccc = result["ccc"]
    result["ccc_macro"] = float(np.mean(ccc)) if all(v is not None for v in ccc) and ccc else None
    result["mse_mean"] = float(np.mean(result["mse"])) if all(v is not None for v in result["mse"]) else None
    result["rmse_mean"] = float(np.mean(result["rmse"])) if all(v is not None for v in result["rmse"]) else None
    result["diagnostics"] = diagnostics
    return result
