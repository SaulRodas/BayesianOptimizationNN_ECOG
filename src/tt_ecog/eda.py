"""Stage-one orchestration, usable identically from CLI and notebook."""

import argparse
import json
from pathlib import Path
import traceback
from uuid import uuid4

import pandas as pd

from .config import load_config
from .decisions import validate_decisions
from .io import read_ecog, validate_development_labels, sha256_file
from .quality_control import analyze_ecog, EVENT_COLUMNS
from .reporting import environment, plot_subject, utc_now, write_json


def run_eda(config_path: str | Path, drive_root: str | Path) -> Path:
    config = load_config(config_path)
    project_root = config.source.parent.parent
    drive_root = Path(drive_root)
    storage = config.storage_root(drive_root)
    if not storage.is_dir():
        raise FileNotFoundError("BLOCKED: persistent storage root does not exist; mount Drive and verify TT_ECoG")
    raw_dir = config.path(drive_root, "raw_dir")
    results_dir = config.path(drive_root, "results_dir")
    if results_dir == raw_dir or results_dir.is_relative_to(raw_dir) or raw_dir.is_relative_to(results_dir):
        raise ValueError("Results and raw data must be separate directory trees")
    run_id = utc_now().replace(":", "").replace("+0000", "Z") + "_" + uuid4().hex[:8]
    output = results_dir / "eda_qc" / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = environment(project_root)
    env["seed_base"] = config.data["project"]["seed_base"]
    env["random_operations"] = False
    write_json(output / "environment.json", env)
    write_json(output / "effective_config.json", config.data)
    (output / "experiment_config.yaml").write_bytes(config.source.read_bytes())
    decisions = json.loads((project_root / "configs/qc_decisions.template.json").read_text(encoding="utf-8"))
    decisions.update(created_at_utc=utc_now(), qc_config_sha256=config.sha256)
    # This command creates proposals only, even if a template was manually edited.
    if decisions["status"] != "pending" or any(s["status"] != "pending" for s in decisions["subjects"].values()):
        raise ValueError("QC template must remain pending")
    if any(s["excluded_channel_indices_0based"] or s["intervals"] for s in decisions["subjects"].values()):
        raise ValueError("QC template cannot contain preloaded exclusions or treatments")
    summaries, channels, events, temporal = [], [], [], []
    metadata = {}
    for subject in config.data["project"]["subjects"]:
        row = {"subject": subject, "status": "PENDING", "details": "", "n_channels": None,
               "n_samples": None, "duration_s": None, "n_candidate_events": None}
        try:
            record = read_ecog(config, drive_root, subject)
            # Metadata validation is independent of QC and contributes no detector inputs.
            label_metadata = validate_development_labels(config, drive_root, record)
            qc = analyze_ecog(record.values, subject, config.data["raw_data"]["ecog_fs_hz"],
                              config.data["quality_control"], config.data["signal"])
            figures = plot_subject(record, qc, output / "qc_figures", config.data["quality_control"]["analysis"]["plot_max_points"])
            metadata[subject] = {**record.metadata, "development_labels": label_metadata}
            decisions["subjects"][subject].update(
                original_channel_count=record.values.shape[0], raw_data_sha256=record.metadata["sha256"],
                evidence_paths=["qc_channels.csv", "qc_events.csv", "qc_temporal.csv", *figures],
                notes="Sin acciones aprobadas. Revisar eventos y evidencia; una lista vacía no significa aprobación.")
            write_json(output / f"{subject}_detectors.json", qc.detector_status)
            channels.append(qc.channels)
            events.append(qc.events)
            temporal.append(qc.temporal)
            pending = any(status.startswith("PENDING") for status in qc.detector_status.values())
            row.update(status="PENDING_THRESHOLDS" if pending else "PENDING_REVIEW",
                       details="Descriptive analysis completed; no data cleaning or approval",
                       n_channels=record.values.shape[0], n_samples=record.values.shape[1],
                       duration_s=record.metadata["duration_s"], n_candidate_events=len(qc.events))
        except FileNotFoundError as exc:
            row.update(status="BLOCKED", details=str(exc))
        except Exception as exc:
            row.update(status="FAIL", details=f"{type(exc).__name__}: {exc}")
            (output / f"{subject}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        summaries.append(row)
    for name, frames, columns in (
        ("qc_channels.csv", channels, ["subject", "channel_0based", "channel_1based"]),
        ("qc_events.csv", events, EVENT_COLUMNS),
        ("qc_temporal.csv", temporal, ["subject", "channel_0based", "start_s", "end_s"]),
    ):
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        frame.to_csv(output / name, index=False)
    pd.DataFrame(summaries).to_csv(output / "qc_summary.csv", index=False)
    write_json(output / "data_manifest.json", metadata)
    validate_decisions(decisions, project_root / "schemas/qc_decisions.schema.json", output, metadata, config.sha256)
    write_json(output / "qc_decisions.json", decisions)
    statuses = {s["status"] for s in summaries}
    status = ("FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses
              else "PENDING_THRESHOLDS" if "PENDING_THRESHOLDS" in statuses else "PENDING_REVIEW")
    summary = {"schema_version": 1, "run_id": run_id, "status": status, "subjects": summaries,
               "qc_approved": False, "pipeline_verified": False, "config_sha256": config.sha256,
               "git_commit": env["git_commit"], "source_sha256": env["source_sha256"],
               "test_accessed": False, "completed_at_utc": utc_now()}
    write_json(output / "run_summary.json", summary)
    review = ["# Revisión de adquisición ECoG", "", f"Estado de ejecución: {status}",
              "", "QC: PENDING. Ninguna acción de limpieza aprobada.",
              "", "Inspección descriptiva global de desarrollo; no es QC anidado por fold.",
              "Las etiquetas de desarrollo solo se validan; nunca alimentan detectores.",
              "No se accedió a variables ni archivos de prueba.",
              "", "Revisar estadísticas, PSD, trazas y eventos para los tres sujetos.",
              "Definir candidatos de umbral con evidencia ECoG en el YAML y repetir el EDA.",
              "Registrar acciones, razones, evidencia, revisor y fecha antes de aprobar QC.",
              "No aplicar preprocesamiento, entrenamiento ni Optuna desde esta etapa.",
              "", "Los intervalos usan [start_sample, end_sample_exclusive); los tiempos son originales.",
              "grouped_candidate_fraction incluye los huecos cortos unidos por event_merge_gap_seconds.",
              "Las figuras crudas pueden submuestrearse solo para visualización; las métricas usan todas las muestras.",
              "PSD usa Welch/Hann y detrend constante para el estimador, sin alterar los datos de origen.",
              "La fracción plana es std local / std del canal; un canal constante tiene fracción cero.",
              "Un tramo final de una sola muestra no admite PSD/variabilidad local y queda fuera de esas estadísticas.",
              "MAD casi nula usa un piso numérico explícito, registrado en el YAML.", ""]
    for row in summaries:
        review.append(f"- {row['subject']}: {row['status']} — {row['details']}")
    (output / "qc_review_log.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    artifacts = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "artifact_manifest.json", {"sha256": artifacts, "self_excluded": True})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only EDA/QC; never approves decisions")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True, help="Mounted MyDrive root, or local equivalent")
    args = parser.parse_args()
    try:
        output = run_eda(args.config, args.drive_root)
    except FileNotFoundError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except (ValueError, PermissionError) as exc:
        print(f"FAIL: {exc}")
        return 1
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    print(f"{summary['status']}: {output}")
    return 1 if summary["status"] == "FAIL" else 2 if summary["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
