"""Schema and semantic checks for manual QC decisions; no auto-approval."""

from pathlib import Path
from typing import Any
from datetime import datetime, timedelta
import json
import math
import re

from jsonschema import Draft202012Validator

from .config import inside
from .io import sha256_file


def validate_decisions(document: dict[str, Any], schema_path: Path,
                       evidence_root: Path, metadata: dict[str, dict],
                       expected_config_sha256: str | None = None) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)
    approved = document["status"] == "approved"
    any_approved = approved or any(s["status"] == "approved" for s in document["subjects"].values())
    if any_approved and (not document.get("reviewed_by") or not document.get("approval_at_utc")):
        raise ValueError("Approval requires reviewer and timestamp")
    if any_approved:
        if not document["reviewed_by"].strip():
            raise ValueError("Approval requires a nonempty reviewer")
        stamp = datetime.fromisoformat(document["approval_at_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("Approval timestamp must use UTC explicitly")
    digest = document.get("qc_config_sha256")
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid configuration SHA-256")
    if expected_config_sha256 is not None and digest != expected_config_sha256:
        raise ValueError("Configuration hash does not match the reviewed run")
    if any_approved and digest is None:
        raise ValueError("Approval requires configuration SHA-256")
    artifact_hashes = {}
    if any_approved:
        manifest_path = evidence_root / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Approval requires an artifact hash manifest")
        artifact_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
        snapshot = evidence_root / "experiment_config.yaml"
        if not snapshot.is_file() or sha256_file(snapshot) != digest:
            raise ValueError("Approval configuration snapshot hash mismatch")
    for subject, decision in document["subjects"].items():
        subject_approved = decision["status"] == "approved"
        if approved and not subject_approved:
            raise ValueError("Global approval requires approval of all subjects")
        source = metadata.get(subject)
        if source is None:
            if (subject_approved or decision["excluded_channel_indices_0based"] or decision["intervals"]):
                raise ValueError(f"{subject}: cannot decide without real source metadata")
            continue
        count = source["n_ecog_channels_original"]
        if decision.get("original_channel_count") != count:
            raise ValueError(f"{subject}: channel count mismatch")
        if decision.get("raw_data_sha256") != source["sha256"]:
            raise ValueError(f"{subject}: source hash mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise ValueError(f"{subject}: invalid source SHA-256")
        if any(c >= count for c in decision["excluded_channel_indices_0based"]):
            raise ValueError(f"{subject}: channel index outside source")
        if len(decision["excluded_channel_indices_0based"]) >= count:
            raise ValueError(f"{subject}: cannot exclude every channel")
        if decision["excluded_channel_indices_0based"] and not decision.get("notes"):
            raise ValueError(f"{subject}: excluded channels need documented reasons in notes")
        evidence = list(decision["evidence_paths"])
        for interval in decision["intervals"]:
            start, end = interval["start_s"], interval["end_s"]
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= source["duration_s"]):
                raise ValueError(f"{subject}: invalid half-open interval")
            if any(c >= count for c in interval.get("channel_indices_0based", [])):
                raise ValueError(f"{subject}: interval channel outside source")
            evidence.extend(interval.get("evidence_paths", []))
            if subject_approved and interval["action"] == "repair":
                raise ValueError("Local repair requires a separately implemented and verified rule; unavailable in EDA")
        if subject_approved:
            if not evidence or not document.get("reviewed_by") or not document.get("approval_at_utc"):
                raise ValueError(f"{subject}: approval needs evidence, reviewer and timestamp")
        for relative in evidence:
            path = inside(evidence_root, relative)
            if not path.is_file():
                raise ValueError(f"{subject}: missing evidence {relative!r}")
            if subject_approved and artifact_hashes.get(relative) != sha256_file(path):
                raise ValueError(f"{subject}: evidence hash mismatch for {relative!r}")
