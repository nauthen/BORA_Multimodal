from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn


LABEL_TO_CLASS = {0: "none", 1: "strong", 2: "medium", 3: "weak"}


def _strip_parallel_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key.removeprefix("module.") if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_wrapped_single_modal_checkpoint(
    checkpoint_path: str | Path,
    prefix_modules: Mapping[str, nn.Module],
    description: str,
) -> None:
    """Strict-load modules from a single-modal wrapper checkpoint by state-dict prefix."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw_state, Mapping):
        raise TypeError(f"{description} checkpoint does not contain a state dictionary: {path}")
    state = _strip_parallel_prefix(raw_state)

    for prefix, module in prefix_modules.items():
        prefix_token = f"{prefix}."
        module_state = {
            key[len(prefix_token) :]: value for key, value in state.items() if key.startswith(prefix_token)
        }
        if not module_state:
            raise RuntimeError(
                f"{description} checkpoint '{path}' has no parameters with required prefix '{prefix_token}'."
            )
        try:
            module.load_state_dict(module_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{description} checkpoint '{path}' is incompatible with {module.__class__.__name__} "
                f"for prefix '{prefix_token}': {exc}"
            ) from exc


def _normalize_component(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/").lower()


def _sample_id_from_stem(stem: str) -> str:
    normalized = Path(_normalize_component(stem)).stem
    for marker in ("_audio_", "_video_", "_sample_"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    return normalized


def canonical_sample_identity(record: Mapping[str, Any]) -> tuple[str, int]:
    """Build path-independent date/session/class/sample identity and validate its label."""
    label = int(record["label"])
    if label not in LABEL_TO_CLASS:
        raise ValueError(f"Unsupported dataset label {label}; expected one of {sorted(LABEL_TO_CLASS)}.")

    sample_key = _normalize_component(record.get("sample_key"))
    path = _normalize_component(record.get("audio_path") or record.get("video_path"))
    parts = [part for part in (sample_key or path).split("/") if part]
    date = _normalize_component(record.get("date")) or (parts[-4] if len(parts) >= 4 else "")
    session = _normalize_component(record.get("session")) or (parts[-3] if len(parts) >= 3 else "")
    class_name = _normalize_component(record.get("class_name")) or (parts[-2] if len(parts) >= 2 else "")
    sample_id = _normalize_component(record.get("sample_id"))
    if not sample_id and parts:
        sample_id = _sample_id_from_stem(parts[-1])
    else:
        sample_id = _sample_id_from_stem(sample_id)

    expected_class = LABEL_TO_CLASS[label]
    if class_name and class_name != expected_class:
        raise ValueError(
            f"Sample class '{class_name}' conflicts with label={label} ('{expected_class}'). Record={dict(record)}"
        )
    class_name = expected_class
    if not date or not session or not sample_id:
        raise ValueError(
            "Cannot derive date/session/sample_id for split-integrity validation. "
            f"Record={dict(record)}"
        )
    return f"{date}/{session}/{class_name}/{sample_id}", label


def _index_records(records: Iterable[Mapping[str, Any]], source: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for record in records:
        identity, label = canonical_sample_identity(record)
        if identity in result:
            raise ValueError(f"Duplicate sample identity '{identity}' in {source}.")
        result[identity] = label
    return result


def _read_split_csv(path: Path) -> list[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint split sidecar not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_checkpoint_split_integrity(
    checkpoint_path: str | Path,
    current_splits: Mapping[str, Iterable[Mapping[str, Any]]],
    description: str,
) -> None:
    """Require exact train/val/test identity and label equality with checkpoint sidecars."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{description} checkpoint not found: {checkpoint}")
    split_dir = checkpoint.parent / "splits"
    current_indexes: Dict[str, Dict[str, int]] = {}
    sidecar_indexes: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        if split_name not in current_splits:
            raise ValueError(f"Current run is missing required '{split_name}' split.")
        current_indexes[split_name] = _index_records(
            current_splits[split_name], f"current {split_name} split"
        )
        sidecar_indexes[split_name] = _index_records(
            _read_split_csv(split_dir / f"{split_name}.csv"),
            f"{description} checkpoint {split_name} sidecar",
        )
        if current_indexes[split_name] != sidecar_indexes[split_name]:
            current_keys = set(current_indexes[split_name])
            sidecar_keys = set(sidecar_indexes[split_name])
            missing = sorted(current_keys - sidecar_keys)[:3]
            unexpected = sorted(sidecar_keys - current_keys)[:3]
            label_mismatch = sorted(
                key
                for key in current_keys & sidecar_keys
                if current_indexes[split_name][key] != sidecar_indexes[split_name][key]
            )[:3]
            raise ValueError(
                f"{description} checkpoint split mismatch for '{split_name}': "
                f"current={len(current_keys)}, checkpoint={len(sidecar_keys)}, "
                f"missing_examples={missing}, unexpected_examples={unexpected}, "
                f"label_mismatch_examples={label_mismatch}."
            )

    owner: Dict[str, str] = {}
    for split_name, index in sidecar_indexes.items():
        for identity in index:
            if identity in owner:
                raise ValueError(
                    f"{description} checkpoint leakage: sample '{identity}' occurs in both "
                    f"'{owner[identity]}' and '{split_name}'."
                )
            owner[identity] = split_name
