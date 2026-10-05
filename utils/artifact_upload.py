from __future__ import annotations

import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, get_token

from config import ArtifactUploadConfig, TrainConfig

logger = logging.getLogger(__name__)


def build_artifact_name(cfg: TrainConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"MultimodalDL_{cfg.audio.backbone}_{cfg.video.backbone}_"
        f"{cfg.fusion.type}_{cfg.evaluation_mode}_{timestamp}.zip"
    )


def zip_source_tree(source_dir: str | Path, output_zip: str | Path) -> Path:
    source_path = Path(source_dir).resolve()
    output_zip = Path(output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    base_name = str(output_zip.with_suffix(""))
    zip_path = shutil.make_archive(base_name, "zip", root_dir=source_path)
    return Path(zip_path)


def _upload_zip(upload_cfg: ArtifactUploadConfig, token: str, zip_path: Path, artifact_name: str) -> None:
    if upload_cfg.create_repo:
        create_repo(
            repo_id=upload_cfg.repo_id,
            repo_type=upload_cfg.repo_type,
            token=token,
            exist_ok=True,
        )
    HfApi(token=token).upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=artifact_name,
        repo_id=upload_cfg.repo_id,
        repo_type=upload_cfg.repo_type,
    )


def upload_cv_fold_if_enabled(
    upload_cfg: Optional[ArtifactUploadConfig],
    train_cfg: TrainConfig,
    fold_dir: str | Path,
    archive_root: str | Path,
    fold_index: int,
    attempts: int = 3,
    retry_seconds: float = 30.0,
) -> Optional[str]:
    """Zip one finished cross-validation fold and upload it right away.

    The archive keeps ``fold_dir`` relative to ``archive_root`` (e.g.
    ``outputs/<experiment>/cross_validation/fold_02/...``), so unzipping it at
    the project root restores the fold where the CV summary looks for it. A
    failed upload is logged, not raised: the fold stays on disk and the
    remaining folds keep training.
    """
    if upload_cfg is None or not upload_cfg.enabled:
        logger.info("Artifact upload is disabled; fold_%02d is kept on disk only.", fold_index)
        return None
    token = get_token()
    if not token:
        logger.error("Hugging Face token not found; fold_%02d was not uploaded. Run 'hf auth login'.", fold_index)
        return None

    fold_path = Path(fold_dir).resolve()
    root_path = Path(archive_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = (
        f"MultimodalDL_{train_cfg.audio.backbone}_{train_cfg.video.backbone}_{train_cfg.fusion.type}_"
        f"cross_validation_fold_{fold_index:02d}_{timestamp}.zip"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(
            shutil.make_archive(
                str(Path(tmp_dir) / artifact_name[: -len(".zip")]),
                "zip",
                root_dir=root_path,
                base_dir=fold_path.relative_to(root_path).as_posix(),
            )
        )
        for attempt in range(1, attempts + 1):
            try:
                _upload_zip(upload_cfg, token, zip_path, artifact_name)
                logger.info("Uploaded fold_%02d to Hugging Face: %s/%s", fold_index, upload_cfg.repo_id, artifact_name)
                return artifact_name
            except Exception:
                logger.exception("Upload of fold_%02d failed (attempt %d/%d).", fold_index, attempt, attempts)
                if attempt < attempts:
                    time.sleep(retry_seconds * attempt)
    logger.error("fold_%02d was NOT uploaded; its results remain in %s.", fold_index, fold_path)
    return None


def upload_artifact_if_enabled(
    upload_cfg: ArtifactUploadConfig,
    train_cfg: TrainConfig,
    source_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    if not upload_cfg.enabled:
        logger.info("Artifact upload is disabled.")
        return None

    token = get_token()
    if not token:
        raise EnvironmentError(
            "Hugging Face authentication token was not found. Run 'hf auth login' in this environment first."
        )

    artifact_name = upload_cfg.path_in_repo or build_artifact_name(train_cfg)
    source_path = Path(source_dir or upload_cfg.source_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / artifact_name
        zip_source_tree(source_path, zip_path)
        _upload_zip(upload_cfg, token, zip_path, artifact_name)
        logger.info("Uploaded artifact to Hugging Face: %s/%s", upload_cfg.repo_id, artifact_name)
        return zip_path
