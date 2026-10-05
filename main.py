from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

from config import fold_config, load_artifact_upload_config, load_train_config
from dataset import create_dataloaders, load_splits
from tasks import MultimodalTrainer, load_cv_fold_results, save_cv_summary
from utils import set_seed
from utils.artifact_upload import upload_artifact_if_enabled, upload_cv_fold_if_enabled
from utils.checkpoint_integrity import validate_checkpoint_split_integrity


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
BORA_TYPES = {"bora_fusion", "temporal_bora_fusion"}


def _log_run_configuration(cfg, root: Path) -> None:
    cross_validation = cfg.evaluation_mode == "cross_validation"
    logger.info("==================================================")
    logger.info("Multimodal Deep Fusion configuration")
    logger.info("  - Evaluation mode:          %s", cfg.evaluation_mode)
    logger.info("  - Audio backbone:           %s", cfg.audio.backbone)
    logger.info("  - Video backbone:           %s", cfg.video.backbone)
    logger.info("  - Fusion head:              %s", cfg.fusion.type)
    if cross_validation:
        logger.info("  - CV folds to run:          %s of %d", cfg.dataset.selected_folds(), cfg.dataset.num_folds)
    if cfg.fusion.type in BORA_TYPES:
        logger.info(
            "  - Audio teacher:            %s",
            cfg.audio.cv_checkpoint_path if cross_validation else cfg.audio.checkpoint_path,
        )
        logger.info(
            "  - Video teacher:            %s",
            cfg.video.cv_checkpoint_path if cross_validation else cfg.video.checkpoint_path,
        )
    logger.info("  - Epochs:                   %s", cfg.epochs)
    logger.info("  - Batch size:               %s", cfg.batch_size)
    logger.info("  - Optimizer:                %s", cfg.optimizer)
    logger.info("  - Learning rate:            %s", cfg.learning_rate)
    logger.info("  - Early stopping:           %s", cfg.early_stopping)
    logger.info("  - Monitor metric:           %s", cfg.monitor)
    logger.info("  - Dataset path:             %s", cfg.dataset.dataset_path)
    logger.info("  - Split strategy:           %s", cfg.dataset.split_strategy)
    logger.info("  - Seed:                     %s", cfg.seed)
    logger.info("  - Output directory:         %s", root)
    logger.info("==================================================")


def _experiment_root(cfg, project_dir: Path) -> Path:
    name = f"{cfg.audio.backbone}_{cfg.video.backbone}_{cfg.fusion.type}"
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    return output_dir / name


def _run_holdout(cfg, root: Path) -> Dict[str, float]:
    logger.info("Starting holdout multimodal training.")
    splits = load_splits(cfg.dataset, mode="holdout")
    if cfg.fusion.type in BORA_TYPES:
        validate_checkpoint_split_integrity(cfg.audio.checkpoint_path, splits, "Audio")
        validate_checkpoint_split_integrity(cfg.video.checkpoint_path, splits, "Video")
    loaders = create_dataloaders(
        splits=splits,
        dataset_cfg=cfg.dataset,
        sample_rate=cfg.audio_features.sample_rate,
        image_size=cfg.video_features.image_size,
        batch_size=cfg.batch_size,
        num_frames=cfg.video_features.num_frames,
    )
    trainer = MultimodalTrainer(
        cfg=cfg,
        loaders=loaders,
        output_dir=root / "holdout",
        splits=splits,
        run_name="holdout",
    )
    return trainer.fit()


def _preflight_cross_validation_teachers(cfg, folds: List[int]) -> None:
    """Check the selected folds' teachers against their splits before any training starts."""
    for fold_index in folds:
        fold_cfg = fold_config(cfg, fold_index)
        splits = load_splits(cfg.dataset, mode="cross_validation", fold_index=fold_index)
        validate_checkpoint_split_integrity(fold_cfg.audio.checkpoint_path, splits, f"Audio fold_{fold_index:02d}")
        validate_checkpoint_split_integrity(fold_cfg.video.checkpoint_path, splits, f"Video fold_{fold_index:02d}")
    logger.info("Teacher checkpoints of folds %s match their fold splits.", folds)


def _run_cross_validation(cfg, root: Path, upload_cfg=None) -> Dict[int, Dict[str, float]]:
    folds = cfg.dataset.selected_folds()
    logger.info(
        "Starting cross-validation multimodal training: folds %s of %d.", folds, cfg.dataset.num_folds
    )
    if cfg.fusion.type in BORA_TYPES:
        _preflight_cross_validation_teachers(cfg, folds)
    cv_dir = root / "cross_validation"
    fold_results: Dict[int, Dict[str, float]] = {}
    for fold_index in folds:
        logger.info("Starting cross-validation fold_%02d.", fold_index)
        # Re-seed so each fold is reproducible on its own.
        set_seed(cfg.seed)
        fold_cfg = fold_config(cfg, fold_index)
        splits = load_splits(cfg.dataset, mode="cross_validation", fold_index=fold_index)
        loaders = create_dataloaders(
            splits=splits,
            dataset_cfg=cfg.dataset,
            sample_rate=cfg.audio_features.sample_rate,
            image_size=cfg.video_features.image_size,
            batch_size=cfg.batch_size,
            num_frames=cfg.video_features.num_frames,
        )
        fold_dir = cv_dir / f"fold_{fold_index:02d}"
        trainer = MultimodalTrainer(
            cfg=fold_cfg,
            loaders=loaders,
            output_dir=fold_dir,
            splits=splits,
            run_name=f"cross_validation_fold_{fold_index:02d}",
        )
        fold_results[fold_index] = trainer.fit()
        # Free this fold's RAM caches, loader workers and GPU memory before the
        # next fold builds its own; otherwise both folds' caches coexist.
        del trainer, loaders, splits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Push the finished fold now, so a lost server costs at most the running fold.
        upload_cv_fold_if_enabled(upload_cfg, fold_cfg, fold_dir, root.parent.parent, fold_index)
        save_cv_summary(load_cv_fold_results(cv_dir, cfg.dataset.num_folds), cv_dir, cfg.dataset.num_folds)
    return fold_results


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    cfg = load_train_config(project_dir / "config" / "train_config.json")
    upload_cfg = load_artifact_upload_config(project_dir / "config" / "artifact_upload_config.json")
    set_seed(cfg.seed)
    root = _experiment_root(cfg, project_dir)
    root.mkdir(parents=True, exist_ok=True)
    _log_run_configuration(cfg, root)

    if cfg.evaluation_mode == "holdout":
        _run_holdout(cfg, root)
    elif cfg.evaluation_mode == "cross_validation":
        _run_cross_validation(cfg, root, upload_cfg)

    upload_artifact_if_enabled(upload_cfg, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Multimodal deep fusion run failed.")
        sys.exit(1)
