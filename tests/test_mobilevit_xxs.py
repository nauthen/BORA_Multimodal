import pytest
import torch

timm = pytest.importorskip("timm")
if "mobilevit_xxs" not in timm.list_models("mobilevit_xxs"):
    pytest.skip("installed timm has no mobilevit_xxs", allow_module_level=True)

from models.multimodal_model import FeatureHook, _video_classifier_and_dim
from models.video import build_video_backbone
from utils.checkpoint_integrity import load_wrapped_single_modal_checkpoint


def test_mobilevit_xxs_feature_hook_exposes_pooled_features_and_logits() -> None:
    backbone = build_video_backbone("MobileViTXXS", classes_num=4, pretrained=False).eval()
    classifier, dim = _video_classifier_and_dim(backbone)
    assert classifier is backbone.model.head.fc
    assert dim == 320

    hook = FeatureHook(module=backbone, classifier=classifier, feature_dim=dim)
    frames = torch.randn(2 * 3, 3, 224, 224)
    with torch.no_grad():
        features = hook(frames)
        direct_logits = backbone(frames)
    assert features.shape == (6, 320)
    assert hook.last_logits.shape == (6, 4)
    torch.testing.assert_close(hook.last_logits, direct_logits)


def test_mobilevit_xxs_strict_loads_single_modal_wrapper_checkpoint(tmp_path) -> None:
    source = build_video_backbone("MobileViTXXS", classes_num=4, pretrained=False)
    state = {f"backbone.{key}": value for key, value in source.state_dict().items()}
    checkpoint = tmp_path / "video_best.pt"
    torch.save({"model_state_dict": state}, checkpoint)

    target = build_video_backbone("MobileViTXXS", classes_num=4, pretrained=False)
    load_wrapped_single_modal_checkpoint(checkpoint, {"backbone": target}, "Video")
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)
