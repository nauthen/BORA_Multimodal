import torch.nn as nn
import timm


class MobileViTXXS(nn.Module):
    """
    MobileViT-XXS image classifier using timm.

    Mirrors the single-modal U_FFIA27K_video wrapper so its checkpoint keys
    (``backbone.model.*``) strict-load unchanged. The classifier is
    ``model.head.fc`` (320 -> classes_num) after global average pooling.
    """
    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        self.model = timm.create_model("mobilevit_xxs", pretrained=pretrained, num_classes=classes_num)
        self.model_name = "mobilevit_xxs"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
