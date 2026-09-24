import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode


class ImageToPIL:
    """Convert one RGB image in [C, H, W] or [H, W, C] format to PIL."""
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            elif image.shape[-1] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            elif image.shape[0] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        return image


class VideoTransform:
    """
    Backward-compatible name for the single-frame image transform pipeline.

    Input:  np.ndarray [H, W, C] or [C, H, W] uint8
    Output: torch.Tensor [C, H, W] float32 normalized with ImageNet statistics.
    """
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(image)

    @staticmethod
    def get_transforms(image_size: int = 224):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = [
            ImageToPIL(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        eval_transform = [
            ImageToPIL(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        return {
            "train": VideoTransform(transforms.Compose(train_transform)),
            "val": VideoTransform(transforms.Compose(eval_transform)),
            "test": VideoTransform(transforms.Compose(eval_transform)),
        }


class VideoClipTransform:
    """Apply one spatial/color transform consistently to every frame in a clip."""

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = int(image_size)
        self.train = bool(train)

    def __call__(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            tensor = torch.from_numpy(frames)
        else:
            tensor = frames
        if tensor.ndim != 4:
            raise ValueError(f"Expected clip [T, C, H, W], got {tuple(tensor.shape)}.")
        if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.shape[1] != 3:
            raise ValueError(f"Expected RGB clips, got {tuple(tensor.shape)}.")
        tensor = tensor.to(torch.float32).div(255.0)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.train:
            # One draw per clip preserves motion instead of creating artificial
            # frame-to-frame jitter through independent image augmentation.
            if torch.rand(()) < 0.5:
                tensor = tensor.flip(-1)
            brightness = 0.85 + 0.30 * torch.rand(())
            tensor = (tensor * brightness).clamp(0.0, 1.0)
        mean = tensor.new_tensor(self.mean).view(1, 3, 1, 1)
        std = tensor.new_tensor(self.std).view(1, 3, 1, 1)
        return (tensor - mean) / std

    @classmethod
    def get_transforms(cls, image_size: int = 224):
        return {
            "train": cls(image_size=image_size, train=True),
            "val": cls(image_size=image_size, train=False),
            "test": cls(image_size=image_size, train=False),
        }
