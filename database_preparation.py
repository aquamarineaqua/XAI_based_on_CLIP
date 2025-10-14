from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import medmnist  # type: ignore
    from medmnist import INFO  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "medmnist is required for the BloodMNIST helpers. Install it via `pip install medmnist`."
    ) from exc


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class PromptConfig:
    label_texts: Sequence[str]
    label_templates: Sequence[str]
    concept_texts: Sequence[str]
    concept_templates: Sequence[str]

    def build_template_map(self) -> Dict[str, List[str]]:
        template_texts: Dict[str, List[str]] = {}
        for idx, template in enumerate(self.concept_templates, start=1):
            template_texts[f"concept_prompts_t{idx:02d}"] = [template.format(text) for text in self.concept_texts]
        for idx, template in enumerate(self.label_templates, start=1):
            template_texts[f"label_prompts_t{idx:02d}"] = [template.format(text) for text in self.label_texts]
        return template_texts


def _ensure_float_tensor(x: Tensor | torch.Tensor) -> Tensor:
    tensor = x.float()
    if tensor.min() < 0.0 or tensor.max() > 1.0:
        tensor = tensor.clamp(0.0, 255.0) / 255.0
    return tensor


def _ensure_label_tensor(label: Tensor | Sequence[int] | int | None) -> Tensor:
    if label is None:
        return torch.zeros(1, dtype=torch.int64)
    if isinstance(label, Tensor):
        tensor = label.long()
    elif isinstance(label, Iterable) and not isinstance(label, (str, bytes)):
        tensor = torch.as_tensor(list(label), dtype=torch.int64)
    else:
        tensor = torch.tensor([int(label)], dtype=torch.int64)
    tensor = tensor.view(-1)
    if tensor.numel() == 0:
        tensor = torch.zeros(1, dtype=torch.int64)
    return tensor[:1]


def _default_collate(batch: Sequence[Tuple[Tensor, Tensor] | Tensor]) -> Tuple[Tensor, Tensor]:
    images: List[Tensor] = []
    labels: List[Tensor] = []
    for item in batch:
        if isinstance(item, tuple) and len(item) >= 2:
            image, label = item[0], item[1]
        else:
            image, label = item, None
        images.append(_ensure_float_tensor(torch.as_tensor(image)))
        labels.append(_ensure_label_tensor(torch.as_tensor(label) if label is not None else None))
    image_tensor = torch.stack(images, dim=0)
    label_tensor = torch.stack(labels, dim=0)
    return image_tensor, label_tensor


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_default_collate,
    )


def build_dataloaders(
    datasets: DatasetBundle,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    return {
        "train": build_dataloader(datasets.train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
        "val": build_dataloader(datasets.val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
        "test": build_dataloader(datasets.test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
    }


def load_bloodmnist_datasets(
    *,
    image_size: int = 224,
    as_rgb: bool = True,
    download: bool = True,
    data_root: Optional[str] = None,
) -> DatasetBundle:
    info = INFO["bloodmnist"]
    data_class = getattr(medmnist, info["python_class"])
    common_tf = transforms.Compose([transforms.ToTensor()])
    kwargs = {
        "size": image_size,
        "as_rgb": as_rgb,
        "mmap_mode": "r",
        "transform": common_tf,
        "download": download,
    }
    if data_root is not None:
        kwargs["root"] = data_root
    train_ds = data_class(split="train", **kwargs)
    test_ds = data_class(split="test", **kwargs)
    val_ds = data_class(split="val", **kwargs)
    return DatasetBundle(train=train_ds, val=val_ds, test=test_ds)


def get_bloodmnist_prompts() -> PromptConfig:
    info = INFO["bloodmnist"]
    label_map = {int(idx): name for idx, name in info["label"].items()}
    label_map[3] = "immature granulocytes"
    label_texts = [label_map[idx] for idx in sorted(label_map.keys())]
    cell_features = [
        "Segmented nucleus",
        "Band nucleus (band form)",
        "Reniform / indented nucleus",
        "Round nucleus",
        "Fine azurophilic granules",
        "Eosinophilic granules",
        "Basophilic granules",
        "Basophilic cytoplasm",
        "Cytoplasmic vacuoles",
        "High nuclear-to-cytoplasmic ratio",
        "Pale cytoplasm",
        "Nucleated erythrocyte (erythroblast)",
        "Platelet fragments / clumps",
        "Stain precipitate (artifact)",
        "Overlapping cell clumps (artifact)",
    ]
    concept_templates = [
        "a cell photo with sign of {}",
        "a photo of a cell with {}",
        "a cell image indicating {}",
        "an image of a cell showing {}",
        "blood cell with {}",
        "a blood cell photo with sign of {}",
        "a photo of a blood cell with {}",
        "a blood cell image indicating {}",
        "an image of blood cell showing {}",
    ]
    label_templates = [
        "a microscopic image of a {} cell",
        "a peripheral blood smear image of a {}",
        "a bloodcell of {}",
    ]
    return PromptConfig(
        label_texts=label_texts,
        label_templates=label_templates,
        concept_texts=cell_features,
        concept_templates=concept_templates,
    )


def prepare_bloodmnist_artifacts(
    *,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = False,
    download: bool = True,
    data_root: Optional[str] = None,
) -> Tuple[Dict[str, DataLoader], PromptConfig]:
    datasets = load_bloodmnist_datasets(
        image_size=image_size,
        as_rgb=True,
        download=download,
        data_root=data_root,
    )
    loaders = build_dataloaders(
        datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    prompts = get_bloodmnist_prompts()
    return loaders, prompts
