from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import device as TorchDevice, nn
from transformers import AutoModel, AutoProcessor

try:
    from huggingface_hub import login as hf_login
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "huggingface_hub is required for loading ConceptCLIP. Install it via `pip install huggingface_hub`."
    ) from exc


@dataclass
class ModelArtifacts:
    model: nn.Module
    processor: AutoProcessor
    device: TorchDevice


def login_to_hf(token: str, *, add_to_git_credential: bool = False) -> None:
    hf_login(token=token, add_to_git_credential=add_to_git_credential)


def resolve_device(explicit: Optional[str | TorchDevice] = None) -> TorchDevice:
    if explicit is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(explicit)


def load_conceptclip_model(
    *,
    hf_token: Optional[str] = None,
    device: Optional[str | TorchDevice] = None,
    trust_remote_code: bool = True,
) -> ModelArtifacts:
    resolved_device = resolve_device(device)
    if hf_token:
        login_to_hf(hf_token)
    model = AutoModel.from_pretrained(
        "JerrryNie/ConceptCLIP",
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(
        "JerrryNie/ConceptCLIP",
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    processor.image_processor.do_rescale = False
    processor.image_processor.do_normalize = True
    model = model.to(resolved_device).eval()
    return ModelArtifacts(model=model, processor=processor, device=resolved_device)
