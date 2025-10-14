from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from database_preparation import PromptConfig
from model_deployment import ModelArtifacts

D_DEFAULT = 1152
T_IMG_DEFAULT = 729
DTYPE_DYNAMIC = "float16"
DTYPE_TEXT = "float32"
DTYPE_STR = h5py.string_dtype(encoding="utf-8")


@dataclass
class FeatureStoreSummary:
    text_templates: Dict[str, int]
    image_samples_appended: Dict[str, int]
    total_new_samples: int


def _serialize_for_attr(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _read_json_attr(f: h5py.File, key: str, default=None):
    if key not in f.attrs:
        return default
    raw = f.attrs[key]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return raw


def _hash_texts(texts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def init_feature_store(
    path: str | os.PathLike,
    *,
    d: int = D_DEFAULT,
    t_img: int = T_IMG_DEFAULT,
    metadata: Optional[Mapping[str, object]] = None,
) -> h5py.File:
    file_path = Path(path)
    if not file_path.exists():
        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "image_features",
                shape=(0, d),
                maxshape=(None, d),
                chunks=(64, d),
                dtype=DTYPE_DYNAMIC,
                compression="lzf",
            )
            f.create_dataset(
                "image_token_features",
                shape=(0, t_img, d),
                maxshape=(None, t_img, d),
                chunks=(4, t_img, d),
                dtype=DTYPE_DYNAMIC,
                compression="lzf",
            )
            f.create_dataset(
                "ids",
                shape=(0,),
                maxshape=(None,),
                chunks=(4096,),
                dtype="int64",
                compression="lzf",
            )
            f.create_dataset(
                "labels",
                shape=(0,),
                maxshape=(None,),
                chunks=(4096,),
                dtype="int64",
                compression="lzf",
            )
            f.create_dataset(
                "split",
                shape=(0,),
                maxshape=(None,),
                chunks=(4096,),
                dtype=DTYPE_STR,
                compression="lzf",
            )
            f.create_group("templates")
            f.attrs["D"] = d
            f.attrs["T_img"] = t_img
            f.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.attrs["version"] = "1.0"
    f = h5py.File(file_path, "a")
    current_len = f["image_features"].shape[0]
    if "image_token_features" not in f:
        f.create_dataset(
            "image_token_features",
            shape=(current_len, t_img, d),
            maxshape=(None, t_img, d),
            chunks=(4, t_img, d),
            dtype=DTYPE_DYNAMIC,
            compression="lzf",
        )
    if "ids" not in f:
        f.create_dataset(
            "ids",
            shape=(current_len,),
            maxshape=(None,),
            chunks=(4096,),
            dtype="int64",
            compression="lzf",
        )
    if "labels" not in f:
        ds_labels = f.create_dataset(
            "labels",
            shape=(current_len,),
            maxshape=(None,),
            chunks=(4096,),
            dtype="int64",
            compression="lzf",
        )
        if current_len > 0:
            ds_labels[:] = np.full((current_len,), -1, dtype="int64")
    if "split" not in f:
        ds_split = f.create_dataset(
            "split",
            shape=(current_len,),
            maxshape=(None,),
            chunks=(4096,),
            dtype=DTYPE_STR,
            compression="lzf",
        )
        if current_len > 0:
            ds_split[:] = np.array(["unspecified"] * current_len, dtype=object)
    if "templates" not in f:
        f.create_group("templates")
    if metadata:
        for key, value in metadata.items():
            try:
                f.attrs[key] = _serialize_for_attr(value)
            except TypeError:
                f.attrs[key] = _serialize_for_attr(str(value))
    return f


def _to_np(x: Tensor | np.ndarray, dtype: str) -> np.ndarray:
    array = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return array.astype(dtype, copy=False)


def append_batch(
    f: h5py.File,
    *,
    image_feats: Tensor,
    image_token_feats: Tensor,
    ids: np.ndarray,
    split_names: Sequence[str],
    labels: Optional[np.ndarray] = None,
) -> None:
    ds_img = f["image_features"]
    ds_tok = f["image_token_features"]
    ds_ids = f["ids"]
    ds_labels = f["labels"]
    ds_split = f["split"]
    batch_size = image_feats.shape[0]
    start = ds_img.shape[0]
    end = start + batch_size
    ds_img.resize(end, axis=0)
    ds_tok.resize(end, axis=0)
    ds_ids.resize(end, axis=0)
    ds_labels.resize(end, axis=0)
    ds_split.resize(end, axis=0)
    ds_img[start:end, :] = _to_np(image_feats, DTYPE_DYNAMIC)
    ds_tok[start:end, :, :] = _to_np(image_token_feats, DTYPE_DYNAMIC)
    ds_ids[start:end] = ids.astype("int64", copy=False)
    if labels is None:
        labels_arr = np.full((batch_size,), -1, dtype="int64")
    else:
        labels_arr = np.asarray(labels, dtype="int64").reshape(-1)
        if labels_arr.shape[0] != batch_size:
            raise ValueError("Label count does not match batch size.")
    ds_labels[start:end] = labels_arr
    split_arr = np.asarray(list(split_names), dtype=object).reshape(-1)
    if split_arr.shape[0] != batch_size:
        raise ValueError("Split count does not match batch size.")
    ds_split[start:end] = split_arr
    f.flush()


def write_template(
    f: h5py.File,
    *,
    template_id: str,
    texts: Sequence[str],
    text_features: Tensor,
    text_token_features: Optional[Tensor] = None,
) -> None:
    root = f["templates"]
    if template_id in root:
        del root[template_id]
    group = root.create_group(template_id)
    tf = _to_np(text_features, DTYPE_TEXT)
    group.create_dataset("text_features", data=tf, compression="lzf")
    if text_token_features is not None:
        ttf = _to_np(text_token_features, DTYPE_TEXT)
        group.create_dataset("text_token_features", data=ttf, compression="lzf")
        t_txt = int(ttf.shape[1])
    else:
        t_txt = -1
    ds_txt = group.create_dataset(
        "texts",
        shape=(len(texts),),
        dtype=h5py.string_dtype(encoding="utf-8"),
        compression="lzf",
    )
    ds_txt[:] = np.array(list(texts), dtype=object)
    group.attrs["K"] = tf.shape[0]
    group.attrs["D"] = tf.shape[1]
    group.attrs["T_txt"] = t_txt
    group.attrs["texts_hash"] = _hash_texts(texts)
    group.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.flush()


def save_model_params(
    f: h5py.File,
    *,
    logit_scale: Optional[float] = None,
    logit_bias: Optional[float] = None,
    concept_logit_scale: Optional[float] = None,
    concept_logit_bias: Optional[float] = None,
) -> None:
    params = {
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
        "concept_logit_scale": concept_logit_scale,
        "concept_logit_bias": concept_logit_bias,
    }
    for key, value in params.items():
        if value is not None:
            f.attrs[key] = float(value)
    f.flush()


def load_model_params(f: h5py.File) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for key in ["logit_scale", "logit_bias", "concept_logit_scale", "concept_logit_bias"]:
        if key in f.attrs:
            params[key] = float(f.attrs[key])
    return params


def _maybe_to_scalar(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu().view(-1)[0].item())
        except Exception:  # pragma: no cover - defensive
            return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _model_param_snapshot(model: torch.nn.Module) -> Dict[str, float]:
    candidate_keys = ["logit_scale", "logit_bias", "concept_logit_scale", "concept_logit_bias"]
    snapshot: Dict[str, float] = {}
    for key in candidate_keys:
        if hasattr(model, key):
            scalar = _maybe_to_scalar(getattr(model, key))
            if scalar is not None:
                snapshot[key] = scalar
    return snapshot


def store_prompt_features(
    artifacts: ModelArtifacts,
    prompt_config: PromptConfig,
    *,
    output_path: str | os.PathLike = "./conceptclip_features.h5",
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, int]:
    template_map = prompt_config.build_template_map()
    metadata = {
        "label_texts": list(prompt_config.label_texts),
        "prompt_temp_for_labels": list(prompt_config.label_templates),
        "concept_texts": list(prompt_config.concept_texts),
        "prompt_temp_for_concepts": list(prompt_config.concept_templates),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    stored: Dict[str, int] = {}
    with init_feature_store(output_path, metadata=metadata) as f:
        with torch.no_grad():
            for template_id, texts in template_map.items():
                text_inputs = artifacts.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = text_inputs["input_ids"].to(artifacts.device)
                text_cls, text_tokens = artifacts.model.encode_text(input_ids, normalize=True)
                text_tokens_proj = (
                    artifacts.model.text_proj(text_tokens)
                    if hasattr(artifacts.model, "text_proj")
                    else text_tokens
                )
                write_template(
                    f,
                    template_id=template_id,
                    texts=texts,
                    text_features=text_cls,
                    text_token_features=text_tokens_proj,
                )
                stored[template_id] = text_cls.shape[0]
        save_model_params(f, **_model_param_snapshot(artifacts.model))
    return stored


def encode_and_store_image_features(
    artifacts: ModelArtifacts,
    split_loaders: Mapping[str, DataLoader],
    *,
    output_path: str | os.PathLike = "./conceptclip_features.h5",
    resume: bool = True,
    show_progress: bool = True,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, int]:
    metadata = extra_metadata or {}
    appended: Dict[str, int] = {}
    with init_feature_store(output_path, metadata=metadata) as f:
        if resume and "split_counts" in f.attrs:
            try:
                existing_counts = json.loads(f.attrs["split_counts"])
            except Exception:
                existing_counts = {}
        else:
            existing_counts = {}
        start_idx = f["image_features"].shape[0]
        for split_name, loader in split_loaders.items():
            recorded = int(existing_counts.get(split_name, 0) or 0) if resume else 0
            processed_in_split = 0
            iterator = tqdm(loader, desc=f"{split_name} split", leave=False) if show_progress else loader
            for images, labels in iterator:
                batch_total = images.shape[0]
                if resume and processed_in_split + batch_total <= recorded:
                    processed_in_split += batch_total
                    continue
                if resume and processed_in_split < recorded:
                    offset = recorded - processed_in_split
                    images = images[offset:]
                    labels = labels[offset:]
                    processed_in_split = recorded
                    batch_total = images.shape[0]
                    if batch_total == 0:
                        continue
                pixel_inputs = artifacts.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                pixel_values = pixel_inputs["pixel_values"].to(artifacts.device)
                with torch.no_grad():
                    img_cls, img_tokens = artifacts.model.encode_image(pixel_values, normalize=True)
                    img_tokens_proj = (
                        artifacts.model.image_proj(img_tokens)
                        if hasattr(artifacts.model, "image_proj")
                        else img_tokens
                    )
                batch_size = img_cls.shape[0]
                batch_ids = np.arange(start_idx, start_idx + batch_size, dtype=np.int64)
                labels_array = labels.view(batch_size, -1)[:, 0].cpu().numpy()
                append_batch(
                    f,
                    image_feats=img_cls,
                    image_token_feats=img_tokens_proj,
                    ids=batch_ids,
                    split_names=[split_name] * batch_size,
                    labels=labels_array,
                )
                start_idx += batch_size
                processed_in_split += batch_size
                appended[split_name] = appended.get(split_name, 0) + batch_size
            if split_name in appended:
                existing_counts[split_name] = int(existing_counts.get(split_name, 0) or 0) + appended[split_name]
        f.attrs["split_counts"] = json.dumps(existing_counts)
        f.flush()
    return appended


def run_full_pipeline(
    artifacts: ModelArtifacts,
    prompt_config: PromptConfig,
    loaders: Mapping[str, DataLoader],
    *,
    output_path: str | os.PathLike = "./conceptclip_features.h5",
    resume: bool = True,
    show_progress: bool = True,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> FeatureStoreSummary:
    stored_templates = store_prompt_features(
        artifacts,
        prompt_config,
        output_path=output_path,
        extra_metadata=extra_metadata,
    )
    image_counts = encode_and_store_image_features(
        artifacts,
        loaders,
        output_path=output_path,
        resume=resume,
        show_progress=show_progress,
        extra_metadata=extra_metadata,
    )
    return FeatureStoreSummary(
        text_templates=stored_templates,
        image_samples_appended=image_counts,
        total_new_samples=sum(image_counts.values()),
    )
