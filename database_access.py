from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import json
import numpy as np


def _fmt_shape(shape) -> str:
    try:
        return "[" + ", ".join(str(int(x)) for x in shape) + "]"
    except Exception:  # pragma: no cover - defensive
        return "[]"


def _dtype(ds) -> str:
    try:
        return str(ds.dtype)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _is_vlen_str(ds) -> bool:
    try:
        dt = ds.dtype
        return (
            hasattr(dt, "metadata")
            and dt.metadata
            and dt.metadata.get("vlen") is str
        ) or str(dt).startswith("|S") or str(dt) == "object"
    except Exception:  # pragma: no cover - defensive
        return False


def _root_attr_keys(f: h5py.File) -> List[str]:
    preferred = [
        "D",
        "T_img",
        "created_at",
        "version",
        "split_counts",
        "logit_scale",
        "logit_bias",
        "concept_logit_scale",
        "concept_logit_bias",
        "label_texts",
        "prompt_temp_for_labels",
        "concept_texts",
        "prompt_temp_for_concepts",
    ]
    exists = [key for key in preferred if key in f.attrs]
    others = sorted([key for key in f.attrs.keys() if key not in exists])
    return exists + (["..."] if others else [])


def _template_attr_keys(g: h5py.Group) -> List[str]:
    preferred = ["K", "D", "T_txt", "texts_hash", "created_at"]
    exists = [key for key in preferred if key in g.attrs]
    others = sorted([key for key in g.attrs.keys() if key not in exists])
    return exists + (["..."] if others else [])


def _build_h5_tree(path: str | Path, max_templates: int = 3) -> List[str]:
    lines: List[str] = []
    with h5py.File(path, "r") as f:
        lines.append(Path(path).name)
        if "image_features" in f:
            lines.append(f"├── image_features               # {_fmt_shape(f['image_features'].shape)}")
        if "image_token_features" in f:
            lines.append(f"├── image_token_features         # {_fmt_shape(f['image_token_features'].shape)}")
        if "ids" in f:
            lines.append(f"├── ids                          # {_fmt_shape(f['ids'].shape)} int64")
        if "labels" in f:
            lines.append(f"├── labels                       # {_fmt_shape(f['labels'].shape)} int64")
        if "split" in f:
            lines.append(f"├── split                        # {_fmt_shape(f['split'].shape)}")
        attr_list = _root_attr_keys(f)
        if attr_list:
            lines.append("├── attrs: {" + ", ".join(attr_list) + "}")
        if "templates" in f:
            lines.append("└── templates/")
            template_names = sorted(list(f["templates"].keys()))
            display = template_names[:max_templates]
            for index, template_id in enumerate(display):
                group = f["templates"][template_id]
                last_template = (index == len(display) - 1) and (len(template_names) <= max_templates)
                branch = "└──" if last_template else "├──"
                lines.append(f"    {branch} {template_id}/")
                prefix = "        " if last_template else "    │   "
                if "text_features" in group:
                    lines.append(
                        f"{prefix}├── text_features        # {_fmt_shape(group['text_features'].shape)} ({_dtype(group['text_features'])})"
                    )
                if "text_token_features" in group:
                    lines.append(
                        f"{prefix}├── text_token_features  # {_fmt_shape(group['text_token_features'].shape)} ({_dtype(group['text_token_features'])})"
                    )
                if "texts" in group:
                    vlen = " (variable length string)" if _is_vlen_str(group["texts"]) else ""
                    lines.append(f"{prefix}├── texts                # {_fmt_shape(group['texts'].shape)}{vlen}")
                lines.append(f"{prefix}└── attrs: {{" + ", ".join(_template_attr_keys(group)) + "}}")
            if len(template_names) > max_templates:
                lines.append("    └── ...")
    return lines


def summarize_h5(path: str | Path = "./conceptclip_features.h5", *, max_templates: int = 3) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with h5py.File(file_path, "r") as f:
        image_shape = f["image_features"].shape if "image_features" in f else None
        ids_preview = f["ids"][:5] if "ids" in f else None
        try:
            split_counts = json.loads(f.attrs.get("split_counts", "{}"))
        except Exception:
            split_counts = {}
        templates = list(f["templates"].keys()) if "templates" in f else []
        print("image_features:", image_shape)
        print("ids preview (first 5):", ids_preview)
        print("split counts:", split_counts)
        print("templates:", templates[:max_templates] + (["..."] if len(templates) > max_templates else []))
    print("\nH5 structure:\n")
    for line in _build_h5_tree(file_path, max_templates=max_templates):
        print(line)


def clear_all_templates(path: str | Path = "./conceptclip_features.h5") -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with h5py.File(file_path, "a") as f:
        if "templates" not in f:
            print("No 'templates' group found. Nothing to clear.")
            return
        template_names = list(f["templates"].keys())
        del f["templates"]
        f.create_group("templates")
        f.flush()
        preview = template_names[:5]
        ellipsis = " ..." if len(template_names) > 5 else ""
        print(f"Cleared {len(template_names)} template groups: {preview}{ellipsis}")


def access_sample_by_index(index: int, path: str | Path = "./conceptclip_features.h5") -> Dict[str, np.ndarray | int | str]:
    with h5py.File(path, "r") as f:
        img_feat = f["image_features"][index]
        img_tokens = f["image_token_features"][index]
        sample_id = int(f["ids"][index])
        label = int(f["labels"][index])
        split_raw = f["split"][index]
        split_name = split_raw.decode("utf-8") if isinstance(split_raw, (bytes, bytearray)) else str(split_raw)
    print(
        f"sample_id: {sample_id}, label: {label}, split_name: {split_name}, "
        f"img_feat.shape: {img_feat.shape}, img_tokens.shape: {img_tokens.shape}"
    )
    return {
        "image_feature": img_feat,
        "image_token_feature": img_tokens,
        "id": sample_id,
        "label": label,
        "split": split_name,
    }


def _to_str_array(arr: np.ndarray) -> np.ndarray:
    return np.array([
        item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
        for item in arr
    ])


def access_batch(
    *,
    split: str = "all",
    idx=None,
    feature: str = "image",
    path: str | Path = "./conceptclip_features.h5",
) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    feature_key = {
        "image": "image_features",
        "patches": "image_token_features",
        "label": "labels",
    }.get(feature.lower())
    if feature_key is None:
        raise ValueError("feature must be one of 'image', 'patches', 'label'")
    with h5py.File(file_path, "r") as f:
        if feature_key not in f:
            raise KeyError(f"Dataset '{feature_key}' not found in file.")
        total = f[feature_key].shape[0]
        split_sel = split.lower() if split else "all"
        if split_sel == "all":
            indices = np.arange(total, dtype=np.int64)
        else:
            if "split" not in f:
                raise KeyError("Dataset 'split' not found for filtering.")
            split_arr = _to_str_array(f["split"][:])
            if split_sel not in {"train", "test", "val"}:
                raise ValueError("split must be one of 'all', 'train', 'test', 'val'")
            indices = np.flatnonzero(split_arr == split_sel).astype(np.int64)
        if idx is None:
            selected = indices
        elif isinstance(idx, slice):
            selected = indices[idx]
        else:
            arr = np.asarray(idx, dtype=np.int64)
            if arr.size > 0 and (arr.min() < 0 or arr.max() >= indices.shape[0]):
                raise IndexError("Index out of range for the selected split.")
            selected = indices[arr]
        result = f[feature_key][selected]
    return np.asarray(result)


def get_template_names(path: str | Path = "./conceptclip_features.h5") -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with h5py.File(file_path, "r") as f:
        return sorted(list(f["templates"].keys())) if "templates" in f else []


def _read_json_attr(f: h5py.File, key: str, default=None):
    if key not in f.attrs:
        return default
    raw = f.attrs[key]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return raw


def fetch_template_features(
    template_id: Optional[Sequence[str]] = None,
    *,
    is_concept: bool = True,
    concept_list: Optional[Sequence[str]] = None,
    is_label: bool = False,
    feature: str = "text",
    path: str | Path = "./conceptclip_features.h5",
) -> Dict[str, np.ndarray] | np.ndarray:
    if is_concept and is_label:
        raise ValueError("is_concept and is_label are mutually exclusive.")
    feature_key = {"text": "text_features", "tokens": "text_token_features"}.get(feature.lower())
    if feature_key is None:
        raise ValueError("feature must be 'text' or 'tokens'")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    requested = None if template_id is None else {str(t) for t in template_id}
    results: Dict[str, np.ndarray] = {}
    with h5py.File(file_path, "r") as f:
        if "templates" not in f:
            raise KeyError("'templates' group not found.")
        template_names = sorted(list(f["templates"].keys()))
        selected = template_names
        if is_concept:
            selected = [name for name in selected if name.startswith("concept")]
        elif is_label:
            selected = [name for name in selected if name.startswith("label")]
        if requested is not None:
            selected = [name for name in selected if name in requested]
            if not selected:
                raise ValueError("No templates match the provided identifiers and filters.")
        concept_idx = None
        if is_concept and concept_list:
            base_concepts = _read_json_attr(f, "concept_texts", default=None)
            if not base_concepts:
                raise KeyError("'concept_texts' attribute missing; cannot filter by concept_list.")
            positions = {str(name): idx for idx, name in enumerate(base_concepts)}
            keep = [positions[str(name)] for name in concept_list if str(name) in positions]
            if not keep:
                raise ValueError("concept_list did not match any stored concept_texts.")
            missing = [name for name in concept_list if str(name) not in positions]
            if missing:
                print(f"Warning: Concepts not found and ignored: {missing}")
            concept_idx = np.asarray(keep, dtype=np.int64)
        for name in selected:
            group = f["templates"][name]
            if feature_key not in group:
                continue
            arr = np.asarray(group[feature_key][:])
            if concept_idx is not None:
                arr = arr[concept_idx]
            results[name] = arr
    if not results:
        raise ValueError("No matching templates with the requested feature were found.")
    if len(results) == 1:
        return next(iter(results.values()))
    return results
