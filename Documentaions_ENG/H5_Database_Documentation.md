# ConceptCLIP H5 Feature Store (HDF5 Database) Documentation

This document describes the structure and semantics of the ConceptCLIP feature store `conceptclip_features.h5`, along with common read / write patterns using the provided Notebooks and Python modules. The file leverages HDF5 (Hierarchical Data Format v5) to persist image and text features produced by the model.

## 1. Overview

- **File path**: By default `./conceptclip_features.h5` in the working directory of the notebook or scripts. Can be customized via `OUT_PATH` (Notebook) or function arguments.
- **Generation**:
  - Run `[Notebook1]Generate_concept_features.ipynb` (execute text → image sections in order), or
  - Call `store_prompt_features()` / `encode_and_store_image_features()` (see `feature_extraction.py`, `feature_store_main.py`).
- **Use cases**: Cache global CLS & patch (token) features for each sample (BloodMNIST or custom dataset), plus concept & label prompt text features, enabling concept retrieval and explainable classification.

## 2. File Structure

The H5 file contains top‑level datasets, a prompt template group (`templates/`), and root attributes:

```
conceptclip_features.h5
├── image_features               # [N, 1152]
├── image_token_features         # [N, 729, 1152]
├── ids                          # [N] int64
├── labels                       # [N] int64
├── split                        # [N]
├── attrs: {D, T_img, created_at, version, split_counts,
│          logit_scale, logit_bias,
│          concept_logit_scale, concept_logit_bias,
│          label_texts, prompt_temp_for_labels,
│          concept_texts, prompt_temp_for_concepts, ...}
└── templates/
    ├── concept_prompts_t01/
    │   ├── text_features        # [K_concept, 1152] (float32)
    │   ├── text_token_features  # [K_concept, T_txt, 1152] (float32, optional)
    │   ├── texts                # [K_concept]
    │   └── attrs: {K, D, T_txt, texts_hash, created_at, ...}
    ├── ...
    └── label_prompts_t01/
        ├── text_features        # [K_label, 1152] (float32)
        ├── text_token_features  # [K_label, T_txt, 1152] (float32, optional)
        ├── texts                # [K_label]
        └── attrs: {K, D, T_txt, texts_hash, created_at, ...}
```

### 2.1 Top‑Level Datasets

| Name | Shape | dtype | Description |
| ---- | ----- | ----- | ----------- |
| `image_features` | `(N, 1152)` | `float16` | Global image CLS embeddings (`model.encode_image(...)[0]`). |
| `image_token_features` | `(N, 729, 1152)` | `float16` | 27×27 patch (token) embeddings (after `model.image_proj`). |
| `ids` | `(N,)` | `int64` | Incremental sample IDs in write order. |
| `labels` | `(N,)` | `int64` | Original labels (write fallback if absent: `0` or custom). |
| `split` | `(N,)` | UTF‑8 vlen string | Data split (`train` / `val` / `test` / custom); default `"unspecified"`. |

`append_batch()` auto‑extends datasets, validates counts, and fills defaults if split / labels are missing.

### 2.2 Root Attributes

| Attribute | Example | Description |
| --------- | ------- | ----------- |
| `D` | `1152` | Shared embedding dimension (image & text). |
| `T_img` | `729` | Number of image tokens (patches). |
| `created_at` | `YYYY-MM-DD HH:MM:SS` | File creation timestamp. |
| `version` | `1.0` | Custom version tag. |
| `split_counts` | `{"train": 11959, "val": 3421, ...}` | Per-split appended counts (JSON); enables resume. |
| `logit_scale`, `logit_bias` | `float` | CLIP alignment logit parameters. |
| `concept_logit_scale`, `concept_logit_bias` | `float` | Concept scoring scale / bias. |
| `label_texts`, `prompt_temp_for_labels` | JSON string | Label vocabulary & associated prompt templates. |
| `concept_texts`, `prompt_temp_for_concepts` | JSON string | Concept vocabulary & associated prompt templates. |

### 2.3 Template Groups (`templates/`)

Each template ID (`concept_prompts_tXX`, `label_prompts_tXX`, or custom) is a subgroup containing:

| Dataset / Attr | Type | Description |
| -------------- | ---- | ----------- |
| `text_features` | float32, `(K, 1152)` | CLS embeddings for each generated prompt. |
| `text_token_features` | float32, `(K, T_txt, 1152)` (optional) | Per-token projected embeddings; omitted if no `text_proj`. |
| `texts` | UTF‑8, `(K,)` | Original prompt strings (e.g. `"a cell photo with sign of Segmented nucleus"`). |
| `K` / `D` / `T_txt` | attrs | Prompt count, embedding dim, token length (`-1` if absent). |
| `texts_hash` | str | SHA256 hash of the prompt list (change detection). |
| `created_at` | str | Write timestamp. |

Default prompt template lists (example from Notebook):

```python
concept_prompt_template_list = [
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

label_prompt_template_list = [
    "a microscopic image of a {} cell",
    "a peripheral blood smear image of a {}",
    "a bloodcell of {}",
]
```

## 3. Notebook Write Workflow

From `[Notebook1]Generate_concept_features.ipynb` the pipeline consists of:

1. **Prepare dataset & prompt metadata**: Load BloodMNIST; build `train_loader` / `val_loader` / `test_loader` (`batch_size=64`, `shuffle=False` for deterministic ID ordering); define `label_list`, `concept_list`, and prompt template lists; pack into `DB_METADATA` root attributes.
2. **Initialize or resume H5**: `init_file()` creates core datasets & `templates/` group with `D=1152`, `T_img=729`; fills defaults for `labels` / `split`; later runs reuse / repair structure (supports interruption recovery).
3. **Write text template features**: Combine template lists with vocab to form `template_texts_map`; call `model.encode_text(input_ids, normalize=True)` to obtain CLS & token features (project tokens if `model.text_proj` exists); persist via `write_template()` including `texts_hash` and model logit params.
4. **Write image & token features**: `encode_and_store_image_features()` iterates splits, calling `model.encode_image(pixel_values, normalize=True)`; applies `model.image_proj` if present; uses `split_counts` and intra-batch offset logic to skip previously written samples—ensuring monotonic `ids` aligned with labels & splits.
5. **Validate**: `validate_h5()` prints shape summaries, `ids` / `labels` preview, `split_counts`, template snapshot & stored model params.

If interrupted, rerun text or image steps—`init_file()` + `split_counts` align incremental state without manual rebuild.

## 4. Reading Patterns

See `[Notebook2]Access_Database.ipynb` for concrete usage (overview, sample & batch access, template filtering, feature retrieval).

## 5. Usage Recommendations

- **Batch size & VRAM**: Default `batch_size=64`; reduce if memory constrained. Keep `shuffle=False` to maintain consistent `ids` / label alignment when resuming.
- **Resume logic**: `split_counts` tracks per-split progress; to fully rebuild, delete the file or remove the attribute before re-running `init_file()`.
- **Prompt iteration**: Run `clear_all_templates()` prior to rewriting modified prompt lists to avoid legacy data contamination; `write_template()` overwrites same IDs.
- **Concept subsetting**: `fetch_template_features()` supports `concept_list` filtering; unmatched names are warned & skipped—verify against `attrs['concept_texts']`.
- **Precision & I/O**: Store image features (including tokens) as `float16`; retain text in `float32` for stability. Given the size of `image_token_features`, prefer selective slicing or CLS-only access first for faster reads.
