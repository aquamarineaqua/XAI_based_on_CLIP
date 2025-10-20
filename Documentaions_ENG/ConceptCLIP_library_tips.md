ConceptCLIP Usage Guide

## 1. Model Initialization
Import, initialize the model, and deploy to GPU.

```python
from transformers import AutoModel, AutoProcessor
from huggingface_hub import login, whoami
login(token="<YOUR_HF_TOKEN>")
print(whoami())
```

```python
# First run downloads the weights
model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
```

```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
```

---

## 2. Processor Settings for Preprocessing (Images & Text)

Typical call:

```python
inputs = processor(
    images=img_batch,
    text=texts,
    return_tensors='pt',
    padding=True,
    truncation=True,
).to(model.device)
```

If your `img_batch` is already a float tensor of shape `[B, 3, 224, 224]` scaled to `[0,1]`, disable rescaling and keep normalization:

```python
processor.image_processor.do_rescale = False
processor.image_processor.do_normalize = True
```

Important: ConceptCLIP expects input images of size `384×384`. Do not change the processor’s output resolution; rely on the processor to produce `pixel_values` shaped `[B,3,384,384]`.

---

## 3. Inputs/Outputs Explained

After `processor`, `inputs` is a dict, e.g.:

```
input_ids:      torch.Size([15, 15])
token_type_ids: torch.Size([15, 15])
attention_mask: torch.Size([15, 15])
pixel_values:   torch.Size([64, 3, 384, 384])
```

Forward inference for text–image similarity logits:

```python
with torch.no_grad():
    outputs = model(**inputs)
    logits = (
        outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].t()
        + outputs['logit_bias']
    )
```

The output dict may contain:

```
image_features:        torch.Size([64, 1152])
text_features:         torch.Size([15, 1152])
logit_scale:           torch.Size([])
image_token_features:  torch.Size([64, 729, 1152])
text_token_features:   torch.Size([15, 15, 1152])
logit_bias:            torch.Size([])
concept_logit_scale:   torch.Size([])
concept_logit_bias:    torch.Size([])
```

Notes:

- `64` is the batch size; `15` is the number of prompts.
- `logit_scale`, `logit_bias`, `concept_logit_scale`, `concept_logit_bias` are learned scalar parameters.
- `image_features`: global image CLS embeddings (`D=1152`).
- `text_features`: global text CLS embeddings (`D=1152`).
- `image_token_features`: per‑patch features (27×27 = 729 patches).
- `text_token_features`: per‑token features for each prompt (length depends on tokenizer; example shows 15).

---

## 4. Text Encoder Only

Prepare `texts` by filling concepts into prompt templates:

```python
texts = [
    'a cell photo with sign of Segmented nucleus',
    'a cell photo with sign of Band nucleus (band form)',
    'a cell photo with sign of Reniform / indented nucleus',
    'a cell photo with sign of Round nucleus',
    'a cell photo with sign of Fine azurophilic granules',
    'a cell photo with sign of Eosinophilic granules',
    'a cell photo with sign of Basophilic granules',
    'a cell photo with sign of Basophilic cytoplasm',
    'a cell photo with sign of Cytoplasmic vacuoles',
]
```

Then encode:

```python
enc = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
print(enc.keys())
with torch.inference_mode():
    text_cls, text_tokens = model.encode_text(enc["input_ids"], normalize=True)  # [N,1152], [N,T_txt,768]
    text_tokens_proj = None if text_tokens is None else model.text_proj(text_tokens)  # -> [N,T_txt,1152]
```

Tokenizer output example:

```
input_ids:      torch.Size([15, 15])
token_type_ids: torch.Size([15, 15])
attention_mask: torch.Size([15, 15])
```

Only `input_ids` are required for `encode_text()` in ConceptCLIP; `attention_mask` and `token_type_ids` are not used in the current signature. The `text_cls` is what you would see as `text_features` in the full forward pass.

---

## 5. Image Encoder Only

```python
inputs_imgs = processor(images=img_batch, return_tensors="pt", padding=True, truncation=True)

with torch.inference_mode():
    pixels = inputs_imgs["pixel_values"].to(device)
    img_cls, img_tokens = model.encode_image(pixels, normalize=True)   # [B,1152], [B,729,1152]

    # To match forward(): project patch features into the joint concept space
    img_tokens_proj = model.image_proj(img_tokens)  # -> image_token_features
```

When only images are provided, the processor dict contains `pixel_values` only. `img_cls` corresponds to `image_features`. The projected `img_tokens_proj` matches `image_token_features` (patch‑level embeddings mapped to the text‑aligned concept space by the projection head).
