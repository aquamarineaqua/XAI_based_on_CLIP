ConceptCLIP Usage and Tips

## 1. Model Initialization
Import, initialize the model, and move it to GPU.

```python
from transformers import AutoModel, AutoProcessor
from huggingface_hub import login, whoami
login(token="<YOUR_HF_TOKEN>")  # Replace with your HF token (do NOT hardcode in production)
print(whoami())
```

```python
# First-time execution will download model weights
model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
```

```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
```

---

## 2. Processor Settings for Image / Text Preprocessing

Typical usage pattern:

```python
inputs = processor(
    images=img_batch,   # List[Image] | torch.Tensor [B,3,H,W]
    text=texts,         # List[str]
    return_tensors='pt',
    padding=True,
    truncation=True,
).to(model.device)
```

If `img_batch` is already a float tensor of shape `[B, 3, 224, 224]` scaled to `[0,1]`, you can disable the internal `do_rescale` and keep normalization:

```python
processor.image_processor.do_rescale = False
processor.image_processor.do_normalize = True
```

---

## 3. Understanding Model Inputs and Outputs

After processing with the `processor`, the resulting `inputs` dict may contain:

```
input_ids:        torch.Size([15, 15])
token_type_ids:   torch.Size([15, 15])
attention_mask:   torch.Size([15, 15])
pixel_values:     torch.Size([64, 3, 384, 384])
```

Forward inference (joint text + image) for similarity / scoring:

```python
with torch.no_grad():
    outputs = model(**inputs)
    logits = (
        outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].T
        + outputs['logit_bias']
    )
```

The output dictionary may include:

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

- `64` is the image batch size; `15` is the number of prompts.
- `logit_scale`, `logit_bias`, `concept_logit_scale`, `concept_logit_bias` are learned scalar parameters.
- `image_features`: global image CLS embeddings (`D=1152`).
- `text_features`: global text CLS embeddings, same dimensionality.
- `image_token_features`: per‑patch embeddings (27×27 = 729 patches).
- `text_token_features`: per-token embeddings of each prompt (length depends on tokenizer; here `15`).

---

## 4. Using the Text Encoder Only

Prepare a list of prompts `texts` (concepts filled into prompt templates):

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

Encode text only (no images needed):

```python
enc = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
print(enc.keys())  # dict keys
with torch.inference_mode():
    text_cls, text_tokens = model.encode_text(enc["input_ids"], normalize=True)  # [N,1152], [N,T_txt,768]
    text_tokens_proj = None if text_tokens is None else model.text_proj(text_tokens)  # -> [N,T_txt,1152]
```

Tokenizer output (a dict) example:

```
input_ids:      torch.Size([15, 15])
token_type_ids: torch.Size([15, 15])
attention_mask: torch.Size([15, 15])
```

Only `input_ids` are required for `encode_text()` in ConceptCLIP; `attention_mask` and `token_type_ids` are **ignored** by the current implementation.
The returned `text_cls` corresponds to the `text_features` you would see in a full forward pass.

---

## 5. Using the Image Encoder Only

```python
inputs_imgs = processor(images=img_batch, return_tensors="pt", padding=True, truncation=True)

with torch.inference_mode():
    pixels = inputs_imgs["pixel_values"].to(device)
    img_cls, img_tokens = model.encode_image(pixels, normalize=True)   # [B,1152], [B,729,1152]

    # To match forward() output (after the projection MLP):
    img_tokens_proj = model.image_proj(img_tokens)  # -> image_token_features
```

When only images are passed to the processor, the returned dict contains just `pixel_values`. The `img_cls` tensor matches the `image_features` field; after projection, `img_tokens_proj` matches `image_token_features`.

---

## 6. Practical Tips & Gotchas

| Topic | Tip |
|-------|-----|
| Token lengths | Prompt token length varies with wording; monitor `input_ids.shape[1]` if memory is tight. |
| Normalization | `normalize=True` makes embeddings unit‑length—critical for cosine similarity based logits. |
| Projection layers | Use `text_proj` / `image_proj` outputs when you need consistency with end-to-end model logits. |
| Mixed precision | Storing image features as `float16` is usually safe; keep text in `float32` for marginally better stability. |

---

## 7. Minimal End-to-End Similarity Example

```python
from transformers import AutoModel, AutoProcessor
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)

texts = ["a cell photo with sign of Segmented nucleus", "a cell photo with sign of Round nucleus"]
images = [...]  # list of PIL Images or a float tensor [B,3,224,224] scaled to [0,1]

with torch.no_grad():
    proc = processor(images=images, text=texts, return_tensors='pt', padding=True, truncation=True).to(device)
    out = model(**proc)
    # Cosine-like logits already scaled
    sim = out['logit_scale'] * out['image_features'] @ out['text_features'].T + out['logit_bias']
print(sim.shape)  # [B, num_texts]
```

---

## 8. Glossary

| Term | Meaning |
|------|---------|
| CLS embedding | Global pooled representation for an image or a prompt. |
| Patch token | Representation of a spatial image patch (here 27×27 = 729). |
| Text token | Embedding of each token produced by the tokenizer. |
| Projection head | The final linear / MLP layer mapping encoder space to joint embedding space. |
| `logit_scale` / `logit_bias` | Learned scalar parameters modulating similarity logits. |
| Concept prompt | Natural language phrase describing a morphological feature or attribute. |

---