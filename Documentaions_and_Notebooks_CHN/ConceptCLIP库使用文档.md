ConceptCLIP使用文档

## 1 模型初始化
导入、初始化模型和GPU部署。

```python
# 导入库
from transformers import AutoModel, AutoProcessor
from huggingface_hub import login, whoami
login(token="<YOUR TOKEN>")
print(whoami())
```

```python
# 模型初始化，初次运行会下载模型权重
model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
```

```python
# 使用GPU推理
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
```



## 2 Processor中关于数据（图像、文本）预处理的设置

processor的处理代码一般为：

```python
inputs = processor(
    images=img_batch,
    text=texts,
    return_tensors='pt',
    padding=True,
    truncation=True,
).to(model.device)
```

注意，如果你的`img_batch`已经是形状为 [B, 3, 224, 224] 的tensor，并且像素值已经scale到 [0,1]，就不需要ConceptCLIPProcessor内置的 do_rescale 功能。关闭的方法：

```python
processor.image_processor.do_rescale = False  # already [0,1] scaled, so disable rescale in processor
processor.image_processor.do_normalize = True  # enable normalize in processor
```

此外，ConceptCLIP要求输入的图像尺寸是：384×384，所以不能改变processor的输出图像尺寸。

## 3 模型输入和输出结构解析

先经过ConceptCLIPProcessor处理后得到的`inputs`是一个dict：

例如，内容可为：

```
key name: input_ids, shape: torch.Size([15, 15])
key name: token_type_ids, shape: torch.Size([15, 15])
key name: attention_mask, shape: torch.Size([15, 15])
key name: pixel_values, shape: torch.Size([64, 3, 384, 384])
```

之后传入 model (ConceptCLIP对象) 进行推理，推理结果output可用于计算各文本prompt与图像的logits score，如：

```python
with torch.no_grad():
    outputs = model(**inputs)
    logits = (outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].t() + outputs['logit_bias'])
```

推理结果output是一个dict，内容可为：

```
Key name: image_features, shape: torch.Size([64, 1152])
Key name: text_features, shape: torch.Size([15, 1152])
Key name: logit_scale, shape: torch.Size([])
Key name: image_token_features, shape: torch.Size([64, 729, 1152])
Key name: text_token_features, shape: torch.Size([15, 15, 1152])
Key name: logit_bias, shape: torch.Size([])
Key name: concept_logit_scale, shape: torch.Size([])
Key name: concept_logit_bias, shape: torch.Size([])
```

64为传入的batch_size。

其中，`logit_scale`,`logit_bias`,`concept_logit_scale`,`concept_logit_bias` 为ConceptCLIP模型预训练后的内置参数，为标量。

`image_features`为图像编码器输出的向量，长度一般为1152。

`text_features`为文本编码器输出的向量，长度和image_features对齐，为1152。15是Prompt的数量，即你传入processor的texts的长度。

`image_token_features`为图像各个patch经过图像编码器输出的向量，这里分成了27×27=729个patch。

`text_token_features`用得较少，为Prompt中各token的编码向量，这里Prompt被分成了15个token。

## 4 如何单独使用文本编码器

我们先准备好Prompt列表`texts`，即将准备好的各个自然语言概念填入Prompt模板，如：

```
texts =
['a cell photo with sign of Segmented nucleus',
 'a cell photo with sign of Band nucleus (band form)',
 'a cell photo with sign of Reniform / indented nucleus',
 'a cell photo with sign of Round nucleus',
 'a cell photo with sign of Fine azurophilic granules',
 'a cell photo with sign of Eosinophilic granules',
 'a cell photo with sign of Basophilic granules',
 'a cell photo with sign of Basophilic cytoplasm',
 'a cell photo with sign of Cytoplasmic vacuoles']
```

---

然后参考如下代码：

```python
enc = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
print(enc.keys())
with torch.inference_mode():
    txt_cls, txt_tokens = model.encode_text(enc["input_ids"], normalize=True)  # [N, 1152], [N, 15, 768]
    txt_tokens_proj = None if txt_tokens is None else model.text_proj(txt_tokens)  # [N, 15, 1152]
```

其中 processor.tokenizer() 输出的 `enc` 是一个dict，如：

```
input_ids: torch.Size([15, 15])
token_type_ids: torch.Size([15, 15])
attention_mask: torch.Size([15, 15])
```

和前面processor输出的inputs相比只少了`pixel_values`字段。我们这里只使用`input_ids`字段作为文本编码器的输入。attention_mask和token_type_ids不需要传入，因为encode_text()函数用不上。

文本编码器 model.encode_text() 输出的 `txt_cls` 即为我们想要的 `text_features` 部分。

## 5 如何单独使用图像编码器

参考如下代码：

```python
inputs_imgs = processor(images=img_batch, return_tensors="pt", padding=True, truncation=True)

with torch.inference_mode():
    pixel = inputs_imgs["pixel_values"].to(device)
    img_cls, img_tokens = model.encode_image(pixel, normalize=True)   # [B,1152], [B,729,1152]

    # 若你希望 img_tokens 特征与 forward() 一致（进入过 image_proj MLP）：
    # img_tokens_proj 即为原model输出中的 image_token_features
    img_tokens_proj = model.image_proj(img_tokens)  # [B,729,1152]
```

其中，processor在只传入images的情况下，输出的字典只有"pixel_values"。

我们使用 model.encode_image() 图像编码器即可得到对应的编码向量和patches向量。

`img_cls`即与原输出字段 "image_features" 内容相同。

img_tokens需要再经过以此 model.image_proj() 映射得到`img_tokens_proj`，最终得到与原输出字段 "image_token_features" 相同的内容（把 patch 级视觉特征映射到“与文本对齐的概念空间”的投影头）。