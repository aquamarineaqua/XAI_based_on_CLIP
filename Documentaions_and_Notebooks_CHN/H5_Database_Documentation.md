# ConceptCLIP H5 数据库说明

本说明介绍 ConceptCLIP 特征库 `conceptclip_features.h5` 的结构、字段含义以及结合 Notebook / Python 脚本的常见读写方式。该文件使用 HDF5（Hierarchical Data Format v5）保存模型推理得到的图像与文本特征。

## 1 数据库概览

- **文件路径**：默认保存在 Notebook 或脚本所在目录下的 `./conceptclip_features.h5`，可通过 `OUT_PATH`（Notebook）或函数参数自定义。
- **生成方式**：
	- 运行 `[Notebook1]Generate_concept_features.ipynb`，按文本→图像的顺序执行单元。
	- 或调用脚本函数 `store_prompt_features()` / `encode_and_store_image_features()`（参见 `feature_extraction.py`、`feature_store_main.py`）。
- **应用场景**：缓存 BloodMNIST（或自定义数据集）中每个样本的图像 CLS / patch 特征，以及概念与标签 Prompt 的文本特征，支持概念检索与可解释分类。

## 2 文件结构

H5 文件由顶层数据集（datasets）、Prompt 模板组（group `templates/`）与全局属性（attributes）组成：

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

### 2.1 顶层数据集

| 名称 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `image_features` | `(N, 1152)` | `float16` | 每个样本的图像 CLS 向量（`model.encode_image(...)[0]`）。|
| `image_token_features` | `(N, 729, 1152)` | `float16` | 图像 27×27 patch 的特征，经过 `model.image_proj`。|
| `ids` | `(N,)` | `int64` | 自增样本 ID，保持写入顺序。|
| `labels` | `(N,)` | `int64` | 原始标签，若缺失则写入 `0`。|
| `split` | `(N,)` | UTF-8 可变长字符串 | 数据划分（`train` / `val` / `test` / 自定义）。缺省为 `"unspecified"`。|

写入时 `append_batch()` 会自动扩容数据集，校验标签/划分数量，并在缺失时填充默认值。

### 2.2 全局属性

| 属性名 | 示例 | 说明 |
| --- | --- | --- |
| `D` | `1152` | 图像 / 文本特征维度。|
| `T_img` | `729` | 图像 token 数量。|
| `created_at` | `YYYY-MM-DD HH:MM:SS` | 文件创建时间。|
| `version` | `1.0` | 自定义版本号。|
| `split_counts` | `{"train": 11959, "val": 3421, ...}` | 各划分写入样本数（JSON 字符串），支持断点续写。|
| `logit_scale`, `logit_bias` | `float` | CLIP 对齐 logits 参数。|
| `concept_logit_scale`, `concept_logit_bias` | `float` | 概念得分缩放 / 偏置。|
| `label_texts`, `prompt_temp_for_labels` | JSON 字符串 | 标签文本与对应 Prompt 模板列表。|
| `concept_texts`, `prompt_temp_for_concepts` | JSON 字符串 | 概念文本与对应 Prompt 模板列表。|

### 2.3 Prompt 模板组 `templates/`

每个模板 ID（`concept_prompts_tXX` / `label_prompts_tXX` / 自定义）对应一个子组，包含：

| 数据集 / 属性 | 类型 | 说明 |
| --- | --- | --- |
| `text_features` | `float32` dataset, `(K, 1152)` | Prompt 的文本 CLS 特征。|
| `text_token_features` | `float32` dataset, `(K, T_txt, 1152)`（可选） | 每个 token 的投影特征；若模型未提供 `text_proj` 则缺省。|
| `texts` | UTF-8 dataset, `(K,)` | Prompt 原文列表（如 “a cell photo with sign of Segmented nucleus”）。|
| `K` / `D` / `T_txt` | attribute | Prompt 数量、特征维度、token 序列长度（无 token 特征时为 `-1`）。|
| `texts_hash` | str | Prompt 文本列表的 SHA256 校验，便于变更检测。|
| `created_at` | str | 写入时间戳。

Notebook 默认的 Prompt 模板：

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


## 3 Notebook数据写入流程

`[Notebook1]Generate_concept_features.ipynb` 中的写入逻辑可以拆解为以下几步：

1. **准备数据集与 Prompt 元信息**：加载 BloodMNIST，构造 `train_loader` / `val_loader` / `test_loader`（torch.utils.data.DataLoader类，均采用 `batch_size=64` 且 `shuffle=False`，确保写入顺序稳定），并定义 `label_list`、`concept_list` 以及对应的 Prompt 模板列表。所有列表会被封装到 `DB_METADATA`，用于写入 H5 根属性。
2. **初始化或恢复 H5 框架**：`init_file()` 会在首次运行时创建顶层数据集与 `templates/` 组，设置 `D=1152`、`T_img=729` 等属性，并将 `labels` / `split` 填充默认值；再次运行时则会复用既有结构并补齐缺失的数据集，支持断点续写。
3. **批量写入文本模板特征**：结合 `concept_prompt_template_list` 与 `label_prompt_template_list` 生成 `template_texts_map`，随后利用 `model.encode_text(input_ids, normalize=True)` 得到 CLS 与 token 特征（若存在 `model.text_proj` 则对 token 进行投影），通过 `write_template()` 存入 `templates/{template_id}/`，并将 Prompt 原文、`texts_hash` 与模型 logits 相关参数一并写入根属性。
4. **批量写入图像与 token 特征**：`encode_and_store_image_features()` 逐个遍历数据划分，对每个 batch 调用 `model.encode_image(pixel_values, normalize=True)`，并在存在 `model.image_proj` 时写入其输出。该函数通过 `split_counts` 记录已处理样本数，并在重跑时根据计数与偏移逻辑跳过已写入的样本，以保证 `ids` 单调递增且与标签、划分对齐。
5. **验证写入结果**：`validate_h5()` 会打印图片/文本特征形状、`ids`/`labels` 预览、`split_counts`、Prompt 模板摘要以及保存的模型参数，可在每轮写入结束后执行。

若流程中断，只需重新运行文本或图像写入单元；`init_file()` 与 `split_counts` 会自动对齐已有数据，无需手动重建文件。

## 4 常用读取方式

详见`[Notebook2]Access_Database.ipynb`。

## 使用建议

- **批量与显存**：Notebook 默认 `batch_size=64`，可根据设备显存调低；若需要重新写入，只要保持 `shuffle=False` 即可让 `ids` 与标签顺序一致。
- **断点控制**：`split_counts` 会记录每个划分的写入量，重跑时可自动跳过已处理样本；若想彻底重建，可删除对应属性或整个文件，再执行 `init_file()`。
- **Prompt 迭代**：在更新 Prompt 模板或文本列表前，运行 `clear_all_templates()` 清空旧模板，再调用文本写入单元，避免混入历史数据；同一模板 ID 会被 `write_template()` 自动覆盖。
- **概念筛选**：`fetch_template_features()` 支持按 `concept_list` 选取子集，如有未命中将提示并忽略；使用时建议基于 `attrs['concept_texts']` 做校验。
- **存储精度与读取成本**：图像相关特征以 `float16` 存储、文本以 `float32` 存储；`image_token_features` 体量较大，建议按需切片读取或仅加载 CLS 特征以提升 I/O 效率。