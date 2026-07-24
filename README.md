# classification-LQ_TinyClassifier

基于 LQ_TinyClassifier（龙邱科技）的分类训练工程，支持两种 backbone 架构：

- **`tiny_custom`** — 原深度可分离卷积架构（~4K 参数），训练速度快、模型极小
- **`mobilenet_v2`** — MobileNetV2（~2.2M 参数），精度更高，适合对算力要求更高的场景

两者完全独立、互不影响。原始代码未做任何修改。

---

## 目录结构

```
classification-LQ_TinyClassifier/
├── train.ipynb                          ← 统一的 Notebook，支持两种 backbone 切换
├── LQ_TinyClassifier/                   ← 原始 tiny_custom 项目（未修改）
│   ├── train_tiny_classifier.py         ← 训练脚本（V1.0.0 原始版本）
│   ├── evaluate_local_accuracy.py       ← 本地精度复核
│   ├── evaluate_random_subset_accuracy.py
│   ├── convert_to_ncnn.ps1              ← ONNX 转 NCNN（Windows）
│   ├── run_all.ps1                      ← 一键训练 + 导出
│   ├── artifacts/                       ← 训练产物（best_model.pt, ONNX, labels.txt, metrics.json）
│   └── datasets/datas/                  ← 数据集（9 个类别）
├── LQ_TinyClassifier_MobileNetV2/       ← MobileNetV2 独立模块
│   ├── train_mobilenet_v2.py            ← 独立训练脚本（自包含，不依赖原始代码）
│   └── artifacts_mobilenetv2/           ← MobileNetV2 训练产物
└── README.md
```

---

## 使用方式

在工程根目录下打开 `train.ipynb`，直接修改 `CFG` 中的参数（`backbone`、`width_mult` 等均可自由修改，不会被强制覆盖）：

```python
CFG = {
    "backbone": "tiny_custom",    # ← 切换： "tiny_custom" 或 "mobilenet_v2"
    "width_mult": 0.35,           # ← 可在此自由修改，setdefault 仅设默认值
    "img_size": 96,
    "batch_size": 64,
    "epochs": 50,
    "lr": 5e-4,
    ...
}
```

- `"tiny_custom"` → 使用 `LQ_TinyClassifier.train_tiny_classifier.TinyClassifier`（原始架构）
- `"mobilenet_v2"` → 使用 `MobileNetV2Classifier`（从 `train_mobilenet_v2.py` 导入）

训练产物根据 `backbone` 自动输出到各自的文件夹：
- `"tiny_custom"` → `LQ_TinyClassifier/artifacts/`
- `"mobilenet_v2"` → `LQ_TinyClassifier_MobileNetV2/artifacts_mobilenetv2/`

ONNX 文件名也自动区分：
- `"tiny_custom"` → `tiny_classifier_fp32.onnx`
- `"mobilenet_v2"` → `mobilenet_v2_fp32.onnx`

---

## 数据集

数据集路径：`LQ_TinyClassifier/datasets/datas/`

格式为 ImageFolder 结构（无需额外配置，Notebook 中直接引用）：

```
LQ_TinyClassifier/datasets/datas/
├── 00mickey_mouse/
├── 01pikachu/
├── 02spongebob_squarepants/
├── ...
└── 09grey_wolf/
```

---

## 转换 ncnn

训练后生成的 ONNX 文件（opset 12，已 onnxsim 简化）可通过 `onnx2ncnn` 或 `pnnx` 转换为 ncnn 格式。  
MobileNetV2 的所有算子均被 ncnn 原生支持，转换无额外兼容性问题。