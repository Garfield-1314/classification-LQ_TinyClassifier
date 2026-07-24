# classification-LQ_TinyClassifier

基于 LQ_TinyClassifier（龙邱科技）的图像分类训练工程。从 PC 训练到嵌入式部署（Seekfree LS2K0300）的全流程解决方案。

> **注意**：`LQ_TinyClassifier/` 是 git 子模块（仓库：https://gitee.com/lq-tech/LQ_TinyClassifier.git），clone 后需执行初始化命令拉取其内容，详见下方[环境安装](#环境安装)。

支持两种 backbone 架构，在同一个 Notebook 中自由切换：

| Backbone | 参数量 | 特点 | 适用场景 |
|----------|--------|------|----------|
| **`tiny_custom`** | ~4K | 超轻量深度可分离卷积，训练极快 | 算力受限的嵌入式设备 |
| **`mobilenet_v2`** | ~2.2M | 基于 torchvision 官方 MobileNetV2，精度更高 | 对精度要求更高的场景 |

两者完全独立、互不影响，原始代码未做任何修改。

---

## 克隆与初始化

```bash
# 克隆仓库（含子模块）
git clone --recurse-submodules git@github.com:Garfield-1314/classification-LQ_TinyClassifier.git

# 如果已克隆但未拉取子模块，执行：
git submodule update --init --recursive
```

`LQ_TinyClassifier/` 是 git 子模块（远程仓库：https://gitee.com/lq-tech/LQ_TinyClassifier.git），**必须初始化**后该目录才会有 `train_tiny_classifier.py` 等文件，否则 notebook 会报 `ModuleNotFoundError: No module named 'train_tiny_classifier'`。

## 环境安装

### 方式一：Conda（推荐）

```bash
# 创建虚拟环境
conda create -n tinycls python=3.11 -y
conda activate tinycls

# 安装依赖（宽松版本，兼容性好）
pip install -r requirements.txt

# （可选）安装 Jupyter Notebook
conda install -n tinycls jupyter -y
```

### 方式二：pip venv

```bash
python -m venv .venv
source .venv/bin/activate    # Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

> **依赖说明**：根目录 `requirements.txt` 使用 `>=` 宽松约束，适配不同环境。
> 如需严格复现训练环境，可使用 `LQ_TinyClassifier/requirements.txt`（conda 导出固定版本）。

### 启动 Notebook

```bash
conda activate tinycls    # 或 source .venv/bin/activate
jupyter notebook
```

打开浏览器，进入 `train.ipynb`。

---

## 目录结构

```
classification-LQ_TinyClassifier/
│
├── train.ipynb                      ← 统一训练 Notebook（支持两种 backbone 切换）
├── requirements.txt                 ← 宽松版本 Python 依赖
├── .gitignore                       ← Git 忽略规则
├── README.md                        ← 本文件
│
├── datasets/                        ← 数据集目录（ImageFolder 格式，不提交 git）
│   ├── 1/                           ← 类别 1 图片
│   ├── 2/                           ← 类别 2 图片
│   └── 3/                           ← 类别 3 图片
│
├── LQ_TinyClassifier/               ← 原始 tiny_custom 项目（git 子模块，未修改）
│   ├── train_tiny_classifier.py     ← tiny_custom 独立训练脚本
│   ├── evaluate_local_accuracy.py   ← 本地精度复核
│   ├── evaluate_random_subset_accuracy.py  ← 随机抽样精度验证
│   ├── convert_to_ncnn.ps1          ← ONNX 转 NCNN（Windows）
│   ├── run_all.ps1                  ← 一键训练 + NCNN 导出（Windows）
│   ├── requirements.txt             ← 固定版本依赖（conda 导出）
│   ├── README.md                    ← 原始项目说明
│   ├── LICENSE                      ← GPL-3.0 许可证
│   ├── artifacts/                   ← tiny_custom 训练产物
│   │   ├── best_model.pt            ← PyTorch 模型权重
│   │   ├── tiny_classifier_fp32.onnx ← ONNX 格式模型
│   │   ├── tiny_classifier_fp32.ncnn.param  ← NCNN 模型参数
│   │   ├── tiny_classifier_fp32.ncnn.bin    ← NCNN 模型权重
│   │   ├── labels.txt               ← 类别标签（每行一个）
│   │   ├── metrics.json             ← 训练指标
│   │   └── random_sample_metrics.json ← 随机抽样指标
│   └── pnnx-20260112-windows/       ← PNNX 工具链（Windows）
│
├── LQ_TinyClassifier_MobileNetV2/   ← MobileNetV2 独立模块
│   ├── train_mobilenet_v2.py        ← MobileNetV2 独立训练脚本
│   └── artifacts_mobilenetv2/       ← MobileNetV2 训练产物
│       ├── best_model.pt
│       ├── mobilenet_v2_fp32.onnx
│       ├── labels.txt
│       └── metrics.json
│
└── Seekfree_LS2K0300_Opensource_Library/  ← LS2K0300 开发板 SDK
    ├── libraries/                   ← 驱动库（ncnn、tflm 等）
    ├── project/
    │   ├── code/                    ← 板端推理代码（lq_ncnn.cpp 等）
    │   ├── model/                   ← 板端模型文件
    │   └── user/                    ← 用户程序入口（main.cpp）
    └── ...
```

---

## 数据集准备

数据集采用 ImageFolder 格式（与 torchvision 兼容）：

```
datasets/
├── class_a/        ← 类别名可自定义
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── class_b/
│   ├── img_001.jpg
│   └── ...
└── class_c/
    └── ...
```

要求：
- 每个子目录代表一个类别
- 支持图片格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
- 至少需 2 个类别
- 类别索引按文件夹名字**字典序**自动生成

**使用方式**：
1. 直接将数据集放在 `datasets/` 目录下
2. 或创建软链接：`ln -s /path/to/your/dataset datasets`
3. 或在 Notebook 中修改 `CFG["data_root"]` 指向你的路径

---

## 训练流程

### 方式一：Notebook（推荐）

在工程根目录打开 `train.ipynb`，按 Cell 顺序执行：

**Step 1**：修改训练参数

```python
CFG = {
    "data_root":     "datasets",       # 数据集路径
    "img_size":      96,               # 输入图片尺寸
    "batch_size":    64,               # Batch size
    "epochs":        50,               # 最大训练轮数
    "lr":            5e-4,             # 学习率
    "weight_decay":  1e-4,             # 权重衰减
    "num_workers":   4,                # 数据加载线程数
    "seed":          42,               # 随机种子
    "width_mult":    0.35,             # 宽度缩放（tiny_custom 默认 0.35，mobilenet_v2 默认 1.0）
    "backbone":      "tiny_custom",    # ← 切换 backbone: "tiny_custom" / "mobilenet_v2"
    "patience":      10,               # 早停耐心值
    "target_acc":    0.98,             # 目标准确率（达标后提前停止）
}
```

**Step 2**：按顺序执行各 Cell
1. **导入依赖** — 加载 torch、numpy 等
2. **检查数据集** — 显示类别数和各类别图片数
3. **参数配置 + 加载数据** — 自动构建训练/验证/测试集
4. **创建模型** — 根据 backbone 选择模型类
5. **训练循环** — 含早停、学习率余弦退火、自动保存最佳模型
6. **测试集评估** — 加载最佳 checkpoint 评估
7. **导出 ONNX + CPU 测速** — 导出 opset 12 ONNX，测 CPU 推理延迟
8. **保存指标** — 输出 metrics.json、labels.txt
9. **绘制训练曲线** — Loss & Accuracy 曲线可视化
10. **训练摘要** — 打印关键指标汇总
11-12. **推理 Demo** — 自动选图做预测演示

### 方式二：独立脚本

#### tiny_custom

```bash
python LQ_TinyClassifier/train_tiny_classifier.py \
  --data-root datasets \
  --img-size 96 \
  --batch-size 64 \
  --epochs 50 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --width-mult 0.35 \
  --patience 10 \
  --target-acc 0.98 \
  --out-dir LQ_TinyClassifier/artifacts
```

#### MobileNetV2

```bash
python LQ_TinyClassifier_MobileNetV2/train_mobilenet_v2.py \
  --data-root datasets \
  --img-size 96 \
  --batch-size 64 \
  --epochs 50 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --width-mult 1.0 \
  --patience 10 \
  --target-acc 0.98 \
  --out-dir LQ_TinyClassifier_MobileNetV2/artifacts_mobilenetv2
```

---

## 训练产物

训练产物根据 `backbone` 自动输出到各自文件夹。

### tiny_custom → `LQ_TinyClassifier/artifacts/`

| 文件 | 说明 |
|------|------|
| `best_model.pt` | PyTorch 最佳模型 checkpoint（含权重、类别映射、归一化参数） |
| `tiny_classifier_fp32.onnx` | ONNX 格式模型（opset 12，已 onnxsim 简化） |
| `tiny_classifier_fp32.ncnn.param` | NCNN 模型结构参数 |
| `tiny_classifier_fp32.ncnn.bin` | NCNN 模型权重 |
| `labels.txt` | 类别标签（每行一个，按字典序与模型输出索引对应） |
| `metrics.json` | 训练指标（准确率、损失、参数数量、推理延迟等） |
| `random_sample_metrics.json` | 随机抽样精度验证结果 |

### mobilenet_v2 → `LQ_TinyClassifier_MobileNetV2/artifacts_mobilenetv2/`

| 文件 | 说明 |
|------|------|
| `best_model.pt` | PyTorch 最佳模型 checkpoint |
| `mobilenet_v2_fp32.onnx` | ONNX 格式模型 |
| `labels.txt` | 类别标签 |
| `metrics.json` | 训练指标 |

### metrics.json 内容示例

```json
{
  "best_val_acc": 0.985,
  "best_epoch": 12,
  "target_acc": 0.98,
  "reached_target_acc": true,
  "test_acc": 0.983,
  "test_loss": 0.0512,
  "torch_cpu_1thread_ms": 3.456,
  "num_params": 4078,
  "backbone": "tiny_custom",
  "classes": ["1", "2", "3"],
  "img_size": 96,
  "width_mult": 0.35
}
```

---

## ONNX 转 NCNN

训练导出的 ONNX 模型可转换为 NCNN 格式以在嵌入式设备上推理。

### Windows（使用 PNNX）

已在 `LQ_TinyClassifier/pnnx-20260112-windows/` 中包含 pnnx 工具：

```powershell
# 转换 tiny_custom
.\pnnx.exe .\LQ_TinyClassifier\artifacts\tiny_classifier_fp32.onnx

# 转换 mobilenet_v2
.\pnnx.exe .\LQ_TinyClassifier_MobileNetV2\artifacts_mobilenetv2\mobilenet_v2_fp32.onnx
```

也可使用一键脚本：

```powershell
.\LQ_TinyClassifier\run_all.ps1
```

### Linux（使用 onnx2ncnn）

```bash
# 需先编译 ncnn 工具链
onnx2ncnn tiny_classifier_fp32.onnx tiny_classifier_fp32.ncnn.param tiny_classifier_fp32.ncnn.bin
```

> MobileNetV2 的所有算子均被 NCNN 原生支持，转换无额外兼容性问题。

---

## 嵌入式部署（LS2K0300）

`Seekfree_LS2K0300_Opensource_Library/` 目录包含了 LS2K0300 开发板的完整 SDK：

- **NCNN 推理引擎**：`libraries/zf_components/ncnn/` — 已编译好的 ncnn 库及头文件
- **TFLite Micro**：`libraries/zf_components/tflm/` — TensorFlow Lite Micro 推理引擎
- **板级驱动**：`libraries/zf_driver/`、`libraries/zf_device/` — 摄像头、屏幕、IMU 等驱动
- **项目模板**：`project/user/main.cpp` — 用户主程序入口
- **板端推理示例**：`project/code/lq_ncnn.cpp` — 使用 ncnn 加载模型进行推理的参考实现

**部署步骤**：

1. 在 PC 上训练模型 → 导出 ONNX → 转换为 NCNN（.param + .bin）
2. 将 NCNN 模型文件复制到 `project/model/` 目录
3. 在 `project/user/main.cpp` 中编写推理逻辑
4. 使用交叉编译工具链编译：`cd project/user && ./build.sh`
5. 将编译产物部署到 LS2K0300 开发板