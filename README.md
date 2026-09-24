# AssistLabel

基于 **Depth Anything V3 + SAM3** 的批量自动标注（预标注）流水线工具。

- **深度真值**：Depth Anything V3（默认）/ V2 批量刷 16-bit 深度图（毫米，KITTI 约定）
- **检测 + 分割**：SAM3 文本概念提示批量生成框、实例掩码、语义标签
- **深度-检测融合**：每个目标写入中位/最小/最大深度（伪激光雷达标注的输入）
- **断点续跑**：manifest.jsonl 驱动，中断后重跑自动跳过已完成图像
- **生态互通**：工作格式为 labelme（X-AnyLabeling 直接打开），可导出 COCO / YOLO

设计文档见 [PLAN.md](PLAN.md)，迭代计划见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。
当前状态：Windows + RTX 5090 真机全流程验证通过，57 项单元/端到端测试。

---

## 1. 环境搭建

> 本仓库已在 Windows + RTX 5090 上完整安装验证（conda env `assistlabel`，
> Python 3.12.14 + torch 2.11.0+cu128，真实模型推理跑通，见 `smoke/` 示例）。

### 1.1 环境搭建（GPU 推理，推荐直接用 requirements.txt）

```bash
cd /d/Code/AssistLabel

conda create -n assistlabel python=3.12 -y
conda activate assistlabel
# （若 conda activate 报错：老版本 conda 在部分终端需先
#   source <你的miniconda路径>/etc/profile.d/conda.sh 再 activate）

# 1) torch 必须走 cu128 专用源（Blackwell 显卡硬要求，PyPI 上的是 CPU 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) 其余依赖 + 本工具（SAM3 走 transformers 集成，纯 pip；深度默认 DA3）
pip install -r requirements.txt -e .
# 3) DA3 官方包（不在 PyPI，源码安装；--no-deps 避免拉入全量重依赖）
pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3"
#    （DA3 实际运行依赖已收录在 requirements.txt：einops/omegaconf/moviepy<2/trimesh/open3d 等）

# 4) 环境体检 + 预下载权重（ModelScope 直连）
python -m assistlabel.cli models check
python -m assistlabel.cli models download sam3             # SAM3 HF 格式 ~3.4GB（默认检测引擎）
python -m assistlabel.cli models download da3-metric-L        # DA3 米制深度 ~1.3GB（默认深度引擎）
```

> **SAM3 加载路径**：transformers ≥5 已原生集成 SAM3（`Sam3Model`/`Sam3Processor`），
> 权重为 HF 格式 `model.safetensors`（`detector_model.*` 键），ModelScope 镜像
> `facebook/sam3` 可直连下载，无需申请 HF 权限。（Meta 原生 `sam3` 包及 `sam3.pt`
> 的 `detector.*` 键格式与本工具不再相关；其视频跟踪 API 未被 transformers
> 高层封装覆盖，故本工具暂不含视频模式。）

### 1.2 仅跑流水线逻辑 / 开发 / 测试（无需 GPU）

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -q   # 56 passed
```

#### 权重下载源（ModelScope 优先）

`run.yaml` 中 `download.source` 控制所有模型的权重获取策略：

| 值 | 行为 |
|---|---|
| `auto`（默认） | **ModelScope 优先**（国内直连），失败自动回退 HuggingFace，再回退引擎自带下载 |
| `modelscope` | 只用 ModelScope，失败即报错 |
| `huggingface` | 只用 HuggingFace |

已验证的 ModelScope 仓库：`facebook/sam3`（HF 格式 safetensors + tokenizer）、
`depth-anything/Depth-Anything-V2-*-hf` 全系列、`depth-anything/DA3METRIC-LARGE`、
`depth-anything/DA3MONO-LARGE`。预下载/换源命令：
`assistlabel models download <key> [--source auto]`，
缓存位置由 ModelScope 默认管理（`%USERPROFILE%\.cache\modelscope`，可用 `MODELSCOPE_CACHE` 改）。

环境自检：

```bash
assistlabel models check   # torch/CUDA、transformers、modelscope、HF_TOKEN 逐项体检
```

---

## 2. 快速开始

### 2.1 无 GPU 冒烟验证（mock 引擎）

```bash
assistlabel init --dir demo --engine mock
# 放几张图进 demo/data/raw，或用任意 jpg 试跑：
assistlabel run -c demo/run.yaml
assistlabel validate -c demo/run.yaml
```

### 2.2 真实标注（三步）

```bash
# 1) 生成配置（默认注册真实模型）
assistlabel init --dir mydata

# 2) 编辑 mydata/ontology.yaml：数据集类别 -> SAM3 提示词
#    classes:
#      car:        {prompt: "car"}
#      pedestrian: {prompt: "pedestrian walking on the road"}

# 3) 跑流水线（depth、detect 串行共享 GPU；可随时 Ctrl+C，--resume 续跑）
assistlabel run -c mydata/run.yaml
```

室内/室外深度模型切换：`run.yaml` 中 `depth.model` 改为
`da3-metric-L`（默认深度引擎，米制）或 `da2-metric-indoor-L` / `da2-metric-outdoor-L`。
全部注册模型见 `assistlabel models list`。

---

## 3. 大规模数据集实战手册

### 3.1 模型权重下载到哪里

| 来源 | 默认缓存位置 | 环境变量改位置 |
|---|---|---|
| ModelScope（默认优先） | `%USERPROFILE%\.cache\modelscope` | `MODELSCOPE_CACHE` |
| HuggingFace（兜底） | `%USERPROFILE%\.cache\huggingface\hub` | `HF_HOME` |

缓存全局共享：多个数据集项目复用同一份权重，只有首次下载有成本。
显存允许时，batch 推理的权重开销是固定的，调大 `batch_size` 几乎白赚吞吐。

### 3.2 推荐作业流程（万级~十万级图像）

```bash
# 1) 试跑：先刷 200 张验证配置/类别/深度尺度是否正确（廉价试错）
assistlabel run -c run.yaml --limit 200
assistlabel validate -c run.yaml          # 检查类别分布、置信度、深度值域是否合理

# 2) 全量：Ctrl+C 随时中断，重跑自动跳过已完成图像
assistlabel run -c run.yaml

# 3) 完整性校验：断电/杀进程后，找出"manifest 标了 done 但文件损坏"的图
assistlabel verify -c run.yaml --repair   # 重置坏条目，然后重跑补齐
assistlabel run -c run.yaml               # 只重做坏的那几张

# 4) 人工复核 + 导出
assistlabel validate -c run.yaml --sample 100   # 低置信度分层复核清单
assistlabel export -c run.yaml --format coco --split 0.9
```

可靠性机制一览：

- **断点续跑**：`manifest.jsonl` 每批次原子落盘（tmp+rename），中断只损失最近几秒进度；
- **坏图隔离**：单张解码失败/推理 OOM 只标记该图 failed，不阻塞批次；resume 会自动重试 failed 图；
- **撕裂写检测**：`verify` 重读每个 "done" 产物的 PNG 头/uint16 类型/有效像素/JSON 完整性；
- **manifest 防爆**：全量重写按时间节流（5s），百万图规模也不会因存进度拖慢作业。

### 3.3 性能调优（run.yaml `depth:` 段）

深度阶段是三级流水线：**预取解码线程 → GPU 批量推理 → 写盘线程池**，
三段与下一批次重叠执行。调优顺序：

| 参数 | 默认 | 建议 |
|---|---|---|
| `batch_size` | 4 | 显存允尽量调大（8~16）；OOM 则减半。对吞吐影响最大 |
| `half` | true | fp16 推理，保持开启 |
| `prefetch` | 4 | 磁盘慢（机械盘/网络盘）时调大，解码彻底不挡 GPU |
| `write_workers` | 2 | NVMe 可 2-4；写入大 16-bit PNG 的收益明显 |
| `save_color: false` | true | 不需要人眼抽检时关掉，省 1/3 写盘量 |
| `save_npz` | false | 保持关闭，除非下游需要全精度浮点 |

其他已在引擎内启用的优化：cuDNN benchmark（固定输入尺寸自动选最快卷积核）、
TF32（Ampere+ 显卡免费加速）、横竖图分桶批处理（避免 padding 浪费）、
批量推理失败自动降级为逐图（防 OOM 连坐）。

SAM3 检测阶段为逐图多提示模型（图像编码一次复用全部类别提示），
耗时主要由模型决定；`detect.conf_thres` 调高可减少无效目标的写盘量。

## 4. 输出说明

```
out_dir/
├── manifest.jsonl      # 每图一行：路径、任务状态、耗时（断点续跑 + 溯源依据）
├── detect/
│   └── annotations.json  # COCO：RLE 实例掩码 + 框 + score + 每目标深度统计
├── semantic/
│   ├── classes.txt     # 像素值 -> 类别名（0=background）
│   └── xxx.png         # 单通道 uint8 语义掩码（像素值=类别索引）
├── depth/
│   └── xxx.png         # 16-bit PNG，uint16 毫米，depth_m = pixel/1000，无效=0
├── detect_viz/         # 检测叠加图（掩码半透明 + 框 + 类别/分数/深度）
├── semantic_viz/       # 语义掩码彩色预览
├── depth_viz/          # 深度 turbo 伪彩
├── report.html         # validate 产物：类别分布、置信度直方图、深度有效率
└── review_list.txt     # 按置信度分层的待人工复核清单

（按需导出：`assistlabel export --format yolo` 生成 ultralytics 训练布局）
```

标注即 COCO：`detect/annotations.json` 的每个 annotation 携带 RLE 实例掩码、
bbox、score，以及融合的每目标深度统计（`depth_median_m` 等），是伪激光雷达/
3D 框数据集的输入。

---

## 5. CLI 参考

| 命令 | 说明 |
|---|---|
| `assistlabel init [--dir DIR] [--engine auto\|mock]` | 生成 run.yaml + ontology.yaml |
| `assistlabel run -c run.yaml [--tasks depth,detect] [--no-resume]` | 批量流水线（可中断续跑） |
| `assistlabel fuse -c run.yaml` | 对已有 depth + labels 单独跑融合 |
| `assistlabel export -c run.yaml --format coco\|yolo\|semantic [--split 0.8] [--viz]` | 导出 COCO / YOLO(ultralytics) / 语义分割 |
| `assistlabel validate -c run.yaml [--sample 50]` | QA 报告 + 分层复核清单 |
| `assistlabel verify -c run.yaml [--repair]` | 产物完整性校验（断电/撕裂写检测与修复） |
| `assistlabel status -c run.yaml` | 各任务进度：done/failed/pending/ETA |
| `assistlabel run ... --redo-detect` | ontology 变更后清除 detect 状态重刷 |
| `assistlabel models list\|info KEY\|download KEY\|check` | 注册表 / 预下载 / 环境预检 |

#### 三种导出格式怎么选

| 格式 | 内容 | 适用 |
|---|---|---|
| `coco` | 实例级：每目标框 + polygon/RLE 掩码，**实例重叠完整保留** | 实例分割/检测训练（推荐默认） |
| `yolo` | ultralytics 训练布局：`images/ labels/ train/val + data.yaml` | YOLO 系检测训练 |
| `semantic` | 每图一张单通道 PNG，像素值=类别索引（0=背景，1..N=ontology 顺序）+ `classes.txt`；`--viz` 附彩色预览 | 语义分割训练（Cityscapes/ADE20K 风格） |

两点注意：

- **语义导出的重叠处理**：实例掩码重叠时按绘制顺序覆盖（后画盖先画）——这是从实例
  合成语图的通用做法。若下游需要严格保留重叠区域，用 `coco`（实例级无损）。
- **掩码边界精度**：`mask_to: polygon`（默认）落盘简化多边形（与原始掩码 IoU
  平均 ~0.99、最差 ~0.90）；`mask_to: rle` 落盘像素级精确掩码（无需
  pycocotools，纯标准库编解码），`semantic`/`coco --segmentation rle` 即按
  像素级真值导出。代价：rle 形状在 X-AnyLabeling 里不可视化编辑。

关键配置项（run.yaml）：

```yaml
tasks: [depth, detect, fuse]   # 任意子集
depth:
  model: da3-metric-L          # 注册表 key（DA3 米制；备选 da2-metric-* / da3-mono-L）
  half: true                   # fp16 推理
detect:
  conf_thres: 0.5              # 置信度阈值
  nms_iou: 0.7                 # 类内 NMS
  mask_to: polygon             # polygon(可编辑/近似) | rle(像素级精确)
fuse:
  stats: [median, min, max]    # 写入 shape.extra 的深度统计
```

自定义模型：把新的 YAML 放进任意目录，设 `ASSISTLABEL_MODELS_DIR` 指向它即可
覆盖/追加注册表（格式参照 `assistlabel/config/models/`）。

---

## 6. 架构

```
assistlabel/
├── cli.py            # typer CLI（本文件第 5 节的全部命令）
├── runner.py         # 编排器：分相流水线（预取解码/批推理/异步写盘）
├── hub.py            # 模型权重下载源（ModelScope 优先，HF 兜底）
├── qa.py             # 统计报告 + 分层抽样
├── verify.py         # 产物完整性校验（撕裂写/源图变更检测）
├── core/
│   ├── geometry.py   # IoU / NMS / mask<->polygon（纯 numpy，无 torch）
│   ├── registry.py   # 模型注册表（X-AnyLabeling model zoo 式 YAML）
│   ├── engine.py     # BaseEngine：懒加载 / unload 显存共享；Detection 数据类
│   ├── ontology.py   # 类别名 <-> SAM3 提示词（autodistill 式，重复键检测）
│   ├── config.py     # pydantic 运行配置
│   └── pipeline.py   # manifest.jsonl 断点续跑 + 单图任务执行
├── engines/
│   ├── depth_anything.py   # DA2（transformers AutoModelForDepthEstimation）
│   ├── depth_anything3.py  # DA3（官方包，批推理 + 原尺寸回采样）
│   ├── sam3.py             # SAM3（transformers 集成，(图,提示) 对批推理）
│   └── mock.py             # 确定性 mock（无 GPU 全链路测试用）
├── io/
│   ├── dataset.py    # 图像发现 / 内容哈希 / 输出路径约定
│   ├── depth_io.py   # 16-bit PNG(mm) 读写 / 伪彩
│   └── annotations.py# labelme 主格式 + COCO/YOLO 导出
├── fusion/box_depth.py    # 掩码内中位深度统计（剔除无效像素）
└── viz/overlay.py         # 检测叠加 / 深度对比图
```

新增引擎只需：继承 `BaseEngine`（load/infer/unload）+ 注册表加一条 YAML。

## 7. 测试

```bash
.venv\Scripts\python -m pytest tests/ -q
```

覆盖：几何（IoU/NMS/掩码转换 roundtrip）、标注格式（labelme/COCO/YOLO roundtrip）、
ontology 解析与错误、深度量化 roundtrip、manifest 断点续跑、注册表、配置校验、
以及 **mock 引擎端到端**（图像→深度+检测+融合→导出→QA 全链路）。

## 8. 许可说明

- 本工具代码：MIT
- SAM3 权重：Meta 自定义 SAM License（商用前请核对 LICENSE）
- Depth Anything V2 权重：Apache-2.0
- Depth Anything 3 代码：Apache-2.0；权重分许可——本工具收录的
  DA3METRIC-LARGE / DA3MONO-LARGE 为 Apache-2.0（GIANT/LARGE 系列为
  CC BY-NC 非商业，未收录）
