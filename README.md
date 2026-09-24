# AssistLabel

基于 **Depth Anything V3 + SAM3** 的批量自动标注（预标注）流水线工具。

- **深度真值**：Depth Anything V3（默认）/ V2 批量刷 16-bit 深度图（毫米，KITTI 约定）
- **检测 + 实例分割**：SAM3 文本概念提示批量生成框、RLE 实例掩码、语义标签
- **语义分割**：同步产出按类别索引的单通道语义 PNG（0=背景）
- **断点续跑**：manifest.jsonl 驱动，中断后重跑自动跳过已完成图像
- **生态互通**：detect 直接产出标准 COCO（ultralytics convert_coco 可读），可视化独立成 `*_viz` 目录

设计文档见 [PLAN.md](PLAN.md)，迭代计划见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。
当前状态：Windows + RTX 5090 真机全流程验证通过，62 项单元/端到端测试。

---

## 1. 安装

> 本仓库已在 Windows + RTX 5090 上完整安装验证（conda env `assistlabel`，
> Python 3.12.14 + torch 2.11.0+cu128，真实模型推理跑通）。

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
assistlabel models check
assistlabel models download sam3           # SAM3 ~3.4GB（默认检测引擎）
assistlabel models download da3-metric-L   # DA3 米制深度 ~1.3GB（默认深度引擎）
```

> **SAM3 加载路径**：transformers ≥5 已原生集成 SAM3（`Sam3Model`/`Sam3Processor`），
> 权重为 HF 格式 `model.safetensors`（`detector_model.*` 键），ModelScope 镜像
> `facebook/sam3` 可直连下载，无需申请 HF 权限。（Meta 原生 `sam3` 包及 `sam3.pt`
> 的 `detector.*` 键格式与本工具不再相关；其视频跟踪 API 未被 transformers
> 高层封装覆盖，故本工具暂不含视频模式。）

### 1.1 权重下载源与缓存

`run.yaml` 中 `download.source` 控制所有模型的权重获取策略：

| 值 | 行为 |
|---|---|
| `auto`（默认） | **ModelScope 优先**（国内直连），失败自动回退 HuggingFace，再回退引擎自带下载 |
| `modelscope` | 只用 ModelScope，失败即报错 |
| `huggingface` | 只用 HuggingFace |

已验证的 ModelScope 仓库：`facebook/sam3`、`depth-anything/Depth-Anything-V2-*-hf`
全系列、`depth-anything/DA3METRIC-LARGE`、`depth-anything/DA3MONO-LARGE`。
预下载/换源命令：`assistlabel models download <key> [--source auto]`。

| 来源 | 默认缓存位置 | 环境变量改位置 |
|---|---|---|
| ModelScope（默认优先） | `%USERPROFILE%\.cache\modelscope` | `MODELSCOPE_CACHE` |
| HuggingFace（兜底） | `%USERPROFILE%\.cache\huggingface\hub` | `HF_HOME` |

缓存全局共享：多个数据集项目复用同一份权重，只有首次下载有成本。

---

## 2. 快速上手

### 2.1 完整标注工作流

```bash
# 1) 生成配置（默认 da3-metric-L 深度 + sam3 检测）
assistlabel init --dir mydata

# 2) 编辑 mydata/ontology.yaml：数据集类别 -> SAM3 提示词
#    classes:
#      car:        {prompt: "car"}
#      pedestrian: {prompt: "pedestrian walking on the road"}

# 3) 试跑 20 张：先确认检出/深度合理，再放量（廉价试错）
assistlabel run -c mydata/run.yaml --limit 20
#    打开 mydata/labeled/ 下 detect_viz\ 与 semantic_viz\ 人工核对

# 4) 全量：Ctrl+C 随时中断，重跑自动跳过已完成图像
assistlabel run -c mydata/run.yaml

# 5) 校验 + 质检
assistlabel verify -c mydata/run.yaml --repair    # 完整性校验（--repair 重置坏条目）
assistlabel validate -c mydata/run.yaml --sample 50   # QA 报告 + 低置信度复核清单

# 6) 导出训练格式
assistlabel export -c mydata/run.yaml --format yolo --split 0.8  # ultralytics 布局
```

### 2.2 无 GPU 冒烟验证（mock 引擎）

```bash
assistlabel init --dir demo --engine mock
# 放几张图进 demo/data/raw，或用任意 jpg 试跑：
assistlabel run -c demo/run.yaml
assistlabel validate -c demo/run.yaml
```

深度模型切换：`run.yaml` 中 `depth.model` 改为 `da3-metric-L`（默认，米制室内外通用）、
`da3-mono-L`（相对深度）或 `da2-metric-indoor-L` / `da2-metric-outdoor-L`。
全部注册模型见 `assistlabel models list`（带中文说明）。

---

## 3. 大规模作业与可靠性

### 3.1 可靠性机制

- **断点续跑**：`manifest.jsonl` 每批次原子落盘（tmp+rename），中断只损失最近几秒进度；
- **坏图隔离**：单张解码失败/推理 OOM 只标记该图 failed，不阻塞批次；resume 会自动重试 failed 图；
- **撕裂写检测**：`verify` 重读每个 "done" 产物的 PNG 头/uint16 类型/有效像素/JSON 完整性；
- **manifest 防爆**：全量重写按时间节流（5s），百万图规模也不会因存进度拖慢作业。

### 3.2 性能调优（run.yaml `depth:` 段）

深度阶段是三级流水线：**预取解码线程 → GPU 批量推理 → 写盘线程池**，
三段与下一批次重叠执行。调优顺序：

| 参数 | 默认 | 建议 |
|---|---|---|
| `batch_size` | 4 | 显存允尽量调大（8~16）；OOM 则减半。对吞吐影响最大 |
| `half` | true | fp16 推理，保持开启 |
| `prefetch` | 4 | 磁盘慢（机械盘/网络盘）时调大，解码彻底不挡 GPU |
| `write_workers` | 2 | NVMe 可 2-4；写入大 16-bit PNG 的收益明显 |
| `viz: false` | true | 不需要深度伪彩抽检时关掉 |

其他已在引擎内启用的优化：cuDNN benchmark（固定输入尺寸自动选最快卷积核）、
TF32（Ampere+ 显卡免费加速）、横竖图分桶批处理（避免 padding 浪费）、
批量推理失败自动降级为逐图（防 OOM 连坐）。

SAM3 检测阶段为逐图多提示模型（图像编码一次复用全部类别提示），
耗时主要由模型决定；`detect.conf_thres` 调高可减少无效目标的写盘量。

---

## 4. 输出说明

```
out_dir/
├── manifest.jsonl      # 每图一行：路径、任务状态、耗时（断点续跑 + 溯源依据）
├── detect/
│   └── annotations.json  # 标准 COCO：RLE 实例掩码 + bbox + score
├── semantic/
│   ├── classes.txt     # 像素值 -> 类别名（0=background）
│   └── xxx.png         # 单通道 uint8 语义掩码（像素值=类别索引）
├── depth/
│   └── xxx.png         # 16-bit PNG，uint16 毫米，depth_m = pixel/1000，无效=0
├── detect_viz/         # 检测叠加图（实例掩码半透明 + 框 + 类别/分数）
├── semantic_viz/       # 语义掩码彩色预览
├── depth_viz/          # 深度 turbo 伪彩
├── report.html         # validate 产物：类别分布、置信度直方图、深度有效率
└── review_list.txt     # 按置信度分层的待人工复核清单

（按需导出：`assistlabel export --format yolo` 生成 ultralytics 训练布局）
```

`detect/annotations.json` 为标准 COCO 字段（RLE 实例掩码 + bbox + score），
ultralytics 官方 `convert_coco` 工具或 pycocotools 均可直接读取；
`semantic/` 与 `depth/` 为像素级对齐的语义/深度真值。

---

## 5. CLI 命令手册

> **所有命令与子命令都支持查询用法**：`assistlabel --help`、
> `assistlabel run --help`、`assistlabel models download --help`……
> 也可以用 `assistlabel help <命令>`（支持多级，如 `assistlabel help models download`）。

### 5.1 init — 生成项目配置

```
assistlabel init [--dir DIR] [--engine auto|mock]
```

| 选项 | 说明 |
|---|---|
| `--dir DIR` | 项目目录（默认当前目录），自动创建 `run.yaml`、`ontology.yaml`、`data/raw/` |
| `--engine auto\|mock` | auto=注册表里的真实模型；mock=测试假引擎（无需 GPU，秒级验证流程） |

### 5.2 run — 批量标注流水线

```
assistlabel run -c run.yaml [选项]
```

| 选项 | 说明 |
|---|---|
| `-c/--config PATH` | run.yaml 路径（默认 `run.yaml`） |
| `--tasks depth,detect` | 覆盖配置里的任务列表（逗号分隔） |
| `--resume/--no-resume` | 断点续跑开关（默认开；跳过 manifest 中已完成的图） |
| `--limit N` | 只处理前 N 张（小批量试跑） |
| `--redo-detect` | 清除全部 detect 状态后重跑（ontology 变更后使用） |
| `--force` | 跳过磁盘剩余空间预检 |

执行中可随时 Ctrl+C 中断：进度已保存，重跑自动续。

### 5.3 status — 查看进度（只读，不执行处理）

```
assistlabel status -c run.yaml [--tasks depth,detect]
```

按任务显示 done / failed / pending 与平均耗时、ETA；若 ontology.yaml
在标注后发生过变更，会提示用 `--redo-detect` 重刷。

### 5.4 export — 导出训练格式

```
assistlabel export -c run.yaml --format yolo [--split 0.75]
```

从 `detect/annotations.json` 派生 ultralytics 训练布局
（`images/ labels/ train/val + data.yaml`）。

| 选项 | 说明 |
|---|---|
| `--format yolo` | 生成 ultralytics 训练布局 |
| `--split 0.8` | 训练集占比（<1.0 时同时生成 val） |

### 5.5 validate — QA 质检

```
assistlabel validate -c run.yaml [--sample 50] [--report PATH]
```

生成 `report.html`（类别分布、置信度直方图、深度有效率）与
`review_list.txt`（按最低置信度排序的人工复核清单，`--sample` 控制条数）。

### 5.6 verify — 产物完整性校验

```
assistlabel verify -c run.yaml [--repair]
```

逐项检查 "done" 产物的完整性：深度 PNG 可读/uint16/有效率、语义 PNG、
COCO 可解析、源图内容是否变更（stale）。`--repair` 把损坏条目
重置为待处理，随后 `run --resume` 只重做坏图。

### 5.7 models — 模型注册表

```
assistlabel models list [--kind depth|detect_segment]   # 浏览模型（含中文说明）
assistlabel models info KEY                             # 单个模型完整规格
assistlabel models download KEY [--source auto]         # 预下载权重
assistlabel models check                                # 环境体检
```

### 5.8 help — 查询命令用法

```
assistlabel help                  # 顶层命令列表
assistlabel help run              # run 的用法与选项
assistlabel help models download  # 多级子命令
```

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
│   ├── depth_io.py   # 16-bit PNG(mm) 读写（Windows 中文路径安全）/ 伪彩
│   └── annotations.py# COCO store 增删存取 + RLE 编解码 + ultralytics 导出
├── fusion/box_depth.py    # 掩码内深度统计（剔除无效像素）
└── viz/overlay.py         # 检测叠加 / 深度对比图
```

新增引擎只需：继承 `BaseEngine`（load/infer/unload）+ 注册表加一条 YAML。

---

## 7. 开发与测试（无需 GPU）

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -q   # 62 passed
```

测试覆盖：几何（IoU/NMS/掩码转换 roundtrip）、标注格式（COCO store、RLE、
ultralytics 导出）、ontology 解析与错误、深度量化 roundtrip、manifest 断点续跑、
注册表、配置校验、CLI 帮助查询，以及 **mock 引擎端到端**
（图像→深度+检测→融合→导出→QA 全链路）。

---

## 8. 许可说明

- 本工具代码：MIT
- SAM3 权重：Meta 自定义 SAM License（商用前请核对 LICENSE）
- Depth Anything V2 权重：Apache-2.0
- Depth Anything 3 代码：Apache-2.0；权重分许可——本工具收录的
  DA3METRIC-LARGE / DA3MONO-LARGE 为 Apache-2.0（GIANT/LARGE 系列为
  CC BY-NC 非商业，未收录）
