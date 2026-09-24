# AssistLabel 自动标注工具 — 项目规划

> 版本: v1.0 (2026-09-23)
> 定位: 基于 SAM3 + Depth Anything 的批量自动标注（预标注）流水线工具
> 本文档是后续实现的唯一蓝图，实现时严格按此执行，有异议先回溯更新本文档。

---

## 1. 项目目标

构建一个**命令行优先**的批量自动标注工具，核心能力：

1. **深度真值生成**：用 Depth Anything V2 批量为图像刷深度图（相对深度 + 度量深度）；
2. **目标框 + 语义标签生成**：用 SAM3 以文本概念提示（concept prompt）批量生成检测框、实例掩码、语义类别；
3. **（进阶）深度-检测融合**：掩码/框内深度统计 → 每目标中位深度、3D 位置等增强标注。

**明确不做的事**（差异化定位，避免重复造轮子）：
- 不做完整的人工标注编辑器 UI —— 人工修正环节导出 labelme/COCO 格式，交给 X-AnyLabeling / CVAT 完成；
- 不做模型训练/蒸馏 —— 只做"刷标签"，训练交给下游（可参考 autodistill 思路）；
- 不做云端服务 —— 单机 GPU 批处理工具。

---

## 2. 开源工具调研结论与借鉴

| 工具 | 形态 | 与本项目关系 | 借鉴点 |
|---|---|---|---|
| [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | Qt 桌面标注应用，已支持 SAM3 / Depth Anything | 竞品 + 互操作目标 | **配置驱动的模型注册表（model zoo）**：每个模型一个 JSON 配置声明来源、类型、推理参数；labelme 格式兼容 |
| [Autodistill](https://github.com/autodistill/autodistill) | base model → ontology → label → target model 的蒸馏流水线 | 理念参考 | **Ontology（本体映射）**：`类别名 → 自然语言概念提示` 的映射表，解耦"数据集类别体系"与"模型提示词" |
| [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Grounding DINO + SAM 组合 demo | 上一代方案 | 文本提示 → 框 → mask 的两段式流程；SAM3 原生支持文本提示，**此组合已被 SAM3 取代** |
| CVAT / Label Studio | Web 标注平台 + ML backend | 互操作 | COCO 导入导出格式规范 |

**结论**：
- 本工具与 X-AnyLabeling 的差异：X-AnyLabeling 是"人在回路"的桌面标注器（单图交互式），本项目是**无人值守的批量流水线**（万级图片、断点续跑、纯 CLI/可脚本化）。两者互补：本工具刷完标签 → 人工在 X-AnyLabeling 里抽检修正。
- 必须支持 **labelme + COCO 双格式导出**，保证与现有标注生态互通。

---

## 3. 模型选型

### 3.1 SAM3（检测 + 分割 + 语义）

> **2026-09 更新**：transformers ≥5 已原生集成 SAM3（`Sam3Model`/`Sam3Processor`/
> `Sam3VideoModel`），本工具因此支持两条加载路径，**权重键名互不兼容**：
> - `sam3`（默认）：transformers 加载 `model.safetensors`（`detector_model.*` 键），
>   ModelScope 镜像 `facebook/sam3` 可直连下载（无需 HF gated 权限）。纯 pip 安装。
> - `sam3`：Meta 原生包（GitHub clone 源码安装），加载 `sam3.pt`（`detector.*` 键）。
>   图片结果一致（实测 truck 0.93 vs 0.86 同框），**视频跟踪必须用原生包**。
>   注意原生包 pyproject 漏声明 einops/triton/pycocotools/psutil。

- 仓库: https://github.com/facebookresearch/sam3 （2025-11 开源，检测器 + 跟踪器共享视觉编码器）
- 权重: HuggingFace `facebook/sam3` / `facebook/sam3.1`；ModelScope 镜像同名同组织可直连（已验证含 HF 格式 safetensors + 原生 sam3.pt）
- 能力: 文本概念提示（如 `"a yellow school bus"`）→ 实例掩码 + 框 + 置信度；也支持点/框/示例图提示；视频模式支持检测 + 跟踪
- transformers 关键 API:
  ```python
  from transformers import Sam3Model, Sam3Processor, Sam3ImageProcessor, AutoTokenizer
  model = Sam3Model.from_pretrained(local_dir).to("cuda").eval()
  processor = Sam3Processor(Sam3ImageProcessor(), AutoTokenizer.from_pretrained(local_dir))
  inputs = processor(images=[img]*len(prompts), text=prompts, return_tensors="pt").to("cuda")
  outputs = model(**inputs)
  results = processor.post_process_instance_segmentation(outputs, threshold=0.4, target_sizes=[(h,w)]*len(prompts))
  # -> [{scores, boxes(xyxy), masks(N,H,W)}, ...] 与 prompts 一一对应
  ```
  （ModelScope 镜像缺 preprocessor_config.json，processor 用类默认参数组装即可，默认值即官方值）
- 环境: transformers ≥5，PyTorch ≥2.7（Blackwell 显卡必须 cu128）
- 许可: 自定义 SAM License（研究/商用基本可用，超大体量产品有限制，商用前自行核对 LICENSE）

### 3.2 Depth Anything V2（深度真值）

- 仓库: https://github.com/DepthAnything/Depth-Anything-V2 （NeurIPS 2024，生态最稳）
- 变体选择（transformers 生态直接可用，`depth-anything/Depth-Anything-V2-*-hf` 系列）:
  - **相对深度**（单目相对，无绝对尺度）: Small(25M) / Base(97M) / Large(335M)，通用场景
  - **度量深度**（绝对米制）: `Metric-Indoor-Hypersim`（室内）/ `Metric-Outdoor-VKITTI`（室外）两个精调版
  - 默认策略: Large 版；室内/室外由配置或图像 EXIF/自动分类决定，默认提供两个 metric 模型 + 1 个 relative 模型三选一
- 调用方式（transformers pipeline，无需自编译）:
  ```python
  from transformers import pipeline
  pipe = pipeline(task="depth-estimation",
                  model="depth-anything/Depth-Anything-V2-Metric-Indoor-Hypersim-Large-hf",
                  device=0)
  out = pipe(image)   # out["predicted_depth"]: tensor(H,W), 米制
  ```
- 备注: Depth Anything 3 已发布（多视角 3D 重建方向，OpenReview），Prompt Depth Anything 支持 LiDAR 提示的 4K 度量深度。本期不引入，作为架构预留（模型注册表天然支持后续扩展）。

---

## 4. 总体架构

```
assistlabel/
├── cli.py                  # typer CLI 入口: depth / detect / fuse / validate
├── core/
│   ├── registry.py         # 模型注册表（借鉴 X-AnyLabeling model zoo）
│   ├── engine.py           # 推理引擎抽象：懒加载、批处理、显存管理、device
│   ├── pipeline.py         # 任务编排：任务 DAG + 断点续跑（manifest 驱动）
│   └── ontology.py         # 类别名 ↔ 概念提示映射（借鉴 autodistill）
├── engines/
│   ├── depth_anything.py   # DepthAnythingEngine (transformers pipeline)
│   └── sam3.py             # SAM3Engine (图片检测分割) / SAM3VideoEngine(跟踪)
├── io/
│   ├── dataset.py          # 图像发现、遍历、哈希索引
│   ├── depth_io.py         # 16-bit PNG 读写 + scale 元数据 + 伪彩可视化
│   └── annotations.py      # labelme / COCO 导入导出，mask↔polygon/RLE 转换
├── fusion/
│   └── box_depth.py        # 框/掩码 × 深度图 → 每目标深度统计（中位/最小/最大）
├── viz/
│   └── overlay.py          # 叠加渲染：mask 半透明、框+类别+分数、深度伪彩
└── config/
    ├── default.yaml
    └── models/             # 每个模型一份注册配置 (JSON/YAML)
```

### 核心抽象（实现时的接口契约）

```python
class BaseEngine(ABC):
    """所有推理引擎的基类。懒加载：首次 infer 时才占显存。"""
    name: str
    def load(self, device: str, half: bool) -> None: ...
    def infer(self, image: np.ndarray, **params) -> dict: ...
    def unload(self) -> None: ...   # 释放显存，支持 depth→detect 串行共卡

class Pipeline:
    """manifest 驱动的批处理。每张图处理完即写 manifest，崩溃后重跑跳过已完成。"""
    def run(self, images: list[Path], tasks: list[str], resume: bool) -> RunReport: ...
```

### 配置驱动（一切可调参数进配置，不硬编码）

`config/run.yaml` 示例：
```yaml
dataset:
  image_dir: "data/raw"
  out_dir: "data/labeled"
  patterns: ["*.jpg", "*.png"]

tasks: [depth, detect]          # 可选: depth / detect / fuse

depth:
  model: "da2-metric-indoor-L"  # 注册表中的 key
  save: {raw: true, color: true, npz: false}

detect:
  model: "sam3"
  ontology: "config/ontology.yaml"   # 见下
  conf_thres: 0.5
  max_objects_per_prompt: 50
  mask_to: "polygon"            # polygon / rle / both（labelme 要 polygon，COCO 两者皆可）

fuse:
  depth_stat: [median, min, max]     # 写入标注的 extra 字段
```

`config/ontology.yaml`（autodistill 式本体映射）:
```yaml
classes:
  car:            {prompt: "car", }
  pedestrian:     {prompt: "pedestrian walking on the road"}
  traffic_cone:   {prompt: "traffic cone"}
  # 数据集类别名 → SAM3 概念提示词，允许一词多映射/别名
```

---

## 5. 输出格式规范（关键设计决策）

### 5.1 深度输出

- **主格式**: 与原图同尺寸 **16-bit PNG**，`uint16`，单位毫米（KITTI 风格约定：`depth_m = pixel / 1000`；无效值 = 0）
- **元数据 sidecar**: `depth_meta.json`，记录 `{model_id, depth_unit: "mm", scale: 1000, valid_pixels_ratio, source_image_hash}`，与图像一一对应
- **可视化**: 并行输出 8-bit 伪彩 JPG（turbo/inferno colormap），仅供人眼抽检，不是真值
- 可选 `.npz`（float16 原始输出，保精度，默认关闭）

### 5.2 检测/分割输出

- **主格式 labelme**（`.json` 每图一份，与 X-AnyLabeling 无缝互通）:
  `shapes[].label = 类别名（ontology key）`，`shape_type = "polygon"`（掩码转多边形，用 cv2.findContours 简化，`epsilon = 0.002 * 周长`），`score` 写入 `shape["score"]`（labelme 非标准字段但 X-AnyLabeling 认）
- **导出格式 COCO**（`assistlabel export --format coco`）: boxes + segmentation(RLE 或 polygon)，`category_id` 按 ontology 顺序
- **YOLO 框格式导出**（可选，`--format yolobox`）

### 5.3 目录结构约定

```
data/labeled/
├── manifest.jsonl            # 每图一行: 路径、哈希、各任务状态、耗时、目标数
├── images/                   # 软链或复制原图
├── depth/
│   ├── xxx.png               # 16-bit 真值
│   ├── xxx_meta.json
│   └── xxx_color.jpg         # 伪彩
├── labels/
│   └── xxx.json              # labelme
├── viz/
│   └── xxx.jpg               # 检测叠加图（抽检用）
└── coco/
    └── annotations.json      # export 产物
```

---

## 6. 功能模块设计

### 6.1 深度流水线（M1，先做——最简单，验证架构）

1. 扫描图像 → manifest 登记
2. Depth Anything 推理（batch=1 起步，transformers 内部优化；resize 策略: 保持长边 ≤518 分辨率由模型内部处理，输出 resize 回原图尺寸 + 最近邻/双线性插值按配置）
3. 米制 → 毫米 uint16 量化落盘 + 伪彩图 + sidecar
4. 汇报：平均耗时、显存峰值、失败列表

### 6.2 SAM3 检测流水线（M2，核心）

1. 逐图 `set_image` → 遍历 ontology 的每个类别提示 `set_text_prompt`
   - 优化: 同一张图的状态可复用多个文本提示（避免重复编码图像）
2. 过滤: `score < conf_thres` 丢弃；每提示 `max_objects` 截断；框 NMS（IoU 0.7，跨类别不抑制）
3. 掩码 → polygon（cv2.findContours + Douglas-Peucker 简化，丢弃 <3 点的形状）
4. 写 labelme json + 叠加可视化图
5. **视频模式（M3 可选）**: `build_sam3_video_predictor` + 首帧文本提示 → 全视频跟踪，输出逐帧 labelme + track_id

### 6.3 深度-检测融合（M3）

- 对每个目标: 掩码内深度统计（中位深度为主，抗掩码边缘噪声；另存 min/max）写入 labelme 的 `shape["extra"]["depth_m"]`
- 有效性: 深度图与检测来自不同模型，分辨率/对齐一致性靠"都 resize 回原图坐标系"保证；掩码内 0 值（无效深度）像素剔除后统计
- 产物意义: 伪激光雷达标注、3D 框数据集的输入

### 6.4 批处理调度（贯穿）

- manifest.jsonl 驱动断点续跑：每图完成即追加落盘（写临时文件 + rename 原子操作）
- GPU OOM 处理: 捕获 → 单图降级重试一次 → 仍失败记入 manifest `failed`，不阻塞批次
- 引擎串行加载（depth 和 sam3 不同时占显存，除非显存 ≥24G 且配置允许）

### 6.5 质检（M4）

- `assistlabel validate`: 统计报告（类别分布、置信度直方图、每图目标数、深度有效像素比）
- `assistlabel sample --stratify conf`: 按置信度分层抽样生成待人工复核清单
- 可视化叠加图 + 伪彩图直接供人工快速翻阅
-（可选）生成 FiftyOne 兼容数据集描述，用 FiftyOne 做交互式 QA

---

## 7. CLI 设计（typer + rich）

```
assistlabel init                          # 交互式生成 run.yaml + ontology.yaml
assistlabel run [-c run.yaml] [--resume] [--tasks depth,detect]
assistlabel video -v input.mp4 --ontology ... # 视频跟踪模式
assistlabel fuse [-c run.yaml]            # 离线融合（已有 depth + labels 时）
assistlabel export --format coco|yolobox [--split 0.8]
assistlabel validate [--report report.html]
assistlabel models list|info <id>         # 注册表查询
```

---

## 8. 技术栈与依赖

| 组件 | 选型 | 理由 |
|---|---|---|
| Python | 3.12（locked） | SAM3 硬性要求 |
| PyTorch | 2.10 + cu128（locked） | SAM3 官方示例版本，避免兼容坑 |
| 深度模型 | `transformers` ≥4.5x | Depth Anything V2 即装即用 |
| SAM3 | `pip install -e .`（vendored 或 submodule） | 官方包未上 PyPI，需源码安装 |
| CLI | typer + rich | 进度条、表格报告 |
| 配置 | pydantic + YAML | 校验 + 默认值 + IDE 提示 |
| 图像 | opencv-python, pillow, pycocotools | mask↔poly、RLE、COCO |
| 打包 | uv + pyproject.toml | 快速环境复现 |
| 测试 | pytest（引擎用小模型/ mocked 推理） | CI 可跑逻辑测试 |

**环境风险（Windows）**: 本机为 win32。SAM3 核心依赖（torch cu128）有 Windows wheel；flash-attn-3 / cc_torch 跳过不影响功能。若 `pip install -e sam3` 在 Windows 编译失败 → 备选方案 **WSL2 + Ubuntu 22.04 + conda**（官方推荐路径），CLI 与数据目录完全无感迁移。

---

## 9. 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **M0 脚手架** (0.5d) | repo 结构、pyproject、配置系统、CLI 骨架、dataset 扫描/manifest | `assistlabel init` 生成可用配置；单测过 |
| **M1 深度流水线** (1d) | DepthAnythingEngine + depth_io + 断点续跑 | 100 张图刷完，16-bit PNG 可被 OpenCV 读回且数值正确，中断重跑只补缺 |
| **M2 SAM3 流水线** (2d) | SAM3Engine + ontology + labelme/COCO 导出 + NMS/过滤 + 可视化 | 抽 20 张，X-AnyLabeling 打开 labelme 无损；COCO 可被 pycocotools 加载 |
| **M3 融合 + 视频** (1.5d) | box_depth 融合；SAM3 视频跟踪模式 | 每目标 depth_m 落盘；10 分钟视频跟踪跑通 |
| **M4 质检与文档** (1d) | validate 报告、分层抽样、README、性能基准 | 一条命令端到端 demo；README 可让新人 30 分钟跑通 |

总计约 6 人日（flash 模型实现时按此顺序推进，每个里程碑独立可验证、可提交）。

---

## 10. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| SAM3 权重是 HF gated repo | 首次运行失败 | 文档写明申请步骤；`assistlabel models check` 预检 `HF_TOKEN` 与权重可及性 |
| SAM3 自定义 License | 商用合规 | README 标注许可摘要，商用前人工核对 |
| Windows 下 SAM3 安装/编译问题 | 环境搭建受阻 | 优先纯 pip 路径；失败即切 WSL2 方案（预案已列） |
| 848M 参数显存占用（估计 6-8GB 推理） | 消费级 GPU 紧张 | 引擎串行加载 + unload；提供 SAM3 smaller 变体（如有）注册位 |
| 掩码→polygon 顶点过多 | labelme 文件膨胀 | Douglas-Peucker 自适应简化；顶点上限（默认 500） |
| 度量深度室内/室外模型选错 | 深度真值尺度错误 | 配置显式声明；`validate` 中输出深度值域分布辅助判断（室内场景 >30m 即可疑） |

---

## 11. 参考资料

- SAM3 论文/仓库: https://github.com/facebookresearch/sam3 ｜ https://ai.meta.com/blog/segment-anything-3/
- SAM3 权重: https://huggingface.co/facebook/sam3 （需申请访问）
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 ｜ HF: `depth-anything/Depth-Anything-V2-Metric-*-hf`
- X-AnyLabeling（模型注册表参考 + 人工修正工具）: https://github.com/CVHub520/X-AnyLabeling
- Autodistill（ontology 模式参考）: https://github.com/autodistill/autodistill
- Depth Anything 3 / Prompt Depth Anything（后续扩展方向）
