# AssistLabel 改进计划 v2（供后续实现）

> 本文档是下一轮迭代的唯一任务清单，按优先级排序。每项含：问题、改哪里、
> 怎么改、验收标准。实现时逐项进行，每完成一项跑通全部测试再进行下一项。
>
> **实现环境约定**（与本仓库当前状态一致）：
> - 测试环境：`source /d/miniconda3/etc/profile.d/conda.sh && conda activate assistlabel`
> - 测试命令：`python -m pytest tests/ -q`（当前 57 passed，改动后只增不减）
> - 郵件源码在 `assistlabel/`，纯逻辑模块不依赖 GPU 即可测试；涉及真实模型的
>   验证用 mock 引擎写单测，不要在测试里加载真实权重
> - 配置默认值改动必须同步 `assistlabel/cli.py` 的 RUN_YAML_TEMPLATE 和 README

---

## P0 — 正确性 bug（必须先修，都是现有代码的真实缺陷）

### P0-1 输出路径冲突：嵌套目录同名文件互相覆盖

**问题**：`assistlabel/io/dataset.py` 的 `out_paths()` 只用 `image_path.stem`
拼输出文件名。数据集是递归扫描的（`data/raw/a/img.png` 和 `data/raw/b/img.png`
都合法），但两者会写到同一个 `out/labels/img.json`，互相覆盖，且 manifest
里两条记录都标 done——**静默丢标注，最危险的一类 bug**。函数 docstring 声称
"relative path flattened with `__`"，实际没实现。

**改法**（`io/dataset.py`）：`out_paths()` 增加可选参数 `image_dir`，有值时
用相对路径把 `/` 和 `\` 替换为 `__` 作为输出 stem（`a/img.png` → `a__img`），
无值时保持现行为（向后兼容顶层平铺数据集）。`runner.py` 所有调用点传入
`cfg.dataset.image_dir`。

**验收**：新增单测——`tmp_path/a/img.png` 与 `tmp_path/b/img.png` 生成不同的
depth/labelme 路径；mock 端到端跑嵌套目录数据集，两张图的产物都在且内容正确。

### P0-2 Windows 非 ASCII 路径（中文文件名）读写会失败

**问题**：`io/depth_io.py` 的 `save_depth_png` / `read_depth_png` /
`save_depth_color` 直接 `cv2.imwrite/imread(str(path))`；OpenCV 在 Windows
上对非 ASCII 路径返回 None / 写失败。`runner.py` 的 `_read_image_rgb` 和
`_write_image_bgr` 已经用了正确的 `np.fromfile` + `imencode.tofile` 方案，
但 depth_io 和 `verify.py`（同样用 cv2.imread）没跟上。数据集里有中文文件名
是常态，这不是边缘情况。

**改法**：在 `io/depth_io.py` 内部封装 `_imwrite_safe(path, arr, params)` /
`_imread_safe(path, flag)`（imencode/tofile 与 imdecode/fromfile），替换所有
直接 cv2 调用；`verify.py` 的深度图读取改用 `read_depth_png`。

**验收**：单测用包含中文的路径（`tmp_path/"图库"/"照片_01.png"`）做深度
save/read roundtrip；verify 对中文路径项目正常工作。

### P0-3 同一 out_dir 并发跑两个 run 会损坏 manifest

**问题**：两个 `assistlabel run -c 同一个run.yaml` 同时启动时，两个进程各自
全量重写 `manifest.jsonl`（tmp+rename 原子，但互相覆盖对方的进度），最后
manifest 与磁盘实际状态不一致。

**改法**（`core/pipeline.py`）：`Manifest` 初始化时在同目录创建锁文件
`manifest.lock`（Windows 用 `msvcrt.locking`，POSIX 用 `fcntl.flock`，都不引新依赖），
拿不到锁时报错退出并提示"另一个实例正在运行"；进程正常退出/崩溃时释放。
CLI `run` 命令捕获该错误给友好提示。

**验收**：单测模拟二次加锁失败（子进程或直接二次 open）；现有 manifest 测试全过。

### P0-4 README 与实现的偏差清理

README（当前版本）需要修正：
- 第 3.1 节表格仍列 `ASSISTLABEL_SAM3_CKPT`（该机制已随原生 sam3 删除）；
- CLI 表格缺 `verify` 命令一行；
- 出现两个 "## 4"（输出说明、CLI 参考），重排为 4/5/6/7/8；
- 第 5 节架构图缺 `hub.py`、`verify.py`、`engines/depth_anything3.py`，
  `sam3.py` 的注释还写着"视频跟踪"（已删）；
- 第 1 节开头说"41 项测试"（现 57）、提到已删除的 `smoke/` 示例；
- 第 7 节许可缺 DA3（代码 Apache-2.0，但 **GIANT/LARGE 权重是 CC BY-NC**，
  本工具只用 Apache 的 METRIC/MONO 变体，应写明）。

**验收**：文档通读无过期引用；`grep -rn "sam3-hf\|ASSISTLABEL_SAM3_CKPT\|video" README.md` 无残留。

---

## P1 — 性能改进（按预期收益排序）

### P1-1 detect 阶段并行化写盘 + 预取（对标 depth 阶段已有的三级流水线）

**现状**：`runner.py::_run_detect_phase` 主循环串行做：读图 → SAM3 推理 →
labelme 写盘 → viz 叠加渲染 + 编码写盘。其中 viz 渲染（掩码半透明合成）
和 JPEG 编码是纯 CPU，实测占比不低；深度阶段已有预取线程 + 写盘线程池，
detect 没有。detect 是 SAM3 主力任务，吞吐 ~1.4 img/s，优化空间最大。

**改法**：复用 depth 阶段的模式——producer 线程预取解码；推理在主线程；
labelme/viz 写盘丢进 `ThreadPoolExecutor(cfg.detect.write_workers)`（新配置项，
默认 2）。注意 manifest 标记 done 必须在写盘完成后（复用 `_depth_write_worker`
的模式：worker 里写盘成功才 mark done）。

**验收**：mock 端到端不变绿；新增单测断言 detect 的 viz/labelme 在写盘线程
完成后 manifest 才标 done（可用极慢 writer 模拟）。

### P1-2 SAM3 跨图批处理（多图一次前向）

**现状**：`engines/sam3.py` 一次前向 = 1 张图 × prompt_batch 个提示。transformers
的 `Sam3Processor(images=[...], text=[...])` 支持 B 图 × P 提示的笛卡尔积布局
（已实测可用）。显存 32G 时 2 图 × 4 提示的矩阵比 1×4 更省激活重复。

**改法**：`engines/sam3.py` 增加 `image_batch` 引擎参数（默认 1），`infer` 改为
接受 `infer_batch(images, prompts)`（BaseEngine 已有默认实现，SAM3 覆盖它）；
`_run_detect_phase` 按 `image_batch` 攒批调用。post_process 返回按批索引的
列表，注意 target_sizes 与 results 的对齐关系。**先实现 `infer_batch` + 单测
（mock 不适用，需真机验证标注一次），CLI/runner 接线在后**。

**验收**：真机（本机 5090）对同一批图分别用 image_batch=1 和 2 跑 detect，
标注 IoU 一致、吞吐提升写入本文件附录。

### P1-3 DA3 深度批推理

**现状**：DA3 官方 `model.inference(images)` 接受图片列表，当前引擎逐图调用。
`DepthAnything3Engine` 覆盖 `infer_batch`：一次传 N 张（同横竖桶），输出按
索引取回再各自 resize 回原尺寸。深度阶段的 batch_size 已有配置，直接生效。

**验收**：mock 单测覆盖批结果与逐图结果的一致性接口约定；真机验证一次
（truck.jpg 合成图，深度中位数与逐图差 <1%）。

### P1-4 深度 PNG 压缩等级可调 + 默认降为快速档

**现状**：`save_depth_png` 用 cv2 默认 PNG 压缩（较高），16-bit 大图压缩是
写盘线程的主要 CPU 消耗。给 `depth.png_compression` 配置（0-9，默认 3），
经 `cv2.IMWRITE_PNG_COMPRESSION` 传入（配合 P0-2 的安全写封装）。

**验收**：单测 roundtrip 不受 compression 影响；同图 c=1 与 c=9 文件大小对比
写进附录（预期小 30-50%、读写快 2 倍+）。

---

## P1 — 易用性改进

### P1-5 `assistlabel status` 命令（零成本看进度）

**需求**：大作业跑完/中断后，用户想知道"多少 done/failed/pending、平均耗时、
预计剩余"必须翻 manifest。加一个只读命令：

```
assistlabel status -c run.yaml
# depth: 8421 done / 3 failed / 1576 pending | avg 0.42s | ETA 11min
# detect: 6000 done / 0 failed / 3 preset ...
```

**改法**：`core/pipeline.py` 的 `Manifest.summary()` 已有雏形，扩展 per-task
统计 + 平均 seconds；CLI 新命令纯读 manifest 与磁盘扫描结果对比。

**验收**：单测覆盖 summary 统计；mock 项目跑一半后 status 输出正确。

### P1-6 YOLO 导出增加 ultralytics 训练布局

**需求**：`export --format yolobox` 现在只输出 labels；用户要训练还得自己
组织 `images/ + labels/ + data.yaml`。加 `--layout ultralytics`：在 out_dir/
`ultralytics/` 下生成标准结构（images/train、images/val、labels/train、labels/val
的硬链接或复制，Windows 硬链接失败自动降级复制）+ `data.yaml`（含 nc/names、
train/val 路径）。

**验收**：单测断言目录结构与 data.yaml 内容；split 生效时 train/val 正确划分。

### P1-7 ontology 变更防呆

**问题**：改了 ontology（加/删类别）后 resume，旧 detect 结果仍标 done，
导出的类别体系与旧标注不一致——静默错误。

**改法**：detect 阶段把 ontology 内容哈希写进 manifest 记录（`manifest.mark(
..., ontology_hash=...)`）；resume 时发现哈希不同则警告并列出受影响数量，
提示 `--redo-detect`（新 CLI 参数：清除全部 detect 状态重跑）。

**验收**：单测改 ontology 后 resume 触发警告路径；--redo-detect 清状态正确。

### P1-8 进度条加 ETA 与失败实时计数

rich 的 `TimeRemainingColumn`（当前 rich 15 有）加入 Progress 列；failed 计数
实时显示在 task description（`detect 3 failed`）。改动极小，纯体验。

---

## P2 — 稳定性/健壮性（重要但可后排）

### P2-1 深度 OOM 后自动降批并记住

**现状**：深度批量 OOM 后逐图降级（已有），但下一批仍用原 batch_size，
每批都要 OOM 一次再降级（每次 OOM 有清理开销）。OOM 捕获处把运行时
batch 减半（不低于 1）并在本次 run 内生效，打印一次提示。

### P2-2 磁盘空间预检

run 启动时估算输出体积（按输入图像总量 × 2.2 系数：16bit PNG + labelme +
viz），对比 out_dir 所在盘剩余空间，不足则拒绝启动并提示（可 `--force` 跳过）。
`shutil.disk_usage` 即可，跨平台。

### P2-3 verify 增强：源图内容变更检测

depth meta 已存 `source_image_hash`；verify 增加校验：当前源图哈希 != meta
记录 → 报 `stale` 类 issue（图换过内容，标注过期），`--repair` 同样重置。

### P2-4 KeyboardInterrupt 干净退出

run 命令捕获 KeyboardInterrupt：保存 manifest、打印已完成统计、exit code 130，
不打印 traceback。（finally 里已保存 manifest，主要补 CLI 层的友好输出。）

### P2-5 深度值域哨兵

推理后检查：全图无效像素(=0)占比 >95% 或值域越界（<0.1m 或 >200m）视为
引擎异常，该图标 failed 而不是落盘坏真值（防模型/预处理错位静默产出垃圾）。

---

## P3 — 远期方向（只记录，不排期）

- 多 GPU 分片：`--gpu 0,1` 每卡一个 worker 进程按图取模分片，manifest 天然支持
- transformers 的 Sam3VideoModel 成熟后恢复 `assistlabel video`
- `init` 交互式问答（图片目录、类别清单）替代手改 yaml
- FiftyOne 数据集描述导出，做交互式 QA
- 深度标定：支持用户给已知距离（如相机高度）做尺度对齐（DA3 metric 的焦距
  估计有系统偏差时用）

---

## 实施顺序与完成定义

1. P0-1 → P0-2 → P0-3 → P0-4（全绿后提交一次）
2. P1-1 → P1-4 → P1-5 → P1-7 → P1-8（每项单测全绿）
3. P1-2 → P1-3（需真机验证，附录记录数据）
4. P1-6 → P2 逐项
5. 最终：README 同步全部新配置/命令，`pytest` 全绿，真机跑一次
   `run --limit 50 + export 三格式 + verify` 收尾

每项完成定义：单测覆盖 + 不破坏既有 57 项测试 + 涉及配置/命令的同步
RUN_YAML_TEMPLATE 与 README。
