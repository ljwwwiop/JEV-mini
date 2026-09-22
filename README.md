# jev-mini

从本地 Kev 项目抽取的 Jev 风格决策模型训练核心。基于 Qwen + LoRA + PointerHead，输出候选项概率，不生成文本，也不是 Jev 官方实现。

## 文件结构与阅读顺序

```text
jev-mini/
├── jev_mini/
│   ├── schema.py       # ① Choice/Noul/Score 请求定义、统一文本格式
│   ├── data.py         # ② JSONL 加载、标签转换、选项数据增强
│   ├── model.py        # ③ token 编码、分支 mask、Qwen/LoRA、指针头
│   ├── losses.py       # ④ CE/软标签、ordinal loss、anchor/permutation KL
│   ├── train.py        # ⑤ 简化训练入口、梯度累积、保存
│   ├── checkpoint.py   # ⑥ 权重元数据、本地/Hub 加载、warm_start API
│   ├── predict.py      # ⑦ 加载 checkpoint，对 JSON 请求输出概率
│   └── device.py       # CPU/CUDA/MPS 设备工具
├── examples/
│   ├── train.jsonl     # 三种题型的带标签示例（仅用于 smoke）
│   └── request.json    # 对应的无标签预测请求
├── tests/test_core.py
├── pyproject.toml
└── LICENSE
```

完整路径是 `JSONL → schema/data → model.encode → model.forward_batch → question_loss → backward → AdamW → checkpoint`。

## 配置文件启动（推荐）

训练参数统一修改 `config/train.yaml`，预测参数修改 `config/eval.yaml`。Shell 设置配置路径、可见 GPU，并为训练输出目录添加时间戳：

```bash
bash run_train.sh                         # 默认 GPU 5、config/train.yaml
CUDA_VISIBLE_DEVICES=0,1 bash run_train.sh # 两张 GPU，torchrun + DDP
bash run_train.sh config/train.yaml       # 指定配置文件
bash run_eval.sh config/eval.yaml         # 独立 test split 批量评估
```

脚本会切换到项目目录，YAML 中的相对路径均以项目目录为基准。训练启动脚本以 `train.yaml` 的 `out` 为前缀，自动追加时间戳（含微秒），例如 `runs/train_2026_09_22_143025_123456`，每次启动生成新的输出路径。直接运行 Python 训练模块时，输出目录仍必须尚不存在。rank 0 保存原始 `train.yaml` 和解析后的 `training_config.json`。

多卡数量从可见设备列表自动计算，每卡一个进程，每卡保存完整模型副本；有效 batch 为 `batch × accum × GPU 数量`。DistributedSampler 在样本数不能整除卡数时会补齐少量重复样本。日志显示 rank 0 的损失。GPU 多卡训练尚未实机验证。

`eval.yaml` 连接 `evaluate.py`，对 `data/test.jsonl` 批量评估。`train.yaml` 读取 `data/train.jsonl`，每 `val_step` 次参数更新（默认 500）由 rank 0 静默评估 `data/val.jsonl` 的前 `val_samples` 条记录（默认 1 条） 并写入 `validation_metrics.jsonl`，训练结束时若未到验证间隔则补做一次验证，验证数据不参与梯度更新。

`data/train.jsonl`、`data/val.jsonl`、`data/test.jsonl` 是指向同级 `dataset/jev-distill-corpus-v3` 的相对软链接；val 对应源文件 `validation.jsonl`。源文件保持原样。加载器自动将 `kind/options/target` 转换为内部请求，保留完整教师分布并归一化；argmax 仅用于辅助标签和准确率。报告中的 accuracy 是与教师最大概率选项的一致率，不是人工真值准确率。另报告软标签交叉熵和相对教师分布的 Brier 分数。

本次按要求使用 `test.jsonl`；数据集另有推荐的 `test_set_30k.jsonl`，未混入训练或验证。上下文限制由 train/eval YAML 的 `max_state`、`max_branch`（含 state）、`max_packed` 指定。默认 `overlength: skip` 跳过超长样本，不截断文本；设为 `error` 则严格报错。训练记录筛选数量到 `data_admission.json`，验证/测试指标包含 `requested_records` 和 `skipped_overlength`，指标仅针对实际接纳的子集。

## 安装与运行

在本目录运行，使用已适配的 jev 环境：

```bash
conda activate jev
python -m pip install -e .
python -m jev_mini.train --data examples/train.jsonl --out runs/smoke
python -m jev_mini.predict --run runs/smoke --input examples/request.json
python -m unittest discover -s tests -v
```

默认基座为本地 `checkpoint/Qwen3-0.6B`（指向 `/lpai/inputs/models/Qwen__Qwen3-0.6B/main`），配置保存绝对路径。模型、tokenizer 和 adapter 加载均设置 `local_files_only=True`，启动脚本开启 Hugging Face 离线模式；文件缺失时直接报错，不下载权重。这个本地目录是 `Qwen3-0.6B`，与此前的 `Qwen3-0.6B-Base` 名称不同。示例仅一条，不能训练出有实际效果的模型。预测 `--run` 也支持原版 Kev checkpoint 目录或 Hub ID（如 `jaredpalmer/kev-4b`）。

CUDA 训练示例：

```bash
python -m jev_mini.train \
  --base Qwen/Qwen3.5-4B-Base \
  --data /path/to/train.jsonl --out runs/custom \
  --device cuda --bf16 --checkpointing \
  --epochs 2 --lr 5e-5 --batch 1 --accum 8 --augment
```

依赖下限沿用源项目；Qwen3.5 的 CUDA 加速内核 `flash-linear-attention`/Triton 没有纳入最小依赖。`--bf16` 是 CUDA 前向 autocast，主干主权重仍为 fp32，不能按 bf16 推理显存估算训练显存。

## 数据格式

每行一个 JSON，请参照 `examples/train.jsonl`。`state` 可为字符串或结构化 JSON，`questions` 是以问题 ID 为键的对象。

| 类型 | criteria | label | 预测结果 |
|---|---|---|---|
| choice | 选项名到描述的字典 | 正确选项名 | 概率和最大概率选项 |
| noul | 可选 true/false 描述 | JSON 布尔值 | P(true) |
| score | 从低到高的等级描述数组 | 从 0 开始的整数 | 等级概率的期望 |

可选 `target` 为选项键到非负权重的字典；加载后归一化为软标签，仍需提供合法 `label`。数据过长按 YAML 的 `overlength` 处理，不会静默截断。原始上下文限制沿用 Kev：state 含标记最多 384 token，state+单问题分支最多 1024 token，整条打包记录最多 2048 token。

`--augment` 启用原版选项乱序、干扰项和“以上都不是”增强；不开启时按原数据训练。自行准备独立验证集，mini 入口不执行数据切分或自动评测。

## 保留与简化

- `model.py` 保留原版训练前向计算，删除推理 KV prefix cache。纯注意力基座使用 block-causal mask；Qwen3.5 hybrid 为每个问题建立独立 causal row。
- `losses.py` 的公式直接抽取自原版。CLI 使用交叉熵/软标签，可通过 `--ord-w` 加入等级 RPS；anchor KL 和 permutation KL 仅保留函数，未接入 CLI。
- `train.py` 重写为短训练循环：固定学习率 AdamW、按问题平均再按记录平均、正确处理最后不足一个梯度累积组的记录、梯度裁剪。省去原版 OneCycleLR、冻结套件管理、混合实验及发布流程，**不是原版发布模型训练配方的完整复现**。
- `checkpoint.py` 沿用原版格式和加载实现，将唯一的 suite 哈希依赖换成 hashlib。保留 warm_start Python API，但 mini CLI 暂不提供 `--init_from`。
- `schema.py` 只保留请求类型与文本转换，不包含 Web 服务或日期预处理。`data.py` 保留 JSONL 路径，移除公开数据集自动下载器，并补充标签范围与软标签有效性校验。
- 保存产物包含 LoRA adapter、`head.pt`、tokenizer 和 `training_config.json`。不包含基座权重、优化器状态或自动温度校准；新训练的温度为 1，不能精确断点续训。

## 来源与验证

源代码来自同级 `kev/kev/{model,api,data,train,checkpoint,device}.py`，原作者 Jared Palmer，Apache-2.0；原始许可证保存在 LICENSE。此目录是可独立安装的精简派生副本，没有导入同级 kev 包。

本次环境已验证 Python 语法、三种题型转换、增强标签一致性、非法标签/软标签拒绝、CE/软标签公式与反向传播、permutation KL 对齐。环境未安装 Transformers/PEFT，未下载真实基座，因此完整训练、真实 checkpoint 加载和 GPU 前向尚未实测。

## casual 环境启动排查

2026-09-22 实测：Python 3.10.18、torch 2.8.0+cu128、transformers 4.49.0；缺少 peft，现有 Transformers 不支持本地 Qwen3 配置。项目声明 Python >=3.12，完整环境依赖见 pyproject.toml。本次未安装或修改任何包。train 启动时执行只读环境预检，在扫描训练数据前报告缺失模块及不支持的模型类型。

原启动错误为 `state exceeds 384 tokens: 490`，现在可选择统计跳过或调整上下文上限。长上下文会增加显存占用；增大上限时应一并评估 batch。

## 已适配的 jev 环境（本地 Qwen3）

使用 `conda activate jev` 后运行 `bash run_train.sh`，不使用 uv。当前验证组合为 Python 3.10、torch 2.8.0+cu128、transformers 4.57.6、peft 0.18.0、accelerate 1.12.0。pyproject.toml 已按此 Qwen3 路径调整；Qwen3.5 不属于这套依赖组合的支持范围。全程从本地 checkpoint 读取模型。

正式训练日志位于 `logs/train.log`，每个优化步骤输出 `rank0_loss`；这是梯度累积组的平均记录损失。先执行全量数据上下文筛选，期间输出 `context check`，筛选完成后开始打印 loss。

当前本地模型入口 `checkpoint/Qwen3-0.6B` 是普通目录，其中的文件软链接到原始模型挂载；旧目录软链接保存在 `checkpoint/Qwen3-0.6B.source`。这样避免挂载层对缺失 `chat_templates/` 等路径的查询阻塞，权重内容未变，也没有联网下载。


### 训练日志和定期验证

`config/train.yaml` 中的 `val_step` 按累计 optimizer step 计数，跨 epoch 不重置。`val_samples: 1` 表示每次固定取验证集第一条记录，不读取后续记录；可增大该值以评估前 N 条。训练进度条保持单行刷新，验证期间不打印结果。每步训练损失写入本次输出目录的 `tensorboard/`，标签为 `train/loss`（多卡时为各 rank 平均值）；验证交叉熵、准确率和 Brier 分数写入 `val/` 标签。

```bash
tensorboard --logdir runs --port 6006
```

每次验证生成 `validation/step_00000500.json` 等独立 JSON 数组文件，逐条保留验证 JSONL 的原始字段及顺序，并添加 `model_prediction`。扁平蒸馏记录包含 `status`、预测 `index`（从 0 开始）、`key`、`option`、`keys` 和 `probabilities`；概率顺序与输入 `options` 一致。原生多问题记录的预测存放在 `model_prediction.questions` 下，按 question ID 对应。超长记录保留并标记 `status: skipped_overlength`，不生成虚构概率。输入不得已有 `model_prediction` 字段。文件完整写入后才发布为 `.json`。

LoRA checkpoint 仍在全部训练和验证正常结束后保存到本次输出目录；验证 JSON 并非模型 checkpoint。新增依赖为 `tensorboard>=2.14`，更新环境可执行 `python -m pip install -e .`。正在运行的训练不会自动应用代码变更。
