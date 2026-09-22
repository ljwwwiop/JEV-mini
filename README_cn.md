# jev-mini [![README English](https://img.shields.io/badge/README-English-blue)](README.md) [![README Chinese](https://img.shields.io/badge/README-中文-red)](README_cn.md)

基于 Qwen + PointerHead 的轻量决策模型，输出候选项概率。代码抽取自 Kev，非 Jev 官方实现。

> Progress always follows an upward spiral. Make classifiers great again!（事物的发展总是螺旋上升的，让分类器再次伟大！）

> [!IMPORTANT]
> 当前方案仍然采用 **Decoder + 分类头** 的策略。对于分类任务，相比 BERT 这类 Encoder 架构，当前设计仍有不少冗余与可精简之处。我们计划近期推出一个**更简单、更直接的版本**，也会尽快训练一个**用于学习和实验的多模态版本**，敬请期待！

## 🚀 快速开始

需要 Python >=3.10，并自行准备本地 Qwen3-0.6B 权重与训练数据。

```bash
python -m pip install -e .
# 修改 config/train.yaml 中的模型、数据和输出路径后运行
python -m jev_mini.train --config config/train.yaml
# 将 config/eval.yaml 的 run 改为实际训练输出目录
bash run_eval.sh config/eval.yaml
# 单条请求预测
python -m jev_mini.predict --run runs/train --input examples/request.json
```

训练输出目录必须尚不存在。`run_train.sh` 支持多卡训练和带时间戳的输出目录，但包含训练机器的 Conda、项目及模型挂载路径，使用前需按本机修改。

## ⚙️ 数据与配置

每行一个 JSON，示例见 [`examples/train.jsonl`](examples/train.jsonl)。支持 `choice`（选项概率）、`noul`（P(true)）和 `score`（等级期望）；蒸馏格式 `state/question/kind/options/target` 也可直接加载，`target` 归一化后用于软标签训练。

- 🏋️ **训练与验证：**[`config/train.yaml`](config/train.yaml)，设置 `base`、`data`、`val_data`、`val_step` 和 `val_samples`。
- 🧪 **独立测试：**[`config/eval.yaml`](config/eval.yaml)，设置 checkpoint 路径 `run` 和测试集 `data`。
- 📏 **上下文：**`max_state`、`max_branch`、`max_packed` 控制长度；`overlength: skip` 跳过超长记录，不截断文本。

验证汇总写入 `validation_metrics.jsonl`，逐条结果写入 `validation/step_*.json`。训练曲线可通过 `tensorboard --logdir runs` 查看。训练结束后保存 adapter、指针头、tokenizer 和训练配置，不包含基座权重及优化器状态。

## 🎯 推理结果示例

以下为第 500 步验证中的一个真实工具选择 sample，完整保留原始字段和预测结果，来源为 [`validation/step_00000500.json`](runs/train_2026_09_22_123806_530328/validation/step_00000500.json)。`probabilities` 与 `options` 顺序一致，`index` 从 0 开始。

```json
{
  "id": "v3_3a61fc66b29476cd_c",
  "kind": "choice",
  "options": [
    "database_query",
    "api_call",
    "file_read",
    "web_search"
  ],
  "target": [
    0.83,
    0.1,
    0.07,
    0.0
  ],
  "state": "An assistant pipeline must validate an email format; additionally requires exact-match lookup in a 5-row table. Context: on the first attempt, under reduced staffing. Reported by the on-call engineer.",
  "question": "Best tool for this scenario.",
  "domain": "tool_selection",
  "family": "agent",
  "source": "yuri_v3",
  "model_prediction": {
    "status": "ok",
    "index": 0,
    "key": "0",
    "option": "database_query",
    "keys": [
      "0",
      "1",
      "2",
      "3"
    ],
    "probabilities": [
      0.7705801129341125,
      0.10950907319784164,
      0.11912398785352707,
      0.0007867840467952192
    ]
  }
}
```

模型选择 `database_query`，概率约 **77.06%**；教师分布中该选项为 **83%**。

同次运行的 [`validation_metrics.jsonl`](runs/train_2026_09_22_123806_530328/validation_metrics.jsonl) 记录了第 500 步的汇总结果：接纳 13,963 条记录，跳过 148 条超长记录；与教师最大概率选项的一致率为 **80.22%**，软标签交叉熵为 **0.7423**，Brier 分数为 **0.0419**。这里的一致率不是人工真值准确率。

## 📂 代码结构

```text
jev_mini/   # 数据、模型、损失、训练、评估与预测
config/     # 训练和评估配置
examples/   # 数据与请求示例
tests/      # 单元测试
runs/       # 运行记录与验证结果
```

🧪 运行测试：`python -m unittest discover -s tests -v`。

## 🤝 来源、许可与致谢

派生自 Jared Palmer 的 [Kev](https://github.com/jaredpalmer/kev) 项目，采用 [Apache-2.0](LICENSE) 许可证。保留核心模型与损失实现，简化训练流程，非原版训练配方的完整复现。

感谢 [Jared Palmer / Kev](https://github.com/jaredpalmer/kev) 提供开源实现，以及 [SargeDev / jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) 提供蒸馏数据集。
