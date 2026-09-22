# jev-mini [![README English](https://img.shields.io/badge/README-English-blue)](README.md) [![README Chinese](https://img.shields.io/badge/README-中文-red)](README_cn.md)

A lightweight decision model built on Qwen + LoRA + PointerHead that outputs probabilities over candidate options. Extracted from Kev; this is not an official Jev implementation.

> Progress always follows an upward spiral. Make classifiers great again!

> [!IMPORTANT]
> The current approach still uses a **Decoder + classification head**. For classification tasks, it retains considerable redundancy and room for simplification compared with Encoder architectures such as BERT. We plan to release a **simpler, more direct version** soon. Stay tuned!

## Quick Start

Requires Python >=3.10. Prepare local Qwen3-0.6B weights and your training data before running.

```bash
python -m pip install -e .
# Update the model, data, and output paths in config/train.yaml
python -m jev_mini.train --config config/train.yaml
# Set run in config/eval.yaml to the actual training output directory
bash run_eval.sh config/eval.yaml
# Predict probabilities for a single request
python -m jev_mini.predict --run runs/train --input examples/request.json
```

The training output directory must not already exist. `run_train.sh` supports multi-GPU training and timestamped output directories, but contains machine-specific Conda, project, and model mount paths. Update these for your environment before use.

## Data and Configuration

Use one JSON object per line; see [`examples/train.jsonl`](examples/train.jsonl). Supported question types are `choice` (option probabilities), `noul` (P(true)), and `score` (expected rating). The distillation format `state/question/kind/options/target` is also supported directly, with `target` normalized for soft-label training.

- Training and validation: [`config/train.yaml`](config/train.yaml). Set `base`, `data`, `val_data`, `val_step`, and `val_samples`.
- Independent testing: [`config/eval.yaml`](config/eval.yaml). Set the checkpoint path `run` and test dataset `data`.
- Context limits: `max_state`, `max_branch`, and `max_packed` control input lengths. `overlength: skip` skips oversized records without truncating text.

Validation summaries are written to `validation_metrics.jsonl`, and per-record predictions to `validation/step_*.json`. View training curves with `tensorboard --logdir runs`. At the end of training, the project saves the LoRA adapter, pointer head, tokenizer, and training configuration; base model weights and optimizer state are not included.

## Example Prediction

This is a real tool-selection sample from validation at step 500, with all original fields and prediction results preserved. Source: [`validation/step_00000500.json`](runs/train_2026_09_22_123806_530328/validation/step_00000500.json). The `probabilities` array follows the order of `options`, and `index` is zero-based.

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

The model selects `database_query` with a probability of approximately **77.06%**, compared with **83%** in the teacher distribution.

The same run's [`validation_metrics.jsonl`](runs/train_2026_09_22_123806_530328/validation_metrics.jsonl) reports the following at step 500: 13,963 accepted records, 148 oversized records skipped, **80.22%** agreement with the teacher's highest-probability option, **0.7423** soft-target cross-entropy, and a **0.0419** Brier score. Teacher agreement is not accuracy against human ground-truth labels.

## Project Structure

```text
jev_mini/   # Data, model, losses, training, evaluation, and prediction
config/     # Training and evaluation configuration
examples/   # Sample data and requests
tests/      # Unit tests
runs/       # Run records and validation results
```

Run tests with `python -m unittest discover -s tests -v`.

## Origins, License, and Acknowledgments

Derived from Jared Palmer's [Kev](https://github.com/jaredpalmer/kev) project under the [Apache-2.0](LICENSE) license. This project retains the core model and loss implementations while simplifying the training workflow; it is not a complete reproduction of the original training recipe.

Thanks to [Jared Palmer / Kev](https://github.com/jaredpalmer/kev) for the open-source implementation and [SargeDev / jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) for the distillation dataset.
