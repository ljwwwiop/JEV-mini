"""JSONL -> Qwen + LoRA + pointer head -> local checkpoint."""
import argparse
import contextlib
import json
import os
import shutil
import random
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .config import parse_config
from .context import Context, add_context_args
from .preflight import check_environment
from .checkpoint import Meta, write_meta
from .data import augment, iter_records, load_records, materialize
from .evaluate import evaluate_to_json
from .device import default_device
from .losses import question_loss
from .model import DecisionModel, MAX_PACKED, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Labelled JSONL file')
    parser.add_argument('--val-data', help='Validation JSONL, evaluated every val-step optimizer updates')
    parser.add_argument('--val-samples', type=int, default=1, help='Evaluate only the first N validation records')
    parser.add_argument('--val-step', type=int, default=500, help='Optimizer updates between validations')
    parser.add_argument('--out', required=True, help='New checkpoint directory')
    parser.add_argument('--base', default='xxx')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lora', type=int, default=16)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--bf16', action='store_true', help='CUDA autocast; master weights stay fp32')
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--augment', action='store_true', help='Original Kev Choice permutation/none/distractor augmentation')
    parser.add_argument('--ord-w', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    add_context_args(parser)
    args = parse_config(parser)
    context = Context.from_args(args)
    check_environment(args.base)
    if min(args.epochs, args.batch, args.accum, args.lora, args.head_dim, args.val_step, args.val_samples) < 1 or args.lr <= 0 or args.ord_w < 0:
        parser.error('counts and learning rate must be positive; ord-w must be nonnegative')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = args.device or default_device()
    distributed = world_size > 1
    if distributed:
        if device != 'cuda':
            parser.error('multi-GPU training requires device: cuda')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend='nccl')
    if args.bf16 and device != 'cuda':
        parser.error('--bf16 requires CUDA')
    out = Path(args.out)
    if out.exists():
        parser.error('output directory already exists; choose a new --out')
    if distributed:
        torch.distributed.barrier()  # all ranks check before rank 0 creates the directory
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if args.val_data and not Path(args.val_data).is_file():
        parser.error(f'validation file does not exist: {args.val_data}')
    records = load_records(args.data)
    tokenizer = load_tokenizer(args.base, revision=args.revision)
    # Filter before the distributed sampler so every rank sees the same admitted data.
    requested_records = len(records)
    admitted = []
    for index, request in enumerate(records, 1):
        try:
            encoded = context.encode(tokenizer, materialize(request))
        except ValueError as exc:
            raise ValueError(f'{args.data}: record {index}: {exc}') from exc
        if encoded is not None:
            admitted.append(request)
        if rank == 0 and index % 10000 == 0:
            print(f'context check: {index}/{requested_records}, admitted={len(admitted)}', flush=True)
    records = admitted
    admission = {'requested': requested_records, 'admitted': len(records),
                 'skipped_overlength': requested_records - len(records), 'context': vars(context)}
    if rank == 0:
        print(f'training data: {json.dumps(admission)}', flush=True)
    if not records:
        raise ValueError('no training records fit the configured context')
    model = DecisionModel(args.base, tokenizer, device, lora=args.lora,
                          revision=args.revision, head_dim=args.head_dim)
    model.lm.config.use_cache = False
    if args.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'data_admission.json').write_text(json.dumps(admission, indent=2), encoding='utf-8')
        (out / 'training_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
        if args.config:
            shutil.copyfile(args.config, out / 'train.yaml')
    if distributed:
        torch.distributed.barrier()
    model.train()
    # DDP needs a forward method accepting a batch of encoded records.
    class BatchForward(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, encodings):
            return self.inner.forward_batch(encodings)

    forward = BatchForward(model)
    if distributed:
        forward = torch.nn.parallel.DistributedDataParallel(
            forward, device_ids=[int(os.environ['LOCAL_RANK'])])
    writer = SummaryWriter(log_dir=str(out / 'tensorboard'), flush_secs=10) if rank == 0 else None

    def validate(step, epoch):
        if distributed:
            torch.distributed.barrier()
        if rank == 0:
            metrics = evaluate_to_json(model, tokenizer, args.val_data,
                                       out / 'validation' / f'step_{step:08d}.json',
                                       args.batch, context=context, max_records=args.val_samples)
            metrics.update(step=step, epoch=epoch)
            with (out / 'validation_metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            for name in ('soft_target_cross_entropy', 'teacher_argmax_accuracy', 'brier_to_target'):
                writer.add_scalar(f'val/{name}', metrics[name], step)
            writer.flush()
        if distributed:
            torch.distributed.barrier()

    try:
        step = 0
        for epoch in range(args.epochs):
            if distributed:
                sampler = torch.utils.data.DistributedSampler(records, num_replicas=world_size, rank=rank, seed=args.seed)
                sampler.set_epoch(epoch)
                epoch_records = [records[i] for i in sampler]
            else:
                rng.shuffle(records)
                epoch_records = records
            # Each optimizer step averages actual source records, including the last partial group.
            steps_per_epoch = (len(epoch_records) + args.batch * args.accum - 1) // (args.batch * args.accum)
            # Only rank 0 renders; tqdm refreshes the same terminal line.
            with tqdm(total=steps_per_epoch, desc=f'Epoch {epoch + 1}/{args.epochs}',
                      unit='step', dynamic_ncols=True, leave=True, disable=rank != 0,
                      postfix=f'step={step} loss=--') as progress:
                for start in range(0, len(epoch_records), args.batch * args.accum):
                    group = epoch_records[start:start + args.batch * args.accum]
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    for offset in range(0, len(group), args.batch):
                        chunk = group[offset:offset + args.batch]
                        recs, encs = [], []
                        for original in chunk:
                            rec = materialize(augment(original, rng) if args.augment else original)
                            enc = context.encode(tokenizer, rec, model.encode)
                            if enc is None:
                                # Augmentation may add tokens: fall back to the admitted original.
                                rec = materialize(original)
                                enc = context.encode(tokenizer, rec, model.encode)
                            if enc is None:
                                raise ValueError('previously admitted record no longer fits')
                            recs.append(rec)
                            encs.append(enc)
                        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext()
                        with autocast:
                            predictions = forward(encs)
                            loss = sum(
                                sum(question_loss(z.float(), q, device, args.ord_w)
                                    for z, q in zip(logits, rec['questions'])) / len(logits)
                                for logits, rec in zip(predictions, recs)
                            ) / len(group)
                        if not torch.isfinite(loss):
                            raise ValueError('non-finite loss')
                        loss.backward()
                        total += loss.item()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    step += 1
                    logged_loss = torch.tensor(total, device=device)
                    if distributed:
                        torch.distributed.all_reduce(logged_loss)
                        logged_loss /= world_size
                    if rank == 0:
                        writer.add_scalar('train/loss', logged_loss.item(), step)
                        progress.set_postfix_str(f'step={step} loss={total:.4f}', refresh=False)
                        progress.update(1)
                    if args.val_data and step % args.val_step == 0:
                        validate(step, epoch + 1)
        # Also evaluate the final weights when training ends between validation steps.
        if args.val_data and step % args.val_step:
            validate(step, args.epochs)
        if rank == 0:
            model.lm.save_pretrained(out)
            tokenizer.save_pretrained(out)
            write_meta(out, Meta(base=args.base, base_revision=args.revision, lora=args.lora,
                                 head_dim=args.head_dim, head=model.head.state_dict(), extra={'args': vars(args)}))
            print(f'Saved checkpoint: {out}')
    finally:
        if writer is not None:
            writer.close()
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()



if __name__ == '__main__':
    main()
