"""Evaluate labelled JSONL; accuracy is agreement with teacher argmax."""
import argparse
import contextlib
from collections import Counter
import itertools
import json
from pathlib import Path

import torch

from .config import parse_config
from .context import Context, add_context_args
from .data import iter_records, materialize


def evaluate_records(model, tokenizer, records, batch=1, context=None, on_prediction=None):
    context = context or Context()
    if batch < 1:
        raise ValueError('batch must be positive')
    was_training = model.training
    model.eval()
    totals = Counter()
    by_kind = {}
    iterator = iter(records)
    try:
        with torch.no_grad():
            while chunk := list(itertools.islice(iterator, batch)):
                recs, encs = [], []
                outputs = []
                for request in chunk:
                    rec = materialize(request)
                    enc = context.encode(tokenizer, rec, model.encode)
                    outputs.append({'status': 'skipped_overlength', 'questions': {}})
                    totals['requested_records'] += 1
                    if enc is None:
                        totals['skipped_overlength'] += 1
                        continue
                    outputs[-1] = {'status': 'ok', 'questions': {}}
                    recs.append((rec, outputs[-1]))
                    encs.append(enc)
                predictions = model.forward_batch(encs) if encs else []
                for logits, (rec, output) in zip(predictions, recs):
                    for z, q in zip(logits, rec['questions']):
                        lp = z.float().log_softmax(-1)
                        if on_prediction is not None:
                            index = int(lp.argmax().item())
                            output['questions'][q['qid']] = {
                                'index': index, 'key': q['keys'][index],
                                'option': q['options'][index],
                                'keys': q['keys'], 'probabilities': lp.exp().cpu().tolist()}
                        target = torch.zeros_like(lp)
                        if 'target' in q:
                            target = torch.tensor(q['target'], device=lp.device, dtype=lp.dtype)
                        else:
                            target[q['label']] = 1
                        stats = {'questions': 1, 'correct': int(lp.argmax().item() == q['label']),
                                 'cross_entropy': -(target * lp).sum().item(),
                                 'brier': (lp.exp() - target).square().sum().item()}
                        totals.update(stats)
                        by_kind.setdefault(q['qtype'], Counter()).update(stats)
                if on_prediction is not None:
                    for request, output in zip(chunk, outputs):
                        on_prediction(request, output)
                totals['records'] += len(recs)
    finally:
        model.train(was_training)
    if not totals['questions']:
        raise ValueError('empty evaluation set')

    def summarize(stats):
        n = stats['questions']
        return {'questions': n, 'teacher_argmax_accuracy': stats['correct'] / n,
                'soft_target_cross_entropy': stats['cross_entropy'] / n,
                'brier_to_target': stats['brier'] / n}

    return {'records': totals['records'], 'requested_records': totals['requested_records'],
            'skipped_overlength': totals['skipped_overlength'], **summarize(totals),
            'by_kind': {kind: summarize(stats) for kind, stats in by_kind.items()}}



def evaluate_to_json(model, tokenizer, data, output, batch=1, context=None, max_records=None):
    """Stream original validation rows plus predictions into an atomic JSON array."""
    if max_records is not None and max_records < 1:
        raise ValueError('max_records must be positive')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write('[\n')
            first = True

            def save_prediction(request, prediction):
                nonlocal first
                row = dict(request['_raw_record'])
                if 'model_prediction' in row:
                    raise ValueError('input already contains model_prediction')
                if 'questions' not in row and prediction['status'] == 'ok':
                    question = prediction['questions']['decision']
                    prediction = {'status': 'ok', **question}
                    prediction['option'] = row['options'][question['index']]
                row['model_prediction'] = prediction
                if not first:
                    stream.write(',\n')
                json.dump(row, stream, ensure_ascii=False, allow_nan=False)
                first = False

            with contextlib.closing(iter_records(data, preserve_raw=True)) as records:
                selected = itertools.islice(records, max_records) if max_records is not None else records
                metrics = evaluate_records(model, tokenizer, selected,
                                           batch, context=context, on_prediction=save_prediction)
            stream.write('\n]\n')
        temporary.replace(output)
        return metrics
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--batch', type=int, default=1)
    add_context_args(parser)
    args = parse_config(parser)
    context = Context.from_args(args)
    if args.batch < 1:
        parser.error('batch must be positive')
    if not Path(args.data).is_file():
        parser.error(f'data file does not exist: {args.data}')
    from .checkpoint import Checkpoint
    from .device import default_device
    tokenizer, model = Checkpoint(args.run).load(args.device or default_device())
    result = evaluate_records(model, tokenizer, iter_records(args.data), args.batch, context=context)
    result.update({'run': args.run, 'data': args.data, 'temperature': model.head.temperature})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
