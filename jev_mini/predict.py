"""Load a Kev-compatible checkpoint and score unlabelled JSON requests."""
import argparse
import json
from pathlib import Path

from .config import parse_config
from .checkpoint import Checkpoint
from .device import default_device
from .model import MAX_PACKED
from .schema import SystemOneRequest, to_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Local checkpoint or Hub id[@revision]')
    parser.add_argument('--input', required=True, help='One JSON request (labels are unnecessary)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--out', help='Optional output JSON path')
    args = parse_config(parser)
    request = SystemOneRequest.model_validate(json.loads(Path(args.input).read_text(encoding='utf-8')))
    rec, metadata = to_record(request)
    tokenizer, model = Checkpoint(args.run).load(args.device or default_device())
    enc = model.encode(tokenizer, rec, strict=True)
    if len(enc['ids']) > MAX_PACKED:
        raise ValueError('request exceeds packed context limit')
    result = {}
    for probabilities, meta in zip(model.probs(enc), metadata):
        p = probabilities.tolist()
        answer = {'probabilities': dict(zip(meta['keys'], p))}
        if meta['type'] == 'choice':
            answer['choice'] = meta['keys'][max(range(len(p)), key=p.__getitem__)]
        elif meta['type'] == 'noul':
            answer['noul'] = p[1]
        else:
            answer['score'] = sum(i * value for i, value in enumerate(p))
        result[meta['id']] = answer
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
