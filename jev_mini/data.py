"""JSONL loading, option augmentation, and conversion to model records."""
import copy
import hashlib
import json
import math
from pathlib import Path
from .schema import SystemOneRequest, to_record

NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
                ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
                ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
                ("other", "An answer not covered by the other options"), ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}

def source_seed(seed, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order (always); sometimes add a 'none of the above' option, either as the correct
    answer (true option removed) or as a wrong alternative (true option kept); sometimes add an irrelevant distractor."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q; continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:                       # soft-target questions: permute only; inserting or swapping options would change the target's meaning
            keys = list(crit); rng.shuffle(keys); out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}; continue
        r = rng.random()
        none_options = [(k, v) for k, v in NONE_OPTIONS if k not in crit]
        distractors = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            nk, nd = rng.choice(none_options); crit.pop(y); crit[nk] = nd; y = nk
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            nk, nd = rng.choice(none_options); crit[nk] = nd
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            k = rng.choice(distractors); crit[k] = DISTRACTORS[k]
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}, "label": y}
    return out


def distill_request(row):
    """Convert a distillation row without discarding its teacher distribution."""
    kind, options, target = row['kind'], row['options'], row['target']
    if kind not in ('choice', 'noul', 'score'):
        raise ValueError(f'unsupported kind: {kind}')
    if not isinstance(options, list) or not options or not all(isinstance(x, str) for x in options):
        raise ValueError('options must be a nonempty list of strings')
    if not isinstance(target, list) or len(target) != len(options):
        raise ValueError('target and options must have matching lengths')
    if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in target):
        raise ValueError('target must contain finite nonnegative numbers')
    mass = sum(target)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('target must have positive finite mass')
    target = [x / mass for x in target]
    label = max(range(len(target)), key=target.__getitem__)
    q = {'type': kind, 'instructions': row['question'], 'src': row.get('source', 'distill')}
    if kind == 'noul':
        if options != ['false', 'true']:
            raise ValueError('noul options must be ["false", "true"]')
        keys = options
        q['label'] = bool(label)
    else:
        # Index keys preserve ordering, even when option descriptions repeat.
        keys = [str(i) for i in range(len(options))]
        q['criteria'] = dict(zip(keys, options)) if kind == 'choice' else options
        q['label'] = str(label) if kind == 'choice' else label
    q['target'] = dict(zip(keys, target))
    return {'state': row['state'], 'questions': {'decision': q},
            '_meta': {k: row[k] for k in ('id', 'source', 'domain', 'family') if k in row}}


def iter_records(path, source='custom', preserve_raw=False):
    """Stream either native labelled requests or flat distillation JSONL."""
    count = 0
    with Path(path).open(encoding='utf-8') as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                raw = copy.deepcopy(r) if preserve_raw else None
                if 'questions' not in r and 'kind' in r:
                    r = distill_request(r)
                if 'state' not in r or not isinstance(r.get('questions'), dict) or not r['questions']:
                    raise ValueError('a record needs state and a nonempty questions object')
                for q in r['questions'].values():
                    if 'label' not in q:
                        raise ValueError('question has no label')
                    q.setdefault('src', f"{source}_{q['type']}")
                r['_meta'] = {'source': source, 'id': f'{source}/{n}', **r.get('_meta', {})}
                # Validate before yielding, for identical train/eval acceptance rules.
                materialize(r)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{path}:{n}: {exc}') from exc
            count += 1
            if preserve_raw:
                r['_raw_record'] = raw
            yield r
    if not count:
        raise ValueError(f'{path}: no records')


def load_records(path, source='custom'):
    return list(iter_records(path, source))


def api_request(record):
    """The /v1/systemone request body for a labelled record: state and typed questions only, never labels, targets or
    metadata (this is what leaves the machine when a remote predictor is scored)."""
    return {"state": record["state"], "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in record["questions"].items()}}


def materialize(req):
    """Labelled request -> internal record via the serving path (api.to_record), attaching int labels and src."""
    rec, meta = to_record(SystemOneRequest.model_validate(api_request(req)))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        y = src_q["label"]
        if m["type"] == "noul" and not isinstance(y, bool):
            raise ValueError(f"{qid}: noul label must be a boolean")
        if m["type"] == "score" and (type(y) is not int or not 0 <= y < len(m["keys"])):
            raise ValueError(f"{qid}: score label must be a valid integer level")
        q["label"] = m["keys"].index(y) if m["type"] == "choice" else int(y)   # noul labels are bools, score labels level indices
        q["src"] = src_q["src"]; q["qtype"] = m["type"]; q["qid"] = qid; q["keys"] = m["keys"]
        if src_q.get("target") is not None:
            # soft target keyed by option name (choice), "false"/"true" (noul) or level index as a string (score); options the
            # target does not name get 0, then the vector is normalised. Used for unknowable records (uniform over the options).
            t = [float(src_q["target"].get(k, 0.0)) for k in q["keys"]]
            if any(not math.isfinite(x) or x < 0 for x in t) or not math.isfinite(sum(t)) or sum(t) <= 0: raise ValueError(f"target for {qid} puts no mass on any option")
            q["target"] = [x / sum(t) for x in t]
    return rec

