"""Original Kev objectives; train.py uses CE/soft targets and optional ordinal loss.

Anchor and permutation KL are retained for reading/extension, not enabled by the mini CLI.
"""
import math
import torch
import torch.nn.functional as F

def permuted_copy(rec, rng):
    """Re-shuffle options of every Choice question with K>=3; return (record, perms) with perms[q] = new->old index or None."""
    out, perms = {"state": rec["state"], "questions": []}, []
    for q in rec["questions"]:
        if q["qtype"] == "choice" and len(q["options"]) >= 3:
            perm = list(range(len(q["options"]))); rng.shuffle(perm)
            out["questions"].append({**q, "options": [q["options"][j] for j in perm], "label": perm.index(q["label"])}); perms.append(perm)
        else:
            out["questions"].append(q); perms.append(None)
    return out, perms


def question_loss(z, q, dev, ord_w, label_smoothing=0.0, brier_w=0.0, focal_gamma=0.0):
    """Cross-entropy (or cross-entropy against a soft target when the question carries one), optionally plus the
    normalized ranked probability score for ordered levels."""
    options = (label_smoothing, brier_w, focal_gamma)
    if not all(math.isfinite(v) and v >= 0 for v in options) or label_smoothing > 1 or sum(v > 0 for v in options) > 1:
        raise ValueError("choose at most one finite, nonnegative loss modifier; smoothing must be <= 1")
    if q.get("target") is not None:
        t = torch.tensor(q["target"], device=dev, dtype=z.dtype)
        return -(t * F.log_softmax(z, -1)).sum()
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y, label_smoothing=label_smoothing)
    if brier_w:
        target = F.one_hot(y[0], len(z)).to(z.dtype)
        loss = loss + brier_w * (F.softmax(z, -1) - target).square().sum()
    if focal_gamma:
        loss = (1 - torch.exp(-loss)).pow(focal_gamma) * loss
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def anchor_loss(z, q, target, dev):
    """KL(teacher || student) for one question, teacher = frozen base zero-shot distribution keyed by option key.
    Skips (returns None) when the current option set is not exactly the teacher's (e.g. a none-option was inserted)."""
    if target is None or set(target) != set(q["keys"]): return None
    t = torch.tensor([target[k] for k in q["keys"]], device=dev, dtype=torch.float32).clamp_min(1e-6); t = t / t.sum()
    return F.kl_div(F.log_softmax(z, -1), t, reduction="sum")


def permutation_kl(z1, z2, perm, dev):
    """Symmetric KL between one question's predictions under two option orders; perm maps the second order's positions
    back to the first (perms from permuted_copy)."""
    lp1 = F.log_softmax(z1, -1); lp2 = F.log_softmax(z2, -1)[torch.tensor([perm.index(j) for j in range(len(perm))], device=dev)]
    return 0.5 * (F.kl_div(lp2, lp1, log_target=True, reduction="sum") + F.kl_div(lp1, lp2, log_target=True, reduction="sum"))

