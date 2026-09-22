"""Typed requests and the shared training/inference text format, extracted from Kev api.py."""
import json
from typing import Any, Literal, Union
from pydantic import BaseModel, Field, model_validator
JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255
class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS: raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]
class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "kev-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list[str]:
    """The keys a question's probabilities are reported under, in option order: the criteria names (choice),
    ["false", "true"] (noul), the level indices as strings (score). Labels, targets and anchors use the same keys."""
    if qtype == "choice": return list(criteria)
    if qtype == "noul": return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: SystemOneRequest):
    """-> internal record for encode(), plus per-question metadata ({"id", "type", "keys", "legend" for score}) to map
    probabilities back."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            opts = [render(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q.instructions), "options": opts, "label": 0}); meta.append(m)
    return {"state": render(req.state), "questions": qs}, meta

