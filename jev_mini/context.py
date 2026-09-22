"""Shared, explicit context admission for training and evaluation."""
from dataclasses import dataclass


@dataclass
class Context:
    max_state: int = 384
    max_branch: int = 1024  # state + one question
    max_packed: int = 2048
    overlength: str = 'error'

    def __post_init__(self):
        if min(self.max_state, self.max_branch, self.max_packed) < 1:
            raise ValueError('context limits must be positive')
        if self.overlength not in ('error', 'skip'):
            raise ValueError('overlength must be error or skip')

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__})

    def encode(self, tokenizer, record, encoder=None):
        if encoder is None:
            from .model import encode
            encoder = encode
        try:
            result = encoder(tokenizer, record, strict=True,
                             max_state=self.max_state, max_branch=self.max_branch)
            if len(result['ids']) > self.max_packed:
                raise ValueError(f"packed record exceeds {self.max_packed} tokens")
            return result
        except ValueError as exc:
            # Only context errors can be skipped, never malformed data.
            if self.overlength == 'skip' and str(exc).startswith(('state exceeds ', 'branch too long:', 'packed record exceeds ')):
                return None
            raise


def add_context_args(parser):
    parser.add_argument('--max-state', type=int, default=384)
    parser.add_argument('--max-branch', type=int, default=1024)
    parser.add_argument('--max-packed', type=int, default=2048)
    parser.add_argument('--overlength', choices=['error', 'skip'], default='error')
