"""Read flat YAML settings through the same validation as command-line flags."""
import argparse
import sys
from pathlib import Path
import yaml


def parse_config(parser):
    parser.add_argument('--config', help='YAML configuration path')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    known, _ = probe.parse_known_args()
    tokens = []
    if known.config:
        values = yaml.safe_load(Path(known.config).read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            parser.error('config must be a YAML mapping')
        actions = {a.dest: a for a in parser._actions if a.dest not in ('help', 'config')}
        for key, value in values.items():
            if key not in actions:
                parser.error(f'unknown config key: {key}')
            action = actions[key]
            if value is None:
                continue
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f'{key} must be a YAML boolean')
                if value:
                    tokens.append(action.option_strings[0])
            else:
                if isinstance(value, (list, dict, bool)):
                    parser.error(f'{key} must be a scalar value')
                tokens.extend([action.option_strings[0], str(value)])
    return parser.parse_args(tokens + sys.argv[1:])
