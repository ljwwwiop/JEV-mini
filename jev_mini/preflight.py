"""Read-only checks: never installs or changes dependencies."""
import importlib.util
import importlib.metadata
import sys


def check_environment(base):
    import json
    from pathlib import Path
    problems = []
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'pydantic', 'yaml'):
        if importlib.util.find_spec(name) is None:
            problems.append(f'missing module: {name}')
    if importlib.util.find_spec('transformers'):
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        config = Path(base) / 'config.json'
        if config.is_file():
            kind = json.loads(config.read_text())['model_type']
            if kind not in CONFIG_MAPPING:
                version = importlib.metadata.version('transformers')
                problems.append(f'transformers {version} does not support model_type={kind}')
    if problems:
        raise RuntimeError('Training environment is not ready: ' + sys.executable + '\n- '
                           + '\n- '.join(problems)
                           + '\nNo packages were installed or modified. See pyproject.toml for the declared environment.')
