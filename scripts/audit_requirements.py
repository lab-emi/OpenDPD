"""Emit the current environment for advisory lookup, without installing anything.

PyTorch's official +cpu/+cuXXX build suffix is not a separate PyPI release;
audit its upstream release instead. Application source is reviewed separately.
"""
import importlib.metadata
import re

seen = set()
for distribution in importlib.metadata.distributions():
    name, version = distribution.metadata['Name'], distribution.version
    normalized = re.sub(r'[-_.]+', '-', name).lower()
    if normalized == 'opendpd' or normalized in seen:
        continue
    seen.add(normalized)  # Match sys.path precedence when using system site packages.
    if normalized in {'torch', 'torchvision', 'torchaudio'} and re.search(r'\+(cpu|cu[0-9]+)$', version):
        version = version.split('+')[0]
    print(f'{name}=={version}')
