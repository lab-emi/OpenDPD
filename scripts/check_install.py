"""Post-install smoke check; no installer hooks or environment mutation."""
import importlib.metadata
import sys
import torch

for package in ('opendpd','torch','fastapi','uvicorn','psutil','python-multipart','pywebview'):
    print(f'{package}: {importlib.metadata.version(package)}')
if sys.platform.startswith('linux'):
    for package in ('PyQt6','PyQt6-WebEngine'):
        print(f'{package}: {importlib.metadata.version(package)}')
elif sys.platform=='darwin':
    print('WKWebView:',importlib.metadata.version('pyobjc-framework-WebKit'))
elif sys.platform=='win32':
    print('pythonnet:',importlib.metadata.version('pythonnet'))
devices=['cpu']
if torch.cuda.is_available():
    devices.append('cuda')
if torch.backends.mps.is_available():
    devices.append('mps')
for device in devices:
    value=torch.tensor([2.,3.],device=device).square().sum().cpu().item()
    assert value==13.,f'{device} tensor smoke failed'
    print(f'{device}: tensor arithmetic passed')
