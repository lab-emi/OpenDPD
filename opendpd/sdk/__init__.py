"""Local experiment SDK used by the MATLAB toolbox.

Projects share Studio's service, queue and workspace. Importing this module
does not import PyTorch or start a service. See ``Matlab/toolbox/README.md``.
"""

from .client import Job, Project, SDKError, open_project
from .diagnostics import doctor

API_VERSION = 1

__all__ = ["API_VERSION", "Job", "Project", "SDKError", "doctor", "open_project"]
