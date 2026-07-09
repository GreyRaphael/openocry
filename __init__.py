from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

import inspect
import importlib.util
import builtins

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# Save original import function
_original_import = builtins.__import__

# Detect if running under static analysis, testing, or doc generation
_ACTIVE_TOOLS = {'pytest', 'pylint', 'mypy', 'sphinx', 'unittest', 'docutils', 'pip'}
_IS_TOOL_ACTIVE = any(
    tool in sys.modules or any(tool in arg.lower() for arg in sys.argv)
    for tool in _ACTIVE_TOOLS
)

if not _IS_TOOL_ACTIVE:
    def _openocr_custom_import(name, globals=None, locals=None, fromlist=None, level=0):
        top_level = name.split('.')[0]
        if top_level in {'tools', 'opendet', 'openrec', 'configs'}:
            # Traverse call stack to verify if the import originates from openocr package
            package_root = __dir__.lower()
            frame = inspect.currentframe()
            try:
                while frame:
                    file_path = frame.f_globals.get('__file__', '')
                    if file_path:
                        abs_file_path = os.path.abspath(file_path).lower()
                        # Redirect import to openocr package namespace if caller is inside package root
                        if abs_file_path.startswith(package_root):
                            return _original_import(f"openocr.{name}", globals, locals, fromlist, level)
                    frame = frame.f_back
            finally:
                del frame
                
        return _original_import(name, globals, locals, fromlist, level)

    # Override builtins.__import__ to redirect absolute imports of subfolders
    builtins.__import__ = _openocr_custom_import

# from .tools.infer_e2e import OpenOCRE2E, OpenDetector, OpenRecognizer
# from .tools.infer_unirec_onnx import UniRecONNX
# from .tools.infer_doc_onnx import OpenDocONNX
from .openocr import OpenOCR, main

__version__ = '0.1.17'

# Lazy import for demo interfaces to avoid initialization on import
def launch_openocr_demo(*args, **kwargs):
    """Launch Gradio OCR demo"""
    from .demo_gradio import launch_demo
    return launch_demo(*args, **kwargs)

def launch_unirec_demo(*args, **kwargs):
    """Launch UniRec demo"""
    from .demo_unirec import launch_demo
    return launch_demo(*args, **kwargs)

def launch_opendoc_demo(*args, **kwargs):
    """Launch OpenDoc demo"""
    from .demo_opendoc import launch_demo
    return launch_demo(*args, **kwargs)

__all__ = [
    'OpenOCRE2E',
    'OpenDetector',
    'OpenRecognizer',
    'UniRecONNX',
    'OpenDocONNX',
    'OpenOCR',
    'main',
    'launch_openocr_demo',
    'launch_unirec_demo',
    'launch_opendoc_demo',
]
