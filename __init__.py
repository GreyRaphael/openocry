from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

import inspect
import importlib.util

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

class OpenOCRImportRedirector:
    """Redirects absolute imports of internal folders (tools, opendet, openrec)
    to openocr package namespace to avoid polluting the Python global namespace.
    Only intercepts imports initiated within the openocr package.
    """
    def find_spec(self, fullname, path, target=None):
        top_level = fullname.split('.')[0]
        if top_level in {'tools', 'opendet', 'openrec', 'configs'}:
            # Only redirect if openocr package itself is available
            try:
                if importlib.util.find_spec("openocr") is None:
                    return None
            except Exception:
                return None

            frame = inspect.currentframe()
            try:
                while frame:
                    file_path = frame.f_globals.get('__file__', '')
                    if file_path and ('openocr' in file_path.lower() or 'openocry' in file_path.lower()):
                        return importlib.util.find_spec(f"openocr.{fullname}")
                    frame = frame.f_back
            finally:
                del frame
        return None

# Register import redirector
sys.meta_path.insert(0, OpenOCRImportRedirector())

# from .tools.infer_e2e import OpenOCRE2E, OpenDetector, OpenRecognizer
# from .tools.infer_unirec_onnx import UniRecONNX
# from .tools.infer_doc_onnx import OpenDocONNX
from .openocr import OpenOCR, main

__version__ = '0.1.8'

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
