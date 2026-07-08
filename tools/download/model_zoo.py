import os
import shutil
from pathlib import Path
from tools.utils.logging import get_logger

logger = get_logger(name='model_zoo')

# Centralized Model Registry
MODEL_REGISTRY = {
    # Text Detection
    'det_repvit': {
        'clean_name': 'openocr_det_repvit_ch.pth',
        'ms_repo': 'topdktu/OpenOCR',
        'hf_repo': 'topdu/OpenOCR',
        'default_url': 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_repvit_ch.pth'
    },
    'det_onnx': {
        'clean_name': 'openocr_det_model.onnx',
        'ms_repo': 'topdktu/OpenOCR',
        'hf_repo': 'topdu/OpenOCR',
        'default_url': 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_model.onnx'
    },
    # Text Recognition
    'rec_repsvtr': {
        'clean_name': 'openocr_repsvtr_ch.pth',
        'ms_repo': 'topdktu/OpenOCR',
        'hf_repo': 'topdu/OpenOCR',
        'default_url': 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_repsvtr_ch.pth'
    },
    'rec_svtrv2_server': {
        'clean_name': 'openocr_svtrv2_ch.pth',
        'ms_repo': 'topdktu/OpenOCR',
        'hf_repo': 'topdu/OpenOCR',
        'default_url': 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth'
    },
    'rec_onnx': {
        'clean_name': 'openocr_rec_model.onnx',
        'ms_repo': 'topdktu/OpenOCR',
        'hf_repo': 'topdu/OpenOCR',
        'default_url': 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_rec_model.onnx'
    },
    # UniRec (VLM)
    'unirec_encoder': {
        'clean_name': 'unirec_encoder.onnx',
        'ms_repo': 'topdktu/unirec_0_1b_onnx',
        'hf_repo': 'topdu/unirec_0_1b_onnx',
        'default_url': ''
    },
    'unirec_decoder': {
        'clean_name': 'unirec_decoder.onnx',
        'ms_repo': 'topdktu/unirec_0_1b_onnx',
        'hf_repo': 'topdu/unirec_0_1b_onnx',
        'default_url': ''
    },
    'unirec_mapping': {
        'clean_name': 'unirec_tokenizer_mapping.json',
        'ms_repo': 'topdktu/unirec_0_1b_onnx',
        'hf_repo': 'topdu/unirec_0_1b_onnx',
        'default_url': ''
    },
    # Layout Detection
    'layout_onnx': {
        'clean_name': 'PP-DoclayoutV2.onnx',
        'ms_repo': 'topdktu/PP_DoclayoutV2_onnx',
        'hf_repo': 'topdu/PP_DoclayoutV2_onnx',
        'default_url': ''
    }
}


def get_cache_dir(subdir=''):
    """Get the localized cache directory for OpenOCR models."""
    cache_dir = Path.home() / '.cache' / 'openocr'
    if subdir:
        cache_dir = cache_dir / subdir
    return cache_dir


def check_and_download_file(model_key, target_path=None, auto_download=True):
    """
    Check if a model exists locally. If not, download from ModelScope or HuggingFace.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: '{model_key}'")

    meta = MODEL_REGISTRY[model_key]
    clean_name = meta['clean_name']

    # Determine final target path
    if target_path is None:
        target_path = get_cache_dir() / clean_name
    else:
        target_path = Path(target_path)

    # Return if it already exists
    if target_path.exists():
        logger.info(f"Model already exists at: {target_path}")
        return str(target_path)

    if not auto_download:
        raise FileNotFoundError(
            f"Model '{clean_name}' not found at {target_path}. "
            f"Please download manually from ModelScope ({meta['ms_repo']}) or HuggingFace ({meta['hf_repo']})"
        )

    # Create parent folder
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try ModelScope first
    logger.info(f"Model not found. Attempting download of '{clean_name}' from ModelScope...")
    try:
        from modelscope.hub.file_download import model_file_download
        downloaded_path = model_file_download(
            model_id=meta['ms_repo'],
            file_path=clean_name,
            cache_dir=str(target_path.parent.parent)
        )
        if downloaded_path != str(target_path):
            shutil.copy2(downloaded_path, str(target_path))
        logger.info(f"Model successfully downloaded from ModelScope: {target_path}")
        return str(target_path)
    except Exception as e:
        logger.warning(f"Failed to download from ModelScope: {e}. Falling back to HuggingFace...")

    # 2. Try HuggingFace hub
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=meta['hf_repo'],
            filename=clean_name,
            cache_dir=str(target_path.parent.parent),
            local_dir=str(target_path.parent) if 'unirec' in model_key or 'layout' in model_key else None,
            local_dir_use_symlinks=False
        )
        if downloaded_path != str(target_path):
            shutil.copy2(downloaded_path, str(target_path))
        logger.info(f"Model successfully downloaded from HuggingFace: {target_path}")
        return str(target_path)
    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}")
        raise RuntimeError(
            f"Failed to download '{clean_name}' from both ModelScope and HuggingFace. "
            f"Please manually download and save to: {target_path}"
        ) from e
