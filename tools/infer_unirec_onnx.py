"""
ONNX inference script for UniRec model.
Standalone version without transformers dependency.

Version: Optimized v2
- Supports optimized KV cache format: [batch_size, num_heads, seq_len, head_dim]
- Compatible with merged QKV/KV projection models
- No reshape overhead during generation
"""

import json
import os
import re
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image


from tools.download.model_zoo import check_and_download_file, get_cache_dir

def download_model_files(model_dir=None):
    """Download ONNX model files from ModelScope or HuggingFace.
    Unifies calls under tools.download.model_zoo.
    """
    if model_dir is None:
        model_dir = get_cache_dir('unirec_0_1b_onnx')
    else:
        model_dir = Path(model_dir)

    enc_path = model_dir / 'unirec_encoder.onnx'
    dec_path = model_dir / 'unirec_decoder.onnx'
    map_path = model_dir / 'unirec_tokenizer_mapping.json'

    check_and_download_file('unirec_encoder', target_path=enc_path)
    check_and_download_file('unirec_decoder', target_path=dec_path)
    check_and_download_file('unirec_mapping', target_path=map_path)

    return str(enc_path), str(dec_path), str(map_path)


def check_and_download_models(encoder_path, decoder_path, mapping_path, auto_download=True):
    """Check if model files exist, download if missing."""
    enc_dir = os.path.dirname(encoder_path)
    if enc_dir and enc_dir != './unirec_0_1b_onnx':
        model_dir = enc_dir
    else:
        model_dir = None

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path) or not os.path.exists(mapping_path):
        if not auto_download:
            raise FileNotFoundError("UniRec model files not found locally.")
        return download_model_files(model_dir)
    return encoder_path, decoder_path, mapping_path



class SimpleImageProcessor:
    """Standalone image processor without transformers dependency."""

    def __init__(
            self,
            max_side=(960, 1408),  # (width, height)
            divided_factor=(64, 64),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
    ):
        self.max_side = max_side
        self.divided_factor = divided_factor
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def _calculate_target_size(self, original_width, original_height):
        """Calculate target size with aspect ratio preservation."""
        max_width, max_height = self.max_side
        aspect_ratio = original_width / original_height

        if original_width > max_width or original_height > max_height:
            if (max_width / max_height) >= aspect_ratio:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
        else:
            new_width, new_height = original_width, original_height

        # Apply divided factor
        div_w, div_h = self.divided_factor
        final_width = max(int(new_width // div_w * div_w), 64)
        final_height = max(int(new_height // div_h * div_h), 64)

        return (final_width, final_height)

    def __call__(self, image):
        """
        Process image for model input using OpenCV resize.

        Args:
            image: PIL Image or numpy array

        Returns:
            dict with 'pixel_values' as numpy array [1, 3, H, W]
        """
        import cv2
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, np.ndarray):
            image_np = image.copy()
        else:
            raise ValueError('Input must be PIL Image or numpy array')

        original_height, original_width = image_np.shape[:2]

        # Calculate target size with aspect ratio preservation
        target_width, target_height = self._calculate_target_size(original_width,
                                                                 original_height)

        # OpenCV cubic resize
        resized = cv2.resize(image_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Handle different channel formats (ensure RGB)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = resized[:, :, :3]

        # Convert to float32 and normalize to [0, 1]
        image_np = resized.astype(np.float32) / 255.0

        # Normalize: (x - mean) / std
        image_np = (image_np - self.image_mean) / self.image_std

        # Transpose to [C, H, W]
        image_np = image_np.transpose(2, 0, 1)

        # Add batch dimension [1, C, H, W]
        image_np = np.expand_dims(image_np, axis=0)

        return {'pixel_values': image_np}


class SimpleTokenizer:
    """Standalone tokenizer without transformers dependency."""

    def __init__(self, mapping_file=None):
        """
        Load vocabulary from mapping file or tokenizer.json.

        Args:
            vocab_file: path to tokenizer.json (deprecated, use mapping_file)
            mapping_file: path to unirec_tokenizer_mapping.json (recommended)
        """

        if mapping_file and os.path.exists(mapping_file):
            # 使用导出的映射文件 (推荐)
            print(f'Loading tokenizer from mapping file: {mapping_file}')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)

            # 直接使用 id_to_token 映射
            self.id_to_token = {
                int(k): v
                for k, v in mapping_data['id_to_token'].items()
            }
            self.vocab_size = mapping_data['vocab_size']

            # 特殊 token
            special_tokens = mapping_data['special_tokens']
            self.bos_token_id = special_tokens['bos_token_id']
            self.eos_token_id = special_tokens['eos_token_id']
            self.pad_token_id = special_tokens['pad_token_id']

        print(f'✅ Loaded vocabulary with {self.vocab_size} tokens')

    def decode(self, token_ids, skip_special_tokens=False):
        """
        Decode token IDs to text.

        Args:
            token_ids: list of token IDs
            skip_special_tokens: whether to skip special tokens

        Returns:
            decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]

                # Skip special tokens if requested
                if skip_special_tokens and token_id in [
                        self.bos_token_id, self.eos_token_id, self.pad_token_id
                ]:
                    continue

                tokens.append(token)
            else:
                tokens.append(f'<unk_{token_id}>')

        # Join tokens
        text = ''.join(tokens)

        return text


def clean_special_tokens(text):
    """Clean special tokens from decoded text."""
    # Remove special formatting tokens
    text = text.replace('Ġ', ' ').replace('Ċ', '\n')
    text = text.replace('<|bos|>', '').replace('<|eos|>',
                                               '').replace('<|pad|>', '')

    # Apply regex rules
    rules = [
        (r'-<\|sn\|>', ''),
        (r' <\|sn\|>', ' '),
        (r'<\|sn\|>', ' '),
        (r'<\|unk\|>', ''),
        (r'<s>', ''),
        (r'</s>', ''),
        (r'\uffff', ''),
        (r'_{4,}', '___'),
        (r'\.{4,}', '...'),
    ]

    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)

    return text


class UniRecONNX:
    """ONNX-based inference for UniRec model (standalone version)."""

    def __init__(
        self,
        encoder_path=None,
        decoder_path=None,
        mapping_path=None,
        use_gpu=None,
        auto_download=True,
        intra_op_num_threads=0,
        inter_op_num_threads=0,
    ):
        """Initialize ONNX inference sessions.

        Args:
            encoder_path: Path to encoder ONNX model. If None, use default cache directory.
            decoder_path: Path to decoder ONNX model. If None, use default cache directory.
            mapping_path: Path to tokenizer mapping JSON. If None, use default cache directory.
            use_gpu: Whether to use GPU. If None, auto-detect. If True, force GPU. If False, force CPU.
            auto_download: If True, automatically download missing model files
            intra_op_num_threads: Number of threads to parallelize the execution of a single operator.
            inter_op_num_threads: Number of threads to parallelize the execution of multiple operators.
        """
        # Set default paths if not provided
        if encoder_path is None or decoder_path is None or mapping_path is None:
            cache_dir = Path.home() / '.cache' / 'openocr'
            model_path = cache_dir / 'unirec_0_1b_onnx'
            if encoder_path is None:
                encoder_path = str(model_path / 'unirec_encoder.onnx')
            if decoder_path is None:
                decoder_path = str(model_path / 'unirec_decoder.onnx')
            if mapping_path is None:
                mapping_path = str(model_path / 'unirec_tokenizer_mapping.json')

        # Check and download models if needed
        encoder_path, decoder_path, mapping_path = check_and_download_models(
            encoder_path, decoder_path, mapping_path, auto_download=auto_download
        )

        print('Loading ONNX models...')

        # Determine execution provider
        providers = self._get_execution_providers(use_gpu)
        print(f'Using execution providers: {providers}')

        # Create ONNX runtime sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads > 0:
            sess_options.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads > 0:
            sess_options.inter_op_num_threads = inter_op_num_threads
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=providers)
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)

        # Initialize processor and tokenizer
        self.processor = SimpleImageProcessor()
        self.tokenizer = SimpleTokenizer(mapping_file=mapping_path)

        # Get model info from decoder session
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        self.num_decoder_layers = None
        self.num_heads = None
        self.head_dim = None

        for inp in self.decoder_session.get_inputs():
            if 'past_key' in inp.name:
                layer_idx = int(inp.name.split('_')[-1])
                if self.num_decoder_layers is None or layer_idx + 1 > self.num_decoder_layers:
                    self.num_decoder_layers = layer_idx + 1
                # Get shape info: [batch_size, num_heads, seq_len, head_dim]
                if len(inp.shape) == 4:
                    if self.num_heads is None and isinstance(
                            inp.shape[1], int):
                        self.num_heads = inp.shape[1]
                    if self.head_dim is None and isinstance(inp.shape[3], int):
                        self.head_dim = inp.shape[3]

        # Calculate d_model
        if self.num_heads and self.head_dim:
            self.d_model = self.num_heads * self.head_dim
        else:
            self.d_model = None

        print('\n✅ Models loaded successfully!')
        print(f'   Number of decoder layers: {self.num_decoder_layers}')
        print(f'   Number of attention heads: {self.num_heads}')
        print(f'   Head dimension: {self.head_dim}')
        print(f'   Model dimension (d_model): {self.d_model}')
        print(f'   Vocabulary size: {self.tokenizer.vocab_size}')
        
        # Determine device type for IO Binding
        self.device_type = 'cuda' if 'CUDAExecutionProvider' in self.decoder_session.get_providers() and any('CUDA' in p for p in providers) else 'cpu'
        self.device_id = 0
        print(f'   IO Binding device: {self.device_type}:{self.device_id}')


    def _get_execution_providers(self, use_gpu):
        """Determine execution providers based on GPU availability and user preference.

        Args:
            use_gpu: None (auto-detect), True (force GPU), or False (force CPU)

        Returns:
            List of execution providers in priority order
        """
        available_providers = ort.get_available_providers()

        if use_gpu is False:
            # Force CPU
            print('🔧 User specified: Using CPU')
            return ['CPUExecutionProvider']

        # Check for GPU providers
        gpu_providers = []
        if 'CUDAExecutionProvider' in available_providers:
            gpu_providers.append('CUDAExecutionProvider')
        # if 'TensorrtExecutionProvider' in available_providers:
        #     gpu_providers.append('TensorrtExecutionProvider')

        if use_gpu is True:
            # Force GPU
            if gpu_providers:
                print(f'🔧 User specified: Using GPU ({gpu_providers[0]})')
                return gpu_providers + ['CPUExecutionProvider']
            else:
                print('⚠️  GPU requested but not available, falling back to CPU')
                return ['CPUExecutionProvider']

        # Auto-detect (use_gpu is None)
        if gpu_providers:
            print(f'✅ GPU detected: Using {gpu_providers[0]}')
            return gpu_providers + ['CPUExecutionProvider']
        else:
            print('ℹ️  No GPU detected, using CPU')
            return ['CPUExecutionProvider']

    def encode_image(self, image):
        """Encode image using encoder ONNX model."""
        # Preprocess image
        data_img = self.processor(image)
        pixel_values = data_img['pixel_values']

        # Run encoder
        encoder_outputs = self.encoder_session.run(
            None, {'pixel_values': pixel_values.astype(np.float32)})

        encoder_hidden_states = encoder_outputs[0]
        cross_k = encoder_outputs[1]
        cross_v = encoder_outputs[2]

        return encoder_hidden_states, cross_k, cross_v

    def decode_step(self,
                    input_id,
                    past_length,
                    cross_k,
                    cross_v,
                    past_key_values,
                    padding_idx=1):
        """Unified decoder step with ORT IO Binding."""
        # Convert inputs to OrtValues if they are numpy arrays
        if isinstance(cross_k, np.ndarray):
            cross_k_val = ort.OrtValue.from_numpy(cross_k.astype(np.float32), self.device_type, self.device_id)
        else:
            cross_k_val = cross_k

        if isinstance(cross_v, np.ndarray):
            cross_v_val = ort.OrtValue.from_numpy(cross_v.astype(np.float32), self.device_type, self.device_id)
        else:
            cross_v_val = cross_v

        # Prepare input_ids and position_ids
        input_ids = np.array([[input_id]], dtype=np.int64)
        position_ids = np.array([[padding_idx + 1 + past_length]], dtype=np.int64)

        input_ids_val = ort.OrtValue.from_numpy(input_ids, self.device_type, self.device_id)
        position_ids_val = ort.OrtValue.from_numpy(position_ids, self.device_type, self.device_id)

        # Setup IO Binding
        io_binding = self.decoder_session.io_binding()

        # Bind inputs
        io_binding.bind_ortvalue_input('input_ids', input_ids_val)
        io_binding.bind_ortvalue_input('position_ids', position_ids_val)
        io_binding.bind_ortvalue_input('cross_k', cross_k_val)
        io_binding.bind_ortvalue_input('cross_v', cross_v_val)

        # Bind past key values
        for i, (past_key, past_value) in enumerate(past_key_values):
            if isinstance(past_key, np.ndarray):
                past_key_val = ort.OrtValue.from_numpy(past_key.astype(np.float32), self.device_type, self.device_id)
            else:
                past_key_val = past_key

            if isinstance(past_value, np.ndarray):
                past_value_val = ort.OrtValue.from_numpy(past_value.astype(np.float32), self.device_type, self.device_id)
            else:
                past_value_val = past_value

            io_binding.bind_ortvalue_input(f'past_key_{i}', past_key_val)
            io_binding.bind_ortvalue_input(f'past_value_{i}', past_value_val)

        # Bind outputs (on the same device to avoid host copying)
        io_binding.bind_output('logits', self.device_type, self.device_id)
        for i in range(self.num_decoder_layers):
            io_binding.bind_output(f'present_key_{i}', self.device_type, self.device_id)
            io_binding.bind_output(f'present_value_{i}', self.device_type, self.device_id)

        # Run session
        self.decoder_session.run_with_iobinding(io_binding)

        # Retrieve outputs as OrtValues
        ort_outputs = io_binding.get_outputs()
        logits_val = ort_outputs[0]
        # We only transfer logits to host memory because it's small and needed for token argmax
        logits = logits_val.numpy()

        present_key_values = []
        for i in range(self.num_decoder_layers):
            present_key_values.append((ort_outputs[1 + i * 2], ort_outputs[1 + i * 2 + 1]))

        return logits, present_key_values

    def _pdf_to_images(self, pdf_path):
        """Convert PDF file to a list of PIL Images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Image objects (RGB format)
        """
        try:
            import fitz
        except ImportError:
            raise ImportError(
                'PyMuPDF is required for PDF support. '
                'Install with: pip install PyMuPDF'
            )

        images = []
        with fitz.open(pdf_path) as pdf:
            for pg in range(pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                # If width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
                images.append(img)
        return images

    def __call__(
        self,
        img_path=None,
        img_numpy=None,
        image=None,
        images_list=None,
        max_length=2048,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
        Unified interface for UniRec inference.

        Args:
            img_path: Path to input image or PDF file (str or Path)
            img_numpy: Input image as numpy array (BGR format)
            image: PIL Image object (RGB format)
            images_list: List of PIL Images or numpy arrays for batched inference
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Tuple of (generated_text, generated_ids) for single image input.
            List of tuples [(generated_text, generated_ids), ...] for PDF or batched input.
        """
        # Handle batched input list directly
        if images_list is not None:
            return self._infer_batch(
                images=images_list,
                max_length=max_length,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        if isinstance(image, list):
            return self._infer_batch(
                images=image,
                max_length=max_length,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        # Handle PDF input: convert to images and process each page dynamically (page-by-page) to save memory
        if img_path is not None and str(img_path).lower().endswith('.pdf'):
            print(f'Processing PDF file: {img_path}')
            try:
                import fitz
            except ImportError:
                raise ImportError(
                    'PyMuPDF is required for PDF support. '
                    'Install with: pip install PyMuPDF'
                )
            results = []
            with fitz.open(img_path) as pdf:
                total_pages = pdf.page_count
                print(f'Found {total_pages} pages in PDF')
                for page_idx in range(total_pages):
                    print(f'\n--- Processing page {page_idx + 1}/{total_pages} ---')
                    page = pdf[page_idx]
                    mat = fitz.Matrix(2, 2)
                    pm = page.get_pixmap(matrix=mat, alpha=False)
                    if pm.width > 2000 or pm.height > 2000:
                        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                    page_image = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
                    
                    result = self._infer_single_image(
                        image=page_image,
                        max_length=max_length,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                    results.append(result)
                    
                    # Release memory reference immediately
                    del page_image
                    del pm
            return results

        # Load image from path, numpy array, or use provided PIL image
        if img_path is not None:
            image = Image.open(img_path).convert('RGB')
        elif img_numpy is not None:
            # Convert BGR to RGB if needed
            if len(img_numpy.shape) == 3 and img_numpy.shape[2] == 3:
                import cv2
                img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_numpy)
        elif image is None:
            raise ValueError('Either img_path, img_numpy, image, or images_list must be provided')

        return self._infer_single_image(
            image=image,
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    def _infer_batch(self, images, max_length=2048, bos_token_id=None, eos_token_id=None, pad_token_id=None):
        """Run batch inference on a list of PIL Images or numpy arrays."""
        if not images:
            return []

        # Get token IDs
        if bos_token_id is None:
            bos_token_id = self.tokenizer.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        batch_size = len(images)
        
        # Preprocess all images
        processed_data = [self.processor(img)['pixel_values'][0] for img in images]
        
        # Pad to max height and width in the batch
        max_h = max(img.shape[1] for img in processed_data)
        max_w = max(img.shape[2] for img in processed_data)
        
        # Stack into [N, 3, max_h, max_w]
        padded_images = np.zeros((batch_size, 3, max_h, max_w), dtype=np.float32)
        for idx, img in enumerate(processed_data):
            c, h, w = img.shape
            padded_images[idx, :, :h, :w] = img

        # Encode batch
        encoder_outputs = self.encoder_session.run(None, {'pixel_values': padded_images})
        encoder_hidden_states, cross_k, cross_v = encoder_outputs[0], encoder_outputs[1], encoder_outputs[2]

        # Convert cross_k/cross_v to OrtValues for IO Binding
        cross_k_ort = ort.OrtValue.from_numpy(cross_k.astype(np.float32), self.device_type, self.device_id)
        cross_v_ort = ort.OrtValue.from_numpy(cross_v.astype(np.float32), self.device_type, self.device_id)

        # Initialize generation: list of generated ids for each sequence in the batch
        generated_ids = [[bos_token_id] for _ in range(batch_size)]
        finished = [False] * batch_size

        # Initialize empty past_key_values OrtValues for first step
        past_key_values = []
        for _ in range(self.num_decoder_layers):
            empty_key = np.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=np.float32)
            empty_value = np.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=np.float32)
            
            empty_key_ort = ort.OrtValue.from_numpy(empty_key, self.device_type, self.device_id)
            empty_value_ort = ort.OrtValue.from_numpy(empty_value, self.device_type, self.device_id)
            past_key_values.append((empty_key_ort, empty_value_ort))

        # Generation loop
        for step in range(max_length - 1):
            # Current token for each sequence
            current_tokens = np.array([[g[-1]] for g in generated_ids], dtype=np.int64) # [N, 1]
            position_ids = np.array([[pad_token_id + 1 + step] for _ in range(batch_size)], dtype=np.int64) # [N, 1]

            # IO Binding
            io_binding = self.decoder_session.io_binding()
            
            input_ids_ort = ort.OrtValue.from_numpy(current_tokens, self.device_type, self.device_id)
            position_ids_ort = ort.OrtValue.from_numpy(position_ids, self.device_type, self.device_id)
            
            io_binding.bind_ortvalue_input('input_ids', input_ids_ort)
            io_binding.bind_ortvalue_input('position_ids', position_ids_ort)
            io_binding.bind_ortvalue_input('cross_k', cross_k_ort)
            io_binding.bind_ortvalue_input('cross_v', cross_v_ort)

            for i, (pk, pv) in enumerate(past_key_values):
                io_binding.bind_ortvalue_input(f'past_key_{i}', pk)
                io_binding.bind_ortvalue_input(f'past_value_{i}', pv)

            # Bind outputs
            io_binding.bind_output('logits', self.device_type, self.device_id)
            for i in range(self.num_decoder_layers):
                io_binding.bind_output(f'present_key_{i}', self.device_type, self.device_id)
                io_binding.bind_output(f'present_value_{i}', self.device_type, self.device_id)

            # Run
            self.decoder_session.run_with_iobinding(io_binding)

            # Outputs
            ort_outputs = io_binding.get_outputs()
            logits = ort_outputs[0].numpy() # [N, 1, vocab_size]

            # Get next token for each sequence
            for idx in range(batch_size):
                if finished[idx]:
                    generated_ids[idx].append(pad_token_id)
                    continue
                
                next_token_id = int(np.argmax(logits[idx, -1, :]))
                generated_ids[idx].append(next_token_id)
                
                if next_token_id == eos_token_id:
                    finished[idx] = True

            if all(finished):
                break

            # Update past_key_values for next step
            past_key_values = []
            for i in range(self.num_decoder_layers):
                past_key_values.append((ort_outputs[1 + i * 2], ort_outputs[1 + i * 2 + 1]))

        # Decode tokens
        results = []
        for g_ids in generated_ids:
            # truncate padding tokens or eos tokens if any
            clean_g_ids = []
            for token in g_ids:
                clean_g_ids.append(token)
                if token == eos_token_id:
                    break
            decoded = self.tokenizer.decode(clean_g_ids, skip_special_tokens=False)
            results.append((clean_special_tokens(decoded), clean_g_ids))

        return results

    def _infer_single_image(
        self,
        image,
        max_length=2048,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """Run inference on a single PIL Image.

        Args:
            image: PIL Image object (RGB format)
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Tuple of (generated_text, generated_ids)
        """
        # Get token IDs
        if bos_token_id is None:
            bos_token_id = self.tokenizer.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        # Encode image
        print('Encoding image...')
        t_start = time.time()
        encoder_hidden_states, cross_k, cross_v = self.encode_image(image)
        print(f'Encoding time: {time.time() - t_start:.2f} seconds')
        print(f'  cross_k shape: {cross_k.shape}')
        print(f'  cross_v shape: {cross_v.shape}')

        # Initialize generation
        print('Generating text...')
        generated_ids = [bos_token_id]

        # Initialize empty past_key_values for first step
        # Shape: [batch_size, num_heads, 0, head_dim]
        batch_size = encoder_hidden_states.shape[0]
        past_key_values = []
        for _ in range(self.num_decoder_layers):
            empty_key = np.zeros(
                (batch_size, self.num_heads, 0, self.head_dim),
                dtype=np.float32)
            empty_value = np.zeros(
                (batch_size, self.num_heads, 0, self.head_dim),
                dtype=np.float32)
            past_key_values.append((empty_key, empty_value))

        # Generation loop
        t_start = time.time()
        for step in range(max_length - 1):
            # Current token to decode
            current_token = generated_ids[-1]

            # past_length is the sequence length in cache
            past_length = step

            # Decode step
            logits, past_key_values = self.decode_step(
                current_token,
                past_length,
                cross_k,
                cross_v,
                past_key_values,
                padding_idx=pad_token_id)

            # Get next token
            next_token_id = int(np.argmax(logits[0, -1, :]))
            generated_ids.append(next_token_id)

            # Check for EOS
            if next_token_id == eos_token_id:
                break

            # Progress indicator
            if (step + 1) % 50 == 0:
                print(f'  Generated {step + 1} tokens...')

        t_end = time.time()
        print(f'✅ Generation complete! Total tokens: {len(generated_ids)}')
        print(f'  Time taken: {t_end - t_start:.2f} seconds')
        print(
            f'  Tokens per second: {len(generated_ids) / (t_end - t_start):.2f}'
        )

        # Decode tokens
        generated_text = self.tokenizer.decode(generated_ids,
                                               skip_special_tokens=False)
        cleaned_text = clean_special_tokens(generated_text)

        return cleaned_text, generated_ids


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='UniRec ONNX Inference (Standalone)')
    parser.add_argument('--image',
                        type=str,
                        required=True,
                        help='Path to input image')
    parser.add_argument('--encoder-model',
                        type=str,
                        default=None,
                        help='Path to encoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx)')
    parser.add_argument('--decoder-model',
                        type=str,
                        default=None,
                        help='Path to decoder ONNX model (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx)')
    parser.add_argument(
        '--mapping',
        type=str,
        default=None,
        help='Path to tokenizer mapping JSON (default: ~/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json)')
    parser.add_argument('--max-length',
                        type=int,
                        default=2048,
                        help='Maximum generation length')
    parser.add_argument('--use-gpu',
                        type=str,
                        default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Use GPU for inference (auto: auto-detect, true: force GPU, false: force CPU)')
    parser.add_argument('--no-auto-download',
                        action='store_true',
                        help='Disable automatic model download')
    args = parser.parse_args()

    # Parse use_gpu argument
    if args.use_gpu == 'auto':
        use_gpu = None
    elif args.use_gpu == 'true':
        use_gpu = True
    else:
        use_gpu = False

    # Load image
    print(f'Loading image: {args.image}')
    image = Image.open(args.image).convert('RGB')

    # Initialize inference
    inference = UniRecONNX(
        encoder_path=args.encoder_model,
        decoder_path=args.decoder_model,
        mapping_path=args.mapping,
        use_gpu=use_gpu,
        auto_download=not args.no_auto_download,
    )

    # Generate
    result_text, generated_ids = inference(
        image=image,
        max_length=args.max_length,
    )

    # Print result
    print('\n' + '=' * 80)
    print('RESULT:')
    print('=' * 80)
    print(result_text)
    print('=' * 80)
    print(f'\nGenerated {len(generated_ids)} tokens')


if __name__ == '__main__':
    main()
