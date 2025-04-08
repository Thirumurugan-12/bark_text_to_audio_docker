import psutil
import logging
from typing import Dict, Optional, Callable, Any, Literal
from collections import OrderedDict
from threading import Lock
import torch
from transformers import BertTokenizer
from encodec import EncodecModel

from core.memory.common import get_cached_or_download_model_from_hf, env
from core.model.bark import GPTConfig, FineGPTConfig, GPT, FineGPT
from core.memory.models import *

# Configure logging for this module
logger = logging.getLogger(__name__)


def clear_cuda_cache() -> None:
    """
    Clear the CUDA memory cache if GPU is available.

    Raises:
        RuntimeError: If CUDA operations fail unexpectedly.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared successfully")
        except RuntimeError as e:
            logger.error(f"Failed to clear CUDA cache: {str(e)}")
            raise RuntimeError(f"CUDA cache clear failed: {str(e)}")


class ModelManager:
    """
    Manager class for loading, caching, and unloading PyTorch models with memory management.

    Prioritizes GPU memory when available, with an optional `offload_to_cpu` flag to use CPU RAM instead.
    Uses an LRU (Least Recently Used) cache to keep only the most recently used models in memory.
    Automatically unloads models when memory usage (GPU or CPU, depending on config) exceeds a threshold
    or the maximum number of cached models is reached.
    """

    def __init__(self, max_models: int = 10, offload_to_cpu: bool = False):
        """
        Initialize the model manager.

        Args:
            max_models (int): Maximum number of models to keep in memory before unloading (default: 5)
            offload_to_cpu (bool): If True, use CPU RAM instead of GPU memory (default: False)
        """
        self._models: OrderedDict = OrderedDict()  # LRU cache for loaded models
        self._lock = Lock()  # Thread lock for safe concurrent access
        self._max_models = max_models  # Max number of models to cache
        # Whether to offload models to CPU instead of GPU
        self._offload_to_cpu = offload_to_cpu
        self._device = torch.device(env.DEVICE)  # Device to load models onto
        logger.info(f"Model manager initialized with device: {self._device}")

    def _check_memory(self) -> bool:
        """
        Check if current memory usage is below the threshold, focusing on GPU unless offloaded to CPU.

        Returns:
            bool: True if memory usage is safe, False if it exceeds the threshold
        """
        if self._offload_to_cpu or not torch.cuda.is_available():
            # Check CPU memory usage
            mem = psutil.virtual_memory()  # System memory stats
            total_mem_used = mem.used / 1e9  # CPU memory used in GB
            total_mem_available = mem.total / 1e9  # Total CPU memory in GB
        else:
            # Check GPU memory usage
            total_mem_used = (
                torch.cuda.memory_allocated() / 1e9
            )  # GPU memory used in GB
            total_mem_available = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )  # Total GPU memory in GB

        usage_ratio = total_mem_used / total_mem_available
        logger.debug(
            f"Memory usage on {self._device}: {usage_ratio:.2%} (threshold: {MEMORY_THRESHOLD})"
        )
        return usage_ratio < MEMORY_THRESHOLD

    def _unload_lru_model(self):
        """Unload the least recently used model to free memory."""
        with self._lock:
            if self._models:
                # Remove oldest entry
                model_info, model_instance = self._models.popitem(last=False)
                logger.info(
                    f"Unloading model {model_info} from {self._device} to free memory"
                )
                # Move model to CPU before deletion to ensure GPU memory is freed
                if not self._offload_to_cpu and torch.cuda.is_available():
                    model_instance.model = model_instance.model.cpu()
                del model_instance  # Explicitly delete reference
                logger.debug(f"Memory freed from {self._device}")

    def get_model(self, model_info: ModelInfo) -> Model:
        """
        Retrieve or load a model, managing memory constraints on the chosen device (GPU or CPU).

        Args:
            model_info (ModelInfo): Metadata for the model to load

        Returns:
            Model: The loaded model instance with config and preprocessor

        Raises:
            ValueError: If model_info is invalid
        """
        assert isinstance(
            model_info, ModelInfo
        ), f"invalid model_info type {type(model_info)}"
        with self._lock:
            # If model is already loaded, move it to the end (most recently used) and return it
            if model_info in self._models:
                self._models.move_to_end(model_info)
                return self._models[model_info]

            # Ensure memory is available by unloading models if necessary
            while not self._check_memory() or len(self._models) >= self._max_models:
                self._unload_lru_model()

            if model_info.load_model is not None:
                model = model_info.load_model(model_info, torch.device(env.DEVICE))
            elif model_info.checkpoint_name is not None:
                model = load_transformers_model(model_info, self._device)
            elif model_info.repo_id is not None and model_info.file_name is not None:
                model_file_path = get_cached_or_download_model_from_hf(
                    repo_id=model_info.repo_id, file_name=model_info.file_name
                )
                model = load_model_from_file(model_info, model_file_path, self._device)
            else:
                raise ValueError(
                    "Invalid model info: must provide checkpoint_name or repo_id/file_name"
                )

            # Cache the loaded model
            self._models[model_info] = model
            clear_cuda_cache()
            logger.info(f"Loaded and cached model {model_info} on {self._device}")
            return model

    def unload_model(self, model_info: ModelInfo):
        """
        Manually unload a specific model from memory.

        Args:
            model_info (ModelInfo): Metadata of the model to unload
        """
        with self._lock:
            if model_info in self._models:
                model_instance = self._models[model_info]
                # Move model to CPU before deletion if on GPU
                if not self._offload_to_cpu and torch.cuda.is_available():
                    model_instance.model = model_instance.model.cpu()
                del self._models[model_info]
                logger.info(f"Manually unloaded model {model_info} from {self._device}")


def load_model_from_file(
    model_info: ModelInfo, model_file_path: str, device: torch.device
) -> Model:
    """
    Load a model from a file (e.g., custom weights from Hugging Face).

    Args:
        model_info (ModelInfo): Metadata for the model
        model_file_path (str): Path to the model weights file
        device (torch.device): Device to load the model onto (CPU or GPU)

    Returns:
        Model: Loaded model instance
    """
    if model_info.repo_id == "suno/bark":
        return load_bark_model(model_info, model_file_path, device)
    if model_info.model_type == "custom_hubert_tokenizer":
        return load_custom_hubert_tokenizer(model_info, model_file_path, device)
    raise ValueError(f"Unknown how to load model {model_info}")


# temporary turnoff this hubert
def load_custom_hubert_tokenizer(
    model_info: ModelInfo, model_file_path: str, device: torch.device
) -> Model:
    # Automatically uses the right layers
    # tokenizer = HuBERTForBarkSemantic.load_from_checkpoint(
    #     model_file_path, torch.device(env.DEVICE)
    # ).to(device)

    # return Model(model=tokenizer)
    return Model(model=None)


def load_transformers_model(model_info: ModelInfo, device: torch.device) -> Model:
    """
    Load a model using Hugging Face's transformers library.

    Args:
        model_info (ModelInfo): Metadata for the model
        device (torch.device): Device to load the model onto (CPU or GPU)

    Returns:
        Model: Loaded model instance
    """
    if model_info.checkpoint_name == "facebook/encodec_24khz":
        model = EncodecModel.encodec_model_24khz()
        model.encode()
        model = model.to(device)
        return Model(model)
    raise NotImplementedError("Only Encodec 24k supported for now")


def load_bark_model(
    model_info: ModelInfo, model_file_path: str, device: torch.device
) -> Model:
    """
    Load a Bark model from a file.

    Args:
        model_info (ModelInfo): Metadata for the Bark model
        model_file_path (str): Path to the model weights file
        device (torch.device): Device to load the model onto (CPU or GPU)

    Returns:
        Model: Loaded Bark model instance with config and optional tokenizer
    """
    # Load checkpoint directly to the specified device
    # weights_only = False only for trusted source
    checkpoint = torch.load(model_file_path, map_location=device, weights_only=False)
    ConfigClass, ModelClass = (
        (GPTConfig, GPT)
        if model_info.model_type in ["text", "coarse"]
        else (FineGPTConfig, FineGPT)
    )

    model_args = preprocess_model_args(checkpoint["model_args"])

    conf = ConfigClass(**model_args)
    model = ModelClass(conf)
    state_dict = _update_bark_state_dict(model, checkpoint["model"])
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    logger.info(f"Loaded Bark model: {model_info} on {device}")

    # Add tokenizer for text models (tokenizer stays on CPU as it doesn't require GPU)
    preprocessor = (
        BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        if model_info.model_type == "text"
        else None
    )
    return Model(model, conf, preprocessor)


def preprocess_model_args(model_args: dict) -> dict:
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    return model_args


def _update_bark_state_dict(model: GPT, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the state dictionary by removing unwanted prefixes (specific to Bark models).

    Args:
        model (GPT): The model instance to align the state dict with
        state_dict (Dict[str, Any]): The loaded state dictionary

    Returns:
        Dict[str, Any]: Updated state dictionary
    """
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    return state_dict


# Instantiate the global model manager with default GPU priority
model_manager = ModelManager(offload_to_cpu=False if env.USE_GPU else True)
