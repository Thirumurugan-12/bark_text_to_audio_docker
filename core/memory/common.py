import os
import logging
from dotenv import load_dotenv
from enum import Enum
from typing import ClassVar, Dict, Any
import torch
from huggingface_hub import hf_hub_download

# Configure logging with a default level (will be updated by EnvVars)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Enumeration of valid logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def grab_best_device(use_gpu: bool, enable_mps: bool) -> str:
    """
    Determine the best available device for PyTorch operations.

    Args:
        use_gpu (bool): Whether to prioritize GPU/MPS over CPU.
        enable_mps (bool): Whether to allow MPS (Metal Performance Shaders) on Apple Silicon.

    Returns:
        str: Device identifier ("cuda", "mps", or "cpu").
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.debug("Selected CUDA device (GPU available)")
    elif use_gpu and enable_mps and torch.backends.mps.is_available():
        device = "mps"
        logger.debug("Selected MPS device (Apple Silicon GPU available)")
    else:
        device = "cpu"
        logger.debug("Selected CPU device (no GPU/MPS available or disabled)")
    return device


class EnvVars:
    """
    Class to manage and expose environment variables with type safety and runtime configurability.

    Loads variables from a .env file or system environment, applies defaults if not found, and allows updates
    at runtime. Variables are stored as instance attributes rather than polluting the global namespace.
    """

    # Default values for environment variables
    _DEFAULTS: ClassVar[Dict[str, Any]] = {
        "GLOBAL_ENABLE_MPS": True,  # Enable PyTorch's Metal Performance Shaders on Apple Silicon
        "AUDIO_SAMPLE_RATE": 24000,  # Default sample rate for audio processing (in Hz)
        "SUNO_USE_SMALL_MODELS": True,  # Use smaller Bark models if True
        "CACHE_DIR": "./models",
        "LOG_LEVEL": LogLevel.INFO,  # Default logging level
        "USE_GPU": True,  # Whether to prioritize GPU/MPS over CPU
    }

    def __init__(self) -> None:
        """Initialize the EnvVars instance and load variables."""
        self._vars: Dict[str, Any] = {}
        self._load_env_vars()
        self._update_attributes()

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file or system, falling back to defaults."""
        load_dotenv()  # Load .env file into os.environ
        for var_name, default_value in self._DEFAULTS.items():
            value = os.getenv(var_name)
            if value is None:
                logger.info(
                    f"{var_name} not found in environment, using default: {default_value}"
                )
                self._vars[var_name] = default_value
            else:
                # Convert value to the appropriate type based on default
                if isinstance(default_value, bool):
                    self._vars[var_name] = value.lower() in ("true", "1", "t")
                elif isinstance(default_value, int):
                    self._vars[var_name] = int(value)
                elif isinstance(default_value, float):
                    self._vars[var_name] = float(value)
                elif isinstance(default_value, LogLevel):
                    self._vars[var_name] = LogLevel(value.upper())
                else:
                    self._vars[var_name] = value
                logger.info(
                    f"{var_name} loaded from environment: {self._vars[var_name]}"
                )

    def _update_attributes(self) -> None:
        """Update instance attributes and apply settings (e.g., logging level, device)."""
        # Set instance attributes
        self.GLOBAL_ENABLE_MPS: bool = self._vars["GLOBAL_ENABLE_MPS"]
        self.AUDIO_SAMPLE_RATE: int = self._vars["AUDIO_SAMPLE_RATE"]
        self.SUNO_USE_SMALL_MODELS: bool = self._vars["SUNO_USE_SMALL_MODELS"]
        self.CACHE_DIR: str = self._vars["CACHE_DIR"]
        self.LOG_LEVEL: LogLevel = self._vars["LOG_LEVEL"]
        self.USE_GPU: bool = self._vars["USE_GPU"]
        self.DEVICE: str = grab_best_device(self.USE_GPU, self.GLOBAL_ENABLE_MPS)
        logging.getLogger().setLevel(self.LOG_LEVEL.value)

    def update(self, var_name: str, value: Any) -> None:
        """
        Update an environment variable at runtime and reapply settings.

        Args:
            var_name (str): Name of the variable to update (must be in _DEFAULTS).
            value (Any): New value for the variable.

        Raises:
            KeyError: If var_name is not a recognized environment variable.
        """
        if var_name not in self._DEFAULTS:
            raise KeyError(f"Unknown environment variable: {var_name}")

        # Convert value to the appropriate type based on default
        default_type = type(self._DEFAULTS[var_name])
        if default_type is bool:
            self._vars[var_name] = bool(
                value.lower() in ("true", "1", "t") if isinstance(value, str) else value
            )
        elif default_type is int:
            self._vars[var_name] = int(value)
        elif default_type is float:
            self._vars[var_name] = float(value)
        elif default_type is LogLevel:
            self._vars[var_name] = LogLevel(
                value.upper() if isinstance(value, str) else value
            )
        else:
            self._vars[var_name] = value

        logger.info(f"Updated {var_name} to {self._vars[var_name]}")
        self._update_attributes()


# Create global instance to access environment variables
env = EnvVars()


def get_cached_or_download_model_from_hf(
    repo_id: str, file_name: str, cache_dir: str = env.CACHE_DIR
) -> str:
    """
    Download a model from Hugging Face Hub if not already cached.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub (e.g., 'suno/bark').
        file_name (str): The name of the model file to download (e.g., 'text.pt').
        cache_dir (str): Directory to store cached models (defaults to env.CACHE_DIR).

    Returns:
        str: The full path to the downloaded or cached model file.

    Raises:
        OSError: If the cache directory cannot be created.
        RuntimeError: If the download from Hugging Face fails.
    """
    # Ensure cache directory exists
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create cache directory {cache_dir}: {str(e)}")
        raise

    # Check if file is already cached
    cached_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cached_path):
        logger.debug(f"Model found in cache: {cached_path}")
        return cached_path

    # Download from Hugging Face if not cached
    logger.info(f"Downloading model {repo_id}/{file_name} to {cache_dir}")
    try:
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=cache_dir)
        logger.debug(f"Model downloaded successfully to {cached_path}")
        return cached_path
    except Exception as e:
        logger.error(f"Failed to download model {repo_id}/{file_name}: {str(e)}")
        raise RuntimeError(f"Failed to download model {repo_id}/{file_name}: {str(e)}")
