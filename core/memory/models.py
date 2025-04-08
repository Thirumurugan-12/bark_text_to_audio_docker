import os
import sys
import logging
from dataclasses import asdict
from typing_extensions import Optional, Callable
from dataclasses import dataclass
from enum import Enum
from transformers import BertTokenizer
from encodec import EncodecModel

import torch
from core.model.bark import GPT
from core.memory.common import env
from core.utils import download_file_from_hf
from core.model.hubert import HuBERTForBarkSemantic, HubertForBarkSemanticConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Memory threshold (in percentage) to trigger unloading of models when memory usage gets too high
# 90% of available memory; applies to GPU unless offloaded to CPU
MEMORY_THRESHOLD = 0.9


@dataclass(frozen=True)
class ModelInfo:
    """Data structure to hold metadata about a model."""

    # Hugging Face repository ID (e.g., "suno/bark")
    repo_id: Optional[str] = None
    # Filename of the model weights (e.g., "text.pt")
    file_name: Optional[str] = None
    # Pretrained checkpoint name (e.g., "facebook/encodec_24khz")
    checkpoint_name: Optional[str] = None
    # Configuration class for the model
    config_class: Optional[type] = None
    # Model class to instantiate
    model_class: Optional[type] = None
    # Preprocessor class (e.g., tokenizer)
    preprocessor_class: Optional[type] = None
    # Type of model (e.g., "text", "coarse", "encodec")
    model_type: Optional[str] = None
    # define the function that load the model
    load_model: Optional[Callable] = None


@dataclass
class Model:
    """Container for a loaded model, its configuration, and preprocessor."""

    model: Callable  # The PyTorch model instance
    config: Optional[Callable] = None  # Model configuration object
    # Preprocessor (e.g., tokenizer for text models)
    preprocessor: Optional[Callable] = None


def _load_encodec_model(model_info: ModelInfo, device: torch.device) -> Model:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    return Model(model)


def _load_hubert_base_for_bark_semantic(
    model_info: ModelInfo, device: torch.device
) -> "Model":
    os.makedirs(env.CACHE_DIR, exist_ok=True)
    local_file_path = os.path.join(env.CACHE_DIR, model_info.file_name)
    if not os.path.isfile(local_file_path):
        logger.info(
            f"Downloading {model_info.file_name} model from {model_info.repo_id}"
        )
        download_file_from_hf(
            model_info.repo_id, "model", model_info.file_name, env.CACHE_DIR
        )

    checkpoint = torch.load(local_file_path, map_location=device)

    assert isinstance(
        checkpoint, dict
    ), "expecting a dictionary, got {type(checkpoint)}"

    state_dict = checkpoint.get("model_state_dict", None)
    assert (
        state_dict is not None
    ), f"model_state_dict not in checkpoint, {checkpoint.keys()}"

    model_config = checkpoint.get("config", None)
    assert model_config is not None, "not found model config in checkpoint"

    config = HubertForBarkSemanticConfig(**model_config)
    model = HuBERTForBarkSemantic(
        config=config, load_hubert_pretrained_weights=False, device=device
    )
    model.load_state_dict(state_dict=state_dict, strict=True)

    return Model(model=model, config=config, preprocessor=None)


# TODO: refactor this class, each ModelInfo should have its own _load_model function for consistency
# and avoid complicated if-else paths
class ModelEnum(Enum):
    """
    Enumeration of supported models with their metadata.
    Each entry maps to a ModelInfo object defining how to load the model.
    """

    BARK_TEXT_SMALL = ModelInfo(
        repo_id="suno/bark",
        file_name="text.pt",
        model_type="text",
        model_class=GPT,
        preprocessor_class=BertTokenizer,
    )
    BARK_COARSE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="coarse.pt", model_type="coarse"
    )
    BARK_FINE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="fine.pt", model_type="fine"
    )

    BARK_TEXT = ModelInfo(repo_id="suno/bark", file_name="text_2.pt", model_type="text")
    BARK_COARSE = ModelInfo(
        repo_id="suno/bark", file_name="coarse_2.pt", model_type="coarse"
    )
    BARK_FINE = ModelInfo(repo_id="suno/bark", file_name="fine_2.pt", model_type="fine")

    CustomHuBERTTokenizer = ModelInfo(
        repo_id="GitMylo/bark-voice-cloning",
        file_name="quantifier_hubert_base_ls960_14.pth",
        model_type="custom_hubert_tokenizer",
    )

    ENCODEC24k = ModelInfo(
        checkpoint_name="facebook/encodec_24khz",
        model_type="encodec",
        load_model=_load_encodec_model,
    )

    HuBERTBaseForBarkSemantic = ModelInfo(
        checkpoint_name="facebook/hubert-base-ls960",
        repo_id="sleeper371/hubert-for-bark-semantic",
        file_name="hubert_epoch_30_2025_04_06_03_23_eval_loss_0.5520355800787607_acc_0.8344086021505376.pt",
        load_model=_load_hubert_base_for_bark_semantic,
    )

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """
        Retrieve ModelInfo for a given model name.

        Args:
            model_name (str): Name of the model (e.g., "BARK_TEXT_SMALL")

        Returns:
            ModelInfo: Metadata for the requested model

        Raises:
            ValueError: If the model name is not recognized
        """
        try:
            return cls[model_name].value
        except KeyError:
            raise ValueError(f"Unknown model name: {model_name}")
