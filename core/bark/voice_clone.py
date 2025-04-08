import torch
import torchaudio
from typing import Optional

from core.utils import read_audio_file
from core.bark import encodec_encode_audio

from core.model.hubert import HuBERTForBarkSemantic
from core.memory import model_manager, ModelEnum
from core.bark.custom_context import InferenceContext
from core.data_model import *


HUBERT_SAMPLE_RATE = 16000


def generate_semantic_tokens_from_hubert(
    waves: torch.Tensor,
    audio_sample_rate: int,
    temperature: float,
    eos_p: float,
    max_length: int,
    device: Optional[torch.device],
    inference_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate semantic tokens from audio using the HuBERT model.

    Args:
        audio: 2D tensor of raw audio samples (shape: [B, T], where T is the number of samples)
        sample_rate: Sample rate of the input audio (default: 24000, matching EnCodec in BARK)
        hubert_model_name: Name of the HuBERT model from Hugging Face (default: facebook/hubert-large-ls960-ft)
        device: Torch device to run the model on (defaults to CUDA if available, else CPU)
        max_length: Maximum length of semantic tokens to return (optional, for truncation)

    Returns:
        torch.Tensor: 1D tensor of semantic tokens (e.g., shape [N], where N is the sequence length)

    Raises:
        RuntimeError: If HuBERT model loading or processing fails
    """
    assert (
        len(waves.shape) == 2
    ), f"expecting a tensor of shape [B, T], got {waves.shape}"
    waves = waves.to(device)

    # # HuBERT expects audio at 16kHz, resample if necessary
    if audio_sample_rate != HUBERT_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=audio_sample_rate, new_freq=HUBERT_SAMPLE_RATE
        ).to(device)
        waves = resampler(waves)

    model = model_manager.get_model(ModelEnum.HuBERTBaseForBarkSemantic.value).model

    assert isinstance(
        model, HuBERTForBarkSemantic
    ), f"expecting HuBERTForBarkSemantic model type, received {type(model)}"

    waves = waves.to(dtype=inference_dtype)
    model = model.to(dtype=inference_dtype)

    with InferenceContext():
        predictions: torch.Tensor = model.generate(
            wav_input=waves, temperature=temperature, eos_p=eos_p, max_length=max_length
        )

    return predictions


def create_bark_prompt(
    audio_file: AudioFile, temperature: float, eos_p: float, device: torch.device
) -> BarkPrompt:
    """
    Turn raw audio into valid BARK prompt. When given a raw audio file, use this function
    to generate a valid BARK prompt
    """
    # Read the audio
    raw_audio = read_audio_file(
        path=audio_file.audio_file_path,
        target_sample_rate=HUBERT_SAMPLE_RATE,
        channels=1,
        max_duration=15,
    )

    audio_tensor = torch.tensor(raw_audio.astype(np.float32), device=device)
    # Generate semantic tokens from audio using HuBERT
    semantic_tokens: torch.Tensor = generate_semantic_tokens_from_hubert(
        waves=audio_tensor.unsqueeze(0),
        audio_sample_rate=16000,
        temperature=temperature,
        eos_p=eos_p,
        max_length=600,
        device=device,
    )

    # Generate codebook tokens using EnCodec
    codes = encodec_encode_audio(
        audio_sample=torch.from_numpy(raw_audio[None]),
        audio_sample_rate=HUBERT_SAMPLE_RATE,
    )

    # Assuming codes has shape [num_codebooks, T], typically 8 codebooks for 24kHz
    return BarkPrompt(semantic_tokens, codes[:2, :], codes[:, :])
