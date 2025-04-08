import sys
import logging
from typing_extensions import Union, List
import numpy as np
import torch
from dataclasses import asdict

from core.bark.generate_semantic import generate_semantic_tokens_from_text
from core.bark.generate_coarse import generate_coarse_tokens_from_semantic
from core.bark.generate_fine import generate_fine_tokens_from_coarse


from core.data_model.bark import BarkPrompt, BarkGenerationConfig
from core.bark.encodec import encodec_decode_fine_tokens_to_audio
from core.bark.constants import SEMANTIC_PAD_TOKEN, SEMANTIC_RATE_HZ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_audio(
    texts: List[str],
    prompt: Union[BarkPrompt, None] = None,
    generation_config: BarkGenerationConfig = None,
    silent: bool = False,
) -> List[np.ndarray]:
    """
    Generate audio from text with an optional audio prompt
    Args:
        text (str): Input text to generate audio. Must be non-empty.
        num_gen (int): number of audio to generate per text
        prompt (Union[str, None]): optional path to a prompt file of type .npz that will be used as the audio prompt
        generation_config: configurations to generate audio

    """
    if prompt is not None:
        semantic_prompt = prompt.semantic_prompt if prompt is not None else None
        # if len(semantic_prompt.shape) == 2:
        #     semantic_prompt = semantic_prompt[0, :]
        assert (
            len(semantic_prompt.shape) == 1
        ), "expecting semantic prompt as a 1D array"
    else:
        semantic_prompt = None

    if generation_config is None:
        logger.info("using BARK default generation config")
        generation_config = BarkGenerationConfig()

    semantic_tokens = generate_semantic_tokens_from_text(
        texts,
        semantic_prompt,
        **asdict(generation_config),
        silent=silent,
    )

    # because we generate audio in batch, all audios in one batch have the same length
    # of the longest audio. We need to remove the random section of the shorter audio
    # after it has ended

    # coarse token generation
    coarse_tokens = generate_coarse_tokens_from_semantic(
        semantic_tokens, prompt, **asdict(generation_config), silent=silent
    )

    # fine token generation
    fine_tokens = generate_fine_tokens_from_coarse(
        coarse_tokens=coarse_tokens,
        history_prompt=prompt,
        temperature=generation_config.generate_fine_temperature,
        use_small_model=generation_config.use_small_model,
        silent=silent,
    )

    # decoding the codes
    audio_wave = encodec_decode_fine_tokens_to_audio(fine_tokens)
    assert (
        len(audio_wave.shape) == 3
    ), f"expecting audio tensor of shape (B, C, T), received {audio_wave.shape}"

    audio_wave = audio_wave.squeeze(1)  # squeeze the channel dimension
    res = remove_padded_segment_from_audio(audio_wave, semantic_tokens.cpu().numpy())
    return res


def remove_padded_segment_from_audio(
    audio_wave: np.ndarray, semantic_tokens: np.ndarray, audio_sample_rate: int = 24000
) -> List[np.ndarray]:
    # Because the semantic token tensor's time step dimension is of the longest audio in the sample
    # all the remaining audio have shorter length would have random sound after its end
    # we will change the values of coarse_token tensor of shorter audios at positions after it end
    # to avoid random sound in the generated results
    # SEMANTIC_PAD_TOKEN is also the end of sentence token
    # this function assume audio_wave has shape (batch, T)
    assert (
        len(audio_wave.shape) == 2
    ), f"expecting ndarray of shape (B, T), received {audio_wave.shape}"
    mask = semantic_tokens == SEMANTIC_PAD_TOKEN
    semantic_eos_indices = np.argmax(mask.astype(np.int32), axis=1)  # Shape [batch]
    wave_eos_indices: np.ndarray = semantic_eos_indices * (
        audio_sample_rate / SEMANTIC_RATE_HZ
    )
    wave_eos_indices = wave_eos_indices.astype(np.int32)
    res = []
    for wave_index in range(audio_wave.shape[0]):
        if wave_eos_indices[wave_index] == 0:
            # zero means this audio is the longest one in the batch and there is no need to cut the padded segment
            res.append(audio_wave[wave_index])
            continue
        start_padding_index = wave_eos_indices[wave_index]
        res.append(audio_wave[wave_index, :start_padding_index])

    return res
