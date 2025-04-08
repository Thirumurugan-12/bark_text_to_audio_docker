from typing import List
import numpy as np
from tqdm import tqdm
from dataclasses import asdict
from core.bark.generate_semantic import generate_semantic_tokens_from_text
from core.bark.generate_coarse import generate_coarse_tokens_from_semantic
from core.bark.generate_fine import generate_fine_tokens_from_coarse
from core.bark.encodec import encodec_decode_fine_tokens_to_audio
from core.bark.generate_audio import remove_padded_segment_from_audio
from core.data_model import WavSemantic, WavSemanticDataset, BarkGenerationConfig
from core.bark.constants import SEMANTIC_PAD_TOKEN


def generate_wav_semantic_dataset(
    text_file_path: str,
    generation_config: BarkGenerationConfig,
    batch_size: int = 16,
    silent: bool = False,
    save_path: str = "./dataset",
    save_data_as_raw_audio: bool = True,
) -> None:
    """
    Generate a dataset of (wav, semantic_tokens) for training a model to predict semantic tokens from audio

    Args
        text_file_path: path to the text file that will be used to generate audio data
        generation_config: the config used to generate data
        batch_size: batch size when generate data
        bark_model_type: either `large` or `small`, the coarse and fine model variant that will be used to generate audio
        max_token_per_example: a criteria to limit the length of an example from text. The text will be tokenized using a BERT tokenizer,
            and the tokenized text will be truncated to not exceed this length
        save_path: path to save the generated dataset
        save_data_as_raw_audio: if True, waves will be saved as raw audio, otherwise it will be saved as compressed .npz file
    """
    texts = read_text_file(text_file_path)
    assert len(texts) > 0, "empty text data"

    mini_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    progress_bar = tqdm(
        total=len(mini_batches), disable=silent, desc="Generating wav-semantic dataset"
    )
    for batch in mini_batches:
        semantic_tokens = generate_semantic_tokens_from_text(
            texts=batch, semantic_prompt=None, silent=True, **asdict(generation_config)
        )

        coarse = generate_coarse_tokens_from_semantic(
            semantic_tokens=semantic_tokens,
            history_prompt=None,
            silent=True,
            **asdict(generation_config)
        )

        fine = generate_fine_tokens_from_coarse(
            coarse_tokens=coarse,
            history_prompt=None,
            temperature=generation_config.generate_fine_temperature,
            use_small_model=generation_config.use_small_model,
            silent=True,
        )

        # generate audio waves from the fine tokens
        waves = encodec_decode_fine_tokens_to_audio(fine)
        # remove the channel dimension
        waves = waves.squeeze(1)

        waves = remove_padded_segment_from_audio(waves, semantic_tokens.cpu().numpy())

        save_semantic_wave_data(
            batch,
            waves,
            semantic_tokens.detach().cpu().numpy(),
            24000,
            generation_config,
            save_path,
            save_data_as_raw_audio,
        )

        progress_bar.update(1)
        del semantic_tokens, coarse, fine, waves


def save_semantic_wave_data(
    texts: List[str],
    waves: List[np.ndarray],
    semantic_tokens: np.ndarray,
    sample_rate: int,
    generation_config: BarkGenerationConfig,
    save_path: str,
    save_raw_audio: bool,
) -> None:
    """
    Save the given data as a WaveSemantic dataset
    """
    examples = []
    assert (
        len(texts) == len(waves) == semantic_tokens.shape[0]
    ), "unexpected array length"

    model_type = "small" if generation_config.use_small_model else "large"

    # remove the padding tokens at the end of the semantic sequences
    mask = semantic_tokens == SEMANTIC_PAD_TOKEN
    semantic_padding_indices = np.argmax(mask.astype(np.int32), axis=1)

    for i, (text, padding_index) in enumerate(zip(texts, semantic_padding_indices)):
        if padding_index == 0:
            padding_index = len(semantic_tokens[i])
        example = WavSemantic(text, waves[i], semantic_tokens[i, :padding_index])
        examples.append(example)

    dataset = WavSemanticDataset(sample_rate, generation_config, model_type, examples)

    dataset.save(save_path, save_raw_audio)


def read_text_file(path: str) -> List[str]:
    with open(path, "r") as file:
        lines = file.readlines()
        # Remove newline characters
        lines = [line.strip() for line in lines]
        return lines
