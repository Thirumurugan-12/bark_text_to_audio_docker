import torch
import numpy as np

from encodec import EncodecModel
from encodec.utils import convert_audio
from core.memory import model_manager, ModelEnum, env
from core.bark.custom_context import inference_mode


def encodec_decode_fine_tokens_to_audio(fine_tokens: torch.Tensor) -> np.ndarray:
    """
    expecting fine_tokens shape [codebook_size, timestep], concretely [8, 75*duration_in_sec]
    Decode the given fine_tokens using the Encodec's decoder
    Returns the audio sample array as an np.ndarray
    Returns
        np.ndarray of shape (B, C, T), C = 1 for mono audio
    """
    model_info = ModelEnum.ENCODEC24k.value

    model_wrapper = model_manager.get_model(model_info)
    model: EncodecModel = model_wrapper.model

    device = next(model.parameters()).device

    input_tensor = fine_tokens.transpose(0, 1).to(device)

    emb = model.quantizer.decode(input_tensor)

    output: torch.Tensor = model.decoder(emb)
    audio_arr = output.detach().cpu().numpy()

    del input_tensor, emb, output

    return audio_arr


def encodec_encode_audio(
    audio_sample: torch.Tensor, audio_sample_rate: int
) -> torch.Tensor:
    """
    Encode the given audio sample using the encodec model
    audio_sample expected shape: (channels, sample)

    Returns codes as a tensor shape [n_q, T]
        where n_q typically is 8 and T is the compressed time step dimension (75 per second for 24khz model)
    """
    model_wrapper = model_manager.get_model(ModelEnum.ENCODEC24k.value)
    model: EncodecModel = model_wrapper.model

    device = next(model.parameters()).device

    wav = convert_audio(
        audio_sample, audio_sample_rate, model.sample_rate, model.channels
    )
    wav = wav.unsqueeze(0).float().to(device)

    # Extract discrete codes from EnCodec
    with inference_mode():
        encoded_frames = model.encode(wav)

    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

    return codes[0, :, :]
