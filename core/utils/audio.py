"""
Helpful functions to process audio
"""

import numpy as np
import soundfile as sf

from typing_extensions import Annotated, Literal, Optional
import torchaudio
import torch

AudioChannel = Literal[1, 2]


def read_audio_file(
    path: str,
    target_sample_rate: int = 16000,
    channels: int = 1,
    normalize: bool = True,
    max_duration: Optional[float] = None,
) -> np.ndarray:
    """Read and resample audio file
    If target_sample_rate is different than the audio's sample rate, this function will resample it
    If GPU is available, the resampling will be on GPU.

    Args:
        path: Path to the audio file (supports WAV, FLAC, OGG)
        target_sample_rate: Target sample rate (default: 24000)
        channels: Number of output channels (1 for mono, 2 for stereo)
        normalize: Whether to normalize audio to [-1, 1]
        max_duration: Maximum duration in seconds (truncates longer files)
        device: Device to process on ("cuda" or "cpu", defaults to cuda if available)

    Returns:
        np.ndarray: Processed audio samples as a numpy array

    Raises:
        RuntimeError: If the file cannot be read or processing fails
    """
    try:
        # Load audio file with torchaudio
        waveform, original_sample_rate = torchaudio.load(path)  # [channels, samples]

        # Truncate to max_duration before resampling
        if max_duration is not None:
            max_samples = int(max_duration * original_sample_rate)
            if waveform.size(1) > max_samples:
                waveform = waveform[:, :max_samples]

        # Downmix to desired channels
        if waveform.size(0) > channels:
            if channels == 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # Mono: average channels
            elif channels == 2:
                waveform = waveform[:2, :]  # Stereo: take first 2 channels

        # Resample if needed
        if original_sample_rate != target_sample_rate:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            waveform = waveform.to(device)
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=target_sample_rate,
                resampling_method="sinc_interp_kaiser",  # Fast and high-quality
            ).to(device)
            waveform = resampler(waveform)

        # Normalize to [-1, 1] if requested
        if normalize:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

        # Move back to CPU and convert to numpy
        data = waveform.cpu().numpy()

        # Ensure correct shape (remove extra dim if mono)
        if channels == 1 and data.shape[0] == 1:
            data = data[0, :]

        return data

    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {path}: {str(e)}")


def save_audio_file(
    audio_array: np.ndarray, sample_rate: int, file_path: str, format="WAV"
):
    """
    Save an audio array to a file.

    Parameters:
    - audio_array: numpy array or list containing the audio samples
    - sample_rate: int, the sample rate of the audio (e.g., 44100 Hz)
    - file_path: str, path where the file will be saved (e.g., 'output.wav')
    - format: str, audio file format (e.g., 'WAV', 'FLAC', 'OGG'), default is 'WAV'
    """
    try:
        if not file_path.endswith(".wav"):
            file_path += ".wav"
        sf.write(file_path, audio_array, sample_rate, format=format)
    except Exception as e:
        print(f"Error saving audio file at {file_path}: {e}")
