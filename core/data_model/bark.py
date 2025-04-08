import os
import json
from pathlib import Path
import torch

from dataclasses import dataclass, asdict, fields
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Literal
from datetime import datetime
from core.utils import save_audio_file, read_audio_file


@dataclass
class BarkGenerationConfig:
    semantic_top_k: Union[int, None] = 1000  # a tenth of the semantic vocab size
    coarse_top_k: Union[int, None] = 100  # a tenth of the coarse codebook size
    semantic_top_p: Union[int, None] = None
    coarse_top_p: Union[int, None] = None
    min_eos_p: float = 0.5
    max_gen_duration_second: Union[float, None] = None
    allow_early_stop: bool = True
    use_kv_caching: bool = True
    max_coarse_history: int = 630
    sliding_window_length: int = 60
    max_token_per_example: int = 256
    # set to None to use argmax sampling
    temperature: float = 0.6
    generate_coarse_temperature: float = 0.6
    # set this to None if you want to use argmax to generate fine token
    generate_fine_temperature: float = 0.6
    use_small_model: bool = True

    def __init__(self, **kwargs):
        # Get field names from dataclass
        valid_fields = {f.name for f in fields(self)}
        # Set only known fields
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BarkGenerationConfig":
        return cls(**data)


@dataclass
class BarkPrompt:
    """
    semantic_prompt shape: (T)
    coarse_prompt shape: (2, T)
    fine_prompt shape: (8, T)
    those T are different depends on the rate of token type per second
    """

    semantic_prompt: torch.Tensor
    coarse_prompt: torch.Tensor
    fine_prompt: torch.Tensor

    def save_prompt(self, file_path: str) -> bool:
        """
        Save all 3 prompts to disk as JSON. Return True if success, False if error
        """
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory:  # If there's a directory component
            os.makedirs(directory, exist_ok=True)

        data = {
            "semantic_prompt": self.semantic_prompt.detach().cpu().tolist(),
            "coarse_prompt": self.coarse_prompt.detach().cpu().tolist(),
            "fine_prompt": self.fine_prompt.detach().cpu().tolist(),
        }

        if not file_path.endswith(".json"):
            file_path += ".json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    @classmethod
    def load_prompt(cls, file_path: str, device: torch.device) -> "BarkPrompt":
        """
        Load a prompt from disk. File to load can be either a .json or .npz file
        """
        try:
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    prompt = json.load(f)

                assert (
                    "semantic_prompt" in prompt
                    and "coarse_prompt" in prompt
                    and "fine_prompt" in prompt
                ), f"invalid prompt data {prompt}"

                semantic_prompt = torch.tensor(prompt["semantic_prompt"])
                coarse_prompt = torch.tensor(prompt["coarse_prompt"])
                fine_prompt = torch.tensor(prompt["fine_prompt"])

            elif file_path.endswith(".npz"):
                with np.load(file_path) as data:
                    assert (
                        "semantic_prompt" in data
                        and "coarse_prompt" in data
                        and "fine_prompt" in data
                    ), f"invalid prompt data in NPZ file"

                    semantic_prompt = torch.from_numpy(data["semantic_prompt"])
                    coarse_prompt = torch.from_numpy(data["coarse_prompt"])
                    fine_prompt = torch.from_numpy(data["fine_prompt"])

            else:
                raise ValueError("Unsupported file format. Use .json or .npz")

            # Convert to device and dtype after loading
            semantic_prompt = semantic_prompt.to(device=device, dtype=torch.int32)
            coarse_prompt = coarse_prompt.to(device=device, dtype=torch.int32)
            fine_prompt = fine_prompt.to(device=device, dtype=torch.int32)

            # Shape checks remain the same
            if len(semantic_prompt.shape) == 2:
                semantic_prompt = semantic_prompt[0, :]
            assert (
                len(semantic_prompt.shape) == 1
            ), "expecting semantic_prompt as a 1D array"

            assert (
                coarse_prompt.shape[0] == 2
            ), "expecting coarse_prompt has 2 code book dimension"

            assert (
                fine_prompt.shape[0] == 8
            ), "expecting fine_prompt has 8 code book dimension"

            return cls(semantic_prompt, coarse_prompt, fine_prompt)

        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")


class AudioFile(BaseModel):
    """Model for validating raw audio prompt inputs."""

    audio_file_path: str = Field(..., description="Path to the audio file")
    max_duration: int = Field(
        ..., ge=1, description="Maximum duration of the audio in seconds"
    )

    def get_default_prompt_name(self) -> str:
        audio_file_name = Path(self.audio_file_path).name
        return f"{audio_file_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"


class TextToAudioInput(BaseModel):
    """Model for validating inputs to the text-to-audio generation function."""

    texts: List[str] = Field(
        ..., min_items=1, description="List of text strings to convert to audio"
    )
    audio_prompt: Optional[Union[AudioFile, str]] = Field(
        None, description="Optional audio prompt (raw or file path)"
    )
    sample_rate: int = Field(
        default=24000, ge=1, description="Sample rate for generated audio"
    )
    device: Optional[str] = Field(
        None, description="Device to use for generation (e.g., 'cuda', 'cpu')"
    )
    save_path: str = Field(
        default="./artifact", description="Directory to save generated audio files"
    )


class TextToAudioModel(Enum):
    BARK = "BARK"


@dataclass
class WavSemantic:
    """
    An example of a pair (wav, semantic) for training a model to predict semantic from audio
    """

    text: str
    wav: np.ndarray
    semantic: np.ndarray


@dataclass
class WavSemanticDataset:
    sample_rate: int
    semantic_generation_config: BarkGenerationConfig
    bark_model_type: Literal["small", "large"]
    data: List[WavSemantic]

    def save(self, save_path: str, save_raw_audio: bool) -> None:
        """
        Save this WavSemanticDataset instance to disk at the specified path with compression.

        Args:
            save_path: Directory path where the dataset will be saved (default: './data').
        """
        # Ensure the save directory exists
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # this allows continuous saving of data, e.g save every new batch of data generated
        if not os.path.exists(save_dir / "metadata.json"):
            # Prepare metadata dictionary using instance attributes
            metadata = {
                "sample_rate": self.sample_rate,
                "semantic_generation_config": self.semantic_generation_config.to_dict(),
                "bark_model_type": self.bark_model_type,
            }

            # Save metadata as JSON
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        next_index = self._get_latest_saved_file_index(save_path) + 1
        # Save each WavSemantic sample
        for i, sample in enumerate(self.data):
            sample_dir = save_dir / f"sample_{i+next_index}"
            sample_dir.mkdir(exist_ok=True)

            # Save text
            with open(sample_dir / "text.txt", "w") as f:
                f.write(sample.text)

            # Save wav and semantic in a single compressed .npz file
            if save_raw_audio:
                save_audio_file(
                    sample.wav, self.sample_rate, str(sample_dir / "audio.wav")
                )
                with open(sample_dir / "semantic.json", "w") as f:
                    json.dump(sample.semantic.tolist(), f)
            else:
                np.savez_compressed(
                    sample_dir / "data.npz", wav=sample.wav, semantic=sample.semantic
                )

    @staticmethod
    def _get_latest_saved_file_index(dataset_path: str) -> int:
        file_names = os.listdir(dataset_path)
        file_names.remove("metadata.json")
        if len(file_names) == 0:
            return -1

        indices = [
            int(file_name.split("_")[-1].split(".")[0]) for file_name in file_names
        ]

        return max(indices)

    @classmethod
    def load(cls, load_path: str, num_samples: int = 5000) -> "WavSemanticDataset":
        """
        Load a WavSemanticDataset from disk at the specified path.

        Args:
            load_path: Directory path where the dataset is saved.
            num_samples: maximum number of samples to load from the folder
        Returns:
            A new WavSemanticDataset instance loaded from disk.
        """
        load_dir = Path(load_path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Directory {load_path} does not exist")

        filenames = os.listdir(load_dir)
        if len(filenames) == 1:
            # when there is a folder inside the load_path folder, step into it
            load_dir = load_dir / filenames[0]
            filenames = os.listdir(load_dir)

        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Reconstruct semantic_generation_config
        config = BarkGenerationConfig.from_dict(metadata["semantic_generation_config"])

        # Load each WavSemantic sample
        data = []
        for i, filename in enumerate(filenames):
            if not "sample" in filename:
                continue
            sample_dir = load_dir / filename

            # Load text
            with open(sample_dir / "text.txt", "r") as f:
                text = f.read()

            # Load compressed wav and semantic from .npz file
            if os.path.isfile(sample_dir / "data.npz"):
                with np.load(sample_dir / "data.npz") as npz_data:
                    wav = npz_data["wav"]
                    semantic = npz_data["semantic"]
            # assuming audio wave file was stored separately from the semantic file
            else:
                # assuming "audio.wav" and "semantic.npz" exist in the folder
                wav = read_audio_file(
                    sample_dir / "audio.wav", metadata["sample_rate"], 1, False, None
                )
                if os.path.isfile(sample_dir / "semantic.npz"):
                    with np.load(sample_dir / "semantic.npz") as npz_data:
                        semantic = npz_data["semantic"]
                elif os.path.isfile(sample_dir / "semantic.json"):
                    with open(sample_dir / "semantic.json") as f:
                        semantic = np.array(json.load(f))

            data.append(WavSemantic(text=text, wav=wav, semantic=semantic))
            if i > num_samples:
                break

        # Reconstruct and return the dataset
        return cls(
            sample_rate=metadata["sample_rate"],
            semantic_generation_config=config,
            bark_model_type=metadata["bark_model_type"],
            data=data,
        )

    def __getitem__(self, idx: int) -> WavSemantic:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)
