import os
from pathlib import Path
from datetime import datetime
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, LinearLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from tqdm import tqdm

from typing import Literal, List, Optional, Tuple, Dict, Callable, Union, Any
from core.data_model import WavSemantic, WavSemanticDataset
from core.utils import read_audio_file, upload_file_to_hf

# cudnn error about non-contiguous input at the lstm layer, disable it fixed the issue
torch.backends.cudnn.enabled = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


HUBERT_SAMPLE_RATE = 16000
# 10_000 and 10_001 are for SOS and EOS tokens
SEMANTIC_PADDING_TOKEN = 10002
SOS_TOKEN = 10_000
EOS_TOKEN = 10_001


class WavSemanticTorchDataset(Dataset):
    """PyTorch Dataset for WavSemantic data with resampling and noise augmentation.
    Padding is carried out in a collator function.

    Args:
        samples: List of WavSemantic objects (speech data).
        orig_sample_rate: Original sample rate of the audio.
        target_sample_rate: Desired sample rate (default: 16000 Hz).
        device: Device to move tensors to (optional).
        noises: List of noise waveforms as NumPy arrays (optional, for augmentation).
            noises audio must already have sample_rate = target_sample rate, this class doesn't resample it
        augment_prob: Probability of applying noise augmentation (default: 0.5).
    """

    def __init__(
        self,
        samples: List["WavSemantic"],
        orig_sample_rate: int,
        target_sample_rate: Optional[int] = 16000,
        device: Optional[torch.device] = None,
        noises: Optional[List[np.ndarray]] = None,
        augment_prob: float = 0.5,
    ):
        self.samples = samples
        self.orig_sample_rate = orig_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.noises = noises
        self.augment_prob = augment_prob
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate, new_freq=target_sample_rate
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to [-1, 1]."""
        max_val = wav.abs().max()
        if max_val > 0:
            wav = wav / max_val
        return wav

    def _add_time_varying_noise(
        self, speech: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Add noise to a random segment of the speech with fade-in/fade-out."""
        speech_len = speech.size(0)
        noise_len = noise.size(0)

        # Match noise length (loop or trim)
        if noise_len < speech_len:
            repeats = int(np.ceil(speech_len / noise_len))
            noise = noise.repeat(repeats)[:speech_len]
        else:
            noise = noise[:speech_len]

        # Random segment (50%-100% of speech length)
        seg_len = int(speech_len * np.random.uniform(0.5, 1.0))
        start = np.random.randint(0, speech_len - seg_len + 1)
        end = start + seg_len

        # Compute noise scaling based on SNR
        speech_energy = torch.mean(speech[start:end] ** 2)
        noise_energy = torch.mean(noise[start:end] ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_scale = torch.sqrt(speech_energy / (noise_energy * snr_linear + 1e-10))

        # Apply noise to segment with fade-in/fade-out
        fade_len = min(1000, seg_len // 4)  # Fade over 1000 samples or 1/4 segment
        fade_in = torch.linspace(0, 1, fade_len)
        fade_out = torch.linspace(1, 0, fade_len)
        mask = torch.ones(seg_len)
        if fade_len > 0:
            mask[:fade_len] = fade_in
            mask[-fade_len:] = fade_out

        noisy_segment = speech[start:end] + (noise_scale * noise[start:end] * mask)
        noisy_speech = speech.clone()
        noisy_speech[start:end] = noisy_segment

        return torch.clamp(noisy_speech, -1, 1)

    def _augment_with_noise(self, wav: torch.Tensor) -> torch.Tensor:
        """Augment waveform with random noise mixture."""
        if not self.noises or len(self.noises) == 0:
            return wav

        # Decide how many noises to mix (1 or 2)
        num_noises = np.random.randint(1, 3)  # 1 or 2 noises
        random_indices = np.random.randint(0, len(self.noises), size=num_noises)
        selected_noises = [self.noises[i] for i in random_indices]
        noisy_wav = wav.clone()
        for noise_np in selected_noises:
            noise = torch.from_numpy(noise_np).float()
            noise = self._normalize_waveform(noise)  # Normalize noise
            snr_db = np.random.uniform(0, 20)  # Random SNR between 0-20 dB
            noisy_wav = self._add_time_varying_noise(noisy_wav, noise, snr_db)

        # Volume normalization: re-normalize after mixing
        noisy_wav = self._normalize_waveform(noisy_wav)
        return noisy_wav

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]

        # Convert NumPy wav to torch tensor and resample
        wav_tensor = torch.from_numpy(sample.wav).float()
        if self.orig_sample_rate != self.target_sample_rate:
            wav_tensor = self.resampler(wav_tensor)

        # Normalize to [-1, 1]
        wav_tensor = self._normalize_waveform(wav_tensor)

        # Apply noise augmentation with probability
        if self.noises and np.random.rand() < self.augment_prob:
            wav_tensor = self._augment_with_noise(wav_tensor)

        # Convert semantic to torch tensor (assuming integer tokens for CTC)
        semantic_tensor = torch.from_numpy(sample.semantic).long()

        # Move to device if specified
        if self.device is not None:
            wav_tensor = wav_tensor.to(self.device)
            semantic_tensor = semantic_tensor.to(self.device)

        return wav_tensor, semantic_tensor


def wav_semantic_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    sos_token: int = SOS_TOKEN,  # Adjust based on your vocab
    eos_token: int = EOS_TOKEN,  # Adjust based on your vocab
    padding_token: int = SEMANTIC_PADDING_TOKEN,  # Adjust based on your vocab
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for wav and semantic token pairs, adding <SOS> and <EOS> to targets.

    Args:
        batch: List of (wav_tensor, semantic_tensor) tuples.
        sos_token: Index of the <SOS> token.
        eos_token: Index of the <EOS> token.
        padding_token: Index of the padding token.

    Returns:
        Tuple of (padded_wavs, padded_targets, wav_lengths, target_lengths).
        - padded_wavs: [B, max_wav_len]
        - padded_targets: [B, max_target_len] with <SOS> and <EOS>
        - wav_lengths: [B] (original wav lengths)
        - target_lengths: [B] (original semantic lengths + 2 for <SOS> and <EOS>)
    """
    waves, semantics = zip(*batch)
    # Add <SOS> and <EOS> to each semantic sequence
    semantics_with_tokens = [
        torch.cat(
            [
                torch.tensor([sos_token], dtype=torch.long, device=semantic.device),
                semantic,
                torch.tensor([eos_token], dtype=torch.long, device=semantic.device),
            ]
        )
        for semantic in semantics
    ]

    # Compute lengths *after* adding <SOS> and <EOS>
    wav_lengths = torch.tensor([wav.size(0) for wav in waves], dtype=torch.long)
    target_lengths = torch.tensor(
        [semantic.size(0) for semantic in semantics_with_tokens], dtype=torch.long
    )

    # Pad waves and targets to max length in batch
    max_wav_len = max(wav_lengths).item()
    max_target_len = max(target_lengths).item()

    padded_wavs = torch.zeros(size=(len(waves), max_wav_len), device=waves[0].device)
    padded_targets = torch.full(
        size=(len(semantics), max_target_len),
        fill_value=padding_token,
        dtype=torch.long,
        device=semantics[0].device,
    )

    for i, (wav, semantic) in enumerate(zip(waves, semantics_with_tokens)):
        padded_wavs[i, : wav.size(0)] = wav
        padded_targets[i, : semantic.size(0)] = semantic

    return padded_wavs, padded_targets, wav_lengths, target_lengths


def load_train_val_dataloaders(
    dataset: WavSemanticDataset,
    train_ratio: float,
    batch_size: int,
    target_sample_rate: int = 16000,
    noises: List[np.ndarray] = None,
    augment_prob: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load train and validation DataLoaders from a WavSemanticDataset with dynamic batch padding.

    Args:
        dataset: The WavSemanticDataset instance to split and load.
        train_ratio: Fraction of data to use for training (0 to 1).
        batch_size: Number of samples per batch.
        target_sample_rate: Target sample rate for resampling (default: 16000 Hz).
        device: Optional device to move tensors to (default: None, stays on CPU).

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # Split dataset into train and val
    total_samples = len(dataset.data)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size
    train_data, val_data = random_split(dataset.data, [train_size, val_size])

    # Create datasets without fixed max_sequence_length
    train_dataset = WavSemanticTorchDataset(
        samples=train_data,
        orig_sample_rate=dataset.sample_rate,
        target_sample_rate=target_sample_rate,
        device=device,
        noises=noises,
        augment_prob=augment_prob,
    )
    val_dataset = WavSemanticTorchDataset(
        samples=val_data,
        orig_sample_rate=dataset.sample_rate,
        target_sample_rate=target_sample_rate,
        device=device,
        noises=noises,
        augment_prob=augment_prob,
    )

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Increase if you have multiple cores
        collate_fn=wav_semantic_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=wav_semantic_collate_fn,
    )

    return train_dataloader, val_dataloader


def train_hubert_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    train_dataloader: DataLoader,
    grad_scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    progress_bar: Optional[tqdm] = None,
    enable_autocast: bool = False,
) -> Dict[str, float]:
    """
    Train the HuBERT model for one epoch using mixed-precision training with CrossEntropyLoss.

    Args:
        model: The HuBERT model with Transformer decoder.
        optimizer: Optimizer for updating model parameters.
        criterion: CrossEntropyLoss function.
        train_dataloader: DataLoader for training data.
        grad_scaler: Gradient scaler for mixed-precision training.
        device: Device to train on (e.g., 'cuda', 'mps', 'cpu').
        progress_bar: Optional tqdm progress bar.

    Returns:
        Dict with 'loss' metric.
    """
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        # DataLoader already moves data to device
        waves, targets = batch[0], batch[1]
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=enable_autocast
        ):

            logits: torch.Tensor = model(waves, targets)

            loss = criterion(logits[:, :-1, :].transpose(1, 2), targets[:, 1:])

        total_loss += loss.detach().item()

        # Mixed precision with scaler (remove scaler if autocast is disabled)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if progress_bar is not None:
            progress_bar.update(1)

    avg_loss = total_loss / len(train_dataloader)
    return {"loss": avg_loss}


def eval_hubert(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    val_dataloader: DataLoader,
    device: torch.device,
    sos_token: int = SOS_TOKEN,
    eos_token: int = EOS_TOKEN,
    padding_token: int = SEMANTIC_PADDING_TOKEN,
) -> Dict[str, float]:
    """
    Evaluate the updated HuBERT model with Transformer decoder on the validation set.

    Args:
        model: The HuBERT model with Transformer decoder.
        criterion: CrossEntropyLoss function.
        val_dataloader: DataLoader for validation data (waves, targets).
        device: Device to evaluate on.
        sos_token: Index of the <SOS> token.
        eos_token: Index of the <EOS> token.
        padding_token: Index of the padding token.

    Returns:
        Dict with 'loss', 'accuracy', and 'num_tokens' metrics.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch in val_dataloader:
        # targets: [B, T'] with <SOS> and <EOS>
        waves, targets = batch[0].to(device), batch[1].to(device)

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            # [B, T', semantic_vocab_size]
            # transformers use batch_first=True
            # targets is a tensor of [B, T'], all including [SOS] and [EOS] tokens
            logits: torch.Tensor = model(waves, targets)

            # remove the last token predictions from the logits
            # remove the first token, which is SOS token from the targets
            # transpose the logits tensor from (B, T, C) to (B, C, T)
            loss = criterion(logits[:, :-1, :].transpose(1, 2), targets[:, 1:])

            # Calculate accuracy (ignoring padding tokens)
            preds = logits.argmax(dim=-1)[:, :-1]
            target_shifted = targets[:, 1:]
            mask = target_shifted != padding_token
            total_correct += (preds[mask] == target_shifted[mask]).sum().item()
            total_tokens += mask.sum().item()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy, "num_tokens": total_tokens}


def _load_noise_dataset(data_path: str, target_sample_rate: int) -> List[np.ndarray]:
    data = []
    # Add more extensions as needed  ".flac", ".ogg", ".aiff"
    audio_extensions = (".wav", ".mp3")

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            # Check if the file has an audio extension
            if filename.lower().endswith(audio_extensions):
                filepath = os.path.join(root, filename)
                try:
                    audio = read_audio_file(
                        filepath,
                        target_sample_rate=target_sample_rate,
                        channels=1,
                        normalize=False,
                    )
                    data.append(audio)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {str(e)}")
                    continue

    if len(data) == 0:
        raise RuntimeError(f"No audio files found in {data_path} or its subdirectories")

    return data


def train_hubert_quantizer(
    model: nn.Module,
    model_config: Dict[str, Any],
    lr: float,
    num_epoch: int,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    data_path: str = "./wav_semantic_dataset",
    checkpoint_path: str = "./checkpoints",
    save_checkpoint_every: int = 2,
    enable_grad_scaler: bool = False,
    augment_data_with_noise: bool = False,
    augment_prob: float = 0.5,
    noise_data_path: str = "./noise_dataset",
    publish_hf: bool = False,
    publish_to_repo: str = "",
    num_samples: int = 5000,
    device: torch.device = "cuda",
) -> nn.Module:
    """
    Train a HuBERT model with mixed-precision training and save checkpoints.

    Args:
        model: The HuBERT model to train.
        lr: Learning rate for the optimizer.
        num_epoch: Number of epochs to train.
        train_ratio: Fraction of data for training.
        batch_size: Batch size for DataLoaders.
        data_path: Path to the saved dataset.
        checkpoint_path: Directory to save checkpoints.
        save_checkpoint_every: Save checkpoint every N epochs.
        augment_data_with_noise: whether to add random noise to training audio
        augment_prob: probability of a sample will be augmented with noise
        num_samples: maximum number of samples to load from the dataset
    Returns:
        The trained model.
    """

    #  else "mps" if torch.backends.mps.is_available()
    # mix precision training doesn't work with mps device at the grad_scaler.step(optimizer) step
    # for testing just run on cpu
    model.to(device)

    # Load dataset and create dataloaders
    dataset = WavSemanticDataset.load(data_path, num_samples=num_samples)
    noises = None
    if augment_data_with_noise:
        logger.info(f"reading noise data from {noise_data_path}")
        noises = _load_noise_dataset(noise_data_path, target_sample_rate=16000)

    train_dataloader, val_dataloader = load_train_val_dataloaders(
        dataset,
        train_ratio=train_ratio,
        batch_size=batch_size,
        target_sample_rate=HUBERT_SAMPLE_RATE,
        noises=noises,
        augment_prob=augment_prob,
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=SEMANTIC_PADDING_TOKEN)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=enable_grad_scaler)
    progress_bar = tqdm(total=num_epoch * len(train_dataloader), desc="Training HuBERT")
    # scheduler = LinearLR(
    #     optimizer, start_factor=1, end_factor=0.5, total_iters=(num_epoch / 2)
    # )
    scheduler = None
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epoch):
        train_result = train_hubert_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            grad_scaler=grad_scaler,
            device=device,
            progress_bar=progress_bar,
            enable_autocast=enable_grad_scaler,
        )
        with torch.no_grad():
            eval_result = eval_hubert(
                model=model,
                criterion=criterion,
                val_dataloader=val_dataloader,
                device=device,
            )

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{num_epoch}, Train: {train_result}, Eval: {eval_result}"
        )

        if (epoch + 1) % save_checkpoint_every == 0:
            checkpoint_file = os.path.join(
                checkpoint_path,
                f"hubert_epoch_{epoch + 1}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}_eval_loss_{eval_result.get('loss', 0)}_acc_{eval_result.get('accuracy', 0)}.pt",
            )
            torch.save(
                {  # should have save the model configuration for later loading
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    "train_result": train_result,
                    "eval_result": eval_result,
                    "config": model_config,
                },
                checkpoint_file,
            )
            logger.info(f"Saved checkpoint to {checkpoint_file}")

            if publish_hf:
                upload_file_to_hf(checkpoint_file, publish_to_repo, "model")

    progress_bar.close()
    return model
