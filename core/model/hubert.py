from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers import HubertModel, AutoConfig, AutoModel


@dataclass
class CustomHubertConfig:
    """Configuration class for CustomHubert model."""

    # e.g., "facebook/hubert-base-ls960" or "facebook/hubert-large-ll60k"
    checkpoint_name: str
    # Layer to extract features from (0-indexed, e.g., 9 for 10th layer)
    feature_layer: int = 11
    # Target audio sample rate in Hz
    target_sample_rate: int = 16000
    # Optional length multiple for audio trimming
    seq_len_multiple_of: Optional[int] = None


@dataclass
class HubertForBarkSemanticConfig:
    """Configuration for HuBERTForBarkSemantic."""

    # # HuBERT model checkpoint for feature extractor layer
    checkpoint_name: Literal["facebook/hubert-base-ls960", "hubert-large-ls960-ft"]
    vocab_size: int
    # Layer to extract features from
    feature_layer: int = 11
    # last three tokens for SOS, EOS and PAD tokens
    # maximum target sequence length
    max_target_length: int = 2000
    num_decoder_layer: int = 12
    sos_token_id: int = 10000
    eos_token_id: int = 10001


class HubertFeatureExtractor(nn.Module):
    """
    A custom HuBERT model that loads a pretrained model from transformers and extracts
    features from a specified layer. Processes raw audio waveforms and returns hidden states.

    Args:
        config (CustomHubertConfig): Configuration specifying checkpoint, layer, and audio settings.
        device (torch.device, optional): Device to run the model on (e.g., "cuda" or "cpu").
    """

    def __init__(
        self,
        config: CustomHubertConfig,
        load_pretrained_weights: bool,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.target_sample_rate = config.target_sample_rate

        # Load pretrained HuBERT model from transformers
        self.hubert_config = AutoConfig.from_pretrained(config.checkpoint_name)
        if load_pretrained_weights:
            self.model = HubertModel.from_pretrained(config.checkpoint_name)
        else:
            # don't download the pretrained weights, init the model from the config
            self.model = AutoModel.from_config(self.hubert_config)

        # Validate feature_layer
        # e.g., 12 for BASE, 24 for LARGE
        num_layers = self.model.config.num_hidden_layers
        if not (0 <= config.feature_layer < num_layers):
            raise ValueError(
                f"feature_layer must be between 0 and {num_layers - 1}, got {config.feature_layer}"
            )
        self.feature_layer = config.feature_layer

        # Move to device if specified
        if device is not None:
            self.to(device)

    @property
    def hidden_size(self) -> int:
        """Returns the hidden size of the HuBERT model (e.g., 768 for BASE, 1024 for LARGE)."""
        return self.model.config.hidden_size

    def forward(
        self,
        wav_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes raw audio waveforms through HuBERT and extracts features from the specified layer.
        Input audio sample rate expected 16k

        Args:
            wav_input (torch.Tensor): Raw audio waveforms, shape [batch_size, audio_length].
            return_shape (Tuple[int, int], optional): If provided, reshapes output to [batch_size, seq_length, hidden_size].

        Returns:
            torch.Tensor: Features from the specified layer. Shape depends on return_shape:
                          - If None: [batch_size * seq_length, hidden_size] (flattened).
                          - If provided: [batch_size, seq_length, hidden_size].
        """

        # Forward pass through HuBERT
        # output_hidden_states=True returns all layer outputs
        outputs: BaseModelOutput = self.model(
            input_values=wav_input, output_hidden_states=True, return_dict=True
        )

        # Extract features from the specified layer (0-indexed)
        # hidden_states is a tuple of [batch_size, seq_length, hidden_size] for each layer
        features = outputs.hidden_states[self.feature_layer]  # e.g., [2, 500, 768]
        features = features.contiguous()
        return features


class HuBERTForBarkSemantic(nn.Module):
    def __init__(
        self,
        config: HubertForBarkSemanticConfig,
        load_hubert_pretrained_weights: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config

        # HuBERT feature extractor
        hubert_config = CustomHubertConfig(
            checkpoint_name=config.checkpoint_name,
            feature_layer=config.feature_layer,
        )
        self.hubert = HubertFeatureExtractor(
            config=hubert_config,
            load_pretrained_weights=load_hubert_pretrained_weights,
            device=device,
        )

        # e.g., 768 for BASE
        input_size = self.hubert.model.config.hidden_size

        # Transformer Decoder
        self.decoder_embedding = nn.Embedding(config.vocab_size, input_size)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_target_length, input_size)
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=config.num_decoder_layer,  # Adjust as needed
        )
        self.fc = nn.Linear(input_size, config.vocab_size)

        if device is not None:
            self.to(device)

    def save_state_dict(self, save_path: str):
        torch.save(self.state_dict(), save_path)

    def forward(self, wav_input: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Extracts HuBERT features and predicts semantic token probabilities.

        Args:
            wav_input: [batch_size, audio_length] (e.g., [2, 160000])
            tgt: the target sequence

        Returns:
            [batch_size, seq_length, vocab_size + 1] (e.g., [2, 500, VOCAB_SIZE])
        """
        memory: torch.Tensor = self.hubert(wav_input)  # [B, T, 768]
        B, T_tgt = tgt.shape
        tgt_emb = self.decoder_embedding(tgt) + self.pos_embedding[:, :T_tgt, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt).to(tgt.device)

        output: torch.Tensor = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.fc(output)
        return logits

    @torch.no_grad
    def generate(
        self,
        wav_input: torch.Tensor,
        temperature: Optional[float] = 0.8,
        eos_p: Optional[float] = 0.5,
        max_length: int = 600,
    ) -> torch.Tensor:
        """
        Inference: autoregressive generation.
        assuming wav_input audio is at 16000 sample rate"""
        self.eval()
        memory = self.hubert(wav_input)
        B = wav_input.shape[0]
        tgt = torch.full(
            size=(B, 1), fill_value=self.config.sos_token_id, device=wav_input.device
        )

        for _ in range(max_length):
            tgt_emb = (
                self.decoder_embedding(tgt) + self.pos_embedding[:, : tgt.shape[1], :]
            )
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
                tgt.device
            )

            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            # logits shape (B, T', vocab_size)
            logits: torch.Tensor = self.fc(output[:, -1, :])

            if temperature is not None and temperature > 0:
                probs = torch.softmax(input=logits / temperature, dim=-1)
                next_token = torch.multinomial(input=probs, num_samples=1)
            else:
                probs = torch.softmax(input=logits, dim=-1)
                next_token = logits.argmax(dim=-1, keepdim=True)

            # stop if the EOS token probabilities are higher than the provided eos_p
            if eos_p is not None and eos_p > 0:
                if torch.all(probs[:, self.config.eos_token_id] > eos_p):
                    break

            # early stopping
            if torch.all(next_token == self.config.eos_token_id):
                break

            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break

        # remove the [SOS] token from the generated semantic sequences
        return tgt[:, 1:]
