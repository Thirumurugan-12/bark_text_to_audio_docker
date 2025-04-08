import argparse
import logging
import os
from dataclasses import asdict
import torch
import torch.nn as nn
from core.trainer import train_hubert_quantizer
from core.model.hubert import (
    HuBERTForBarkSemantic,
    HubertForBarkSemanticConfig,
)
from core.utils import download_dataset_from_hf
from core.bark.constants import HUBERT_OUTPUT_VOCAB_SIZE


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
WORKSPACE = "./"

# HF repo id to the dataset
DATASET_REPO_ID = "sleeper371/bark-wave-semantic"
# if choose to publish checkpoint to HF, this will be the repo-id to publish checkpoint
CHECKPOINT_REPO_ID = "sleeper371/hubert-for-bark-semantic"
# name of the noise data file on the HF dataset repo
HF_NOISE_FILE_NAME = "environmental_sound.zip"


# local path that has the noise data use to enhance the training data
_LOCAL_NOISE_DATA_PATH = "noise_dataset"
# local path to the training audio folder
_LOCAL_TRAINING_DATA_PATH = "wav_semantic_dataset"
# local folder path to save trained checkpoint
_LOCAL_CHECKPOINTS_PATH = "checkpoints"


def prefix_workspace(workspace_path: str, path: str) -> str:
    return os.path.join(workspace_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description="HuBERT Training Script")
    parser.add_argument(
        "--hubert-checkpoint-name",
        type=str,
        default="facebook/hubert-base-ls960",
        help="checkpoint name that will be used as the feature extractor layer for CustomHuBERT",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=11,
        help="layer at which to use features for the LSTM",
    )

    parser.add_argument(
        "--mix-precision",
        action="store_true",
        help="train model with mix precision bfloat16 and gradient scaler",
    )

    parser.add_argument(
        "--lr", type=float, default=8e-5, help="Learning rate (default: 8e-5)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--dataset-file-name",
        type=str,
        default="short_sentences.zip",
        help="name of the dataset file in the HF repo to download",
    )

    parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )

    parser.add_argument(
        "--model-bfloat16",
        action="store_true",
        default=False,
        help="set true to convert and train model in bfloat16",
    )

    parser.add_argument(
        "--augment-data-with-noise",
        action="store_true",
        default=False,
        help="load and add noise randomly to training data as a regularization technique",
    )

    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.5,
        help="noise will be added to audio sample with this probability",
    )

    parser.add_argument(
        "--publish-hf",
        action="store_true",
        default=False,
        help="if set, publish checkpoints to huggingface hub",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=WORKSPACE,
        help="workspace folder to store data",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="number of examples to load from the dataset",
    )

    return parser.parse_args()


def ensure_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def calculate_model_memory(model: nn.Module):
    """
    Calculate and print the memory usage of a PyTorch model's parameters based on their detected data type.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    # Dictionary mapping PyTorch dtypes to bytes per parameter
    bytes_per_param_dict = {
        torch.float32: 4,  # 32 bits = 4 bytes
        torch.float16: 2,  # 16 bits = 2 bytes
        torch.int8: 1,  # 8 bits = 1 byte
        torch.int32: 4,  # 32 bits = 4 bytes
        torch.int64: 8,  # 64 bits = 8 bytes
    }

    # Detect the data type from the first parameter
    param_iter = iter(model.parameters())
    try:
        first_param = next(param_iter)
        dtype = first_param.dtype
    except StopIteration:
        print("Model has no parameters!")
        return

    # Get bytes per parameter based on detected dtype
    # Default to 4 bytes if dtype not found
    bytes_per_param = bytes_per_param_dict.get(dtype, 4)
    dtype_name = str(dtype).replace("torch.", "")  # Clean up dtype name for printing

    # Count total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Count total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate total memory in bytes
    total_memory_bytes = total_params * bytes_per_param

    # Convert to KB, MB, and GB for readability
    total_memory_kb = total_memory_bytes / 1024
    total_memory_mb = total_memory_kb / 1024
    total_memory_gb = total_memory_mb / 1024

    # Print results
    logger.info(f"Model Memory Usage (Detected dtype: {dtype_name}):")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Total Memory: {total_memory_gb:,.2f} GB")


def main():
    args = parse_args()

    # local path that has the noise data use to enhance the training data
    LOCAL_NOISE_DATA_PATH = prefix_workspace(args.workspace, _LOCAL_NOISE_DATA_PATH)
    # local path to the training audio folder
    LOCAL_TRAINING_DATA_PATH = prefix_workspace(
        args.workspace, _LOCAL_TRAINING_DATA_PATH
    )
    # local folder path to save trained checkpoint
    LOCAL_CHECKPOINTS_PATH = prefix_workspace(args.workspace, _LOCAL_CHECKPOINTS_PATH)

    # Create necessary directories
    ensure_directory(LOCAL_CHECKPOINTS_PATH)

    logger.info("Starting HuBERT training")

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    config = HubertForBarkSemanticConfig(
        vocab_size=HUBERT_OUTPUT_VOCAB_SIZE,
        checkpoint_name=args.hubert_checkpoint_name,
        feature_layer=args.feature_layer,
        num_decoder_layer=6,
    )
    model = HuBERTForBarkSemantic(
        config=config, load_hubert_pretrained_weights=True, device=device
    )

    if args.model_bfloat16:
        model = model.to(torch.bfloat16)
        logger.info("Training model in bfloat16 precision")

    calculate_model_memory(model)

    # Download datasets if needed
    if not os.path.exists(LOCAL_TRAINING_DATA_PATH):
        download_dataset_from_hf(
            DATASET_REPO_ID,
            args.dataset_file_name,
            LOCAL_TRAINING_DATA_PATH,
        )

    if args.augment_data_with_noise and not os.path.exists(LOCAL_NOISE_DATA_PATH):
        download_dataset_from_hf(
            DATASET_REPO_ID,
            HF_NOISE_FILE_NAME,
            LOCAL_NOISE_DATA_PATH,
        )

    # Train the model
    trained_model = train_hubert_quantizer(
        model=model,
        model_config=asdict(config),
        lr=args.lr,
        num_epoch=args.num_epochs,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        data_path=LOCAL_TRAINING_DATA_PATH,
        checkpoint_path=LOCAL_CHECKPOINTS_PATH,
        save_checkpoint_every=args.save_checkpoint_every,
        augment_data_with_noise=args.augment_data_with_noise,
        augment_prob=args.augment_prob,
        noise_data_path=LOCAL_NOISE_DATA_PATH,
        publish_hf=args.publish_hf,
        publish_to_repo=CHECKPOINT_REPO_ID,
        device=device,
        num_samples=args.num_samples,
        enable_grad_scaler=args.mix_precision,
    )
    logger.info("Training completed")

    return trained_model


if __name__ == "__main__":
    main()
