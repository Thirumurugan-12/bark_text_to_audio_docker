import argparse
import logging
import os
from typing import Optional

from core.bark.generate_audio_semantic_dataset import (
    generate_wav_semantic_dataset,
    BarkGenerationConfig,
)
from core.utils import upload_file_to_hf, zip_folder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_dataset_args(args_list=None):
    """Parse arguments specific to dataset creation."""
    parser = argparse.ArgumentParser(description="Audio Semantic Dataset Creation")

    parser.add_argument(
        "--text-file",
        type=str,
        default="data/test_data.txt",
        help="Path to text file for dataset generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for processing (default: 1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset",
        help="Output directory for generated files (default: ./dataset)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per example (default: 256)",
    )
    parser.add_argument(
        "--use-small-model",
        action="store_true",
        help="Use small model for generation",
    )
    parser.add_argument(
        "--save-raw-audio",
        action="store_true",
        help="Store generated audio as .wav instead of .npz",
    )
    parser.add_argument(
        "--publish-hf",
        action="store_true",
        help="Publish dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repo ID to publish to",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        help="Path in HF repo",
        default=None,
    )
    parser.add_argument(
        "--silent", action="store_true", help="Suppress progress output"
    )

    return parser.parse_args(args_list)


def create_audio_semantic_dataset(
    text_file: str,
    output_dir: str = "./dataset",
    batch_size: int = 1,
    max_tokens: int = 256,
    use_small_model: bool = False,
    save_raw_audio: bool = False,
    publish_hf: bool = False,
    repo_id: Optional[str] = None,
    path_in_repo: Optional[str] = None,
    silent: bool = False,
) -> None:
    """Create audio semantic dataset from text file.

    Can be called directly with parameters or via command line using parse_dataset_args().

    Args:
        text_file: Path to input text file
        output_dir: Directory to save generated dataset
        batch_size: Batch size for processing
        max_tokens: Maximum tokens per example
        use_small_model: Whether to use small model
        save_raw_audio: Save as raw audio (.wav) instead of .npz
        publish_hf: Whether to publish to HuggingFace Hub
        repo_id: HF repo ID to publish to
        path_in_repo: Path in HF repo
        silent: Suppress progress output
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")

    logger.info(f"Starting dataset generation from {text_file}")
    generation_config = BarkGenerationConfig(
        temperature=None,
        generate_coarse_temperature=None,
        generate_fine_temperature=None,
        use_small_model=use_small_model,
    )

    generate_wav_semantic_dataset(
        text_file_path=text_file,
        generation_config=generation_config,
        batch_size=batch_size,
        save_path=output_dir,
        save_data_as_raw_audio=save_raw_audio,
        silent=silent,
    )
    logger.info("Dataset generation completed")

    if publish_hf and repo_id:
        logger.info("Publishing dataset to huggingface hub")
        zip_path = "./dataset.zip"
        success = zip_folder(output_dir, zip_path)
        if not success:
            raise RuntimeError(f"Unable to zip folder {output_dir}")
        upload_file_to_hf(zip_path, repo_id, "dataset", path_in_repo=path_in_repo)


if __name__ == "__main__":
    args = parse_dataset_args()
    create_audio_semantic_dataset(
        text_file=args.text_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        use_small_model=args.use_small_model,
        save_raw_audio=args.save_raw_audio,
        publish_hf=args.publish_hf,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        silent=args.silent,
    )
