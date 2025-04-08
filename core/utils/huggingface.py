import logging
import sys
from typing import Optional, Literal
import os
import shutil
from zipfile import ZipFile
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

__all__ = ["download_dataset_from_hf", "upload_file_to_hf", "download_file_from_hf"]


def download_dataset_from_hf(
    repo_id: str,
    filename: str,
    dest_path: str,
    token: str = None,
    local_dir: str = "./downloads",
    remove_downloaded_file: bool = True,
) -> None:
    """
    Download a file from Hugging Face repository and unzip it to destination path

    Args:
        repo_id (str): Hugging Face repository ID (username/repo_name)
        filename (str): Name of the file to download from the repository
        dest_path (str): Destination path where contents will be unzipped
        token (str, optional): Hugging Face token, if None will prompt for login
    """
    # Ensure destination directory exists
    os.makedirs(dest_path, exist_ok=True)
    if token is None:
        logger.info("reading HF_TOKEN variable from environment")
        token = os.getenv("HF_TOKEN")

    # Download the file
    downloaded_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",  # Specify dataset repository
        local_dir=local_dir,  # Temporary download location
        token=token,
    )
    logger.info(f"Downloaded {filename} to {downloaded_file}")

    # Check if it's a zip file
    if filename.endswith(".zip"):
        # Extract the zip file
        with ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        logger.info(f"Unzipped contents to {dest_path}")

        # Clean up the downloaded zip file
        if remove_downloaded_file:
            os.remove(downloaded_file)
            logger.info(f"Cleaned up temporary file: {downloaded_file}")
    else:
        # If not a zip, just move the file
        final_path = os.path.join(dest_path, filename)
        shutil.move(downloaded_file, final_path)
        logger.info(f"Moved {filename} to {final_path}")


def download_file_from_hf(
    repo_id: str,
    repo_type: Literal["model", "dataset"],
    filename: str,
    dest_path: str,
    token: str = None,
) -> None:
    """
    Download a file from Hugging Face repository and unzip it to destination path

    Args:
        repo_id (str): Hugging Face repository ID (username/repo_name)
        repo_type: model for model repo, dataset for dataset repo
        filename (str): Name of the file to download from the repository
        dest_path (str): Destination path where contents will be unzipped
        token (str, optional): Hugging Face token, if None will prompt for login

    """
    # Ensure destination directory exists
    os.makedirs(dest_path, exist_ok=True)
    if token is None:
        logger.info("reading HF_TOKEN variable from environment")
        token = os.getenv("HF_TOKEN")

    # Download the file
    downloaded_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir="./downloads",  # Temporary download location
        token=token,
    )
    logger.info(f"Downloaded {filename} to {downloaded_file}")

    # Check if it's a zip file
    if filename.endswith(".zip"):
        # Extract the zip file
        with ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        logger.info(f"Unzipped contents to {dest_path}")

        # Clean up the downloaded zip file
        os.remove(downloaded_file)
        logger.info(f"Cleaned up temporary file: {downloaded_file}")
    else:
        # If not a zip, just move the file
        final_path = os.path.join(dest_path, filename)
        shutil.move(downloaded_file, final_path)
        logger.info(f"Moved {filename} to {final_path}")


def upload_file_to_hf(
    local_file_path: str,
    repo_id: str,
    repo_type: Literal["model", "dataset"],
    token: Optional[str] = None,
    path_in_repo: Optional[str] = None,
    commit_message: str = "Upload file",
) -> None:
    """
    Upload a file to Hugging Face hub.

    Args:
        local_file_path (str): Path to the local .pt checkpoint file
        repo_id (str): Repository ID in format "username/repo_name"
        repo_type (str, optional): Type of repository, either "model" or "dataset"
        token (str): Hugging Face authentication token. Read from environment variable HF_TOKEN if don't provide
        path_in_repo (str, optional): Destination path in the repository.
            Defaults to the filename from local_checkpoint_path
        commit_message (str, optional): Commit message for the upload

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        ValueError: If the repository ID is invalid
    """
    # Validate file exists
    if not os.path.isfile(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    # Use filename as default path_in_repo if not specified
    if path_in_repo is None:
        path_in_repo = Path(local_file_path).name

    if token is None:
        logger.info("reading HF_TOKEN variable from environment")
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise RuntimeError("not found HF_TOKEN variable from environment")

    upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
    )
    logger.info(f"Successfully uploaded {local_file_path} to {repo_id}/{path_in_repo}")
