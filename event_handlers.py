from typing import List, Tuple, Optional, Dict, Any
import traceback
import torch
import gradio as gr
import numpy as np
import time
import os
import re
import wave
import contextlib
import logging
import pandas as pd
import gc

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from core.data_model import AudioFile
from core.bark.voice_clone import create_bark_prompt
from core.bark.generate_audio import generate_audio
from core.data_model import BarkPrompt, BarkGenerationConfig
from core.utils.audio import save_audio_file
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# return list of available devices and the best device to be used as default for all inference
def get_available_torch_devices() -> Tuple[List[str], str]:
    devices = ["cpu"]
    best_device = "cpu"
    if torch.mps.is_available():
        devices.append("mps")
        best_device = "mps"
    if torch.cuda.is_available():
        devices.append("cuda")
        best_device = "cuda"

    return devices, best_device


# --- Helper Functions ---
# (Keep get_wav_duration, load_existing_audio, get_safe_filename,
#  generate_sine_wave, save_audio, parse_text_prompts, get_available_prompts,
#  create_audio_prompt as they are, they are mostly backend logic)
def get_wav_duration(filepath):
    """Gets the duration of a WAV file in seconds."""
    try:
        with contextlib.closing(wave.open(filepath, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            if rate > 0:
                duration = frames / float(rate)
                return duration
            else:
                logger.info(f"Warning: Framerate is 0 for {filepath}")
                return 0
    except wave.Error as e:
        logger.info(f"Warning: Could not read wave file header for {filepath}: {e}")
        return 0
    except Exception as e:
        logger.info(f"Warning: Could not get duration for {filepath}: {e}")
        return 0


def load_existing_audio() -> List[Dict[str, Any]]:
    """Scans the audio directory and loads metadata for existing WAV files."""
    logger.info("\n--- Loading Existing Audio Files ---")
    existing_files_metadata = []
    if not os.path.isdir(GENERATED_AUDIO_DIR):
        logger.info(f"Directory not found: {GENERATED_AUDIO_DIR}")
        return []

    try:
        for filename in os.listdir(GENERATED_AUDIO_DIR):
            if filename.lower().endswith(".wav"):
                filepath = os.path.join(GENERATED_AUDIO_DIR, filename)
                if not os.path.isfile(filepath):
                    continue

                match = re.match(r"^(.*)_(\d{13})\.wav$", filename)
                text_guess = "Unknown (from filename)"
                timestamp_ms = 0
                if match:
                    text_guess = match.group(1).replace("_", " ")
                    try:
                        timestamp_ms = int(match.group(2))
                    except ValueError:
                        timestamp_ms = 0
                else:
                    text_guess = os.path.splitext(filename)[0].replace("_", " ")

                timestamp_sec = (
                    timestamp_ms / 1000.0
                    if timestamp_ms > 0
                    else os.path.getmtime(filepath)
                )
                duration = get_wav_duration(filepath)

                metadata = {
                    "text": text_guess,
                    "path": filepath,
                    "duration": duration,
                    "timestamp": timestamp_sec,
                }
                existing_files_metadata.append(metadata)

    except Exception as e:
        logger.error(f"Error loading existing audio files: {e}")

    existing_files_metadata.sort(key=lambda x: x.get("timestamp", 0))
    logger.info(
        f"--- Finished Loading {len(existing_files_metadata)} Existing Files ---"
    )
    return existing_files_metadata


def get_safe_filename(base_name: str, extension: str, directory: str) -> str:
    """Creates a safe and unique filename in the target directory."""
    safe_base = "".join(
        c if c.isalnum() or c in ["_", "-"] else "_" for c in base_name[:50]
    )
    timestamp = int(time.time() * 1000)
    filename = f"{safe_base}_{timestamp}.{extension}"
    filepath = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(filepath):
        filename = f"{safe_base}_{timestamp}_{counter}.{extension}"
        filepath = os.path.join(directory, filename)
        counter += 1
    return filepath


def update_audio_list(
    newly_generated_metadata: List[Dict[str, Any]],
    current_audio_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Appends new metadata to the list and sorts it by timestamp."""
    logger.info(f"\n--- Updating Audio List State ---")
    if not isinstance(current_audio_list, list):
        logger.info("Current audio list was not a list, initializing.")
        current_audio_list = []
    if not isinstance(newly_generated_metadata, list):
        logger.info("Newly generated metadata is not a list, skipping update.")
        return current_audio_list

    logger.info(f"Current list size: {len(current_audio_list)}")
    logger.info(f"Adding {len(newly_generated_metadata)} new items.")
    updated_list = current_audio_list + newly_generated_metadata
    updated_list.sort(key=lambda x: x.get("timestamp", 0))
    logger.info(f"Updated list state size: {len(updated_list)}")
    logger.info("--- Finished Updating Audio List State ---")
    return updated_list


def format_audio_list_for_dataframe(audio_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts the list of audio metadata dicts into a pandas DataFrame for display."""
    logger.info("\n--- Formatting List for DataFrame ---")
    if not audio_list:
        logger.info("Audio list is empty, returning empty DataFrame.")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["File", "Prompt", "Duration (s)"])

    display_data = []
    for item in audio_list:
        filepath = item.get("path", "N/A")
        filename = os.path.basename(filepath) if filepath != "N/A" else "N/A"
        # Truncate long text prompts for display in the table
        text_prompt = item.get("text", "N/A")
        display_text = (
            (text_prompt[:75] + "...") if len(text_prompt) > 75 else text_prompt
        )
        duration = item.get("duration", 0)
        display_data.append(
            {
                "File": filename,
                "Prompt": display_text,
                "Duration (s)": f"{duration:.2f}" if duration else "N/A",
                # Store the full path implicitly by list order, not shown in df
            }
        )

    df = pd.DataFrame(display_data)
    logger.info(f"Created DataFrame with {len(df)} rows.")
    logger.info("--- Finished Formatting List for DataFrame ---")
    return df


def handle_row_selection(
    audio_list: List[Dict[str, Any]], evt: gr.SelectData
) -> Tuple[Optional[str], int]:
    """
    Handles the selection event from the DataFrame.
    Updates the audio player with the selected file's path.
    Returns the filepath and the selected index.
    """
    logger.info("\n--- Handling Row Selection ---")
    selected_index = evt.index[0] if evt.index else None  # Get row index
    logger.info(f"DataFrame row selected. Event data: {evt}")

    if selected_index is not None and 0 <= selected_index < len(audio_list):
        selected_item = audio_list[selected_index]
        filepath = selected_item.get("path")
        logger.info(f"Selected item at index {selected_index}: {selected_item}")
        if filepath and os.path.exists(filepath):
            logger.info(f"Updating audio player with: {filepath}")
            logger.info("--- Finished Handling Row Selection (Success) ---")
            return filepath, selected_index
        else:
            logger.info(f"File not found for selected item: {filepath}")
            gr.Warning(
                f"File not found for selected row: {os.path.basename(filepath or 'N/A')}"
            )
            logger.info("--- Finished Handling Row Selection (File Not Found) ---")
            return None, selected_index  # Keep index, but clear player
    else:
        logger.info("Invalid selection index or empty list.")
        logger.info("--- Finished Handling Row Selection (Invalid Index) ---")
        return None, -1  # Clear player and indicate no valid selection


def handle_delete_selected(
    selected_index: int, current_audio_list: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """
    Deletes the audio file corresponding to the selected index.
    Updates the main audio list state.
    Clears the selection index and audio player.
    """
    logger.info("\n--- Handling Delete Selected ---")
    logger.info(f"Attempting deletion for selected index: {selected_index}")

    if (
        selected_index is None
        or selected_index < 0
        or selected_index >= len(current_audio_list)
    ):
        gr.Warning("No valid audio selected for deletion.")
        logger.info("No valid index provided.")
        # Return current list, clear index, clear player
        return current_audio_list, -1, None

    item_to_delete = current_audio_list[selected_index]
    filepath_to_delete = item_to_delete.get("path")
    logger.info(f"Item to delete: {item_to_delete}")

    # Create the new list excluding the item
    # Corrected slicing logic: include elements before and after the index
    new_audio_list = (
        current_audio_list[:selected_index] + current_audio_list[selected_index + 1 :]
    )
    logger.info(f"New list size after filtering: {len(new_audio_list)}")

    # Try to delete the file from disk
    deletion_successful_on_disk = False
    try:
        if filepath_to_delete and os.path.exists(filepath_to_delete):
            os.remove(filepath_to_delete)
            logger.info(f"Successfully deleted file: {filepath_to_delete}")
            gr.Info(f"Deleted {os.path.basename(filepath_to_delete)}")
            deletion_successful_on_disk = True
        elif filepath_to_delete:
            logger.info(f"File not found for deletion: {filepath_to_delete}")
            gr.Warning("Audio entry removed from list, but file was not found on disk.")
            deletion_successful_on_disk = True  # Consider list update successful
        else:
            logger.info("Invalid filepath in selected item.")
            gr.Warning("Could not delete: Invalid file path associated with selection.")
            # Revert list change if filepath was invalid from the start? Or keep it removed?
            # Let's keep it removed from the list for consistency.
            deletion_successful_on_disk = True  # Treat as success for list update

    except OSError as e:
        logger.info(f"Error deleting file {filepath_to_delete}: {e}")
        traceback.logger.info_exc()
        gr.Error(f"Error deleting file: {e}")
        # If file deletion fails, we still return the updated list (item removed).
        # If you want to revert the list change on OS error, return `current_audio_list` here.

    logger.info("--- Finished Deleting Selected Item ---")
    # Return the updated list, clear the selected index, clear the audio player
    return new_audio_list, -1, None


def get_available_prompts() -> List[str]:
    """Loads available prompt file names."""
    try:
        prompts = [
            f
            for f in os.listdir(PROMPT_DIR)
            if os.path.isfile(os.path.join(PROMPT_DIR, f))
            and f.lower().endswith((".npz", ".npy", ".json"))
        ]

        if len(prompts) == 0:
            gr.Info("No prompts found.", duration=3)

        return ["None"] + prompts
    except Exception as e:
        logger.info(f"Error loading prompts: {e}")
        gr.Info(f"Error loading prompts {e}", duration=3, title="Error")
        return ["None"]


def update_available_prompts() -> gr.update:
    try:
        prompts = [
            f
            for f in os.listdir(PROMPT_DIR)
            if os.path.isfile(os.path.join(PROMPT_DIR, f))
            and f.lower().endswith((".npz", ".npy", ".json"))
        ]

        if len(prompts) == 0:
            gr.Info("No prompts found.", duration=3)

        return gr.update(choices=["None"] + prompts)
    except Exception as e:
        logger.info(f"Error loading prompts: {e}")
        gr.Info(f"Error loading prompts {e}", duration=3, title="Error")
        return gr.update()


def generate_batch_audio(
    text: str,
    semantic_temp: float,
    coarse_temp: float,
    fine_temp: float,
    manual_seed: int,
    model_type: str,
    inference_device: str,
    selected_prompt_name: Optional[str],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Generates audio (sine wave) for each line of text input.
    Returns metadata for generated files.
    """
    gc.collect()

    torch.manual_seed(manual_seed)
    if not text:
        gr.Warning("No valid text prompts provided.")
        return []

    generated_metadata = []

    bark_prompt = None
    if selected_prompt_name != "None":
        gr.Info("Loading audio prompt...")
        prompt_path = os.path.join(PROMPT_DIR, selected_prompt_name)
        bark_prompt = BarkPrompt.load_prompt(
            prompt_path, torch.device(inference_device)
        )

    generation_config = BarkGenerationConfig(
        temperature=semantic_temp,
        generate_coarse_temperature=coarse_temp,
        generate_fine_temperature=fine_temp,
        use_small_model=True if model_type == "small" else False,
    )

    # split the text into sentences
    sentences = sent_tokenize(text)

    gr.Info("Generating Audio....", duration=120)
    waves = generate_audio(
        texts=sentences,
        prompt=bark_prompt,
        generation_config=generation_config,
        silent=True,
    )
    audio = np.concat(waves, axis=-1)

    output_filepath = get_safe_filename(text, "wav", GENERATED_AUDIO_DIR)
    save_audio_file(audio, DEFAULT_AUDIO_SAMPLE_RATE, output_filepath)
    duration_sec = audio.shape[0] // DEFAULT_AUDIO_SAMPLE_RATE
    metadata = {
        "text": text,
        "path": output_filepath,
        "duration": duration_sec,
        "timestamp": time.time(),
    }
    generated_metadata.append(metadata)
    gr.Info("Done!", duration=5)
    return generated_metadata


def create_audio_prompt(
    uploaded_audio_file: Optional[str],
    device: str,
    progress: gr.Progress = gr.Progress(),
) -> gr.update:
    """Processes an uploaded audio file to create a voice prompt file (stub)."""
    logger.info("\n--- Starting Prompt Creation ---")
    if uploaded_audio_file is None or len(uploaded_audio_file) == 0:
        gr.Warning("No audio file uploaded!")
        return gr.update()

    logger.info(f"Processing uploaded file: {uploaded_audio_file}")

    try:
        progress(0, desc="Starting prompt creation...")
        new_prompt_filename = None
        progress(0.2, desc="Extracting prompt features...")
        audio_file = AudioFile(audio_file_path=uploaded_audio_file, max_duration=10)
        prompt = create_bark_prompt(
            audio_file=audio_file, temperature=1, eos_p=0.2, device=torch.device(device)
        )

        progress(0.8, desc="Saving prompt file...")
        original_basename = os.path.splitext(os.path.basename(uploaded_audio_file))[0]
        prompt_filepath = get_safe_filename(original_basename, "json", PROMPT_DIR)
        new_prompt_filename = os.path.basename(prompt_filepath)

        ok = prompt.save_prompt(prompt_filepath)
        if ok:
            progress(1.0, desc="Prompt creation complete.")

        else:
            progress(1.0, desc="Error when saving prompt")

        new_choices = get_available_prompts()

        return gr.update(choices=new_choices, value=new_prompt_filename)

    except Exception as e:
        logger.info(f"Error creating prompt: {e}")
        gr.Error(f"Prompt creation failed: {e}")
        return f"Error creating prompt: {e}", gr.update()
