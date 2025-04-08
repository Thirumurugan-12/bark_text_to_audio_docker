import os

# --- Configuration ---
PROMPT_DIR = "./prompts"
GENERATED_AUDIO_DIR = "./generated_audio"
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)

# Constants for audio generation
DEFAULT_AUDIO_SAMPLE_RATE = 24000
DEFAULT_DURATION = 3
DEFAULT_FREQ = 440
