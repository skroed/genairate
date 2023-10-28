import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_file_path: Path) -> dict:
    """
    Load a configuration file.

    Args:
        config_file_path (Path): The path to the configuration file.

    Returns:
        dict: The configuration file.
    """
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config


def merge_audio_files(full_audio: list[tuple], output_path: Path) -> None:
    """
    Merge all the audio files into one.

    Args:
        full_audio (list[tuple]): A list of tuples containing the audio files.
            every tuple has the format (previous_song, moderation, next_song).
        output_path (Path): where to save the final audio file.
    """
    for idx, (prev_song, mod, next_song) in enumerate(full_audio):
        prev_song = prev_song.fade_in(1000).fade_out(1000)
        next_song = next_song.fade_in(1000).fade_out(1000)
        if idx == 0:
            full_audio = prev_song + mod + next_song

        else:
            full_audio += prev_song + mod + next_song

    logger.info(
        f'Total duration of audio: {full_audio.duration_seconds} seconds.',
    )
    logger.info(f'Saving audio to {output_path}.')
    full_audio.export(output_path / 'merged_audio.mp3', format='mp3')
