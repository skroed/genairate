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
            every tuple has the format (previous_song, moderation).
        output_path (Path): where to save the final audio file.
    """
    for idx, (prev_song, mod) in enumerate(full_audio):
        prev_song = prev_song.fade_in(1000).fade_out(1000)
        if idx == 0:
            full_audio = prev_song + mod

        else:
            full_audio += prev_song + mod

    logger.info(
        f'Total duration of audio: {full_audio.duration_seconds} seconds.',
    )
    logger.info(f'Saving audio to {output_path}.')
    full_audio.export(output_path / 'merged_audio.mp3', format='mp3')


def create_sentence_chunks(sentences: list[str]) -> list[str]:
    """
    Merge sentences together such that are not longer sections than 30 words.

    Args:
        sentences (list[str]): A list of sentences.

    Returns:
        list[str]: A list of sentences merged together.
    """
    sentence_chunks = []
    sentence_chunk = ''
    for sentence in sentences:
        lenght_sentence = len(sentence.split(' '))
        if len(sentence_chunk.split(' ')) + lenght_sentence < 30:
            sentence_chunk += sentence + ' '
        else:
            sentence_chunks.append(sentence_chunk)
            sentence_chunk = sentence + ' '

    sentence_chunks.append(sentence_chunk)

    return sentence_chunks
