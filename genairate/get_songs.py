import logging
from pathlib import Path
from typing import List

import click
import yaml

from genairate.language_models import get_language_model
from genairate.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(name='get-songs')
@click.option(
    '--config-file-paths',
    '-c',
    required=True,
    type=click.Path(exists=True),
    multiple=True,
    help='Paths to config file.',
)
@click.option(
    '--song-file-path',
    required=True,
    type=click.Path(exists=False),
    help='Path to dump song file yaml that contains the song names and descriptions',
)
def get_songs(
    config_file_paths: List[str] = None,
    song_file_path: str = None,
):
    """
    Function to get songs from the language model.

    Args:
        config_file_paths (List[str], optional): The song configuration file.
        song_file_path (str, optional): Where to store the songs.
    """
    song_file_path = Path(song_file_path)
    song_file_path.mkdir(parents=True, exist_ok=True)

    for config_file_path in config_file_paths:
        logger.info(f'Getting songs for {config_file_path}.')
        # Check configuration file
        config = load_config(config_file_path)
        song_prompt = config['prompt']
        logger.info('Loaded config file.')

        language_model = get_language_model(config)
        logger.info('Loaded language model.')

        logger.info(f"Getting {config['n_examples']} songs.")
        for _ in range(config['n_examples']):
            try:
                song_title, artist, description = language_model.get(
                    song_prompt,
                )

                song_title = song_title.replace('"', '')
                description = description.replace('"', '')
                artist = artist.replace('"', '')
                song_dict = {
                    'song_title': song_title,
                    'artist': artist,
                    'description': description,
                }
                with open(
                    Path(song_file_path)
                    / f"{song_title.lower().replace(' ', '_')}.yaml",
                    'w',
                ) as f:
                    yaml.dump(song_dict, f)
                logger.info(f'Saved song: {song_title}')

            except Exception as e:
                logger.error(e)
                continue
