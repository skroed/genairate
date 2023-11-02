import logging
import random
from pathlib import Path

import click
import yaml

from genairate.language_models import get_language_model
from genairate.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(name="get-moderation")
@click.option(
    "--config-file-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to config file.",
)
@click.option(
    "--song-file-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to song file yaml that contains the descriptions",
)
@click.option(
    "--moderation-file-path",
    type=click.Path(writable=True),
    help="Path to output directory.",
    required=True,
)
def get_moderation(
    config_file_path: str = None,
    song_file_path: str = None,
    moderation_file_path: str = None,
):
    """Function to get moderation from the language model.

    Args:
        config_file_path (str, optional): The moderation configuration file.
        song_file_path (str, optional): The path to all the song yaml files.
        moderation_file_path (str, optional): Where to store moderations.
    """
    # Check configuration file
    config = load_config(config_file_path)
    logger.info("Loaded config file.")

    moderation_file_path = Path(moderation_file_path)
    moderation_file_path.mkdir(parents=True, exist_ok=True)

    # Load all the song titles and descriptions from the song file
    song_files = list(Path(song_file_path).rglob("*.yaml"))
    random.shuffle(song_files)

    logger.info(f"Found {len(song_files)} song files.")
    logger.info(f"Getting {config['n_examples']} moderations.")

    previous = load_config(song_files.pop())
    for idx_ex in range(config["n_examples"]):
        next = load_config(song_files.pop())
        language_model = get_language_model(config)
        try:
            prompt = (
                f"previous: {previous['song_title']}|||{previous['artist']}|||{previous['description']}\n"
                f"next: {next['song_title']}|||{next['artist']}|||{next['description']}"
            )

            moderation = language_model.get(prompt)[0]

            with open(
                Path(moderation_file_path)
                / f"idx_{idx_ex:02d}_{previous['song_title'].lower().replace(' ', '_')}_to_{next['song_title'].lower().replace(' ', '_')}.yaml",
                "w",
            ) as f:
                yaml.dump(
                    {
                        "moderation": moderation,
                        "title_previous": previous["song_title"],
                        "artist_previous": previous["artist"],
                        "description_previous": previous["description"],
                        "title_next": next["song_title"],
                        "artist_next": next["artist"],
                        "description_next": next["description"],
                    },
                    f,
                )
            logger.info(
                f"Got moderation for {previous['song_title']} to {next['song_title']}",
            )
            # Set the next song as the previous song
            previous = next
        except Exception as e:
            logger.error(e)
            continue
