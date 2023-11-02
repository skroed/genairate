import logging
from pathlib import Path

import click
from pydub import AudioSegment

from genairate.audio_models import get_audio_model
from genairate.utils import load_config, merge_audio_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(name="get-audio")
@click.option(
    "--config-file-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to config file.",
)
@click.option(
    "--moderation-file-path",
    type=click.Path(writable=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--audio-output-path",
    type=click.Path(writable=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--overwrite-songs",
    is_flag=True,
    show_default=True,
    default=False,
    required=False,
    help="Process all audio files into one.",
)
@click.option(
    "--overwrite-moderation",
    is_flag=True,
    show_default=True,
    default=False,
    required=False,
    help="Process all audio files into one.",
)
@click.option(
    "--merge-audio",
    is_flag=True,
    show_default=True,
    default=False,
    required=False,
    help="Process all audio files into one.",
)
def get_audio(
    config_file_path: Path = None,
    moderation_file_path: Path = None,
    audio_output_path: Path = None,
    overwrite_songs: bool = None,
    overwrite_moderation: bool = None,
    merge_audio: bool = None,
):
    """Function to get audio from the moderation and song models. As input we
    need a moderation file and a config file that specifies which models to
    use.

    Args:
        config_file_path (Path, optional): The configuration file.
        moderation_file_path (Path, optional): The moderation file yaml
            that contains all the information about the previous and
            next songs as well as the moderation.
        audio_output_path (Path, optional): Where to save the audio files.
        overwrite_songs (bool, optional): If all songs should be
            overwritten.
        overwrite_moderation (bool, optional): If all moderation files
            should be overwritten.
        merge_audio (bool, optional): If all audio files should
            be merged into one.
    """

    audio_output_path = Path(audio_output_path)
    audio_output_path.mkdir(parents=True, exist_ok=True)

    # Check configuration file
    config = load_config(config_file_path)

    logger.info(
        f"Loading model moderation: {config['moderation']}"
        f" and song: {config['song']}",
    )
    moderation_model = get_audio_model(config["moderation"])
    song_model = get_audio_model(config["song"])

    config_list = sorted(list(Path(moderation_file_path).rglob("*.yaml")))
    full_audio = []
    logger.info(f"Getting audio for {config['n_examples']} examples.")
    for example_config_path in config_list[: config["n_examples"]]:
        example_config = load_config(example_config_path)
        previous_title = (
            example_config["title_previous"]
            .lower()
            .replace(
                " ",
                "_",
            )
        )
        next_title = example_config["title_next"].lower().replace(" ", "_")

        logger.info(f"Getting audio for {previous_title}")
        song_previous_file = audio_output_path / f"{previous_title}.mp3"
        if song_previous_file.exists() and not overwrite_songs:
            logger.info(f"Song {previous_title} already exists.")
            song_previous = AudioSegment.from_file(song_previous_file)
        else:
            song_previous = song_model.get(example_config["description_previous"])
            song_previous.export(song_previous_file, format="mp3")
            logger.info(f"Saved audio to {song_previous_file}")

        logger.info("Getting audio for moderation")
        output_path = audio_output_path / f"{previous_title}_to_{next_title}.mp3"
        if output_path.exists() and not overwrite_moderation:
            logger.info(f"Moderation {previous_title} to {next_title} already exists.")
            moderation = AudioSegment.from_file(output_path)
        else:
            moderation = moderation_model.get(example_config["moderation"])
            moderation.export(output_path, format="mp3")
            logger.info(f"Saved audio to {output_path}")

        full_audio.append((song_previous, moderation))

    logger.info("Done processing all examples.")
    if merge_audio:
        merge_audio_files(full_audio, audio_output_path)
