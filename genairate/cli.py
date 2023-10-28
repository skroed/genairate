import click
from genairate.get_audio import get_audio
from genairate.get_moderation import get_moderation
from genairate.get_songs import get_songs


@click.group()
def genairate_cli():
    pass


genairate_cli.add_command(get_audio)
genairate_cli.add_command(get_moderation)
genairate_cli.add_command(get_songs)

if __name__ == '__main__':
    genairate_cli()
