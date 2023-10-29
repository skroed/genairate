# genairate 🔉
A repository to create the content for a generative AI based radio station.
<!-- ![genairate](https://github.com/skroed/genairate/assets/83976953/9e3e07bd-2e2b-4337-bb0f-4955a20302a3 | width=100) -->
<img src="https://github.com/skroed/genairate/assets/83976953/9e3e07bd-2e2b-4337-bb0f-4955a20302a3" width="600">

## how it works
I used LLMs to create song descriptions for various music kinds. Afterwards the moderations between the different songs are also created with LLMs and include a random invented topic like `tech`, `gossip` or `weather`.
Finally the the songs are rendered by generative music models and the moderations are transferred into audio with text to speech models. The results shown here were obtained with the following models: \
`song descriptions`: OpenAI GPT4 \
`moderations`: OpenAI GPT4 \
`song generation`: [facebook musicgen medium](https://huggingface.co/facebook/musicgen-medium) \
`text to speech`: [suno bark](https://github.com/suno-ai/bark)
## tune in


https://github.com/skroed/genairate/assets/83976953/6336bc7f-e57c-4467-a305-b6cd1b1d6d23


## change Log
- 2023-10-29: Initial working version with added audio examples

## TODO
- [ ] support better speech synthesis
- [ ] make moderation and song roles configurable
- [x] unify audio conversion in model

## how to install
You can install the repository via checkout and then
```shell
poetry install --sync
```
This command should install the package together with the genairate entry point.
This configuration should work on MacOS.
## openAI and HF Tokens
Make sure that you have a token for OpenAI registered under OPENAI_API_KEY. For using Llama models via huggingface a PRO membership is necessary.
## how to use
In order to use `genairate` to produce your own generative AI radio station 3 steps are needed:
1. Use the `get-songs` entrypoint to create a number of songs. This means creating a semantic descriptions for the songs and not the audio itself, which will be done in step 3. For some ideas check the configurations, for instance `configs/electronic_music`. A typical command would look like:
```shell
genairate get-songs --config-file-path configs/classic_music.yaml --song-file-path my/putput/path
```
2. After the songs have been created we need to create the text for moderations between the songs. The configuration for the moderation is defined in the precondition prompt in `language_models.py`. Basically we request a humorous moderation between the songs but feel free to adjust if needed. For this we will use the `get-moderation` entrypoint:
```shell
genairate get-moderation --config-file-path configs/moderation.yaml --song-file-path my/song/path --moderation-file-path my/moderation/path
```
3. Last but not least we need to convert everything into audio. This is the most involved step and requires the longest time by far.
```shell
genairate get-audio --config-file-path configs/audio_remote.yaml --moderation-file-path my/moderation/path --audio-output-path audio/out/path
```
### configuration
Unless you have a super powerful machine it is recommended to use remote configurations for LLMs and generative audio. For LLMs the best performance is reached with OpenAI.
The following repositories can be used for [Bark](https://huggingface.co/skroed/bark) and [musicgen-medium](https://huggingface.co/skroed/musicgen-medium) to create inference endpoints.
### examples songs description
`song_title`: Dance with Joy \
`artist`:  Bubblegum Beats \
`description`: This is a lively pop song with an infectious chorus, upbeat rhythm, and
  is written in a major key.
### examples moderation script
Example of a moderation between two songs: \
`moderation`: Well, folks, isn't "Dance with Joy" by Bubblegum Beats just the best song
  to lighten up your mood and get your feet tapping? Speaking of dancing feet, let
  me share some fantastic news from the world of sports. Rumor has it, an underground
  group of professional figure skaters have begun translating popular pop songs into
  skating routines! They've even figured out how to create sequin stars that skid
  across the ice when they power slide! Just imagine the sparkles, folks! But alright,
  time to slow things down a bit, as we pivot from our Dance with Joy to our next
  beautiful track. Relax your mind to the soothing melodies of "Dance of the Misty
  Moon" by Arabella Steinway. Enjoy!
### example audio
#### [example song 1](https://github.com/skroed/genairate/blob/main/examples/sunset_melody.mp3?raw=true)
`artist`: DJ Starfire \
`description`: An uplifting EDM track with a pulsing rhythm and memorable melody. \
`song_title`: Sunset Melody

<audio src="https://github.com/skroed/genairate/blob/main/examples/sunset_melody.mp3?raw=true" controls="controls">
</audio>

#### [example song 2](https://github.com/skroed/genairate/blob/main/examples/spiritual_reverie.mp3?raw=true)
`artist`: Cordelia Royal \
`description`: An enchanting piece of classical music, played on a grand piano, unveiling
  a tranquil mood, gentle progression, and invoking contemplative tranquility. \
`song_title`: Spiritual Reverie

<audio src="https://github.com/skroed/genairate/blob/main/examples/spiritual_reverie.mp3?raw=true" controls="controls">
</audio>

#### [example moderation](https://github.com/skroed/genairate/blob/main/examples/sunshine_morning_to_bass_enigma.mp3?raw=true)
Song: Sunshine morning to Song: Bass enigma

<audio src="https://github.com/skroed/genairate/blob/main/examples/sunshine_morning_to_bass_enigma.mp3?raw=true" controls="controls">
</audio>

#### more examples
Feel free to browse through all the [examples](examples) provided with the repository.
The folder contains both songs and moderations between the songs. The full audio of ~15 minutes of genairate radio is located
in [examples/merged_audio.mp3](examples/merged_audio.mp3).
