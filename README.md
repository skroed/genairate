# 🔉 genairate 🔉
A repository to create the content for a generative AI based radio station.
<!-- ![genairate](https://github.com/skroed/genairate/assets/83976953/9e3e07bd-2e2b-4337-bb0f-4955a20302a3 | width=100) -->
<img src="https://github.com/skroed/genairate/assets/83976953/9e3e07bd-2e2b-4337-bb0f-4955a20302a3" width="600">

## how it works
I used LLMs to create song descriptions for various music kinds. Afterwards the moderations between the different songs are also created with LLMs and include a random invented topic like `tech`, `gossip` or `weather`.
Finally the the songs are rendered by generative music models and the moderations are transferred into audio with text to speech models. The results shown here were obtained with the following models: \
`song descriptions`: OpenAI GPT4 \
`moderations`: OpenAI GPT4 \
`song generation`: [musicgen-large](https://huggingface.co/facebook/musicgen-large) \
`text to speech`: [OpenAI tts-1-hd](https://platform.openai.com/docs/guides/text-to-speech)

other supported models are `audioldm2`, `bark`, `vits` for audio and `Llama` as LLM backend model.
### Requirements
- OpenAI access via their [API](https://openai.com/blog/openai-api) and sufficient funds.
- At least one Nvidia Tesla T6 (16GB of memory) or a deployment of my [huggingface-repo](https://huggingface.co/skroed/audiocraft_handler).

## tune in


https://github.com/skroed/genairate/assets/83976953/6bceb7f5-78ad-4b5b-afa9-34b9e2af6811




## change Log
- 2023-10-29: Initial working version with added audio examples

## TODO
- [x] support better speech synthesis model
- [x] allow use LDM2 for music generation
- [ ] allow stereo sound musicgen generation
- [x] support musicgen > 30s sounds by continuation
- [ ] make moderation and song roles configurable
- [ ] make moderation topics configurable
- [ ] use links to news websites as links for moderation
- [x] intro and exit
- [ ] create interview style moderations
- [x] unify audio conversion in model

## how to install
You can install the repository via checkout and then
```shell
poetry install --sync
```
This command should install the package together with the genairate entry point.
This configuration should work on MacOS.
## openAI and HF Tokens
Make sure that you have a token for OpenAI registered under `OPENAI_API_KEY`. For using `Llama` models via huggingface a PRO membership is necessary.
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
genairate get-audio --config-file-path configs/audio_remote_custom_audiocraft.yaml --moderation-file-path my/moderation/path --audio-output-path audio/out/path
```
### configuration
Unless you have a super powerful machine it is recommended to use remote configurations for LLMs and generative audio. For LLMs the best performance is reached with OpenAI.
The following repositories can be used for [Bark](https://huggingface.co/skroed/bark) and [musicgen-large](https://huggingface.co/skroed/audiocraft_handler) to create inference endpoints.
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
#### [example song 1](https://github.com/skroed/genairate/blob/main/examples/mystic_drift.mp3?raw=true)
`artist`: Neon Geometrics \
`description`: Deep House genre blended harmoniously with driving rhythms and unique
  acoustic iterations clocking in at 120 BPM. \
`song_title`: Mystic Drift

#### [example song 2](https://github.com/skroed/genairate/blob/main/examples/sun_arrayvisions_2.0.mp3?raw=true)
`artist`: Marble dust \
`description`: Upbeat electronic song pulsating merrinebarty with earelementingly compiled
  catchy melodial tactics wrapped up in captivating summer sided conversations atop
  an upbeat fragmentolo funded sonic constituent - 110 BPM electronic explosive happiness \
`song_title`: Sun arrayvisions 2.0

#### [example moderation](https://github.com/skroed/genairate/blob/main/examples/burning_circuit_to_sun_arrayvisions_2.0.mp3?raw=true)
Song: burning circuit to sun arrayvisions 2.0

#### more examples
Feel free to browse through all the [examples](examples) provided with the repository.
The folder contains both songs and moderations between the songs. The full audio of ~15 minutes of genairate radio is located
in [examples/merged_audio.mp3](examples/merged_audio.mp3).
