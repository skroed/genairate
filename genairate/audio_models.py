import logging
import random
from functools import reduce
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import numpy as np
import scipy
import torch
from audiocraft.models import MusicGen
from huggingface_hub import InferenceClient
from nltk import sent_tokenize
from pydub import AudioSegment
from transformers import AutoModel, AutoProcessor, AutoTokenizer, VitsModel

from genairate.utils import create_sentence_chunks

logger = logging.getLogger(__name__)

GLOBAL_SAMPLE_RATE = 22050


class AudioModel(object):
    """
    Abstract class for audio models.
    """

    def __init__(
        self,
        name: None,
        local: bool = False,
        inference_params: dict = None,
    ):
        """Initializes the AudioModel class.

        Args:
            name (None): the name of the model.
            local (bool, optional): If the model is running locally.
              Defaults to None.
            inference_params (dict, optional): Additional params to
                pass for inference. Defaults to None.
        """
        self.name = name
        self.local = local

        if not inference_params:
            inference_params = {}

        self.inference_params = inference_params

    def get(self, prompt: str) -> AudioSegment:
        """
        Abstract method for getting audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.


        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        raise NotImplementedError('AudioModel is an abstract class.')


class BarkAudioModel(AudioModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local: bool = True,
        audio_type: str = 'tts',
        inference_params: dict = None,
    ):
        """
        Initializes the BarkAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
            local (bool, optional): If the bark model should run locally.
                Defaults to True.
            audio_type (str, optional): The type of the audio.
                 Defaults to "tts".
            inference_params (dict, optional): Additional inference params.
                Defaults to None.

        """
        super().__init__(
            name='Bark',
            local=local,
            inference_params=inference_params,
        )

        if audio_type.lower() != 'tts':
            raise NotImplementedError(
                'BarkAudioModel only supports TTS at the moment.',
            )
        # Select a random speaker
        speakers = [f'v2/en_speaker_{speaker}' for speaker in range(10)]
        random.shuffle(speakers)
        self.speaker = speakers.pop()

        if local:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        else:
            self.model = InferenceClient(model=model_name_or_path)
            if '.huggingface.cloud' in model_name_or_path:
                self.custom_endpoint = True
                logger.info(
                    f'Using custom endpoint {model_name_or_path} for bark',
                )
            else:
                self.custom_endpoint = False

    def get(self, prompt: str) -> AudioSegment:
        """
        Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        sentences = sent_tokenize(prompt)
        sentences = create_sentence_chunks(sentences)

        audio_arrays = []
        for sentence in sentences:
            logger.info(f'Generating audio for sentence: {sentence}')
            if self.local:
                audio = (
                    self.model.generate(
                        **self.processor(
                            text=[sentence],
                            return_tensors='pt',
                            voice_preset=self.speaker,
                        ),
                        do_sample=False,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                audio = AudioSegment(
                    audio.tobytes(),
                    frame_rate=24000,
                    sample_width=audio.dtype.itemsize,
                    channels=1,
                )

            else:
                if self.custom_endpoint:
                    audio_raw = self.model.post(
                        json={
                            'inputs': sentence,
                            'voice_preset': self.speaker,
                        },
                    )
                    audio = (
                        np.array(
                            eval(audio_raw)[0]['generated_audio'][0],
                        )
                        .squeeze()
                        .astype(np.float32)
                    )
                    audio = librosa.resample(
                        audio,
                        orig_sr=24000,
                        target_sr=GLOBAL_SAMPLE_RATE,
                    )
                    with TemporaryDirectory() as temp_dir:
                        file_path = Path(temp_dir) / Path('tmp_audio.wav')
                        scipy.io.wavfile.write(
                            file_path,
                            GLOBAL_SAMPLE_RATE,
                            audio,
                        )

                        audio = AudioSegment.from_file(file_path)
                else:
                    logger.warning(
                        'speaker is not being used in remote setting. Use custom endpoint for this.',
                    )
                    audio_raw = self.model.post(
                        json={
                            'inputs': sentence,
                        },
                    )
                    with TemporaryDirectory() as temp_dir:
                        file_path = Path(temp_dir) / Path('tmp_audio.flac')
                        file_path.write_bytes(audio_raw)
                        audio = AudioSegment.from_file(file_path, 'flac')
            audio_arrays.append(audio)

        return reduce(lambda x, y: x + y, audio_arrays)


class AudiocraftAudioModel(AudioModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local: bool = True,
        audio_type: str = 'music',
        duration: int = 25,
        inference_params: dict = None,
    ):
        """
        Initializes the AudiocraftAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
            local (bool, optional): If the model is running locally.
              Defaults to True.
            audio_type (str, optional): The audio type. Defaults to "music".
            duration (int, optional): How long the pieve should be.
                Defaults to 25 seconds.
            inference_params (dict, optional): additional inference params.
                Defaults to None.

        """
        super().__init__(
            name='Audiocraft',
            local=local,
            inference_params=inference_params,
        )

        if audio_type.lower() != 'music':
            raise NotImplementedError(
                'AudiocraftAudioModel only supports music at the moment.',
            )

        if local:
            self.model = MusicGen.get_pretrained(model_name_or_path).to(device)
            self.model.set_generation_params(duration=duration)
        else:
            # Duration is 30s by default
            self.model = InferenceClient(model=model_name_or_path)
            if '.huggingface.cloud' in model_name_or_path:
                logger.info(
                    f'Using custom endpoint {model_name_or_path} for musicgen',
                )
                self.custom_endpoint = True
            else:
                self.custom_endpoint = False

    def get(self, prompt: str) -> AudioSegment:
        """
        Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        if self.local:
            audio = self.model.generate(prompt).cpu().numpy()
            audio = AudioSegment(
                audio.tobytes(),
                frame_rate=self.model.sample_rate,
                sample_width=audio.dtype.itemsize,
                channels=1,
            )
            return audio
        else:
            if self.custom_endpoint:
                audio_raw = self.model.post(
                    json={
                        'inputs': prompt,
                    },
                )
                audio = (
                    np.array(
                        eval(audio_raw)[0]['generated_audio'][0],
                    )
                    .squeeze()
                    .astype(np.float32)
                )
                audio = librosa.resample(
                    audio,
                    orig_sr=32000,
                    target_sr=GLOBAL_SAMPLE_RATE,
                )
                with TemporaryDirectory() as temp_dir:
                    file_path = Path(temp_dir) / Path('tmp_audio.wav')
                    scipy.io.wavfile.write(
                        file_path,
                        GLOBAL_SAMPLE_RATE,
                        audio,
                    )
                    audio = AudioSegment.from_file(file_path)

            else:
                audio_raw = self.model.text_to_speech(
                    prompt,
                    **self.inference_params,
                )
                with TemporaryDirectory() as temp_dir:
                    file_path = Path(temp_dir) / Path('tmp_audio.flac')
                    file_path.write_bytes(audio_raw)
                    audio = AudioSegment.from_file(file_path, 'flac')
            return audio


class VitsAudioModel(AudioModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local: bool = True,
        audio_type: str = 'tts',
        inference_params: dict = None,
    ):
        """
        Initializes the VitsAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
            local (bool, optional): If its running locally.
                Defaults to True.
            audio_type (str, optional): Which audio type. Defaults to "tts".
            inference_params (dict, optional): additional inference params.
                Defaults to None.

        """
        super().__init__(
            name='Vits',
            local=local,
            inference_params=inference_params,
        )

        if audio_type.lower() != 'tts':
            raise NotImplementedError(
                'VitsAudioModel only supports TTS at the moment.',
            )

        if local:
            self.model = VitsModel.from_pretrained(
                model_name_or_path,
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = InferenceClient(model=model_name_or_path)

    def get(self, prompt: str) -> AudioSegment:
        """
        Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        sentences = sent_tokenize(prompt)

        audio_arrays = []
        for sentence in sentences:
            logger.info(f'Generating audio for sentence: {sentence}')
            if self.local:
                inputs = self.tokenizer(sentence, return_tensors='pt')
                with torch.no_grad():
                    audio = (
                        self.model(
                            **inputs,
                        )
                        .waveform.cpu()
                        .numpy()
                        .squeeze()
                    )
                # NOTE: Fix write to file with scipy, did not work with pydub
                with TemporaryDirectory() as temp_dir:
                    file_path = Path(temp_dir) / Path('tmp_audio.wav')
                    scipy.io.wavfile.write(
                        file_path,
                        self.model.config.sampling_rate,
                        audio,
                    )
                    audio = AudioSegment.from_file(file_path, 'wav')

            else:
                audio_raw = self.model.post(
                    json={
                        'inputs': sentence,
                    },
                )
                with TemporaryDirectory() as temp_dir:
                    file_path = Path(temp_dir) / Path('tmp_audio.flac')
                    file_path.write_bytes(audio_raw)
                    audio = AudioSegment.from_file(file_path, 'flac')
            audio_arrays.append(audio)

        return reduce(lambda x, y: x + y, audio_arrays)


def get_audio_model(config: dict) -> AudioModel:
    """
    Helper function to get an audio model from the config.
    Currently supports Bark, Audiocraft, and Vits.

    Args:
        config (dict): The config for the audio model.

    Returns:
        AudioModel: The initialized audio model.
    """
    if config['name'].lower() == 'bark':
        return BarkAudioModel(
            model_name_or_path=config['model_name_or_path'],
            device=config.get('device', 'cpu'),
            local=config['local'],
            audio_type=config['audio_type'],
        )
    elif config['name'].lower() == 'audiocraft':
        return AudiocraftAudioModel(
            model_name_or_path=config['model_name_or_path'],
            device=config.get('device', 'cpu'),
            local=config['local'],
            audio_type=config['audio_type'],
            duration=config.get('duration', None),
        )
    elif config['name'].lower() == 'vits':
        return VitsAudioModel(
            model_name_or_path=config['model_name_or_path'],
            device=config.get('device', 'cpu'),
            local=config['local'],
            audio_type=config['audio_type'],
        )
    else:
        raise ValueError(f"Model {config['name']} not supported.")
