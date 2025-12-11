
"""
installation info: https://speech.fish.audio/install/

"""
from pathlib import Path
import torch
import scipy.io.wavfile
import pandas as pd
from datasets import Dataset, Audio
import sys
import torchaudio
import soundfile as sf

tras = 'PATH/dataset.csv'
audios = 'PATH'

dataset = pd.read_csv(tras)
dataset["audio"] = audios + "/" + dataset["path"]
hf_dataset = Dataset.from_pandas(dataset)
hf_dataset = hf_dataset.cast_column('audio', Audio(sampling_rate=16000))


import sys
sys.path.insert(0, './fish_speech')  
from fish_speech.models.dac.inference import load_model


model = load_model(config_name="modded_dac_vq", checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth", device="cpu")

@torch.no_grad()  
def generate_openaudio_direct(example, output_dir="openaudio_generated", audio_column="audio"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    audio_data = example.get(audio_column)
    
    if isinstance(audio_data, dict) and 'path' in audio_data:
        ref_audio_path = audio_data['path']
        original_filename = Path(audio_data['path']).name
    elif isinstance(audio_data, str):
        ref_audio_path = audio_data
        original_filename = Path(audio_data).name
    else:
        raise ValueError("Invalid audio data format")
    
    output_filename = original_filename
    output_path = Path(output_dir) / output_filename
    
    audio, sr = torchaudio.load(str(ref_audio_path))
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, model.sample_rate)
    
    audios = audio[None].to("cpu")  
    
    audio_lengths = torch.tensor([audios.shape[2]], device="cpu", dtype=torch.long)
    indices, indices_lens = model.encode(audios, audio_lengths)
    
    if indices.ndim == 3:
        indices = indices[0]
    
    fake_audios, audio_lengths = model.decode(indices, indices_lens)
    
    fake_audio = fake_audios[0, 0].float().detach().cpu().numpy()
    sf.write(str(output_path), fake_audio, model.sample_rate)
    
    example['generated_audio_path'] = str(output_path)
    
    return example


generated_dataset = hf_dataset.map(generate_openaudio_direct, fn_kwargs={"output_dir": "openaudio_output","audio_column": "audio"},num_proc=1)