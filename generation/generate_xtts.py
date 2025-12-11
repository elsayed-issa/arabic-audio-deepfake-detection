"""
This is for voice cloning
coqui/XTTS-v2

"""
from TTS.api import TTS
from pathlib import Path
import soundfile as sf
from datasets import Audio, Dataset
import pandas as pd

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

tras = 'dataset.csv'
audios = 'PATH'

dataset = pd.read_csv(tras)
dataset["audio"] = audios + "/" + dataset["path"]
hf_dataset = Dataset.from_pandas(dataset)
hf_dataset = hf_dataset.cast_column('audio', Audio(sampling_rate=16000))


def gen_xtts(example, output_dir="cloned_audio", text_column="sentence", audio_column="audio", speaker_column="speaker_id"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    text = example[text_column]
    speaker_wav = example[audio_column]
    
    original_filename = None
    
    if isinstance(speaker_wav, dict) and 'path' in speaker_wav:
        original_filename = Path(speaker_wav['path']).name
        speaker_wav = speaker_wav['path']
    elif isinstance(speaker_wav, dict) and 'array' in speaker_wav:
        temp_path = f"{output_dir}/temp_reference.wav"
        sf.write(temp_path, speaker_wav['array'], speaker_wav['sampling_rate'])
        speaker_wav = temp_path
        
        if 'path' in example.get(audio_column, {}):
            original_filename = Path(example[audio_column]['path']).name
    elif isinstance(speaker_wav, str):
        original_filename = Path(speaker_wav).name
    
    if original_filename:
        output_filename = original_filename
    else:
        speaker_id = example.get(speaker_column, "unknown")
        output_filename = f"{speaker_id}_{hash(text) % 10**8}.wav"
    
    output_path = Path(output_dir) / output_filename
    
    tts.tts_to_file(text=text, file_path=str(output_path), speaker_wav=speaker_wav, language="ar")
    
    example['cloned_audio_path'] = str(output_path)
    
    return example

# run
d = hf_dataset.map(gen_xtts, num_proc=1) 
