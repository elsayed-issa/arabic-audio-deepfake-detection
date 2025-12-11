from pathlib import Path
import torch
import pandas as pd
import numpy as np
import scipy.io.wavfile
from datasets import Dataset, Audio
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-ara", token="")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara", token="")

tras = 'PATH/dataset.csv'
audios = 'PATH'

dataset = pd.read_csv(tras)
dataset["audio"] = audios + "/" + dataset["path"]
hf_dataset = Dataset.from_pandas(dataset)
hf_dataset = hf_dataset.cast_column('audio', Audio(sampling_rate=16000))

def generate_speech_from_dataset(example, output_dir="mms-tts", text_column="sentence", audio_column="audio"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    text = example[text_column]
    
    original_filename = None
    audio_data = example.get(audio_column)
    
    if audio_data:
        if isinstance(audio_data, dict) and 'path' in audio_data:
            original_filename = Path(audio_data['path']).name
        elif isinstance(audio_data, str):
            original_filename = Path(audio_data).name
    
    if original_filename:
        output_filename = original_filename
    else:
        output_filename = f"audio_{hash(text) % 10**8}.wav"
    
    output_path = Path(output_dir) / output_filename
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    output = output.cpu()
    data_np = output.numpy()
    data_np_squeezed = np.squeeze(data_np)
    
    scipy.io.wavfile.write(
        str(output_path), 
        rate=model.config.sampling_rate, 
        data=data_np_squeezed
    )
    
    example['generated_audio_path'] = str(output_path)
    
    return example




hf_dataset = hf_dataset.map(generate_speech_from_dataset,fn_kwargs={"output_dir": "mms_generated_audio", "text_column": "sentence","audio_column": "audio"})

