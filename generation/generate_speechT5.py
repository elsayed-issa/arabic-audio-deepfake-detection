
from pathlib import Path
import torch
import pandas as pd
from datasets import Dataset, Audio, load_dataset
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def dataset(tras, audios):
    dataset = pd.read_csv(tras, sep="\t")
    dataset["audio"] = audios + "/" + dataset["path"]
    hf_dataset = Dataset.from_pandas(dataset)
    hf_dataset = hf_dataset.cast_column('audio', Audio(sampling_rate=16000))
    return hf_dataset


processor = SpeechT5Processor.from_pretrained("MBZUAI/speecht5_tts_clartts_ar")
model = SpeechT5ForTextToSpeech.from_pretrained("MBZUAI/speecht5_tts_clartts_ar")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("herwoww/arabic_xvector_embeddings", split="validation")


def generate_speecht5(example, output_dir="speecht5_generated", text_column="sentence", audio_column="audio", speaker_id=105):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    text = example[text_column]
    
    original_filename = None
    audio_data = example.get(audio_column)
    
    if audio_data:
        if isinstance(audio_data, dict) and 'path' in audio_data:
            original_filename = Path(audio_data['path']).name
        elif isinstance(audio_data, str):
            original_filename = Path(audio_data).name
    
    output_filename = original_filename if original_filename else f"audio_{hash(text) % 10**8}.wav"
    output_path = Path(output_dir) / output_filename
    
    inputs = processor(text=text, return_tensors="pt")
    
    speaker_embedding = torch.tensor(embeddings_dataset[speaker_id]["speaker_embeddings"]).unsqueeze(0)
    
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    
    sf.write(str(output_path), speech.detach().cpu().numpy(), samplerate=16000)
    
    example['generated_audio_path'] = str(output_path)
    return example

if __name__ == "__main__":
    tras = 'data.tsv'
    audios = "PATH/wavs"

    d = dataset(tras, audios)

    generated_dataset = d.map(
        generate_speecht5,
        fn_kwargs={
            "output_dir": "speecht5_output",
            "text_column": "sentence",
            "audio_column": "audio",
            "speaker_id": 105  # speaker 105 for all samples
        },
        num_proc=1
    )