
import os
import time
import pathlib
import requests
from typing import Iterable, Literal, Optional, List
from resemble import Resemble
import pandas as pd
from datasets import Dataset

from dotenv import load_dotenv

load_dotenv()



AudioFmt = Literal["wav", "mp3"]

def dataset(path, gender):
    batch = pd.read_csv(path)
    selected_columns = ['name', 'trans']
    df_subset = batch[selected_columns]
    df_subset['gender'] = gender
    hf_dataset = Dataset.from_pandas(df_subset)
    return hf_dataset

def synthesis(
    texts: Iterable[str],
    *,
    project_uuid: str,
    voice_uuid: str,
    api_key: Optional[str] = None,   
    out_dir: str = "tts_outputs-trainxxx",
    base_name: str = "clip",
    output_format: AudioFmt = "wav",
    sample_rate: Optional[int] = None,
    include_timestamps: Optional[bool] = None,
    pause_sec: float = 0.0,
    overwrite: bool = True,
    timeout: int = 120,
    use_ssml_lang: bool = False,     
    lang_tag: str = "ar",           
) -> List[str]:
    
    api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set RESEMBLE_API_KEY or pass api_key.")
    Resemble.api_key(api_key)

    outp = pathlib.Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for idx, text in enumerate(texts, start=1):
        fname = f"{base_name}_{idx:04d}.{output_format}"
        fpath = outp / fname
        if fpath.exists() and not overwrite:
            written.append(str(fpath))
            continue

        body = f'<speak><lang xml:lang="{lang_tag}">{text}</lang></speak>' if use_ssml_lang else text

        resp = Resemble.v2.clips.create_sync(
            project_uuid,
            voice_uuid,
            body,                         
            is_archived=False,
            title=None,
            sample_rate=sample_rate,
            output_format=output_format,
            precision=None,               
            include_timestamps=include_timestamps,
            raw=None,
            # language=language_override, 
        )
        if not resp.get("success"):
            raise RuntimeError(f"Resemble create_sync failed: {resp}")

        item = resp["item"]
        audio_src = item.get("audio_src")
        if isinstance(audio_src, dict):
            audio_url = audio_src.get("url") or audio_src.get("signed_url") or audio_src.get("href")
        else:
            audio_url = audio_src

        if not audio_url:
            raise RuntimeError(f"No audio URL found in response item: {item}")

        r = requests.get(audio_url, timeout=timeout)
        r.raise_for_status()
        fpath.write_bytes(r.content)
        written.append(str(fpath))

        if pause_sec > 0:
            time.sleep(pause_sec)

    return written



# run
input_df = "PATH/file.csv"
data = dataset(input_df, "male")
data = data['trans']


files = synthesis(
    texts=data,
    project_uuid="4f93cfd3",
    voice_uuid="779842bf",
    out_dir="ClArTTS_train_200-fake",
    base_name="sample",
    output_format="wav",
    sample_rate=44100,
    include_timestamps=False,
    use_ssml_lang=True,  
    lang_tag="ar-SA",        
)
print(files)







