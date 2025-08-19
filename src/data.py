"""
data.py
Module for loading and preprocessing the Free Spoken Digit Dataset (FSDD).
"""

import numpy as np
import pandas as pd
import librosa

def load_fsdd_parquet(train_path, test_path, sr=8000):
    """
    Loads FSDD from Parquet files. Returns train and test lists of (audio, label).
    """
    def decode_audio(row):
        import io
        import soundfile as sf
        # Parquet stores audio as dict with 'bytes' key containing WAV data
        wav_bytes = row['audio']['bytes']
        y, _ = sf.read(io.BytesIO(wav_bytes))
        return y

    def load_split(path):
        df = pd.read_parquet(path)
        data = []
        for _, row in df.iterrows():
            y = decode_audio(row)
            label = int(row['label']) if 'label' in row else int(row['digit'])
            data.append((y, label))
        return data

    train = load_split(train_path)
    test = load_split(test_path)
    return train, test
