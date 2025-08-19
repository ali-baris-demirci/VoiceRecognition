"""
features.py
Audio feature extraction functions (MFCCs, Mel-spectrograms).
"""
import numpy as np
import librosa


def extract_mfcc(y, sr=8000, n_mfcc=13, n_fft=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    return np.mean(mfcc, axis=1)


def extract_mel_spectrogram(y, sr=8000, n_mels=40, n_fft=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft)
    S_db = librosa.power_to_db(S, ref=np.max)
    return np.mean(S_db, axis=1)
