"""Extract features from waveforms, augment them, and save them to disk."""

from os import listdir, mkdir, path

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

from utils import get_dir_files

FOA_DIR = 'data/foa_dev'
METADATA_DIR = 'data/metadata_dev'
FEATURES_DIR = 'data/features_dev'

SAMPLE_RATE = 24000
FRAME_LENGTH = SAMPLE_RATE // 10
MAX_EVENTS = 5

columns = ['frame', 'class', 'source', 'azimuth', 'elevation', 'distance']


def main():
    foa_files = get_dir_files(FOA_DIR)
    metadata_files = get_dir_files(METADATA_DIR)
    mkdir(FEATURES_DIR)
    for dir in listdir(FOA_DIR):
        mkdir(path.join(FEATURES_DIR, dir))

    for foa_file, metadata_file in tqdm(
        zip(foa_files, metadata_files), total=len(foa_files)
    ):
        extract_and_save(foa_file, metadata_file, stft_tr, mel_tr, logmel_tr)


def extract_and_save(foa_file, metadata_file, stft_tr, mel_tr, logmel_tr):
    waveform, _ = torchaudio.load(path.join(FOA_DIR, foa_file))
    metadata = pd.read_csv(path.join(METADATA_DIR, metadata_file), names=columns)

    # Remove the last incomplete frame
    remainder = waveform.shape[1] % FRAME_LENGTH
    waveform = waveform[:, : waveform.shape[1] - remainder]

    # Extract features and labels for each augmentation and save them
    for i in range(len(doa_augmentations)):
        # Test files are not augmented, i.e., augmentation 2
        if 'dev-test' in foa_file and i != 2:
            continue

        features, labels = features_and_labels(waveform, metadata, i)
        file_name = foa_file.split('.')[0]
        torch.save(features, path.join(FEATURES_DIR, f'{file_name}_{i}_features.pt'))
        torch.save(labels, path.join(FEATURES_DIR, f'{file_name}_{i}_labels.pt'))


def features_and_labels(waveform, metadata, doa_transformation):
    """Extract features and labels from the waveform and metadata for the
    given data augmentation transformation."""
    waveform, metadata = augment(waveform, metadata, doa_transformation)

    frames = waveform.shape[1] // FRAME_LENGTH
    features = extract_features(waveform)

    # For each event, we annotate: class, azimuth, elevation, distance
    labels = torch.zeros(frames, MAX_EVENTS, 4, dtype=torch.long)
    events = metadata.groupby('frame')[columns[1:]].apply(lambda x: x.values.tolist())
    for frame, events in events.items():
        # Some annotations seem incorrect
        if frame >= frames:
            continue

        # A single audio file has 6 simultaneous events, but we omit it
        events = sorted(events)[:MAX_EVENTS]
        for i, (cls, source, azimuth, elevation, distance) in enumerate(events):
            labels[frame, i] = torch.tensor([cls, azimuth, elevation, distance])

    return features, labels


stft_tr = T.Spectrogram(
    n_fft=1024,
    win_length=SAMPLE_RATE // 25,  # 40ms
    hop_length=SAMPLE_RATE // 50,  # 20ms
    power=None,
)
mel_tr = T.MelScale(
    n_mels=64,
    sample_rate=SAMPLE_RATE,
    n_stft=1024 // 2 + 1,
)
logmel_tr = T.AmplitudeToDB(stype='power')


def extract_features(waveform):
    """Extract Log-Mel spectrogram and intensity vector features. Intensity
    vector features are defined in paper: https://arxiv.org/pdf/2002.05994."""
    stft = stft_tr(waveform)
    stft = stft[:, :, : waveform.shape[1] // (SAMPLE_RATE // 50)]
    logmel = logmel_tr(mel_tr(stft.abs().pow(2)))

    W = stft[0:1]
    h = stft[1:]
    iv = torch.real(torch.conj(W) * h)
    iv = iv / (iv.norm(dim=0, keepdim=True) + 1e-8)

    mel_fbanks = F.melscale_fbanks(iv.shape[1], 0, SAMPLE_RATE // 2, 64, SAMPLE_RATE)
    iv = torch.matmul(iv.permute(2, 0, 1), mel_fbanks).permute(1, 2, 0)

    return torch.cat([logmel, iv], dim=0)


def shift_azimuth(azimuth, shift):
    azimuth += shift
    if azimuth > 180:
        azimuth -= 360
    elif azimuth < -180:
        azimuth += 360
    return azimuth


doa_augmentations = [
    (lambda phi: shift_azimuth(phi, -90), lambda theta: -theta),
    (lambda phi: shift_azimuth(-phi, -90), lambda theta: theta),
    (lambda phi: phi, lambda theta: theta),
    (lambda phi: -phi, lambda theta: -theta),
    (lambda phi: shift_azimuth(phi, 90), lambda theta: -theta),
    (lambda phi: shift_azimuth(-phi, 90), lambda theta: theta),
    (lambda phi: shift_azimuth(phi, 180), lambda theta: theta),
    (lambda phi: shift_azimuth(-phi, 180), lambda theta: -theta),
]

foa_augmentations = [
    ([0, 3, 2, 1], torch.tensor([[1, -1, -1, 1]]).T),
    ([0, 3, 2, 1], torch.tensor([[1, -1, 1, -1]]).T),
    ([0, 1, 2, 3], torch.tensor([[1, 1, 1, 1]]).T),
    ([0, 1, 2, 3], torch.tensor([[1, -1, -1, 1]]).T),
    ([0, 3, 2, 1], torch.tensor([[1, 1, -1, -1]]).T),
    ([0, 3, 2, 1], torch.tensor([[1, 1, 1, 1]]).T),
    ([0, 1, 2, 3], torch.tensor([[1, -1, 1, -1]]).T),
    ([0, 1, 2, 3], torch.tensor([[1, 1, -1, -1]]).T),
]


def augment(waveform, metadata, data_augmentation):
    order, signs = foa_augmentations[data_augmentation]
    waveform = waveform[order] * signs
    metadata = metadata.copy()
    metadata['azimuth'] = metadata['azimuth'].apply(
        doa_augmentations[data_augmentation][0]
    )
    metadata['elevation'] = metadata['elevation'].apply(
        doa_augmentations[data_augmentation][1]
    )
    return waveform, metadata


if __name__ == '__main__':
    main()
