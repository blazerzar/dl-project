"""
Dataset classes for PyTorch models. For features, we use the Log-Mel
spectrogram and ambisonics intensity vector. For RNN based models, sequences
are padded to the same length in a batch, but for transformer based models,
the sequences are split into smaller chunks.

Each audio file can be augmented in 8 different ways. Features and labels of
each transformation are saved locally for faster loading.
"""

from os import path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import get_dir_files


class VariableLengthDataset(Dataset):
    """Dataset for audio files with variable number of frames."""

    def __init__(self, features_dir, *, split='train', augments=[2]):
        """Dataset for features and labels for the RNN model. With augments,
        we can only use some of the files created in data augmentation. The
        default files (non-augmented) have index 2."""
        if split == 'test' and augments != [2]:
            raise ValueError('Invalid augmentation for test set')

        files = get_dir_files(features_dir, split=split)
        features = [f for f in files if 'features' in f]
        labels = [f for f in files if 'labels' in f]

        self.features_dir = features_dir
        self.features = sorted(self._filter_files(features, augments))
        self.labels = sorted(self._filter_files(labels, augments))

    def _filter_files(self, files, augments):
        return [f for f in files if any((f'_{a}_' in f) for a in augments)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.load(path.join(self.features_dir, self.features[idx]))
        labels = torch.load(path.join(self.features_dir, self.labels[idx]))
        return features, labels


def collate_fn(batch):
    logmels, labels = zip(*batch)
    lengths = torch.tensor([logmel.shape[2] for logmel in logmels])
    logmels_padded = pad_sequence(
        [lm.permute(2, 0, 1) for lm in logmels], batch_first=True
    ).permute(0, 2, 3, 1)
    labels_padded = pad_sequence(labels, batch_first=True)

    return logmels_padded, labels_padded, lengths
