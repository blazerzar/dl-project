"""
Dataset classes for PyTorch models. For features, we use the Log-Mel
spectrogram and ambisonics intensity vector. For RNN based models, sequences
are padded to the same length in a batch, but for transformer based models,
the sequences are split into smaller chunks.

Each audio file can be augmented in 8 different ways. Features and labels of
each transformation are saved locally for faster loading.
"""

from os import path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils import get_dir_files

FEATURES_DIR = 'data/features_dev'
NORMALIZED_DIR = 'data/normalized_dev'


class VariableLengthDataset(Dataset):
    """Dataset for audio files with variable number of frames."""

    def __init__(self, features_dir, *, split='train', augments=[2], normalized_dir=''):
        """Dataset for features and labels for the RNN model. With augments,
        we can only use some of the files created in data augmentation. The
        default files (non-augmented) have index 2."""
        if split == 'test' and augments != [2]:
            raise ValueError('Invalid augmentation for test set')

        files = get_dir_files(features_dir, split=split)
        features = [f for f in files if 'features' in f]
        labels = [f for f in files if 'labels' in f]

        self.features_dir = features_dir
        self.labels_dir = features_dir
        if normalized_dir:
            self.features_dir = normalized_dir

        self.features = sorted(self._filter_files(features, augments))
        self.labels = sorted(self._filter_files(labels, augments))

    def _filter_files(self, files, augments):
        return [f for f in files if any((f'_{a}_' in f) for a in augments)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.load(path.join(self.features_dir, self.features[idx]))
        labels = torch.load(path.join(self.labels_dir, self.labels[idx]))
        return features, labels


class FixedLengthDataset(Dataset):
    """Dataset for audio files split into 50 frames chunks. Because the hop
    length is 20 ms, 50 frames are represented by 250 values."""

    def __init__(self, features_dir, split='train', augments=[2], normalized_dir=''):
        if split == 'test' and augments != [2]:
            raise ValueError('Invalid augmentation for test set')

        files = get_dir_files(features_dir, split=split)
        features = [f for f in files if 'features' in f]
        labels = [f for f in files if 'labels' in f]

        self.features_dir = features_dir
        self.labels_dir = features_dir
        if normalized_dir:
            self.features_dir = normalized_dir

        self.features = sorted(self._filter_files(features, augments))
        self.labels = sorted(self._filter_files(labels, augments))
        self.cum_lengths = self._get_cumulative_lengths()

        self.cache = {}
        self.cache_queue = []

    def _filter_files(self, files, augments):
        return [f for f in files if any((f'_{a}_' in f) for a in augments)]

    def _get_cumulative_lengths(self):
        lengths = [0]
        for f in self.features:
            features = torch.load(path.join(self.features_dir, f))
            frames = features.shape[2] // 5
            sequences = int(np.ceil(frames / 50))
            prev = lengths[-1] if lengths else 0
            lengths.append(prev + sequences)
        return lengths

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, index):
        file_index = np.searchsorted(self.cum_lengths, index, side='right') - 1
        if file_index in self.cache:
            features, labels = self.cache[file_index]
        else:
            features = torch.load(
                path.join(self.features_dir, self.features[file_index])
            )
            labels = torch.load(path.join(self.labels_dir, self.labels[file_index]))
            self.cache[file_index] = (features, labels)
            self.cache_queue.append(file_index)

            if len(self.cache_queue) > 10:
                oldest_index = self.cache_queue.pop(0)
                del self.cache[oldest_index]

        start_f = (index - self.cum_lengths[file_index]) * 250
        start_l = start_f // 5
        features = features[:, :, start_f : start_f + 250]
        labels = labels[start_l : start_l + 50]
        if features.shape[2] < 250:
            padding = 0, 250 - features.shape[2]
            features = F.pad(features, padding, value=0)
        if labels.shape[0] < 50:
            padding = 0, 0, 0, 0, 0, 50 - labels.shape[0]
            labels = F.pad(labels, padding, value=0)

        # Need to compute predictions
        frame_offset = start_f // 5

        return features, labels, self.features[file_index], frame_offset


def collate_fn(batch):
    logmels, labels = zip(*batch)
    lengths = torch.tensor([logmel.shape[2] for logmel in logmels])
    logmels_padded = pad_sequence(
        [lm.permute(2, 0, 1) for lm in logmels], batch_first=True
    ).permute(0, 2, 3, 1)
    labels_padded = pad_sequence(labels, batch_first=True)

    return logmels_padded, labels_padded, lengths


def create_dataloaders(batch_size, *, augments=[2], normalized=False):
    train_dataset = FixedLengthDataset(
        FEATURES_DIR,
        split='train',
        augments=augments,
        normalized_dir=NORMALIZED_DIR if normalized else None,
    )
    test_dataset = FixedLengthDataset(
        FEATURES_DIR,
        split='test',
        normalized_dir=NORMALIZED_DIR if normalized else None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_dataloader, test_dataloader
