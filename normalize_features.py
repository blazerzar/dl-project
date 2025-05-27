from os import listdir, mkdir, path

import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

FEATURES_DIR = 'data/features_dev'
NORMALIZED_DIR = 'data/normalized_dev'


def main():
    dirs = listdir(FEATURES_DIR)
    train_dirs = [d for d in dirs if 'train' in d]
    test_dirs = [d for d in dirs if 'test' in d]

    scaler = StandardScaler()
    for dir in train_dirs:
        fit_scaler(path.join(FEATURES_DIR, dir), scaler)

    mkdir(NORMALIZED_DIR)
    for dir in train_dirs + test_dirs:
        mkdir(path.join(NORMALIZED_DIR, dir))
        transform_features(path.join(FEATURES_DIR, dir), scaler)


def fit_scaler(dir, scaler):
    for file in tqdm(listdir(dir)):
        if 'features' not in file:
            continue
        features = torch.load(path.join(dir, file)).permute(2, 0, 1)
        features = features.reshape(features.shape[0], -1)
        scaler.partial_fit(features.numpy())


def transform_features(dir, scaler):
    for file in tqdm(listdir(dir)):
        if 'features' not in file:
            continue
        features = torch.load(path.join(dir, file)).permute(2, 0, 1)
        shape = features.shape
        features = features.reshape(features.shape[0], -1)
        features = scaler.transform(features.numpy())
        features = features.reshape(*shape)
        features = torch.from_numpy(features).permute(1, 2, 0)
        torch.save(features, path.join(NORMALIZED_DIR, dir.split('/')[-1], file))


if __name__ == '__main__':
    main()
