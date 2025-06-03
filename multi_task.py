import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from conformer import SELDConformerBackbone
from SELD_evaluation_metrics import SELDMetrics
from seld_net import SELDNetBackbone
from utils import spherical_to_cartesian

NUM_CLASSES = 13


class MultiTaskSELD(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes,
        num_events,
        input_dim,
        dropout,
        hidden_dim,
        rnn_layers=0,
        mhsa_layers=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_events = num_events

        if backbone == 'seldnet':
            self.backbone = SELDNetBackbone(
                num_classes,
                num_events,
                input_dim,
                hidden_dim,
                dropout,
                rnn_layers,
                mhsa_layers,
                variable_length=False,
            )
        elif backbone == 'conformer':
            self.backbone = SELDConformerBackbone(input_dim, dropout)
        else:
            raise ValueError(f'Invalid backbone: "{backbone}"')

        # Techniques based on: https://arxiv.org/pdf/2403.11827, idea from:
        # https://arxiv.org/pdf/2010.15306, predict (x, y z) for each class
        self.doa = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes * 3),
            nn.Tanh(),
        )
        # Predict distance for each class
        self.sde = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        doa = self.doa(x).reshape(*x.shape[:2], self.num_classes, 3)
        sde = self.sde(x).reshape(*x.shape[:2], self.num_classes, 1)
        x = torch.cat([doa, sde], dim=-1)
        return x


def train_model(
    model_args,
    train_dataloader,
    test_dataloader,
    epochs,
    *,
    device='cpu',
    sde_weight=0.0,
):
    model = MultiTaskSELD(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def loss_fn(*x, predictions=False):
        return multi_task_loss(*x, sde_weight=sde_weight, predictions=predictions)

    # ER, F, LE, LR
    best_epoch = 0
    best_metrics_micro = 0, 0, 0, 0
    best_metrics_macro = 0, 0, 0, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}')
        for j, (features, labels, files, offsets) in enumerate(train_dataloader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss, *_ = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / (j + 1))
            pbar.update(1)

        model.eval()
        test_loss = 0
        test_metrics = SELDMetrics(nb_classes=model_args['num_classes'])
        with torch.no_grad():
            for features, labels, files, offsets in test_dataloader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss, predictions, labels = loss_fn(outputs, labels, predictions=True)
                test_loss += loss.item()

                test_metrics.update_seld_scores(predictions, labels)

            pbar.set_postfix(
                loss=train_loss / len(train_dataloader),
                test_loss=test_loss / len(test_dataloader),
            )
            pbar.close()

        er, f, le, lr, *_ = test_metrics.compute_seld_scores()
        if f > best_metrics_micro[1]:
            best_epoch = epoch + 1
            best_metrics_micro = er, f, le, lr
        if (epoch + 1) % 10 == 0:
            print(f'Macro: ER={er:.2f}, F={f:.2f}, LE={le:.2f}, LR={lr:.2f}')

        test_metrics._average = 'micro'
        er, f, le, lr, *_ = test_metrics.compute_seld_scores()
        if best_epoch == epoch + 1:
            best_metrics_macro = er, f, le, lr, epoch + 1
        if (epoch + 1) % 10 == 0:
            print(f'Micro: ER={er:.2f}, F={f:.2f}, LE={le:.2f}, LR={lr:.2f}')

    print(
        f'Micro @ epoch {best_epoch}: '
        f'ER={best_metrics_micro[0]:.2f}, F={best_metrics_micro[1]:.2f}, '
        f'LE={best_metrics_micro[2]:.2f}, LR={best_metrics_micro[3]:.2f}'
    )
    print(
        f'Macro @ epoch {best_epoch}: '
        f'ER={best_metrics_macro[0]:.2f}, F={best_metrics_macro[1]:.2f}, '
        f'LE={best_metrics_macro[2]:.2f}, LR={best_metrics_macro[3]:.2f}'
    )

    return model


def multi_task_labels(labels):
    """Convert labels to a form suitable for multi-task loss.

    Parameters:
        - labels: (*, track, 4 = class, azimuth, elevation, distance)

    Returns:
        - out: (*, num classes, 3 + 1)
    """
    device = labels.device
    x, y, z = spherical_to_cartesian(labels[:, :, :, 1:3])
    dist = labels[:, :, :, 3].float() / 100

    out = torch.zeros(*labels.shape[:-2], NUM_CLASSES, 3 + 1, device=device)
    classes = labels[:, :, :, 0]
    for i in range(NUM_CLASSES):
        class_mask = classes == i + 1
        # Get batch, frame and track indices for all class events
        batch_frame_track = torch.nonzero(class_mask)
        batch_frame = batch_frame_track[:, :2]

        # Remove duplicate batch and frame indices (keep minimal track)
        _, inv = torch.unique(batch_frame, return_inverse=True, dim=0)
        good_indices = torch.ones_like(inv, dtype=torch.bool)
        good_indices[1:] = inv[1:] != inv[:-1]
        batch_frame_track = batch_frame_track[good_indices]

        batches = batch_frame_track[:, 0]
        frames = batch_frame_track[:, 1]
        tracks = batch_frame_track[:, 2]
        out[batches, frames, i, 0] = x[batches, frames, tracks]
        out[batches, frames, i, 1] = y[batches, frames, tracks]
        out[batches, frames, i, 2] = z[batches, frames, tracks]
        out[batches, frames, i, 3] = dist[batches, frames, tracks]
    return out


def multi_task_predictions(outputs):
    """Create predictions from outputs or labels in multi-task format.

    Parameters:
        - outputs: (batch size, time steps, num classes, 3 + 1)

    Returns:
        - predictions: {block index: {class index: [[frames, doa]]}}
    """
    # Each batch has 50 frames = 5 seconds = 5 blocks
    predictions = {x: {} for x in range(outputs.shape[0] * 5)}
    batch_size, frames, classes, output_dim = outputs.shape
    outputs = outputs.reshape(batch_size, 5, 10, classes, output_dim)

    for i, batch in enumerate(outputs):
        for j, block in enumerate(batch):
            block_index = i * 5 + j

            radius = torch.sqrt(block[..., :3].pow(2).sum(dim=-1))
            active = radius > 0.5
            x = block[..., 0][active] / radius[active]
            y = block[..., 1][active] / radius[active]
            z = block[..., 2][active] / radius[active]

            frames_classes = torch.nonzero(active)
            for k, (frame, cls_t) in enumerate(frames_classes):
                cls = cls_t.item()
                if cls not in predictions[block_index]:
                    predictions[block_index][cls] = [[[], []]]

                doa = [[0, x[k].item(), y[k].item(), z[k].item()]]
                predictions[block_index][cls][0][0].append(frame.item())
                predictions[block_index][cls][0][1].append(doa)

    return predictions


def multi_task_loss(outputs, labels, *, sde_weight=1.0, predictions=False):
    """Compute loss using MSE for DOA and SDE branches. The technique was
    proposed in https://arxiv.org/pdf/2403.11827 based on the idea from
    https://arxiv.org/pdf/2010.15306.

    Parameters:
        - outputs: (batch size, time steps, num classes, 3 + 1)
        - labels: (batch size, time steps, max num events, num classes)

    Returns:
        - loss: Tensor scalar
        - predictions, labels: {block index: {class index: [[frames, doa]]}}
    """
    mt_labels = multi_task_labels(labels)

    doa_loss = F.mse_loss(outputs[:, :, :, :3], mt_labels[:, :, :, :3])
    sde_loss = F.mse_loss(outputs[:, :, :, 3:], mt_labels[:, :, :, 3:])
    loss = (1 - sde_weight) * doa_loss + sde_weight * sde_loss

    if predictions:
        predictions = multi_task_predictions(outputs)
        labels = multi_task_predictions(mt_labels)
    else:
        predictions, labels = {}, {}
    return loss, predictions, labels
