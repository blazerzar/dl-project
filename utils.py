from os import listdir, path

import numpy as np

NUM_CLASSES = 13


def get_dir_files(dir, *, split=''):
    """Return all files in all directories in the given directory."""
    files = []
    for directory in listdir(dir):
        if split and split not in directory:
            continue

        for filename in listdir(path.join(dir, directory)):
            files.append(path.join(directory, filename))
    return sorted(files)


def split_into_segments(detections, segment_length, max_frame):
    """Split detections into segments with the given number of frames."""
    segments = []
    frames = detections[:, 0]
    for i in range(0, max_frame + 1, segment_length):
        mask = (frames >= i) & (frames < i + segment_length)
        segments.append(detections[mask])
    return segments


def seld_evaluation_metrics(predictions, ground_truth):
    """Calculate and return SELD evaluation metrics.

    Parameters:
        - predictions, ground_truth: list of np.array for each batch element,
            i.e., audio file, where each array is of shape
            (valid time steps, 5 = frame, class, x, y, z)
    """
    # TP_c, TP_20, FP_20, FN_c, sum theta_k
    metrics = np.zeros((NUM_CLASSES, 5))

    for pred, gt in zip(predictions, ground_truth):
        max_frame = gt[-1, 0].astype(int)
        if pred.shape[0] > 0 and pred[-1, 0] > max_frame:
            max_frame = pred[-1, 0].astype(int)

        pred_segments = split_into_segments(pred, 1, max_frame)
        gt_segments = split_into_segments(gt, 10, max_frame)

        for cls in range(NUM_CLASSES):
            for pred_segment, gt_segment in zip(pred_segments, gt_segments):
                preds = pred_segment[pred_segment[:, 1] == cls]
                gts = gt_segment[gt_segment[:, 1] == cls]

                P_c = int(preds.shape[0] > 0)
                R_c = int(gts.shape[0] > 0)
                FN_c = np.maximum(0, R_c - P_c)
                TP_c = np.minimum(P_c, R_c)

                metrics[cls, 0] += TP_c
                metrics[cls, 3] += FN_c

                if TP_c == 1:
                    preds_doa = np.mean(preds[:, 2:5], axis=0)
                    gt_doa = np.mean(gts[:, 2:5], axis=0)

                    dot_p = np.dot(preds_doa, gt_doa)
                    preds_norm = np.linalg.norm(preds_doa)
                    gt_norm = np.linalg.norm(gt_doa)

                    angular_distance = np.acos(dot_p / (preds_norm * gt_norm + 1e-10))
                    angular_distance = np.degrees(angular_distance)
                    metrics[cls, 4] += angular_distance

                    if angular_distance <= 20:
                        metrics[cls, 1] += 1
                    else:
                        metrics[cls, 2] += 1

    pr_20_c = metrics[:, 1] / (metrics[:, 1] + metrics[:, 2] + 1e-10)
    re_20_c = metrics[:, 1] / (metrics[:, 1] + metrics[:, 3] + 1e-10)
    f_20_c = 2 * (pr_20_c * re_20_c) / (pr_20_c + re_20_c + 1e-10)
    er_c = (metrics[:, 2] + metrics[:, 3]) / (metrics[:, 0] + metrics[:, 3])
    le_c = metrics[:, 4] / np.where(metrics[:, 0] == 0, np.nan, metrics[:, 0])
    lr_c = metrics[:, 0] / (metrics[:, 0] + metrics[:, 3])

    f_20 = np.mean(f_20_c)
    er = np.mean(er_c)
    le = 0 if np.sum(~np.isnan(le_c)) == 0 else np.nanmean(le_c)
    lr = np.mean(lr_c)

    return f_20, er, le, lr
